use std::collections::HashMap;
use std::error::Error;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path as FsPath, PathBuf};

use ai_engine_crf::blocks::{CachedSentence, PageBlocks};
use ai_engine_crf::corpus::{self, Index, Limits, Sentence};
use ai_engine_crf::output;
use ai_engine_crf::{BOS, EOS, Model, Path as CrfPath, nbest};
use ai_engine_llm::{CandidateBatchStats, CandidatePrefix, LlamaCppConfig, LlamaCppScorer};
use serde::{Deserialize, Serialize};

const DEFAULT_NBEST: usize = 128;
const DEFAULT_EXPANSION_LIMIT: usize = 500_000;
const DEFAULT_TRAIN_SENTENCES: usize = 10_000;
const DEFAULT_MAX_WORDS: usize = 48;
const DEFAULT_CONTEXT_LENGTH: u32 = 32_768;
const DEFAULT_SEED: u64 = 20_261_007;

#[derive(Debug)]
struct Options {
    model_in: PathBuf,
    page_cache: PathBuf,
    test_sample: PathBuf,
    llama_model: PathBuf,
    model_name: Option<String>,
    model_out_dir: PathBuf,
    work_dir: PathBuf,
    train_sentences: usize,
    train_start_sentence: usize,
    train_stride: usize,
    test_sentences: usize,
    test_start_sentence: usize,
    test_stride: usize,
    nbest: usize,
    expansion_limit: usize,
    max_words: usize,
    context_length: u32,
    threads: i32,
    gpu_layers: i32,
    seed: u64,
    fit_metrics: Option<PathBuf>,
    training_features: Option<PathBuf>,
    fixed_alpha: Option<f64>,
    rescore_training_features: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct CandidateFeature {
    text: String,
    crf_score: f64,
    llm_mean_softmax: f64,
    gold: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ScoredExample {
    page_id: u64,
    title: String,
    input: String,
    gold_text: String,
    gold_candidate: Option<usize>,
    raw_nbest_gold_cover: bool,
    #[serde(default)]
    llm_unique_candidate_count: usize,
    #[serde(default)]
    llm_decode_batch_count: usize,
    #[serde(default)]
    llm_max_sequences_per_decode_batch: usize,
    candidates: Vec<CandidateFeature>,
}

#[derive(Clone, Debug, Serialize)]
struct EvaluationMetrics {
    sentence_count: usize,
    raw_nbest_gold_cover_rate: f64,
    bigram_top1_accuracy: f64,
    crf_post_trigram_top1_accuracy: f64,
    llm_crf_top1_accuracy: f64,
    bigram_character_level_accuracy: f64,
    crf_post_trigram_character_level_accuracy: f64,
    llm_crf_character_level_accuracy: f64,
    requested_llm_sequences_per_batch: usize,
    total_llm_unique_candidate_texts: usize,
    total_llm_decode_batches: usize,
    max_llm_sequences_per_decode_batch: usize,
    sentences_reaching_requested_llm_batch: usize,
    alpha: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct FitMetrics {
    training_sentences: usize,
    gold_augmented_sentences: usize,
    crf_score_rms: f64,
    feature_centering_rms: f64,
    normalization: String,
    initial_alpha: f64,
    learned_alpha: f64,
    iterations: usize,
    converged: bool,
    listwise_nll_before: f64,
    listwise_nll_after: f64,
    listwise_top1_before: f64,
    listwise_top1_after: f64,
}

#[derive(Serialize)]
struct RunMetadata {
    version: u32,
    model_in: String,
    llama_model: String,
    page_cache: String,
    test_sample: String,
    nbest: usize,
    llama_context_length: u32,
    llama_cpu_threads: i32,
    llama_gpu_layers: i32,
    llama_sequence_capacity: usize,
    expansion_limit: usize,
    training_sentences: usize,
    training_start_sentence: usize,
    training_stride: usize,
    fixed_alpha: Option<f64>,
    training_features_rescored: bool,
    fixed_test_sentences: usize,
    test_start_sentence: usize,
    test_stride: usize,
    seed: u64,
    score_definition: &'static str,
    score_normalization: &'static str,
    objective: &'static str,
    crf_parameters_updated: &'static str,
    alpha_fit_source: Option<String>,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("crf-llm-alpha error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let options = parse_options()?;
    let llama_model = resolve_llama_model(&options.llama_model, options.model_name.as_deref())?;
    if !options.model_in.is_file() {
        return Err(format!("CRF model does not exist: {}", options.model_in.display()).into());
    }
    if !options.test_sample.is_file() {
        return Err(format!(
            "fixed test sample does not exist: {}",
            options.test_sample.display()
        )
        .into());
    }
    if options.nbest == 0 || options.train_sentences == 0 || options.test_sentences == 0 {
        return Err("N-best, training, and test sentence counts must be positive".into());
    }
    prepare_fresh_directory(&options.work_dir)?;
    prepare_fresh_directory(&options.model_out_dir)?;
    if options.fit_metrics.is_some() && options.training_features.is_some() {
        return Err("use only one of --fit-metrics or --training-features".into());
    }
    if options.rescore_training_features && options.training_features.is_none() {
        return Err("--rescore-training-features requires --training-features".into());
    }
    if options.fixed_alpha.is_some_and(|alpha| !alpha.is_finite()) {
        return Err("--fixed-alpha must be finite".into());
    }
    if options.fixed_alpha.is_some() && options.fit_metrics.is_some() {
        return Err("--fixed-alpha cannot be combined with --fit-metrics; use rescored training features to estimate per-model normalization".into());
    }
    if options.fixed_alpha.is_some()
        && options.training_features.is_some()
        && !options.rescore_training_features
    {
        return Err("fixed-alpha comparisons with saved training features must rescore those features for this GGUF".into());
    }
    write_readme(
        &options.work_dir,
        &options.model_out_dir,
        &llama_model,
        options
            .fit_metrics
            .as_deref()
            .or(options.training_features.as_deref()),
        options.fixed_alpha,
        options.rescore_training_features,
    )?;

    let mut model = output::load(&options.model_in)?;
    validate_model(&model)?;
    let model_index = Index::new(&model)?;
    let supplied_fit = options
        .fit_metrics
        .as_ref()
        .map(|path| read_json::<FitMetrics>(path))
        .transpose()?;
    let mut cached_training_examples = options
        .training_features
        .as_ref()
        .map(|path| read_jsonl::<ScoredExample>(path))
        .transpose()?
        .map(|examples| {
            select_spaced(
                &examples,
                options.train_start_sentence,
                options.train_sentences,
                options.train_stride,
                "saved training features",
            )
        })
        .transpose()?;
    if supplied_fit
        .as_ref()
        .is_some_and(|fit| !fit.learned_alpha.is_finite())
    {
        return Err("supplied fit-metrics has a non-finite learned_alpha".into());
    }
    let all_test: Vec<Sentence> =
        serde_json::from_reader(BufReader::new(File::open(&options.test_sample)?))?;
    let test = select_spaced(
        &all_test,
        options.test_start_sentence,
        options.test_sentences,
        options.test_stride,
        "fixed test sample",
    )?;
    validate_sentences(&test, &model)?;
    if supplied_fit.is_some() && options.train_start_sentence != 0 {
        return Err("--train-start-sentence cannot be used with --fit-metrics".into());
    }
    let training = if supplied_fit.is_none() && cached_training_examples.is_none() {
        let blocks = PageBlocks::load(&options.page_cache)?;
        let requested_end = selection_end(
            options.train_start_sentence,
            options.train_sentences,
            options.train_stride,
            "training sentence range",
        )?;
        if blocks.training_sentences < requested_end {
            return Err(format!(
                "page cache has only {} training sentences; requested range {}..{}",
                blocks.training_sentences, options.train_start_sentence, requested_end
            )
            .into());
        }
        let sampled = select_training_sentences(&blocks, requested_end, options.seed)?;
        let training = select_spaced(
            &sampled,
            options.train_start_sentence,
            options.train_sentences,
            options.train_stride,
            "selected training sample",
        )?;
        validate_sentences(&training, &model)?;
        Some(training)
    } else {
        None
    };
    let training_count = supplied_fit.as_ref().map_or_else(
        || {
            cached_training_examples.as_ref().map_or_else(
                || training.as_ref().map_or(0, |items| items.len()),
                Vec::len,
            )
        },
        |fit| fit.training_sentences,
    );

    let metadata = RunMetadata {
        version: 4,
        model_in: options.model_in.display().to_string(),
        llama_model: llama_model.display().to_string(),
        page_cache: options.page_cache.display().to_string(),
        test_sample: options.test_sample.display().to_string(),
        nbest: options.nbest,
        llama_context_length: options.context_length,
        llama_cpu_threads: options.threads,
        llama_gpu_layers: options.gpu_layers,
        llama_sequence_capacity: options.nbest,
        expansion_limit: options.expansion_limit,
        training_sentences: training_count,
        training_start_sentence: options.train_start_sentence,
        training_stride: options.train_stride,
        fixed_alpha: options.fixed_alpha,
        training_features_rescored: options.rescore_training_features,
        fixed_test_sentences: test.len(),
        test_start_sentence: options.test_start_sentence,
        test_stride: options.test_stride,
        seed: options.seed,
        score_definition: "For each candidate surface, prepend BOS, score every candidate token through its last token with the target-token softmax probability, omit EOS, and take the arithmetic mean over that candidate's token count.",
        score_normalization: "Per-sentence N-best mean centering; CRF and LLM values divided by their pooled within-list RMS from the training split; combined score z_crf + alpha*z_llm.",
        objective: if options.fixed_alpha.is_some() {
            "Fixed-alpha held-out comparison; per-GGUF normalization scales are estimated from training-only candidate features; no alpha fit and no fitting on fixed-test sentences."
        } else if supplied_fit.is_some() || cached_training_examples.is_some() {
            "Held-out evaluation using alpha fitted or loaded from training-only candidate features; no fitting on fixed-test sentences."
        } else {
            "Unregularized N-best listwise negative log likelihood; inject the gold path only when absent from the raw first-order N-best set; optimize one scalar alpha while freezing every CRF parameter."
        },
        crf_parameters_updated: if options.fixed_alpha.is_some() {
            "none (fixed alpha; per-GGUF RMS scales from training-only rescored features; CRF frozen)"
        } else if supplied_fit.is_some() {
            "none (loaded alpha and normalizers only; CRF frozen)"
        } else if cached_training_examples.is_some() {
            "llm_alpha learned; CRF/LLM RMS scales estimated from saved training features; CRF weights frozen"
        } else {
            "llm_alpha learned; CRF/LLM RMS scales estimated from training features; CRF weights frozen"
        },
        alpha_fit_source: options
            .fit_metrics
            .as_ref()
            .or(options.training_features.as_ref())
            .map(|path| path.display().to_string()),
    };
    write_json(&options.work_dir.join("run.json"), &metadata)?;

    eprintln!(
        "loading local GGUF model={} context={} threads={} gpu_layers={} sequence_capacity={}",
        llama_model.display(),
        options.context_length,
        options.threads,
        options.gpu_layers,
        options.nbest
    );
    let scorer = LlamaCppScorer::load(
        &llama_model,
        LlamaCppConfig {
            context_length: options.context_length,
            threads: options.threads,
            gpu_layers: options.gpu_layers,
            max_sequences: options.nbest.try_into()?,
        },
    )?;

    let limits = Limits {
        words: options.max_words,
        nodes: 20_000,
        edges: 1_000_000,
    };

    let fit = if let Some(fit) = supplied_fit {
        model.llm_alpha = fit.learned_alpha;
        model.llm_crf_score_rms = fit.crf_score_rms;
        model.llm_feature_rms = fit.feature_centering_rms;
        write_json(&options.work_dir.join("fit-metrics.json"), &fit)?;
        fit
    } else if let Some(mut fit_examples) = cached_training_examples.take() {
        if fit_examples.len() != options.train_sentences {
            return Err(
                "selected saved training feature count does not match requested count".into(),
            );
        }
        if options.rescore_training_features {
            let mut total_batches = 0_usize;
            let mut total_unique = 0_usize;
            let mut max_sequences = 0_usize;
            let started = std::time::Instant::now();
            let training_count = fit_examples.len();
            for (index, example) in fit_examples.iter_mut().enumerate() {
                let stats = rescore_example_llm(example, &scorer)?;
                total_batches += stats.decode_batch_count;
                total_unique += stats.candidate_count;
                max_sequences = max_sequences.max(stats.max_sequences_per_decode_batch);
                if (index + 1) % 25 == 0 || index + 1 == training_count {
                    let elapsed = started.elapsed().as_secs_f64().max(0.001);
                    eprintln!(
                        "llm_alpha phase=rescore_training_features sentences={}/{} rate={:.2}/sec elapsed={:.1}s decoder_max_sequences={}/{} decode_batches={} mean_unique_candidates={:.1}",
                        index + 1,
                        training_count,
                        (index + 1) as f64 / elapsed,
                        elapsed,
                        max_sequences,
                        options.nbest,
                        total_batches,
                        total_unique as f64 / (index + 1) as f64,
                    );
                }
            }
        }
        write_jsonl(
            &options.work_dir.join("training-features.jsonl"),
            &fit_examples,
        )?;
        let fit = if let Some(alpha) = options.fixed_alpha {
            fit_alpha_at_fixed_value(&fit_examples, alpha)?
        } else {
            fit_alpha(&fit_examples, 0.0)?
        };
        write_json(&options.work_dir.join("fit-metrics.json"), &fit)?;
        fit
    } else {
        // This is a fresh scalar-only experiment. Do not inherit alpha from a prior run.
        let initial_alpha = 0.0;
        model.llm_alpha = initial_alpha;
        let training = training.as_ref().expect("training examples selected above");
        let train_path = options.work_dir.join("training-features.jsonl");
        let mut train_writer = BufWriter::new(create_new(&train_path)?);
        let mut fit_examples = Vec::with_capacity(training.len());
        let train_started = std::time::Instant::now();
        for (index, sentence) in training.iter().enumerate() {
            let example = score_sentence(
                &model,
                &model_index,
                &scorer,
                sentence,
                limits,
                options.nbest,
                options.expansion_limit,
                true,
            )?;
            serde_json::to_writer(&mut train_writer, &example)?;
            writeln!(train_writer)?;
            fit_examples.push(example);
            if (index + 1) % 25 == 0 || index + 1 == training.len() {
                let elapsed = train_started.elapsed().as_secs_f64().max(0.001);
                eprintln!(
                    "llm_alpha phase=training_features sentences={}/{} rate={:.2}/sec elapsed={:.1}s",
                    index + 1,
                    training.len(),
                    (index + 1) as f64 / elapsed,
                    elapsed
                );
            }
        }
        train_writer.flush()?;
        let fit = if let Some(alpha) = options.fixed_alpha {
            fit_alpha_at_fixed_value(&fit_examples, alpha)?
        } else {
            fit_alpha(&fit_examples, initial_alpha)?
        };
        write_json(&options.work_dir.join("fit-metrics.json"), &fit)?;
        fit
    };
    model.llm_alpha = fit.learned_alpha;
    model.llm_crf_score_rms = fit.crf_score_rms;
    model.llm_feature_rms = fit.feature_centering_rms;
    eprintln!(
        "llm_alpha fit n={} alpha={:.12e} nll={:.6}->{:.6} top1={:.4}->{:.4} iterations={} converged={}",
        fit.training_sentences,
        fit.learned_alpha,
        fit.listwise_nll_before,
        fit.listwise_nll_after,
        fit.listwise_top1_before,
        fit.listwise_top1_after,
        fit.iterations,
        fit.converged
    );

    let test_path = options.work_dir.join("test-features.jsonl");
    let mut test_writer = BufWriter::new(create_new(&test_path)?);
    let mut test_examples = Vec::with_capacity(test.len());
    let mut test_decode_batches = 0_usize;
    let mut test_unique_candidate_texts = 0_usize;
    let mut test_max_sequences_per_decode_batch = 0_usize;
    let test_started = std::time::Instant::now();
    for (index, sentence) in test.iter().enumerate() {
        let example = score_sentence(
            &model,
            &model_index,
            &scorer,
            sentence,
            limits,
            options.nbest,
            options.expansion_limit,
            false,
        )?;
        serde_json::to_writer(&mut test_writer, &example)?;
        writeln!(test_writer)?;
        test_decode_batches += example.llm_decode_batch_count;
        test_unique_candidate_texts += example.llm_unique_candidate_count;
        test_max_sequences_per_decode_batch =
            test_max_sequences_per_decode_batch.max(example.llm_max_sequences_per_decode_batch);
        test_examples.push(example);
        if (index + 1) % 25 == 0 || index + 1 == test.len() {
            let elapsed = test_started.elapsed().as_secs_f64().max(0.001);
            eprintln!(
                "llm_alpha phase=fixed_test sentences={}/{} rate={:.2}/sec elapsed={:.1}s decoder_max_sequences={}/{} decode_batches={} mean_unique_candidates={:.1}",
                index + 1,
                test.len(),
                (index + 1) as f64 / elapsed,
                elapsed,
                test_max_sequences_per_decode_batch,
                options.nbest,
                test_decode_batches,
                test_unique_candidate_texts as f64 / (index + 1) as f64,
            );
        }
    }
    test_writer.flush()?;

    let evaluation = evaluate(&test_examples, &fit, options.nbest)?;
    eprintln!(
        "llm_alpha batch_stats requested_sequences={} max_sequences_per_decode={} total_decode_batches={} mean_unique_candidates_per_sentence={:.1} sentences_full_batch={}/{}",
        evaluation.requested_llm_sequences_per_batch,
        evaluation.max_llm_sequences_per_decode_batch,
        evaluation.total_llm_decode_batches,
        evaluation.total_llm_unique_candidate_texts as f64 / evaluation.sentence_count as f64,
        evaluation.sentences_reaching_requested_llm_batch,
        evaluation.sentence_count,
    );
    write_json(&options.work_dir.join("metrics.json"), &evaluation)?;
    output::save(&model, &options.model_out_dir.join("model.json"))?;
    drop(test_examples);
    drop(training);
    drop(test);
    drop(scorer);
    drop(model);
    let reloaded = output::load(&options.model_out_dir.join("model.json"))?;
    let roundtrip = (reloaded.llm_alpha - fit.learned_alpha).abs() <= 1e-12
        && (reloaded.llm_crf_score_rms - fit.crf_score_rms).abs() <= 1e-12
        && (reloaded.llm_feature_rms - fit.feature_centering_rms).abs() <= 1e-12;
    if !roundtrip {
        return Err(format!(
            "LLM score parameter save/reload mismatch: alpha {:.17e}/{:.17e}, CRF RMS {:.17e}/{:.17e}, LLM RMS {:.17e}/{:.17e}",
            fit.learned_alpha,
            reloaded.llm_alpha,
            fit.crf_score_rms,
            reloaded.llm_crf_score_rms,
            fit.feature_centering_rms,
            reloaded.llm_feature_rms,
        )
        .into());
    }
    write_json(
        &options.model_out_dir.join("verification.json"),
        &serde_json::json!({
            "status":"completed",
            "model_save_reload":"passed",
            "alpha_and_normalization_roundtrip":"passed",
            "llm_alpha":reloaded.llm_alpha,
            "llm_crf_score_rms":reloaded.llm_crf_score_rms,
            "llm_feature_rms":reloaded.llm_feature_rms,
            "training_sentences":fit.training_sentences,
            "fixed_test_sentences":evaluation.sentence_count,
            "crf_weights_changed":false,
        }),
    )?;
    println!(
        "completed train={} test={} alpha={:.12e} raw_nbest_cover={:.4} crf_top1={:.4} llm_crf_top1={:.4} character_accuracy={:.4}->{:.4}",
        fit.training_sentences,
        evaluation.sentence_count,
        fit.learned_alpha,
        evaluation.raw_nbest_gold_cover_rate,
        evaluation.crf_post_trigram_top1_accuracy,
        evaluation.llm_crf_top1_accuracy,
        evaluation.crf_post_trigram_character_level_accuracy,
        evaluation.llm_crf_character_level_accuracy
    );
    Ok(())
}

fn parse_options() -> Result<Options, Box<dyn Error>> {
    let mut values = HashMap::<String, String>::new();
    let mut args = std::env::args().skip(1);
    while let Some(key) = args.next() {
        if key == "--help" || key == "-h" {
            print_help();
            std::process::exit(0);
        }
        if !key.starts_with("--") {
            return Err(format!("unexpected argument: {key}").into());
        }
        let value = if key == "--rescore-training-features" {
            "true".to_owned()
        } else {
            args.next()
                .ok_or_else(|| format!("missing value for {key}"))?
        };
        if values.insert(key.clone(), value).is_some() {
            return Err(format!("duplicate option: {key}").into());
        }
    }
    let path = |key: &str| -> Result<PathBuf, Box<dyn Error>> {
        Ok(PathBuf::from(values.get(key).ok_or_else(|| {
            format!("required option {key} is missing")
        })?))
    };
    let number = |key: &str, default: usize| -> Result<usize, Box<dyn Error>> {
        values
            .get(key)
            .map_or(Ok(default), |value| Ok(value.parse()?))
    };
    let signed = |key: &str, default: i32| -> Result<i32, Box<dyn Error>> {
        values
            .get(key)
            .map_or(Ok(default), |value| Ok(value.parse()?))
    };
    let context_length: u32 = match values.get("--context-length") {
        Some(value) => value.parse()?,
        None => DEFAULT_CONTEXT_LENGTH,
    };
    let seed: u64 = match values.get("--seed") {
        Some(value) => value.parse()?,
        None => DEFAULT_SEED,
    };
    let known = [
        "--model-in",
        "--page-cache",
        "--test-sample",
        "--llama-model",
        "--model-name",
        "--model-out-dir",
        "--work-dir",
        "--train-sentences",
        "--train-start-sentence",
        "--train-stride",
        "--test-sentences",
        "--test-start-sentence",
        "--test-stride",
        "--nbest",
        "--expansion-limit",
        "--max-words",
        "--context-length",
        "--threads",
        "--gpu-layers",
        "--seed",
        "--fit-metrics",
        "--training-features",
        "--fixed-alpha",
        "--rescore-training-features",
    ];
    if let Some(unknown) = values.keys().find(|key| !known.contains(&key.as_str())) {
        return Err(format!("unknown option: {unknown}").into());
    }
    Ok(Options {
        model_in: path("--model-in")?,
        page_cache: path("--page-cache")?,
        test_sample: path("--test-sample")?,
        llama_model: path("--llama-model")?,
        model_name: values.get("--model-name").cloned(),
        model_out_dir: path("--model-out-dir")?,
        work_dir: path("--work-dir")?,
        train_sentences: number("--train-sentences", DEFAULT_TRAIN_SENTENCES)?,
        train_start_sentence: number("--train-start-sentence", 0)?,
        train_stride: number("--train-stride", 1)?,
        test_sentences: number("--test-sentences", 5_000)?,
        test_start_sentence: number("--test-start-sentence", 0)?,
        test_stride: number("--test-stride", 1)?,
        nbest: number("--nbest", DEFAULT_NBEST)?,
        expansion_limit: number("--expansion-limit", DEFAULT_EXPANSION_LIMIT)?,
        max_words: number("--max-words", DEFAULT_MAX_WORDS)?,
        context_length,
        threads: signed("--threads", 8)?,
        gpu_layers: signed("--gpu-layers", 0)?,
        seed,
        fit_metrics: values.get("--fit-metrics").map(PathBuf::from),
        training_features: values.get("--training-features").map(PathBuf::from),
        fixed_alpha: values
            .get("--fixed-alpha")
            .map(|value| value.parse())
            .transpose()?,
        rescore_training_features: values.contains_key("--rescore-training-features"),
    })
}

fn print_help() {
    println!(
        "Usage: crf-llm-alpha --model-in <model.json> --page-cache <page-blocks-dir> --test-sample <test-sample.json> --llama-model <file.gguf|directory> --model-out-dir <dir> --work-dir <dir> [options]\n\nOptions:\n  --model-name <name>       Select one exact GGUF filename from a directory\n  --train-sentences <n>     Default 10000\n  --test-sentences <n>      Default 5000\n  --nbest <n>               Default 128\n  --expansion-limit <n>     Default 500000\n  --max-words <n>           Default 48\n  --context-length <n>      Default 32768 (wide N-best batch)\n  --threads <n>             Default 8\n  --gpu-layers <n>          Default 0 (CPU)\n  --seed <n>                Default 20261007\n  --fixed-alpha <value>     Hold alpha fixed; estimate per-GGUF scales on training scores\n  --rescore-training-features  Re-score saved N-best candidates using this GGUF\n\nIf --llama-model is a directory, use --model-name to select one .gguf file."
    );
    println!(
        "  --train-start-sentence <n> Start row in training source; default 0\n  --train-stride <n>         Select every nth training-source row; default 1\n  --test-start-sentence <n>  Start row in fixed test sample; default 0\n  --test-stride <n>          Select every nth fixed-test row; default 1"
    );
}

fn resolve_llama_model(input: &FsPath, name: Option<&str>) -> Result<PathBuf, Box<dyn Error>> {
    if input.is_file() {
        if name.is_some() {
            return Err("--model-name is only valid when --llama-model is a directory".into());
        }
        if input
            .extension()
            .is_none_or(|extension| !extension.eq_ignore_ascii_case("gguf"))
        {
            return Err("--llama-model must name a .gguf file or a directory".into());
        }
        return Ok(input.to_owned());
    }
    if !input.is_dir() {
        return Err(format!("GGUF file or directory does not exist: {}", input.display()).into());
    }
    let mut models = fs::read_dir(input)?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file()
                && path
                    .extension()
                    .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
        })
        .collect::<Vec<_>>();
    models.sort();
    let Some(name) = name else {
        let choices = models
            .iter()
            .filter_map(|path| path.file_name().map(|name| name.to_string_lossy()))
            .collect::<Vec<_>>()
            .join(", ");
        return Err(format!(
            "{} contains {} GGUF files; select one with --model-name <filename>: {choices}",
            input.display(),
            models.len()
        )
        .into());
    };
    models
        .into_iter()
        .find(|path| path.file_name().is_some_and(|file| file == name))
        .ok_or_else(|| format!("GGUF filename {name} was not found in {}", input.display()).into())
}

fn prepare_fresh_directory(path: &FsPath) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(path)?;
    if fs::read_dir(path)?.next().is_some() {
        return Err(format!("output directory is not empty: {}", path.display()).into());
    }
    Ok(())
}

fn write_readme(
    work: &FsPath,
    model: &FsPath,
    llama_model: &FsPath,
    fit_metrics: Option<&FsPath>,
    fixed_alpha: Option<f64>,
    rescore_training_features: bool,
) -> Result<(), Box<dyn Error>> {
    let mut work_readme = create_new(&work.join("README.md"))?;
    writeln!(work_readme, "# Local LLM alpha experiment work data\n")?;
    if let Some(alpha) = fixed_alpha {
        writeln!(
            work_readme,
            "Fixed-alpha comparison (alpha `{alpha}`); per-GGUF RMS normalization is measured from the training-only candidate set. The fixed test sample is never used for fitting. `{}` records training features and `metrics.json` records held-out results.\n",
            if rescore_training_features {
                "training-features.jsonl"
            } else {
                "fit-metrics.json"
            }
        )?;
    } else if let Some(fit_metrics) = fit_metrics {
        writeln!(
            work_readme,
            "Evaluation pass using saved training-only fit/features from `{}`; the fixed test sample was not used for fitting. `fit-metrics.json` records the scalar alpha and normalization scales; `test-features.jsonl` records each raw N-best candidate's CRF score and mean token softmax probability; `metrics.json` contains held-out results. Gold paths absent from raw N-best remain absent and count as misses.\n",
            fit_metrics.display()
        )?;
    } else {
        writeln!(
            work_readme,
            "Source training examples are selected from the normalized page-block cache; the fixed test sample is read-only and is not used to fit alpha. `training-features.jsonl` and `test-features.jsonl` contain each candidate's CRF score and mean token softmax probability. `fit-metrics.json` records the scalar-only fit; `metrics.json` contains held-out results. These files can be deleted after the results have been retained.\n"
        )?;
    }
    writeln!(work_readme, "Selected GGUF: `{}`.", llama_model.display())?;
    writeln!(
        work_readme,
        "The N-best scorer requests one independent llama.cpp sequence per candidate (`n_seq_max = --nbest`) and uses the configured context length as the token-batch capacity. `run.json` records the request; `metrics.json` records the maximum actual sequence width and decode-call count, since long candidates can force a split."
    )?;
    let mut model_readme = create_new(&model.join("README.md"))?;
    writeln!(model_readme, "# Local LLM alpha model output\n")?;
    writeln!(
        model_readme,
        "`model.json` stores alpha and pooled RMS normalizers. `path_score_with_llm` applies centered-and-scaled CRF and LLM features; the per-list centering constants are omitted because they are common to all candidates in one N-best list. `verification.json` confirms save/reload. Source GGUF: `{}`.\n",
        llama_model.display()
    )?;
    Ok(())
}

fn validate_model(model: &Model) -> Result<(), Box<dyn Error>> {
    if model.words.len() <= EOS as usize {
        return Err("CRF vocabulary lacks the expected BOS/EOS special word IDs".into());
    }
    if !model.llm_alpha.is_finite()
        || !model.llm_crf_score_rms.is_finite()
        || model.llm_crf_score_rms < 0.0
        || !model.llm_feature_rms.is_finite()
        || model.llm_feature_rms < 0.0
    {
        return Err("input CRF model has invalid LLM alpha/normalization parameters".into());
    }
    Ok(())
}

fn validate_sentences(sentences: &[Sentence], model: &Model) -> Result<(), Box<dyn Error>> {
    for sentence in sentences {
        if sentence.spans.is_empty()
            || sentence.spans.iter().any(|span| {
                span.word as usize >= model.words.len()
                    || span.start > span.end
                    || span.end > sentence.reading.len()
                    || !sentence.reading.is_char_boundary(span.start)
                    || !sentence.reading.is_char_boundary(span.end)
            })
        {
            return Err(format!(
                "sentence {} has invalid or model-incompatible normalized spans",
                sentence.page_id
            )
            .into());
        }
    }
    Ok(())
}

fn select_slice<T: Clone>(
    items: &[T],
    start: usize,
    count: usize,
    label: &str,
) -> Result<Vec<T>, Box<dyn Error>> {
    let end = start
        .checked_add(count)
        .ok_or_else(|| format!("{label} sentence range overflows usize"))?;
    if items.len() < end {
        return Err(format!(
            "{label} has {} sentences; requested range {}..{}",
            items.len(),
            start,
            end
        )
        .into());
    }
    Ok(items[start..end].to_vec())
}

fn selection_end(
    start: usize,
    count: usize,
    stride: usize,
    label: &str,
) -> Result<usize, Box<dyn Error>> {
    if count == 0 || stride == 0 {
        return Err(format!("{label} count and stride must be positive").into());
    }
    let span = (count - 1)
        .checked_mul(stride)
        .and_then(|distance| distance.checked_add(1))
        .ok_or_else(|| format!("{label} range overflows usize"))?;
    start
        .checked_add(span)
        .ok_or_else(|| format!("{label} range overflows usize").into())
}

fn select_spaced<T: Clone>(
    items: &[T],
    start: usize,
    count: usize,
    stride: usize,
    label: &str,
) -> Result<Vec<T>, Box<dyn Error>> {
    let end = selection_end(start, count, stride, label)?;
    if stride == 1 {
        return select_slice(items, start, count, label);
    }
    if items.len() < end {
        return Err(format!(
            "{label} has {} sentences; requested {} rows starting at {} with stride {} (last index {})",
            items.len(),
            count,
            start,
            stride,
            end - 1
        )
        .into());
    }
    Ok((0..count)
        .map(|index| items[start + index * stride].clone())
        .collect())
}

fn select_training_sentences(
    blocks: &PageBlocks,
    requested: usize,
    seed: u64,
) -> Result<Vec<Sentence>, Box<dyn Error>> {
    let mut selected = Vec::with_capacity(requested);
    let total_blocks = blocks.blocks.len();
    for (block_index, block) in blocks.blocks.iter().enumerate() {
        let quota = ((block_index + 1) * requested / total_blocks)
            - (block_index * requested / total_blocks);
        if quota == 0 {
            continue;
        }
        if block.sentences < quota {
            return Err(format!(
                "page block {} has {} rows, fewer than its sample quota {quota}",
                block.file, block.sentences
            )
            .into());
        }
        let span = block.sentences - quota + 1;
        let offset = (mix64(seed ^ block_index as u64) as usize) % span;
        let end = offset + quota;
        let mut reader = blocks.open(block_index)?;
        let mut line = String::new();
        for row in 0..end {
            line.clear();
            if reader.read_line(&mut line)? == 0 {
                return Err(format!("page block {} ended early", block.file).into());
            }
            if row >= offset {
                let cached: CachedSentence = serde_json::from_str(&line)?;
                selected.push(cached.sentence);
            }
        }
        if (block_index + 1) % 10 == 0 || block_index + 1 == total_blocks {
            eprintln!(
                "llm_alpha phase=training_sample blocks={}/{} selected={}/{}",
                block_index + 1,
                total_blocks,
                selected.len(),
                requested
            );
        }
    }
    if selected.len() != requested {
        return Err(format!(
            "stratified page-block sampler selected {} sentences, expected {requested}",
            selected.len()
        )
        .into());
    }
    Ok(selected)
}

fn mix64(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn score_sentence(
    model: &Model,
    index: &Index,
    scorer: &LlamaCppScorer,
    sentence: &Sentence,
    limits: Limits,
    nbest_count: usize,
    expansion_limit: usize,
    inject_gold: bool,
) -> Result<ScoredExample, Box<dyn Error>> {
    let prepared = corpus::prepare_sentence(sentence, model, index, limits)?;
    let raw_paths = nbest(model, &prepared.graph, nbest_count, expansion_limit)?;
    let raw_cover = raw_paths
        .iter()
        .any(|path| path.nodes == prepared.graph.gold);
    let mut paths = raw_paths;
    if inject_gold && !raw_cover {
        paths.push(CrfPath {
            nodes: prepared.graph.gold.clone(),
            score: 0.0,
        });
    }
    let gold_candidate = paths
        .iter()
        .position(|path| path.nodes == prepared.graph.gold);
    if inject_gold && gold_candidate.is_none() {
        return Err("gold path was not present after training-candidate augmentation".into());
    }

    let mut unique_texts = Vec::<String>::new();
    let mut text_index = HashMap::<String, usize>::new();
    let mut candidate_texts = Vec::with_capacity(paths.len());
    for path in &paths {
        let text = path_surface(model, &prepared.reading, &prepared.graph.nodes, &path.nodes)?;
        let unique_index = *text_index.entry(text.clone()).or_insert_with(|| {
            unique_texts.push(text);
            unique_texts.len() - 1
        });
        candidate_texts.push(unique_index);
    }
    let prefixes = unique_texts
        .iter()
        .map(|text| CandidatePrefix::new(text.clone()))
        .collect::<Vec<_>>();
    let (unique_scores, batch_stats) =
        scorer.score_open_candidate_mean_softmax_with_stats("", &prefixes)?;
    if unique_scores.len() != unique_texts.len() {
        return Err("llama.cpp returned a different number of scores than candidates".into());
    }
    let mut candidates = Vec::with_capacity(paths.len());
    for (path_index, path) in paths.iter().enumerate() {
        let crf_score = model.path_score(&prepared.graph, &path.nodes)
            + model.trigram_score(&prepared.graph, &path.nodes);
        let llm_mean_softmax = f64::from(unique_scores[candidate_texts[path_index]]);
        if !crf_score.is_finite() || !llm_mean_softmax.is_finite() {
            return Err("non-finite CRF or LLM candidate score".into());
        }
        candidates.push(CandidateFeature {
            text: unique_texts[candidate_texts[path_index]].clone(),
            crf_score,
            llm_mean_softmax,
            gold: Some(path_index) == gold_candidate,
        });
    }
    Ok(ScoredExample {
        page_id: sentence.page_id,
        title: sentence.title.clone(),
        input: sentence.sentence.clone(),
        gold_text: sentence.sentence.clone(),
        gold_candidate,
        raw_nbest_gold_cover: raw_cover,
        llm_unique_candidate_count: batch_stats.candidate_count,
        llm_decode_batch_count: batch_stats.decode_batch_count,
        llm_max_sequences_per_decode_batch: batch_stats.max_sequences_per_decode_batch,
        candidates,
    })
}

fn path_surface(
    model: &Model,
    reading: &str,
    nodes: &[ai_engine_crf::Node],
    path: &[usize],
) -> Result<String, Box<dyn Error>> {
    let mut surface = String::new();
    for &node_index in path {
        let node = nodes
            .get(node_index)
            .ok_or("N-best path has invalid node index")?;
        if node.word == BOS || node.word == EOS {
            continue;
        }
        let word = model
            .words
            .get(node.word as usize)
            .ok_or("N-best path has invalid word ID")?;
        if word.model_only {
            let span = reading
                .get(node.start..node.end)
                .ok_or("model-only path node has invalid reading span")?;
            surface.push_str(span);
        } else {
            surface.push_str(&word.surface);
        }
    }
    if surface.is_empty() {
        return Err("N-best path rendered to an empty candidate".into());
    }
    Ok(surface)
}

fn rescore_example_llm(
    example: &mut ScoredExample,
    scorer: &LlamaCppScorer,
) -> Result<CandidateBatchStats, Box<dyn Error>> {
    let mut unique_texts = Vec::<String>::new();
    let mut text_indices = HashMap::<String, usize>::new();
    let mut candidate_indices = Vec::with_capacity(example.candidates.len());
    for candidate in &example.candidates {
        let index = if let Some(&index) = text_indices.get(&candidate.text) {
            index
        } else {
            let index = unique_texts.len();
            unique_texts.push(candidate.text.clone());
            text_indices.insert(candidate.text.clone(), index);
            index
        };
        candidate_indices.push(index);
    }
    let prefixes = unique_texts
        .iter()
        .map(|text| CandidatePrefix::new(text.clone()))
        .collect::<Vec<_>>();
    let (scores, stats) = scorer.score_open_candidate_mean_softmax_with_stats("", &prefixes)?;
    if scores.len() != unique_texts.len() {
        return Err("llama.cpp returned a different number of scores than saved candidates".into());
    }
    for (candidate, &index) in example.candidates.iter_mut().zip(&candidate_indices) {
        let score = f64::from(scores[index]);
        if !score.is_finite() {
            return Err("llama.cpp returned a non-finite score for a saved candidate".into());
        }
        candidate.llm_mean_softmax = score;
    }
    example.llm_unique_candidate_count = stats.candidate_count;
    example.llm_decode_batch_count = stats.decode_batch_count;
    example.llm_max_sequences_per_decode_batch = stats.max_sequences_per_decode_batch;
    Ok(stats)
}

fn fit_alpha_at_fixed_value(
    examples: &[ScoredExample],
    alpha: f64,
) -> Result<FitMetrics, Box<dyn Error>> {
    if examples.is_empty() {
        return Err("no alpha training examples".into());
    }
    if !alpha.is_finite() {
        return Err("fixed alpha is non-finite".into());
    }
    let crf_rms = pooled_centered_rms(examples, |candidate| candidate.crf_score);
    let llm_rms = pooled_centered_rms(examples, |candidate| candidate.llm_mean_softmax);
    if !crf_rms.is_finite() || !llm_rms.is_finite() {
        return Err("score normalization produced a non-finite RMS".into());
    }
    let (loss, top1) = objective_and_top1(examples, alpha, crf_rms, llm_rms)?;
    Ok(FitMetrics {
        training_sentences: examples.len(),
        gold_augmented_sentences: examples
            .iter()
            .filter(|example| !example.raw_nbest_gold_cover)
            .count(),
        crf_score_rms: crf_rms,
        feature_centering_rms: llm_rms,
        normalization:
            "per-sentence mean centering; pooled within-list RMS for CRF and LLM features"
                .to_owned(),
        initial_alpha: alpha,
        learned_alpha: alpha,
        iterations: 0,
        converged: true,
        listwise_nll_before: loss,
        listwise_nll_after: loss,
        listwise_top1_before: top1,
        listwise_top1_after: top1,
    })
}

fn fit_alpha(examples: &[ScoredExample], initial_alpha: f64) -> Result<FitMetrics, Box<dyn Error>> {
    if examples.is_empty() {
        return Err("no alpha training examples".into());
    }
    let crf_rms = pooled_centered_rms(examples, |candidate| candidate.crf_score);
    let llm_rms = pooled_centered_rms(examples, |candidate| candidate.llm_mean_softmax);
    if !crf_rms.is_finite() || !llm_rms.is_finite() {
        return Err("score normalization produced a non-finite RMS".into());
    }
    if llm_rms <= f64::EPSILON {
        let (loss, top1) = objective_and_top1(examples, 0.0, crf_rms, llm_rms)?;
        return Ok(FitMetrics {
            training_sentences: examples.len(),
            gold_augmented_sentences: examples
                .iter()
                .filter(|example| !example.raw_nbest_gold_cover)
                .count(),
            crf_score_rms: crf_rms,
            feature_centering_rms: llm_rms,
            normalization:
                "per-sentence mean centering; pooled within-list RMS for CRF and LLM features"
                    .to_owned(),
            initial_alpha,
            learned_alpha: 0.0,
            iterations: 0,
            converged: true,
            listwise_nll_before: loss,
            listwise_nll_after: loss,
            listwise_top1_before: top1,
            listwise_top1_after: top1,
        });
    }

    let mut alpha = 0.0_f64;
    let (loss_before, top1_before) = objective_and_top1(examples, alpha, crf_rms, llm_rms)?;
    let mut iterations = 0;
    let mut converged = false;
    for iteration in 0..100 {
        let (current_loss, gradient, hessian) =
            objective_gradient_hessian(examples, alpha, crf_rms, llm_rms)?;
        if gradient.abs() < 1e-8 {
            converged = true;
            iterations = iteration;
            break;
        }
        if !hessian.is_finite() || hessian <= 1e-14 {
            iterations = iteration;
            break;
        }
        let step = (-gradient / hessian).clamp(-100.0, 100.0);
        let mut fraction = 1.0;
        let mut accepted = false;
        while fraction >= 1.0 / 1_048_576.0 {
            let candidate_alpha = (alpha + fraction * step).clamp(-100_000.0, 100_000.0);
            let (candidate_loss, _) =
                objective_and_top1(examples, candidate_alpha, crf_rms, llm_rms)?;
            if candidate_loss <= current_loss + 1e-12 {
                let improvement = current_loss - candidate_loss;
                alpha = candidate_alpha;
                accepted = true;
                if improvement < 1e-12 || fraction * step.abs() < 1e-8 {
                    converged = true;
                }
                break;
            }
            fraction *= 0.5;
        }
        iterations = iteration + 1;
        if !accepted || converged {
            break;
        }
    }
    let learned_alpha = alpha;
    if !learned_alpha.is_finite() {
        return Err("scalar alpha optimizer produced a non-finite parameter".into());
    }
    let (loss_after, top1_after) = objective_and_top1(examples, alpha, crf_rms, llm_rms)?;
    Ok(FitMetrics {
        training_sentences: examples.len(),
        gold_augmented_sentences: examples
            .iter()
            .filter(|example| !example.raw_nbest_gold_cover)
            .count(),
        crf_score_rms: crf_rms,
        feature_centering_rms: llm_rms,
        normalization:
            "per-sentence mean centering; pooled within-list RMS for CRF and LLM features"
                .to_owned(),
        initial_alpha,
        learned_alpha,
        iterations,
        converged,
        listwise_nll_before: loss_before,
        listwise_nll_after: loss_after,
        listwise_top1_before: top1_before,
        listwise_top1_after: top1_after,
    })
}

fn objective_gradient_hessian(
    examples: &[ScoredExample],
    alpha: f64,
    crf_rms: f64,
    llm_rms: f64,
) -> Result<(f64, f64, f64), Box<dyn Error>> {
    let mut loss_sum = 0.0;
    let mut gradient_sum = 0.0;
    let mut hessian_sum = 0.0;
    for example in examples {
        let crf_list_mean = crf_mean(example);
        let llm_list_mean = feature_mean(example);
        let scores = example
            .candidates
            .iter()
            .map(|candidate| {
                normalize(candidate.crf_score, crf_list_mean, crf_rms)
                    + alpha * normalize(candidate.llm_mean_softmax, llm_list_mean, llm_rms)
            })
            .collect::<Vec<_>>();
        let normalizer = log_sum_exp(&scores)?;
        let probabilities = scores
            .iter()
            .map(|score| (*score - normalizer).exp())
            .collect::<Vec<_>>();
        let gold_candidate = example
            .gold_candidate
            .ok_or("alpha fitting requires a gold candidate in every training set")?;
        let gold_score = scores[gold_candidate];
        let gold_z = normalize(
            example.candidates[gold_candidate].llm_mean_softmax,
            llm_list_mean,
            llm_rms,
        );
        let mut expected_z = 0.0;
        let mut expected_z2 = 0.0;
        for (candidate, probability) in example.candidates.iter().zip(probabilities) {
            let z = normalize(candidate.llm_mean_softmax, llm_list_mean, llm_rms);
            expected_z += probability * z;
            expected_z2 += probability * z * z;
        }
        loss_sum += normalizer - gold_score;
        gradient_sum += expected_z - gold_z;
        hessian_sum += (expected_z2 - expected_z * expected_z).max(0.0);
    }
    let count = examples.len() as f64;
    Ok((loss_sum / count, gradient_sum / count, hessian_sum / count))
}

fn objective_and_top1(
    examples: &[ScoredExample],
    alpha: f64,
    crf_rms: f64,
    llm_rms: f64,
) -> Result<(f64, f64), Box<dyn Error>> {
    let mut loss = 0.0;
    let mut correct = 0_usize;
    for example in examples {
        let crf_list_mean = crf_mean(example);
        let llm_list_mean = feature_mean(example);
        let scores = example
            .candidates
            .iter()
            .map(|candidate| {
                normalize(candidate.crf_score, crf_list_mean, crf_rms)
                    + alpha * normalize(candidate.llm_mean_softmax, llm_list_mean, llm_rms)
            })
            .collect::<Vec<_>>();
        let normalizer = log_sum_exp(&scores)?;
        let gold_candidate = example
            .gold_candidate
            .ok_or("alpha fitting requires a gold candidate in every training set")?;
        loss += normalizer - scores[gold_candidate];
        if argmax(&scores) == gold_candidate {
            correct += 1;
        }
    }
    Ok((
        loss / examples.len() as f64,
        correct as f64 / examples.len() as f64,
    ))
}

fn feature_mean(example: &ScoredExample) -> f64 {
    example
        .candidates
        .iter()
        .map(|candidate| candidate.llm_mean_softmax)
        .sum::<f64>()
        / example.candidates.len() as f64
}

fn crf_mean(example: &ScoredExample) -> f64 {
    example
        .candidates
        .iter()
        .map(|candidate| candidate.crf_score)
        .sum::<f64>()
        / example.candidates.len() as f64
}

fn pooled_centered_rms(
    examples: &[ScoredExample],
    value: impl Fn(&CandidateFeature) -> f64,
) -> f64 {
    (examples
        .iter()
        .map(|example| {
            let mean = example.candidates.iter().map(&value).sum::<f64>()
                / example.candidates.len() as f64;
            example
                .candidates
                .iter()
                .map(|candidate| (value(candidate) - mean).powi(2))
                .sum::<f64>()
                / example.candidates.len() as f64
        })
        .sum::<f64>()
        / examples.len() as f64)
        .sqrt()
}

fn normalize(value: f64, list_mean: f64, pooled_rms: f64) -> f64 {
    if pooled_rms <= f64::EPSILON {
        0.0
    } else {
        (value - list_mean) / pooled_rms
    }
}

fn log_sum_exp(values: &[f64]) -> Result<f64, Box<dyn Error>> {
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !max.is_finite() {
        return Err("candidate scores are non-finite".into());
    }
    let sum = values.iter().map(|value| (*value - max).exp()).sum::<f64>();
    if !sum.is_finite() || sum <= 0.0 {
        return Err("candidate score normalization failed".into());
    }
    Ok(max + sum.ln())
}

fn argmax(values: &[f64]) -> usize {
    let mut best = 0;
    for index in 1..values.len() {
        if values[index] > values[best] {
            best = index;
        }
    }
    best
}

fn evaluate(
    examples: &[ScoredExample],
    fit: &FitMetrics,
    requested_batch_size: usize,
) -> Result<EvaluationMetrics, Box<dyn Error>> {
    if examples.is_empty() {
        return Err("no fixed test examples".into());
    }
    let mut cover = 0_usize;
    let mut bigram_correct = 0_usize;
    let mut crf_correct = 0_usize;
    let mut llm_correct = 0_usize;
    let mut bigram_edits = 0_usize;
    let mut crf_edits = 0_usize;
    let mut llm_edits = 0_usize;
    let mut bigram_denominator = 0_usize;
    let mut crf_denominator = 0_usize;
    let mut llm_denominator = 0_usize;
    let mut unique_candidate_texts = 0_usize;
    let mut decode_batch_count = 0_usize;
    let mut max_sequences_per_decode_batch = 0_usize;
    let mut full_batch_sentences = 0_usize;
    for example in examples {
        cover += usize::from(example.raw_nbest_gold_cover);
        unique_candidate_texts += example.llm_unique_candidate_count;
        decode_batch_count += example.llm_decode_batch_count;
        max_sequences_per_decode_batch =
            max_sequences_per_decode_batch.max(example.llm_max_sequences_per_decode_batch);
        full_batch_sentences +=
            usize::from(example.llm_max_sequences_per_decode_batch >= requested_batch_size);
        let bigram_best = 0; // nbest() returns the first-order CRF paths in score order.
        let crf_list_mean = crf_mean(example);
        let llm_list_mean = feature_mean(example);
        let crf_scores = example
            .candidates
            .iter()
            .map(|candidate| normalize(candidate.crf_score, crf_list_mean, fit.crf_score_rms))
            .collect::<Vec<_>>();
        let combined_scores = example
            .candidates
            .iter()
            .map(|candidate| {
                normalize(candidate.crf_score, crf_list_mean, fit.crf_score_rms)
                    + fit.learned_alpha
                        * normalize(
                            candidate.llm_mean_softmax,
                            llm_list_mean,
                            fit.feature_centering_rms,
                        )
            })
            .collect::<Vec<_>>();
        let crf_best = argmax(&crf_scores);
        let llm_best = argmax(&combined_scores);
        bigram_correct += usize::from(example.gold_candidate == Some(bigram_best));
        crf_correct += usize::from(example.gold_candidate == Some(crf_best));
        llm_correct += usize::from(example.gold_candidate == Some(llm_best));
        let gold_chars = example.gold_text.chars().collect::<Vec<_>>();
        let bigram_chars = candidate_chars(&example.candidates[bigram_best].text);
        let crf_chars = candidate_chars(&example.candidates[crf_best].text);
        let llm_chars = candidate_chars(&example.candidates[llm_best].text);
        bigram_edits += levenshtein(&bigram_chars, &gold_chars);
        crf_edits += levenshtein(&crf_chars, &gold_chars);
        llm_edits += levenshtein(&llm_chars, &gold_chars);
        bigram_denominator += gold_chars.len().max(bigram_chars.len()).max(1);
        crf_denominator += gold_chars.len().max(crf_chars.len()).max(1);
        llm_denominator += gold_chars.len().max(llm_chars.len()).max(1);
    }
    let count = examples.len() as f64;
    Ok(EvaluationMetrics {
        sentence_count: examples.len(),
        raw_nbest_gold_cover_rate: cover as f64 / count,
        bigram_top1_accuracy: bigram_correct as f64 / count,
        crf_post_trigram_top1_accuracy: crf_correct as f64 / count,
        llm_crf_top1_accuracy: llm_correct as f64 / count,
        bigram_character_level_accuracy: 1.0
            - bigram_edits as f64 / bigram_denominator.max(1) as f64,
        crf_post_trigram_character_level_accuracy: 1.0
            - crf_edits as f64 / crf_denominator.max(1) as f64,
        llm_crf_character_level_accuracy: 1.0 - llm_edits as f64 / llm_denominator.max(1) as f64,
        requested_llm_sequences_per_batch: requested_batch_size,
        total_llm_unique_candidate_texts: unique_candidate_texts,
        total_llm_decode_batches: decode_batch_count,
        max_llm_sequences_per_decode_batch: max_sequences_per_decode_batch,
        sentences_reaching_requested_llm_batch: full_batch_sentences,
        alpha: fit.learned_alpha,
    })
}

fn candidate_chars(text: &str) -> Vec<char> {
    text.chars().collect()
}

fn levenshtein(left: &[char], right: &[char]) -> usize {
    if left.is_empty() {
        return right.len();
    }
    if right.is_empty() {
        return left.len();
    }
    let (short, long) = if left.len() <= right.len() {
        (left, right)
    } else {
        (right, left)
    };
    let mut previous = (0..=short.len()).collect::<Vec<_>>();
    let mut current = vec![0; short.len() + 1];
    for (row, long_char) in long.iter().enumerate() {
        current[0] = row + 1;
        for (column, short_char) in short.iter().enumerate() {
            current[column + 1] = (previous[column + 1] + 1)
                .min(current[column] + 1)
                .min(previous[column] + usize::from(long_char != short_char));
        }
        std::mem::swap(&mut previous, &mut current);
    }
    previous[short.len()]
}

fn create_new(path: &FsPath) -> Result<File, Box<dyn Error>> {
    Ok(OpenOptions::new().write(true).create_new(true).open(path)?)
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &FsPath) -> Result<T, Box<dyn Error>> {
    Ok(serde_json::from_reader(BufReader::new(File::open(path)?))?)
}

fn read_jsonl<T: for<'de> Deserialize<'de>>(path: &FsPath) -> Result<Vec<T>, Box<dyn Error>> {
    let reader = BufReader::new(File::open(path)?);
    let mut values = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if !line.trim().is_empty() {
            values.push(serde_json::from_str(&line)?);
        }
    }
    Ok(values)
}

fn write_jsonl<T: Serialize>(path: &FsPath, values: &[T]) -> Result<(), Box<dyn Error>> {
    let file = create_new(path)?;
    let mut writer = BufWriter::new(file);
    for value in values {
        serde_json::to_writer(&mut writer, value)?;
        writeln!(writer)?;
    }
    writer.flush()?;
    Ok(())
}

fn write_json(path: &FsPath, value: &impl Serialize) -> Result<(), Box<dyn Error>> {
    let file = create_new(path)?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, value)?;
    writeln!(writer)?;
    writer.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        CandidateFeature, FitMetrics, ScoredExample, evaluate, fit_alpha, levenshtein,
        select_slice, select_spaced,
    };

    #[test]
    fn selects_middle_training_and_test_windows() {
        let training = (0..2_000).collect::<Vec<_>>();
        assert_eq!(
            select_slice(&training, 500, 1_000, "training").unwrap(),
            (500..1_500).collect::<Vec<_>>()
        );
        let test = (0..5_000).collect::<Vec<_>>();
        assert_eq!(
            select_slice(&test, 2_250, 500, "test").unwrap(),
            (2_250..2_750).collect::<Vec<_>>()
        );
    }

    #[test]
    fn slice_rejects_short_input_and_overflow() {
        assert!(select_slice(&[1, 2, 3], 2, 2, "sample").is_err());
        assert!(select_slice(&[1], usize::MAX, 2, "sample").is_err());
    }

    #[test]
    fn selects_evenly_spaced_training_and_test_rows() {
        let training = (0..2_000).collect::<Vec<_>>();
        assert_eq!(
            select_spaced(&training, 0, 1_000, 2, "training").unwrap(),
            (0..2_000).step_by(2).collect::<Vec<_>>()
        );
        let test = (0..5_000).collect::<Vec<_>>();
        assert_eq!(
            select_spaced(&test, 0, 500, 10, "test").unwrap(),
            (0..5_000).step_by(10).collect::<Vec<_>>()
        );
    }

    #[test]
    fn spaced_selection_rejects_zero_stride_and_short_input() {
        assert!(select_spaced(&[1, 2], 0, 2, 0, "test").is_err());
        assert!(select_spaced(&[1, 2], 0, 2, 2, "test").is_err());
        assert!(select_spaced(&[1], usize::MAX, 2, 1, "test").is_err());
    }

    #[test]
    fn scalar_alpha_uses_only_the_llm_feature_and_improves_listwise_fit() {
        let examples = (0..4)
            .map(|index| ScoredExample {
                page_id: index,
                title: String::new(),
                input: String::new(),
                gold_text: String::new(),
                gold_candidate: Some(1),
                raw_nbest_gold_cover: index % 2 == 0,
                llm_unique_candidate_count: 2,
                llm_decode_batch_count: 1,
                llm_max_sequences_per_decode_batch: 2,
                candidates: vec![
                    CandidateFeature {
                        text: "incorrect".to_owned(),
                        crf_score: 0.5,
                        llm_mean_softmax: 0.1,
                        gold: false,
                    },
                    CandidateFeature {
                        text: "correct".to_owned(),
                        crf_score: 0.0,
                        llm_mean_softmax: 0.8,
                        gold: true,
                    },
                ],
            })
            .collect::<Vec<_>>();
        let fit = fit_alpha(&examples, 0.0).expect("scalar fit");
        assert!(fit.learned_alpha > 0.0);
        assert!(fit.listwise_nll_after < fit.listwise_nll_before);
        assert!(fit.listwise_top1_after > fit.listwise_top1_before);
        assert_eq!(fit.training_sentences, 4);
    }

    #[test]
    fn character_error_uses_unicode_scalar_levenshtein_distance() {
        assert_eq!(levenshtein(&['あ', 'い'], &['あ', 'う']), 1);
        assert_eq!(levenshtein(&['猫'], &['猫', '犬']), 1);
    }

    #[test]
    fn held_out_gold_missing_from_nbest_counts_as_miss_without_failing() {
        let example = ScoredExample {
            page_id: 1,
            title: String::new(),
            input: "せいかい".to_owned(),
            gold_text: "正解".to_owned(),
            gold_candidate: None,
            raw_nbest_gold_cover: false,
            llm_unique_candidate_count: 2,
            llm_decode_batch_count: 1,
            llm_max_sequences_per_decode_batch: 2,
            candidates: vec![
                CandidateFeature {
                    text: "誤変換".to_owned(),
                    crf_score: 2.0,
                    llm_mean_softmax: 0.1,
                    gold: false,
                },
                CandidateFeature {
                    text: "正しい".to_owned(),
                    crf_score: 1.0,
                    llm_mean_softmax: 0.9,
                    gold: false,
                },
            ],
        };
        let fit = FitMetrics {
            training_sentences: 1000,
            gold_augmented_sentences: 0,
            crf_score_rms: 1.0,
            feature_centering_rms: 1.0,
            normalization: "test".to_owned(),
            initial_alpha: 0.0,
            learned_alpha: 1.0,
            iterations: 0,
            converged: true,
            listwise_nll_before: 0.0,
            listwise_nll_after: 0.0,
            listwise_top1_before: 0.0,
            listwise_top1_after: 0.0,
        };
        let metrics = evaluate(&[example], &fit, 2).expect("held-out evaluation");
        assert_eq!(metrics.raw_nbest_gold_cover_rate, 0.0);
        assert_eq!(metrics.bigram_top1_accuracy, 0.0);
        assert_eq!(metrics.crf_post_trigram_top1_accuracy, 0.0);
        assert_eq!(metrics.llm_crf_top1_accuracy, 0.0);
        assert_eq!(metrics.total_llm_unique_candidate_texts, 2);
        assert_eq!(metrics.total_llm_decode_batches, 1);
        assert_eq!(metrics.max_llm_sequences_per_decode_batch, 2);
        assert_eq!(metrics.sentences_reaching_requested_llm_batch, 1);
    }
}
