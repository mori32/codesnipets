//! Bounded, reproducible offline CRF experiment on tagged `MeCab` JSONL.
#![allow(clippy::cast_precision_loss, clippy::too_many_lines)]
#![recursion_limit = "256"]
use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use ai_engine_crf::blocks::{self, CachedSentence as Buffered, PageBlocks};
use ai_engine_crf::corpus::{self, Index, Limits, Prepared, Sentence};
use ai_engine_crf::cutoff;
use ai_engine_crf::hogwild as hogwild_crf;
use ai_engine_crf::parallel::{BatchConfig, BatchTiming, Trainer};
use ai_engine_crf::{Feature, Model, ParameterLimit, Result, first_nll, nbest, output, rerank};
use serde::{Deserialize, Serialize};
mod error_analysis;
mod resume;
use resume::Resume;

#[derive(Debug, Deserialize, Serialize)]
#[allow(clippy::struct_excessive_bools)]
struct Options {
    input: PathBuf,
    corpus_statistics: PathBuf,
    system_dictionary: PathBuf,
    user_additions: PathBuf,
    output: PathBuf,
    #[serde(default)]
    work_dir: PathBuf,
    #[serde(default)]
    model_in: Option<PathBuf>,
    #[serde(default)]
    evaluation_sample_in: Option<PathBuf>,
    #[serde(default)]
    error_trigram_analysis: bool,
    #[serde(default)]
    feature_ablation_evaluation: bool,
    #[serde(default)]
    prepared_data_in: Option<PathBuf>,
    epochs: usize,
    max_sentences: usize,
    max_validation_sentences: usize,
    max_records: usize,
    skip_records: usize,
    max_words: usize,
    max_nodes: usize,
    max_edges: usize,
    max_astar_expansions: usize,
    nbest: usize,
    learning_rate: f64,
    l2: f64,
    #[serde(default)]
    primary_nbest_margin: Option<f64>,
    #[serde(default)]
    primary_nbest_listwise: bool,
    missing_pos_probability: f64,
    seed: u64,
    shuffle_buffer: usize,
    max_eval_sentences: usize,
    max_parameter_mib: usize,
    checkpoint_every: usize,
    #[serde(default = "default_recovery_interval")]
    recovery_interval: usize,
    threads: usize,
    batch_size: usize,
    page_block_size: usize,
    save_initial_model: bool,
    full_corpus: bool,
    #[serde(default)]
    reference_kernels: bool,
    #[serde(default)]
    hogwild: bool,
    #[serde(default)]
    gold_path_augmentation: bool,
    #[serde(default)]
    missed_feedback: bool,
    #[serde(default)]
    test_prefix_sentences: usize,
    #[serde(default)]
    resume_from: Option<PathBuf>,
    #[serde(default)]
    allow_resume_batch_size_change: bool,
    #[serde(default)]
    delta_cutoff: f64,
    #[serde(default = "default_delta_cutoff_max")]
    delta_cutoff_max: f64,
    #[serde(default)]
    initial_bigram_count: usize,
    #[serde(default = "default_initial_bigram_delta")]
    initial_bigram_delta: f64,
    #[serde(default)]
    initial_trigram_count: usize,
    #[serde(default = "default_initial_trigram_weight")]
    initial_trigram_weight: f64,
    #[serde(default = "default_progress_interval")]
    progress_interval: usize,
    #[serde(default = "default_progress_interval_seconds")]
    progress_interval_seconds: u64,
}

fn default_delta_cutoff_max() -> f64 {
    1e-2
}

fn default_initial_bigram_delta() -> f64 {
    0.9
}

fn default_initial_trigram_weight() -> f64 {
    0.1
}

fn default_progress_interval() -> usize {
    250_000
}

fn default_progress_interval_seconds() -> u64 {
    1_800
}

fn default_recovery_interval() -> usize {
    250_000
}

fn primary_objective(o: &Options) -> &'static str {
    if o.primary_nbest_listwise {
        "raw-nbest-listwise"
    } else if o.primary_nbest_margin.is_some() {
        "raw-nbest-all-pairwise-hinge"
    } else {
        "full-lattice-crf-nll"
    }
}

fn primary_objective_definition(o: &Options) -> &'static str {
    if o.primary_nbest_listwise {
        "Listwise softmax over raw N-best plus the gold path when absent. Minimize logsumexp(primary_score(candidate)) - primary_score(gold); update primary features with expected candidate feature counts minus gold feature counts. Raw N-best hit is measured before gold inclusion."
    } else if o.primary_nbest_margin.is_some() {
        "For every non-gold path in raw N-best, add max(0, margin + score(path) - score(gold)). Sum active pairwise losses and gradients; each active gradient is features(path)-features(gold)."
    } else {
        "Full-lattice log-partition negative log likelihood."
    }
}

fn options() -> Result<Options> {
    let mut o = Options {
        input: PathBuf::new(),
        corpus_statistics: PathBuf::new(),
        system_dictionary: PathBuf::from("C:/Program Files/MeCab/dic/ipadic"),
        user_additions: PathBuf::from("dictionary/additions"),
        output: PathBuf::new(),
        work_dir: PathBuf::new(),
        model_in: None,
        evaluation_sample_in: None,
        error_trigram_analysis: false,
        feature_ablation_evaluation: false,
        prepared_data_in: None,
        epochs: 5,
        max_sentences: 1000,
        max_validation_sentences: 200,
        max_records: 20000,
        skip_records: 0,
        max_words: 48,
        max_nodes: 20000,
        max_edges: 1_000_000,
        max_astar_expansions: 500_000,
        nbest: 256,
        learning_rate: 0.03,
        l2: 0.001,
        primary_nbest_margin: None,
        primary_nbest_listwise: false,
        missing_pos_probability: 1e-8,
        seed: 20_261_003,
        shuffle_buffer: 256,
        max_eval_sentences: 256,
        max_parameter_mib: 512,
        checkpoint_every: 0,
        recovery_interval: default_recovery_interval(),
        threads: std::thread::available_parallelism().map_or(1, |n| n.get().min(16)),
        batch_size: 16,
        page_block_size: 1000,
        save_initial_model: false,
        full_corpus: false,
        reference_kernels: false,
        hogwild: false,
        gold_path_augmentation: false,
        missed_feedback: false,
        test_prefix_sentences: 0,
        resume_from: None,
        allow_resume_batch_size_change: false,
        delta_cutoff: 0.0,
        delta_cutoff_max: default_delta_cutoff_max(),
        initial_bigram_count: 0,
        initial_bigram_delta: default_initial_bigram_delta(),
        initial_trigram_count: 0,
        initial_trigram_weight: default_initial_trigram_weight(),
        progress_interval: default_progress_interval(),
        progress_interval_seconds: default_progress_interval_seconds(),
    };
    let mut args = std::env::args().skip(1);
    while let Some(name) = args.next() {
        if name == "--help" || name == "-h" {
            println!(
                "ai_engine_crf --input TAGGED.jsonl.bz2 --corpus-statistics GLOBAL_DIR --output NEW_DIR --work-dir WORK_DIR\n\
                [--system-dictionary IPADIC_DIR] [--user-additions ADDITIONS_DIR] [--model-in model.json]\n\
                [--evaluation-sample-in normalized-sentences.json] (fixed-sample evaluation; requires --epochs 0 and --model-in)\n\
                [--error-trigram-analysis] (write char-diff-focused gold/predicted trigram rankings during fixed-sample evaluation)\n\
                [--feature-ablation-evaluation] (compare fixed-sample metrics with selected read-length/POS features disabled)\n\
                [--prepared-data-in prior-run-output] (reuse its normalized page cache and fixed samples; start weights from --model-in)\n\
                [--epochs 5] [--max-sentences 1000] [--max-validation-sentences 200] [--max-records 20000]\n\
                [--skip-records 0] [--max-words 48] [--nbest 256] [--learning-rate 0.03] [--l2 0.001]\n\
                [--primary-nbest-margin 0.0] (sum pairwise hinge over raw N-best paths; 0 means stop at a tie with gold)\n\
                [--primary-nbest-listwise] (softmax candidate-set loss over raw N-best plus gold if absent)\n\
                [--max-nodes 20000] [--max-edges 1000000] [--max-astar-expansions 500000]\n\
                [--missing-pos-probability 1e-8] [--seed 20261003]\n\
                [--shuffle-buffer 256] [--max-eval-sentences 256] [--max-parameter-mib 512] [--checkpoint-every 0]\n\
                [--recovery-interval 250000] (persist exact recovery state after this many committed sentences)\n\
                [--threads AVAILABLE_UP_TO_16] [--batch-size 16]\n\
                [--page-block-size 1000] (0 disables page-block caching/shuffling)\n\
                [--save-initial-model] (save the immutable run-entry model for controlled comparisons)\n\
                [--full-corpus] (read to EOF; hold out entire pages; bounded whole-corpus evaluation samples)\n\
                [--reference-kernels] (diagnostic implementation for controlled comparisons)\n\
                [--hogwild] (experimental asynchronous per-sentence scc updates; bounded runs only)\n\
                [--gold-path-augmentation] (training only: append gold if raw N-best misses it)\n\
                [--missed-feedback] (mine final reranked top-1 errors for the next epoch)\n\
                [--test-prefix-sentences 5000] (reserve the first N accepted sentences as fixed test data)\n\
                [--resume-from STOPPED_RUN_DIR] (reuse exact page cache, pending window and RNG)\n\
                [--allow-resume-batch-size-change] (explicitly permit re-batching the saved pending window)\n\
                [--delta-cutoff 0] (0 disables pressure cutoff; suggested enabled value 1e-4)\n\
                [--delta-cutoff-max 1e-2] (maximum pressure threshold; retained-table target 75% of budget)\n\
                [--initial-bigram-count 0] (seed this many highest-count static word-bigram corrections)\n\
                [--initial-bigram-delta 0.9] (weight on POS/unigram baseline; empirical bigram gets 1-delta)\n\
                [--progress-interval 250000] (fixed-heldout N-best hit-rate evaluation interval)\n\
                [--progress-interval-seconds 1800] (also evaluate at this wall-clock interval)\n\
                [--initial-trigram-count 0] (seed this many highest-count static word-trigram corrections)\n\
                [--initial-trigram-weight 0.1] (weight on empirical trigram; remainder backs off to bigram)\n\
                Larger scores are better. Existing output files are never overwritten. No LLM/server required."
            );
            std::process::exit(0);
        }
        if name == "--save-initial-model" {
            o.save_initial_model = true;
            continue;
        }
        if name == "--full-corpus" {
            o.full_corpus = true;
            continue;
        }
        if name == "--reference-kernels" {
            o.reference_kernels = true;
            continue;
        }
        if name == "--hogwild" {
            o.hogwild = true;
            continue;
        }
        if name == "--gold-path-augmentation" {
            o.gold_path_augmentation = true;
            continue;
        }
        if name == "--error-trigram-analysis" {
            o.error_trigram_analysis = true;
            continue;
        }
        if name == "--feature-ablation-evaluation" {
            o.feature_ablation_evaluation = true;
            continue;
        }
        if name == "--missed-feedback" {
            o.missed_feedback = true;
            continue;
        }
        if name == "--primary-nbest-listwise" {
            o.primary_nbest_listwise = true;
            continue;
        }
        if name == "--allow-resume-batch-size-change" {
            o.allow_resume_batch_size_change = true;
            continue;
        }
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for {name}"))?;
        match name.as_str() {
            "--input" => o.input = value.into(),
            "--corpus-statistics" => o.corpus_statistics = value.into(),
            "--system-dictionary" => o.system_dictionary = value.into(),
            "--user-additions" => o.user_additions = value.into(),
            "--output" => o.output = value.into(),
            "--work-dir" => o.work_dir = value.into(),
            "--model-in" => o.model_in = Some(value.into()),
            "--evaluation-sample-in" => o.evaluation_sample_in = Some(value.into()),
            "--prepared-data-in" => o.prepared_data_in = Some(value.into()),
            "--resume-from" => o.resume_from = Some(value.into()),
            "--epochs" => o.epochs = value.parse()?,
            "--max-sentences" => o.max_sentences = value.parse()?,
            "--max-validation-sentences" => o.max_validation_sentences = value.parse()?,
            "--max-records" => o.max_records = value.parse()?,
            "--skip-records" => o.skip_records = value.parse()?,
            "--max-words" => o.max_words = value.parse()?,
            "--max-nodes" => o.max_nodes = value.parse()?,
            "--max-edges" => o.max_edges = value.parse()?,
            "--max-astar-expansions" => o.max_astar_expansions = value.parse()?,
            "--nbest" => o.nbest = value.parse()?,
            "--learning-rate" => o.learning_rate = value.parse()?,
            "--l2" => o.l2 = value.parse()?,
            "--primary-nbest-margin" => o.primary_nbest_margin = Some(value.parse()?),
            "--missing-pos-probability" => o.missing_pos_probability = value.parse()?,
            "--seed" => o.seed = value.parse()?,
            "--shuffle-buffer" => o.shuffle_buffer = value.parse()?,
            "--max-eval-sentences" => o.max_eval_sentences = value.parse()?,
            "--max-parameter-mib" => o.max_parameter_mib = value.parse()?,
            "--checkpoint-every" => o.checkpoint_every = value.parse()?,
            "--recovery-interval" => o.recovery_interval = value.parse()?,
            "--threads" => o.threads = value.parse()?,
            "--batch-size" => o.batch_size = value.parse()?,
            "--page-block-size" => o.page_block_size = value.parse()?,
            "--delta-cutoff" => o.delta_cutoff = value.parse()?,
            "--delta-cutoff-max" => o.delta_cutoff_max = value.parse()?,
            "--initial-bigram-count" => o.initial_bigram_count = value.parse()?,
            "--initial-bigram-delta" => o.initial_bigram_delta = value.parse()?,
            "--initial-trigram-count" => o.initial_trigram_count = value.parse()?,
            "--initial-trigram-weight" => o.initial_trigram_weight = value.parse()?,
            "--progress-interval" => o.progress_interval = value.parse()?,
            "--progress-interval-seconds" => o.progress_interval_seconds = value.parse()?,
            "--test-prefix-sentences" => o.test_prefix_sentences = value.parse()?,
            _ => return Err(format!("unknown option {name}").into()),
        }
    }
    if let Some(directory) = &o.resume_from {
        if o.model_in.is_some() {
            return Err("--resume-from cannot be combined with --model-in".into());
        }
        o.model_in = Some(Resume::resolve_checkpoint(directory)?.join("stopped-model.json"));
    }
    if o.allow_resume_batch_size_change && o.resume_from.is_none() {
        return Err("--allow-resume-batch-size-change requires --resume-from".into());
    }
    if o.work_dir.as_os_str().is_empty() && !o.output.as_os_str().is_empty() {
        o.work_dir = o.output.join("work");
    }
    if o.model_in.is_some() && (o.initial_bigram_count > 0 || o.initial_trigram_count > 0) {
        return Err("initial n-gram counts cannot be combined with --model-in".into());
    }
    if o.input.as_os_str().is_empty()
        || o.output.as_os_str().is_empty()
        || o.work_dir.as_os_str().is_empty()
        || (o.model_in.is_none() && o.corpus_statistics.as_os_str().is_empty())
    {
        return Err("required: --input, --output, and --corpus-statistics (or --model-in)".into());
    }
    if o.evaluation_sample_in.is_some()
        && (o.model_in.is_none() || o.epochs != 0 || o.resume_from.is_some())
    {
        return Err(
            "--evaluation-sample-in requires --model-in, --epochs 0, and no --resume-from".into(),
        );
    }
    if o.evaluation_sample_in.is_some() && o.primary_nbest_margin.is_some() {
        return Err("--primary-nbest-margin is a training objective and cannot be combined with fixed-sample evaluation-only mode".into());
    }
    if o.evaluation_sample_in.is_some() && o.primary_nbest_listwise {
        return Err("--primary-nbest-listwise is a training objective and cannot be combined with fixed-sample evaluation-only mode".into());
    }
    if o.error_trigram_analysis && o.evaluation_sample_in.is_none() {
        return Err("--error-trigram-analysis requires --evaluation-sample-in".into());
    }
    if o.feature_ablation_evaluation && o.evaluation_sample_in.is_none() {
        return Err("--feature-ablation-evaluation requires --evaluation-sample-in".into());
    }
    if o.prepared_data_in.is_some()
        && (o.resume_from.is_some()
            || o.evaluation_sample_in.is_some()
            || o.hogwild
            || !o.full_corpus
            || o.test_prefix_sentences == 0
            || o.page_block_size == 0)
    {
        return Err("--prepared-data-in requires a fresh synchronous full-corpus run with page blocks and a fixed test prefix".into());
    }
    if o.hogwild
        && (o.full_corpus
            || o.resume_from.is_some()
            || o.delta_cutoff > 0.0
            || o.max_sentences > 10_000)
    {
        return Err(
            "experimental --hogwild currently requires a fresh run of at most 10,000 sentences, without --full-corpus, resume, or delta cutoff".into(),
        );
    }
    if o.primary_nbest_margin.is_some() && (o.hogwild || o.reference_kernels) {
        return Err("--primary-nbest-margin requires the optimized synchronous trainer and cannot be combined with --hogwild or --reference-kernels".into());
    }
    if o.primary_nbest_listwise && (o.hogwild || o.reference_kernels) {
        return Err("--primary-nbest-listwise requires the optimized synchronous trainer and cannot be combined with --hogwild or --reference-kernels".into());
    }
    if o.primary_nbest_listwise && o.primary_nbest_margin.is_some() {
        return Err(
            "--primary-nbest-listwise cannot be combined with --primary-nbest-margin".into(),
        );
    }
    if o.missed_feedback
        && (o.test_prefix_sentences == 0
            || o.hogwild
            || o.epochs < 2
            || o.page_block_size == 0
            || !o.full_corpus)
    {
        return Err("--missed-feedback requires --full-corpus, --page-block-size > 0, --test-prefix-sentences, synchronous training, and at least two epochs".into());
    }
    if o.full_corpus {
        if o.skip_records != 0 {
            return Err("--full-corpus cannot be combined with --skip-records".into());
        }
        o.max_sentences = usize::MAX;
        o.max_records = usize::MAX;
    }
    if [
        o.max_sentences,
        o.max_validation_sentences,
        o.max_records,
        o.max_words,
        o.max_nodes,
        o.max_edges,
        o.max_astar_expansions,
        o.nbest,
        o.shuffle_buffer,
        o.max_eval_sentences,
        o.max_parameter_mib,
        o.threads,
        o.batch_size,
        o.progress_interval,
        usize::try_from(o.progress_interval_seconds).unwrap_or(usize::MAX),
        o.recovery_interval,
    ]
    .contains(&0)
        || o.shuffle_buffer < o.batch_size
        || o.seed == 0
        || !o.learning_rate.is_finite()
        || o.learning_rate <= 0.0
        || !o.l2.is_finite()
        || o.l2 < 0.0
        || o.learning_rate * o.l2 >= 1.0
        || o.primary_nbest_margin
            .is_some_and(|margin| !margin.is_finite() || margin < 0.0)
        || !o.initial_bigram_delta.is_finite()
        || !(0.0..=1.0).contains(&o.initial_bigram_delta)
        || !o.initial_trigram_weight.is_finite()
        || !(0.0..=1.0).contains(&o.initial_trigram_weight)
    {
        return Err("invalid positive limits, seed, or learning rate/L2".into());
    }
    o.max_records
        .checked_add(o.skip_records)
        .ok_or("record limit overflow")?;
    o.max_parameter_mib
        .checked_mul(1 << 20)
        .ok_or("parameter budget overflow")?;
    if !o.delta_cutoff.is_finite() || o.delta_cutoff < 0.0 {
        return Err("delta cutoff must be finite and nonnegative".into());
    }
    if o.delta_cutoff > 0.0 {
        cutoff_policy(&o).validate()?;
    }
    if o.output.exists() && fs::read_dir(&o.output)?.next().is_some() {
        return Err("output directory is not empty; choose a new run directory".into());
    }
    if o.resume_from.is_none() && o.work_dir.exists() && fs::read_dir(&o.work_dir)?.next().is_some()
    {
        return Err("work directory is not empty; choose a new work directory".into());
    }
    Ok(o)
}

#[derive(Debug, Default, Deserialize, Serialize)]
struct InputReport {
    records_read: usize,
    skipped_prefix: usize,
    training_sentences: usize,
    validation_sentences: usize,
    training_pages: usize,
    validation_pages: usize,
    forced_tokens: usize,
    rejected: BTreeMap<String, usize>,
    rejection_examples: Vec<String>,
    load_seconds: f64,
    stopped_at_limit: bool,
    source_training_pages: usize,
    source_heldout_pages: usize,
    heldout_sentences: usize,
    heldout_pages: usize,
    evaluation_sampling: String,
}

fn canonical_path_key(path: &Path) -> Result<String> {
    Ok(fs::canonicalize(path)?
        .to_string_lossy()
        .replace('/', "\\")
        .to_lowercase())
}

fn load_prepared_data(
    o: &Options,
    model: &Model,
) -> Result<(Vec<Sentence>, Vec<Sentence>, InputReport, PageBlocks)> {
    let source = o
        .prepared_data_in
        .as_ref()
        .ok_or("missing prepared-data source")?;
    let source_config: Options =
        serde_json::from_reader(BufReader::new(fs::File::open(source.join("config.json"))?))?;
    let (Some(source_model), Some(current_model)) = (&source_config.model_in, &o.model_in) else {
        return Err(
            "prepared-data reuse requires the same explicit initial model in both runs".into(),
        );
    };
    if canonical_path_key(&source_config.input)? != canonical_path_key(&o.input)?
        || canonical_path_key(source_model)? != canonical_path_key(current_model)?
        || source_config.max_words != o.max_words
        || source_config.missing_pos_probability.to_bits() != o.missing_pos_probability.to_bits()
        || source_config.page_block_size != o.page_block_size
        || source_config.test_prefix_sentences != o.test_prefix_sentences
        || source_config.max_eval_sentences != o.max_eval_sentences
        || source_config.seed != o.seed
        || !source_config.full_corpus
    {
        return Err(
            "prepared-data source is incompatible with this input, model, normalization, or split"
                .into(),
        );
    }
    let info: serde_json::Value = serde_json::from_reader(BufReader::new(fs::File::open(
        source.join("resume-info.json"),
    )?))?;
    let cache_path = info["page_cache_directory"]
        .as_str()
        .ok_or("prepared-data source does not identify a page cache")?;
    let blocks = PageBlocks::load(Path::new(cache_path))?;
    let report: InputReport = serde_json::from_reader(BufReader::new(fs::File::open(
        source.join("input-report.json"),
    )?))?;
    let test: Vec<Sentence> = serde_json::from_reader(BufReader::new(fs::File::open(
        source.join("test-sample.json"),
    )?))?;
    let train: Vec<Sentence> = serde_json::from_reader(BufReader::new(fs::File::open(
        source.join("training-probe.json"),
    )?))?;
    if blocks.training_sentences != report.training_sentences
        || blocks.pages_per_block != o.page_block_size
        || report.validation_sentences != o.test_prefix_sentences
        || test.len() != o.test_prefix_sentences
        || train.len() > o.max_eval_sentences
        || model.words.len() < 3
    {
        return Err("prepared-data manifest, input report, or fixed samples disagree".into());
    }
    Ok((train, test, report, blocks))
}

#[derive(Serialize)]
struct BorrowedTestSentence<'a> {
    record: usize,
    sentence: &'a Sentence,
}

fn sample(
    o: &Options,
    index: &Index,
    mut page_cache: Option<&mut blocks::Builder>,
) -> Result<(Vec<Sentence>, Vec<Sentence>, InputReport)> {
    let started = Instant::now();
    let mut reader = corpus::open(&o.input)?;
    let mut train = Vec::new();
    let mut dev = Vec::new();
    let mut training_probe =
        corpus::Reservoir::new(o.max_eval_sentences, o.seed ^ 0x0074_7261_696e);
    let mut heldout_probe =
        corpus::Reservoir::new(o.max_validation_sentences, o.seed ^ 0x7465_7374);
    let mut report = InputReport::default();
    let mut training_pages = std::collections::BTreeSet::new();
    let mut heldout_pages = std::collections::BTreeSet::new();
    let mut source_training_pages = std::collections::BTreeSet::new();
    let mut source_heldout_pages = std::collections::BTreeSet::new();
    let test_target = if o.test_prefix_sentences > 0 {
        o.test_prefix_sentences
    } else {
        o.max_validation_sentences
    };
    let mut test_writer = if o.full_corpus {
        Some(bzip2::write::BzEncoder::new(
            BufWriter::new(
                OpenOptions::new()
                    .create_new(true)
                    .write(true)
                    .open(o.output.join("test-corpus.normalized.jsonl.bz2"))?,
            ),
            bzip2::Compression::fast(),
        ))
    } else {
        None
    };
    let mut line = String::new();
    loop {
        if !o.full_corpus
            && report.training_sentences >= o.max_sentences
            && dev.len() >= test_target
        {
            report.stopped_at_limit = true;
            break;
        }
        if report.records_read >= o.max_records + o.skip_records {
            report.stopped_at_limit = true;
            break;
        }
        let Some(record) = corpus::read_record(&mut *reader, &mut line, report.records_read + 1)?
        else {
            break;
        };
        report.records_read += 1;
        if report.records_read <= o.skip_records {
            report.skipped_prefix += 1;
            continue;
        }
        if let Some(cache) = page_cache.as_deref_mut() {
            cache.observe(record.page_id, report.records_read)?;
        }
        // With an explicit prefix test set, reserve the first N accepted,
        // normalizable sentences in corpus order. Otherwise retain the
        // page-disjoint hash split used by existing bounded experiments.
        let is_dev = if o.test_prefix_sentences > 0 {
            dev.len() < test_target
        } else {
            corpus::validation_page(record.page_id)
        };
        if is_dev {
            source_heldout_pages.insert(record.page_id);
        } else {
            source_training_pages.insert(record.page_id);
        }
        if !o.full_corpus
            && ((is_dev && dev.len() >= test_target)
                || (!is_dev && report.training_sentences >= o.max_sentences))
        {
            continue;
        }
        match corpus::normalize(record, index, o.max_words) {
            Ok(prepared) => {
                report.forced_tokens += prepared.forced_tokens();
                if is_dev {
                    report.heldout_sentences += 1;
                    heldout_pages.insert(prepared.page_id);
                    if o.full_corpus {
                        if let Some(writer) = test_writer.as_mut() {
                            serde_json::to_writer(
                                &mut *writer,
                                &BorrowedTestSentence {
                                    record: report.records_read,
                                    sentence: &prepared,
                                },
                            )?;
                            writeln!(writer)?;
                        }
                        if o.test_prefix_sentences > 0 {
                            dev.push(prepared);
                        } else {
                            heldout_probe.push(prepared);
                        }
                    } else {
                        dev.push(prepared);
                    }
                } else {
                    if let Some(cache) = page_cache.as_deref_mut() {
                        cache.add(report.records_read, &prepared)?;
                    }
                    training_pages.insert(prepared.page_id);
                    report.training_sentences += 1;
                    if o.full_corpus {
                        training_probe.push(prepared);
                    } else if train.len() < o.max_eval_sentences {
                        train.push(prepared);
                    }
                }
            }
            Err(e) => {
                let text = e.to_string();
                let kind = if text.starts_with("source word is missing:") {
                    "source word is missing"
                } else {
                    &text
                };
                *report.rejected.entry(kind.to_owned()).or_default() += 1;
                if report.rejection_examples.len() < 10 {
                    report
                        .rejection_examples
                        .push(format!("record {}: {text}", report.records_read));
                }
            }
        }
        if report.records_read % (if o.full_corpus { 10_000 } else { 1000 }) == 0 {
            eprintln!(
                "input records={} pages={} train={} heldout={} elapsed={:.1}s",
                report.records_read,
                source_training_pages.len() + source_heldout_pages.len(),
                report.training_sentences,
                report.heldout_sentences,
                started.elapsed().as_secs_f64()
            );
        }
    }
    if o.full_corpus {
        train = training_probe.into_values();
        if o.test_prefix_sentences == 0 {
            dev = heldout_probe.into_values();
        }
        if let Some(writer) = test_writer {
            writer.finish()?.flush()?;
        }
        write_json(
            &o.output.join("corpus-split.json"),
            &serde_json::json!({
                "version":2,
                "rule":if o.test_prefix_sentences > 0 { format!("first {} accepted normalized sentences in input order => test; all later accepted sentences => training", o.test_prefix_sentences) } else { "splitmix64(page_id + constant) % 5 == 0 => heldout; otherwise training".to_owned() },
                "page_id_disjoint":if o.test_prefix_sentences > 0 { serde_json::Value::Null } else { serde_json::Value::Bool(source_training_pages.is_disjoint(&source_heldout_pages)) },
                "training_page_ids":source_training_pages, "heldout_page_ids":source_heldout_pages,
                "test_corpus":"test-corpus.normalized.jsonl.bz2", "test_sample":"test-sample.json",
                "test_sampling":if o.test_prefix_sentences > 0 { "all accepted prefix test sentences" } else { "uniform sentence reservoir over all accepted heldout records; fixed independent RNG" },
                "test_sample_capacity":if o.test_prefix_sentences > 0 { o.test_prefix_sentences } else { o.max_validation_sentences },"seed":o.seed,
                "initial_statistics_leakage":true,
                "supervision":"MeCab pseudo-gold; this split holds out CRF updates only, not initial statistics"
            }),
        )?;
    }
    if o.test_prefix_sentences > 0 && dev.len() != o.test_prefix_sentences {
        return Err(format!(
            "test prefix requested {} accepted sentences but only {} were normalized",
            o.test_prefix_sentences,
            dev.len()
        )
        .into());
    }
    if train.is_empty() || dev.is_empty() {
        return Err(format!("need nonempty page-heldout train/dev sets; got train={} dev={}; increase --max-records", train.len(), dev.len()).into());
    }
    report.validation_sentences = dev.len();
    report.training_pages = training_pages.len();
    report.validation_pages = dev
        .iter()
        .map(|p| p.page_id)
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    report.load_seconds = started.elapsed().as_secs_f64();
    report.source_training_pages = source_training_pages.len();
    report.source_heldout_pages = source_heldout_pages.len();
    report.heldout_pages = heldout_pages.len();
    report.evaluation_sampling = if o.full_corpus {
        "whole-corpus uniform sentence reservoir"
    } else {
        "prefix"
    }
    .into();
    Ok((train, dev, report))
}

#[derive(Debug, Serialize)]
struct Metrics {
    sentences: usize,
    nbest: usize,
    mean_first_nll: f64,
    mean_rerank_nll_gold_present: Option<f64>,
    first_exact_sentences: usize,
    first_exact_accuracy: f64,
    first_top1_boundary_error_sentences: usize,
    first_top1_same_boundary_surface_error_sentences: usize,
    first_top1_same_boundary_identity_error_sentences: usize,
    nbest_gold_recall: f64,
    reranked_exact_sentences: usize,
    reranked_exact_accuracy: f64,
    reranked_top1_boundary_error_sentences: usize,
    reranked_top1_same_boundary_surface_error_sentences: usize,
    reranked_top1_same_boundary_identity_error_sentences: usize,
    gold_present_sentences: usize,
    first_exact_accuracy_given_gold_present: Option<f64>,
    reranked_exact_accuracy_given_gold_present: Option<f64>,
    rerank_top1_helped_sentences: usize,
    rerank_top1_harmed_sentences: usize,
    first_surface_accuracy: f64,
    reranked_surface_accuracy: f64,
    nbest_rerank_worker_seconds: f64,
    seconds: f64,
}

struct CurveLog {
    writer: BufWriter<fs::File>,
    interval: usize,
    total_training_sentences: usize,
    next_at: usize,
    wall_interval: Duration,
    next_wall: Instant,
    last: Option<(usize, usize)>,
}

impl CurveLog {
    fn create(
        path: &std::path::Path,
        interval: usize,
        wall_interval_seconds: u64,
        total_training_sentences: usize,
    ) -> Result<Self> {
        let file = OpenOptions::new().create_new(true).write(true).open(path)?;
        Ok(Self {
            writer: BufWriter::new(file),
            interval,
            total_training_sentences,
            next_at: interval,
            wall_interval: Duration::from_secs(wall_interval_seconds),
            next_wall: Instant::now() + Duration::from_secs(wall_interval_seconds),
            last: None,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn maybe_record(
        &mut self,
        model: &Model,
        index: &Index,
        validation: &[Sentence],
        o: &Options,
        trainer: &Trainer,
        pass: &Pass,
        epoch: usize,
        started: Instant,
    ) -> Result<()> {
        let now = Instant::now();
        let count_due = pass.processed >= self.next_at;
        let wall_due = now >= self.next_wall;
        while pass.processed >= self.next_at {
            self.record(
                model,
                index,
                validation,
                o,
                trainer,
                pass,
                epoch,
                started,
                if wall_due {
                    "interval_and_wallclock"
                } else {
                    "interval"
                },
            )?;
            self.next_at = self
                .next_at
                .checked_add(self.interval)
                .ok_or("progress interval overflow")?;
        }
        if wall_due {
            if !count_due {
                self.record(
                    model,
                    index,
                    validation,
                    o,
                    trainer,
                    pass,
                    epoch,
                    started,
                    "wallclock",
                )?;
            }
            self.next_wall = Instant::now() + self.wall_interval;
        }
        Ok(())
    }

    fn needs_record(&self, processed: usize) -> bool {
        processed >= self.next_at || Instant::now() >= self.next_wall
    }

    #[allow(clippy::too_many_arguments)]
    fn record_if_new(
        &mut self,
        model: &Model,
        index: &Index,
        validation: &[Sentence],
        o: &Options,
        trainer: &Trainer,
        pass: &Pass,
        epoch: usize,
        started: Instant,
        kind: &str,
    ) -> Result<()> {
        if self.last != Some((epoch, pass.processed)) {
            self.record(
                model, index, validation, o, trainer, pass, epoch, started, kind,
            )?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn record(
        &mut self,
        model: &Model,
        index: &Index,
        validation: &[Sentence],
        o: &Options,
        trainer: &Trainer,
        pass: &Pass,
        epoch: usize,
        started: Instant,
        kind: &str,
    ) -> Result<()> {
        let metrics = evaluate(model, index, validation, o, trainer)?;
        let character_evaluation_started = Instant::now();
        let character_metrics = character_accuracy_metrics(model, index, validation, o, trainer)?;
        let character_evaluation_seconds = character_evaluation_started.elapsed().as_secs_f64();
        let bigram_character_accuracy =
            character_metrics["bigram_stage"]["character_level_accuracy"]
                .as_f64()
                .ok_or("character evaluation omitted bigram-stage accuracy")?;
        let post_trigram_character_accuracy =
            character_metrics["post_trigram_stage"]["character_level_accuracy"]
                .as_f64()
                .ok_or("character evaluation omitted post-trigram accuracy")?;
        let training_hit = (pass.processed > 0)
            .then(|| (pass.processed - pass.missed) as f64 / pass.processed as f64);
        let training_top1 =
            (pass.top1_evaluated > 0).then(|| pass.top1_hits as f64 / pass.top1_evaluated as f64);
        let primary_objective = primary_objective(o);
        let max_abs_first_delta = max_abs(&model.first);
        let max_abs_word_bigram_delta = max_abs_word_bigrams(&model.first);
        let max_abs_word_trigram_delta = max_abs(&model.third);
        let max_abs_pos_trigram_delta = max_abs(&model.third_pos);
        let max_abs_reading_length_trigram_delta = max_abs(&model.third_reading_length);
        let mut row = serde_json::json!({
            "epoch": epoch,
            "event": kind,
            "processed_training_sentences": pass.processed,
            "total_training_sentences": self.total_training_sentences,
            "training_primary_objective": primary_objective,
            "primary_nbest_margin_log_score": o.primary_nbest_margin,
            "primary_nbest_listwise": o.primary_nbest_listwise,
            "primary_objective_mean_loss": (pass.processed > 0).then(|| pass.first_loss / pass.processed as f64),
            "primary_margin_active_sentences": pass.primary_nbest_margin_active_sentences,
            "primary_margin_active_rate": (pass.processed > 0).then(|| pass.primary_nbest_margin_active_sentences as f64 / pass.processed as f64),
            "primary_listwise_active_sentences": pass.primary_nbest_listwise_active_sentences,
            "training_nbest_gold_hit_rate": training_hit,
            "training_nbest_gold_hits": pass.processed.saturating_sub(pass.missed),
            "training_preupdate_top1_gold_hit_rate": training_top1,
            "training_preupdate_top1_gold_hits": pass.top1_hits,
            "training_preupdate_top1_evaluated_sentences": pass.top1_evaluated,
            "training_post_trigram_top1_gold_hit_rate": (pass.post_trigram_top1_evaluated > 0).then(|| pass.post_trigram_top1_hits as f64 / pass.post_trigram_top1_evaluated as f64),
            "training_post_trigram_top1_gold_hits": pass.post_trigram_top1_hits,
            "training_post_trigram_top1_evaluated_sentences": pass.post_trigram_top1_evaluated,
            "training_gold_augmented_sentences": pass.gold_augmented,
            "learned_word_trigram_count": model.third.values.len(),
            "max_abs_first_order_delta": max_abs_first_delta,
            "max_abs_word_bigram_delta": max_abs_word_bigram_delta,
            "max_abs_word_trigram_delta": max_abs_word_trigram_delta,
            "max_abs_pos_trigram_delta": max_abs_pos_trigram_delta,
            "max_abs_reading_length_trigram_delta": max_abs_reading_length_trigram_delta,
            "validation_sentences": metrics.sentences,
            "validation_first_exact_sentences": metrics.first_exact_sentences,
            "validation_nbest_gold_hit_rate": metrics.nbest_gold_recall,
            "validation_bigram_nbest_gold_cover_rate": metrics.nbest_gold_recall,
            "validation_nbest_gold_hits": metrics.gold_present_sentences,
            "validation_reranked_exact_sentences": metrics.reranked_exact_sentences,
            "validation_first_nll": metrics.mean_first_nll,
            "validation_first_exact_accuracy": metrics.first_exact_accuracy,
            "validation_bigram_top1_accuracy": metrics.first_exact_accuracy,
            "validation_reranked_exact_accuracy": metrics.reranked_exact_accuracy,
            "validation_post_trigram_top1_accuracy": metrics.reranked_exact_accuracy,
            "validation_character_level_accuracy_bigram": bigram_character_accuracy,
            "validation_character_level_accuracy_post_trigram": post_trigram_character_accuracy,
            "validation_character_metrics": character_metrics,
            "validation_gold_present_sentences": metrics.gold_present_sentences,
            "validation_first_exact_given_gold_present": metrics.first_exact_accuracy_given_gold_present,
            "validation_reranked_exact_given_gold_present": metrics.reranked_exact_accuracy_given_gold_present,
            "validation_rerank_top1_helped_sentences": metrics.rerank_top1_helped_sentences,
            "validation_rerank_top1_harmed_sentences": metrics.rerank_top1_harmed_sentences,
            "evaluation_seconds": metrics.seconds,
            "character_evaluation_seconds": character_evaluation_seconds,
            "nbest_rerank_worker_seconds": metrics.nbest_rerank_worker_seconds,
            "elapsed_seconds": started.elapsed().as_secs_f64(),
        });
        row["validation_top1_error_breakdown"] = serde_json::json!({
            "bigram": {
                "boundary_error_sentences": metrics.first_top1_boundary_error_sentences,
                "same_boundary_surface_error_sentences": metrics.first_top1_same_boundary_surface_error_sentences,
                "same_boundary_identity_error_sentences": metrics.first_top1_same_boundary_identity_error_sentences,
            },
            "post_trigram": {
                "boundary_error_sentences": metrics.reranked_top1_boundary_error_sentences,
                "same_boundary_surface_error_sentences": metrics.reranked_top1_same_boundary_surface_error_sentences,
                "same_boundary_identity_error_sentences": metrics.reranked_top1_same_boundary_identity_error_sentences,
            }
        });
        serde_json::to_writer(&mut self.writer, &row)?;
        writeln!(self.writer)?;
        self.writer.flush()?;
        self.last = Some((epoch, pass.processed));
        eprintln!(
            "curve epoch={epoch} event={kind} trained={} primary_objective={primary_objective} primary_mean_loss={:.6} primary_active={}/{} train_preupdate_top1={} train_nbest_hit={} heldout_bigram_top1={:.4} ({}/{}) heldout_post_trigram_top1={:.4} ({}/{}) heldout_bigram_nbest_cover={:.4} ({}/{}) heldout_char_accuracy={:.4}->{:.4} learned_word_trigrams={} max_abs_first_delta={:.4} max_abs_word_bigram_delta={:.4} max_abs_word_trigram_delta={:.4} max_abs_pos_tri_delta={:.4} max_abs_readlen_tri_delta={:.4} rerank_helped={} rerank_harmed={} heldout_nll={:.6} eval_seconds={:.1} char_eval_seconds={:.1}",
            pass.processed,
            if pass.processed > 0 {
                pass.first_loss / pass.processed as f64
            } else {
                0.0
            },
            if o.primary_nbest_listwise {
                pass.primary_nbest_listwise_active_sentences
            } else {
                pass.primary_nbest_margin_active_sentences
            },
            pass.processed,
            training_top1.map_or_else(
                || format!("n/a ({}/{})", pass.top1_hits, pass.top1_evaluated),
                |x| format!("{x:.4} ({}/{})", pass.top1_hits, pass.top1_evaluated)
            ),
            training_hit.map_or_else(|| "n/a".to_owned(), |x| format!("{x:.4}")),
            metrics.first_exact_accuracy,
            metrics.first_exact_sentences,
            metrics.sentences,
            metrics.reranked_exact_accuracy,
            metrics.reranked_exact_sentences,
            metrics.sentences,
            metrics.nbest_gold_recall,
            metrics.gold_present_sentences,
            metrics.sentences,
            bigram_character_accuracy,
            post_trigram_character_accuracy,
            model.third.values.len(),
            max_abs_first_delta,
            max_abs_word_bigram_delta,
            max_abs_word_trigram_delta,
            max_abs_pos_trigram_delta,
            max_abs_reading_length_trigram_delta,
            metrics.rerank_top1_helped_sentences,
            metrics.rerank_top1_harmed_sentences,
            metrics.mean_first_nll,
            metrics.seconds,
            character_evaluation_seconds
        );
        eprintln!(
            "heldout_top1_errors epoch={epoch} event={kind} bigram_boundary={} bigram_same_boundary_surface={} bigram_same_boundary_identity={} post_trigram_boundary={} post_trigram_same_boundary_surface={} post_trigram_same_boundary_identity={} sentences={}",
            metrics.first_top1_boundary_error_sentences,
            metrics.first_top1_same_boundary_surface_error_sentences,
            metrics.first_top1_same_boundary_identity_error_sentences,
            metrics.reranked_top1_boundary_error_sentences,
            metrics.reranked_top1_same_boundary_surface_error_sentences,
            metrics.reranked_top1_same_boundary_identity_error_sentences,
            metrics.sentences
        );
        Ok(())
    }
}

fn max_abs<K: ai_engine_crf::sparse::Key>(parameters: &ai_engine_crf::Parameters<K>) -> f64 {
    parameters
        .values
        .iter()
        .map(|(_, value)| (value * parameters.scale).abs())
        .fold(0.0, f64::max)
}

fn max_abs_word_bigrams(parameters: &ai_engine_crf::Parameters<Feature>) -> f64 {
    parameters
        .values
        .iter()
        .filter_map(|(feature, value)| {
            matches!(feature, Feature::Bigram(_, _)).then_some((value * parameters.scale).abs())
        })
        .fold(0.0, f64::max)
}

fn surfaces(model: &Model, sample: &Prepared, nodes: &[usize]) -> Vec<String> {
    nodes
        .iter()
        .copied()
        .filter(|&n| n != 0 && n != sample.graph.nodes.len() - 1)
        .map(|n| {
            let node = &sample.graph.nodes[n];
            let word = &model.words[node.word as usize];
            if word.model_only {
                sample.reading[node.start..node.end].to_owned()
            } else {
                word.surface.clone()
            }
        })
        .collect()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Top1Outcome {
    Correct,
    BoundaryError,
    SameBoundarySurfaceError,
    SameBoundaryIdentityError,
}

fn classify_top1(exact_path: bool, same_boundaries: bool, same_surfaces: bool) -> Top1Outcome {
    if exact_path {
        Top1Outcome::Correct
    } else if !same_boundaries {
        Top1Outcome::BoundaryError
    } else if !same_surfaces {
        Top1Outcome::SameBoundarySurfaceError
    } else {
        Top1Outcome::SameBoundaryIdentityError
    }
}

fn same_path_boundaries(sample: &Prepared, predicted: &[usize]) -> bool {
    let content_spans = |path: &[usize]| {
        path.iter()
            .copied()
            .filter(|&node| node != 0 && node != sample.graph.nodes.len() - 1)
            .map(|node| {
                let node = &sample.graph.nodes[node];
                (node.start, node.end)
            })
            .collect::<Vec<_>>()
    };
    content_spans(predicted) == content_spans(&sample.graph.gold)
}

fn evaluate(
    model: &Model,
    index: &Index,
    samples: &[Sentence],
    o: &Options,
    trainer: &Trainer,
) -> Result<Metrics> {
    let started = Instant::now();
    let mut nll = 0.0;
    let mut tri_nll = 0.0;
    let mut first_correct = 0;
    let mut tri_correct = 0;
    let mut first_boundary_errors = 0;
    let mut first_same_boundary_surface_errors = 0;
    let mut first_same_boundary_identity_errors = 0;
    let mut tri_boundary_errors = 0;
    let mut tri_same_boundary_surface_errors = 0;
    let mut tri_same_boundary_identity_errors = 0;
    let mut recall = 0_usize;
    let mut first_correct_given_gold = 0_usize;
    let mut tri_correct_given_gold = 0_usize;
    let mut rerank_helped = 0_usize;
    let mut rerank_harmed = 0_usize;
    let mut first_surface = 0;
    let mut tri_surface = 0;
    let mut nbest_rerank_worker_seconds = 0.0;
    let results = trainer.map(samples, |sentence| {
        let s = corpus::prepare_sentence(sentence, model, index, limits(o))?;
        let first_loss = first_nll(model, &s.graph)?;
        let nbest_started = Instant::now();
        let paths = nbest(model, &s.graph, o.nbest, o.max_astar_expansions)?;
        let ranked = rerank(model, &s.graph, &paths)?;
        let nbest_worker_seconds = nbest_started.elapsed().as_secs_f64();
        let gold = surfaces(model, &s, &s.graph.gold);
        let first_surfaces = surfaces(model, &s, &paths[0].nodes);
        let reranked_surfaces = surfaces(model, &s, &paths[ranked.best].nodes);
        Ok((
            first_loss,
            ranked.loss,
            classify_top1(
                paths[0].nodes == s.graph.gold,
                same_path_boundaries(&s, &paths[0].nodes),
                first_surfaces == gold,
            ),
            classify_top1(
                paths[ranked.best].nodes == s.graph.gold,
                same_path_boundaries(&s, &paths[ranked.best].nodes),
                reranked_surfaces == gold,
            ),
            first_surfaces == gold,
            reranked_surfaces == gold,
            nbest_worker_seconds,
        ))
        // This sample's lattice is dropped before the result leaves its worker.
    })?;
    for (
        first_loss,
        third_loss,
        first_outcome,
        third_outcome,
        first_display,
        third_display,
        nbest_worker_seconds,
    ) in results
    {
        nll += first_loss;
        if let Some(loss) = third_loss {
            recall += 1;
            tri_nll += loss;
            let first_match = first_outcome == Top1Outcome::Correct;
            let third_match = third_outcome == Top1Outcome::Correct;
            first_correct_given_gold += usize::from(first_match);
            tri_correct_given_gold += usize::from(third_match);
            rerank_helped += usize::from(!first_match && third_match);
            rerank_harmed += usize::from(first_match && !third_match);
        }
        first_correct += usize::from(first_outcome == Top1Outcome::Correct);
        tri_correct += usize::from(third_outcome == Top1Outcome::Correct);
        match first_outcome {
            Top1Outcome::Correct => {}
            Top1Outcome::BoundaryError => first_boundary_errors += 1,
            Top1Outcome::SameBoundarySurfaceError => first_same_boundary_surface_errors += 1,
            Top1Outcome::SameBoundaryIdentityError => first_same_boundary_identity_errors += 1,
        }
        match third_outcome {
            Top1Outcome::Correct => {}
            Top1Outcome::BoundaryError => tri_boundary_errors += 1,
            Top1Outcome::SameBoundarySurfaceError => tri_same_boundary_surface_errors += 1,
            Top1Outcome::SameBoundaryIdentityError => tri_same_boundary_identity_errors += 1,
        }
        first_surface += usize::from(first_display);
        tri_surface += usize::from(third_display);
        nbest_rerank_worker_seconds += nbest_worker_seconds;
    }
    let n = samples.len() as f64;
    Ok(Metrics {
        sentences: samples.len(),
        nbest: o.nbest,
        mean_first_nll: nll / n,
        mean_rerank_nll_gold_present: (recall > 0).then(|| tri_nll / recall as f64),
        first_exact_sentences: first_correct,
        first_exact_accuracy: first_correct as f64 / n,
        first_top1_boundary_error_sentences: first_boundary_errors,
        first_top1_same_boundary_surface_error_sentences: first_same_boundary_surface_errors,
        first_top1_same_boundary_identity_error_sentences: first_same_boundary_identity_errors,
        nbest_gold_recall: recall as f64 / n,
        reranked_exact_sentences: tri_correct,
        reranked_exact_accuracy: tri_correct as f64 / n,
        reranked_top1_boundary_error_sentences: tri_boundary_errors,
        reranked_top1_same_boundary_surface_error_sentences: tri_same_boundary_surface_errors,
        reranked_top1_same_boundary_identity_error_sentences: tri_same_boundary_identity_errors,
        gold_present_sentences: recall,
        first_exact_accuracy_given_gold_present: (recall > 0)
            .then(|| first_correct_given_gold as f64 / recall as f64),
        reranked_exact_accuracy_given_gold_present: (recall > 0)
            .then(|| tri_correct_given_gold as f64 / recall as f64),
        rerank_top1_helped_sentences: rerank_helped,
        rerank_top1_harmed_sentences: rerank_harmed,
        first_surface_accuracy: first_surface as f64 / n,
        reranked_surface_accuracy: tri_surface as f64 / n,
        nbest_rerank_worker_seconds,
        seconds: started.elapsed().as_secs_f64(),
    })
}

fn evaluate_feature_ablations(
    model: &mut Model,
    index: &Index,
    samples: &[Sentence],
    o: &Options,
    trainer: &Trainer,
    baseline: &Metrics,
) -> Result<serde_json::Value> {
    let baseline_character_metrics = character_accuracy_metrics(model, index, samples, o, trainer)?;
    let without_reading_length_bigram = {
        let initial = std::mem::replace(&mut model.initial_reading_length_bigrams, [0.0; 81]);
        let learned = model
            .first
            .values
            .iter()
            .filter_map(|(feature, value)| match feature {
                Feature::ReadingLengthBigram(..) => Some((feature, value)),
                _ => None,
            })
            .collect::<Vec<_>>();
        model
            .first
            .values
            .retain(|feature, _| !matches!(feature, Feature::ReadingLengthBigram(..)));
        let result = evaluate(model, index, samples, o, trainer);
        let character_result = character_accuracy_metrics(model, index, samples, o, trainer);
        model.initial_reading_length_bigrams = initial;
        for (feature, value) in learned {
            model.first.values.insert(feature, value);
        }
        serde_json::json!({
            "metrics": result?,
            "character_metrics": character_result?
        })
    };

    let without_reading_length_trigram = {
        let initial = std::mem::replace(&mut model.initial_reading_length_trigrams, [0.0; 729]);
        let learned = std::mem::take(&mut model.third_reading_length);
        let result = evaluate(model, index, samples, o, trainer);
        let character_result = character_accuracy_metrics(model, index, samples, o, trainer);
        model.initial_reading_length_trigrams = initial;
        model.third_reading_length = learned;
        serde_json::json!({
            "metrics": result?,
            "character_metrics": character_result?
        })
    };

    let without_reading_length_bigram_and_trigram = {
        let initial_bigrams =
            std::mem::replace(&mut model.initial_reading_length_bigrams, [0.0; 81]);
        let initial_trigrams =
            std::mem::replace(&mut model.initial_reading_length_trigrams, [0.0; 729]);
        let learned_trigrams = std::mem::take(&mut model.third_reading_length);
        let learned_bigrams = model
            .first
            .values
            .iter()
            .filter_map(|(feature, value)| match feature {
                Feature::ReadingLengthBigram(..) => Some((feature, value)),
                _ => None,
            })
            .collect::<Vec<_>>();
        model
            .first
            .values
            .retain(|feature, _| !matches!(feature, Feature::ReadingLengthBigram(..)));
        let result = evaluate(model, index, samples, o, trainer);
        let character_result = character_accuracy_metrics(model, index, samples, o, trainer);
        model.initial_reading_length_bigrams = initial_bigrams;
        model.initial_reading_length_trigrams = initial_trigrams;
        model.third_reading_length = learned_trigrams;
        for (feature, value) in learned_bigrams {
            model.first.values.insert(feature, value);
        }
        serde_json::json!({
            "metrics": result?,
            "character_metrics": character_result?
        })
    };

    let without_pos_trigram = {
        let initial = std::mem::take(&mut model.initial_pos_trigrams);
        let learned = std::mem::take(&mut model.third_pos);
        let result = evaluate(model, index, samples, o, trainer);
        let character_result = character_accuracy_metrics(model, index, samples, o, trainer);
        model.initial_pos_trigrams = initial;
        model.third_pos = learned;
        serde_json::json!({
            "metrics": result?,
            "character_metrics": character_result?
        })
    };

    let without_pos_word_and_word_pos_bigrams = {
        let initial = model
            .initial_bigrams
            .iter()
            .filter_map(|(feature, value)| {
                matches!(feature, Feature::PosWord(..) | Feature::WordPos(..))
                    .then_some((feature, value))
            })
            .collect::<Vec<_>>();
        model
            .initial_bigrams
            .retain(|feature, _| !matches!(feature, Feature::PosWord(..) | Feature::WordPos(..)));
        let learned = model
            .first
            .values
            .iter()
            .filter_map(|(feature, value)| {
                matches!(feature, Feature::PosWord(..) | Feature::WordPos(..))
                    .then_some((feature, value))
            })
            .collect::<Vec<_>>();
        model
            .first
            .values
            .retain(|feature, _| !matches!(feature, Feature::PosWord(..) | Feature::WordPos(..)));
        let result = evaluate(model, index, samples, o, trainer);
        let character_result = character_accuracy_metrics(model, index, samples, o, trainer);
        for (feature, value) in initial {
            model.initial_bigrams.insert(feature, value);
        }
        for (feature, value) in learned {
            model.first.values.insert(feature, value);
        }
        serde_json::json!({
            "metrics": result?,
            "character_metrics": character_result?
        })
    };

    Ok(serde_json::json!({
        "sentences": samples.len(),
        "nbest": o.nbest,
        "baseline": {
            "metrics": baseline,
            "character_metrics": baseline_character_metrics
        },
        "without_reading_length_bigram": without_reading_length_bigram,
        "without_reading_length_trigram": without_reading_length_trigram,
        "without_reading_length_bigram_and_trigram": without_reading_length_bigram_and_trigram,
        "without_pos_trigram": without_pos_trigram,
        "without_pos_word_and_word_pos_bigrams": without_pos_word_and_word_pos_bigrams,
        "definitions": {
            "without_reading_length_bigram": "Zero immutable initial read-length bigram scores and omit learned read-length bigram deltas; retain read-length trigram scores and deltas.",
            "without_reading_length_trigram": "Zero immutable initial read-length trigram scores and omit learned read-length trigram deltas; retain read-length bigram scores.",
            "without_reading_length_bigram_and_trigram": "Zero immutable initial read-length bigram/trigram scores and omit learned read-length bigram/trigram deltas.",
            "without_pos_trigram": "Omit immutable initial POS-trigram scores and learned POS-trigram deltas.",
            "without_pos_word_and_word_pos_bigrams": "Omit both immutable initial POS-WORD/WORD-POS bigram scores and their learned first-order deltas."
        }
    }))
}

fn character_accuracy_metrics(
    model: &Model,
    index: &Index,
    samples: &[Sentence],
    o: &Options,
    trainer: &Trainer,
) -> Result<serde_json::Value> {
    let report = error_analysis::analyze(model, index, samples, o, trainer)?;
    Ok(serde_json::json!({
        "method": report.method,
        "character_unit": report.character_unit,
        "bigram_stage": {
            "character_edit_distance": report.bigram_stage.character_edit_distance,
            "character_denominator": report.bigram_stage.character_denominator,
            "character_level_accuracy": report.bigram_stage.character_level_accuracy
        },
        "post_trigram_stage": {
            "character_edit_distance": report.post_trigram_stage.character_edit_distance,
            "character_denominator": report.post_trigram_stage.character_denominator,
            "character_level_accuracy": report.post_trigram_stage.character_level_accuracy
        }
    }))
}

#[derive(Debug, Serialize)]
struct EpochReport {
    epoch: usize,
    learning_rate: f64,
    training_update_seconds: f64,
    mean_update_first_nll: Option<f64>,
    mean_update_primary_nbest_margin_loss: Option<f64>,
    primary_nbest_margin_active_sentences: usize,
    mean_update_primary_nbest_listwise_loss: Option<f64>,
    primary_nbest_listwise_active_sentences: usize,
    mean_update_rerank_nll: Option<f64>,
    training_gold_missing_nbest: usize,
    nbest_gold_hits: usize,
    training_nbest_gold_hit_rate: Option<f64>,
    training_preupdate_top1_gold_hits: usize,
    training_preupdate_top1_evaluated_sentences: usize,
    training_preupdate_top1_gold_hit_rate: Option<f64>,
    training_post_trigram_top1_gold_hits: usize,
    training_post_trigram_top1_evaluated_sentences: usize,
    training_post_trigram_top1_gold_hit_rate: Option<f64>,
    training_gold_augmented_sentences: usize,
    feature_counts: BTreeMap<String, usize>,
    train: Metrics,
    validation: Metrics,
    training_sentences_processed: usize,
    parameter_table_bytes: usize,
    initial_bigram_table_bytes: usize,
    initial_trigram_table_bytes: usize,
    maximum_lattice_nodes: usize,
    maximum_lattice_edges: usize,
    training_batches: usize,
    maximum_batch_nodes: usize,
    maximum_batch_edges: usize,
    page_block_order: Vec<usize>,
    batch_timing: BatchTiming,
    lattice_prepare_seconds: f64,
    cutoff_events: Vec<CutoffEvent>,
}

fn cutoff_policy(o: &Options) -> cutoff::Policy {
    cutoff::Policy {
        minimum_delta: o.delta_cutoff,
        maximum_delta: o.delta_cutoff_max,
        target_fraction: 0.75,
    }
}

#[derive(Debug, Serialize)]
struct CutoffEvent {
    epoch: usize,
    committed_sentences: usize,
    committed_batches: usize,
    retry_source_record: usize,
    reason: String,
    cutoff: cutoff::Report,
    validation_before: Metrics,
    validation_after: Metrics,
}

#[allow(clippy::too_many_arguments)]
fn recover_capacity(
    model: &mut Model,
    index: &Index,
    o: &Options,
    trainer: &Trainer,
    validation: &[Sentence],
    epoch: usize,
    pass: &Pass,
    record: usize,
    reason: &ParameterLimit,
) -> Result<CutoffEvent> {
    eprintln!(
        "parameter_pressure epoch={epoch} trained={} retry_record={record} {reason}; applying delta cutoff",
        pass.processed
    );
    let validation_before = evaluate(model, index, validation, o, trainer)?;
    let report = cutoff::apply_for_retry(
        model,
        cutoff_policy(o),
        o.max_parameter_mib * (1 << 20),
        [
            reason.additional_first_bound,
            reason.additional_third_bound,
            reason.additional_pos_third_bound,
            reason.additional_reading_length_third_bound,
        ],
    )?;
    let validation_after = evaluate(model, index, validation, o, trainer)?;
    eprintln!(
        "delta_cutoff removed={} table_bytes_after={} retained_minimum_bytes={} retry_capacity_reached={} allocation_error={:?} dev_nll={:.6}->{:.6} dev_rerank={:.4}->{:.4}; retrying SAME batch",
        report.removed_total,
        report.table_bytes_after,
        report.retained_minimum_table_bytes,
        report.retry_capacity_reached,
        reason.allocation_error,
        validation_before.mean_first_nll,
        validation_after.mean_first_nll,
        validation_before.reranked_exact_accuracy,
        validation_after.reranked_exact_accuracy,
    );
    let event = CutoffEvent {
        epoch,
        committed_sentences: pass.processed,
        committed_batches: pass.batches,
        retry_source_record: record,
        reason: reason.to_string(),
        cutoff: report,
        validation_before,
        validation_after,
    };
    let mut writer = BufWriter::new(
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(o.output.join("delta-cutoff.jsonl"))?,
    );
    serde_json::to_writer(&mut writer, &event)?;
    writeln!(writer)?;
    writer.flush()?;
    Ok(event)
}

fn limits(o: &Options) -> Limits {
    Limits {
        words: o.max_words,
        nodes: o.max_nodes,
        edges: o.max_edges,
    }
}

#[derive(Debug, Deserialize, Serialize)]
struct TrainingStop {
    epoch: usize,
    kind: String,
    reason: String,
    source_records_read: usize,
    shuffle_state: u64,
    pending_window: Vec<Buffered>,
    page_block_order: Vec<usize>,
    active_page_block: Option<usize>,
}
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
struct MissedRecord {
    record_id: usize,
    page_id: u64,
}

fn missed_directory(work_dir: &Path) -> PathBuf {
    work_dir.join("missed-chunk003")
}

fn missed_file(work_dir: &Path, epoch: usize, partial: bool) -> PathBuf {
    let suffix = if partial { ".partial.tsv" } else { ".tsv" };
    missed_directory(work_dir).join(format!("epoch-{epoch:04}{suffix}"))
}

fn ensure_missed_readme(directory: &Path) -> Result<()> {
    fs::create_dir_all(directory)?;
    let path = directory.join("README.md");
    if !path.exists() {
        let mut writer = OpenOptions::new().create_new(true).write(true).open(path)?;
        writer.write_all(
            b"# missed-chunk003\n\n- Source: the current run's normalized training examples; the original tagged JSONL is referenced by record ID.\n- Output: `epoch-NNNN.tsv`, containing source record ID and page ID for examples whose final top-1 after trigram reranking was not gold.\n- Use: epoch N+1 trains only these record IDs, after retrieving the normalized examples from the run's page cache.\n- `epoch-NNNN.partial.tsv` is a resumable prefix for an interrupted epoch.\n- Deletion: this is derived and regenerable training data; delete the folder only after the run is no longer needed.\n",
        )?;
        writer.flush()?;
    }
    Ok(())
}

fn write_missed_records(
    work_dir: &Path,
    epoch: usize,
    records: &[MissedRecord],
    partial: bool,
) -> Result<PathBuf> {
    let directory = missed_directory(work_dir);
    ensure_missed_readme(&directory)?;
    let mut sorted = records.to_vec();
    sorted.sort_unstable_by_key(|record| record.record_id);
    if sorted
        .windows(2)
        .any(|pair| pair[0].record_id == pair[1].record_id)
    {
        return Err(format!("duplicate source record in epoch {epoch} missed list").into());
    }
    let path = missed_file(work_dir, epoch, partial);
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_nanos();
    let temporary = path.with_file_name(format!(".epoch-{epoch:04}-{nonce}.tmp"));
    write_missed_rows(&temporary, &sorted)?;
    if path.exists() {
        fs::remove_file(&path)?;
    }
    fs::rename(&temporary, &path)?;
    Ok(path)
}

fn ensure_readme(directory: &Path, contents: &[u8]) -> Result<()> {
    fs::create_dir_all(directory)?;
    let readme = directory.join("README.md");
    if !readme.exists() {
        let mut writer = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(readme)?;
        writer.write_all(contents)?;
        writer.flush()?;
    }
    Ok(())
}

fn write_missed_rows(path: &Path, records: &[MissedRecord]) -> Result<()> {
    let mut writer = OpenOptions::new().create_new(true).write(true).open(path)?;
    writeln!(writer, "# record_id\tpage_id")?;
    for record in records {
        writeln!(writer, "{}\t{}", record.record_id, record.page_id)?;
    }
    writer.flush()?;
    Ok(())
}

fn read_missed_for_epoch(work_dir: &Path, epoch: usize) -> Result<Vec<MissedRecord>> {
    let path = missed_file(work_dir, epoch, false);
    read_missed_records(&path)
}

fn save_recovery_checkpoint(
    o: &Options,
    model: &Model,
    pass: &mut Pass,
    stop: &TrainingStop,
    page_cache: &Path,
) -> Result<PathBuf> {
    let recovery = o.work_dir.join("recovery");
    ensure_readme(
        &recovery,
        b"# CRF recovery checkpoints\n\nGenerated complete checkpoint generations are atomic directories. Use the newest committed generation to resume. Keep at most the latest two; incomplete `.tmp` directories can be ignored after verifying no CRF process is running. These files are derived from the source corpus and initial model.\n",
    )?;
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_nanos();
    let stem = format!(
        "checkpoint-epoch-{:04}-batch-{:010}-{nonce}",
        stop.epoch, pass.batches
    );
    let temporary = recovery.join(format!("{stem}.tmp"));
    let committed = recovery.join(&stem);
    fs::create_dir(&temporary)?;
    output::save(model, &temporary.join("stopped-model.json"))?;
    for name in [
        "config.json",
        "input-report.json",
        "training-probe.json",
        "test-sample.json",
    ] {
        let source = o.output.join(name);
        if source.is_file() {
            fs::copy(source, temporary.join(name))?;
        }
    }
    write_json(
        &temporary.join("resume-info.json"),
        &serde_json::json!({
            "page_cache_directory":page_cache,
            "work_directory":o.work_dir,
            "recovery_snapshot":true
        }),
    )?;
    write_json(&temporary.join("training-stop.json"), stop)?;
    let fingerprint = output::fingerprint(model);
    write_json(&temporary.join("resume-fingerprint.json"), &fingerprint)?;
    let missed_path = temporary.join("missed-records.tsv");
    pass.missed_records
        .sort_unstable_by_key(|record| record.record_id);
    if pass
        .missed_records
        .windows(2)
        .any(|pair| pair[0].record_id == pair[1].record_id)
    {
        return Err("duplicate source record in recovery missed list".into());
    }
    write_missed_rows(&missed_path, &pass.missed_records)?;
    write_json(
        &temporary.join("partial-training.json"),
        &serde_json::json!({
            "epoch":stop.epoch,
            "training_sentences_processed":pass.processed,
            "training_batches":pass.batches,
            "training_update_seconds":0.0,
            "checkpoint":"stopped-model.json",
            "first_loss_sum":pass.first_loss,
            "primary_objective":primary_objective(o),
            "primary_objective_loss_sum":pass.first_loss,
            "primary_nbest_margin_active_sentences":pass.primary_nbest_margin_active_sentences,
            "primary_nbest_listwise_active_sentences":pass.primary_nbest_listwise_active_sentences,
            "rerank_loss_sum":pass.tri_loss,
            "rerank_sentences":pass.tri_count,
            "gold_missing_nbest":pass.missed,
            "training_preupdate_top1_gold_hits":pass.top1_hits,
            "training_preupdate_top1_evaluated_sentences":pass.top1_evaluated,
            "training_post_trigram_top1_gold_hits":pass.post_trigram_top1_hits,
            "training_post_trigram_top1_evaluated_sentences":pass.post_trigram_top1_evaluated,
            "gold_augmented_sentences":pass.gold_augmented,
            "missed_records_file":"missed-records.tsv",
            "exact_resume":true
        }),
    )?;
    write_json(
        &temporary.join("metrics.json"),
        &serde_json::json!([{
            "epoch":stop.epoch,
            "training_sentences_processed":pass.processed,
            "training_gold_missing_nbest":pass.missed
        }]),
    )?;
    let mut marker = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(temporary.join("COMMITTED"))?;
    writeln!(marker, "{nonce}")?;
    marker.sync_all()?;
    // Windows can reject a directory rename while a file handle inside the
    // directory is still open. The marker is complete and durable, so close it
    // before publishing the checkpoint generation.
    drop(marker);
    fs::rename(&temporary, &committed)?;
    if o.missed_feedback {
        let published = write_missed_records(&o.work_dir, stop.epoch, &pass.missed_records, true)?;
        eprintln!("recovery_missed_snapshot={}", published.display());
    }
    eprintln!(
        "recovery_checkpoint committed={} epoch={} sentences={} batches={} missed={}",
        committed.display(),
        stop.epoch,
        pass.processed,
        pass.batches,
        pass.missed_records.len()
    );
    let mut generations = fs::read_dir(&recovery)?
        .filter_map(std::result::Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_dir()
                && path
                    .file_name()
                    .is_some_and(|name| name.to_string_lossy().starts_with("checkpoint-"))
                && path.join("COMMITTED").is_file()
        })
        .collect::<Vec<_>>();
    generations.sort();
    for old in generations.iter().take(generations.len().saturating_sub(2)) {
        fs::remove_dir_all(old)?;
    }
    Ok(committed)
}

fn read_missed_records(path: &Path) -> Result<Vec<MissedRecord>> {
    let mut records = Vec::new();
    for line in std::fs::read_to_string(path)?.lines() {
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let (record_id, page_id) = line
            .split_once('\t')
            .ok_or_else(|| format!("invalid missed record row in {}", path.display()))?;
        records.push(MissedRecord {
            record_id: record_id.parse()?,
            page_id: page_id.parse()?,
        });
    }
    if records
        .windows(2)
        .any(|pair| pair[0].record_id >= pair[1].record_id)
    {
        return Err(format!(
            "missed record IDs are not strictly sorted in {}",
            path.display()
        )
        .into());
    }
    Ok(records)
}

#[derive(Debug, Default)]
struct Pass {
    processed: usize,
    selected: usize,
    first_loss: f64,
    primary_nbest_margin_active_sentences: usize,
    primary_nbest_listwise_active_sentences: usize,
    tri_loss: f64,
    tri_count: usize,
    missed: usize,
    top1_hits: usize,
    top1_evaluated: usize,
    post_trigram_top1_hits: usize,
    post_trigram_top1_evaluated: usize,
    missed_records: Vec<MissedRecord>,
    gold_augmented: usize,
    max_nodes: usize,
    max_edges: usize,
    batches: usize,
    max_batch_nodes: usize,
    max_batch_edges: usize,
    block_order: Vec<usize>,
    active_page_block: Option<usize>,
    stop: Option<TrainingStop>,
    timing: BatchTiming,
    lattice_prepare_seconds: f64,
    cutoff_events: Vec<CutoffEvent>,
}

struct HogwildRuntime {
    trainer: hogwild_crf::Trainer,
    master: hogwild_crf::Master,
}

impl HogwildRuntime {
    fn new(model: &Model, threads: usize) -> Result<Self> {
        Ok(Self {
            trainer: hogwild_crf::Trainer::new(threads)?,
            master: hogwild_crf::Master::from_model(model),
        })
    }

    fn config(o: &Options, rate: f64) -> hogwild_crf::Config {
        // Matching the previous mean-gradient update's nominal per-example
        // contribution: η / batch_size. Online stale reads still change the path.
        let per_sentence_rate = rate / o.batch_size as f64;
        hogwild_crf::Config {
            rate: per_sentence_rate,
            trigram_rate: per_sentence_rate,
            l2: o.l2,
            nbest: o.nbest,
            astar_limit: o.max_astar_expansions,
            add_gold_for_trigram: o.gold_path_augmentation,
        }
    }

    fn snapshot_into(&self, model: &mut Model) {
        self.master.snapshot_into_model(model);
    }
}

#[allow(clippy::too_many_arguments)] // One window owns the complete streaming update context.
fn flush_window(
    buffer: &mut Vec<Buffered>,
    model: &mut Model,
    index: &Index,
    o: &Options,
    epoch: usize,
    state: &mut u64,
    pass: &mut Pass,
    source_records: usize,
    rate: f64,
    started: Instant,
    trainer: &Trainer,
    validation: &[Sentence],
    curve: &mut CurveLog,
    hogwild: Option<&HogwildRuntime>,
    shuffle: bool,
    recovery_page_cache: Option<&Path>,
) -> bool {
    if shuffle {
        corpus::shuffle(buffer, state);
    }
    if let Some(hogwild) = hogwild {
        let sentences = buffer
            .iter()
            .map(|entry| &entry.sentence)
            .collect::<Vec<_>>();
        let report = match hogwild.master.train_sentences(
            &hogwild.trainer,
            model,
            &sentences,
            index,
            limits(o),
            HogwildRuntime::config(o, rate),
        ) {
            Ok(report) => report,
            Err(error) => {
                hogwild.snapshot_into(model);
                pass.stop = Some(TrainingStop {
                    epoch,
                    kind: "hogwild-error".into(),
                    reason: error.to_string(),
                    source_records_read: source_records,
                    shuffle_state: *state,
                    pending_window: buffer.clone(),
                    page_block_order: pass.block_order.clone(),
                    active_page_block: pass.active_page_block,
                });
                return false;
            }
        };
        pass.processed += report.sentences;
        pass.batches += 1; // One completed target-store window, not a gradient batch.
        pass.first_loss += report.primary_loss_sum;
        pass.tri_loss += report.trigram_loss_sum;
        pass.tri_count += report.raw_nbest_gold_hits + report.gold_added_for_trigram;
        pass.missed += report.sentences - report.raw_nbest_gold_hits;
        pass.top1_hits += report.viterbi_gold_hits;
        pass.top1_evaluated += report.sentences;
        pass.gold_augmented += report.gold_added_for_trigram;
        pass.timing.joint_parallel_wall_seconds += report.wall_seconds;

        // Window completion is a quiescent point for heldout evaluation and
        // checkpoint snapshots; sentence updates never wait for one another.
        if curve.needs_record(pass.processed) {
            hogwild.snapshot_into(model);
            if let Err(error) =
                curve.maybe_record(model, index, validation, o, trainer, pass, epoch, started)
            {
                pass.stop = Some(TrainingStop {
                    epoch,
                    kind: "progress-evaluation".into(),
                    reason: error.to_string(),
                    source_records_read: source_records,
                    shuffle_state: *state,
                    pending_window: Vec::new(),
                    page_block_order: pass.block_order.clone(),
                    active_page_block: pass.active_page_block,
                });
                return false;
            }
        }
        // Hogwild runs are currently bounded, and every completed shuffle
        // window is the natural point to publish cumulative progress rates.
        let top1 =
            (pass.top1_evaluated > 0).then(|| pass.top1_hits as f64 / pass.top1_evaluated as f64);
        let nbest = (pass.processed > 0)
            .then(|| (pass.processed - pass.missed) as f64 / pass.processed as f64);
        eprintln!(
            "epoch={epoch} trained={} hogwild_window={} online_top1={} online_nbest_hit={} elapsed={:.1}s first_keys={} trigram_keys={}",
            pass.processed,
            report.sentences,
            top1.map_or_else(
                || "n/a".to_owned(),
                |x| format!("{x:.4} ({}/{})", pass.top1_hits, pass.top1_evaluated)
            ),
            nbest.map_or_else(
                || "n/a".to_owned(),
                |x| format!(
                    "{x:.4} ({}/{})",
                    pass.processed - pass.missed,
                    pass.processed
                )
            ),
            started.elapsed().as_secs_f64(),
            hogwild.master.first_len(),
            hogwild.master.third_len()
        );
        if let Some(error) = report.error {
            hogwild.snapshot_into(model);
            pass.stop = Some(TrainingStop {
                epoch,
                kind: "hogwild-error".into(),
                reason: error,
                source_records_read: source_records,
                shuffle_state: *state,
                pending_window: buffer.clone(),
                page_block_order: pass.block_order.clone(),
                active_page_block: pass.active_page_block,
            });
            return false;
        }
        if report.sentences != buffer.len() {
            hogwild.snapshot_into(model);
            pass.stop = Some(TrainingStop {
                epoch,
                kind: "hogwild-incomplete-window".into(),
                reason: format!(
                    "processed {} of {} sentences",
                    report.sentences,
                    buffer.len()
                ),
                source_records_read: source_records,
                shuffle_state: *state,
                pending_window: buffer.clone(),
                page_block_order: pass.block_order.clone(),
                active_page_block: pass.active_page_block,
            });
            return false;
        }
        buffer.clear();
        return true;
    }
    for (batch_index, entries) in buffer.chunks(o.batch_size).enumerate() {
        let sentences = entries
            .iter()
            .map(|entry| &entry.sentence)
            .collect::<Vec<_>>();
        let prepare_started = Instant::now();
        let prepared = match trainer.prepare(&sentences, model, index, limits(o)) {
            Ok(prepared) => prepared,
            Err(error) => {
                pass.stop = Some(TrainingStop {
                    epoch,
                    kind: "lattice-error".into(),
                    reason: format!(
                        "batch starting record {} page {}: {error}",
                        entries[0].record, entries[0].sentence.page_id
                    ),
                    source_records_read: source_records,
                    shuffle_state: *state,
                    pending_window: buffer[batch_index * o.batch_size..].to_vec(),
                    page_block_order: pass.block_order.clone(),
                    active_page_block: pass.active_page_block,
                });
                return false;
            }
        };
        pass.lattice_prepare_seconds += prepare_started.elapsed().as_secs_f64();
        for sentence in &prepared {
            pass.max_nodes = pass.max_nodes.max(sentence.graph.nodes.len());
            pass.max_edges = pass.max_edges.max(sentence.graph.edges.len());
        }
        pass.max_batch_nodes = pass
            .max_batch_nodes
            .max(prepared.iter().map(|p| p.graph.nodes.len()).sum());
        pass.max_batch_edges = pass
            .max_batch_edges
            .max(prepared.iter().map(|p| p.graph.edges.len()).sum());
        let graphs = prepared.iter().map(|p| &p.graph).collect::<Vec<_>>();
        let config = BatchConfig {
            rate,
            l2: o.l2,
            nbest: o.nbest,
            astar_limit: o.max_astar_expansions,
            parameter_bytes: o.max_parameter_mib * (1 << 20),
            gold_path_augmentation: o.gold_path_augmentation,
            primary_nbest_margin: o.primary_nbest_margin,
            primary_nbest_listwise: o.primary_nbest_listwise,
        };
        let processed_before_batch = pass.processed;
        let mut cutoff_applied = false;
        let mut result = trainer.train_batch(model, &graphs, config);
        if o.delta_cutoff > 0.0 {
            if let Some(reason) = result
                .as_ref()
                .err()
                .and_then(|e| e.downcast_ref::<ParameterLimit>())
            {
                // train_batch left both parameter families unmodified. Cutoff is a separate,
                // logged model operation; then recompute this same batch once.
                match recover_capacity(
                    model,
                    index,
                    o,
                    trainer,
                    validation,
                    epoch,
                    pass,
                    entries[0].record,
                    reason,
                ) {
                    Ok(event) => {
                        pass.cutoff_events.push(event);
                        cutoff_applied = true;
                        result = trainer.train_batch(model, &graphs, config);
                    }
                    Err(error) => result = Err(error),
                }
            }
        }
        match result {
            Ok(report) => {
                pass.timing.add(&report.timing);
                pass.processed += report.sentences;
                pass.batches += 1;
                pass.first_loss += report.first_loss_sum;
                pass.primary_nbest_margin_active_sentences +=
                    report.primary_margin_active_sentences;
                pass.primary_nbest_listwise_active_sentences +=
                    report.primary_listwise_active_sentences;
                pass.tri_count += report.rerank_sentences;
                pass.tri_loss += report.rerank_loss_sum;
                pass.missed += report.sentences - report.nbest_gold_hits;
                pass.top1_hits += report.first_path_gold_hits;
                pass.top1_evaluated += report.sentences;
                pass.post_trigram_top1_hits += report.post_trigram_top1_gold_hits;
                pass.post_trigram_top1_evaluated += report.sentences;
                if o.missed_feedback {
                    for &missed_index in &report.post_trigram_top1_missed_indices {
                        let Some(entry) = entries.get(missed_index) else {
                            pass.stop = Some(TrainingStop {
                                epoch,
                                kind: "internal-report-error".into(),
                                reason: "post-trigram miss index outside committed batch".into(),
                                source_records_read: source_records,
                                shuffle_state: *state,
                                pending_window: buffer[(batch_index + 1) * o.batch_size..].to_vec(),
                                page_block_order: pass.block_order.clone(),
                                active_page_block: pass.active_page_block,
                            });
                            return false;
                        };
                        pass.missed_records.push(MissedRecord {
                            record_id: entry.record,
                            page_id: entry.sentence.page_id,
                        });
                    }
                }
                pass.gold_augmented += report.gold_augmented_sentences;
            }
            Err(e) => {
                pass.stop = Some(TrainingStop {
                    epoch,
                    kind: if e
                        .downcast_ref::<ParameterLimit>()
                        .is_some_and(|limit| limit.allocation_error.is_none())
                    {
                        "parameter-limit"
                    } else {
                        "training-error"
                    }
                    .into(),
                    reason: format!(
                        "batch starting record {} page {}: {e}",
                        entries[0].record, entries[0].sentence.page_id
                    ),
                    source_records_read: source_records,
                    shuffle_state: *state,
                    pending_window: buffer[batch_index * o.batch_size..].to_vec(),
                    page_block_order: pass.block_order.clone(),
                    active_page_block: pass.active_page_block,
                });
                return false;
            }
        }
        let crossed_recovery_interval =
            pass.processed / o.recovery_interval > processed_before_batch / o.recovery_interval;
        if crossed_recovery_interval && hogwild.is_none() {
            for event in cutoff::apply_count_caps(model) {
                eprintln!(
                    "count_cutoff epoch={epoch} trained={} family={} entries={}->{} selected_min_abs_delta={:.6e} temporary_selection_mib={:.1}",
                    pass.processed,
                    event.family,
                    event.entries_before,
                    event.entries_after,
                    event.selected_minimum_abs_delta,
                    event.temporary_selection_bytes as f64 / f64::from(1 << 20),
                );
                if let Err(error) = append_count_cutoff(o, epoch, pass, &event) {
                    pass.stop = Some(TrainingStop {
                        epoch,
                        kind: "count-cutoff-log-error".into(),
                        reason: error.to_string(),
                        source_records_read: source_records,
                        shuffle_state: *state,
                        pending_window: buffer[(batch_index + 1) * o.batch_size..].to_vec(),
                        page_block_order: pass.block_order.clone(),
                        active_page_block: pass.active_page_block,
                    });
                    return false;
                }
            }
        }
        if let Err(error) =
            curve.maybe_record(model, index, validation, o, trainer, pass, epoch, started)
        {
            pass.stop = Some(TrainingStop {
                epoch,
                kind: "progress-evaluation".into(),
                reason: error.to_string(),
                source_records_read: source_records,
                shuffle_state: *state,
                pending_window: buffer[(batch_index + 1) * o.batch_size..].to_vec(),
                page_block_order: pass.block_order.clone(),
                active_page_block: pass.active_page_block,
            });
            return false;
        }
        if cutoff_applied || crossed_recovery_interval {
            if let Some(page_cache) = recovery_page_cache {
                let stop = TrainingStop {
                    epoch,
                    kind: "recovery-checkpoint".into(),
                    reason: "periodic committed-batch recovery point".into(),
                    source_records_read: source_records,
                    shuffle_state: *state,
                    pending_window: buffer[(batch_index + 1) * o.batch_size..].to_vec(),
                    page_block_order: pass.block_order.clone(),
                    active_page_block: pass.active_page_block,
                };
                if let Err(error) = save_recovery_checkpoint(o, model, pass, &stop, page_cache) {
                    pass.stop = Some(TrainingStop {
                        kind: "recovery-checkpoint-error".into(),
                        reason: error.to_string(),
                        ..stop
                    });
                    return false;
                }
            }
        }
        if pass.processed % 100 < entries.len() {
            let nbest_hit_rate = (pass.processed > 0)
                .then(|| (pass.processed - pass.missed) as f64 / pass.processed as f64);
            let top1_rate = (pass.top1_evaluated > 0)
                .then(|| pass.top1_hits as f64 / pass.top1_evaluated as f64);
            eprintln!(
                "epoch={epoch} trained={} batches={} online_top1={} online_nbest_hit={} elapsed={:.1}s sparse_mib={:.1} lattice_nodes={} lattice_edges={}",
                pass.processed,
                pass.batches,
                top1_rate.map_or_else(
                    || format!("n/a ({}/{})", pass.top1_hits, pass.top1_evaluated),
                    |x| format!("{x:.4} ({}/{})", pass.top1_hits, pass.top1_evaluated)
                ),
                nbest_hit_rate.map_or_else(
                    || "n/a".to_owned(),
                    |x| format!(
                        "{x:.4} ({}/{})",
                        pass.processed - pass.missed,
                        pass.processed
                    )
                ),
                started.elapsed().as_secs_f64(),
                (model.first.values.projected_bytes(0) + model.third.values.projected_bytes(0))
                    as f64
                    / f64::from(1 << 20),
                pass.max_nodes,
                pass.max_edges
            );
        }
        // The complete batch's lattices are dropped before the next batch.
    }
    buffer.clear();
    true
}

#[allow(clippy::too_many_arguments)] // Shared epoch resources, not duplicated model state.
fn run_epoch(
    o: &Options,
    model: &mut Model,
    index: &Index,
    input: &InputReport,
    epoch: usize,
    missed_input: Option<&[MissedRecord]>,
    state: &mut u64,
    rate: f64,
    trainer: &Trainer,
    page_blocks: Option<&PageBlocks>,
    resume: Option<&Resume>,
    validation: &[Sentence],
    curve: &mut CurveLog,
    hogwild: Option<&HogwildRuntime>,
) -> Result<Pass> {
    if let Some(blocks) = page_blocks {
        let result = run_block_epoch(
            o,
            model,
            index,
            epoch,
            missed_input,
            state,
            rate,
            trainer,
            blocks,
            resume,
            validation,
            curve,
            hogwild,
        );
        if let Some(hogwild) = hogwild {
            hogwild.snapshot_into(model);
        }
        return result;
    }
    let started = Instant::now();
    let mut reader = corpus::open(&o.input)?;
    let mut line = String::new();
    let mut buffer = Vec::with_capacity(o.shuffle_buffer);
    let mut pass = Pass::default();
    for number in 1..=input.records_read {
        let record = corpus::read_record(&mut *reader, &mut line, number)?
            .ok_or("corpus ended before selected prefix; input changed")?;
        if number <= o.skip_records
            || corpus::validation_page(record.page_id)
            || pass.selected >= input.training_sentences
        {
            continue;
        }
        let Ok(sentence) = corpus::normalize(record, index, o.max_words) else {
            continue;
        };
        if let Some(misses) = missed_input {
            if misses
                .binary_search_by_key(&number, |miss| miss.record_id)
                .is_err()
            {
                continue;
            }
        }
        pass.selected += 1;
        buffer.push(Buffered {
            record: number,
            sentence,
        });
        if buffer.len() == o.shuffle_buffer
            && !flush_window(
                &mut buffer,
                model,
                index,
                o,
                epoch,
                state,
                &mut pass,
                number,
                rate,
                started,
                trainer,
                validation,
                curve,
                hogwild,
                true,
                None,
            )
        {
            return Ok(pass);
        }
    }
    if !buffer.is_empty() {
        flush_window(
            &mut buffer,
            model,
            index,
            o,
            epoch,
            state,
            &mut pass,
            input.records_read,
            rate,
            started,
            trainer,
            validation,
            curve,
            hogwild,
            true,
            None,
        );
    }
    if pass.stop.is_none() && pass.processed != input.training_sentences {
        return Err("selected training records changed between passes".into());
    }
    if let Some(hogwild) = hogwild {
        hogwild.snapshot_into(model);
    }
    Ok(pass)
}

#[allow(clippy::too_many_arguments)] // The same shared resources as the streaming epoch.
fn run_block_epoch(
    o: &Options,
    model: &mut Model,
    index: &Index,
    epoch: usize,
    missed_input: Option<&[MissedRecord]>,
    state: &mut u64,
    rate: f64,
    trainer: &Trainer,
    blocks: &PageBlocks,
    resume: Option<&Resume>,
    validation: &[Sentence],
    curve: &mut CurveLog,
    hogwild: Option<&HogwildRuntime>,
) -> Result<Pass> {
    let started = Instant::now();
    let mut pass = Pass {
        block_order: blocks.order(o.seed, epoch),
        ..Pass::default()
    };
    let mut buffer = Vec::with_capacity(o.shuffle_buffer);
    let mut line = String::new();
    let mut start_position = 0;
    if let Some(resume) = resume {
        if pass.block_order != resume.stop.page_block_order {
            return Err("resume block order differs from seed/epoch".into());
        }
        resume.restore(&mut pass, state);
        start_position = resume.position(blocks)?;
        // Validate the active cache prefix BEFORE committing any resumed update.
        resume.validate_prefix(blocks, missed_input)?;
    }
    for position in start_position..pass.block_order.len() {
        let block = pass.block_order[position];
        pass.active_page_block = Some(block);
        eprintln!(
            "epoch={epoch} page_block={block} order={}/{} pages={} sentences={}",
            position + 1,
            pass.block_order.len(),
            blocks.blocks[block].pages,
            blocks.blocks[block].sentences
        );
        let mut reader = blocks.open(block)?;
        let mut source_record = blocks.blocks[block].source_first_record;
        let mut cache_seen = 0;
        if position == start_position {
            if let Some(resume) = resume {
                buffer.clone_from(&resume.stop.pending_window);
                // These entries are already shuffled, and RNG is after that
                // shuffle. Shuffling again would change batches and weights.
                if !flush_window(
                    &mut buffer,
                    model,
                    index,
                    o,
                    epoch,
                    state,
                    &mut pass,
                    resume.stop.source_records_read,
                    rate,
                    started,
                    trainer,
                    validation,
                    curve,
                    hogwild,
                    false,
                    Some(&blocks.directory),
                ) {
                    return Ok(pass);
                }
            }
        }
        while let Some(entry) = blocks::read_cached(&mut *reader, &mut line)? {
            source_record = entry.record;
            cache_seen += 1;
            if position == start_position
                && resume.is_some_and(|r| entry.record <= r.stop.source_records_read)
            {
                continue;
            }
            if let Some(misses) = missed_input {
                let Ok(miss_index) =
                    misses.binary_search_by_key(&entry.record, |miss| miss.record_id)
                else {
                    continue;
                };
                if misses[miss_index].page_id != entry.sentence.page_id {
                    return Err(format!("missed record {} changed page ID", entry.record).into());
                }
            }
            pass.selected += 1;
            buffer.push(entry);
            if buffer.len() == o.shuffle_buffer
                && !flush_window(
                    &mut buffer,
                    model,
                    index,
                    o,
                    epoch,
                    state,
                    &mut pass,
                    source_record,
                    rate,
                    started,
                    trainer,
                    validation,
                    curve,
                    hogwild,
                    true,
                    Some(&blocks.directory),
                )
            {
                return Ok(pass);
            }
        }
        if cache_seen != blocks.blocks[block].sentences {
            return Err("page block sentence count changed".into());
        }
        if !buffer.is_empty()
            && !flush_window(
                &mut buffer,
                model,
                index,
                o,
                epoch,
                state,
                &mut pass,
                source_record,
                rate,
                started,
                trainer,
                validation,
                curve,
                hogwild,
                true,
                Some(&blocks.directory),
            )
        {
            return Ok(pass);
        }
    }
    let expected_sentences = missed_input.map_or(blocks.training_sentences, <[MissedRecord]>::len);
    if pass.processed != expected_sentences {
        return Err("page block training count mismatch".into());
    }
    pass.active_page_block = None;
    Ok(pass)
}

fn feature_counts(model: &Model) -> BTreeMap<String, usize> {
    let mut sizes = [0_usize; 8];
    for feature in model.first.values.keys() {
        let index = match feature {
            Feature::Word(_) => 0,
            Feature::Pos(..) => 1,
            Feature::Bigram(..) => 2,
            Feature::PosWord(..) => 3,
            Feature::WordPos(..) => 4,
            Feature::Script(_) => 5,
            Feature::ScriptBigram(..) => 6,
            Feature::ReadingLengthBigram(..) => 7,
        };
        sizes[index] += 1;
    }
    let mut counts = BTreeMap::new();
    for (name, count) in [
        "word",
        "pos-bigram",
        "word-bigram",
        "pos-word-bigram",
        "word-pos-bigram",
        "surface-unigram",
        "surface-bigram",
        "reading-length-bigram",
    ]
    .into_iter()
    .zip(sizes)
    {
        if count > 0 {
            counts.insert(name.to_owned(), count);
        }
    }
    counts.insert("word-trigram".to_owned(), model.third.values.len());
    counts.insert("pos-trigram".to_owned(), model.third_pos.values.len());
    counts.insert(
        "reading-length-trigram".to_owned(),
        model.third_reading_length.values.len(),
    );
    counts.insert(
        "static-initial-word-bigram".to_owned(),
        model.initial_bigrams.len(),
    );
    counts.insert(
        "static-initial-word-trigram".to_owned(),
        model.initial_trigrams.len(),
    );
    counts
}

fn write_json(path: &std::path::Path, value: &impl Serialize) -> Result<()> {
    let mut writer = BufWriter::new(OpenOptions::new().create_new(true).write(true).open(path)?);
    serde_json::to_writer_pretty(&mut writer, value)?;
    writeln!(writer)?;
    writer.flush()?;
    Ok(())
}

fn append_count_cutoff(
    o: &Options,
    epoch: usize,
    pass: &Pass,
    event: &cutoff::CountCapEvent,
) -> Result<()> {
    let path = o.output.join("count-cutoff.jsonl");
    let mut writer = BufWriter::new(OpenOptions::new().create(true).append(true).open(path)?);
    serde_json::to_writer(
        &mut writer,
        &serde_json::json!({
            "epoch": epoch,
            "committed_sentences": pass.processed,
            "committed_batches": pass.batches,
            "family": event.family,
            "trigger_entries": event.trigger_entries,
            "entries_before": event.entries_before,
            "entries_after": event.entries_after,
            "selected_minimum_abs_delta": event.selected_minimum_abs_delta,
            "temporary_selection_bytes": event.temporary_selection_bytes,
        }),
    )?;
    writeln!(writer)?;
    writer.flush()?;
    Ok(())
}

fn run(o: &Options) -> Result<()> {
    let started = Instant::now();
    let resume = o
        .resume_from
        .as_ref()
        .map(|path| Resume::load(path, o))
        .transpose()?;
    let trainer = Trainer::with_reference_kernels(o.threads, o.reference_kernels)?;
    eprintln!(
        "internal_threads={} batch_size={} execution_mode={} l2={}",
        trainer.threads(),
        o.batch_size,
        if o.hogwild {
            "Hogwild sentence-stream; scc updates immediately, join at progress window"
        } else {
            "synchronous mini-batch; first+trigram updates after batch join"
        },
        o.l2
    );
    if let Some(path) = &o.model_in {
        eprintln!("loading immutable input model {}", path.display());
    } else {
        eprintln!("loading dictionary seeds (all raw word bigrams for POS statistics; no cutoff)");
    }
    let mut model = if let Some(path) = &o.model_in {
        output::load(path)?
    } else if o.initial_bigram_count > 0 || o.initial_trigram_count > 0 {
        let (seeds, bigrams, trigrams) = ai_engine_dict::training_seeds_with_ngrams(
            &o.corpus_statistics,
            &o.system_dictionary,
            &o.user_additions,
            o.initial_bigram_count,
            o.initial_trigram_count,
        )?;
        let model = Model::from_seeds_with_ngrams(
            seeds,
            bigrams,
            trigrams,
            o.missing_pos_probability,
            o.initial_bigram_delta,
            o.initial_trigram_weight,
        )?;
        eprintln!(
            "static_initial_word_bigrams={} mixture_delta={} (baseline weight; empirical weight={})",
            model.initial_bigrams.len(),
            o.initial_bigram_delta,
            1.0 - o.initial_bigram_delta
        );
        eprintln!(
            "static_initial_word_trigrams={} empirical_weight={} bigram_backoff_weight={}",
            model.initial_trigrams.len(),
            o.initial_trigram_weight,
            1.0 - o.initial_trigram_weight
        );
        model
    } else {
        Model::from_seeds(
            ai_engine_dict::training_seeds(
                &o.corpus_statistics,
                &o.system_dictionary,
                &o.user_additions,
            )?,
            o.missing_pos_probability,
        )?
    };
    eprintln!(
        "dictionary words={} POS={} seed_load={:.1}s sparse_bigrams={} learned_trigrams={} initial_trigrams={}",
        model.words.len(),
        model.pos_names.len() - 1,
        started.elapsed().as_secs_f64(),
        feature_counts(&model)
            .get("word-bigram")
            .copied()
            .unwrap_or(0),
        model.third.values.len(),
        model.initial_trigrams.len()
    );
    let initial_fingerprint = output::fingerprint(&model);
    let hogwild = o
        .hogwild
        .then(|| HogwildRuntime::new(&model, o.threads))
        .transpose()?;
    if o.hogwild {
        eprintln!(
            "hogwild_per_sentence_rate={} (learning_rate / batch_size); sentence-local lattice is dropped after immediate scc update",
            o.learning_rate / o.batch_size as f64
        );
        eprintln!(
            "warning: --max-parameter-mib is not enforced in --hogwild mode; this bounded run has no exact resume or delta cutoff"
        );
    }
    if let Some(resume) = &resume {
        resume.validate_model(&initial_fingerprint)?;
    }
    let index = Index::new(&model)?;
    fs::create_dir_all(&o.output)?;
    ensure_readme(
        &o.output,
        format!(
            "# CRF chunk003 outputs\n\n- Status: in progress.\n- Source corpus: `{}`\n- Initial/model source: `{}`\n- Purpose: trained CRF model, per-epoch metrics, fixed test sample and final verification.\n- Work data: `{}`\n- Deletion: derived outputs; do not remove while a CRF process is active. The immutable input corpus and initial model are not owned by this run.\n",
            o.input.display(),
            o.model_in.as_deref().unwrap_or(&o.corpus_statistics).display(),
            o.work_dir.display()
    )
        .as_bytes(),
    )?;
    fs::create_dir_all(&o.work_dir)?;
    if let Some(sample_path) = &o.evaluation_sample_in {
        ensure_readme(
            &o.work_dir,
            format!(
                "# CRF fixed-sample evaluation work\n\n- Model source: `{}`\n- Fixed test source: `{}`\n- Purpose: evaluation-only; no training, model updates, or corpus cache is written here.\n- Evaluation reports: `{}`\n- Deletion: this run-specific derived work directory can be removed after reviewing the report.\n",
                o.model_in.as_deref().unwrap_or(&o.corpus_statistics).display(),
                sample_path.display(),
                o.output.display()
            )
            .as_bytes(),
        )?;
    }
    let page_cache_note = o.prepared_data_in.as_ref().map_or_else(
        || {
            format!(
                "- Page cache: created from the current source corpus and this initial model's word/POS IDs at `{}`; no prior normalized cache or learned checkpoint is reused.\n",
                o.work_dir.display()
            )
        },
        |source| {
            format!(
                "- Page cache: reused from `{}`; it contains normalized source examples, not learned weights.\n",
                source.display()
            )
        },
    );
    ensure_readme(
        &o.work_dir,
        format!(
            "# CRF chunk003 work data\n\n- Status: in progress.\n- Source corpus: `{}`\n- Initial model: `{}`\n- Contents: exact recovery snapshots, a fresh page-block cache, and `missed-chunk003` lists for later epochs.\n{page_cache_note}- Next use: same run's next epoch and exact resume.\n- Deletion: derived data; preserve this folder until training/recovery are complete.\n",
            o.input.display(),
            o.model_in.as_deref().unwrap_or(&o.corpus_statistics).display()
        )
        .as_bytes(),
    )?;
    write_json(&o.output.join("config.json"), o)?;
    write_json(
        &o.output.join("training-update-policy.json"),
        &serde_json::json!({
            "primary_objective":primary_objective(o),
            "primary_nbest_margin_log_score":o.primary_nbest_margin,
            "primary_nbest_listwise":o.primary_nbest_listwise,
            "primary_objective_definition":primary_objective_definition(o),
            "l2":o.l2,
            "l2_effect":"0 disables decay of learned deltas toward the static initial scores",
            "word_bigram_reduction":"sum all minibatch sentence gradients; do not divide by batch size",
            "word_trigram_reduction":"sum all eligible N-best/gold-augmented sentence gradients; do not divide by batch size",
            "other_primary_feature_reduction":"mean across the full minibatch, treating absent feature gradients as zero",
            "pos_and_reading_length_trigram_reduction":"mean across sentences with a valid trigram candidate set",
            "synchronous_update":"compute all gradients against one pre-update model; apply once after worker join"
        }),
    )?;
    if let Some(sample_path) = &o.evaluation_sample_in {
        let samples: Vec<Sentence> =
            serde_json::from_reader(BufReader::new(fs::File::open(sample_path)?))?;
        if samples.is_empty() {
            return Err("fixed evaluation sample is empty".into());
        }
        let metrics = evaluate(&model, &index, &samples, o, &trainer)?;
        write_json(
            &o.output.join("evaluation.json"),
            &serde_json::json!({
                "model_in":o.model_in,
                "evaluation_sample_in":sample_path,
                "gold_path_augmentation_applied":false,
                "gold_path_augmentation_note":"augmentation is training-only; this report measures raw N-best recall",
                "metrics":metrics
            }),
        )?;
        if o.feature_ablation_evaluation {
            let report =
                evaluate_feature_ablations(&mut model, &index, &samples, o, &trainer, &metrics)?;
            write_json(&o.output.join("feature-ablation-evaluation.json"), &report)?;
        }
        if o.error_trigram_analysis {
            let report = error_analysis::analyze(&model, &index, &samples, o, &trainer)?;
            write_json(&o.output.join("error-trigram-analysis.json"), &report)?;
            let paired_trigram_rows = error_analysis::write_paired_tsv(
                &report,
                &o.output.join("paired-gold-vs-wrong-trigrams-top100.tsv"),
            )?;
            write_json(
                &o.output.join("summary.json"),
                &serde_json::json!({
                    "evaluation_sample_in":sample_path,
                    "sentences":metrics.sentences,
                    "nbest":metrics.nbest,
                    "character_level_accuracy_definition":"1 - total Unicode-scalar Levenshtein distance / sum(max(gold character count, converted character count)); micro-averaged over sentences",
                    "bigram_stage":{
                        "top1_accuracy":metrics.first_exact_accuracy,
                        "character_edit_distance":report.bigram_stage.character_edit_distance,
                        "character_denominator":report.bigram_stage.character_denominator,
                        "character_level_accuracy":report.bigram_stage.character_level_accuracy
                    },
                    "post_trigram_stage":{
                        "final_top1_accuracy":metrics.reranked_exact_accuracy,
                        "character_edit_distance":report.post_trigram_stage.character_edit_distance,
                        "character_denominator":report.post_trigram_stage.character_denominator,
                        "character_level_accuracy":report.post_trigram_stage.character_level_accuracy
                    },
                    "paired_trigram_report":"paired-gold-vs-wrong-trigrams-top100.tsv",
                    "paired_trigram_rows":paired_trigram_rows
                }),
            )?;
        }
        println!(
            "evaluation completed nbest={} sentences={} raw_nbest_gold_hit={:.4} wall_seconds={:.3} nbest_rerank_worker_seconds={:.3}",
            metrics.nbest,
            metrics.sentences,
            metrics.nbest_gold_recall,
            metrics.seconds,
            metrics.nbest_rerank_worker_seconds
        );
        return Ok(());
    }
    if o.save_initial_model {
        output::save(&model, &o.output.join("initial-model.json"))?;
    }
    fs::create_dir_all(&o.work_dir)?;
    let mut page_cache = if resume.is_none()
        && o.prepared_data_in.is_none()
        && o.epochs > 0
        && o.page_block_size > 0
    {
        Some(blocks::Builder::new(
            &o.work_dir.join("page-blocks"),
            o.page_block_size,
        )?)
    } else {
        None
    };
    let (train, dev, input_report, page_blocks) = if let Some(resume) = &resume {
        let (train, dev, report) = resume.samples()?;
        let blocks = PageBlocks::load(&resume.page_cache)?;
        if blocks.training_sentences != report.training_sentences
            || blocks.pages_per_block != o.page_block_size
        {
            return Err("resume cache/input-report mismatch".into());
        }
        write_json(
            &o.output.join("resume-info.json"),
            &serde_json::json!({
                "source_run":resume.directory,"page_cache_directory":resume.page_cache,
                "training_sentences_already_processed":resume.processed,"resume_epoch":resume.stop.epoch,
                "checkpoint_fingerprint":initial_fingerprint,"exact_model_resume":true
            }),
        )?;
        (train, dev, report, Some(blocks))
    } else if o.prepared_data_in.is_some() {
        let (train, dev, report, blocks) = load_prepared_data(o, &model)?;
        write_json(
            &o.output.join("resume-info.json"),
            &serde_json::json!({
                "page_cache_directory":blocks.directory,
                "prepared_data_source":o.prepared_data_in,
                "training_sentences_already_processed":0,
                "fresh_model_start":true,
                "model_fingerprint":initial_fingerprint
            }),
        )?;
        write_json(
            &o.output.join("prepared-data-info.json"),
            &serde_json::json!({
                "source_output":o.prepared_data_in,
                "source_page_cache":blocks.directory,
                "source_training_sentences":blocks.training_sentences,
                "weights_loaded_from":o.model_in,
                "learned_checkpoint_reused":false,
                "normalization_compatibility_checked":true
            }),
        )?;
        (train, dev, report, Some(blocks))
    } else {
        let (train, dev, report) = sample(o, &index, page_cache.as_mut())?;
        let page_blocks = page_cache.map(blocks::Builder::finish).transpose()?;
        write_json(
            &o.output.join("resume-info.json"),
            &serde_json::json!({
                "page_cache_directory":page_blocks.as_ref().map(|cache| cache.directory.clone()),
                "work_directory":o.work_dir,
                "test_prefix_sentences":o.test_prefix_sentences,
                "exact_resume_supported":page_blocks.is_some() && !o.hogwild
            }),
        )?;
        (train, dev, report, page_blocks)
    };
    // Resume always reuses the original ID-local evaluation samples, not a new
    // reservoir or a new corpus normalization against changed dictionaries.
    if resume.is_some() || o.prepared_data_in.is_some() || !o.full_corpus {
        write_json(&o.output.join("training-probe.json"), &train)?;
        write_json(&o.output.join("test-sample.json"), &dev)?;
    }
    eprintln!(
        "sample train={} ({} pages) dev={} ({} pages) rejected={:?}",
        input_report.training_sentences,
        input_report.training_pages,
        dev.len(),
        input_report.validation_pages,
        input_report.rejected
    );
    write_json(&o.output.join("input-report.json"), &input_report)?;
    let mut curve = CurveLog::create(
        &o.output.join("training-curve.jsonl"),
        o.progress_interval,
        o.progress_interval_seconds,
        input_report.training_sentences,
    )?;
    let mut reports = vec![];
    let mut random = resume.as_ref().map_or(o.seed, |r| r.stop.shuffle_state);
    let mut stop = None;
    for epoch in 0..=o.epochs {
        if epoch > 0 && resume.as_ref().is_some_and(|r| epoch < r.stop.epoch) {
            continue;
        }
        let update_started = Instant::now();
        let rate = o.learning_rate / (1.0 + epoch.saturating_sub(1) as f64 * 0.2);
        let missed_input = if o.missed_feedback && epoch > 1 {
            Some(read_missed_for_epoch(&o.work_dir, epoch - 1)?)
        } else {
            None
        };
        if missed_input.as_ref().is_some_and(Vec::is_empty) {
            eprintln!("epoch={epoch} skipped: previous epoch had no post-trigram top-1 misses");
            break;
        }
        let pass = if epoch > 0 {
            match run_epoch(
                o,
                &mut model,
                &index,
                &input_report,
                epoch,
                missed_input.as_deref(),
                &mut random,
                rate,
                &trainer,
                page_blocks.as_ref(),
                resume.as_ref().filter(|r| r.stop.epoch == epoch),
                &dev,
                &mut curve,
                hogwild.as_ref(),
            ) {
                Ok(pass) => pass,
                Err(error) => {
                    // Even a cache/IO failure must not discard already committed updates.
                    if let Some(hogwild) = &hogwild {
                        hogwild.snapshot_into(&mut model);
                    }
                    write_json(
                        &o.output.join("emergency-stop.json"),
                        &serde_json::json!({
                            "epoch":epoch,"error":error.to_string(),"checkpoint":"emergency-model.json",
                            "exact_resume":false
                        }),
                    )?;
                    output::save(&model, &o.output.join("emergency-model.json"))?;
                    return Err(error);
                }
            }
        } else {
            Pass::default()
        };
        let missed_path = if epoch > 0 && o.missed_feedback {
            Some(write_missed_records(
                &o.work_dir,
                epoch,
                &pass.missed_records,
                pass.stop.is_some(),
            )?)
        } else {
            None
        };
        if let Some(path) = &missed_path {
            eprintln!(
                "epoch={epoch} post_trigram_top1_hits={} evaluated={} missed={} missed_file={}",
                pass.post_trigram_top1_hits,
                pass.post_trigram_top1_evaluated,
                pass.missed_records.len(),
                path.display()
            );
        }
        let update_seconds = update_started.elapsed().as_secs_f64()
            + resume
                .as_ref()
                .filter(|r| r.stop.epoch == epoch)
                .map_or(0.0, |r| r.previous_seconds);
        if let Some(reason) = &pass.stop {
            // Save BEFORE evaluation/export: a difficult evaluation sentence could
            // itself fail, and must not destroy the successfully committed model.
            write_json(&o.output.join("training-stop.json"), reason)?;
            write_json(
                &o.output.join("resume-fingerprint.json"),
                &output::fingerprint(&model),
            )?;
            output::save(&model, &o.output.join("stopped-model.json"))?;
            write_json(
                &o.output.join("partial-training.json"),
                &serde_json::json!({
                    "epoch":epoch,"training_sentences_processed":pass.processed,
                    "training_batches":pass.batches,"training_update_seconds":update_seconds,
                    "checkpoint":"stopped-model.json",
                    "first_loss_sum":pass.first_loss,"rerank_loss_sum":pass.tri_loss,
                    "primary_objective":primary_objective(o),
                    "primary_objective_loss_sum":pass.first_loss,
                    "primary_nbest_margin_active_sentences":pass.primary_nbest_margin_active_sentences,
                    "primary_nbest_listwise_active_sentences":pass.primary_nbest_listwise_active_sentences,
                    "rerank_sentences":pass.tri_count,
                    "gold_missing_nbest":pass.missed,
                    "training_preupdate_top1_gold_hits":pass.top1_hits,
                    "training_preupdate_top1_evaluated_sentences":pass.top1_evaluated,
                    "training_post_trigram_top1_gold_hits":pass.post_trigram_top1_hits,
                    "training_post_trigram_top1_evaluated_sentences":pass.post_trigram_top1_evaluated,
                    "gold_augmented_sentences":pass.gold_augmented,
                    "missed_records_file":missed_path.as_ref().map(|path| path.to_string_lossy()),
                    "exact_resume":page_blocks.is_some() && !o.hogwild
                }),
            )?;
        }
        let train_metrics = evaluate(&model, &index, &train, o, &trainer)?;
        let validation_metrics = evaluate(&model, &index, &dev, o, &trainer)?;
        eprintln!(
            "epoch={epoch} train_nll={:.6} train_exact={:.3} dev_nll={:.6} dev_first={:.3} dev_rerank={:.3} dev_recall={:.3}",
            train_metrics.mean_first_nll,
            train_metrics.reranked_exact_accuracy,
            validation_metrics.mean_first_nll,
            validation_metrics.first_exact_accuracy,
            validation_metrics.reranked_exact_accuracy,
            validation_metrics.nbest_gold_recall
        );
        curve.record_if_new(
            &model,
            &index,
            &dev,
            o,
            &trainer,
            &pass,
            epoch,
            started,
            if epoch == 0 {
                "initial"
            } else if pass.stop.is_some() {
                "stopped"
            } else {
                "epoch-end"
            },
        )?;
        reports.push(EpochReport {
            epoch,
            learning_rate: rate,
            training_update_seconds: update_seconds,
            mean_update_first_nll: (pass.processed > 0
                && o.primary_nbest_margin.is_none()
                && !o.primary_nbest_listwise)
                .then(|| pass.first_loss / pass.processed as f64),
            mean_update_primary_nbest_margin_loss: (pass.processed > 0
                && o.primary_nbest_margin.is_some())
            .then(|| pass.first_loss / pass.processed as f64),
            primary_nbest_margin_active_sentences: pass.primary_nbest_margin_active_sentences,
            mean_update_primary_nbest_listwise_loss: (pass.processed > 0
                && o.primary_nbest_listwise)
                .then(|| pass.first_loss / pass.processed as f64),
            primary_nbest_listwise_active_sentences: pass.primary_nbest_listwise_active_sentences,
            mean_update_rerank_nll: (pass.tri_count > 0)
                .then(|| pass.tri_loss / pass.tri_count as f64),
            training_gold_missing_nbest: pass.missed,
            nbest_gold_hits: pass.processed - pass.missed,
            training_nbest_gold_hit_rate: (pass.processed > 0)
                .then(|| (pass.processed - pass.missed) as f64 / pass.processed as f64),
            training_preupdate_top1_gold_hits: pass.top1_hits,
            training_preupdate_top1_evaluated_sentences: pass.top1_evaluated,
            training_preupdate_top1_gold_hit_rate: (pass.top1_evaluated > 0)
                .then(|| pass.top1_hits as f64 / pass.top1_evaluated as f64),
            training_post_trigram_top1_gold_hits: pass.post_trigram_top1_hits,
            training_post_trigram_top1_evaluated_sentences: pass.post_trigram_top1_evaluated,
            training_post_trigram_top1_gold_hit_rate: (pass.post_trigram_top1_evaluated > 0).then(
                || pass.post_trigram_top1_hits as f64 / pass.post_trigram_top1_evaluated as f64,
            ),
            training_gold_augmented_sentences: pass.gold_augmented,
            feature_counts: feature_counts(&model),
            train: train_metrics,
            validation: validation_metrics,
            training_sentences_processed: pass.processed,
            parameter_table_bytes: model.first.values.projected_bytes(0)
                + model.third.values.projected_bytes(0)
                + model.third_pos.values.projected_bytes(0)
                + model.third_reading_length.values.projected_bytes(0),
            initial_bigram_table_bytes: model.initial_bigrams.projected_bytes(0),
            initial_trigram_table_bytes: model.initial_trigrams.projected_bytes(0),
            maximum_lattice_nodes: pass.max_nodes,
            maximum_lattice_edges: pass.max_edges,
            training_batches: pass.batches,
            maximum_batch_nodes: pass.max_batch_nodes,
            maximum_batch_edges: pass.max_batch_edges,
            page_block_order: pass.block_order,
            batch_timing: pass.timing,
            lattice_prepare_seconds: pass.lattice_prepare_seconds,
            cutoff_events: pass.cutoff_events,
        });
        if pass.stop.is_some() {
            stop = pass.stop;
            break;
        }
        if epoch > 0 && o.checkpoint_every > 0 && epoch % o.checkpoint_every == 0 {
            output::save(
                &model,
                &o.output.join(format!("checkpoint-epoch-{epoch:04}.json")),
            )?;
        }
    }
    write_json(&o.output.join("metrics.json"), &reports)?;
    output::export(&model, &o.output)?;
    let expected_fingerprint = output::fingerprint(&model);
    let initial_bigram_count = model.initial_bigrams.len();
    let initial_bigram_bytes = model.initial_bigrams.projected_bytes(0);
    let initial_trigram_count = model.initial_trigrams.len();
    let initial_trigram_bytes = model.initial_trigrams.projected_bytes(0);
    let reference = evaluate(&model, &index, &dev, o, &trainer)?;
    // Do not keep a second full trained model alongside the first for verification.
    drop(model);
    drop(index);
    let loaded = output::load(&o.output.join("model.json"))?;
    if output::fingerprint(&loaded) != expected_fingerprint {
        return Err("checkpoint round-trip mismatch".into());
    }
    let loaded_index = Index::new(&loaded)?;
    let roundtrip = evaluate(&loaded, &loaded_index, &dev, o, &trainer)?;
    if reference.mean_first_nll.to_bits() != roundtrip.mean_first_nll.to_bits()
        || reference.reranked_exact_accuracy.to_bits()
            != roundtrip.reranked_exact_accuracy.to_bits()
    {
        return Err("checkpoint scoring round-trip mismatch".into());
    }
    write_json(
        &o.output.join("verification.json"),
        &serde_json::json!({
            "checkpoint_roundtrip":"passed", "total_seconds":started.elapsed().as_secs_f64(),
            "supervision":"MeCab pseudo-gold; not human-verified", "source_bytes":fs::metadata(&o.input)?.len(),
            "initial_static_word_bigrams":initial_bigram_count,
            "initial_static_word_bigram_table_bytes":initial_bigram_bytes,
            "initial_bigram_mixture_delta":o.initial_bigram_delta,
            "initial_bigram_mixture_definition":"log(delta * P(POS_b|POS_a)P(word_b|POS_b) + (1-delta) * P(word_b|word_a)) minus baseline log probability",
            "initial_static_word_trigrams":initial_trigram_count,
            "initial_static_word_trigram_table_bytes":initial_trigram_bytes,
            "initial_trigram_empirical_weight":o.initial_trigram_weight,
            "initial_trigram_bigram_backoff_weight":1.0-o.initial_trigram_weight,
            "primary_objective":primary_objective(o),
            "primary_nbest_margin_log_score":o.primary_nbest_margin,
            "primary_nbest_listwise":o.primary_nbest_listwise,
            "primary_margin_active_sentences":reports.iter().map(|report| report.primary_nbest_margin_active_sentences).sum::<usize>(),
            "initial_trigram_mixture_definition":"log((1-lambda)*P(word_c|word_b) + lambda*P(word_c|word_a,word_b)) minus log P(word_c|word_b), for retained static trigrams only; unretained triples default to zero correction",
            "unseen_trigram_initial_correction":0.0,
            "unseen_learned_corrections_are_zero":true,
            "gold_injection":o.gold_path_augmentation,
            "gold_injection_scope":"training trigram candidate sets only; raw N-best metrics and inference are never augmented",
            "pos_counts_cutoff":null, "integer_costs":false,
            "execution_mode":if o.hogwild {"hogwild-scc-per-sentence"} else {"synchronous-minibatch"},
            "scope":if o.hogwild {
                "streaming prefix selection; bounded shuffle windows; per-sentence first/trigram gradients; immediate scc updates; no batch gradient mean"
            } else {
                "streaming prefix selection; bounded batch lattices; per-sentence first/trigram gradients from one immutable model state; one batch join then atomic model updates"
            },
            "validation_sentences":dev.len(),"training_probe_sentences":train.len(),
            "shuffle_buffer":o.shuffle_buffer,"max_parameter_mib":o.max_parameter_mib,
            "parameter_budget_enforced":!o.hogwild,
            "threads":trainer.threads(),"batch_size":o.batch_size,
            "gradient_reduction":if o.hogwild {
                "per-sentence immediate scc updates; learning_rate / batch_size; no batch mean"
            } else {
                "word-bigram and word-trigram gradients summed without batch divisor; all other feature families averaged over eligible sentences; both gradient families computed from one pre-update model and applied after batch join"
            },
            "l2":o.l2,
            "page_block_size":o.page_block_size,"page_blocks":page_blocks.as_ref().map_or(0, |b| b.blocks.len()),
            "status":stop.as_ref().map_or("completed", |s| s.kind.as_str()),"parameter_pruning":o.delta_cutoff > 0.0,
            "delta_cutoff_policy":if o.delta_cutoff > 0.0 {Some(cutoff_policy(o))} else {None},
            "delta_cutoff_events":reports.iter().map(|r| r.cutoff_events.len()).sum::<usize>(),
            "full_corpus":o.full_corpus,"source_training_pages":input_report.source_training_pages,
            "source_heldout_pages":input_report.source_heldout_pages,"heldout_sentences":input_report.heldout_sentences,
            "fingerprint":expected_fingerprint,"initial_fingerprint":initial_fingerprint
        }),
    )?;
    if let Some(reason) = stop {
        return Err(format!(
            "{}; committed model saved; see training-stop.json",
            reason.reason
        )
        .into());
    }
    println!(
        "completed output={} elapsed={:.1}s",
        o.output.display(),
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

fn main() {
    if let Err(e) = options().and_then(|o| run(&o)) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod top1_error_tests {
    use super::{Top1Outcome, classify_top1};

    #[test]
    fn groups_top1_misses_by_boundary_surface_and_identity() {
        assert_eq!(classify_top1(true, true, true), Top1Outcome::Correct);
        assert_eq!(
            classify_top1(false, false, false),
            Top1Outcome::BoundaryError
        );
        assert_eq!(
            classify_top1(false, true, false),
            Top1Outcome::SameBoundarySurfaceError
        );
        assert_eq!(
            classify_top1(false, true, true),
            Top1Outcome::SameBoundaryIdentityError
        );
    }
}
