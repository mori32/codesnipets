use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use ai_engine_crf::blocks::{self, PageBlocks};
use ai_engine_crf::corpus::{self, Index, Limits, Sentence};
use ai_engine_crf::output;
use ai_engine_crf::{Model, nbest};
use serde::Deserialize;

const NBEST: usize = 128;
const EXPANSION_LIMIT: usize = 500_000;
const TRAIN_POOL_SIZE: usize = 2_000;
const TRAIN_STRIDE: usize = 2;
const TEST_STRIDE: usize = 10;
const SAMPLE_SEED: u64 = 20_261_007;

#[derive(Debug)]
struct Options {
    model: PathBuf,
    page_cache: PathBuf,
    train_features: PathBuf,
    test_sample: PathBuf,
    test_features: PathBuf,
}

#[derive(Debug, Deserialize)]
struct CandidateFeature {
    text: String,
    crf_score: f64,
}

#[derive(Debug, Deserialize)]
struct ScoredExample {
    page_id: u64,
    title: String,
    input: String,
    gold_text: String,
    gold_candidate: Option<usize>,
    raw_nbest_gold_cover: bool,
    candidates: Vec<CandidateFeature>,
}

#[derive(Default)]
struct Top1Counts {
    sentences: usize,
    covered: usize,
    hits: usize,
    hits_when_covered: usize,
}

impl Top1Counts {
    fn add(&mut self, covered: bool, hit: bool) {
        self.sentences += 1;
        self.covered += usize::from(covered);
        self.hits += usize::from(hit);
        self.hits_when_covered += usize::from(covered && hit);
    }

    fn json(&self) -> serde_json::Value {
        serde_json::json!({
            "sentences": self.sentences,
            "raw_nbest_gold_cover": self.covered,
            "raw_nbest_gold_cover_rate": ratio(self.covered, self.sentences),
            "top1_correct": self.hits,
            "top1_accuracy_all_sentences": ratio(self.hits, self.sentences),
            "top1_correct_when_gold_covered": self.hits_when_covered,
            "top1_accuracy_when_gold_covered": ratio(self.hits_when_covered, self.covered),
        })
    }
}

#[derive(Default)]
struct Results {
    raw_bigram: Top1Counts,
    word_trigram: Top1Counts,
    reading_length_trigram: Top1Counts,
    pos_trigram: Top1Counts,
    active_trigrams: Top1Counts,
    full_crf: Top1Counts,
    full_without_word_trigram: Top1Counts,
    raw_candidate_paths: usize,
    candidate_surface_mismatches: usize,
    crf_score_mismatches: usize,
    gold_cover_mismatches: usize,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("crf-trigram-top1 error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let options = parse_options()?;
    let model = output::load(&options.model)?;
    let index = Index::new(&model)?;
    let page_blocks = PageBlocks::load(&options.page_cache)?;
    let expected_train = read_jsonl::<ScoredExample>(&options.train_features)?;
    let expected_test = read_jsonl::<ScoredExample>(&options.test_features)?;
    let test_pool = read_json::<Vec<Sentence>>(&options.test_sample)?;

    let train_pool = select_cached_training(&page_blocks, TRAIN_POOL_SIZE, SAMPLE_SEED)?;
    let train = select_spaced(&train_pool, expected_train.len(), TRAIN_STRIDE, "training")?;
    let test = select_spaced(&test_pool, expected_test.len(), TEST_STRIDE, "test")?;
    validate_sentence_alignment(&train, &expected_train, "training")?;
    validate_sentence_alignment(&test, &expected_test, "test")?;

    let limits = Limits {
        words: 48,
        nodes: 20_000,
        edges: 1_000_000,
    };
    let mut train_results = Results::default();
    let mut test_results = Results::default();
    evaluate_split(
        &model,
        &index,
        &train,
        &expected_train,
        limits,
        true,
        &mut train_results,
    )?;
    evaluate_split(
        &model,
        &index,
        &test,
        &expected_test,
        limits,
        false,
        &mut test_results,
    )?;

    if train_results.candidate_surface_mismatches != 0
        || train_results.crf_score_mismatches != 0
        || train_results.gold_cover_mismatches != 0
        || test_results.candidate_surface_mismatches != 0
        || test_results.crf_score_mismatches != 0
        || test_results.gold_cover_mismatches != 0
    {
        return Err(format!(
            "reconstructed candidates do not match saved features: train_surface={} train_score={} train_cover={} test_surface={} test_score={} test_cover={}",
            train_results.candidate_surface_mismatches,
            train_results.crf_score_mismatches,
            train_results.gold_cover_mismatches,
            test_results.candidate_surface_mismatches,
            test_results.crf_score_mismatches,
            test_results.gold_cover_mismatches
        )
        .into());
    }

    let report = serde_json::json!({
        "model": options.model,
        "nbest": NBEST,
        "candidate_pool": "same raw first-order N-best paths as the saved LLM feature files; no gold injection in ranking",
        "trigram_score_definition": "rank by this component alone, using the frozen model's static plus learned feature weights",
        "pos_trigrams_enabled": model.pos_trigrams_enabled,
        "reading_length_trigrams_included_in_active_trigram": true,
        "raw_candidate_paths_scored": train_results.raw_candidate_paths + test_results.raw_candidate_paths,
        "validation": {
            "training_candidate_surface_mismatches": train_results.candidate_surface_mismatches,
            "training_crf_score_mismatches": train_results.crf_score_mismatches,
            "training_gold_cover_mismatches": train_results.gold_cover_mismatches,
            "test_candidate_surface_mismatches": test_results.candidate_surface_mismatches,
            "test_crf_score_mismatches": test_results.crf_score_mismatches,
            "test_gold_cover_mismatches": test_results.gold_cover_mismatches,
        },
        "training_1000": {
            "bigram_first_candidate": train_results.raw_bigram.json(),
            "word_trigram_only": train_results.word_trigram.json(),
            "reading_length_trigram_only": train_results.reading_length_trigram.json(),
            "pos_trigram_only": train_results.pos_trigram.json(),
            "active_trigram_sum": train_results.active_trigrams.json(),
            "full_crf_score": train_results.full_crf.json(),
            "full_crf_without_word_trigram": train_results.full_without_word_trigram.json(),
        },
        "test_500": {
            "bigram_first_candidate": test_results.raw_bigram.json(),
            "word_trigram_only": test_results.word_trigram.json(),
            "reading_length_trigram_only": test_results.reading_length_trigram.json(),
            "pos_trigram_only": test_results.pos_trigram.json(),
            "active_trigram_sum": test_results.active_trigrams.json(),
            "full_crf_score": test_results.full_crf.json(),
            "full_crf_without_word_trigram": test_results.full_without_word_trigram.json(),
        },
    });
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn evaluate_split(
    model: &Model,
    index: &Index,
    sentences: &[Sentence],
    expected: &[ScoredExample],
    limits: Limits,
    training: bool,
    results: &mut Results,
) -> Result<(), Box<dyn std::error::Error>> {
    if sentences.len() != expected.len() {
        return Err("sentence and saved feature counts differ".into());
    }
    for (sentence, example) in sentences.iter().zip(expected) {
        let prepared = corpus::prepare_sentence(sentence, model, index, limits)?;
        let paths = nbest(model, &prepared.graph, NBEST, EXPANSION_LIMIT)?;
        let raw_cover = paths
            .iter()
            .any(|path| path.nodes == prepared.graph.gold);
        results.gold_cover_mismatches += usize::from(raw_cover != example.raw_nbest_gold_cover);
        let raw_expected_count = example.candidates.len() - usize::from(training && !raw_cover);
        if paths.len() != raw_expected_count {
            return Err(format!(
                "{} candidate count changed for page {}: regenerated {} raw paths, saved features contain {}",
                if training { "training" } else { "test" },
                sentence.page_id,
                paths.len(),
                raw_expected_count
            )
            .into());
        }
        results.raw_candidate_paths += paths.len();

        let gold_index = paths
            .iter()
            .position(|path| path.nodes == prepared.graph.gold);
        let expected_gold_candidate = gold_index.or_else(|| training.then_some(paths.len()));
        results.gold_cover_mismatches +=
            usize::from(example.gold_candidate != expected_gold_candidate);
        for (path_index, path) in paths.iter().enumerate() {
            let score = model.path_score(&prepared.graph, &path.nodes)
                + model.trigram_score(&prepared.graph, &path.nodes);
            if (score - example.candidates[path_index].crf_score).abs() > 1e-9 {
                results.crf_score_mismatches += 1;
            }
            let surface = path_surface(model, &prepared.reading, &prepared.graph.nodes, &path.nodes)?;
            if surface != example.candidates[path_index].text {
                results.candidate_surface_mismatches += 1;
            }
        }

        let word_scores = paths
            .iter()
            .map(|path| {
                Model::trigram_features(&prepared.graph, &path.nodes)
                    .into_iter()
                    .map(|(key, count)| count * model.trigram_weight(key))
                    .sum::<f64>()
            })
            .collect::<Vec<_>>();
        let pos_scores = paths
            .iter()
            .map(|path| {
                Model::pos_trigram_features(&prepared.graph, &path.nodes, model)
                    .into_iter()
                    .map(|(key, count)| count * model.pos_trigram_weight(key))
                    .sum::<f64>()
            })
            .collect::<Vec<_>>();
        let reading_length_scores = paths
            .iter()
            .map(|path| {
                Model::reading_length_trigram_features(&prepared.graph, &path.nodes, model)
                    .into_iter()
                    .map(|(key, count)| count * model.reading_length_trigram_weight(key))
                    .sum::<f64>()
            })
            .collect::<Vec<_>>();
        let active_scores = paths
            .iter()
            .map(|path| model.trigram_score(&prepared.graph, &path.nodes))
            .collect::<Vec<_>>();
        let full_scores = paths
            .iter()
            .map(|path| model.path_score(&prepared.graph, &path.nodes) + model.trigram_score(&prepared.graph, &path.nodes))
            .collect::<Vec<_>>();
        let full_without_word_trigram_scores = full_scores
            .iter()
            .zip(&word_scores)
            .map(|(full, word)| full - word)
            .collect::<Vec<_>>();

        results.raw_bigram.add(raw_cover, gold_index == Some(0));
        results
            .word_trigram
            .add(raw_cover, Some(argmax(&word_scores)) == gold_index);
        results
            .reading_length_trigram
            .add(raw_cover, Some(argmax(&reading_length_scores)) == gold_index);
        results
            .pos_trigram
            .add(raw_cover, Some(argmax(&pos_scores)) == gold_index);
        results
            .active_trigrams
            .add(raw_cover, Some(argmax(&active_scores)) == gold_index);
        results
            .full_crf
            .add(raw_cover, Some(argmax(&full_scores)) == gold_index);
        results
            .full_without_word_trigram
            .add(raw_cover, Some(argmax(&full_without_word_trigram_scores)) == gold_index);
    }
    Ok(())
}

fn argmax(scores: &[f64]) -> usize {
    let mut best = 0;
    for index in 1..scores.len() {
        if scores[index] > scores[best] {
            best = index;
        }
    }
    best
}

fn path_surface(
    model: &Model,
    reading: &str,
    nodes: &[ai_engine_crf::Node],
    path: &[usize],
) -> Result<String, Box<dyn std::error::Error>> {
    let mut surface = String::new();
    for &node_index in path {
        let node = nodes.get(node_index).ok_or("invalid N-best node index")?;
        if node.word == ai_engine_crf::BOS || node.word == ai_engine_crf::EOS {
            continue;
        }
        let word = model.words.get(node.word as usize).ok_or("invalid word id")?;
        if word.model_only {
            surface.push_str(reading.get(node.start..node.end).ok_or("invalid reading span")?);
        } else {
            surface.push_str(&word.surface);
        }
    }
    Ok(surface)
}

fn select_cached_training(
    blocks: &PageBlocks,
    requested: usize,
    seed: u64,
) -> Result<Vec<Sentence>, Box<dyn std::error::Error>> {
    if blocks.training_sentences < requested {
        return Err("page cache is smaller than the original 2,000-sentence training sample".into());
    }
    let mut selected = Vec::with_capacity(requested);
    let total_blocks = blocks.blocks.len();
    for (block_index, block) in blocks.blocks.iter().enumerate() {
        let quota = ((block_index + 1) * requested / total_blocks)
            - (block_index * requested / total_blocks);
        if quota == 0 {
            continue;
        }
        let offset = (mix64(seed ^ block_index as u64) as usize) % (block.sentences - quota + 1);
        let end = offset + quota;
        let mut reader = blocks.open(block_index)?;
        let mut line = String::new();
        for row in 0..end {
            let cached = blocks::read_cached(&mut *reader, &mut line)?
                .ok_or("page cache ended before sampled training rows")?;
            if row >= offset {
                selected.push(cached.sentence);
            }
        }
    }
    if selected.len() != requested {
        return Err(format!("selected {} training rows, expected {requested}", selected.len()).into());
    }
    Ok(selected)
}

fn mix64(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn select_spaced<T: Clone>(
    input: &[T],
    count: usize,
    stride: usize,
    label: &str,
) -> Result<Vec<T>, Box<dyn std::error::Error>> {
    let end = count
        .checked_sub(1)
        .and_then(|last| last.checked_mul(stride))
        .and_then(|last| last.checked_add(1))
        .ok_or("selection range overflow")?;
    if input.len() < end {
        return Err(format!("{label} has {} rows, needs at least {end}", input.len()).into());
    }
    Ok((0..count).map(|index| input[index * stride].clone()).collect())
}

fn validate_sentence_alignment(
    sentences: &[Sentence],
    examples: &[ScoredExample],
    label: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if sentences.len() != examples.len() {
        return Err(format!("{label} row count mismatch").into());
    }
    for (sentence, example) in sentences.iter().zip(examples) {
        if sentence.page_id != example.page_id
            || sentence.title != example.title
            || sentence.sentence != example.input
            || example.gold_text != sentence.sentence
        {
            return Err(format!("{label} sample mismatch at page {}", example.page_id).into());
        }
    }
    Ok(())
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, Box<dyn std::error::Error>> {
    Ok(serde_json::from_reader(BufReader::new(File::open(path)?))?)
}

fn read_jsonl<T: for<'de> Deserialize<'de>>(
    path: &Path,
) -> Result<Vec<T>, Box<dyn std::error::Error>> {
    let mut result = Vec::new();
    for line in BufReader::new(File::open(path)?).lines() {
        let line = line?;
        if !line.trim().is_empty() {
            result.push(serde_json::from_str(&line)?);
        }
    }
    Ok(result)
}

fn parse_options() -> Result<Options, Box<dyn std::error::Error>> {
    let mut values = HashMap::new();
    let mut args = env::args().skip(1);
    while let Some(key) = args.next() {
        if !key.starts_with("--") {
            return Err(format!("unexpected argument: {key}").into());
        }
        let value = args.next().ok_or_else(|| format!("missing value for {key}"))?;
        if values.insert(key.clone(), value).is_some() {
            return Err(format!("duplicate argument: {key}").into());
        }
    }
    let take = |key: &str| -> Result<PathBuf, Box<dyn std::error::Error>> {
        values
            .get(key)
            .map(PathBuf::from)
            .ok_or_else(|| format!("missing required argument {key}").into())
    };
    Ok(Options {
        model: take("--model")?,
        page_cache: take("--page-cache")?,
        train_features: take("--train-features")?,
        test_sample: take("--test-sample")?,
        test_features: take("--test-features")?,
    })
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}
