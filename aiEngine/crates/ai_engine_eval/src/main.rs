//! Dataset evaluator for classical and GGUF-backed candidate ranking.
//!
//! The evaluator deliberately runs the same dictionary and candidate pipeline
//! used by the CLI. It reports top-1 accuracy, MRR, and latency percentiles so
//! model changes can be compared without changing the conversion path.

use std::env;
use std::error::Error;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use ai_engine_core::{Candidate, KanaKanjiEngine};

#[cfg(feature = "llama-cpp")]
use ai_engine_core::RerankedCandidate;
#[cfg(feature = "llama-cpp")]
use ai_engine_llm::score_and_rerank;

#[cfg(feature = "llama-cpp")]
use ai_engine_llm::{ContextScorer, LlamaCppConfig, LlamaCppScorer};

#[derive(Clone, Debug, Eq, PartialEq)]
struct EvaluationCase {
    context: String,
    reading: String,
    expected: String,
}

#[cfg(feature = "llama-cpp")]
#[derive(Clone, Debug, Eq, PartialEq)]
struct SentenceAlternative {
    text: String,
    expected: bool,
}

#[cfg(feature = "llama-cpp")]
#[derive(Clone, Debug, Eq, PartialEq)]
struct SentenceCase {
    name: String,
    sentences: Vec<SentenceAlternative>,
}

#[cfg(feature = "llama-cpp")]
#[derive(Clone, Debug, Eq, PartialEq)]
struct LearningAlternative {
    text: String,
    expected: bool,
}

#[cfg(feature = "llama-cpp")]
#[derive(Clone, Debug, Eq, PartialEq)]
struct LearningCase {
    name: String,
    context: String,
    candidates: Vec<LearningAlternative>,
}

#[cfg(feature = "llama-cpp")]
#[derive(Clone, Debug, Eq, PartialEq)]
struct GenerationCase {
    prompt: String,
    max_tokens: u32,
}

#[derive(Clone, Debug)]
struct CaseResult {
    rank: Option<usize>,
    elapsed: Duration,
}

#[derive(Clone, Debug)]
struct EvaluationSummary {
    results: Vec<CaseResult>,
}

impl EvaluationSummary {
    #[allow(clippy::cast_precision_loss)]
    fn top1_accuracy(&self) -> f64 {
        if self.results.is_empty() {
            return 0.0;
        }
        let hits = self
            .results
            .iter()
            .filter(|result| result.rank == Some(1))
            .count();
        hits as f64 / self.results.len() as f64
    }

    #[allow(clippy::cast_precision_loss)]
    fn mean_reciprocal_rank(&self) -> f64 {
        if self.results.is_empty() {
            return 0.0;
        }
        self.results
            .iter()
            .map(|result| result.rank.map_or(0.0, |rank| 1.0 / rank as f64))
            .sum::<f64>()
            / self.results.len() as f64
    }

    fn latency_percentile(&self, percentile: usize) -> Duration {
        if self.results.is_empty() {
            return Duration::ZERO;
        }
        let mut latencies: Vec<_> = self.results.iter().map(|result| result.elapsed).collect();
        latencies.sort_unstable();
        let position = (latencies.len() * percentile)
            .div_ceil(100)
            .saturating_sub(1);
        latencies[position.min(latencies.len() - 1)]
    }

    fn mean_latency(&self) -> Duration {
        if self.results.is_empty() {
            return Duration::ZERO;
        }
        let total_nanos: u128 = self
            .results
            .iter()
            .map(|result| result.elapsed.as_nanos())
            .sum();
        let mean_nanos = total_nanos / self.results.len() as u128;
        Duration::from_nanos(u64::try_from(mean_nanos).unwrap_or(u64::MAX))
    }
}

#[derive(Clone, Debug)]
struct Arguments {
    cases: PathBuf,
    dictionaries: Vec<PathBuf>,
    static_dictionaries: Vec<PathBuf>,
    model: Option<PathBuf>,
    sentences: Vec<String>,
    sentence_cases: Option<PathBuf>,
    sentence_cases_open: Option<PathBuf>,
    learning_cases: Option<PathBuf>,
    generation_cases: Option<PathBuf>,
    japanese_only: bool,
    report: Option<PathBuf>,
    candidates: usize,
    score_weight: f32,
    context_length: u32,
    threads: i32,
    gpu_layers: i32,
}

impl Default for Arguments {
    fn default() -> Self {
        Self {
            cases: PathBuf::from("examples/context-cases.tsv"),
            dictionaries: Vec::new(),
            static_dictionaries: Vec::new(),
            model: None,
            sentences: Vec::new(),
            sentence_cases: None,
            sentence_cases_open: None,
            learning_cases: None,
            generation_cases: None,
            japanese_only: false,
            report: None,
            candidates: 5,
            score_weight: 1.0,
            context_length: 2048,
            threads: 0,
            gpu_layers: 0,
        }
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let arguments = parse_arguments()?;
    if let Some(path) = arguments.generation_cases.as_ref() {
        return run_generation_cases(&arguments, path);
    }
    if let Some(path) = arguments.sentence_cases_open.as_ref() {
        if arguments.sentence_cases.is_some() {
            return Err("--sentence-cases and --sentence-cases-open cannot be combined".into());
        }
        return run_sentence_cases_open(&arguments, path);
    }
    if let Some(path) = arguments.sentence_cases.as_ref() {
        return run_sentence_cases(&arguments, path);
    }
    if let Some(path) = arguments.learning_cases.as_ref() {
        return run_learning_cases(&arguments, path);
    }
    if !arguments.sentences.is_empty() {
        return run_sentence_comparison(&arguments);
    }
    let cases = load_cases(&arguments.cases)?;
    if cases.is_empty() {
        return Err("evaluation dataset is empty".into());
    }

    let mut engine = KanaKanjiEngine::new();
    for dictionary in &arguments.dictionaries {
        engine.load_dictionary(dictionary)?;
    }
    for dictionary in &arguments.static_dictionaries {
        engine.load_static_dictionary(dictionary)?;
    }

    let classical = evaluate_classical(&engine, &cases, arguments.candidates);
    print_summary("classical", &classical);

    #[cfg(feature = "llama-cpp")]
    let model_summary = if let Some(model_path) = arguments.model.as_ref() {
        let scorer = LlamaCppScorer::load(
            model_path,
            LlamaCppConfig {
                context_length: arguments.context_length,
                threads: arguments.threads,
                gpu_layers: arguments.gpu_layers,
                max_sequences: 8,
            },
        )?;
        let summary = evaluate_model(
            &engine,
            &cases,
            arguments.candidates,
            &scorer,
            arguments.score_weight,
        )?;
        print_summary("model", &summary);
        Some(summary)
    } else {
        None
    };

    #[cfg(not(feature = "llama-cpp"))]
    let model_summary = if arguments.model.is_some() {
        return Err("--model requires rebuilding ai_engine_eval with the llama-cpp feature".into());
    } else {
        None
    };

    print_case_results(&cases, &classical, model_summary.as_ref());
    if let Some(report) = arguments.report {
        write_report(&report, &cases, &classical, model_summary.as_ref())?;
        println!("report: {}", report.display());
    }
    Ok(())
}

fn parse_arguments() -> Result<Arguments, Box<dyn Error>> {
    let mut arguments = Arguments::default();
    let mut values = env::args().skip(1);
    while let Some(argument) = values.next() {
        match argument.as_str() {
            "--cases" => arguments.cases = next_path(&mut values, "--cases")?,
            "--dictionary" => arguments
                .dictionaries
                .push(next_path(&mut values, "--dictionary")?),
            "--static-dictionary" => arguments
                .static_dictionaries
                .push(next_path(&mut values, "--static-dictionary")?),
            "--model" => arguments.model = Some(next_path(&mut values, "--model")?),
            "--sentence" => arguments
                .sentences
                .push(next_value(&mut values, "--sentence")?),
            "--sentence-cases" => {
                arguments.sentence_cases = Some(next_path(&mut values, "--sentence-cases")?);
            }
            "--sentence-cases-open" => {
                arguments.sentence_cases_open =
                    Some(next_path(&mut values, "--sentence-cases-open")?);
            }
            "--learning-cases" => {
                arguments.learning_cases = Some(next_path(&mut values, "--learning-cases")?);
            }
            "--generation-cases" => {
                arguments.generation_cases = Some(next_path(&mut values, "--generation-cases")?);
            }
            "--japanese-only" => arguments.japanese_only = true,
            "--report" => arguments.report = Some(next_path(&mut values, "--report")?),
            "--candidates" => {
                arguments.candidates = next_value(&mut values, "--candidates")?.parse()?;
                if arguments.candidates == 0 {
                    return Err("--candidates must be greater than zero".into());
                }
            }
            "--score-weight" => {
                arguments.score_weight = next_value(&mut values, "--score-weight")?.parse()?;
            }
            "--context-length" => {
                arguments.context_length = next_value(&mut values, "--context-length")?.parse()?;
            }
            "--threads" => arguments.threads = next_value(&mut values, "--threads")?.parse()?,
            "--gpu-layers" => {
                arguments.gpu_layers = next_value(&mut values, "--gpu-layers")?.parse()?;
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            value if value.starts_with('-') => {
                return Err(format!("unknown option: {value}").into());
            }
            value => return Err(format!("unexpected positional argument: {value}").into()),
        }
    }
    if arguments.dictionaries.is_empty() {
        arguments
            .dictionaries
            .push(PathBuf::from("examples/context-words.tsv"));
    }
    Ok(arguments)
}

#[cfg(feature = "llama-cpp")]
fn run_sentence_comparison(arguments: &Arguments) -> Result<(), Box<dyn Error>> {
    if arguments.sentences.len() < 2 {
        return Err("--sentence must be supplied at least twice for comparison".into());
    }
    let model_path = arguments
        .model
        .as_ref()
        .ok_or("sentence comparison requires --model MODEL.gguf")?;
    let model_scorer = LlamaCppScorer::load(
        model_path,
        LlamaCppConfig {
            context_length: arguments.context_length,
            threads: arguments.threads,
            gpu_layers: arguments.gpu_layers,
            max_sequences: 8,
        },
    )?;
    let prefixes: Vec<_> = arguments
        .sentences
        .iter()
        .cloned()
        .map(ai_engine_llm::CandidatePrefix::new)
        .collect();
    let started = Instant::now();
    let scores = model_scorer
        .score_complete_sentences_to_eos(&prefixes)
        .map_err(|error| format!("sentence scoring failed: {error}"))?;
    let elapsed = started.elapsed();
    if scores.len() != arguments.sentences.len() {
        return Err("model returned a score count different from the sentence count".into());
    }
    let mut ranked: Vec<_> = arguments.sentences.iter().zip(scores).collect();
    ranked.sort_by(|left, right| right.1.total_cmp(&left.1));

    println!(
        "sentence-comparison: candidates={} elapsed_ms={:.3}",
        ranked.len(),
        duration_ms(elapsed),
    );
    for (rank, (sentence, score)) in ranked.iter().enumerate() {
        println!("  {}: score={score:.4} sentence={sentence:?}", rank + 1);
    }
    Ok(())
}

#[cfg(feature = "llama-cpp")]
fn run_sentence_cases(arguments: &Arguments, path: &Path) -> Result<(), Box<dyn Error>> {
    run_sentence_cases_mode(arguments, path, true)
}

#[cfg(feature = "llama-cpp")]
fn run_sentence_cases_open(arguments: &Arguments, path: &Path) -> Result<(), Box<dyn Error>> {
    run_sentence_cases_mode(arguments, path, false)
}

#[cfg(feature = "llama-cpp")]
#[allow(clippy::cast_precision_loss)]
fn run_sentence_cases_mode(
    arguments: &Arguments,
    path: &Path,
    include_eos: bool,
) -> Result<(), Box<dyn Error>> {
    let cases = load_sentence_cases(path)?;
    if cases.is_empty() {
        return Err("sentence case dataset is empty".into());
    }
    let model_path = arguments
        .model
        .as_ref()
        .ok_or("sentence case comparison requires --model MODEL.gguf")?;
    let model_scorer = LlamaCppScorer::load(
        model_path,
        LlamaCppConfig {
            context_length: arguments.context_length,
            threads: arguments.threads,
            gpu_layers: arguments.gpu_layers,
            max_sequences: 8,
        },
    )?;

    let started = Instant::now();
    let mut groups = 0_usize;
    let mut hits = 0_usize;
    let mut alternatives = 0_usize;
    for case in &cases {
        if case.sentences.len() < 2 {
            return Err(format!(
                "sentence case {:?} must contain at least two alternatives",
                case.name
            )
            .into());
        }
        let expected = case
            .sentences
            .iter()
            .position(|sentence| sentence.expected)
            .ok_or_else(|| format!("sentence case {:?} has no expected alternative", case.name))?;
        let prefixes: Vec<_> = case
            .sentences
            .iter()
            .map(|sentence| ai_engine_llm::CandidatePrefix::new(sentence.text.clone()))
            .collect();
        let scores = if include_eos {
            model_scorer.score_complete_sentences_to_eos(&prefixes)
        } else {
            model_scorer.score_prefixes("", &prefixes)
        }
        .map_err(|error| format!("sentence case {:?} failed: {error}", case.name))?;
        let predicted = scores
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1))
            .map(|(index, _)| index)
            .ok_or_else(|| format!("sentence case {:?} returned no scores", case.name))?;
        groups += 1;
        alternatives += case.sentences.len();
        if predicted == expected {
            hits += 1;
        }
        println!(
            "case={} expected={:?} predicted={:?} hit={}",
            case.name,
            case.sentences[expected].text,
            case.sentences[predicted].text,
            predicted == expected
        );
        let mut ranked: Vec<_> = case
            .sentences
            .iter()
            .zip(scores)
            .map(|(sentence, score)| (sentence.text.as_str(), score))
            .collect();
        ranked.sort_by(|left, right| right.1.total_cmp(&left.1));
        for (rank, (sentence, score)) in ranked.iter().enumerate() {
            println!("  {}: score={score:.4} sentence={sentence:?}", rank + 1);
        }
    }
    println!(
        "sentence-cases mode={} groups={groups} alternatives={alternatives} top1_accuracy={:.3} elapsed_ms={:.3}",
        if include_eos { "eos" } else { "open-kmin" },
        hits as f64 / groups as f64,
        duration_ms(started.elapsed()),
    );
    Ok(())
}

#[cfg(feature = "llama-cpp")]
#[allow(clippy::cast_precision_loss)]
fn run_learning_cases(arguments: &Arguments, path: &Path) -> Result<(), Box<dyn Error>> {
    let cases = load_learning_cases(path)?;
    if cases.is_empty() {
        return Err("learning case dataset is empty".into());
    }
    let model_path = arguments
        .model
        .as_ref()
        .ok_or("learning case evaluation requires --model MODEL.gguf")?;
    let model_scorer = LlamaCppScorer::load(
        model_path,
        LlamaCppConfig {
            context_length: arguments.context_length,
            threads: arguments.threads,
            gpu_layers: arguments.gpu_layers,
            max_sequences: 8,
        },
    )?;

    let started = Instant::now();
    let mut hits = 0_usize;
    let mut alternatives = 0_usize;
    for case in &cases {
        if case.candidates.len() < 2 {
            return Err(format!(
                "learning case {:?} must contain at least two candidates",
                case.name
            )
            .into());
        }
        let expected = case
            .candidates
            .iter()
            .position(|candidate| candidate.expected)
            .ok_or_else(|| format!("learning case {:?} has no expected candidate", case.name))?;
        let prefixes: Vec<_> = case
            .candidates
            .iter()
            .map(|candidate| ai_engine_llm::CandidatePrefix::new(candidate.text.clone()))
            .collect();
        let scores = model_scorer
            .score_prefixes(&case.context, &prefixes)
            .map_err(|error| format!("learning case {:?} failed: {error}", case.name))?;
        let predicted = scores
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1))
            .map(|(index, _)| index)
            .ok_or_else(|| format!("learning case {:?} returned no scores", case.name))?;
        let hit = predicted == expected;
        if hit {
            hits += 1;
        }
        alternatives += case.candidates.len();
        println!(
            "learning-case name={} expected={:?} predicted={:?} hit={} context={:?}",
            case.name,
            case.candidates[expected].text,
            case.candidates[predicted].text,
            hit,
            case.context
        );
        let mut ranked: Vec<_> = case
            .candidates
            .iter()
            .zip(scores)
            .map(|(candidate, score)| (candidate.text.as_str(), score))
            .collect();
        ranked.sort_by(|left, right| right.1.total_cmp(&left.1));
        for (rank, (candidate, score)) in ranked.iter().enumerate() {
            println!("  {}: score={score:.4} candidate={candidate:?}", rank + 1);
        }
    }
    println!(
        "learning-cases: groups={} alternatives={} top1_accuracy={:.3} elapsed_ms={:.3}",
        cases.len(),
        alternatives,
        hits as f64 / cases.len() as f64,
        duration_ms(started.elapsed()),
    );
    Ok(())
}

#[cfg(feature = "llama-cpp")]
fn run_generation_cases(arguments: &Arguments, path: &Path) -> Result<(), Box<dyn Error>> {
    let cases = load_generation_cases(path)?;
    if cases.is_empty() {
        return Err("generation case dataset is empty".into());
    }
    let model_path = arguments
        .model
        .as_ref()
        .ok_or("generation case evaluation requires --model MODEL.gguf")?;
    let model_scorer = LlamaCppScorer::load(
        model_path,
        LlamaCppConfig {
            context_length: arguments.context_length,
            threads: arguments.threads,
            gpu_layers: arguments.gpu_layers,
            max_sequences: 8,
        },
    )?;
    let started = Instant::now();
    for case in &cases {
        let generated = if arguments.japanese_only {
            model_scorer.generate_greedy_japanese_only(&case.prompt, case.max_tokens)
        } else {
            model_scorer.generate_greedy(&case.prompt, case.max_tokens)
        }
        .map_err(|error| format!("generation case {:?} failed: {error}", case.prompt))?;
        println!(
            "generation-case prompt={:?} max_tokens={} generated={:?}",
            case.prompt, case.max_tokens, generated
        );
    }
    println!(
        "generation-cases: cases={} elapsed_ms={:.3}",
        cases.len(),
        duration_ms(started.elapsed())
    );
    Ok(())
}

#[cfg(not(feature = "llama-cpp"))]
fn run_sentence_comparison(arguments: &Arguments) -> Result<(), Box<dyn Error>> {
    let _ = arguments;
    Err("--sentence requires rebuilding ai_engine_eval with the llama-cpp feature".into())
}

#[cfg(not(feature = "llama-cpp"))]
fn run_sentence_cases(arguments: &Arguments, path: &Path) -> Result<(), Box<dyn Error>> {
    let _ = (arguments, path);
    Err("--sentence-cases requires rebuilding ai_engine_eval with the llama-cpp feature".into())
}

#[cfg(not(feature = "llama-cpp"))]
fn run_sentence_cases_open(arguments: &Arguments, path: &Path) -> Result<(), Box<dyn Error>> {
    let _ = (arguments, path);
    Err(
        "--sentence-cases-open requires rebuilding ai_engine_eval with the llama-cpp feature"
            .into(),
    )
}

#[cfg(not(feature = "llama-cpp"))]
fn run_learning_cases(arguments: &Arguments, path: &Path) -> Result<(), Box<dyn Error>> {
    let _ = (arguments, path);
    Err("--learning-cases requires rebuilding ai_engine_eval with the llama-cpp feature".into())
}

#[cfg(not(feature = "llama-cpp"))]
fn run_generation_cases(arguments: &Arguments, path: &Path) -> Result<(), Box<dyn Error>> {
    let _ = (arguments, path);
    Err("--generation-cases requires rebuilding ai_engine_eval with the llama-cpp feature".into())
}

fn next_value<I>(values: &mut I, option: &str) -> Result<String, Box<dyn Error>>
where
    I: Iterator<Item = String>,
{
    values
        .next()
        .ok_or_else(|| format!("{option} requires a value").into())
}

fn next_path<I>(values: &mut I, option: &str) -> Result<PathBuf, Box<dyn Error>>
where
    I: Iterator<Item = String>,
{
    Ok(PathBuf::from(next_value(values, option)?))
}

fn load_cases(path: &Path) -> Result<Vec<EvaluationCase>, Box<dyn Error>> {
    let content = fs::read_to_string(path)?;
    content
        .lines()
        .enumerate()
        .filter(|(_, line)| {
            let trimmed = line.trim();
            !trimmed.is_empty() && !trimmed.starts_with('#')
        })
        .map(|(line_number, line)| {
            let fields: Vec<_> = line.split('\t').collect();
            if fields.len() != 3 || fields.iter().any(|field| field.is_empty()) {
                return Err(format!(
                    "{}:{} must contain context, reading, and expected surface separated by tabs",
                    path.display(),
                    line_number + 1
                )
                .into());
            }
            Ok(EvaluationCase {
                context: fields[0].to_owned(),
                reading: fields[1].to_owned(),
                expected: fields[2].to_owned(),
            })
        })
        .collect()
}

#[cfg(feature = "llama-cpp")]
fn load_sentence_cases(path: &Path) -> Result<Vec<SentenceCase>, Box<dyn Error>> {
    let content = fs::read_to_string(path)?;
    let mut cases = Vec::new();
    for (line_number, line) in content.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let fields: Vec<_> = line.split('\t').collect();
        if fields.len() != 3 || fields.iter().any(|field| field.is_empty()) {
            return Err(format!(
                "{}:{} must contain group, sentence, and expected flag separated by tabs",
                path.display(),
                line_number + 1
            )
            .into());
        }
        let expected = match fields[2] {
            "1" | "true" | "TRUE" => true,
            "0" | "false" | "FALSE" => false,
            value => {
                return Err(format!(
                    "{}:{} expected flag must be 0/1 or true/false, got {:?}",
                    path.display(),
                    line_number + 1,
                    value
                )
                .into());
            }
        };
        if let Some(case) = cases
            .iter_mut()
            .find(|case: &&mut SentenceCase| case.name == fields[0])
        {
            if expected && case.sentences.iter().any(|sentence| sentence.expected) {
                return Err(format!(
                    "{}:{} sentence case {:?} has more than one expected alternative",
                    path.display(),
                    line_number + 1,
                    fields[0]
                )
                .into());
            }
            case.sentences.push(SentenceAlternative {
                text: fields[1].to_owned(),
                expected,
            });
        } else {
            cases.push(SentenceCase {
                name: fields[0].to_owned(),
                sentences: vec![SentenceAlternative {
                    text: fields[1].to_owned(),
                    expected,
                }],
            });
        }
    }
    for case in &cases {
        if case.sentences.len() < 2 || !case.sentences.iter().any(|sentence| sentence.expected) {
            return Err(format!(
                "sentence case {:?} must have at least two alternatives and one expected alternative",
                case.name
            )
            .into());
        }
    }
    Ok(cases)
}

#[cfg(feature = "llama-cpp")]
fn load_learning_cases(path: &Path) -> Result<Vec<LearningCase>, Box<dyn Error>> {
    let content = fs::read_to_string(path)?;
    let mut cases = Vec::new();
    for (line_number, line) in content.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let fields: Vec<_> = line.split('\t').collect();
        if fields.len() != 4 || fields.iter().any(|field| field.is_empty()) {
            return Err(format!(
                "{}:{} must contain group, context, candidate, and expected flag separated by tabs",
                path.display(),
                line_number + 1
            )
            .into());
        }
        let expected = match fields[3] {
            "1" | "true" | "TRUE" => true,
            "0" | "false" | "FALSE" => false,
            value => {
                return Err(format!(
                    "{}:{} expected flag must be 0/1 or true/false, got {:?}",
                    path.display(),
                    line_number + 1,
                    value
                )
                .into());
            }
        };
        if let Some(case) = cases
            .iter_mut()
            .find(|case: &&mut LearningCase| case.name == fields[0])
        {
            let normalized_context = fields[1].trim_end();
            if case.context != normalized_context {
                return Err(format!(
                    "{}:{} learning case {:?} has inconsistent contexts",
                    path.display(),
                    line_number + 1,
                    fields[0]
                )
                .into());
            }
            if expected && case.candidates.iter().any(|candidate| candidate.expected) {
                return Err(format!(
                    "{}:{} learning case {:?} has more than one expected candidate",
                    path.display(),
                    line_number + 1,
                    fields[0]
                )
                .into());
            }
            case.candidates.push(LearningAlternative {
                text: fields[2].to_owned(),
                expected,
            });
        } else {
            cases.push(LearningCase {
                name: fields[0].to_owned(),
                context: fields[1].trim_end().to_owned(),
                candidates: vec![LearningAlternative {
                    text: fields[2].to_owned(),
                    expected,
                }],
            });
        }
    }
    for case in &cases {
        if case.candidates.len() < 2 || !case.candidates.iter().any(|candidate| candidate.expected)
        {
            return Err(format!(
                "learning case {:?} must have at least two candidates and one expected candidate",
                case.name
            )
            .into());
        }
    }
    Ok(cases)
}

#[cfg(feature = "llama-cpp")]
fn load_generation_cases(path: &Path) -> Result<Vec<GenerationCase>, Box<dyn Error>> {
    let content = fs::read_to_string(path)?;
    content
        .lines()
        .enumerate()
        .filter(|(_, line)| {
            let trimmed = line.trim();
            !trimmed.is_empty() && !trimmed.starts_with('#')
        })
        .map(|(line_number, line)| {
            let fields: Vec<_> = line.split('\t').collect();
            if fields.len() != 2 || fields[0].is_empty() || fields[1].is_empty() {
                return Err(format!(
                    "{}:{} must contain prompt and max_tokens separated by a tab",
                    path.display(),
                    line_number + 1
                )
                .into());
            }
            let max_tokens: u32 = fields[1].parse().map_err(|error| {
                format!(
                    "{}:{} max_tokens is invalid: {error}",
                    path.display(),
                    line_number + 1
                )
            })?;
            if max_tokens == 0 {
                return Err(format!(
                    "{}:{} max_tokens must be greater than zero",
                    path.display(),
                    line_number + 1
                )
                .into());
            }
            Ok(GenerationCase {
                prompt: fields[0].to_owned(),
                max_tokens,
            })
        })
        .collect()
}

fn evaluate_classical(
    engine: &KanaKanjiEngine,
    cases: &[EvaluationCase],
    limit: usize,
) -> EvaluationSummary {
    let results = cases
        .iter()
        .map(|case| {
            let started = Instant::now();
            let candidates = engine.convert(&case.reading, limit);
            CaseResult {
                rank: candidate_rank(&candidates, &case.expected),
                elapsed: started.elapsed(),
            }
        })
        .collect();
    EvaluationSummary { results }
}

#[cfg(feature = "llama-cpp")]
fn evaluate_model<S: ContextScorer>(
    engine: &KanaKanjiEngine,
    cases: &[EvaluationCase],
    limit: usize,
    scorer: &S,
    score_weight: f32,
) -> Result<EvaluationSummary, Box<dyn Error>> {
    let mut results = Vec::with_capacity(cases.len());
    for (index, case) in cases.iter().enumerate() {
        let candidates = engine.convert(&case.reading, limit);
        let started = Instant::now();
        let ranked = score_and_rerank(scorer, &case.context, &candidates, score_weight)
            .map_err(|error| format!("model scoring failed for case {}: {error}", index + 1))?;
        results.push(CaseResult {
            rank: reranked_rank(&ranked, &case.expected),
            elapsed: started.elapsed(),
        });
    }
    Ok(EvaluationSummary { results })
}

fn candidate_rank(candidates: &[Candidate], expected: &str) -> Option<usize> {
    candidates
        .iter()
        .position(|candidate| candidate.text == expected)
        .map(|index| index + 1)
}

#[cfg(feature = "llama-cpp")]
fn reranked_rank(candidates: &[RerankedCandidate], expected: &str) -> Option<usize> {
    candidates
        .iter()
        .position(|candidate| candidate.candidate.text == expected)
        .map(|index| index + 1)
}

fn print_summary(name: &str, summary: &EvaluationSummary) {
    println!(
        "{name}: cases={} top1={:.3} mrr={:.3} latency_ms(mean/p50/p95)={:.3}/{:.3}/{:.3}",
        summary.results.len(),
        summary.top1_accuracy(),
        summary.mean_reciprocal_rank(),
        duration_ms(summary.mean_latency()),
        duration_ms(summary.latency_percentile(50)),
        duration_ms(summary.latency_percentile(95)),
    );
}

fn print_case_results(
    cases: &[EvaluationCase],
    classical: &EvaluationSummary,
    model: Option<&EvaluationSummary>,
) {
    println!("cases:");
    for (index, case) in cases.iter().enumerate() {
        let classical_rank = display_rank(classical.results[index].rank);
        let model_rank = model.map_or_else(
            || "-".to_owned(),
            |summary| display_rank(summary.results[index].rank),
        );
        println!(
            "  {}: context={:?} reading={:?} expected={:?} classical_rank={} model_rank={}",
            index + 1,
            case.context,
            case.reading,
            case.expected,
            classical_rank,
            model_rank,
        );
    }
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn write_report(
    path: &Path,
    cases: &[EvaluationCase],
    classical: &EvaluationSummary,
    model: Option<&EvaluationSummary>,
) -> Result<(), Box<dyn Error>> {
    if classical.results.len() != cases.len()
        || model.is_some_and(|summary| summary.results.len() != cases.len())
    {
        return Err("evaluation result length does not match dataset".into());
    }
    let mut report = String::from(
        "index\tcontext\treading\texpected\tclassical_rank\tclassical_latency_ms\tmodel_rank\tmodel_latency_ms\n",
    );
    for (index, case) in cases.iter().enumerate() {
        let classical_result = &classical.results[index];
        let model_result = model.map(|summary| &summary.results[index]);
        let model_rank = model_result.map_or_else(String::new, |result| display_rank(result.rank));
        let model_latency = model_result.map_or_else(String::new, |result| {
            format!("{:.3}", duration_ms(result.elapsed))
        });
        writeln!(
            report,
            "{}\t{}\t{}\t{}\t{}\t{:.3}\t{}\t{}",
            index + 1,
            escape_tsv(&case.context),
            escape_tsv(&case.reading),
            escape_tsv(&case.expected),
            display_rank(classical_result.rank),
            duration_ms(classical_result.elapsed),
            model_rank,
            model_latency,
        )?;
    }
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, report)?;
    Ok(())
}

fn display_rank(rank: Option<usize>) -> String {
    rank.map_or_else(|| "miss".to_owned(), |rank| rank.to_string())
}

fn escape_tsv(value: &str) -> String {
    value.replace(['\t', '\r', '\n'], " ")
}

fn print_help() {
    println!(
        "ai-engine-eval [--cases PATH] [--dictionary PATH] [--static-dictionary PATH] [--model MODEL.gguf] [--sentence TEXT ... | --sentence-cases PATH | --sentence-cases-open PATH | --learning-cases PATH | --generation-cases PATH] [--japanese-only] [--report PATH] [--candidates N] [--score-weight N] [--context-length N] [--threads N] [--gpu-layers N]"
    );
    println!();
    println!("Without --model, evaluates the classical dictionary ranking.");
    println!("With --model, also evaluates the llama.cpp GGUF ranking.");
    println!(
        "Repeat --sentence to compare closed sentences/IME segments through EOS (no length penalty)."
    );
    println!("Use --sentence-cases PATH for grouped TSV comparisons with expected answers.");
    println!(
        "Use --sentence-cases-open PATH for open-end comparisons at common shortest token depth."
    );
    println!("Use --learning-cases PATH for long-term selection and input-history context tests.");
    println!("Use --generation-cases PATH for greedy next-word generation cases.");
    println!("Use --japanese-only with --generation-cases to mask non-Japanese token pieces.");
}

#[cfg(test)]
mod tests {
    use super::{CaseResult, EvaluationCase, EvaluationSummary, candidate_rank};
    use std::time::Duration;

    #[cfg(feature = "llama-cpp")]
    use super::load_generation_cases;
    #[cfg(feature = "llama-cpp")]
    use super::load_learning_cases;
    #[cfg(feature = "llama-cpp")]
    use super::load_sentence_cases;

    #[test]
    fn computes_top1_mrr_and_percentiles() {
        let summary = EvaluationSummary {
            results: vec![
                CaseResult {
                    rank: Some(1),
                    elapsed: Duration::from_millis(1),
                },
                CaseResult {
                    rank: Some(2),
                    elapsed: Duration::from_millis(2),
                },
                CaseResult {
                    rank: None,
                    elapsed: Duration::from_millis(3),
                },
            ],
        };
        assert!((summary.top1_accuracy() - 1.0 / 3.0).abs() < f64::EPSILON);
        assert!((summary.mean_reciprocal_rank() - 0.5).abs() < f64::EPSILON);
        assert_eq!(summary.latency_percentile(50), Duration::from_millis(2));
        assert_eq!(summary.latency_percentile(95), Duration::from_millis(3));
    }

    #[test]
    fn ranks_expected_surface_or_reports_a_miss() {
        let candidates = vec![
            ai_engine_core::Candidate {
                text: "first".to_owned(),
                reading: "r".to_owned(),
                score: 0,
                segments: Vec::new(),
            },
            ai_engine_core::Candidate {
                text: "second".to_owned(),
                reading: "r".to_owned(),
                score: 0,
                segments: Vec::new(),
            },
        ];
        assert_eq!(candidate_rank(&candidates, "second"), Some(2));
        assert_eq!(candidate_rank(&candidates, "missing"), None);
    }

    #[test]
    fn case_shape_is_explicit() {
        let case = EvaluationCase {
            context: "庭で犬を".to_owned(),
            reading: "かう".to_owned(),
            expected: "飼う".to_owned(),
        };
        assert_eq!(case.context, "庭で犬を");
    }

    #[cfg(feature = "llama-cpp")]
    #[test]
    fn loads_all_qiita_reranker_samples() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("examples")
            .join("qiita-reranker-cases.tsv");
        let cases = load_sentence_cases(&path).expect("Qiita fixture should parse");
        assert_eq!(cases.len(), 14);
        assert_eq!(
            cases.iter().map(|case| case.sentences.len()).sum::<usize>(),
            30
        );
        assert!(cases.iter().all(|case| {
            case.sentences
                .iter()
                .filter(|sentence| sentence.expected)
                .count()
                == 1
        }));
    }

    #[cfg(feature = "llama-cpp")]
    #[test]
    fn loads_next_word_generation_cases() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("examples")
            .join("next-word-cases.tsv");
        let cases = load_generation_cases(&path).expect("next-word fixture should parse");
        assert_eq!(cases.len(), 10);
        assert_eq!(cases[0].prompt, "先日は大変");
        assert!(cases.iter().all(|case| case.max_tokens == 4));
    }

    #[cfg(feature = "llama-cpp")]
    #[test]
    fn loads_learning_cases_with_context_overrides() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("examples")
            .join("learning-cases.tsv");
        let cases = load_learning_cases(&path).expect("learning fixture should parse");
        assert_eq!(cases.len(), 16);
        assert_eq!(
            cases
                .iter()
                .map(|case| case.candidates.len())
                .sum::<usize>(),
            32
        );
        assert!(cases.iter().all(|case| {
            case.candidates
                .iter()
                .filter(|candidate| candidate.expected)
                .count()
                == 1
        }));
        assert!(
            cases
                .iter()
                .any(|case| case.name.starts_with("override-long-"))
        );
        assert!(
            cases
                .iter()
                .any(|case| case.name.starts_with("override-history-"))
        );
    }
}
