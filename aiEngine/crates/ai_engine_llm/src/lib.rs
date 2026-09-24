//! Context scoring boundary and asynchronous orchestration.
//!
//! This crate deliberately contains no model runtime yet. `NoopScorer` and
//! `FixedScorer` make the integration deterministic while a llama.cpp,
//! Candle, or ONNX implementation can be added behind the same trait later.

use std::error::Error;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, mpsc};
use std::thread;

use ai_engine_core::{Candidate, KanaKanjiEngine, RerankedCandidate, rerank_candidates};

/// The equal-depth candidate prefix passed to an evaluator.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CandidatePrefix {
    pub text: String,
}

impl CandidatePrefix {
    #[must_use]
    pub fn new(text: impl Into<String>) -> Self {
        Self { text: text.into() }
    }
}

/// Errors returned by a context scorer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ScoreError {
    message: String,
}

impl ScoreError {
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for ScoreError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl Error for ScoreError {}

/// Model-independent evaluator interface.
pub trait ContextScorer: Send + Sync {
    /// Returns one higher-is-better score for every supplied candidate.
    ///
    /// # Errors
    ///
    /// Returns a model or input error when scoring cannot be completed.
    fn score_prefixes(
        &self,
        context: &str,
        candidates: &[CandidatePrefix],
    ) -> Result<Vec<f32>, ScoreError>;
}

/// Evaluator that leaves the classical ranking unchanged.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoopScorer;

impl ContextScorer for NoopScorer {
    fn score_prefixes(
        &self,
        _context: &str,
        candidates: &[CandidatePrefix],
    ) -> Result<Vec<f32>, ScoreError> {
        Ok(vec![0.0; candidates.len()])
    }
}

/// Deterministic evaluator for tests and wiring experiments.
#[derive(Clone, Debug)]
pub struct FixedScorer {
    scores: Vec<f32>,
}

impl FixedScorer {
    #[must_use]
    pub fn new(scores: Vec<f32>) -> Self {
        Self { scores }
    }
}

impl ContextScorer for FixedScorer {
    fn score_prefixes(
        &self,
        _context: &str,
        candidates: &[CandidatePrefix],
    ) -> Result<Vec<f32>, ScoreError> {
        if self.scores.len() != candidates.len() {
            return Err(ScoreError::new(format!(
                "fixed scorer has {} scores for {} candidates",
                self.scores.len(),
                candidates.len()
            )));
        }
        Ok(self.scores.clone())
    }
}

/// Scores core candidates synchronously and applies the combined ranking.
///
/// # Errors
///
/// Returns the evaluator error without modifying the input candidates.
pub fn score_and_rerank<S: ContextScorer>(
    scorer: &S,
    context: &str,
    candidates: &[Candidate],
    weight: f32,
) -> Result<Vec<RerankedCandidate>, ScoreError> {
    let prefixes: Vec<_> = candidates
        .iter()
        .map(|candidate| CandidatePrefix::new(candidate.text.clone()))
        .collect();
    let llm_scores = scorer.score_prefixes(context, &prefixes)?;
    Ok(rerank_candidates(candidates, &llm_scores, weight))
}

/// End-to-end synchronous P3 conversion pipeline.
///
/// The pipeline keeps classical candidate generation in `ai_engine_core` and
/// makes context scoring an optional, replaceable step. This is the API that
/// the orchestrator can use before the platform-specific C-ABI is introduced.
#[derive(Debug)]
pub struct ConversionPipeline<S> {
    engine: KanaKanjiEngine,
    scorer: S,
    weight: f32,
}

impl<S: ContextScorer> ConversionPipeline<S> {
    /// Creates a pipeline with an existing core engine and evaluator.
    #[must_use]
    pub fn new(engine: KanaKanjiEngine, scorer: S, weight: f32) -> Self {
        Self {
            engine,
            scorer,
            weight,
        }
    }

    /// Creates a pipeline using the built-in core dictionary.
    #[must_use]
    pub fn with_default_engine(scorer: S, weight: f32) -> Self {
        Self::new(KanaKanjiEngine::new(), scorer, weight)
    }

    /// Generates classical candidates and applies the context evaluator.
    ///
    /// # Errors
    ///
    /// Returns the evaluator error. Callers that must never interrupt input
    /// can use [`Self::convert_or_fallback`] instead.
    pub fn convert(
        &self,
        input: &str,
        context: &str,
        limit: usize,
    ) -> Result<Vec<RerankedCandidate>, ScoreError> {
        let candidates = self.engine.convert(input, limit);
        score_and_rerank(&self.scorer, context, &candidates, self.weight)
    }

    /// Generates candidates without invoking the evaluator.
    #[must_use]
    pub fn convert_classical(&self, input: &str, limit: usize) -> Vec<Candidate> {
        self.engine.convert(input, limit)
    }

    /// Converts with evaluator scoring and falls back to classical ranking.
    #[must_use]
    pub fn convert_or_fallback(
        &self,
        input: &str,
        context: &str,
        limit: usize,
    ) -> Vec<RerankedCandidate> {
        let candidates = self.engine.convert(input, limit);
        match score_and_rerank(&self.scorer, context, &candidates, self.weight) {
            Ok(ranked) => ranked,
            Err(_) => classical_reranked(candidates),
        }
    }

    /// Records a selected candidate in the core engine's user learning data.
    pub fn select(&mut self, candidate: &Candidate) {
        self.engine.select(candidate);
    }

    /// Returns the current core engine.
    #[must_use]
    pub fn engine(&self) -> &KanaKanjiEngine {
        &self.engine
    }

    /// Returns mutable access for dictionary loading and persistence.
    pub fn engine_mut(&mut self) -> &mut KanaKanjiEngine {
        &mut self.engine
    }

    /// Returns the configured evaluator.
    #[must_use]
    pub fn scorer(&self) -> &S {
        &self.scorer
    }

    /// Returns the classical/LLM score mixing weight.
    #[must_use]
    pub const fn weight(&self) -> f32 {
        self.weight
    }

    /// Updates the classical/LLM score mixing weight.
    pub fn set_weight(&mut self, weight: f32) {
        self.weight = weight;
    }
}

#[allow(clippy::cast_precision_loss)]
fn classical_reranked(candidates: Vec<Candidate>) -> Vec<RerankedCandidate> {
    candidates
        .into_iter()
        .map(|candidate| RerankedCandidate {
            combined_score: candidate.score as f32,
            candidate,
            llm_score: 0.0,
        })
        .collect()
}

/// Non-blocking candidate/evaluator pipeline.
///
/// Classical candidates are produced before `submit` returns. Only the
/// context-scoring step runs on a worker thread. A new submission replaces the
/// pending candidate snapshot, so callers cannot accidentally apply a result
/// to a newer input composition.
#[derive(Debug)]
pub struct AsyncConversionPipeline<S> {
    engine: KanaKanjiEngine,
    scorer: AsyncScorer<S>,
    weight: f32,
    pending: Option<(u64, Vec<Candidate>)>,
}

impl<S: ContextScorer + 'static> AsyncConversionPipeline<S> {
    /// Creates a non-blocking pipeline with an existing core engine.
    #[must_use]
    pub fn new(engine: KanaKanjiEngine, scorer: S, weight: f32) -> Self {
        Self {
            engine,
            scorer: AsyncScorer::new(scorer),
            weight,
            pending: None,
        }
    }

    /// Creates a non-blocking pipeline using the built-in core dictionary.
    #[must_use]
    pub fn with_default_engine(scorer: S, weight: f32) -> Self {
        Self::new(KanaKanjiEngine::new(), scorer, weight)
    }

    /// Starts context scoring and returns its generation.
    ///
    /// The returned generation identifies both the candidate snapshot and the
    /// evaluator response. Submitting again makes this request stale.
    pub fn submit(&mut self, input: &str, context: &str, limit: usize) -> u64 {
        let candidates = self.engine.convert(input, limit);
        let prefixes = candidates
            .iter()
            .map(|candidate| CandidatePrefix::new(candidate.text.clone()))
            .collect();
        let generation = self.scorer.submit(context.to_owned(), prefixes);
        self.pending = Some((generation, candidates));
        generation
    }

    /// Returns the latest response when it is ready; otherwise returns `None`.
    pub fn poll(&mut self, generation: u64) -> Option<Result<Vec<RerankedCandidate>, ScoreError>> {
        let response = self.scorer.take_generation(generation).into_iter().next()?;
        let (pending_generation, candidates) = self.pending.take()?;
        if pending_generation != generation {
            self.pending = Some((pending_generation, candidates));
            return None;
        }
        Some(
            response
                .scores
                .map(|scores| rerank_candidates(&candidates, &scores, self.weight)),
        )
    }

    /// Returns whether a generation still corresponds to the newest input.
    #[must_use]
    pub fn is_current(&self, generation: u64) -> bool {
        self.scorer.is_current(generation)
    }

    /// Generates an immediate classical snapshot for UI display.
    #[must_use]
    pub fn convert_classical(&self, input: &str, limit: usize) -> Vec<Candidate> {
        self.engine.convert(input, limit)
    }

    /// Records a selected candidate in the core engine's user learning data.
    pub fn select(&mut self, candidate: &Candidate) {
        self.engine.select(candidate);
    }

    /// Returns mutable access for dictionary loading and persistence.
    pub fn engine_mut(&mut self) -> &mut KanaKanjiEngine {
        &mut self.engine
    }
}

/// A completed asynchronous scoring response.
#[derive(Debug)]
pub struct ScoreResponse {
    pub generation: u64,
    pub scores: Result<Vec<f32>, ScoreError>,
}

/// Fire-and-forget evaluator. Newer generations supersede older responses.
/// Worker threads may finish an old request, but callers must accept only the
/// response whose generation equals `latest_generation()`.
#[derive(Debug)]
pub struct AsyncScorer<S> {
    scorer: Arc<S>,
    next_generation: AtomicU64,
    response_sender: mpsc::Sender<ScoreResponse>,
    response_receiver: mpsc::Receiver<ScoreResponse>,
}

impl<S: ContextScorer + 'static> AsyncScorer<S> {
    #[must_use]
    pub fn new(scorer: S) -> Self {
        let (response_sender, response_receiver) = mpsc::channel();
        Self {
            scorer: Arc::new(scorer),
            next_generation: AtomicU64::new(0),
            response_sender,
            response_receiver,
        }
    }

    /// Starts a request and returns its monotonically increasing generation.
    pub fn submit(&self, context: impl Into<String>, candidates: Vec<CandidatePrefix>) -> u64 {
        let generation = self.next_generation.fetch_add(1, Ordering::Relaxed) + 1;
        let scorer = Arc::clone(&self.scorer);
        let sender = self.response_sender.clone();
        let context = context.into();
        thread::spawn(move || {
            let llm_scores = scorer.score_prefixes(&context, &candidates);
            let _ = sender.send(ScoreResponse {
                generation,
                scores: llm_scores,
            });
        });
        generation
    }

    #[must_use]
    pub fn latest_generation(&self) -> u64 {
        self.next_generation.load(Ordering::Relaxed)
    }

    #[must_use]
    pub fn is_current(&self, generation: u64) -> bool {
        generation != 0 && generation == self.latest_generation()
    }

    /// Drains ready responses for one generation; stale responses are ignored.
    pub fn take_generation(&self, generation: u64) -> Vec<ScoreResponse> {
        self.response_receiver
            .try_iter()
            .filter(|response| response.generation == generation)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use ai_engine_core::KanaKanjiEngine;

    use super::{
        AsyncConversionPipeline, AsyncScorer, CandidatePrefix, ContextScorer, ConversionPipeline,
        FixedScorer, NoopScorer, score_and_rerank,
    };

    #[test]
    fn noop_scorer_preserves_classical_order() {
        let engine = KanaKanjiEngine::new();
        let candidates = engine.convert("にほんごへんかん", 3);
        let ranked = score_and_rerank(&NoopScorer, "", &candidates, 100.0).expect("score");
        assert_eq!(ranked[0].candidate.text, "日本語変換");
        assert!(ranked[0].llm_score.abs() < f32::EPSILON);
    }

    #[test]
    fn fixed_scorer_can_rerank_candidates() {
        let engine = KanaKanjiEngine::new();
        let candidates = engine.convert("とうきょう", 1);
        let scorer = FixedScorer::new(vec![-1.0, 3.0]);
        assert!(
            scorer
                .score_prefixes("", &[CandidatePrefix::new("東京")])
                .is_err()
        );
        let scorer = FixedScorer::new(vec![3.0]);
        let ranked = score_and_rerank(&scorer, "", &candidates, 1.0).expect("score");
        assert_eq!(ranked.len(), 1);
    }

    #[test]
    fn conversion_pipeline_runs_core_and_evaluator_together() {
        let pipeline = ConversionPipeline::with_default_engine(FixedScorer::new(vec![3.0]), 2.0);
        let ranked = pipeline
            .convert("にほんごへんかん", "前の文脈", 1)
            .expect("pipeline conversion");
        assert_eq!(ranked[0].candidate.text, "日本語変換");
        assert!(
            (ranked[0].combined_score - 89.0).abs() < f32::EPSILON,
            "unexpected combined score: {}",
            ranked[0].combined_score
        );
    }

    #[test]
    fn conversion_pipeline_falls_back_when_evaluator_fails() {
        let pipeline = ConversionPipeline::with_default_engine(FixedScorer::new(Vec::new()), 1.0);
        let ranked = pipeline.convert_or_fallback("とうきょう", "", 1);
        assert_eq!(ranked[0].candidate.text, "東京");
        assert!(ranked[0].llm_score.abs() < f32::EPSILON);
        assert!((ranked[0].combined_score - 35.0).abs() < f32::EPSILON);
    }

    #[test]
    fn async_conversion_pipeline_returns_only_the_latest_snapshot() {
        let mut pipeline = AsyncConversionPipeline::with_default_engine(NoopScorer, 1.0);
        let old = pipeline.submit("とうきょう", "old", 1);
        let latest = pipeline.submit("にほんごへんかん", "new", 1);
        assert!(!pipeline.is_current(old));
        assert!(pipeline.is_current(latest));

        std::thread::sleep(Duration::from_millis(10));
        let ranked = pipeline
            .poll(latest)
            .expect("latest response ready")
            .expect("latest response succeeds");
        assert_eq!(ranked[0].candidate.text, "日本語変換");
    }

    #[test]
    fn async_generation_marks_older_work_stale() {
        let scorer = AsyncScorer::new(NoopScorer);
        let first = scorer.submit("old", vec![CandidatePrefix::new("東京")]);
        let second = scorer.submit("new", vec![CandidatePrefix::new("東京")]);
        assert!(!scorer.is_current(first));
        assert!(scorer.is_current(second));
        std::thread::sleep(Duration::from_millis(10));
        let responses = scorer.take_generation(second);
        assert!(!responses.is_empty());
        assert!(
            responses
                .iter()
                .all(|response| response.generation == second)
        );
    }
}
