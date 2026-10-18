//! Shared-model synchronous mini-batches on a persistent private thread pool.
//!
//! Workers only borrow one immutable pre-update model state. Ordered gradient
//! aggregation and both updates happen after a single parallel join. Scheduling
//! cannot change accumulation order.
#![allow(clippy::cast_precision_loss)]

use std::time::Instant;

use rayon::prelude::*;

use crate::corpus::{self, Index, Limits, Prepared, Sentence};
use crate::optimized::{self, Entries};
use crate::sparse::Key;
use crate::{
    Feature, Lattice, Model, ParameterLimit, Parameters, Path, PosTriple, ReadingLengthTriple,
    Result, first_order, nbest, rerank,
};

#[derive(Clone, Copy, Debug)]
pub struct BatchConfig {
    pub rate: f64,
    pub l2: f64,
    pub nbest: usize,
    pub astar_limit: usize,
    pub parameter_bytes: usize,
    /// Add the reference path to the trigram-training set when raw N-best misses it.
    /// Raw N-best hit metrics are still measured before this augmentation.
    pub gold_path_augmentation: bool,
    /// Replace full-lattice primary NLL with summed pairwise hinges between
    /// every non-gold raw N-best path and gold. A zero margin targets gold top-1
    /// within the candidate set. The margin is in additive log-score units.
    pub primary_nbest_margin: Option<f64>,
    /// Use a normalized listwise loss over raw N-best plus gold if absent.
    pub primary_nbest_listwise: bool,
}

#[derive(Debug)]
pub struct BatchReport {
    pub sentences: usize,
    pub first_loss_sum: f64,
    pub primary_margin_active_sentences: usize,
    pub primary_listwise_active_sentences: usize,
    pub rerank_loss_sum: f64,
    pub rerank_sentences: usize,
    pub first_path_gold_hits: usize,
    pub post_trigram_top1_gold_hits: usize,
    /// Input-batch indexes whose raw-N-best final top-1 is not gold.
    pub post_trigram_top1_missed_indices: Vec<usize>,
    pub nbest_gold_hits: usize,
    pub gold_augmented_sentences: usize,
    pub timing: BatchTiming,
}

/// The joint parallel wall time is exact. Per-stage worker times are summed CPU
/// work across sentences, so they intentionally need not sum to the wall time.
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct BatchTiming {
    pub joint_parallel_wall_seconds: f64,
    pub first_worker_seconds: f64,
    pub nbest_rerank_worker_seconds: f64,
    pub gradient_reduction_seconds: f64,
    pub reserve_and_update_seconds: f64,
}
impl BatchTiming {
    pub fn add(&mut self, other: &Self) {
        self.joint_parallel_wall_seconds += other.joint_parallel_wall_seconds;
        self.first_worker_seconds += other.first_worker_seconds;
        self.nbest_rerank_worker_seconds += other.nbest_rerank_worker_seconds;
        self.gradient_reduction_seconds += other.gradient_reduction_seconds;
        self.reserve_and_update_seconds += other.reserve_and_update_seconds;
    }
}

#[derive(Debug)]
pub struct Trainer {
    pool: rayon::ThreadPool,
    reference_kernels: bool,
}

fn results<T>(values: Vec<std::result::Result<T, String>>) -> Result<Vec<T>> {
    // Inspect failures in input order, not the worker completion order.
    values
        .into_iter()
        .map(|value| value.map_err(Into::into))
        .collect()
}

#[derive(Debug)]
#[allow(clippy::struct_excessive_bools)]
struct SentenceGradients {
    first_loss: f64,
    first: Entries<Feature>,
    primary_margin_active: bool,
    primary_listwise_active: bool,
    third_loss: Option<f64>,
    third_word: Entries<crate::Triple>,
    third_pos: Entries<PosTriple>,
    third_reading_length: Entries<ReadingLengthTriple>,
    first_path_gold_hit: bool,
    post_trigram_gold_hit: bool,
    raw_nbest_gold_hit: bool,
    gold_augmented: bool,
    first_worker_seconds: f64,
    nbest_rerank_worker_seconds: f64,
}

/// Undo only modified keys and the shared scale. A rare L2 renormalization needs
/// a full parameter backup; the ordinary path never copies the complete model.
#[derive(Debug)]
enum Undo<K: Key> {
    Keys {
        scale: f64,
        values: Vec<(K::Packed, Option<f64>)>,
    },
    Renormalized(Parameters<K>),
}
impl<K: Key> Undo<K> {
    fn capture(parameters: &Parameters<K>, gradient: &Entries<K>, c: BatchConfig) -> Self {
        if parameters.scale * (1.0 - c.rate * c.l2) < 1e-100 {
            Self::Renormalized(parameters.clone())
        } else {
            Self::Keys {
                scale: parameters.scale,
                values: gradient
                    .iter()
                    .filter(|(_, value)| value.abs() > 1e-12)
                    .map(|(key, _)| (key.pack(), parameters.values.get(&key).copied()))
                    .collect(),
            }
        }
    }
    fn new_keys(&self, parameters: &Parameters<K>, gradient: &Entries<K>) -> usize {
        match self {
            Self::Keys { values, .. } => values.iter().filter(|(_, old)| old.is_none()).count(),
            Self::Renormalized(_) => gradient
                .iter()
                .filter(|(k, v)| v.abs() > 1e-12 && !parameters.values.contains_key(k))
                .count(),
        }
    }
    fn restore(self, parameters: &mut Parameters<K>) {
        match self {
            Self::Renormalized(old) => *parameters = old,
            Self::Keys { scale, values } => {
                parameters.scale = scale;
                for (key, value) in values {
                    let key = K::unpack(key);
                    if let Some(value) = value {
                        parameters.values.insert(key, value);
                    } else {
                        parameters.values.remove(&key);
                    }
                }
            }
        }
    }
}

impl Trainer {
    pub fn new(threads: usize) -> Result<Self> {
        Self::with_reference_kernels(threads, false)
    }

    /// Reference kernels are retained for fixed-data performance/equality checks.
    pub fn with_reference_kernels(threads: usize, reference_kernels: bool) -> Result<Self> {
        if threads == 0 {
            return Err("thread count must be positive".into());
        }
        Ok(Self {
            pool: rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .thread_name(|n| format!("crf-worker-{n}"))
                .build()?,
            reference_kernels,
        })
    }

    #[must_use]
    pub fn threads(&self) -> usize {
        self.pool.current_num_threads()
    }

    /// Indexed parallel work, followed by input-ordered error/result collection.
    pub fn map<T: Sync, R: Send>(
        &self,
        items: &[T],
        compute: impl Fn(&T) -> Result<R> + Sync,
    ) -> Result<Vec<R>> {
        results(self.pool.install(|| {
            items
                .par_iter()
                .map(|item| compute(item).map_err(|e| e.to_string()))
                .collect()
        }))
    }

    pub fn prepare(
        &self,
        sentences: &[&Sentence],
        model: &Model,
        index: &Index,
        limits: Limits,
    ) -> Result<Vec<Prepared>> {
        results(self.pool.install(|| {
            sentences
                .par_iter()
                .enumerate()
                .map(|(i, sentence)| {
                    corpus::prepare_sentence(sentence, model, index, limits)
                        .map_err(|e| format!("batch sentence {i}, page {}: {e}", sentence.page_id))
                })
                .collect()
        }))
    }

    /// Compute both gradient families and evaluate each N-best set against one
    /// immutable pre-update model. A single parallel join precedes both updates.
    #[allow(clippy::too_many_lines)]
    fn batch_gradients(
        &self,
        model: &Model,
        graphs: &[&Lattice],
        c: BatchConfig,
    ) -> Result<Vec<SentenceGradients>> {
        results(self.pool.install(|| {
            graphs
                .par_iter()
                .enumerate()
                .map(|(i, graph)| {
                    let compute = || -> Result<SentenceGradients> {
                        let (
                            first_loss,
                            first_gradient,
                            primary_margin_active,
                            primary_listwise_active,
                            first_worker_seconds,
                            third_loss,
                            third_word,
                            third_pos,
                            third_reading_length,
                            first_path_gold_hit,
                            post_trigram_gold_hit,
                            raw_nbest_gold_hit,
                            gold_augmented,
                            nbest_rerank_worker_seconds,
                        ) = if self.reference_kernels {
                            let first_started = Instant::now();
                            let first = first_order(model, graph)?;
                            let first_worker_seconds = first_started.elapsed().as_secs_f64();
                            let nbest_started = Instant::now();
                            let mut paths = nbest(model, graph, c.nbest, c.astar_limit)?;
                            let first_path_gold_hit =
                                paths.first().is_some_and(|path| path.nodes == graph.gold);
                            let hit = paths.iter().any(|path| path.nodes == graph.gold);
                            let augmented = c.gold_path_augmentation && !hit;
                            let raw_path_count = paths.len();
                            if augmented {
                                paths.push(Path {
                                    nodes: graph.gold.clone(),
                                    score: model.path_score(graph, &graph.gold),
                                });
                            }
                            let third = rerank(model, graph, &paths)?;
                            let mut best_raw = 0;
                            for candidate in 1..raw_path_count {
                                if third.scores[candidate] > third.scores[best_raw] {
                                    best_raw = candidate;
                                }
                            }
                            let post_trigram_gold_hit = paths[best_raw].nodes == graph.gold;
                            (
                                first.loss,
                                Entries::from_reference(first.gradient),
                                false,
                                false,
                                first_worker_seconds,
                                third.loss,
                                Entries::from_reference(third.gradient),
                                Entries::from_reference(third.pos_gradient),
                                Entries::from_reference(third.reading_length_gradient),
                                first_path_gold_hit,
                                post_trigram_gold_hit,
                                hit,
                                augmented,
                                nbest_started.elapsed().as_secs_f64(),
                            )
                        } else if let Some(margin) = c.primary_nbest_margin {
                            let nbest_started = Instant::now();
                            let third = optimized::third(
                                model,
                                graph,
                                c.nbest,
                                c.astar_limit,
                                c.gold_path_augmentation,
                                Some(margin),
                                false,
                            )?;
                            (
                                third.primary_margin_loss.unwrap_or(0.0),
                                third.primary_margin_gradient,
                                third.primary_margin_active,
                                false,
                                0.0,
                                third.loss,
                                third.word_gradient,
                                third.pos_gradient,
                                third.reading_length_gradient,
                                third.first_path_gold_hit,
                                third.post_trigram_gold_hit,
                                third.raw_nbest_gold_hit,
                                third.gold_augmented,
                                nbest_started.elapsed().as_secs_f64(),
                            )
                        } else if c.primary_nbest_listwise {
                            let nbest_started = Instant::now();
                            let third = optimized::third(
                                model,
                                graph,
                                c.nbest,
                                c.astar_limit,
                                c.gold_path_augmentation,
                                None,
                                true,
                            )?;
                            (
                                third.primary_listwise_loss.unwrap_or(0.0),
                                third.primary_listwise_gradient,
                                false,
                                third.primary_listwise_active,
                                0.0,
                                third.loss,
                                third.word_gradient,
                                third.pos_gradient,
                                third.reading_length_gradient,
                                third.first_path_gold_hit,
                                third.post_trigram_gold_hit,
                                third.raw_nbest_gold_hit,
                                third.gold_augmented,
                                nbest_started.elapsed().as_secs_f64(),
                            )
                        } else {
                            let first_started = Instant::now();
                            let first = optimized::first(model, graph)?;
                            let first_worker_seconds = first_started.elapsed().as_secs_f64();
                            let nbest_started = Instant::now();
                            let third = optimized::third(
                                model,
                                graph,
                                c.nbest,
                                c.astar_limit,
                                c.gold_path_augmentation,
                                None,
                                false,
                            )?;
                            (
                                first.0,
                                first.1,
                                false,
                                false,
                                first_worker_seconds,
                                third.loss,
                                third.word_gradient,
                                third.pos_gradient,
                                third.reading_length_gradient,
                                third.first_path_gold_hit,
                                third.post_trigram_gold_hit,
                                third.raw_nbest_gold_hit,
                                third.gold_augmented,
                                nbest_started.elapsed().as_secs_f64(),
                            )
                        };
                        Ok(SentenceGradients {
                            first_loss,
                            first: first_gradient,
                            primary_margin_active,
                            primary_listwise_active,
                            third_loss,
                            third_word,
                            third_pos,
                            third_reading_length,
                            first_path_gold_hit,
                            post_trigram_gold_hit,
                            raw_nbest_gold_hit,
                            gold_augmented,
                            first_worker_seconds,
                            nbest_rerank_worker_seconds,
                        })
                    };
                    compute().map_err(|e| format!("batch sentence {i}, first/N-best: {e}"))
                })
                .collect()
        }))
    }

    /// Calculate first-order and trigram gradients from the same pre-update
    /// model, then apply updates after one parallel join. Word-bigram and
    /// word-trigram gradients are summed across the batch; other feature
    /// families are averaged. Gold augmentation affects only the trigram
    /// candidate set, never raw-hit accounting or inference/evaluation.
    #[allow(clippy::too_many_lines)]
    pub fn train_batch(
        &self,
        model: &mut Model,
        graphs: &[&Lattice],
        c: BatchConfig,
    ) -> Result<BatchReport> {
        if graphs.is_empty()
            || c.nbest == 0
            || c.astar_limit == 0
            || !c.rate.is_finite()
            || c.rate <= 0.0
            || !c.l2.is_finite()
            || c.l2 < 0.0
            || c.rate * c.l2 >= 1.0
            || c.primary_nbest_margin
                .is_some_and(|margin| !margin.is_finite() || margin < 0.0)
            || (c.primary_nbest_margin.is_some() && self.reference_kernels)
            || (c.primary_nbest_margin.is_some() && c.primary_nbest_listwise)
            || (c.primary_nbest_listwise && self.reference_kernels)
        {
            return Err("invalid batch/update settings".into());
        }
        let mut timing = BatchTiming::default();
        let phase = Instant::now();
        let sentence_results = self.batch_gradients(model, graphs, c)?;
        timing.joint_parallel_wall_seconds = phase.elapsed().as_secs_f64();
        let phase = Instant::now();
        let mut first_loss_sum = 0.0;
        let mut primary_margin_active_sentences = 0;
        let mut primary_listwise_active_sentences = 0;
        let mut rerank_loss_sum = 0.0;
        let mut rerank_sentences = 0;
        let mut first_path_gold_hits = 0;
        let mut post_trigram_top1_gold_hits = 0;
        let mut post_trigram_top1_missed_indices = Vec::new();
        let mut nbest_gold_hits = 0;
        let mut gold_augmented_sentences = 0;
        let mut first_results = Vec::with_capacity(sentence_results.len());
        let mut third_word_results = Vec::with_capacity(sentence_results.len());
        let mut third_pos_results = Vec::with_capacity(sentence_results.len());
        let mut third_reading_length_results = Vec::with_capacity(sentence_results.len());
        for (sentence_index, result) in sentence_results.into_iter().enumerate() {
            first_loss_sum += result.first_loss;
            primary_margin_active_sentences += usize::from(result.primary_margin_active);
            primary_listwise_active_sentences += usize::from(result.primary_listwise_active);
            timing.first_worker_seconds += result.first_worker_seconds;
            timing.nbest_rerank_worker_seconds += result.nbest_rerank_worker_seconds;
            first_path_gold_hits += usize::from(result.first_path_gold_hit);
            post_trigram_top1_gold_hits += usize::from(result.post_trigram_gold_hit);
            if !result.post_trigram_gold_hit {
                post_trigram_top1_missed_indices.push(sentence_index);
            }
            nbest_gold_hits += usize::from(result.raw_nbest_gold_hit);
            gold_augmented_sentences += usize::from(result.gold_augmented);
            first_results.push(result.first);
            if let Some(loss) = result.third_loss {
                rerank_loss_sum += loss;
                rerank_sentences += 1;
                third_word_results.push(result.third_word);
                third_pos_results.push(result.third_pos);
                third_reading_length_results.push(result.third_reading_length);
            }
        }
        let first = optimized::average_except(first_results, |feature| {
            matches!(
                feature,
                Feature::Bigram(_, _) | Feature::PosWord(_, _) | Feature::WordPos(_, _)
            )
        });
        let third_word = (rerank_sentences > 0).then(|| optimized::sum(third_word_results));
        let third_pos = (rerank_sentences > 0).then(|| optimized::average(third_pos_results));
        let third_reading_length =
            (rerank_sentences > 0).then(|| optimized::average(third_reading_length_results));
        timing.gradient_reduction_seconds = phase.elapsed().as_secs_f64();

        let phase = Instant::now();
        let first_undo = Undo::capture(&model.first, &first, c);
        let new_first = first_undo.new_keys(&model.first, &first);
        let mut third_word_undo = third_word
            .as_ref()
            .map(|gradient| Undo::capture(&model.third, gradient, c));
        let mut third_pos_undo = third_pos
            .as_ref()
            .map(|gradient| Undo::capture(&model.third_pos, gradient, c));
        let mut third_reading_length_undo = third_reading_length
            .as_ref()
            .map(|gradient| Undo::capture(&model.third_reading_length, gradient, c));
        let additional_first_bound = first
            .iter()
            .filter(|(_, value)| value.abs() > 1e-12)
            .count();
        let additional_third_bound = third_word.as_ref().map_or(0, |gradient| {
            gradient
                .iter()
                .filter(|(_, value)| value.abs() > 1e-12)
                .count()
        });
        let additional_pos_third_bound = third_pos.as_ref().map_or(0, |gradient| {
            gradient
                .iter()
                .filter(|(_, value)| value.abs() > 1e-12)
                .count()
        });
        let additional_reading_length_third_bound =
            third_reading_length.as_ref().map_or(0, |gradient| {
                gradient
                    .iter()
                    .filter(|(_, value)| value.abs() > 1e-12)
                    .count()
            });
        let needed_bytes = model
            .first
            .values
            .projected_bytes(new_first)
            .saturating_add(model.third.values.projected_bytes(
                third_word_undo.as_ref().map_or(0, |undo| {
                    undo.new_keys(&model.third, third_word.as_ref().unwrap())
                }),
            ))
            .saturating_add(model.third_pos.values.projected_bytes(
                third_pos_undo.as_ref().map_or(0, |undo| {
                    undo.new_keys(&model.third_pos, third_pos.as_ref().unwrap())
                }),
            ))
            .saturating_add(model.third_reading_length.values.projected_bytes(
                third_reading_length_undo.as_ref().map_or(0, |undo| {
                    undo.new_keys(
                        &model.third_reading_length,
                        third_reading_length.as_ref().unwrap(),
                    )
                }),
            ));
        if needed_bytes > c.parameter_bytes {
            return Err(Box::new(ParameterLimit {
                needed_bytes,
                limit_bytes: c.parameter_bytes,
                additional_first_bound,
                additional_third_bound,
                additional_pos_third_bound,
                additional_reading_length_third_bound,
                allocation_error: None,
            }));
        }
        if let Err(error) = model.first.values.reserve(new_first) {
            return Err(Box::new(ParameterLimit {
                needed_bytes,
                limit_bytes: c.parameter_bytes,
                additional_first_bound,
                additional_third_bound,
                additional_pos_third_bound,
                additional_reading_length_third_bound,
                allocation_error: Some(error.to_string()),
            }));
        }
        if let (Some(gradient), Some(undo)) = (&third_pos, &third_pos_undo) {
            if let Err(error) = model
                .third_pos
                .values
                .reserve(undo.new_keys(&model.third_pos, gradient))
            {
                return Err(Box::new(ParameterLimit {
                    needed_bytes,
                    limit_bytes: c.parameter_bytes,
                    additional_first_bound,
                    additional_third_bound,
                    additional_pos_third_bound,
                    additional_reading_length_third_bound,
                    allocation_error: Some(error.to_string()),
                }));
            }
        }
        if let (Some(gradient), Some(undo)) = (&third_word, &third_word_undo) {
            if let Err(error) = model
                .third
                .values
                .reserve(undo.new_keys(&model.third, gradient))
            {
                return Err(Box::new(ParameterLimit {
                    needed_bytes,
                    limit_bytes: c.parameter_bytes,
                    additional_first_bound,
                    additional_third_bound,
                    additional_pos_third_bound,
                    additional_reading_length_third_bound,
                    allocation_error: Some(error.to_string()),
                }));
            }
        }
        if let (Some(gradient), Some(undo)) = (&third_reading_length, &third_reading_length_undo) {
            if let Err(error) = model
                .third_reading_length
                .values
                .reserve(undo.new_keys(&model.third_reading_length, gradient))
            {
                return Err(Box::new(ParameterLimit {
                    needed_bytes,
                    limit_bytes: c.parameter_bytes,
                    additional_first_bound,
                    additional_third_bound,
                    additional_pos_third_bound,
                    additional_reading_length_third_bound,
                    allocation_error: Some(error.to_string()),
                }));
            }
        }
        if let Err(e) = model.first.update_entries(first.iter(), c.rate, c.l2) {
            first_undo.restore(&mut model.first);
            return Err(e);
        }
        if let (Some(gradient), Some(_)) = (&third_word, &third_word_undo) {
            if let Err(error) = model.third.update_entries(gradient.iter(), c.rate, c.l2) {
                third_word_undo.take().unwrap().restore(&mut model.third);
                first_undo.restore(&mut model.first);
                return Err(error);
            }
        }
        if let (Some(gradient), Some(_)) = (&third_pos, &third_pos_undo) {
            if let Err(e) = model
                .third_pos
                .update_entries(gradient.iter(), c.rate, c.l2)
            {
                third_pos_undo.take().unwrap().restore(&mut model.third_pos);
                if let Some(undo) = third_word_undo.take() {
                    undo.restore(&mut model.third);
                }
                first_undo.restore(&mut model.first);
                return Err(e);
            }
        }
        if let (Some(gradient), Some(_)) = (&third_reading_length, &third_reading_length_undo) {
            if let Err(e) = model
                .third_reading_length
                .update_entries(gradient.iter(), c.rate, c.l2)
            {
                third_reading_length_undo
                    .take()
                    .unwrap()
                    .restore(&mut model.third_reading_length);
                if let Some(undo) = third_pos_undo.take() {
                    undo.restore(&mut model.third_pos);
                }
                if let Some(undo) = third_word_undo.take() {
                    undo.restore(&mut model.third);
                }
                first_undo.restore(&mut model.first);
                return Err(e);
            }
        }
        timing.reserve_and_update_seconds = phase.elapsed().as_secs_f64();
        Ok(BatchReport {
            sentences: graphs.len(),
            first_loss_sum,
            primary_margin_active_sentences,
            primary_listwise_active_sentences,
            rerank_loss_sum,
            rerank_sentences,
            first_path_gold_hits,
            post_trigram_top1_gold_hits,
            post_trigram_top1_missed_indices,
            nbest_gold_hits,
            gold_augmented_sentences,
            timing,
        })
    }
}
