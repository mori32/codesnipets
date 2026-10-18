//! Asynchronous sentence-at-a-time structured training.
//!
//! Each worker scores one private lattice from live shared parameters, creates
//! only that sentence's Viterbi/N-best paths, and atomically adds its correction
//! to the shared correction maps before taking another sentence. There is no batch-gradient mean or
//! model-wide training lock. A path search uses the edge scores it read while
//! filling its own edge buffer; different edges may therefore reflect different
//! moments, which is an intentional Hogwild approximation.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap, HashMap};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering as AtomicOrdering};
use std::time::Instant;

use rayon::prelude::*;
use scc::HashMap as ConcurrentMap;

use crate::corpus::{self, Index, Limits, Sentence};
use crate::sparse::Key;
use crate::{Edge, Feature, Lattice, Model, Path, Result, SparseMap, Triple, log_add};

type FirstMap = ConcurrentMap<u64, TimedWeight>;
type ThirdMap = ConcurrentMap<Triple, TimedWeight>;

#[derive(Clone, Copy, Debug)]
struct TimedWeight {
    value: f64,
    log_scale: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct Config {
    /// Per-sentence primary full-lattice CRF step.
    pub rate: f64,
    /// Per-sentence conditional N-best trigram step.
    pub trigram_rate: f64,
    /// Per-example L2 shrink, lazily materialized for sparse concurrent-map entries.
    pub l2: f64,
    pub nbest: usize,
    pub astar_limit: usize,
    /// Append gold only to the private trigram candidate set when raw N-best misses it.
    pub add_gold_for_trigram: bool,
}

#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct Report {
    pub sentences: usize,
    pub attempted_sentences: usize,
    pub error: Option<String>,
    pub viterbi_gold_hits: usize,
    pub raw_nbest_gold_hits: usize,
    pub gold_added_for_trigram: usize,
    pub rerank_gold_hits: usize,
    pub rerank_helped: usize,
    pub rerank_harmed: usize,
    pub primary_key_updates: usize,
    pub trigram_key_updates: usize,
    pub trigram_loss_sum: f64,
    pub primary_loss_sum: f64,
    pub worker_seconds: f64,
    pub wall_seconds: f64,
}

impl Report {
    /// Primary Viterbi top-1 gold accuracy over the sentences in this report.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn top1_gold_rate(&self) -> Option<f64> {
        (self.sentences > 0).then(|| self.viterbi_gold_hits as f64 / self.sentences as f64)
    }

    /// Raw N-best gold inclusion rate, measured before trigram reranking or
    /// optional training-only gold augmentation.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn nbest_gold_inclusion_rate(&self) -> Option<f64> {
        (self.sentences > 0).then(|| self.raw_nbest_gold_hits as f64 / self.sentences as f64)
    }
}

/// Concurrent sparse corrections, separate from the immutable statistical base.
///
/// The maps are the master data. `snapshot_into_model` is deliberately a
/// quiescent-boundary operation (e.g. end of a page block), not a training lock.
#[derive(Debug, Default)]
pub struct Master {
    first: FirstMap,
    third: ThirdMap,
    first_log_scale: AtomicU64,
    third_log_scale: AtomicU64,
}

#[derive(Debug)]
pub struct Trainer {
    pool: rayon::ThreadPool,
}

impl Trainer {
    pub fn new(threads: usize) -> Result<Self> {
        if threads == 0 {
            return Err("Hogwild thread count must be positive".into());
        }
        Ok(Self {
            pool: rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .thread_name(|index| format!("crf-hogwild-{index}"))
                .build()?,
        })
    }

    #[must_use]
    pub fn threads(&self) -> usize {
        self.pool.current_num_threads()
    }
}

impl Master {
    #[must_use]
    pub fn from_model(model: &Model) -> Self {
        let first = FirstMap::with_capacity(model.first.values.len());
        for (key, value) in model.first.values.iter() {
            first
                .insert_sync(
                    key.pack(),
                    TimedWeight {
                        value: value * model.first.scale,
                        log_scale: 0.0,
                    },
                )
                .expect("model feature keys must be unique");
        }

        let third = ThirdMap::with_capacity(model.third.values.len());
        for (key, value) in model.third.values.iter() {
            third
                .insert_sync(
                    key.pack(),
                    TimedWeight {
                        value: value * model.third.scale,
                        log_scale: 0.0,
                    },
                )
                .expect("model trigram keys must be unique");
        }
        Self {
            first,
            third,
            first_log_scale: AtomicU64::new(0.0_f64.to_bits()),
            third_log_scale: AtomicU64::new(0.0_f64.to_bits()),
        }
    }

    #[must_use]
    pub fn first_len(&self) -> usize {
        self.first.len()
    }

    #[must_use]
    pub fn third_len(&self) -> usize {
        self.third.len()
    }

    /// Copy current learned deltas back to the ordinary serializable model.
    /// Call only after all sentence workers for the desired checkpoint have joined.
    pub fn snapshot_into_model(&self, model: &mut Model) {
        let first_scale = load_log_scale(&self.first_log_scale);
        let mut first = SparseMap::default();
        self.first.iter_sync(|key, value| {
            first.insert(Feature::unpack(*key), materialize(*value, first_scale));
            true
        });

        let third_scale = load_log_scale(&self.third_log_scale);
        let mut third = SparseMap::default();
        self.third.iter_sync(|key, value| {
            third.insert(*key, materialize(*value, third_scale));
            true
        });
        model.first.values = first;
        model.first.scale = 1.0;
        model.third.values = third;
        model.third.scale = 1.0;
    }

    /// Train all supplied sentences in parallel. Updates are applied by each
    /// worker as soon as that sentence's private paths and corrections are ready.
    pub fn train(
        &self,
        trainer: &Trainer,
        model: &Model,
        graphs: &[&Lattice],
        c: Config,
    ) -> Result<Report> {
        if graphs.is_empty()
            || c.nbest == 0
            || c.astar_limit == 0
            || !c.rate.is_finite()
            || c.rate <= 0.0
            || !c.trigram_rate.is_finite()
            || c.trigram_rate <= 0.0
            || !c.l2.is_finite()
            || c.l2 < 0.0
            || c.rate * c.l2 >= 1.0
            || c.trigram_rate * c.l2 >= 1.0
        {
            return Err("invalid Hogwild update settings".into());
        }
        let started = Instant::now();
        let results = trainer.pool.install(|| {
            graphs
                .par_iter()
                .enumerate()
                .map(|(index, graph)| {
                    let worker_started = Instant::now();
                    let result = self
                        .train_sentence(model, graph, c)
                        .map_err(|error| error.to_string());
                    result.map(|mut report| {
                        report.worker_seconds = worker_started.elapsed().as_secs_f64();
                        (index, report)
                    })
                })
                .collect::<Vec<_>>()
        });

        let mut report = Report {
            attempted_sentences: graphs.len(),
            wall_seconds: started.elapsed().as_secs_f64(),
            ..Report::default()
        };
        for result in results {
            let (_, sentence) = result.map_err(std::io::Error::other)?;
            add_sentence_report(&mut report, &sentence);
        }
        Ok(report)
    }

    /// Stream a sentence slice through a fixed set of workers. Each worker
    /// claims one sentence, builds only that sentence's lattice, updates the
    /// shared master immediately, drops the lattice, then claims another.
    /// The only join is after the supplied progress/checkpoint window is done.
    pub fn train_sentences(
        &self,
        trainer: &Trainer,
        model: &Model,
        sentences: &[&Sentence],
        index: &Index,
        limits: Limits,
        c: Config,
    ) -> Result<Report> {
        validate_config(c)?;
        if sentences.is_empty() {
            return Err("Hogwild sentence window must not be empty".into());
        }
        let started = Instant::now();
        let next = AtomicUsize::new(0);
        let failed = AtomicBool::new(false);
        let error = Mutex::new(None::<String>);
        let worker_reports = trainer.pool.install(|| {
            (0..trainer.threads())
                .into_par_iter()
                .map(|_| {
                    let worker_started = Instant::now();
                    let mut report = Report::default();
                    loop {
                        if failed.load(AtomicOrdering::Acquire) {
                            break;
                        }
                        let sentence_index = next.fetch_add(1, AtomicOrdering::Relaxed);
                        let Some(sentence) = sentences.get(sentence_index) else {
                            break;
                        };
                        let result = corpus::prepare_sentence(sentence, model, index, limits)
                            .and_then(|prepared| self.train_sentence(model, &prepared.graph, c));
                        match result {
                            Ok(sentence_report) => {
                                add_sentence_report(&mut report, &sentence_report);
                            }
                            Err(cause) => {
                                failed.store(true, AtomicOrdering::Release);
                                let mut first_error = error.lock().expect("Hogwild error mutex");
                                first_error.get_or_insert_with(|| {
                                    format!("sentence index {sentence_index}: {cause}")
                                });
                                break;
                            }
                        }
                    }
                    report.worker_seconds = worker_started.elapsed().as_secs_f64();
                    report
                })
                .collect::<Vec<_>>()
        });
        let mut report = Report {
            attempted_sentences: sentences.len(),
            wall_seconds: started.elapsed().as_secs_f64(),
            ..Report::default()
        };
        for worker in worker_reports {
            add_report(&mut report, &worker);
        }
        report.error = error.into_inner().expect("Hogwild error mutex");
        Ok(report)
    }

    fn train_sentence(&self, model: &Model, graph: &Lattice, c: Config) -> Result<Report> {
        let view = LiveScores {
            model,
            first: &self.first,
            first_log_scale: &self.first_log_scale,
            third: &self.third,
            third_log_scale: &self.third_log_scale,
        };

        // The edge vector is private to this sentence. It is filled by reading
        // live shared-map values; workers may update those values during this loop.
        let edges = graph
            .edges
            .iter()
            .map(|&edge| view.edge_score(graph, edge))
            .collect::<Vec<_>>();
        let paths = nbest_from_edges(graph, &edges, c.nbest, c.astar_limit)?;
        let top = paths.first().ok_or("A* returned no path")?;
        let viterbi_hit = top.nodes == graph.gold;
        let raw_hit = paths.iter().any(|path| path.nodes == graph.gold);
        let augmented = c.add_gold_for_trigram && !raw_hit;
        let mut candidates = paths;
        if augmented {
            candidates.push(Path {
                nodes: graph.gold.clone(),
                score: path_score_from_edges(graph, &edges, &graph.gold)?,
            });
        }

        // Preserve the original primary objective: exact forward-backward over
        // every edge of the original lattice, independent of the truncated N-best set.
        let (primary_loss, first_gradient) = primary_correction(model, graph, &edges)?;

        // Trigram loss is conditional on this sentence's N-best set. It changes
        // only ordering within that set; raw N-best recall is measured separately.
        let (tri_loss, tri_delta, reranked_hit) = trigram_correction(&view, graph, &candidates)?;
        let rerank_hit = reranked_hit;
        let trigram_has_gold = candidates.iter().any(|path| path.nodes == graph.gold);

        // Finish all reads/path calculations before this sentence writes either
        // table. Global L2 remains lazy: increment the family-wide scale now,
        // then materialize that scale only on touched keys and snapshots.
        // Other workers remain free to write throughout.
        let first_scale = advance_log_scale(&self.first_log_scale, 1.0 - c.rate * c.l2);
        let third_scale = if trigram_has_gold {
            advance_log_scale(&self.third_log_scale, 1.0 - c.trigram_rate * c.l2)
        } else {
            load_log_scale(&self.third_log_scale)
        };
        for (key, gradient) in &first_gradient {
            let delta = -c.rate * gradient;
            if delta != 0.0 {
                update_timed(&self.first, key.pack(), delta, first_scale);
            }
        }
        for (key, gradient) in &tri_delta {
            let delta = -c.trigram_rate * gradient;
            if delta != 0.0 {
                update_timed(&self.third, *key, delta, third_scale);
            }
        }

        Ok(Report {
            sentences: 1,
            viterbi_gold_hits: usize::from(viterbi_hit),
            raw_nbest_gold_hits: usize::from(raw_hit),
            gold_added_for_trigram: usize::from(augmented),
            rerank_gold_hits: usize::from(rerank_hit),
            rerank_helped: usize::from(!viterbi_hit && rerank_hit),
            rerank_harmed: usize::from(viterbi_hit && !rerank_hit),
            primary_key_updates: first_gradient.len(),
            trigram_key_updates: tri_delta.len(),
            trigram_loss_sum: tri_loss,
            primary_loss_sum: primary_loss,
            ..Report::default()
        })
    }
}

fn validate_config(c: Config) -> Result<()> {
    if c.nbest == 0
        || c.astar_limit == 0
        || !c.rate.is_finite()
        || c.rate <= 0.0
        || !c.trigram_rate.is_finite()
        || c.trigram_rate <= 0.0
        || !c.l2.is_finite()
        || c.l2 < 0.0
        || c.rate * c.l2 >= 1.0
        || c.trigram_rate * c.l2 >= 1.0
    {
        return Err("invalid Hogwild update settings".into());
    }
    Ok(())
}

fn add_sentence_report(target: &mut Report, sentence: &Report) {
    target.sentences += 1;
    target.viterbi_gold_hits += sentence.viterbi_gold_hits;
    target.raw_nbest_gold_hits += sentence.raw_nbest_gold_hits;
    target.gold_added_for_trigram += sentence.gold_added_for_trigram;
    target.rerank_gold_hits += sentence.rerank_gold_hits;
    target.rerank_helped += sentence.rerank_helped;
    target.rerank_harmed += sentence.rerank_harmed;
    target.primary_key_updates += sentence.primary_key_updates;
    target.trigram_key_updates += sentence.trigram_key_updates;
    target.trigram_loss_sum += sentence.trigram_loss_sum;
    target.primary_loss_sum += sentence.primary_loss_sum;
    target.worker_seconds += sentence.worker_seconds;
}

fn add_report(target: &mut Report, worker: &Report) {
    target.sentences += worker.sentences;
    target.viterbi_gold_hits += worker.viterbi_gold_hits;
    target.raw_nbest_gold_hits += worker.raw_nbest_gold_hits;
    target.gold_added_for_trigram += worker.gold_added_for_trigram;
    target.rerank_gold_hits += worker.rerank_gold_hits;
    target.rerank_helped += worker.rerank_helped;
    target.rerank_harmed += worker.rerank_harmed;
    target.primary_key_updates += worker.primary_key_updates;
    target.trigram_key_updates += worker.trigram_key_updates;
    target.trigram_loss_sum += worker.trigram_loss_sum;
    target.primary_loss_sum += worker.primary_loss_sum;
    target.worker_seconds += worker.worker_seconds;
}

fn load_log_scale(scale: &AtomicU64) -> f64 {
    f64::from_bits(scale.load(AtomicOrdering::Acquire))
}

fn advance_log_scale(scale: &AtomicU64, decay: f64) -> f64 {
    let delta = decay.ln();
    let mut old = scale.load(AtomicOrdering::Acquire);
    loop {
        let next = (f64::from_bits(old) + delta).to_bits();
        match scale.compare_exchange_weak(
            old,
            next,
            AtomicOrdering::AcqRel,
            AtomicOrdering::Acquire,
        ) {
            Ok(_) => return f64::from_bits(next),
            Err(observed) => old = observed,
        }
    }
}

fn materialize(weight: TimedWeight, current_log_scale: f64) -> f64 {
    weight.value * (current_log_scale - weight.log_scale).exp()
}

fn update_timed<K>(map: &ConcurrentMap<K, TimedWeight>, key: K, delta: f64, target_log_scale: f64)
where
    K: std::hash::Hash + Eq + Copy,
{
    let mut value_to_insert = TimedWeight {
        value: delta,
        log_scale: target_log_scale,
    };
    loop {
        if map
            .update_sync(&key, |_, current| {
                if target_log_scale <= current.log_scale {
                    *current = TimedWeight {
                        value: materialize(*current, target_log_scale) + delta,
                        log_scale: target_log_scale,
                    };
                } else {
                    // This worker's update was computed before a concurrent newer
                    // update to the same key. Carry its correction through the
                    // intervening global shrink without overwriting that update.
                    current.value += delta * (current.log_scale - target_log_scale).exp();
                }
            })
            .is_some()
        {
            return;
        }
        match map.insert_sync(key, value_to_insert) {
            Ok(()) => return,
            Err((_, value)) => value_to_insert = value,
        }
    }
}

struct LiveScores<'a> {
    model: &'a Model,
    first: &'a FirstMap,
    first_log_scale: &'a AtomicU64,
    third: &'a ThirdMap,
    third_log_scale: &'a AtomicU64,
}

impl LiveScores<'_> {
    fn weight(&self, key: Feature) -> f64 {
        if !self.model.cross_bigrams_enabled
            && matches!(key, Feature::PosWord(..) | Feature::WordPos(..))
        {
            return 0.0;
        }
        let dynamic = self.first.read_sync(&key.pack(), |_, value| *value);
        self.model.initial(key)
            + dynamic.map_or(0.0, |value| {
                materialize(value, load_log_scale(self.first_log_scale))
            })
    }

    fn trigram_weight(&self, key: Triple) -> f64 {
        let initial = self
            .model
            .initial_trigrams
            .get(&key)
            .copied()
            .unwrap_or(0.0);
        initial
            + self
                .third
                .read_sync(&key, |_, value| *value)
                .map_or(0.0, |value| {
                    materialize(value, load_log_scale(self.third_log_scale))
                })
    }

    fn features(&self, graph: &Lattice, edge: Edge) -> [Feature; 7] {
        let a = &graph.nodes[edge.from];
        let b = &graph.nodes[edge.to];
        let source_pos = self.model.words[a.word as usize].pos;
        let target_pos = self.model.words[b.word as usize].pos;
        [
            Feature::Word(b.word),
            Feature::Pos(source_pos, target_pos),
            Feature::Bigram(a.word, b.word),
            Feature::PosWord(self.model.pos_token_ids[source_pos as usize], b.word),
            Feature::WordPos(a.word, self.model.pos_token_ids[target_pos as usize]),
            Feature::Script(b.script),
            Feature::ScriptBigram(a.script, b.script),
        ]
    }

    fn edge_score(&self, graph: &Lattice, edge: Edge) -> f64 {
        self.features(graph, edge)
            .into_iter()
            .map(|feature| self.weight(feature))
            .sum()
    }

    fn trigram_score(&self, graph: &Lattice, path: &[usize]) -> f64 {
        path.windows(3)
            .map(|p| {
                self.trigram_weight([
                    graph.nodes[p[0]].word,
                    graph.nodes[p[1]].word,
                    graph.nodes[p[2]].word,
                ])
            })
            .sum()
    }
}

#[derive(Clone, Debug)]
struct Agenda {
    bound: f64,
    path: Path,
}
impl PartialEq for Agenda {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Agenda {}
impl PartialOrd for Agenda {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Agenda {
    fn cmp(&self, other: &Self) -> Ordering {
        self.bound
            .total_cmp(&other.bound)
            .then_with(|| other.path.nodes.cmp(&self.path.nodes))
    }
}

fn nbest_from_edges(
    graph: &Lattice,
    edge_scores: &[f64],
    count: usize,
    expansion_limit: usize,
) -> Result<Vec<Path>> {
    let last = graph.nodes.len() - 1;
    let mut suffix = vec![f64::NEG_INFINITY; graph.nodes.len()];
    suffix[last] = 0.0;
    for node in (0..last).rev() {
        for &edge in &graph.outgoing[node] {
            let target = graph.edges[edge].to;
            suffix[node] = suffix[node].max(edge_scores[edge] + suffix[target]);
        }
    }
    if !suffix[0].is_finite() {
        return Err("no finite Viterbi path".into());
    }
    let mut agenda = BinaryHeap::from([Agenda {
        bound: suffix[0],
        path: Path {
            nodes: vec![0],
            score: 0.0,
        },
    }]);
    let mut paths = Vec::with_capacity(count);
    let mut expansions = 0;
    while let Some(state) = agenda.pop() {
        let node = *state.path.nodes.last().ok_or("empty A* path")?;
        if node == last {
            paths.push(state.path);
            if paths.len() == count {
                break;
            }
            continue;
        }
        expansions += 1;
        if expansions > expansion_limit {
            return Err("A* expansion budget exceeded".into());
        }
        for &edge in &graph.outgoing[node] {
            let target = graph.edges[edge].to;
            if !suffix[target].is_finite() {
                continue;
            }
            let mut path = state.path.clone();
            path.nodes.push(target);
            path.score += edge_scores[edge];
            agenda.push(Agenda {
                bound: path.score + suffix[target],
                path,
            });
        }
    }
    Ok(paths)
}

fn path_score_from_edges(graph: &Lattice, scores: &[f64], path: &[usize]) -> Result<f64> {
    let mut total = 0.0;
    for pair in path.windows(2) {
        let edge = graph.outgoing[pair[0]]
            .iter()
            .copied()
            .find(|&index| graph.edges[index].to == pair[1])
            .ok_or("gold path contains a missing lattice edge")?;
        total += scores[edge];
    }
    Ok(total)
}

fn primary_correction(
    model: &Model,
    graph: &Lattice,
    edge_scores: &[f64],
) -> Result<(f64, BTreeMap<Feature, f64>)> {
    if edge_scores.len() != graph.edges.len() {
        return Err("primary score/lattice edge count mismatch".into());
    }
    let last = graph.nodes.len() - 1;
    let mut alpha = vec![f64::NEG_INFINITY; graph.nodes.len()];
    alpha[0] = 0.0;
    for node in 1..graph.nodes.len() {
        for &edge in &graph.incoming[node] {
            alpha[node] = log_add(
                alpha[node],
                alpha[graph.edges[edge].from] + edge_scores[edge],
            );
        }
    }
    let log_z = alpha[last];
    if !log_z.is_finite() {
        return Err("no finite path in primary Hogwild lattice".into());
    }
    let mut beta = vec![f64::NEG_INFINITY; graph.nodes.len()];
    beta[last] = 0.0;
    for node in (0..last).rev() {
        for &edge in &graph.outgoing[node] {
            beta[node] = log_add(beta[node], edge_scores[edge] + beta[graph.edges[edge].to]);
        }
    }

    let mut gradient = BTreeMap::<Feature, f64>::new();
    for (edge_index, edge) in graph.edges.iter().enumerate() {
        let probability =
            (alpha[edge.from] + edge_scores[edge_index] + beta[edge.to] - log_z).exp();
        if probability == 0.0 {
            continue;
        }
        for feature in model.features(graph, *edge).into_iter().flatten() {
            *gradient.entry(feature).or_default() += probability;
        }
    }
    for pair in graph.gold.windows(2) {
        for feature in model
            .features(
                graph,
                Edge {
                    from: pair[0],
                    to: pair[1],
                },
            )
            .into_iter()
            .flatten()
        {
            *gradient.entry(feature).or_default() -= 1.0;
        }
    }
    let gold_score = path_score_from_edges(graph, edge_scores, &graph.gold)?;
    let loss = log_z - gold_score;
    if !loss.is_finite() || loss < -1e-8 {
        return Err("invalid primary full-lattice likelihood".into());
    }
    Ok((loss.max(0.0), gradient))
}

fn trigram_correction(
    view: &LiveScores<'_>,
    graph: &Lattice,
    paths: &[Path],
) -> Result<(f64, HashMap<Triple, f64>, bool)> {
    if paths.is_empty() {
        return Err("empty N-best set".into());
    }
    let gold_index = paths.iter().position(|path| path.nodes == graph.gold);
    let Some(gold_index) = gold_index else {
        return Ok((0.0, HashMap::new(), false));
    };
    let mut path_scores = Vec::with_capacity(paths.len());
    let mut path_counts = Vec::with_capacity(paths.len());
    let mut log_z = f64::NEG_INFINITY;
    let mut best = 0;
    for (index, path) in paths.iter().enumerate() {
        let score = path.score + view.trigram_score(graph, &path.nodes);
        if !score.is_finite() {
            return Err("nonfinite Hogwild reranking score".into());
        }
        if index == 0 || score > path_scores[best] {
            best = index;
        }
        log_z = log_add(log_z, score);
        path_scores.push(score);
        path_counts.push(trigram_counts(graph, &path.nodes));
    }
    let mut gradient = HashMap::<Triple, f64>::new();
    for (score, counts) in path_scores.iter().zip(&path_counts) {
        let probability = (*score - log_z).exp();
        for (&key, &count) in counts {
            *gradient.entry(key).or_default() += probability * count;
        }
    }
    for (&key, &count) in &path_counts[gold_index] {
        *gradient.entry(key).or_default() -= count;
    }
    Ok((
        (log_z - path_scores[gold_index]).max(0.0),
        gradient,
        paths[best].nodes == graph.gold,
    ))
}

fn trigram_counts(graph: &Lattice, path: &[usize]) -> HashMap<Triple, f64> {
    let mut counts = HashMap::new();
    for window in path.windows(3) {
        *counts
            .entry([
                graph.nodes[window[0]].word,
                graph.nodes[window[1]].word,
                graph.nodes[window[2]].word,
            ])
            .or_default() += 1.0;
    }
    counts
}

#[cfg(test)]
mod tests {
    use super::{
        Config, FirstMap, Master, TimedWeight, Trainer, advance_log_scale, materialize,
        nbest_from_edges, primary_correction, update_timed,
    };
    use crate::corpus::{Index, Limits, Sentence, Span};
    use crate::sparse::{Key, SparseMap};
    use crate::{BOS, EOS, Feature, Lattice, Model, Node, Parameters, Word};

    #[allow(clippy::too_many_lines)]
    fn fixture() -> (Model, Lattice) {
        let words = vec![
            Word {
                surface: String::new(),
                reading: String::new(),
                pos: 0,
                initial: 0.0,
                model_only: true,
                reading_length: 0,
            },
            Word {
                surface: String::new(),
                reading: String::new(),
                pos: 1,
                initial: 0.0,
                model_only: true,
                reading_length: 0,
            },
            Word {
                surface: String::new(),
                reading: String::new(),
                pos: 2,
                initial: 0.0,
                model_only: true,
                reading_length: 0,
            },
            Word {
                surface: "甲".into(),
                reading: "あ".into(),
                pos: 3,
                initial: 0.0,
                model_only: false,
                reading_length: 1,
            },
            Word {
                surface: "乙".into(),
                reading: "あ".into(),
                pos: 3,
                initial: 0.0,
                model_only: false,
                reading_length: 1,
            },
            Word {
                surface: "丙".into(),
                reading: "い".into(),
                pos: 3,
                initial: 0.0,
                model_only: false,
                reading_length: 1,
            },
            Word {
                surface: "丁".into(),
                reading: "い".into(),
                pos: 3,
                initial: 0.0,
                model_only: false,
                reading_length: 1,
            },
        ];
        let pos_token_ids = (0..4)
            .map(|pos| u32::try_from(words.len() + pos).unwrap())
            .collect();
        let model = Model {
            words,
            pos_names: vec!["reserved".into(), "BOS".into(), "EOS".into(), "名詞".into()],
            pos_token_ids,
            affix_pos_selections: Vec::new(),
            cross_bigrams_enabled: false,
            pos_trigrams_enabled: false,
            initial_pos: vec![0.0; 16],
            initial_bigrams: SparseMap::default(),
            initial_trigrams: SparseMap::default(),
            initial_pos_trigrams: SparseMap::default(),
            initial_reading_length_bigrams: [0.0; 81],
            initial_reading_length_trigrams: [0.0; 729],
            first: Parameters::default(),
            third: Parameters::default(),
            third_pos: Parameters::default(),
            third_reading_length: Parameters::default(),
            llm_alpha: 0.0,
            llm_crf_score_rms: 1.0,
            llm_feature_rms: 1.0,
            missing_pos_log: 0.0,
        };
        let graph = Lattice::new(
            vec![
                Node {
                    start: 0,
                    end: 0,
                    word: BOS,
                    script: 0,
                },
                Node {
                    start: 0,
                    end: 1,
                    word: 3,
                    script: 4,
                },
                Node {
                    start: 0,
                    end: 1,
                    word: 4,
                    script: 4,
                },
                Node {
                    start: 1,
                    end: 2,
                    word: 5,
                    script: 4,
                },
                Node {
                    start: 1,
                    end: 2,
                    word: 6,
                    script: 4,
                },
                Node {
                    start: 2,
                    end: 2,
                    word: EOS,
                    script: 1,
                },
            ],
            vec![0, 1, 3, 5],
            100,
        )
        .unwrap();
        (model, graph)
    }

    #[test]
    fn primary_correction_uses_the_full_lattice_not_the_nbest_prefix() {
        let (model, graph) = fixture();
        let edge_scores = vec![0.0; graph.edges.len()];
        let viterbi_only = nbest_from_edges(&graph, &edge_scores, 1, 100).unwrap();
        assert_eq!(viterbi_only.len(), 1);
        let (_, gradient) = primary_correction(&model, &graph, &edge_scores).unwrap();
        assert!((gradient[&Feature::Word(3)] + 0.5).abs() < 1e-12);
        assert!((gradient[&Feature::Word(4)] - 0.5).abs() < 1e-12);
        assert!((gradient[&Feature::Word(5)] + 0.5).abs() < 1e-12);
        assert!((gradient[&Feature::Word(6)] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn shared_map_lazily_decays_untouched_keys_and_preserves_additions() {
        let map = FirstMap::new();
        let key = Feature::Word(99).pack();
        map.insert_sync(
            key,
            TimedWeight {
                value: 2.0,
                log_scale: 0.0,
            },
        )
        .unwrap();
        let scale = std::sync::atomic::AtomicU64::new(0.0_f64.to_bits());
        let first_scale = advance_log_scale(&scale, 0.5);
        let first_value = map.read_sync(&key, |_, value| *value).unwrap();
        assert!((materialize(first_value, first_scale) - 1.0).abs() < 1e-12);

        let second_scale = advance_log_scale(&scale, 0.5);
        let second_value = map.read_sync(&key, |_, value| *value).unwrap();
        assert!((materialize(second_value, second_scale) - 0.5).abs() < 1e-12);

        update_timed(&map, key, 0.25, second_scale);
        let updated = map.read_sync(&key, |_, value| *value).unwrap();
        assert!((materialize(updated, second_scale) - 0.75).abs() < 1e-12);
    }

    #[test]
    fn shared_map_does_not_lose_contended_updates() {
        let map = FirstMap::new();
        let key = Feature::Word(101).pack();
        let threads = 8;
        let updates_per_thread = 2_000;
        std::thread::scope(|scope| {
            for _ in 0..threads {
                let map = &map;
                scope.spawn(move || {
                    for _ in 0..updates_per_thread {
                        update_timed(map, key, 0.25, 0.0);
                    }
                });
            }
        });
        let actual = map.read_sync(&key, |_, value| value.value).unwrap();
        let expected = f64::from(threads * updates_per_thread) * 0.25;
        assert!(
            (actual - expected).abs() < 1e-9,
            "actual={actual}, expected={expected}"
        );
    }

    #[test]
    fn hogwild_updates_master_and_roundtrips_to_model() {
        let (mut model, graph) = fixture();
        let master = Master::from_model(&model);
        let trainer = Trainer::new(2).unwrap();
        let graphs = vec![&graph; 64];
        let report = master
            .train(
                &trainer,
                &model,
                &graphs,
                Config {
                    rate: 0.03,
                    trigram_rate: 0.03,
                    l2: 0.0,
                    nbest: 4,
                    astar_limit: 100,
                    add_gold_for_trigram: true,
                },
            )
            .unwrap();
        assert_eq!(report.sentences, 64);
        assert_eq!(report.raw_nbest_gold_hits, 64);
        assert!(report.primary_key_updates > 0);
        assert!(report.trigram_key_updates > 0);
        assert!(master.first_len() > 0);
        assert!(master.third_len() > 0);
        master.snapshot_into_model(&mut model);
        assert!((model.first.scale - 1.0).abs() < f64::EPSILON);
        assert!((model.third.scale - 1.0).abs() < f64::EPSILON);
        let gold = model.path_score(&graph, &graph.gold);
        let alternative = model.path_score(&graph, &[0, 2, 3, 5]);
        assert!(gold > alternative, "gold={gold}, alternative={alternative}");
    }

    #[test]
    fn sentence_stream_claims_examples_and_aggregates_both_hit_rates() {
        let (model, _) = fixture();
        let index = Index::new(&model).unwrap();
        let sentence = Sentence {
            page_id: 1,
            title: "test".into(),
            sentence: "甲丙".into(),
            reading: "あい".into(),
            spans: vec![
                Span {
                    start: 0,
                    end: 3,
                    word: 3,
                    script: 4,
                    forced: false,
                },
                Span {
                    start: 3,
                    end: 6,
                    word: 5,
                    script: 4,
                    forced: false,
                },
            ],
        };
        let sentences = vec![&sentence; 64];
        let master = Master::from_model(&model);
        let trainer = Trainer::new(4).unwrap();
        let report = master
            .train_sentences(
                &trainer,
                &model,
                &sentences,
                &index,
                Limits {
                    words: 48,
                    nodes: 100,
                    edges: 1000,
                },
                Config {
                    rate: 0.001,
                    trigram_rate: 0.001,
                    l2: 0.0,
                    nbest: 4,
                    astar_limit: 1000,
                    add_gold_for_trigram: true,
                },
            )
            .unwrap();
        assert_eq!(report.attempted_sentences, sentences.len());
        assert_eq!(
            report.sentences,
            sentences.len(),
            "error={:?}",
            report.error
        );
        assert!(report.error.is_none(), "error={:?}", report.error);
        assert_eq!(report.top1_gold_rate(), Some(1.0));
        assert_eq!(report.nbest_gold_inclusion_rate(), Some(1.0));
    }
}
