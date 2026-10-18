//! Training kernels with reusable worker buffers and packed-key accumulation.
//!
//! Hash iteration never determines floating-point summation order. Contributions
//! arrive in reference edge/path order, and results are sorted before reduction.
//! The public diagnostic kernels remain an independent comparison oracle.
use std::cell::RefCell;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::hash::BuildHasherDefault;
use std::marker::PhantomData;

use crate::sparse::{IntegerHasher, Key};
use crate::{
    Edge, Feature, Lattice, Model, Path, PosTriple, ReadingLengthTriple, Result, Triple, astar,
    log_add,
};

type RerankEntries = (
    Option<f64>,
    Entries<Triple>,
    Entries<PosTriple>,
    Entries<ReadingLengthTriple>,
    bool,
);

/// Sentence-local scores rebuilt on every call from the current model borrow.
/// Node/shape terms are shared by many edges; only POS and word-pair corrections
/// still need a lookup per edge. Addition order is unchanged.
#[derive(Default)]
pub(crate) struct EdgeScores {
    pub(crate) values: Vec<f64>,
    nodes: Vec<(f64, u32)>,
}
impl EdgeScores {
    pub(crate) fn fill(&mut self, model: &Model, graph: &Lattice) {
        self.values.clear();
        if graph.edges.len() < 100 {
            self.values
                .extend(graph.edges.iter().map(|&e| model.edge_score(graph, e)));
            return;
        }
        self.nodes.clear();
        self.nodes.extend(graph.nodes.iter().map(|n| {
            (
                model.weight(Feature::Word(n.word)),
                model.words[n.word as usize].pos,
            )
        }));
        self.values.extend(graph.edges.iter().map(|e| {
            let a = &graph.nodes[e.from];
            let b = &graph.nodes[e.to];
            [
                self.nodes[e.to].0,
                model.weight(Feature::Pos(self.nodes[e.from].1, self.nodes[e.to].1)),
                model.weight(Feature::Bigram(a.word, b.word)),
                model.weight(Feature::PosWord(
                    model.pos_token_ids[self.nodes[e.from].1 as usize],
                    b.word,
                )),
                model.weight(Feature::WordPos(
                    a.word,
                    model.pos_token_ids[self.nodes[e.to].1 as usize],
                )),
                model.weight(Feature::ReadingLengthBigram(
                    model.words[a.word as usize].reading_length,
                    model.words[b.word as usize].reading_length,
                )),
            ]
            .into_iter()
            .sum::<f64>()
        }));
    }
}

#[derive(Debug)]
pub(crate) struct Entries<K: Key> {
    values: Vec<(K::Packed, f64)>,
    marker: PhantomData<K>,
}
impl<K: Key> Entries<K> {
    pub(crate) fn empty() -> Self {
        Self {
            values: Vec::new(),
            marker: PhantomData,
        }
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = (K, f64)> + '_ {
        self.values
            .iter()
            .map(|&(key, value)| (K::unpack(key), value))
    }
    pub(crate) fn from_reference(gradient: std::collections::BTreeMap<K, f64>) -> Self {
        Self {
            values: gradient.into_iter().map(|(k, v)| (k.pack(), v)).collect(),
            marker: PhantomData,
        }
    }
}

#[derive(Debug)]
pub(crate) struct Accumulator<K: Key> {
    table: HashMap<K::Packed, f64, BuildHasherDefault<IntegerHasher>>,
    marker: PhantomData<K>,
}

#[derive(Debug)]
#[allow(clippy::struct_excessive_bools)]
pub(crate) struct ThirdGradient {
    pub(crate) primary_margin_loss: Option<f64>,
    pub(crate) primary_margin_gradient: Entries<Feature>,
    pub(crate) primary_margin_active: bool,
    pub(crate) primary_listwise_loss: Option<f64>,
    pub(crate) primary_listwise_gradient: Entries<Feature>,
    pub(crate) primary_listwise_active: bool,
    pub(crate) loss: Option<f64>,
    pub(crate) word_gradient: Entries<Triple>,
    pub(crate) pos_gradient: Entries<PosTriple>,
    pub(crate) reading_length_gradient: Entries<ReadingLengthTriple>,
    pub(crate) first_path_gold_hit: bool,
    pub(crate) post_trigram_gold_hit: bool,
    pub(crate) raw_nbest_gold_hit: bool,
    pub(crate) gold_augmented: bool,
}
impl<K: Key> Default for Accumulator<K> {
    fn default() -> Self {
        Self {
            table: HashMap::default(),
            marker: PhantomData,
        }
    }
}
impl<K: Key> Accumulator<K> {
    fn add(&mut self, key: K, value: f64) {
        *self.table.entry(key.pack()).or_default() += value;
    }
    fn finish(&mut self) -> Entries<K> {
        let mut values = self.table.drain().collect::<Vec<_>>();
        values.sort_unstable_by_key(|p| p.0);
        Entries {
            values,
            marker: PhantomData,
        }
    }
}

/// Merge already sorted worker gradients with a per-key divisor. The heap has
/// at most one item per sentence, not one per feature. Ties use sentence order,
/// retaining each key's exact floating-point addition order without another
/// hash table or re-sort.
fn reduce<K: Key>(mut gradients: Vec<Entries<K>>, divisor: impl Fn(K) -> f64) -> Entries<K> {
    let mut heap = BinaryHeap::new();
    for (sentence, gradient) in gradients.iter().enumerate() {
        if let Some(&(key, _)) = gradient.values.first() {
            heap.push(Reverse((key, sentence, 0)));
        }
    }
    let mut values = Vec::new();
    let mut pending: Option<(K::Packed, f64)> = None;
    while let Some(mut cursor) = heap.peek_mut() {
        let (key, sentence, position) = cursor.0;
        let value = gradients[sentence].values[position].1;
        if pending
            .as_ref()
            .is_some_and(|&(previous, _)| previous != key)
        {
            let (previous, sum) = pending.take().expect("previous gradient key");
            values.push((previous, sum / divisor(K::unpack(previous))));
        }
        // Include the reference's initial +0 addition (important for signed zero).
        pending.get_or_insert((key, 0.0)).1 += value;
        if let Some(&(next, _)) = gradients[sentence].values.get(position + 1) {
            cursor.0 = (next, sentence, position + 1);
        } else {
            gradients[sentence].values = Vec::new(); // Release an exhausted stream immediately.
            std::collections::binary_heap::PeekMut::pop(cursor);
        }
    }
    drop(gradients);
    if let Some((key, sum)) = pending {
        values.push((key, sum / divisor(K::unpack(key))));
    }
    Entries {
        values,
        marker: PhantomData,
    }
}

/// Sum sentence gradients without a batch-size divisor.
pub(crate) fn sum<K: Key>(gradients: Vec<Entries<K>>) -> Entries<K> {
    reduce(gradients, |_| 1.0)
}

/// Average gradients except for keys whose gradients must be summed.
pub(crate) fn average_except<K: Key>(
    gradients: Vec<Entries<K>>,
    sum_key: impl Fn(K) -> bool,
) -> Entries<K> {
    #[allow(clippy::cast_precision_loss)]
    let mean_divisor = gradients.len() as f64;
    reduce(
        gradients,
        |key| {
            if sum_key(key) { 1.0 } else { mean_divisor }
        },
    )
}

/// Mean-reduce each feature across the supplied sentence gradients.
pub(crate) fn average<K: Key>(gradients: Vec<Entries<K>>) -> Entries<K> {
    average_except(gradients, |_| false)
}

#[derive(Default)]
struct Workspace {
    scores: EdgeScores,
    alpha: Vec<f64>,
    beta: Vec<f64>,
    first: Accumulator<Feature>,
    third_word: Accumulator<Triple>,
    third_pos: Accumulator<PosTriple>,
    third_reading_length: Accumulator<ReadingLengthTriple>,
    triples: Vec<Triple>,
    pos_triples: Vec<PosTriple>,
    reading_length_triples: Vec<ReadingLengthTriple>,
    counts: Vec<(Triple, f64)>,
    pos_counts: Vec<(PosTriple, f64)>,
    reading_length_counts: Vec<(ReadingLengthTriple, f64)>,
    ranges: Vec<std::ops::Range<usize>>,
    pos_ranges: Vec<std::ops::Range<usize>>,
    reading_length_ranges: Vec<std::ops::Range<usize>>,
    path_scores: Vec<f64>,
    astar: astar::Workspace,
}
thread_local! {
    // One buffer set per persistent Rayon worker, never a copy of the model.
    static WORKSPACE: RefCell<Workspace> = RefCell::new(Workspace::default());
}

pub(crate) fn first(model: &Model, graph: &Lattice) -> Result<(f64, Entries<Feature>)> {
    WORKSPACE.with(|workspace| workspace.borrow_mut().first(model, graph))
}
pub(crate) fn third(
    model: &Model,
    graph: &Lattice,
    count: usize,
    cap: usize,
    augment_gold: bool,
    primary_nbest_margin: Option<f64>,
    primary_nbest_listwise: bool,
) -> Result<ThirdGradient> {
    // N-best and its loss are evaluated against the same immutable pre-update
    // model as the first-order gradient. Only one owned gold path is appended;
    // no per-edge correction table is materialized.
    WORKSPACE.with(|workspace| {
        let mut workspace = workspace.borrow_mut();
        let mut paths = workspace.astar.nbest(model, graph, count, cap)?;
        let raw_path_count = paths.len();
        let first_path_gold_hit = paths.first().is_some_and(|path| path.nodes == graph.gold);
        let raw_nbest_gold_hit = paths.iter().any(|path| path.nodes == graph.gold);
        let (primary_margin_loss, primary_margin_gradient, primary_margin_active) =
            if let Some(margin) = primary_nbest_margin {
                let (loss, gradient, active) =
                    workspace.primary_nbest_margin(model, graph, &paths, margin)?;
                (Some(loss), gradient, active)
            } else {
                (None, Entries::empty(), false)
            };
        let (primary_listwise_loss, primary_listwise_gradient, primary_listwise_active) =
            if primary_nbest_listwise {
                let (loss, gradient, active) =
                    workspace.primary_nbest_listwise(model, graph, &paths)?;
                (Some(loss), gradient, active)
            } else {
                (None, Entries::empty(), false)
            };
        let gold_augmented = augment_gold && !raw_nbest_gold_hit;
        if gold_augmented {
            paths.push(Path {
                nodes: graph.gold.clone(),
                score: model.path_score(graph, &graph.gold),
            });
        }
        let (loss, word_gradient, pos_gradient, reading_length_gradient, post_trigram_gold_hit) =
            workspace.rerank(model, graph, &paths, raw_path_count)?;
        Ok(ThirdGradient {
            primary_margin_loss,
            primary_margin_gradient,
            primary_margin_active,
            primary_listwise_loss,
            primary_listwise_gradient,
            primary_listwise_active,
            loss,
            word_gradient,
            pos_gradient,
            reading_length_gradient,
            first_path_gold_hit,
            post_trigram_gold_hit,
            raw_nbest_gold_hit,
            gold_augmented,
        })
    })
}

#[cfg(test)]
#[allow(clippy::type_complexity)]
pub(crate) fn rerank_for_test(
    model: &Model,
    graph: &Lattice,
    paths: &[Path],
) -> Result<(Option<f64>, Vec<(Triple, f64)>)> {
    WORKSPACE.with(|workspace| {
        workspace
            .borrow_mut()
            .rerank(model, graph, paths, paths.len())
            .map(|(loss, word, _, _, _)| (loss, word.iter().collect()))
    })
}

impl Workspace {
    /// Candidate-set CRF loss over raw N-best plus the gold path if missing.
    /// The returned gradient is expected primary feature counts minus gold
    /// counts. The raw candidate-hit metric is computed separately, before
    /// this training-only gold inclusion.
    fn primary_nbest_listwise(
        &mut self,
        model: &Model,
        graph: &Lattice,
        paths: &[Path],
    ) -> Result<(f64, Entries<Feature>, bool)> {
        if paths.is_empty() {
            return Err("empty primary N-best set".into());
        }
        self.first.table.clear();
        let gold_in_paths = paths.iter().any(|path| path.nodes == graph.gold);
        let gold_score = if gold_in_paths {
            paths
                .iter()
                .find(|path| path.nodes == graph.gold)
                .expect("gold candidate exists")
                .score
        } else {
            model.path_score(graph, &graph.gold)
        };
        if !gold_score.is_finite() {
            return Err("nonfinite primary N-best gold score".into());
        }
        let mut log_z = f64::NEG_INFINITY;
        for path in paths {
            if !path.score.is_finite() {
                return Err("nonfinite primary N-best candidate score".into());
            }
            log_z = log_add(log_z, path.score);
        }
        if !gold_in_paths {
            log_z = log_add(log_z, gold_score);
        }
        if !log_z.is_finite() {
            return Err("nonfinite primary N-best log partition".into());
        }
        for path in paths {
            let probability = (path.score - log_z).exp();
            if probability == 0.0 {
                continue;
            }
            self.add_path_features(model, graph, &path.nodes, probability);
        }
        if !gold_in_paths {
            let probability = (gold_score - log_z).exp();
            if probability > 0.0 {
                self.add_path_features(model, graph, &graph.gold, probability);
            }
        }
        self.add_path_features(model, graph, &graph.gold, -1.0);
        let loss = (log_z - gold_score).max(0.0);
        if !loss.is_finite() {
            return Err("nonfinite primary N-best listwise loss".into());
        }
        let active = paths.len() + usize::from(!gold_in_paths) > 1;
        Ok((loss, self.first.finish(), active))
    }

    fn add_path_features(&mut self, model: &Model, graph: &Lattice, path: &[usize], scale: f64) {
        for pair in path.windows(2) {
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
                self.first.add(feature, scale);
            }
        }
    }

    fn primary_nbest_margin(
        &mut self,
        model: &Model,
        graph: &Lattice,
        paths: &[Path],
        margin: f64,
    ) -> Result<(f64, Entries<Feature>, bool)> {
        self.first.table.clear();
        let gold_score = model.path_score(graph, &graph.gold);
        let mut loss_sum = 0.0;
        let mut active = false;
        for path in paths {
            if path.nodes == graph.gold {
                continue;
            }
            let path_score = model.path_score(graph, &path.nodes);
            let loss = margin + path_score - gold_score;
            if !loss.is_finite() {
                return Err("nonfinite primary N-best pairwise hinge loss".into());
            }
            if loss <= 0.0 {
                continue;
            }
            loss_sum += loss;
            active = true;
            // Each violating raw-N-best path contributes its own negative
            // feature counts. Matching gold features cancel path by path.
            for pair in path.nodes.windows(2) {
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
                    self.first.add(feature, 1.0);
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
                    self.first.add(feature, -1.0);
                }
            }
        }
        if !loss_sum.is_finite() {
            return Err("nonfinite summed primary N-best pairwise hinge loss".into());
        }
        Ok((loss_sum, self.first.finish(), active))
    }

    fn first(&mut self, model: &Model, graph: &Lattice) -> Result<(f64, Entries<Feature>)> {
        self.first.table.clear();
        self.scores.fill(model, graph);
        self.alpha.resize(graph.nodes.len(), f64::NEG_INFINITY);
        self.alpha.fill(f64::NEG_INFINITY);
        self.alpha[0] = 0.0;
        for node in 1..graph.nodes.len() {
            for &e in &graph.incoming[node] {
                self.alpha[node] = log_add(
                    self.alpha[node],
                    self.alpha[graph.edges[e].from] + self.scores.values[e],
                );
            }
        }
        let last = graph.nodes.len() - 1;
        let log_z = self.alpha[last];
        if !log_z.is_finite() {
            return Err("no finite BOS-to-EOS path".into());
        }
        self.beta.resize(graph.nodes.len(), f64::NEG_INFINITY);
        self.beta.fill(f64::NEG_INFINITY);
        self.beta[last] = 0.0;
        for node in (0..last).rev() {
            for &e in &graph.outgoing[node] {
                self.beta[node] = log_add(
                    self.beta[node],
                    self.scores.values[e] + self.beta[graph.edges[e].to],
                );
            }
        }
        for (e, edge) in graph.edges.iter().enumerate() {
            let probability =
                (self.alpha[edge.from] + self.scores.values[e] + self.beta[edge.to] - log_z).exp();
            if probability == 0.0 {
                continue;
            }
            for key in model.features(graph, *edge).into_iter().flatten() {
                self.first.add(key, probability);
            }
        }
        for pair in graph.gold.windows(2) {
            for key in model
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
                self.first.add(key, -1.0);
            }
        }
        let loss = log_z - model.path_score(graph, &graph.gold);
        if !loss.is_finite() || loss < -1e-8 {
            return Err("invalid first-order likelihood".into());
        }
        // Training never consumes the full per-edge marginal vector.
        Ok((loss.max(0.0), self.first.finish()))
    }

    fn append_counts(&mut self, graph: &Lattice, path: &[usize]) -> std::ops::Range<usize> {
        self.triples.clear();
        self.triples.extend(path.windows(3).map(|p| {
            [
                graph.nodes[p[0]].word,
                graph.nodes[p[1]].word,
                graph.nodes[p[2]].word,
            ]
        }));
        self.triples.sort_unstable();
        let start = self.counts.len();
        for &key in &self.triples {
            if self.counts.len() > start {
                let (previous, count) = self.counts.last_mut().expect("nonempty count range");
                if *previous == key {
                    *count += 1.0;
                    continue;
                }
            }
            self.counts.push((key, 1.0));
        }
        start..self.counts.len()
    }

    fn append_pos_counts(
        &mut self,
        graph: &Lattice,
        path: &[usize],
        model: &Model,
    ) -> std::ops::Range<usize> {
        self.pos_triples.clear();
        let start = self.pos_counts.len();
        if !model.pos_trigrams_enabled {
            return start..start;
        }
        self.pos_triples.extend(path.windows(3).map(|p| {
            [
                u16::try_from(model.words[graph.nodes[p[0]].word as usize].pos)
                    .expect("POS IDs fit in 16 bits"),
                u16::try_from(model.words[graph.nodes[p[1]].word as usize].pos)
                    .expect("POS IDs fit in 16 bits"),
                u16::try_from(model.words[graph.nodes[p[2]].word as usize].pos)
                    .expect("POS IDs fit in 16 bits"),
            ]
        }));
        self.pos_triples.sort_unstable();
        for &key in &self.pos_triples {
            if self.pos_counts.len() > start {
                let (previous, count) = self.pos_counts.last_mut().expect("POS count range");
                if *previous == key {
                    *count += 1.0;
                    continue;
                }
            }
            self.pos_counts.push((key, 1.0));
        }
        start..self.pos_counts.len()
    }

    fn append_reading_length_counts(
        &mut self,
        graph: &Lattice,
        path: &[usize],
        model: &Model,
    ) -> std::ops::Range<usize> {
        self.reading_length_triples.clear();
        self.reading_length_triples.extend(path.windows(3).map(|p| {
            [
                model.words[graph.nodes[p[0]].word as usize].reading_length,
                model.words[graph.nodes[p[1]].word as usize].reading_length,
                model.words[graph.nodes[p[2]].word as usize].reading_length,
            ]
        }));
        self.reading_length_triples.sort_unstable();
        let start = self.reading_length_counts.len();
        for &key in &self.reading_length_triples {
            if self.reading_length_counts.len() > start {
                let (previous, count) = self
                    .reading_length_counts
                    .last_mut()
                    .expect("reading-length count range");
                if *previous == key {
                    *count += 1.0;
                    continue;
                }
            }
            self.reading_length_counts.push((key, 1.0));
        }
        start..self.reading_length_counts.len()
    }

    fn rerank(
        &mut self,
        model: &Model,
        graph: &Lattice,
        paths: &[Path],
        prediction_count: usize,
    ) -> Result<RerankEntries> {
        if paths.is_empty() {
            return Err("empty N-best set".into());
        }
        self.third_pos.table.clear();
        self.third_word.table.clear();
        self.third_reading_length.table.clear();
        self.counts.clear();
        self.pos_counts.clear();
        self.reading_length_counts.clear();
        self.ranges.clear();
        self.pos_ranges.clear();
        self.reading_length_ranges.clear();
        self.path_scores.clear();
        let mut log_z = f64::NEG_INFINITY;
        let mut gold = None;
        for (index, path) in paths.iter().enumerate() {
            let range = self.append_counts(graph, &path.nodes);
            let pos_range = self.append_pos_counts(graph, &path.nodes, model);
            let reading_length_range = self.append_reading_length_counts(graph, &path.nodes, model);
            let trigram: f64 = self.counts[range.clone()]
                .iter()
                .map(|&(k, n)| n * model.trigram_weight(k))
                .sum();
            let pos_trigram: f64 = self.pos_counts[pos_range.clone()]
                .iter()
                .map(|&(k, n)| n * model.pos_trigram_weight(k))
                .sum();
            let reading_length_trigram: f64 = self.reading_length_counts
                [reading_length_range.clone()]
            .iter()
            .map(|&(k, n)| n * model.reading_length_trigram_weight(k))
            .sum();
            // These paths were generated within the same model borrow; no stale
            // score from before an update can enter this private training kernel.
            let score = path.score + (trigram + pos_trigram + reading_length_trigram);
            if !score.is_finite() {
                return Err("nonfinite reranking score".into());
            }
            if path.nodes == graph.gold {
                gold = Some(index);
            }
            log_z = log_add(log_z, score);
            self.path_scores.push(score);
            self.ranges.push(range);
            self.pos_ranges.push(pos_range);
            self.reading_length_ranges.push(reading_length_range);
        }
        let mut best_prediction = 0;
        for index in 1..prediction_count {
            if self.path_scores[index] > self.path_scores[best_prediction] {
                best_prediction = index;
            }
        }
        let post_trigram_gold_hit = paths[best_prediction].nodes == graph.gold;
        let loss = if let Some(gold) = gold {
            for (index, score) in self.path_scores.iter().enumerate() {
                let probability = (*score - log_z).exp();
                for &(key, count) in &self.counts[self.ranges[index].clone()] {
                    self.third_word.add(key, probability * count);
                }
            }
            for &(key, count) in &self.counts[self.ranges[gold].clone()] {
                self.third_word.add(key, -count);
            }
            for (range, score) in self.pos_ranges.iter().zip(&self.path_scores) {
                let probability = (*score - log_z).exp();
                for &(key, count) in &self.pos_counts[range.clone()] {
                    self.third_pos.add(key, probability * count);
                }
            }
            for (range, score) in self.reading_length_ranges.iter().zip(&self.path_scores) {
                let probability = (*score - log_z).exp();
                for &(key, count) in &self.reading_length_counts[range.clone()] {
                    self.third_reading_length.add(key, probability * count);
                }
            }
            for &(key, count) in &self.pos_counts[self.pos_ranges[gold].clone()] {
                self.third_pos.add(key, -count);
            }
            for &(key, count) in
                &self.reading_length_counts[self.reading_length_ranges[gold].clone()]
            {
                self.third_reading_length.add(key, -count);
            }
            Some((log_z - self.path_scores[gold]).max(0.0))
        } else {
            None
        };
        Ok((
            loss,
            self.third_word.finish(),
            self.third_pos.finish(),
            self.third_reading_length.finish(),
            post_trigram_gold_hit,
        ))
    }
}
