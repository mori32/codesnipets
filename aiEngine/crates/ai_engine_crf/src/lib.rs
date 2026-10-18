//! Offline IME training: first-order lattice CRF, then N-best trigram learning.
//!
//! All scores are additive f64 potentials. Sparse corrections default to zero.
//! No model server, integer `MeCab` costs, or text-generation runtime is involved.
#![allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]

mod astar;
pub mod blocks;
pub mod corpus;
pub mod cutoff;
pub mod hogwild;
mod optimized;
pub mod output;
pub mod parallel;
pub mod sparse;

use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap};

use ai_engine_dict::{
    TrainingAffixPosSelection, TrainingBigram, TrainingCrfNgrams, TrainingFeatureCounts,
    TrainingSeeds, TrainingTrigram,
};
use serde::{Deserialize, Serialize};
use sparse::{Key, SparseMap};

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
pub const BOS: u32 = 1;
pub const EOS: u32 = 2;
pub const SCRIPT_NAMES: [&str; 10] = [
    "BOS", "EOS", "hiragana", "katakana", "kanji", "symbol", "latin", "number", "mixed", "other",
];
pub const READING_LENGTH_CLASS_COUNT: usize = 9;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Word {
    pub surface: String,
    pub reading: String,
    pub pos: u32,
    pub initial: f64,
    pub model_only: bool,
    #[serde(default)]
    pub reading_length: u8,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub enum Feature {
    Word(u32),
    Pos(u32, u32),
    Bigram(u32, u32),
    Script(u8),
    ScriptBigram(u8, u8),
    ReadingLengthBigram(u8, u8),
    /// Dedicated synthetic POS token ID followed by an ordinary word ID.
    PosWord(u32, u32),
    /// Ordinary word ID followed by a dedicated synthetic POS token ID.
    WordPos(u32, u32),
}

pub type Gradient = BTreeMap<Feature, f64>;
pub type Triple = [u32; 3];
/// POS IDs fit in 16 bits; this keeps a POS-trigram hash key to six bytes.
pub type PosTriple = [u16; 3];
pub type ReadingLengthTriple = [u8; 3];
pub type TriGradient = BTreeMap<Triple, f64>;
pub type PosTriGradient = BTreeMap<PosTriple, f64>;
pub type ReadingLengthTriGradient = BTreeMap<ReadingLengthTriple, f64>;

/// Globally decayed sparse deltas, regularized toward their zero initialization.
/// Keeping a common scale applies L2 to *all* weights, including untouched ones.
#[derive(Clone, Debug)]
pub struct Parameters<K: Key> {
    pub values: SparseMap<K>,
    pub scale: f64,
}

impl<K: Key> Default for Parameters<K> {
    fn default() -> Self {
        Self {
            values: SparseMap::default(),
            scale: 1.0,
        }
    }
}

impl<K: Key> Parameters<K> {
    pub fn get(&self, key: K) -> f64 {
        self.values.get(&key).copied().unwrap_or(0.0) * self.scale
    }

    pub fn update(&mut self, gradient: &BTreeMap<K, f64>, rate: f64, l2: f64) -> Result<()> {
        self.update_entries(gradient.iter().map(|(&k, &v)| (k, v)), rate, l2)
    }

    /// Entries must have the same deterministic order as the reference gradient.
    pub(crate) fn update_entries(
        &mut self,
        gradient: impl IntoIterator<Item = (K, f64)>,
        rate: f64,
        l2: f64,
    ) -> Result<()> {
        if !rate.is_finite() || rate <= 0.0 || !l2.is_finite() || l2 < 0.0 || rate * l2 >= 1.0 {
            return Err("invalid learning rate/L2 (require rate > 0, L2 >= 0, rate*L2 < 1)".into());
        }
        self.scale *= 1.0 - rate * l2;
        if self.scale < 1e-100 {
            for value in self.values.values_mut() {
                *value *= self.scale;
            }
            self.scale = 1.0;
        }
        for (key, derivative) in gradient {
            if !derivative.is_finite() {
                return Err("nonfinite gradient".into());
            }
            // Suppress floating-point cancellation, not semantic feature pruning.
            if derivative.abs() <= 1e-12 {
                continue;
            }
            let weight = self.values.entry_value(key);
            *weight -= rate * derivative / self.scale;
            if !weight.is_finite() {
                return Err("nonfinite updated weight".into());
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct Model {
    /// NULL=0, BOS=1, EOS=2; subsequent IDs are training-local source identities.
    pub words: Vec<Word>,
    pub pos_names: Vec<String>,
    /// Synthetic word-namespace IDs, one for each POS ID, in POS-ID order.
    pub pos_token_ids: Vec<u32>,
    pub affix_pos_selections: Vec<TrainingAffixPosSelection>,
    /// Whether POS-WORD/WORD-POS first-order features are active. Old checkpoints
    /// default to enabled; the legacy-POS experiment stores this as false.
    pub cross_bigrams_enabled: bool,
    /// Old checkpoints default to enabled; the next experiment disables POS trigrams.
    pub pos_trigrams_enabled: bool,
    pub initial_pos: Vec<f64>,
    /// Immutable log-ratio corrections that interpolate retained static word bigrams.
    pub initial_bigrams: SparseMap<Feature>,
    /// Immutable log-ratio corrections that interpolate retained static word trigrams.
    pub initial_trigrams: SparseMap<Triple>,
    /// Immutable POS-trigram log-ratio corrections against POS-bigram probabilities.
    pub initial_pos_trigrams: SparseMap<PosTriple>,
    /// Dense 9x9 and 9x9x9 immutable corrections; missing events default to zero.
    pub initial_reading_length_bigrams: [f64; 81],
    pub initial_reading_length_trigrams: [f64; 729],
    pub first: Parameters<Feature>,
    pub third: Parameters<Triple>,
    pub third_pos: Parameters<PosTriple>,
    pub third_reading_length: Parameters<ReadingLengthTriple>,
    /// Scalar multiplying the normalized local-LLM mean-token-softmax feature
    /// during N-best reranking. Older checkpoints load this as zero.
    pub llm_alpha: f64,
    /// Pooled within-N-best RMS for first-order+trigram CRF scores used by
    /// normalized local-LLM reranking. Older checkpoints default to one.
    pub llm_crf_score_rms: f64,
    /// Pooled within-N-best RMS for the local-LLM mean-token-softmax feature.
    /// Older checkpoints default to one.
    pub llm_feature_rms: f64,
    pub missing_pos_log: f64,
}

impl Model {
    pub fn from_seeds(seeds: TrainingSeeds, missing_probability: f64) -> Result<Self> {
        Self::from_seeds_with_bigrams(seeds, Vec::new(), missing_probability, 0.9)
    }

    #[allow(clippy::cast_precision_loss, clippy::too_many_lines)]
    pub fn from_seeds_with_bigrams(
        seeds: TrainingSeeds,
        bigrams: Vec<TrainingBigram>,
        missing_probability: f64,
        mixture_delta: f64,
    ) -> Result<Self> {
        Self::from_seeds_with_ngrams(
            seeds,
            bigrams,
            Vec::new(),
            missing_probability,
            mixture_delta,
            0.1,
        )
    }

    #[allow(clippy::cast_precision_loss, clippy::too_many_lines)]
    pub fn from_seeds_with_ngrams(
        seeds: TrainingSeeds,
        bigrams: Vec<TrainingBigram>,
        trigrams: Vec<TrainingTrigram>,
        missing_probability: f64,
        bigram_mixture_delta: f64,
        trigram_mixture_weight: f64,
    ) -> Result<Self> {
        Self::from_crf_ngrams(
            TrainingCrfNgrams {
                seeds,
                bigrams,
                trigrams,
                pos_word_bigrams: Vec::new(),
                word_pos_bigrams: Vec::new(),
                affix_pos_selections: Vec::new(),
                feature_counts: TrainingFeatureCounts::default(),
            },
            missing_probability,
            bigram_mixture_delta,
            trigram_mixture_weight,
            0.3,
            0.15,
        )
    }

    #[allow(clippy::cast_precision_loss, clippy::too_many_lines)]
    pub fn from_crf_ngrams(
        training: TrainingCrfNgrams,
        missing_probability: f64,
        bigram_baseline_weight: f64,
        word_trigram_empirical_weight: f64,
        pos_trigram_empirical_weight: f64,
        reading_length_empirical_weight: f64,
    ) -> Result<Self> {
        let TrainingCrfNgrams {
            seeds,
            bigrams,
            trigrams,
            pos_word_bigrams,
            word_pos_bigrams,
            affix_pos_selections,
            feature_counts,
        } = training;
        let cross_bigrams_enabled = !pos_word_bigrams.is_empty() || !word_pos_bigrams.is_empty();
        if !missing_probability.is_finite()
            || missing_probability <= 0.0
            || missing_probability > 1.0
        {
            return Err("missing POS probability must be in (0,1]".into());
        }
        if !bigram_baseline_weight.is_finite() || !(0.0..=1.0).contains(&bigram_baseline_weight) {
            return Err("initial bigram baseline weight must be in [0,1]".into());
        }
        for weight in [
            word_trigram_empirical_weight,
            pos_trigram_empirical_weight,
            reading_length_empirical_weight,
        ] {
            if !weight.is_finite() || !(0.0..=1.0).contains(&weight) {
                return Err("initial empirical n-gram weight must be in [0,1]".into());
            }
        }
        let mut pos_names = vec!["[RESERVED]".to_owned()];
        pos_names.extend(seeds.pos_names);
        let pos_id = |name: &str| -> Result<u32> {
            Ok(u32::try_from(
                pos_names
                    .iter()
                    .position(|p| p == name)
                    .ok_or_else(|| format!("missing model POS {name}"))?,
            )?)
        };
        let mut words = vec![
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
                pos: pos_id("[BOS]")?,
                initial: 0.0,
                model_only: true,
                reading_length: 0,
            },
            Word {
                surface: String::new(),
                reading: String::new(),
                pos: pos_id("[EOS]")?,
                initial: 0.0,
                model_only: true,
                reading_length: 0,
            },
        ];
        let mut seed_to_word = vec![None; seeds.words.len()];
        for (seed_id, seed) in seeds.words.into_iter().enumerate() {
            if seed.pos_id == words[BOS as usize].pos || seed.pos_id == words[EOS as usize].pos {
                seed_to_word[seed_id] = Some(if seed.pos_id == words[BOS as usize].pos {
                    BOS
                } else {
                    EOS
                });
                continue;
            }
            if seed.pos_id == 0
                || seed.pos_id as usize >= pos_names.len()
                || !seed.log_unigram.is_finite()
            {
                return Err("invalid word training seed".into());
            }
            let reading_length =
                reading_length_bucket(&seed.reading, &pos_names[seed.pos_id as usize]);
            seed_to_word[seed_id] = Some(u32::try_from(words.len())?);
            words.push(Word {
                surface: seed.surface,
                reading: seed.reading,
                pos: seed.pos_id,
                initial: seed.log_unigram,
                model_only: seed.model_only,
                reading_length,
            });
        }
        if words.len() > sparse::MAX_ID as usize + 1
            || pos_names.len() > sparse::MAX_ID as usize + 1
            || words.len().saturating_add(pos_names.len()) > sparse::MAX_ID as usize + 1
        {
            return Err("training dictionary exceeds 24-bit ID capacity".into());
        }
        if pos_names.len() > usize::from(u16::MAX) + 1 {
            return Err("POS inventory exceeds 16-bit trigram ID capacity".into());
        }
        let missing_pos_log = missing_probability.ln();
        let pos_token_ids = (0..pos_names.len())
            .map(|pos| u32::try_from(words.len().saturating_add(pos)))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let mut initial_pos = vec![
            missing_pos_log;
            pos_names
                .len()
                .checked_mul(pos_names.len())
                .ok_or("POS matrix overflow")?
        ];
        for (a, b, score) in seeds.pos_bigrams {
            if a == 0
                || b == 0
                || a as usize >= pos_names.len()
                || b as usize >= pos_names.len()
                || !score.is_finite()
            {
                return Err("invalid POS training seed".into());
            }
            initial_pos[a as usize * pos_names.len() + b as usize] = score;
        }
        let mut initial_bigrams = SparseMap::default();
        for bigram in bigrams {
            if bigram.count == 0 || bigram.outgoing_count < bigram.count {
                return Err("invalid initial word bigram counts".into());
            }
            let source = seed_to_word
                .get(bigram.source as usize)
                .copied()
                .flatten()
                .ok_or("initial bigram source has no model word")?;
            let target = seed_to_word
                .get(bigram.target as usize)
                .copied()
                .flatten()
                .ok_or("initial bigram target has no model word")?;
            let source_pos = words[source as usize].pos as usize;
            let target_pos = words[target as usize].pos as usize;
            let base_log = initial_pos[source_pos * pos_names.len() + target_pos]
                + words[target as usize].initial;
            let empirical_log = (bigram.count as f64 / bigram.outgoing_count as f64).ln();
            let mixture_log = match (bigram_baseline_weight, 1.0 - bigram_baseline_weight) {
                (0.0, _) => empirical_log,
                (_, 0.0) => base_log,
                (delta, complement) => {
                    logaddexp(delta.ln() + base_log, complement.ln() + empirical_log)
                }
            };
            let correction = mixture_log - base_log;
            if !correction.is_finite() {
                return Err("nonfinite initial word bigram correction".into());
            }
            let key = Feature::Bigram(source, target);
            if initial_bigrams.insert(key, correction).is_some() {
                return Err("duplicate initial word bigram seed".into());
            }
        }
        for bigram in pos_word_bigrams {
            let target = seed_to_word
                .get(bigram.target_word as usize)
                .copied()
                .flatten()
                .ok_or("initial POS-word target has no model word")?;
            if bigram.count == 0
                || bigram.source_pos == 0
                || bigram.source_pos as usize >= pos_names.len()
                || feature_counts
                    .pos_context_counts
                    .get(bigram.source_pos as usize)
                    .copied()
                    .unwrap_or(0)
                    < bigram.count
            {
                return Err("invalid initial POS-word bigram counts".into());
            }
            let source_pos = bigram.source_pos as usize;
            let target_pos = words[target as usize].pos as usize;
            let base_log = initial_pos[source_pos * pos_names.len() + target_pos]
                + words[target as usize].initial;
            let denominator = feature_counts.pos_context_counts[source_pos];
            let empirical_log = (bigram.count as f64 / denominator as f64).ln();
            let correction =
                interpolation_delta(base_log, empirical_log, 1.0 - bigram_baseline_weight);
            let key = Feature::PosWord(pos_token_ids[source_pos], target);
            if !correction.is_finite() || initial_bigrams.insert(key, correction).is_some() {
                return Err("invalid or duplicate initial POS-word bigram seed".into());
            }
        }
        for bigram in word_pos_bigrams {
            let source = seed_to_word
                .get(bigram.source_word as usize)
                .copied()
                .flatten()
                .ok_or("initial word-POS source has no model word")?;
            let source_pos = words[source as usize].pos as usize;
            let target_pos = bigram.target_pos as usize;
            if bigram.count == 0
                || target_pos == 0
                || target_pos >= pos_names.len()
                || bigram.outgoing_count < bigram.count
            {
                return Err("invalid initial word-POS bigram counts".into());
            }
            let base_log = initial_pos[source_pos * pos_names.len() + target_pos];
            let empirical_log = (bigram.count as f64 / bigram.outgoing_count as f64).ln();
            let correction =
                interpolation_delta(base_log, empirical_log, 1.0 - bigram_baseline_weight);
            let key = Feature::WordPos(source, pos_token_ids[target_pos]);
            if !correction.is_finite() || initial_bigrams.insert(key, correction).is_some() {
                return Err("invalid or duplicate initial word-POS bigram seed".into());
            }
        }
        let mut initial_trigrams = SparseMap::default();
        for trigram in trigrams {
            if trigram.count == 0 || trigram.parent_count < trigram.count {
                return Err("invalid initial word trigram counts".into());
            }
            let first = seed_to_word
                .get(trigram.first as usize)
                .copied()
                .flatten()
                .ok_or("initial trigram first word has no model ID")?;
            let middle = seed_to_word
                .get(trigram.middle as usize)
                .copied()
                .flatten()
                .ok_or("initial trigram middle word has no model ID")?;
            let target = seed_to_word
                .get(trigram.target as usize)
                .copied()
                .flatten()
                .ok_or("initial trigram target has no model ID")?;
            let middle_pos = words[middle as usize].pos as usize;
            let target_pos = words[target as usize].pos as usize;
            let base_bigram_log = initial_pos[middle_pos * pos_names.len() + target_pos]
                + words[target as usize].initial;
            let trigram_log = (trigram.count as f64).ln() - (trigram.parent_count as f64).ln();
            let correction =
                interpolation_delta(base_bigram_log, trigram_log, word_trigram_empirical_weight);
            if !correction.is_finite() {
                return Err("nonfinite initial word trigram correction".into());
            }
            if initial_trigrams
                .insert([first, middle, target], correction)
                .is_some()
            {
                return Err("duplicate initial word trigram seed".into());
            }
        }
        let mut initial_pos_trigrams = SparseMap::default();
        for trigram in feature_counts
            .include_pos_trigrams
            .then_some(&feature_counts.pos_trigrams)
            .into_iter()
            .flatten()
        {
            if trigram.count == 0
                || trigram.context_count < trigram.count
                || trigram.first == 0
                || trigram.middle == 0
                || trigram.target == 0
                || [trigram.first, trigram.middle, trigram.target]
                    .iter()
                    .any(|&id| id as usize >= pos_names.len())
            {
                return Err("invalid POS trigram seed".into());
            }
            let base_log =
                initial_pos[trigram.middle as usize * pos_names.len() + trigram.target as usize];
            let empirical_log = (trigram.count as f64 / trigram.context_count as f64).ln();
            let correction =
                interpolation_delta(base_log, empirical_log, pos_trigram_empirical_weight);
            let key = [trigram.first, trigram.middle, trigram.target];
            if initial_pos_trigrams.insert(key, correction).is_some() {
                return Err("duplicate initial POS trigram seed".into());
            }
        }
        let (base_reading_length_bigrams, base_reading_length_trigrams) =
            reading_length_base_probabilities(&words, &pos_names, &initial_pos, &feature_counts)?;
        let mut initial_reading_length_bigrams = [0.0; 81];
        for (index, correction) in initial_reading_length_bigrams.iter_mut().enumerate() {
            let count = feature_counts.reading_length_bigram_counts[index];
            let context = feature_counts.reading_length_bigram_context_counts[index / 9];
            let empirical = (context > 0 && count > 0).then(|| count as f64 / context as f64);
            *correction = interpolation_delta_probability(
                base_reading_length_bigrams[index],
                empirical,
                reading_length_empirical_weight,
            );
        }
        let mut initial_reading_length_trigrams = [0.0; 729];
        for (index, correction) in initial_reading_length_trigrams.iter_mut().enumerate() {
            let count = feature_counts.reading_length_trigram_counts[index];
            let context = feature_counts.reading_length_trigram_context_counts[index / 9];
            let empirical = (context > 0 && count > 0).then(|| count as f64 / context as f64);
            *correction = interpolation_delta_probability(
                base_reading_length_trigrams[index],
                empirical,
                reading_length_empirical_weight,
            );
        }
        Ok(Self {
            words,
            pos_names,
            pos_token_ids,
            affix_pos_selections,
            cross_bigrams_enabled,
            pos_trigrams_enabled: feature_counts.include_pos_trigrams,
            initial_pos,
            initial_bigrams,
            initial_trigrams,
            initial_pos_trigrams,
            initial_reading_length_bigrams,
            initial_reading_length_trigrams,
            first: Parameters::default(),
            third: Parameters::default(),
            third_pos: Parameters::default(),
            third_reading_length: Parameters::default(),
            llm_alpha: 0.0,
            llm_crf_score_rms: 1.0,
            llm_feature_rms: 1.0,
            missing_pos_log,
        })
    }

    #[must_use]
    pub fn initial(&self, feature: Feature) -> f64 {
        match feature {
            Feature::Word(id) => self.words[id as usize].initial,
            Feature::Pos(a, b) => self.initial_pos[a as usize * self.pos_names.len() + b as usize],
            feature @ Feature::Bigram(..) => {
                self.initial_bigrams.get(&feature).copied().unwrap_or(0.0)
            }
            feature @ (Feature::PosWord(..) | Feature::WordPos(..))
                if self.cross_bigrams_enabled =>
            {
                self.initial_bigrams.get(&feature).copied().unwrap_or(0.0)
            }
            Feature::PosWord(..) | Feature::WordPos(..) => 0.0,
            Feature::ReadingLengthBigram(a, b) => {
                self.initial_reading_length_bigrams[usize::from(a) * 9 + usize::from(b)]
            }
            _ => 0.0,
        }
    }

    #[must_use]
    pub fn weight(&self, feature: Feature) -> f64 {
        if !self.cross_bigrams_enabled
            && matches!(feature, Feature::PosWord(..) | Feature::WordPos(..))
        {
            return 0.0;
        }
        self.initial(feature) + self.first.get(feature)
    }

    #[must_use]
    pub fn features(&self, graph: &Lattice, edge: Edge) -> [Option<Feature>; 6] {
        let a = &graph.nodes[edge.from];
        let b = &graph.nodes[edge.to];
        let source_pos = self.words[a.word as usize].pos;
        let target_pos = self.words[b.word as usize].pos;
        [
            Some(Feature::Word(b.word)),
            Some(Feature::Pos(source_pos, target_pos)),
            Some(Feature::Bigram(a.word, b.word)),
            self.cross_bigrams_enabled.then_some(Feature::PosWord(
                self.pos_token_ids[source_pos as usize],
                b.word,
            )),
            self.cross_bigrams_enabled.then_some(Feature::WordPos(
                a.word,
                self.pos_token_ids[target_pos as usize],
            )),
            Some(Feature::ReadingLengthBigram(
                self.words[a.word as usize].reading_length,
                self.words[b.word as usize].reading_length,
            )),
        ]
    }

    #[must_use]
    pub fn edge_score(&self, graph: &Lattice, edge: Edge) -> f64 {
        self.features(graph, edge)
            .into_iter()
            .flatten()
            .map(|f| self.weight(f))
            .sum()
    }

    #[must_use]
    pub fn path_score(&self, graph: &Lattice, path: &[usize]) -> f64 {
        path.windows(2)
            .map(|p| {
                self.edge_score(
                    graph,
                    Edge {
                        from: p[0],
                        to: p[1],
                    },
                )
            })
            .sum()
    }

    #[must_use]
    pub fn trigram_features(graph: &Lattice, path: &[usize]) -> TriGradient {
        let mut counts = BTreeMap::new();
        for triple in path.windows(3) {
            *counts
                .entry([
                    graph.nodes[triple[0]].word,
                    graph.nodes[triple[1]].word,
                    graph.nodes[triple[2]].word,
                ])
                .or_default() += 1.0;
        }
        counts
    }

    #[must_use]
    pub fn pos_trigram_features(graph: &Lattice, path: &[usize], model: &Self) -> PosTriGradient {
        let mut counts = BTreeMap::new();
        if !model.pos_trigrams_enabled {
            return counts;
        }
        for triple in path.windows(3) {
            let key = [
                u16::try_from(model.words[graph.nodes[triple[0]].word as usize].pos)
                    .expect("POS IDs were checked to fit in 16 bits"),
                u16::try_from(model.words[graph.nodes[triple[1]].word as usize].pos)
                    .expect("POS IDs were checked to fit in 16 bits"),
                u16::try_from(model.words[graph.nodes[triple[2]].word as usize].pos)
                    .expect("POS IDs were checked to fit in 16 bits"),
            ];
            *counts.entry(key).or_default() += 1.0;
        }
        counts
    }

    #[must_use]
    pub fn reading_length_trigram_features(
        graph: &Lattice,
        path: &[usize],
        model: &Self,
    ) -> ReadingLengthTriGradient {
        let mut counts = BTreeMap::new();
        for triple in path.windows(3) {
            let key = [
                model.words[graph.nodes[triple[0]].word as usize].reading_length,
                model.words[graph.nodes[triple[1]].word as usize].reading_length,
                model.words[graph.nodes[triple[2]].word as usize].reading_length,
            ];
            *counts.entry(key).or_default() += 1.0;
        }
        counts
    }

    #[must_use]
    pub fn trigram_score(&self, graph: &Lattice, path: &[usize]) -> f64 {
        let word = Self::trigram_features(graph, path)
            .into_iter()
            .map(|(key, count)| count * self.trigram_weight(key))
            .sum::<f64>();
        let pos = Self::pos_trigram_features(graph, path, self)
            .into_iter()
            .map(|(key, count)| count * self.pos_trigram_weight(key))
            .sum::<f64>();
        let reading_length = Self::reading_length_trigram_features(graph, path, self)
            .into_iter()
            .map(|(key, count)| count * self.reading_length_trigram_weight(key))
            .sum::<f64>();
        word + pos + reading_length
    }

    /// Full fixed-CRF candidate score plus the learned local-LLM feature.
    /// `mean_token_softmax` is the arithmetic mean of the target-token
    /// probabilities for the complete candidate text, scored from BOS and
    /// without EOS. It must be computed separately by the configured scorer.
    #[must_use]
    pub fn path_score_with_llm(
        &self,
        graph: &Lattice,
        path: &[usize],
        mean_token_softmax: f64,
    ) -> f64 {
        let crf = self.path_score(graph, path) + self.trigram_score(graph, path);
        let normalized_crf = if self.llm_crf_score_rms <= f64::EPSILON {
            0.0
        } else {
            crf / self.llm_crf_score_rms
        };
        let normalized_llm = if self.llm_feature_rms <= f64::EPSILON {
            0.0
        } else {
            mean_token_softmax / self.llm_feature_rms
        };
        normalized_crf + self.llm_alpha * normalized_llm
    }

    #[must_use]
    pub fn trigram_weight(&self, key: Triple) -> f64 {
        self.initial_trigrams.get(&key).copied().unwrap_or(0.0) + self.third.get(key)
    }

    #[must_use]
    pub fn pos_trigram_weight(&self, key: PosTriple) -> f64 {
        if !self.pos_trigrams_enabled {
            return 0.0;
        }
        self.initial_pos_trigrams.get(&key).copied().unwrap_or(0.0) + self.third_pos.get(key)
    }

    #[must_use]
    pub fn reading_length_trigram_weight(&self, key: ReadingLengthTriple) -> f64 {
        let index = (usize::from(key[0]) * 9 + usize::from(key[1])) * 9 + usize::from(key[2]);
        self.initial_reading_length_trigrams[index] + self.third_reading_length.get(key)
    }
}

fn logaddexp(a: f64, b: f64) -> f64 {
    let maximum = a.max(b);
    maximum + ((a - maximum).exp() + (b - maximum).exp()).ln()
}

fn interpolation_delta(base_log: f64, empirical_log: f64, empirical_weight: f64) -> f64 {
    let base_weight = 1.0 - empirical_weight;
    let mixture_log = match (base_weight, empirical_weight) {
        (0.0, _) => empirical_log,
        (_, 0.0) => base_log,
        (base_weight, empirical_weight) => logaddexp(
            base_weight.ln() + base_log,
            empirical_weight.ln() + empirical_log,
        ),
    };
    mixture_log - base_log
}

fn interpolation_delta_probability(
    base_probability: f64,
    empirical_probability: Option<f64>,
    empirical_weight: f64,
) -> f64 {
    let base_probability = base_probability.max(1e-300);
    let empirical_probability = empirical_probability
        .unwrap_or(base_probability)
        .max(1e-300);
    interpolation_delta(
        base_probability.ln(),
        empirical_probability.ln(),
        empirical_weight,
    )
}

fn reading_length_bucket(reading: &str, mecab_feature: &str) -> u8 {
    if reading.is_empty() || mecab_feature.starts_with("記号,") || mecab_feature == "記号" {
        return 0;
    }
    u8::try_from(
        reading
            .chars()
            .filter(|&ch| !matches!(ch, 'ゃ' | 'ゅ' | 'ょ'))
            .count()
            .min(8),
    )
    .expect("reading length is clamped to 8")
}

#[allow(clippy::cast_precision_loss)]
fn reading_length_base_probabilities(
    words: &[Word],
    pos_names: &[String],
    initial_pos: &[f64],
    counts: &TrainingFeatureCounts,
) -> Result<([f64; 81], [f64; 729])> {
    let pos_count = pos_names.len();
    if initial_pos.len() != pos_count * pos_count {
        return Err("POS matrix dimensions do not match reading-length baseline".into());
    }
    let mut word_length = vec![[1e-12_f64; 9]; pos_count];
    for word in words.iter().skip(3) {
        let row = word_length
            .get_mut(word.pos as usize)
            .ok_or("word POS is outside reading-length baseline")?;
        row[usize::from(word.reading_length)] += word.initial.exp();
    }
    // BOS/EOS are deterministic boundary classes with read-length zero.
    for boundary in [BOS, EOS] {
        let pos = words[boundary as usize].pos as usize;
        word_length[pos] = [1e-12; 9];
        word_length[pos][0] = 1.0;
    }
    for distribution in &mut word_length {
        let total: f64 = distribution.iter().sum();
        for value in distribution {
            *value /= total;
        }
    }

    let mut transitions = vec![vec![0.0_f64; pos_count]; pos_count];
    for a in 1..pos_count {
        let mut total = 0.0;
        for b in 1..pos_count {
            let probability = initial_pos[a * pos_count + b].exp();
            transitions[a][b] = probability;
            total += probability;
        }
        if total <= 0.0 || !total.is_finite() {
            return Err("invalid POS transition row in reading-length baseline".into());
        }
        for transition in transitions[a].iter_mut().skip(1) {
            *transition /= total;
        }
    }
    let mut prior = vec![1e-12_f64; pos_count];
    for (pos, &count) in counts.pos_context_counts.iter().enumerate().skip(1) {
        if let Some(weight) = prior.get_mut(pos) {
            *weight += count as f64;
        }
    }
    let prior_total: f64 = prior.iter().skip(1).sum();
    if prior_total <= 0.0 || !prior_total.is_finite() {
        let uniform = 1.0 / (pos_count - 1) as f64;
        prior
            .iter_mut()
            .skip(1)
            .for_each(|weight| *weight = uniform);
    } else {
        for weight in prior.iter_mut().skip(1) {
            *weight /= prior_total;
        }
    }

    let mut forward_by_pos = vec![[0.0_f64; 9]; pos_count];
    for a in 1..pos_count {
        for b in 1..pos_count {
            for la in 0..9 {
                forward_by_pos[b][la] += prior[a] * word_length[a][la] * transitions[a][b];
            }
        }
    }
    let mut next_length_by_pos = vec![[0.0_f64; 9]; pos_count];
    for b in 1..pos_count {
        for c in 1..pos_count {
            for lc in 0..9 {
                next_length_by_pos[b][lc] += transitions[b][c] * word_length[c][lc];
            }
        }
    }

    let mut bi_joint = [0.0_f64; 81];
    let mut tri_joint = [0.0_f64; 729];
    for b in 1..pos_count {
        for la in 0..9 {
            for lb in 0..9 {
                let prefix = forward_by_pos[b][la] * word_length[b][lb];
                bi_joint[la * 9 + lb] += prefix;
                for lc in 0..9 {
                    tri_joint[(la * 9 + lb) * 9 + lc] += prefix * next_length_by_pos[b][lc];
                }
            }
        }
    }
    let mut bi = [0.0_f64; 81];
    for la in 0..9 {
        let total: f64 = bi_joint[la * 9..la * 9 + 9].iter().sum();
        for lb in 0..9 {
            bi[la * 9 + lb] = bi_joint[la * 9 + lb] / total.max(1e-300);
        }
    }
    let mut tri = [0.0_f64; 729];
    for context in 0..81 {
        let offset = context * 9;
        let total: f64 = tri_joint[offset..offset + 9].iter().sum();
        for lc in 0..9 {
            tri[offset + lc] = tri_joint[offset + lc] / total.max(1e-300);
        }
    }
    Ok((bi, tri))
}

#[derive(Clone, Debug)]
pub struct Node {
    pub start: usize,
    pub end: usize,
    pub word: u32,
    pub script: u8,
}

#[derive(Clone, Copy, Debug)]
pub struct Edge {
    pub from: usize,
    pub to: usize,
}

#[derive(Clone, Debug)]
pub struct Lattice {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
    pub incoming: Vec<Vec<usize>>,
    pub outgoing: Vec<Vec<usize>>,
    pub gold: Vec<usize>,
}

impl Lattice {
    /// Nodes must be in topological order, BOS first and EOS last.
    pub fn new(nodes: Vec<Node>, gold: Vec<usize>, edge_limit: usize) -> Result<Self> {
        if nodes.len() < 2 || nodes[0].word != BOS || nodes.last().map(|n| n.word) != Some(EOS) {
            return Err("lattice boundaries must be BOS/EOS".into());
        }
        let mut starts = BTreeMap::<usize, Vec<usize>>::new();
        for (index, node) in nodes.iter().enumerate().skip(1) {
            starts.entry(node.start).or_default().push(index);
        }
        let mut graph = Self {
            incoming: vec![vec![]; nodes.len()],
            outgoing: vec![vec![]; nodes.len()],
            nodes,
            edges: vec![],
            gold,
        };
        for a in 0..graph.nodes.len() - 1 {
            for &b in starts.get(&graph.nodes[a].end).into_iter().flatten() {
                if b <= a {
                    return Err("lattice is not an acyclic topologically ordered graph".into());
                }
                if graph.edges.len() >= edge_limit {
                    return Err("lattice edge budget exceeded".into());
                }
                let index = graph.edges.len();
                graph.edges.push(Edge { from: a, to: b });
                graph.outgoing[a].push(index);
                graph.incoming[b].push(index);
            }
        }
        if graph.gold.first() != Some(&0) || graph.gold.last() != Some(&(graph.nodes.len() - 1)) {
            return Err("invalid gold boundaries".into());
        }
        for pair in graph.gold.windows(2) {
            if !graph.outgoing[pair[0]]
                .iter()
                .any(|&e| graph.edges[e].to == pair[1])
            {
                return Err("gold path is absent from lattice".into());
            }
        }
        Ok(graph)
    }
}

#[derive(Clone, Debug)]
pub struct FirstOrder {
    pub loss: f64,
    pub log_z: f64,
    /// Gradient of negative log likelihood: expected count minus gold count.
    pub gradient: Gradient,
    pub edge_marginals: Vec<f64>,
}

#[must_use]
pub fn log_add(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let high = a.max(b);
    high + (-(a - b).abs()).exp().ln_1p()
}

/// Exact first-order forward-backward over the entire word-segmentation DAG.
pub fn first_order(model: &Model, graph: &Lattice) -> Result<FirstOrder> {
    let scores = graph
        .edges
        .iter()
        .map(|&e| model.edge_score(graph, e))
        .collect::<Vec<_>>();
    let last = graph.nodes.len() - 1;
    let mut alpha = vec![f64::NEG_INFINITY; graph.nodes.len()];
    alpha[0] = 0.0;
    for node in 1..graph.nodes.len() {
        for &e in &graph.incoming[node] {
            alpha[node] = log_add(alpha[node], alpha[graph.edges[e].from] + scores[e]);
        }
    }
    let log_z = alpha[last];
    if !log_z.is_finite() {
        return Err("no finite BOS-to-EOS path".into());
    }
    let mut beta = vec![f64::NEG_INFINITY; graph.nodes.len()];
    beta[last] = 0.0;
    for node in (0..last).rev() {
        for &e in &graph.outgoing[node] {
            beta[node] = log_add(beta[node], scores[e] + beta[graph.edges[e].to]);
        }
    }
    let mut gradient = Gradient::new();
    let mut edge_marginals = Vec::with_capacity(scores.len());
    for (e, edge) in graph.edges.iter().enumerate() {
        let probability = (alpha[edge.from] + scores[e] + beta[edge.to] - log_z).exp();
        edge_marginals.push(probability);
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
    let loss = log_z - model.path_score(graph, &graph.gold);
    if !loss.is_finite() || loss < -1e-8 {
        return Err("invalid first-order likelihood".into());
    }
    Ok(FirstOrder {
        loss: loss.max(0.0),
        log_z,
        gradient,
        edge_marginals,
    })
}

/// Exact data NLL for evaluation, without backward/marginal/gradient allocation.
pub fn first_nll(model: &Model, graph: &Lattice) -> Result<f64> {
    let mut alpha = vec![f64::NEG_INFINITY; graph.nodes.len()];
    alpha[0] = 0.0;
    for node in 1..graph.nodes.len() {
        for &e in &graph.incoming[node] {
            alpha[node] = log_add(
                alpha[node],
                alpha[graph.edges[e].from] + model.edge_score(graph, graph.edges[e]),
            );
        }
    }
    let loss = alpha[graph.nodes.len() - 1] - model.path_score(graph, &graph.gold);
    if !loss.is_finite() || loss < -1e-8 {
        return Err("invalid first-order likelihood".into());
    }
    Ok(loss.max(0.0))
}

#[derive(Clone, Debug)]
pub struct Path {
    pub nodes: Vec<usize>,
    pub score: f64,
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

/// Reverse Viterbi supplies an exact admissible A* upper bound. No beam pruning.
pub fn nbest(
    model: &Model,
    graph: &Lattice,
    count: usize,
    expansion_limit: usize,
) -> Result<Vec<Path>> {
    if count == 0 {
        return Err("N-best size must be positive".into());
    }
    let scores = graph
        .edges
        .iter()
        .map(|&e| model.edge_score(graph, e))
        .collect::<Vec<_>>();
    let last = graph.nodes.len() - 1;
    let mut suffix = vec![f64::NEG_INFINITY; graph.nodes.len()];
    suffix[last] = 0.0;
    for node in (0..last).rev() {
        for &e in &graph.outgoing[node] {
            suffix[node] = suffix[node].max(scores[e] + suffix[graph.edges[e].to]);
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
    let mut results = Vec::new();
    let mut expansions = 0;
    while let Some(state) = agenda.pop() {
        let node = *state.path.nodes.last().ok_or("empty A* path")?;
        if node == last {
            results.push(state.path);
            if results.len() == count {
                break;
            }
            continue;
        }
        expansions += 1;
        if expansions > expansion_limit {
            return Err("A* expansion budget exceeded (not silently beam-pruned)".into());
        }
        for &e in &graph.outgoing[node] {
            let target = graph.edges[e].to;
            if !suffix[target].is_finite() {
                continue;
            }
            let mut path = state.path.clone();
            path.nodes.push(target);
            path.score += scores[e];
            agenda.push(Agenda {
                bound: path.score + suffix[target],
                path,
            });
        }
    }
    Ok(results)
}

#[derive(Clone, Debug)]
pub struct Rerank {
    pub loss: Option<f64>,
    pub gradient: TriGradient,
    pub pos_gradient: PosTriGradient,
    pub reading_length_gradient: ReadingLengthTriGradient,
    pub best: usize,
    pub scores: Vec<f64>,
}

/// Conditional log-linear loss over the *actual* N-best set. Gold is not injected.
/// Only trigram weights receive this gradient; first-order scores are offsets.
pub fn rerank(model: &Model, graph: &Lattice, paths: &[Path]) -> Result<Rerank> {
    if paths.is_empty() {
        return Err("empty N-best set".into());
    }
    let mut seen = std::collections::BTreeSet::new();
    let mut scores = Vec::new();
    let mut best = 0;
    let mut log_z = f64::NEG_INFINITY;
    let mut gold_index = None;
    for (index, path) in paths.iter().enumerate() {
        if !seen.insert(&path.nodes) {
            return Err("duplicate N-best path".into());
        }
        // Recompute, rather than trust a score cached before a parameter update.
        let score = model.path_score(graph, &path.nodes) + model.trigram_score(graph, &path.nodes);
        if !score.is_finite() {
            return Err("nonfinite reranking score".into());
        }
        if index == 0 || score > scores[best] {
            best = index;
        }
        if path.nodes == graph.gold {
            gold_index = Some(index);
        }
        log_z = log_add(log_z, score);
        scores.push(score);
    }
    let mut gradient = TriGradient::new();
    let mut pos_gradient = PosTriGradient::new();
    let mut reading_length_gradient = ReadingLengthTriGradient::new();
    let loss = if let Some(gold_index) = gold_index {
        for (path, score) in paths.iter().zip(&scores) {
            let probability = (*score - log_z).exp();
            for (key, count) in Model::trigram_features(graph, &path.nodes) {
                *gradient.entry(key).or_default() += probability * count;
            }
            for (key, count) in Model::pos_trigram_features(graph, &path.nodes, model) {
                *pos_gradient.entry(key).or_default() += probability * count;
            }
            for (key, count) in Model::reading_length_trigram_features(graph, &path.nodes, model) {
                *reading_length_gradient.entry(key).or_default() += probability * count;
            }
        }
        for (key, count) in Model::trigram_features(graph, &graph.gold) {
            *gradient.entry(key).or_default() -= count;
        }
        for (key, count) in Model::pos_trigram_features(graph, &graph.gold, model) {
            *pos_gradient.entry(key).or_default() -= count;
        }
        for (key, count) in Model::reading_length_trigram_features(graph, &graph.gold, model) {
            *reading_length_gradient.entry(key).or_default() -= count;
        }
        Some((log_z - scores[gold_index]).max(0.0))
    } else {
        None
    };
    Ok(Rerank {
        loss,
        gradient,
        pos_gradient,
        reading_length_gradient,
        best,
        scores,
    })
}

#[derive(Clone, Debug)]
pub struct UpdateReport {
    pub first_loss_before: f64,
    pub first_loss_after: f64,
    pub rerank_loss_before: Option<f64>,
    pub rerank_loss_after: Option<f64>,
    pub gold_in_nbest: bool,
}

#[derive(Debug)]
pub struct ParameterLimit {
    pub needed_bytes: usize,
    pub limit_bytes: usize,
    /// Conservative number of nonzero gradient keys that may need slots after
    /// a cutoff removes keys also touched by the pending batch.
    pub additional_first_bound: usize,
    pub additional_third_bound: usize,
    pub additional_pos_third_bound: usize,
    pub additional_reading_length_third_bound: usize,
    pub allocation_error: Option<String>,
}
impl std::fmt::Display for ParameterLimit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "sparse parameter capacity requires {} bytes, budget is {} bytes",
            self.needed_bytes, self.limit_bytes
        )?;
        if let Some(error) = &self.allocation_error {
            write!(f, "; allocator rejected reservation: {error}")?;
        }
        Ok(())
    }
}
impl std::error::Error for ParameterLimit {}

#[derive(Debug)]
pub struct OnlineReport {
    pub first_loss: f64,
    pub rerank_loss: Option<f64>,
}

/// Exact same updates as `train_sentence`, without per-sentence post-update diagnostics.
pub fn train_online(
    model: &mut Model,
    graph: &Lattice,
    rate: f64,
    l2: f64,
    n: usize,
    cap: usize,
    parameter_bytes: usize,
) -> Result<OnlineReport> {
    let first = first_order(model, graph)?;
    let new_first = first
        .gradient
        .iter()
        .filter(|(k, v)| v.abs() > 1e-12 && !model.first.values.contains_key(k))
        .count();
    // Positive-width reading nodes: at most input byte length words per path.
    // A conservative reservation bounds the POS/read-length N-best triple unions.
    let new_third_bound = n
        .checked_mul(graph.nodes.last().ok_or("missing EOS")?.end)
        .ok_or("trigram reservation overflow")?;
    let needed_bytes = model
        .first
        .values
        .projected_bytes(new_first)
        .saturating_add(model.third.values.projected_bytes(new_third_bound))
        .saturating_add(model.third_pos.values.projected_bytes(new_third_bound))
        .saturating_add(
            model
                .third_reading_length
                .values
                .projected_bytes(new_third_bound),
        );
    if needed_bytes > parameter_bytes {
        return Err(Box::new(ParameterLimit {
            needed_bytes,
            limit_bytes: parameter_bytes,
            additional_first_bound: new_first,
            additional_third_bound: 0,
            additional_pos_third_bound: new_third_bound,
            additional_reading_length_third_bound: new_third_bound,
            allocation_error: None,
        }));
    }
    model.first.values.reserve(new_first)?;
    model.third.values.reserve(new_third_bound)?;
    model.third_pos.values.reserve(new_third_bound)?;
    model.third_reading_length.values.reserve(new_third_bound)?;
    model.first.update(&first.gradient, rate, l2)?;
    let paths = nbest(model, graph, n, cap)?;
    let before = rerank(model, graph, &paths)?;
    if before.loss.is_some() {
        model.third.update(&before.gradient, rate, l2)?;
        model.third_pos.update(&before.pos_gradient, rate, l2)?;
        model
            .third_reading_length
            .update(&before.reading_length_gradient, rate, l2)?;
    }
    Ok(OnlineReport {
        first_loss: first.loss,
        rerank_loss: before.loss,
    })
}

pub fn train_sentence(
    model: &mut Model,
    graph: &Lattice,
    rate: f64,
    l2: f64,
    n: usize,
    cap: usize,
) -> Result<UpdateReport> {
    let first = first_order(model, graph)?;
    model.first.update(&first.gradient, rate, l2)?;
    let first_after = first_order(model, graph)?.loss;
    let paths = nbest(model, graph, n, cap)?;
    let before = rerank(model, graph, &paths)?;
    if before.loss.is_some() {
        model.third.update(&before.gradient, rate, l2)?;
        model.third_pos.update(&before.pos_gradient, rate, l2)?;
        model
            .third_reading_length
            .update(&before.reading_length_gradient, rate, l2)?;
    }
    let after = rerank(model, graph, &paths)?;
    Ok(UpdateReport {
        first_loss_before: first.loss,
        first_loss_after: first_after,
        rerank_loss_before: before.loss,
        rerank_loss_after: after.loss,
        gold_in_nbest: before.loss.is_some(),
    })
}

/// A mutually exclusive *surface* shape, including mixed scripts and boundaries.
#[must_use]
pub fn script(surface: &str) -> u8 {
    let mut category = None;
    for ch in surface.chars() {
        let current = match ch {
            '\u{3041}'..='\u{309f}' => 2,
            '\u{30a0}'..='\u{30ff}' | '\u{ff66}'..='\u{ff9f}' => 3,
            '\u{3400}'..='\u{4dbf}'
            | '\u{4e00}'..='\u{9fff}'
            | '\u{20000}'..='\u{323af}'
            | '々'
            | '〆' => 4,
            'a'..='z' | 'A'..='Z' | '\u{ff21}'..='\u{ff3a}' | '\u{ff41}'..='\u{ff5a}' => 6,
            _ if ch.is_numeric() => 7,
            _ if ch.is_whitespace() || !ch.is_alphanumeric() => 5,
            _ => 9,
        };
        // A prolonged-sound mark is compatible with either kana family.
        if ch == 'ー' && matches!(category, Some(2 | 3)) {
            continue;
        }
        if let Some(previous) = category {
            if previous != current {
                return 8;
            }
        } else {
            category = Some(current);
        }
    }
    category.unwrap_or(9)
}

#[cfg(test)]
mod tests;
