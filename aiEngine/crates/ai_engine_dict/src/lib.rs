//! Dictionary compiler and shared, unquantized training seeds.

mod dict_char;
mod sdict;

use std::{error::Error, path::Path};

pub const DEFAULT_NGRAM_CUTOFF: u64 = sdict::DEFAULT_NGRAM_CUTOFF;

/// A source word. IDs are training-local, not `.sdict` byte offsets.
#[derive(Clone, Debug)]
pub struct TrainingWord {
    pub surface: String,
    pub reading: String,
    pub pos_id: u32,
    pub log_unigram: f64,
    pub model_only: bool,
}

#[derive(Clone, Debug)]
pub struct TrainingSeeds {
    pub words: Vec<TrainingWord>,
    /// ID zero is reserved; name at index zero has POS ID one.
    pub pos_names: Vec<String>,
    pub pos_bigrams: Vec<(u32, u32, f64)>,
}

/// A count-ranked corpus word bigram before conversion to an additive CRF score.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TrainingBigram {
    /// Index into `TrainingSeeds::words`.
    pub source: u32,
    /// Index into `TrainingSeeds::words`.
    pub target: u32,
    pub count: u64,
    /// Total outgoing bigram count for `source`, including pairs below the dictionary cutoff.
    pub outgoing_count: u64,
}

/// A count-ranked corpus word trigram before conversion to an additive CRF score.
/// IDs index `TrainingSeeds::words`; all denominators use the same merged source IDs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TrainingTrigram {
    pub first: u32,
    pub middle: u32,
    pub target: u32,
    pub count: u64,
    pub parent_count: u64,
}

/// Count-ranked initial POS-token → word bigram. `source_pos` is encoded as a
/// dedicated synthetic word ID by the CRF model; the denominator is the full
/// outgoing corpus count for that POS.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TrainingPosWordBigram {
    pub source_pos: u32,
    pub target_word: u32,
    pub count: u64,
}

/// Count-ranked initial word → POS-token bigram. The denominator is the source
/// word's full outgoing corpus count.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TrainingWordPosBigram {
    pub source_word: u32,
    pub target_pos: u32,
    pub count: u64,
    pub outgoing_count: u64,
}

/// One affix lexeme promoted to a lexeme-specific refined POS class.
#[derive(Clone, Debug, Eq, Hash, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct TrainingAffixPosSelection {
    pub surface: String,
    pub reading: String,
    pub original_pos: String,
    pub assigned_pos: String,
    /// Raw corpus unigram count (not P(word | POS)).
    pub unigram_count: u64,
}

pub type TrainingNgrams = (TrainingSeeds, Vec<TrainingBigram>, Vec<TrainingTrigram>);

/// Counted POS-trigram seed. IDs are local POS IDs from `TrainingSeeds`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TrainingPosTrigram {
    pub first: u16,
    pub middle: u16,
    pub target: u16,
    pub count: u64,
    pub context_count: u64,
}

/// Full-corpus counts needed to seed POS/read-length features. Read-length IDs
/// are 0..=8 (8 means eight or more); arrays are indexed lexicographically.
#[derive(Clone, Debug)]
pub struct TrainingFeatureCounts {
    pub pos_context_counts: Vec<u64>,
    pub pos_trigrams: Vec<TrainingPosTrigram>,
    pub include_pos_trigrams: bool,
    pub reading_length_bigram_counts: [u64; 81],
    pub reading_length_bigram_context_counts: [u64; 9],
    pub reading_length_trigram_counts: [u64; 729],
    pub reading_length_trigram_context_counts: [u64; 81],
}

impl Default for TrainingFeatureCounts {
    fn default() -> Self {
        Self {
            pos_context_counts: Vec::new(),
            pos_trigrams: Vec::new(),
            include_pos_trigrams: true,
            reading_length_bigram_counts: [0; 81],
            reading_length_bigram_context_counts: [0; 9],
            reading_length_trigram_counts: [0; 729],
            reading_length_trigram_context_counts: [0; 81],
        }
    }
}

#[derive(Debug)]
pub struct TrainingCrfNgrams {
    pub seeds: TrainingSeeds,
    pub bigrams: Vec<TrainingBigram>,
    pub trigrams: Vec<TrainingTrigram>,
    pub pos_word_bigrams: Vec<TrainingPosWordBigram>,
    pub word_pos_bigrams: Vec<TrainingWordPosBigram>,
    pub affix_pos_selections: Vec<TrainingAffixPosSelection>,
    pub feature_counts: TrainingFeatureCounts,
}

/// Read the complete source vocabulary and corpus counts before quantization.
/// POS transitions use *all* raw word bigrams, not the cutoff table.
///
/// # Errors
/// Returns an error for missing, malformed, or inconsistent source files.
pub fn training_seeds(
    corpus: &Path,
    system: &Path,
    additions: &Path,
) -> Result<TrainingSeeds, Box<dyn Error>> {
    sdict::training_seeds(corpus, system, additions)
}

/// Read ordinary POS/word seeds and optionally return the most frequent retained word bigrams.
///
/// Bigram rows are mapped through the exact same source vocabulary and normalized word IDs as
/// the training seeds. `limit == 0` avoids reading the retained bigram table.
///
/// # Errors
/// Returns an error for missing, malformed, or inconsistent source files.
pub fn training_seeds_with_bigrams(
    corpus: &Path,
    system: &Path,
    additions: &Path,
    limit: usize,
) -> Result<(TrainingSeeds, Vec<TrainingBigram>), Box<dyn Error>> {
    let (seeds, bigrams, _) =
        sdict::training_seeds_with_ngrams(corpus, system, additions, limit, 0)?;
    Ok((seeds, bigrams))
}

/// Read model seeds and the highest-count retained word bigrams/trigrams.
/// Trigram probabilities are later interpolated against their `middle -> target`
/// bigram probabilities by the CRF model builder.
///
/// # Errors
/// Returns an error for missing, malformed, or inconsistent source files.
pub fn training_seeds_with_ngrams(
    corpus: &Path,
    system: &Path,
    additions: &Path,
    bigram_limit: usize,
    trigram_limit: usize,
) -> Result<TrainingNgrams, Box<dyn Error>> {
    sdict::training_seeds_with_ngrams(corpus, system, additions, bigram_limit, trigram_limit)
}

/// Build the same word/POS seeds plus full-corpus cross-bigram and reading-length n-gram counts.
/// POS trigrams are deliberately disabled for the current CRF experiment; only word
/// bigram/trigram and cross-bigram seed tables are frequency-limited.
///
/// # Errors
/// Returns an error for missing, malformed, or inconsistent source files.
pub fn training_seeds_with_crf_features(
    corpus: &Path,
    system: &Path,
    additions: &Path,
    bigram_limit: usize,
    trigram_limit: usize,
) -> Result<TrainingCrfNgrams, Box<dyn Error>> {
    sdict::training_seeds_with_crf_features(corpus, system, additions, bigram_limit, trigram_limit)
}

#[must_use]
pub fn model_pos(surface: &str, feature: &str) -> String {
    sdict::model_pos_for_word(surface, feature)
}

#[must_use]
pub fn normalize_surface(surface: &str) -> String {
    sdict::normalize_source_surface(surface)
}

#[must_use]
pub fn normalize_reading(reading: &str) -> String {
    sdict::to_hiragana(reading)
}

/// Compile the statistical dictionary (not a CRF model).
///
/// # Errors
/// Returns an error for invalid inputs or I/O failures.
pub fn compile(
    input: &Path,
    output: &Path,
    system: &Path,
    additions: &Path,
    cutoff: u64,
) -> Result<(), Box<dyn Error>> {
    sdict::compile(input, output, system, additions, cutoff)
}

/// # Errors
/// Returns an error for invalid containers or I/O failures.
pub fn dump(input: &Path, output: &Path) -> Result<(), Box<dyn Error>> {
    sdict::dump(input, output)
}
