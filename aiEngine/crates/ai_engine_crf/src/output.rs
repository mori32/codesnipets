//! Unquantized training checkpoints and compiler-facing text potentials.
use std::fs::{self, File, OpenOptions};
use std::hash::{Hash, Hasher};
use std::io::{BufReader, BufWriter, Write};
use std::marker::PhantomData;
use std::path::Path;

use serde::ser::SerializeSeq;
use serde::{Deserialize, Serialize};

use crate::sparse::{IntegerHasher, Key, SparseMap};
use crate::{
    Feature, Model, Parameters, PosTriple, ReadingLengthTriple, Result, SCRIPT_NAMES, Triple, Word,
};

#[derive(Debug, Deserialize)]
struct Snapshot {
    version: u32,
    words: Vec<Word>,
    pos_names: Vec<String>,
    #[serde(default)]
    pos_token_ids: Vec<u32>,
    #[serde(default)]
    affix_pos_selections: Vec<ai_engine_dict::TrainingAffixPosSelection>,
    #[serde(default = "enabled_by_default")]
    cross_bigrams_enabled: bool,
    #[serde(default = "enabled_by_default")]
    pos_trigrams_enabled: bool,
    initial_pos: Vec<f64>,
    #[serde(default)]
    initial_bigram_values: Decoded<Feature>,
    #[serde(default)]
    initial_trigram_values: Decoded<Triple>,
    #[serde(default)]
    initial_pos_trigram_values: Decoded<PosTriple>,
    #[serde(default = "zero_reading_length_bigrams")]
    initial_reading_length_bigrams: Vec<f64>,
    #[serde(default = "zero_reading_length_trigrams")]
    initial_reading_length_trigrams: Vec<f64>,
    missing_pos_log: f64,
    first_scale: f64,
    first_values: Decoded<Feature>,
    third_scale: f64,
    third_values: Decoded<Triple>,
    #[serde(default = "one_scale")]
    third_pos_scale: f64,
    #[serde(default)]
    third_pos_values: Decoded<PosTriple>,
    #[serde(default = "one_scale")]
    third_reading_length_scale: f64,
    #[serde(default)]
    third_reading_length_values: Decoded<ReadingLengthTriple>,
    #[serde(default)]
    llm_alpha: f64,
    #[serde(default = "one_scale")]
    llm_crf_score_rms: f64,
    #[serde(default = "one_scale")]
    llm_feature_rms: f64,
}

fn zero_reading_length_bigrams() -> Vec<f64> {
    vec![0.0; 81]
}

fn zero_reading_length_trigrams() -> Vec<f64> {
    vec![0.0; 729]
}

fn one_scale() -> f64 {
    1.0
}

fn enabled_by_default() -> bool {
    true
}

trait SavedKey: Key {
    fn valid(self) -> bool;
}
impl SavedKey for Feature {
    fn valid(self) -> bool {
        match self {
            Self::Word(w) => w <= crate::sparse::MAX_ID,
            Self::Pos(a, b) | Self::Bigram(a, b) | Self::PosWord(a, b) | Self::WordPos(a, b) => {
                a <= crate::sparse::MAX_ID && b <= crate::sparse::MAX_ID
            }
            Self::Script(s) => (s as usize) < SCRIPT_NAMES.len(),
            Self::ScriptBigram(a, b) => {
                (a as usize) < SCRIPT_NAMES.len() && (b as usize) < SCRIPT_NAMES.len()
            }
            Self::ReadingLengthBigram(a, b) => a < 9 && b < 9,
        }
    }
}
impl SavedKey for Triple {
    fn valid(self) -> bool {
        self.into_iter().all(|w| w <= crate::sparse::MAX_ID)
    }
}
impl SavedKey for PosTriple {
    fn valid(self) -> bool {
        self.into_iter().all(|pos| pos != 0)
    }
}
impl SavedKey for ReadingLengthTriple {
    fn valid(self) -> bool {
        self.into_iter().all(|length| length < 9)
    }
}

/// Decode each entry directly into its destination table; no full Vec of
/// weights coexists with the loaded hash table.
#[derive(Debug)]
struct Decoded<K: Key>(SparseMap<K>);
impl<K: Key> Default for Decoded<K> {
    fn default() -> Self {
        Self(SparseMap::default())
    }
}
impl<'de, K: SavedKey + Deserialize<'de>> Deserialize<'de> for Decoded<K> {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        struct Visitor<K>(PhantomData<K>);
        impl<'de, K: SavedKey + Deserialize<'de>> serde::de::Visitor<'de> for Visitor<K> {
            type Value = Decoded<K>;
            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("an array of distinct finite correction weights")
            }
            fn visit_seq<A: serde::de::SeqAccess<'de>>(
                self,
                mut sequence: A,
            ) -> std::result::Result<Self::Value, A::Error> {
                let mut table = SparseMap::default();
                while let Some((key, value)) = sequence.next_element::<(K, f64)>()? {
                    // Reject oversized IDs before packing, not by a pack assert.
                    if !key.valid() || !value.is_finite() {
                        return Err(serde::de::Error::custom("invalid checkpoint key/value"));
                    }
                    if table.insert(key, value).is_some() {
                        return Err(serde::de::Error::custom("duplicate checkpoint feature"));
                    }
                    if table.len() % 10_000_000 == 0 {
                        #[allow(clippy::cast_precision_loss)]
                        let mib = table.projected_bytes(0) as f64 / f64::from(1 << 20);
                        eprintln!(
                            "checkpoint family={} loaded_keys={} table_mib={mib:.1}",
                            std::any::type_name::<K>(),
                            table.len()
                        );
                    }
                }
                Ok(Decoded(table))
            }
        }
        deserializer.deserialize_seq(Visitor(PhantomData))
    }
}

#[derive(Debug)]
struct Entries<'a, K: Key>(&'a SparseMap<K>);
impl<K: Key + Serialize> Serialize for Entries<'_, K> {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        // Map entry order is not part of the checkpoint format. Stream directly
        // from the HashMap instead of duplicating a potentially multi-GB table
        // into a sorted Vec before serialization.
        let mut sequence = serializer.serialize_seq(Some(self.0.len()))?;
        for (key, value) in self.0.iter_raw() {
            sequence.serialize_element(&(K::unpack(key), value))?;
        }
        sequence.end()
    }
}

#[derive(Debug, Serialize)]
struct SnapshotView<'a> {
    version: u32,
    words: &'a [Word],
    pos_names: &'a [String],
    pos_token_ids: &'a [u32],
    affix_pos_selections: &'a [ai_engine_dict::TrainingAffixPosSelection],
    cross_bigrams_enabled: bool,
    pos_trigrams_enabled: bool,
    initial_pos: &'a [f64],
    initial_bigram_values: Entries<'a, Feature>,
    initial_trigram_values: Entries<'a, Triple>,
    initial_pos_trigram_values: Entries<'a, PosTriple>,
    initial_reading_length_bigrams: &'a [f64],
    initial_reading_length_trigrams: &'a [f64],
    missing_pos_log: f64,
    first_scale: f64,
    first_values: Entries<'a, Feature>,
    third_scale: f64,
    third_values: Entries<'a, Triple>,
    third_pos_scale: f64,
    third_pos_values: Entries<'a, PosTriple>,
    third_reading_length_scale: f64,
    third_reading_length_values: Entries<'a, ReadingLengthTriple>,
    llm_alpha: f64,
    llm_crf_score_rms: f64,
    llm_feature_rms: f64,
}

#[derive(Debug, Eq, PartialEq, Serialize)]
pub struct Fingerprint {
    metadata: u64,
    first: (u64, u64),
    third: (u64, u64),
    third_pos: (u64, u64),
    third_reading_length: (u64, u64),
}

/// All metadata and raw weight bits; diagnostic, not a cryptographic file hash.
#[must_use]
pub fn fingerprint(model: &Model) -> Fingerprint {
    let mut h = IntegerHasher::default();
    model.words.len().hash(&mut h);
    model.pos_names.hash(&mut h);
    model.pos_token_ids.hash(&mut h);
    model.affix_pos_selections.hash(&mut h);
    model.cross_bigrams_enabled.hash(&mut h);
    model.pos_trigrams_enabled.hash(&mut h);
    model.llm_alpha.to_bits().hash(&mut h);
    model.llm_crf_score_rms.to_bits().hash(&mut h);
    model.llm_feature_rms.to_bits().hash(&mut h);
    for word in &model.words {
        word.surface.hash(&mut h);
        word.reading.hash(&mut h);
        word.pos.hash(&mut h);
        word.initial.to_bits().hash(&mut h);
        word.model_only.hash(&mut h);
        word.reading_length.hash(&mut h);
    }
    for value in &model.initial_pos {
        value.to_bits().hash(&mut h);
    }
    if !model.initial_bigrams.is_empty() {
        0x494e_4954_4249_u64.hash(&mut h);
        model.initial_bigrams.len().hash(&mut h);
        for (key, value) in model.initial_bigrams.sorted_raw() {
            key.hash(&mut h);
            value.to_bits().hash(&mut h);
        }
    }
    if !model.initial_trigrams.is_empty() {
        0x0049_4e49_5454_5249_u64.hash(&mut h);
        model.initial_trigrams.len().hash(&mut h);
        for (key, value) in model.initial_trigrams.sorted_raw() {
            key.hash(&mut h);
            value.to_bits().hash(&mut h);
        }
    }
    for value in model.initial_reading_length_bigrams {
        value.to_bits().hash(&mut h);
    }
    for value in model.initial_reading_length_trigrams {
        value.to_bits().hash(&mut h);
    }
    for (key, value) in model.initial_pos_trigrams.sorted_raw() {
        key.hash(&mut h);
        value.to_bits().hash(&mut h);
    }
    model.missing_pos_log.to_bits().hash(&mut h);
    model.first.scale.to_bits().hash(&mut h);
    model.third.scale.to_bits().hash(&mut h);
    model.third_pos.scale.to_bits().hash(&mut h);
    model.third_reading_length.scale.to_bits().hash(&mut h);
    model.first.values.len().hash(&mut h);
    model.third.values.len().hash(&mut h);
    model.third_pos.values.len().hash(&mut h);
    model.third_reading_length.values.len().hash(&mut h);
    Fingerprint {
        metadata: h.finish(),
        first: model.first.values.fingerprint(),
        third: model.third.values.fingerprint(),
        third_pos: model.third_pos.values.fingerprint(),
        third_reading_length: model.third_reading_length.values.fingerprint(),
    }
}

fn create(path: &Path) -> Result<BufWriter<File>> {
    Ok(BufWriter::new(
        OpenOptions::new().write(true).create_new(true).open(path)?,
    ))
}

pub fn save(model: &Model, path: &Path) -> Result<()> {
    let snapshot = SnapshotView {
        version: 9,
        words: &model.words,
        pos_names: &model.pos_names,
        pos_token_ids: &model.pos_token_ids,
        affix_pos_selections: &model.affix_pos_selections,
        cross_bigrams_enabled: model.cross_bigrams_enabled,
        pos_trigrams_enabled: model.pos_trigrams_enabled,
        initial_pos: &model.initial_pos,
        initial_bigram_values: Entries(&model.initial_bigrams),
        initial_trigram_values: Entries(&model.initial_trigrams),
        initial_pos_trigram_values: Entries(&model.initial_pos_trigrams),
        initial_reading_length_bigrams: &model.initial_reading_length_bigrams,
        initial_reading_length_trigrams: &model.initial_reading_length_trigrams,
        missing_pos_log: model.missing_pos_log,
        first_scale: model.first.scale,
        first_values: Entries(&model.first.values),
        third_scale: model.third.scale,
        third_values: Entries(&model.third.values),
        third_pos_scale: model.third_pos.scale,
        third_pos_values: Entries(&model.third_pos.values),
        third_reading_length_scale: model.third_reading_length.scale,
        third_reading_length_values: Entries(&model.third_reading_length.values),
        llm_alpha: model.llm_alpha,
        llm_crf_score_rms: model.llm_crf_score_rms,
        llm_feature_rms: model.llm_feature_rms,
    };
    let mut writer = create(path)?;
    serde_json::to_writer(&mut writer, &snapshot)?;
    writer.flush()?;
    Ok(())
}

#[allow(clippy::too_many_lines)]
pub fn load(path: &Path) -> Result<Model> {
    let saved: Snapshot = serde_json::from_reader(BufReader::new(File::open(path)?))?;
    let word_count = saved.words.len();
    let pos_count = saved.pos_names.len();
    let pos_token_ids = if saved.pos_token_ids.is_empty() {
        (0..pos_count)
            .map(|pos| u32::try_from(word_count.saturating_add(pos)))
            .collect::<std::result::Result<Vec<_>, _>>()?
    } else {
        saved.pos_token_ids.clone()
    };
    if !matches!(saved.version, 1..=9)
        || word_count < 3
        || pos_count < 3
        || word_count > crate::sparse::MAX_ID as usize + 1
        || pos_count > crate::sparse::MAX_ID as usize + 1
        || pos_count.checked_mul(pos_count) != Some(saved.initial_pos.len())
        || pos_token_ids.len() != pos_count
        || pos_token_ids.iter().enumerate().any(|(pos, &id)| {
            id as usize != word_count.saturating_add(pos) || id > crate::sparse::MAX_ID
        })
        || !saved.first_scale.is_finite()
        || saved.first_scale <= 0.0
        || !saved.third_scale.is_finite()
        || saved.third_scale <= 0.0
        || !saved.third_pos_scale.is_finite()
        || saved.third_pos_scale <= 0.0
        || !saved.third_reading_length_scale.is_finite()
        || saved.third_reading_length_scale <= 0.0
        || !saved.missing_pos_log.is_finite()
        || !saved.llm_alpha.is_finite()
        || !saved.llm_crf_score_rms.is_finite()
        || saved.llm_crf_score_rms < 0.0
        || !saved.llm_feature_rms.is_finite()
        || saved.llm_feature_rms < 0.0
        || saved.initial_pos.iter().any(|x| !x.is_finite())
        || saved.initial_reading_length_bigrams.len() != 81
        || saved.initial_reading_length_trigrams.len() != 729
        || saved
            .initial_reading_length_bigrams
            .iter()
            .chain(&saved.initial_reading_length_trigrams)
            .any(|x| !x.is_finite())
        || saved
            .words
            .iter()
            .any(|w| w.pos as usize >= pos_count || !w.initial.is_finite() || w.reading_length >= 9)
    {
        return Err("invalid checkpoint dimensions/values".into());
    }
    let valid_feature = |f: Feature| match f {
        Feature::Word(w) => (w as usize) < word_count,
        Feature::Pos(a, b) => (a as usize) < pos_count && (b as usize) < pos_count,
        Feature::Bigram(a, b) => (a as usize) < word_count && (b as usize) < word_count,
        Feature::PosWord(a, b) => {
            (a as usize) >= word_count
                && (a as usize) < word_count.saturating_add(pos_count)
                && (b as usize) < word_count
        }
        Feature::WordPos(a, b) => {
            (a as usize) < word_count
                && (b as usize) >= word_count
                && (b as usize) < word_count.saturating_add(pos_count)
        }
        Feature::Script(s) => (s as usize) < SCRIPT_NAMES.len(),
        Feature::ScriptBigram(a, b) => {
            (a as usize) < SCRIPT_NAMES.len() && (b as usize) < SCRIPT_NAMES.len()
        }
        Feature::ReadingLengthBigram(a, b) => a < 9 && b < 9,
    };
    if saved
        .first_values
        .0
        .iter()
        .any(|(k, v)| !valid_feature(k) || !v.is_finite() || !(v * saved.first_scale).is_finite())
        || saved.third_values.0.iter().any(|(k, v)| {
            k.iter().any(|&w| w as usize >= word_count)
                || !v.is_finite()
                || !(v * saved.third_scale).is_finite()
        })
        || saved.third_pos_values.0.iter().any(|(k, v)| {
            k.iter().any(|&pos| pos as usize >= pos_count)
                || !v.is_finite()
                || !(v * saved.third_pos_scale).is_finite()
        })
        || saved.third_reading_length_values.0.iter().any(|(k, v)| {
            k.iter().any(|&length| length >= 9)
                || !v.is_finite()
                || !(v * saved.third_reading_length_scale).is_finite()
        })
        || saved.initial_bigram_values.0.iter().any(|(k, v)| {
            !matches!(
                k,
                Feature::Bigram(_, _) | Feature::PosWord(_, _) | Feature::WordPos(_, _)
            ) || !valid_feature(k)
                || !v.is_finite()
        })
        || saved
            .initial_trigram_values
            .0
            .iter()
            .any(|(k, v)| k.iter().any(|&w| w as usize >= word_count) || !v.is_finite())
        || saved
            .initial_pos_trigram_values
            .0
            .iter()
            .any(|(k, v)| k.iter().any(|&pos| pos as usize >= pos_count) || !v.is_finite())
    {
        return Err("invalid checkpoint feature".into());
    }
    let first = Parameters {
        values: saved.first_values.0,
        scale: saved.first_scale,
    };
    let third = Parameters {
        values: saved.third_values.0,
        scale: saved.third_scale,
    };
    let third_pos = Parameters {
        values: saved.third_pos_values.0,
        scale: saved.third_pos_scale,
    };
    let third_reading_length = Parameters {
        values: saved.third_reading_length_values.0,
        scale: saved.third_reading_length_scale,
    };
    let initial_reading_length_bigrams: [f64; 81] = saved
        .initial_reading_length_bigrams
        .try_into()
        .map_err(|_| "invalid reading-length bigram dimensions")?;
    let initial_reading_length_trigrams: [f64; 729] = saved
        .initial_reading_length_trigrams
        .try_into()
        .map_err(|_| "invalid reading-length trigram dimensions")?;
    Ok(Model {
        words: saved.words,
        pos_names: saved.pos_names,
        pos_token_ids,
        affix_pos_selections: saved.affix_pos_selections,
        cross_bigrams_enabled: saved.cross_bigrams_enabled,
        pos_trigrams_enabled: saved.pos_trigrams_enabled,
        initial_pos: saved.initial_pos,
        initial_bigrams: saved.initial_bigram_values.0,
        initial_trigrams: saved.initial_trigram_values.0,
        initial_pos_trigrams: saved.initial_pos_trigram_values.0,
        initial_reading_length_bigrams,
        initial_reading_length_trigrams,
        llm_alpha: saved.llm_alpha,
        llm_crf_score_rms: saved.llm_crf_score_rms,
        llm_feature_rms: saved.llm_feature_rms,
        missing_pos_log: saved.missing_pos_log,
        first,
        third,
        third_pos,
        third_reading_length,
    })
}

/// POS/word/read-length score families, stable source-local ID maps, and a lossless checkpoint.
/// These are CRF potentials, NOT probabilities to feed the existing P(*) codec.
#[allow(clippy::too_many_lines)]
pub fn export(model: &Model, directory: &Path) -> Result<()> {
    fs::create_dir_all(directory)?;
    save(model, &directory.join("model.json"))?;
    let mut vocab = create(&directory.join("vocabulary.jsonl"))?;
    for (id, word) in model.words.iter().enumerate() {
        serde_json::to_writer(
            &mut vocab,
            &serde_json::json!({"word_id":id,"surface":word.surface,
            "reading":word.reading,"reading_length":word.reading_length,
            "pos_id":word.pos,"model_only":word.model_only}),
        )?;
        writeln!(vocab)?;
    }
    if model.cross_bigrams_enabled {
        for (pos_id, &word_id) in model.pos_token_ids.iter().enumerate() {
            let name = model
                .pos_names
                .get(pos_id)
                .ok_or("POS token ID has no name")?;
            serde_json::to_writer(
                &mut vocab,
                &serde_json::json!({"word_id":word_id,"surface":format!("[POS:{name}]"),
                "reading":"","reading_length":0,"pos_id":pos_id,"model_only":true,
                "pos_token":true}),
            )?;
            writeln!(vocab)?;
        }
    }
    vocab.flush()?;
    let mut pos = create(&directory.join("pos.tsv"))?;
    writeln!(pos, "pos_id\tmecab_feature")?;
    for (id, name) in model.pos_names.iter().enumerate() {
        writeln!(pos, "{id}\t{name}")?;
    }
    pos.flush()?;
    if model.cross_bigrams_enabled {
        let mut pos_tokens = create(&directory.join("pos-token.tsv"))?;
        writeln!(pos_tokens, "synthetic_word_id\tpos_id\tpos_name")?;
        for (pos_id, &word_id) in model.pos_token_ids.iter().enumerate() {
            writeln!(
                pos_tokens,
                "{word_id}\t{pos_id}\t{}",
                model.pos_names[pos_id]
            )?;
        }
        pos_tokens.flush()?;
    }
    let mut affixes = create(&directory.join("affix-pos-selection.tsv"))?;
    writeln!(
        affixes,
        "surface\treading\toriginal_pos\trefined_pos\tunigram_count"
    )?;
    for selection in &model.affix_pos_selections {
        writeln!(
            affixes,
            "{}\t{}\t{}\t{}\t{}",
            selection.surface,
            selection.reading,
            selection.original_pos,
            selection.assigned_pos,
            selection.unigram_count
        )?;
    }
    affixes.flush()?;
    let mut words = create(&directory.join("word-cost.tsv"))?;
    writeln!(words, "word_id\tinitial_log_p_word_given_pos\tdelta\tscore")?;
    for (id, word) in model.words.iter().enumerate().skip(1) {
        let f = Feature::Word(u32::try_from(id)?);
        writeln!(
            words,
            "{id}\t{:.17e}\t{:.17e}\t{:.17e}",
            word.initial,
            model.first.get(f),
            model.weight(f)
        )?;
    }
    words.flush()?;
    let mut transitions = create(&directory.join("pos-bigram.tsv"))?;
    writeln!(transitions, "pos_a\tpos_b\tinitial_log_p\tdelta\tscore")?;
    for a in 1..model.pos_names.len() {
        for b in 1..model.pos_names.len() {
            let f = Feature::Pos(u32::try_from(a)?, u32::try_from(b)?);
            writeln!(
                transitions,
                "{a}\t{b}\t{:.17e}\t{:.17e}\t{:.17e}",
                model.initial(f),
                model.first.get(f),
                model.weight(f)
            )?;
        }
    }
    transitions.flush()?;
    let mut bi = create(&directory.join("word-bigram.tsv"))?;
    writeln!(bi, "word_a\tword_b\tinitial_delta\tlearned_delta\tscore")?;
    // The learned map can contain hundreds of millions of keys. Stream it and
    // use the small immutable seed map for lookup; then emit seed-only entries.
    // TSV row order is intentionally unspecified, as is checkpoint array order.
    for (feature, raw) in model.first.values.iter() {
        if let Feature::Bigram(a, b) = feature {
            let initial_delta = model.initial_bigrams.get(&feature).copied().unwrap_or(0.0);
            let learned_delta = raw * model.first.scale;
            writeln!(
                bi,
                "{a}\t{b}\t{initial_delta:.17e}\t{learned_delta:.17e}\t{:.17e}",
                initial_delta + learned_delta
            )?;
        }
    }
    for (feature, initial_delta) in model.initial_bigrams.iter() {
        if let Feature::Bigram(a, b) = feature {
            if !model.first.values.contains_key(&feature) {
                writeln!(
                    bi,
                    "{a}\t{b}\t{initial_delta:.17e}\t{:.17e}\t{initial_delta:.17e}",
                    0.0
                )?;
            }
        }
    }
    bi.flush()?;
    if model.cross_bigrams_enabled {
        write_cross_bigrams(model, directory, true)?;
        write_cross_bigrams(model, directory, false)?;
    }
    let mut tri = create(&directory.join("word-trigram.tsv"))?;
    writeln!(
        tri,
        "word_a\tword_b\tword_c\tinitial_delta\tlearned_delta\tscore"
    )?;
    for (key, raw) in model.third.values.iter() {
        let learned = raw * model.third.scale;
        let initial = model.initial_trigrams.get(&key).copied().unwrap_or(0.0);
        writeln!(
            tri,
            "{}\t{}\t{}\t{initial:.17e}\t{learned:.17e}\t{:.17e}",
            key[0],
            key[1],
            key[2],
            initial + learned
        )?;
    }
    for (key, initial) in model.initial_trigrams.iter() {
        if !model.third.values.contains_key(&key) {
            writeln!(
                tri,
                "{}\t{}\t{}\t{initial:.17e}\t{:.17e}\t{:.17e}",
                key[0], key[1], key[2], 0.0, initial
            )?;
        }
    }
    tri.flush()?;
    if model.pos_trigrams_enabled {
        let mut pos_tri = create(&directory.join("pos-trigram.tsv"))?;
        writeln!(
            pos_tri,
            "pos_a\tpos_b\tpos_c\tinitial_delta\tlearned_delta\tscore"
        )?;
        for (key, initial) in model.initial_pos_trigrams.iter() {
            let learned = model.third_pos.get(key);
            writeln!(
                pos_tri,
                "{}\t{}\t{}\t{initial:.17e}\t{learned:.17e}\t{:.17e}",
                key[0],
                key[1],
                key[2],
                initial + learned
            )?;
        }
        for (key, raw) in model.third_pos.values.iter() {
            if !model.initial_pos_trigrams.contains_key(&key) {
                let learned = raw * model.third_pos.scale;
                writeln!(
                    pos_tri,
                    "{}\t{}\t{}\t{:.17e}\t{learned:.17e}\t{learned:.17e}",
                    key[0], key[1], key[2], 0.0
                )?;
            }
        }
        pos_tri.flush()?;
    }

    let mut readlen_bi = create(&directory.join("reading-length-bigram.tsv"))?;
    writeln!(
        readlen_bi,
        "length_a\tlength_b\tinitial_delta\tlearned_delta\tscore"
    )?;
    for a in 0..9 {
        for b in 0..9 {
            let key = Feature::ReadingLengthBigram(a, b);
            let initial = model.initial(key);
            let learned = model.first.get(key);
            writeln!(
                readlen_bi,
                "{a}\t{b}\t{initial:.17e}\t{learned:.17e}\t{:.17e}",
                initial + learned
            )?;
        }
    }
    readlen_bi.flush()?;

    let mut readlen_tri = create(&directory.join("reading-length-trigram.tsv"))?;
    writeln!(
        readlen_tri,
        "length_a\tlength_b\tlength_c\tinitial_delta\tlearned_delta\tscore"
    )?;
    for a in 0..9 {
        for b in 0..9 {
            for c in 0..9 {
                let key = [a, b, c];
                let index = (usize::from(a) * 9 + usize::from(b)) * 9 + usize::from(c);
                let initial = model.initial_reading_length_trigrams[index];
                let learned = model.third_reading_length.get(key);
                writeln!(
                    readlen_tri,
                    "{a}\t{b}\t{c}\t{initial:.17e}\t{learned:.17e}\t{:.17e}",
                    initial + learned
                )?;
            }
        }
    }
    readlen_tri.flush()?;
    Ok(())
}

fn write_cross_bigrams(model: &Model, directory: &Path, pos_word: bool) -> Result<()> {
    let (filename, header) = if pos_word {
        (
            "pos-word-bigram.tsv",
            "source_pos_id\tsource_pos_token_id\ttarget_word_id\tinitial_delta\tlearned_delta\tscore",
        )
    } else {
        (
            "word-pos-bigram.tsv",
            "source_word_id\ttarget_pos_id\ttarget_pos_token_id\tinitial_delta\tlearned_delta\tscore",
        )
    };
    let mut writer = create(&directory.join(filename))?;
    writeln!(writer, "{header}")?;
    for (feature, raw) in model.first.values.iter() {
        let row = match feature {
            Feature::PosWord(token, target) if pos_word => {
                let pos = (token as usize)
                    .checked_sub(model.words.len())
                    .filter(|&pos| pos < model.pos_names.len())
                    .and_then(|pos| u32::try_from(pos).ok());
                pos.map(|pos| (pos, token, target))
            }
            Feature::WordPos(source, token) if !pos_word => {
                let pos = (token as usize)
                    .checked_sub(model.words.len())
                    .filter(|&pos| pos < model.pos_names.len())
                    .and_then(|pos| u32::try_from(pos).ok());
                pos.map(|pos| (source, pos, token))
            }
            _ => None,
        };
        if let Some((a, b, c)) = row {
            let initial = model.initial_bigrams.get(&feature).copied().unwrap_or(0.0);
            let learned = raw * model.first.scale;
            writeln!(
                writer,
                "{a}\t{b}\t{c}\t{initial:.17e}\t{learned:.17e}\t{:.17e}",
                initial + learned
            )?;
        }
    }
    for (feature, initial) in model.initial_bigrams.iter() {
        let row = match feature {
            Feature::PosWord(token, target) if pos_word => {
                let pos = (token as usize)
                    .checked_sub(model.words.len())
                    .filter(|&pos| pos < model.pos_names.len())
                    .and_then(|pos| u32::try_from(pos).ok());
                pos.map(|pos| (pos, token, target))
            }
            Feature::WordPos(source, token) if !pos_word => {
                let pos = (token as usize)
                    .checked_sub(model.words.len())
                    .filter(|&pos| pos < model.pos_names.len())
                    .and_then(|pos| u32::try_from(pos).ok());
                pos.map(|pos| (source, pos, token))
            }
            _ => None,
        };
        if let Some((a, b, c)) = row {
            if !model.first.values.contains_key(&feature) {
                writeln!(
                    writer,
                    "{a}\t{b}\t{c}\t{initial:.17e}\t{:.17e}\t{initial:.17e}",
                    0.0
                )?;
            }
        }
    }
    writer.flush()?;
    Ok(())
}
