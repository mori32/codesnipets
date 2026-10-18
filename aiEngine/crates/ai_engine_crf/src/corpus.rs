//! Streaming tagged JSONL input and reading-based variable-segmentation lattices.
#![allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]

use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::{BOS, EOS, Lattice, Model, Node, Result, script};

#[derive(Clone, Debug, Deserialize)]
pub struct Token {
    pub surface: String,
    pub feature: String,
    pub reading: Option<String>,
    pub unknown: bool,
    #[serde(default)]
    pub segment_index: u32,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Record {
    pub page_id: u64,
    pub title: String,
    pub sentence: String,
    pub tokens: Vec<Token>,
}

pub fn open(path: &Path) -> Result<Box<dyn BufRead>> {
    let input = BufReader::with_capacity(1 << 20, File::open(path)?);
    if path.extension().is_some_and(|s| s == "bz2") {
        Ok(Box::new(BufReader::with_capacity(
            1 << 20,
            bzip2::read::MultiBzDecoder::new(input),
        )))
    } else {
        Ok(Box::new(input))
    }
}

/// Bounded UTF-8 JSONL read, including corrupt-line allocation protection.
pub fn read_record(
    reader: &mut dyn BufRead,
    line: &mut String,
    number: usize,
) -> Result<Option<Record>> {
    line.clear();
    if reader.take((8 << 20) + 1).read_line(line)? == 0 {
        return Ok(None);
    }
    if line.len() > 8 << 20 {
        return Err("JSONL record exceeds 8 MiB safety limit".into());
    }
    Ok(Some(
        serde_json::from_str(line).map_err(|e| format!("JSON record {number}: {e}"))?,
    ))
}

#[derive(Clone, Debug)]
pub struct Index {
    pub keys: HashMap<String, u32>,
    pub readings: HashMap<String, Vec<u32>>,
    pub classes: HashMap<String, u32>,
    /// Model-specific lexicalized POS refinements for selected affix lexemes.
    pub affix_pos: HashMap<String, String>,
    pub max_reading_bytes: usize,
}

#[must_use]
pub fn key(surface: &str, reading: &str, pos: &str) -> String {
    format!("{surface}\u{1f}{reading}\u{1f}{pos}")
}

impl Index {
    pub fn new(model: &Model) -> Result<Self> {
        let mut result = Self {
            keys: HashMap::new(),
            readings: HashMap::new(),
            classes: HashMap::new(),
            affix_pos: HashMap::new(),
            max_reading_bytes: 0,
        };
        for selection in &model.affix_pos_selections {
            let source_key = key(
                &selection.surface,
                &selection.reading,
                &selection.original_pos,
            );
            if result
                .affix_pos
                .insert(source_key, selection.assigned_pos.clone())
                .is_some()
            {
                return Err("duplicate selected affix POS identity".into());
            }
        }
        for (id, word) in model.words.iter().enumerate().skip(3) {
            let id = u32::try_from(id)?;
            let pos = &model.pos_names[word.pos as usize];
            if result
                .keys
                .insert(key(&word.surface, &word.reading, pos), id)
                .is_some()
            {
                return Err("duplicate source word identity".into());
            }
            if word.model_only {
                result.classes.insert(pos.clone(), id);
            } else if !word.reading.is_empty() && word.reading != "*" {
                result
                    .readings
                    .entry(word.reading.clone())
                    .or_default()
                    .push(id);
                result.max_reading_bytes = result.max_reading_bytes.max(word.reading.len());
            }
        }
        Ok(result)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Limits {
    pub words: usize,
    pub nodes: usize,
    pub edges: usize,
}

#[derive(Clone, Debug)]
pub struct Prepared {
    pub page_id: u64,
    pub title: String,
    pub sentence: String,
    pub reading: String,
    pub graph: Lattice,
    pub forced_tokens: usize,
}

/// Lightweight normalized gold spans; no lattice nodes/edges retained here.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Sentence {
    pub page_id: u64,
    pub title: String,
    pub sentence: String,
    pub reading: String,
    pub spans: Vec<Span>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Span {
    pub start: usize,
    pub end: usize,
    pub word: u32,
    pub script: u8,
    pub forced: bool,
}
impl Sentence {
    #[must_use]
    pub fn forced_tokens(&self) -> usize {
        self.spans.iter().filter(|s| s.forced).count()
    }
}

#[derive(Clone, Debug)]
struct GoldToken {
    id: u32,
    reading: String,
    script: u8,
    forced: bool,
}

fn english(surface: &str) -> bool {
    let mut letter = false;
    for ch in surface.chars() {
        if ch.is_alphabetic() && !(('あ'..='ヿ').contains(&ch) || ('㐀'..='鿿').contains(&ch)) {
            letter = true;
        } else if !ch.is_numeric() {
            return false;
        }
    }
    letter
}

fn class_token(surface: String, pos: &str, index: &Index) -> Result<GoldToken> {
    let id = *index
        .classes
        .get(pos)
        .ok_or_else(|| format!("missing POS-only class: {pos}"))?;
    Ok(GoldToken {
        id,
        script: script(&surface),
        reading: surface,
        forced: true,
    })
}

fn lexical(surface: &str, reading: String, pos: &str, index: &Index) -> Result<GoldToken> {
    let id = *index
        .keys
        .get(&key(surface, &reading, pos))
        .ok_or_else(|| format!("source word is missing: {surface}/{reading}/{pos}"))?;
    Ok(GoldToken {
        id,
        script: script(surface),
        reading,
        forced: false,
    })
}

fn normalize_token(token: &Token, index: &Index) -> Result<Vec<GoldToken>> {
    if token.surface.is_empty() {
        return Err("empty corpus token".into());
    }
    let surface = ai_engine_dict::normalize_surface(&token.surface);
    // Each symbol is one word, irrespective of an unreviewed MeCab POS guess.
    if surface
        .chars()
        .all(|ch| !ch.is_alphanumeric() && ch != 'ー')
    {
        return surface
            .chars()
            .map(|ch| {
                let text = ch.to_string();
                lexical(&text, text.clone(), "記号,一般,*,*,*,*,*", index)
            })
            .collect();
    }
    if surface.chars().all(char::is_numeric)
        || (!token.unknown && token.feature.starts_with("名詞,数,"))
    {
        return Ok(vec![class_token(surface, "名詞,数,*,*,*,*,*", index)?]);
    }
    if english(&surface) {
        return Ok(vec![class_token(surface, "英単語", index)?]);
    }
    if token.unknown || token.feature.starts_with("記号,単漢字,") {
        // Never adopt MeCab's unchecked unknown-word POS assignment.
        return Ok(vec![class_token(surface, "未知語", index)?]);
    }
    if token.feature.split(',').count() < 6 {
        return Err("invalid grammatical feature".into());
    }
    let reading = token
        .reading
        .as_deref()
        .filter(|s| !s.is_empty() && *s != "*")
        .ok_or("known word has no reading")?;
    let reading = ai_engine_dict::normalize_reading(reading);
    let base_pos = ai_engine_dict::model_pos(&surface, &token.feature);
    let source_key = key(&surface, &reading, &base_pos);
    let pos = index
        .affix_pos
        .get(&source_key)
        .map_or(base_pos.as_str(), String::as_str);
    Ok(vec![lexical(&surface, reading, pos, index)?])
}

pub fn normalize(record: Record, index: &Index, max_words: usize) -> Result<Sentence> {
    if record.tokens.iter().any(|t| t.segment_index != 0) {
        return Err("multi-segment record must be split before training".into());
    }
    let mut tokens = Vec::new();
    for token in &record.tokens {
        tokens.extend(normalize_token(token, index)?);
    }
    if tokens.is_empty() || tokens.len() > max_words {
        return Err("sentence word limit exceeded or empty sentence".into());
    }
    let mut reading = String::new();
    let mut spans = Vec::new();
    for token in &tokens {
        let start = reading.len();
        reading.push_str(&token.reading);
        let end = reading.len();
        spans.push(Span {
            start,
            end,
            word: token.id,
            script: token.script,
            forced: token.forced,
        });
    }
    Ok(Sentence {
        page_id: record.page_id,
        title: record.title,
        sentence: record.sentence,
        reading,
        spans,
    })
}

pub fn prepare(record: Record, model: &Model, index: &Index, limits: Limits) -> Result<Prepared> {
    let sentence = normalize(record, index, limits.words)?;
    prepare_sentence(&sentence, model, index, limits)
}

pub fn prepare_sentence(
    sentence: &Sentence,
    model: &Model,
    index: &Index,
    limits: Limits,
) -> Result<Prepared> {
    let reading = &sentence.reading;
    let forced = sentence
        .spans
        .iter()
        .filter(|s| s.forced)
        .map(|s| Node {
            start: s.start,
            end: s.end,
            word: s.word,
            script: s.script,
        })
        .collect::<Vec<_>>();
    // Unknown, numeric and Latin spans have no dictionary reading annotation.
    // Preserve these literal input spans; do not hallucinate Japanese readings.
    let mut nodes = Vec::new();
    let boundaries = reading
        .char_indices()
        .map(|(i, _)| i)
        .chain(std::iter::once(reading.len()))
        .collect::<Vec<_>>();
    for (start_index, &start) in boundaries.iter().enumerate().take(boundaries.len() - 1) {
        for &end in &boundaries[start_index + 1..] {
            if end - start > index.max_reading_bytes {
                break;
            }
            if forced.iter().any(|f| start < f.end && end > f.start) {
                continue;
            }
            if let Some(ids) = index.readings.get(&reading[start..end]) {
                for &id in ids {
                    nodes.push(Node {
                        start,
                        end,
                        word: id,
                        script: script(&model.words[id as usize].surface),
                    });
                }
            }
        }
        if nodes.len() + forced.len() > limits.nodes {
            return Err("lattice node budget exceeded".into());
        }
    }
    nodes.extend(forced);
    nodes.sort_by_key(|n| (n.end, n.start, n.word));
    let mut all = vec![Node {
        start: 0,
        end: 0,
        word: BOS,
        script: 0,
    }];
    all.extend(nodes);
    all.push(Node {
        start: reading.len(),
        end: reading.len(),
        word: EOS,
        script: 1,
    });
    let lookup = all
        .iter()
        .enumerate()
        .map(|(id, n)| ((n.start, n.end, n.word), id))
        .collect::<BTreeMap<_, _>>();
    let mut gold = vec![0];
    for span in &sentence.spans {
        let spec = (span.start, span.end, span.word);
        gold.push(
            *lookup
                .get(&spec)
                .ok_or("gold source word has no reading-lattice node")?,
        );
    }
    gold.push(all.len() - 1);
    let graph = Lattice::new(all, gold, limits.edges)?;
    Ok(Prepared {
        page_id: sentence.page_id,
        title: sentence.title.clone(),
        sentence: sentence.sentence.clone(),
        reading: reading.clone(),
        graph,
        forced_tokens: sentence.forced_tokens(),
    })
}

/// Page-level split. Related sentences from the same page never cross the split.
#[must_use]
pub fn validation_page(page: u64) -> bool {
    let mut mixed = page.wrapping_add(0x9e37_79b9_7f4a_7c15);
    mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    (mixed ^ (mixed >> 31)) % 5 == 0
}

/// A bounded, reproducible uniform sample over a stream, not its first N items.
/// Keep a separate RNG from training so evaluation sampling never changes SGD.
#[derive(Debug)]
pub struct Reservoir<T> {
    values: Vec<T>,
    seen: usize,
    capacity: usize,
    random: u64,
}

impl<T> Reservoir<T> {
    #[must_use]
    pub fn new(capacity: usize, seed: u64) -> Self {
        Self {
            values: Vec::new(),
            seen: 0,
            capacity,
            random: seed.max(1),
        }
    }

    pub fn push(&mut self, value: T) {
        self.seen += 1;
        if self.values.len() < self.capacity {
            self.values.push(value);
        } else if self.capacity > 0 {
            let bound = u64::try_from(self.seen).expect("stream length fits u64");
            // Rejection sampling avoids the modulo bias in a bounded draw.
            let threshold = bound.wrapping_neg() % bound;
            let draw = loop {
                self.random ^= self.random << 13;
                self.random ^= self.random >> 7;
                self.random ^= self.random << 17;
                let random = self.random.wrapping_mul(0x2545_f491_4f6c_dd1d);
                if random >= threshold {
                    break random % bound;
                }
            };
            if let Ok(slot) = usize::try_from(draw) {
                if slot < self.capacity {
                    self.values[slot] = value;
                }
            }
        }
    }

    #[must_use]
    pub fn into_values(self) -> Vec<T> {
        self.values
    }
}

/// Seeded Fisher-Yates shuffle without introducing a platform RNG dependency.
pub fn shuffle<T>(values: &mut [T], state: &mut u64) {
    for i in (1..values.len()).rev() {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        let j = usize::try_from(*state % u64::try_from(i + 1).expect("slice fits u64"))
            .expect("index fits usize");
        values.swap(i, j);
    }
}

#[cfg(test)]
mod tests {
    use super::{Index, Token, key, normalize_token};
    use std::collections::HashMap;

    #[test]
    fn corpus_normalization_uses_the_selected_refined_affix_pos() {
        let base_pos = "名詞,接尾,一般,*,*,*,*";
        let refined_pos = "名詞,接尾,一般,*,*,*,*,CRF頻出接尾語:市/し";
        let source_key = key("市", "し", base_pos);
        let index = Index {
            keys: HashMap::from([(key("市", "し", refined_pos), 17)]),
            readings: HashMap::new(),
            classes: HashMap::new(),
            affix_pos: HashMap::from([(source_key, refined_pos.to_owned())]),
            max_reading_bytes: 0,
        };
        let token = Token {
            surface: "市".to_owned(),
            feature: "名詞,接尾,一般,*,*,*,*,シ,シ".to_owned(),
            reading: Some("シ".to_owned()),
            unknown: false,
            segment_index: 0,
        };

        let normalized = normalize_token(&token, &index).expect("selected affix token");
        assert_eq!(normalized.len(), 1);
        assert_eq!(normalized[0].id, 17);
        assert_eq!(index.affix_pos.len(), 1);
    }
}
