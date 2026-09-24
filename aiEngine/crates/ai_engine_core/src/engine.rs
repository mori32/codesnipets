//! Dictionary lattice and bounded N-best search.

use std::cmp::Ordering;
use std::collections::HashSet;
use std::io;
use std::path::Path;

use crate::dictionary::{Dictionary, Word};
use crate::normalize::to_hiragana;

/// One complete conversion candidate.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Candidate {
    pub text: String,
    pub reading: String,
    pub score: i32,
    pub segments: Vec<Word>,
}

/// Candidate with an optional context-model score applied.
#[derive(Clone, Debug)]
pub struct RerankedCandidate {
    pub candidate: Candidate,
    pub llm_score: f32,
    pub combined_score: f32,
}

/// Merges classical scores with evaluator scores and sorts highest first.
///
/// `llm_scores` uses a higher-is-better convention. The evaluator may provide
/// log probabilities (usually negative), normalized scores, or fixed test
/// values; `weight` controls how strongly they affect the classical score.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn rerank_candidates(
    candidates: &[Candidate],
    llm_scores: &[f32],
    weight: f32,
) -> Vec<RerankedCandidate> {
    let mut ranked: Vec<_> = candidates
        .iter()
        .enumerate()
        .map(|(index, candidate)| {
            let llm_score = llm_scores.get(index).copied().unwrap_or(0.0);
            let llm_score = if llm_score.is_finite() {
                llm_score
            } else {
                0.0
            };
            RerankedCandidate {
                candidate: candidate.clone(),
                llm_score,
                combined_score: candidate.score as f32 + weight * llm_score,
            }
        })
        .collect();
    ranked.sort_by(|left, right| {
        right
            .combined_score
            .total_cmp(&left.combined_score)
            .then_with(|| right.candidate.score.cmp(&left.candidate.score))
    });
    ranked
}

#[derive(Clone, Debug)]
struct PathState {
    score: i32,
    segments: Vec<Word>,
    last_right_id: Option<u32>,
}

/// Kana-kanji conversion engine with a replaceable dictionary.
#[derive(Clone, Debug, Default)]
pub struct KanaKanjiEngine {
    dictionary: Dictionary,
}

impl KanaKanjiEngine {
    /// Creates an engine using the built-in Phase 1 lexicon.
    #[must_use]
    pub fn new() -> Self {
        Self {
            dictionary: Dictionary::new(),
        }
    }

    /// Adds a user entry. Romaji is accepted as a convenience.
    pub fn add_word(&mut self, reading: &str, surface: &str, priority: i32) {
        self.dictionary.add_user_word(reading, surface, priority);
    }

    /// Loads an external UTF-8 TSV dictionary without adding entries to user learning.
    ///
    /// # Errors
    ///
    /// Returns an I/O error for unreadable files or malformed rows.
    pub fn load_dictionary(&mut self, path: impl AsRef<Path>) -> io::Result<()> {
        self.dictionary.load_dictionary(path)
    }

    /// Loads the provisional P5 binary dictionary.
    ///
    /// # Errors
    ///
    /// Returns an I/O or invalid-data error when the dictionary cannot be
    /// loaded or validated.
    pub fn load_static_dictionary(&mut self, path: impl AsRef<Path>) -> io::Result<()> {
        self.dictionary.load_static_dictionary(path)
    }

    /// Loads user entries from a UTF-8 TSV file.
    ///
    /// # Errors
    ///
    /// Returns an I/O error for unreadable files or malformed rows.
    pub fn load_user_dictionary(&mut self, path: impl AsRef<Path>) -> io::Result<()> {
        self.dictionary.load_user_dictionary(path)
    }

    /// Saves user entries to a UTF-8 TSV file.
    ///
    /// # Errors
    ///
    /// Returns an I/O error when the destination cannot be created or written.
    pub fn save_user_dictionary(&self, path: impl AsRef<Path>) -> io::Result<()> {
        self.dictionary.save_user_dictionary(path)
    }

    /// Records the selected candidate for future ranking.
    pub fn select(&mut self, candidate: &Candidate) {
        for segment in &candidate.segments {
            self.dictionary.record_selection(segment);
        }
    }

    /// Converts romaji, hiragana, or katakana into up to `limit` candidates.
    #[must_use]
    pub fn convert(&self, input: &str, limit: usize) -> Vec<Candidate> {
        if limit == 0 {
            return Vec::new();
        }
        let reading = to_hiragana(input);
        let characters: Vec<char> = reading.chars().collect();
        if characters.is_empty() {
            return vec![Candidate {
                text: String::new(),
                reading,
                score: 0,
                segments: Vec::new(),
            }];
        }

        let beam_width = limit.saturating_mul(8).max(8);
        let mut paths: Vec<Vec<PathState>> = vec![Vec::new(); characters.len() + 1];
        paths[0].push(PathState {
            score: 0,
            segments: Vec::new(),
            last_right_id: None,
        });

        for start in 0..characters.len() {
            if paths[start].is_empty() {
                continue;
            }
            let matches = self.dictionary.matching_words(&characters, start);
            let entries = if matches.is_empty() {
                vec![Word {
                    reading: characters[start].to_string(),
                    surface: characters[start].to_string(),
                    priority: -100,
                    frequency: 0,
                    left_id: 0,
                    right_id: 0,
                    cost: 0,
                }]
            } else {
                matches
            };

            for word in entries {
                let end = start + word.reading.chars().count();
                for path in paths[start].clone() {
                    let mut segments = path.segments;
                    segments.push(word.clone());
                    let has_context = word_has_context(&word);
                    let connection_cost = if has_context {
                        path.last_right_id.map_or(0, |right_id| {
                            self.dictionary.connection_cost(right_id, word.left_id)
                        })
                    } else {
                        0
                    };
                    paths[end].push(PathState {
                        score: path
                            .score
                            .saturating_add(word_score(&word))
                            .saturating_sub(connection_cost),
                        segments,
                        last_right_id: has_context.then_some(word.right_id),
                    });
                }
                prune(&mut paths[end], beam_width);
            }
        }

        let mut final_paths = paths.pop().unwrap_or_default();
        final_paths.sort_by(compare_paths);
        let mut seen = HashSet::new();
        let mut candidates = Vec::with_capacity(limit);
        for path in final_paths {
            let text = path
                .segments
                .iter()
                .map(|word| word.surface.as_str())
                .collect::<String>();
            if seen.insert(text.clone()) {
                candidates.push(Candidate {
                    text,
                    reading: reading.clone(),
                    score: path.score,
                    segments: path.segments,
                });
                if candidates.len() == limit {
                    break;
                }
            }
        }
        candidates
    }
}

fn word_score(word: &Word) -> i32 {
    let frequency = i32::try_from(word.frequency).unwrap_or(i32::MAX / 5);
    let length = i32::try_from(word.reading.chars().count()).unwrap_or(i32::MAX);
    word.priority
        .saturating_sub(word.cost)
        .saturating_add(frequency.saturating_mul(5))
        .saturating_add(length)
}

fn word_has_context(word: &Word) -> bool {
    word.cost != 0 || word.left_id != 0 || word.right_id != 0
}

fn prune(paths: &mut Vec<PathState>, beam_width: usize) {
    if paths.len() <= beam_width {
        return;
    }
    paths.sort_by(compare_paths);
    paths.truncate(beam_width);
}

fn compare_paths(left: &PathState, right: &PathState) -> Ordering {
    right.score.cmp(&left.score).then_with(|| {
        let left_text = left
            .segments
            .iter()
            .map(|word| word.surface.as_str())
            .collect::<String>();
        let right_text = right
            .segments
            .iter()
            .map(|word| word.surface.as_str())
            .collect::<String>();
        left_text.cmp(&right_text)
    })
}
