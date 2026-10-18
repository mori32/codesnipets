//! Character-diff focused aggregation of word trigrams around top-1 errors.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt::Write as _;
use std::path::Path;

use ai_engine_crf::corpus::{self, Index, Limits, Prepared, Sentence};
use ai_engine_crf::parallel::Trainer;
use ai_engine_crf::{BOS, EOS, Model, Result, nbest, rerank};
use serde::Serialize;

use crate::Options;

const MAX_LCS_CELLS: usize = 4_000_000;
const MAX_DIFF_EXAMPLES: usize = 100;

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
pub struct WordLabel {
    word_id: u32,
    surface: String,
    pos: String,
}

#[derive(Clone, Debug, Serialize)]
struct CharDiffHunk {
    gold_char_range: [usize; 2],
    converted_char_range: [usize; 2],
    gold_text: String,
    converted_text: String,
}

#[derive(Clone, Debug, Serialize)]
struct DiffExample {
    page_id: u64,
    title: String,
    gold_text: String,
    converted_text: String,
    char_diff: Vec<CharDiffHunk>,
}

#[derive(Clone, Debug, Serialize)]
struct RankedTrigram {
    rank: usize,
    trigram: [WordLabel; 3],
    occurrences: usize,
    error_sentences: usize,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct TrigramPairKey {
    wrong_pattern: String,
    gold: [WordLabel; 3],
    wrong: [WordLabel; 3],
}

#[derive(Clone, Debug, Serialize)]
struct RankedTrigramPair {
    rank: usize,
    wrong_pattern: String,
    gold_trigram: [WordLabel; 3],
    wrong_trigram: [WordLabel; 3],
    occurrences: usize,
    error_sentences: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct StageReport {
    top1_error_sentences: usize,
    character_different_error_sentences: usize,
    same_text_but_different_path_sentences: usize,
    character_diff_hunks: usize,
    pub character_edit_distance: usize,
    pub character_denominator: usize,
    pub character_level_accuracy: f64,
    gold_side_trigrams: Vec<RankedTrigram>,
    converted_side_trigrams: Vec<RankedTrigram>,
    paired_gold_vs_wrong_trigrams: Vec<RankedTrigramPair>,
    examples: Vec<DiffExample>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ErrorTrigramReport {
    pub method: &'static str,
    pub character_unit: &'static str,
    pub sentences: usize,
    pub nbest: usize,
    pub bigram_stage: StageReport,
    pub post_trigram_stage: StageReport,
}

#[derive(Clone, Debug)]
struct TokenChars {
    start: usize,
    end: usize,
}

#[derive(Clone, Debug)]
struct PathDisplay {
    text: String,
    labels: Vec<WordLabel>,
    token_chars: Vec<TokenChars>,
}

#[derive(Clone, Debug)]
struct PathSet {
    gold: PathDisplay,
    bigram: PathDisplay,
    post_trigram: PathDisplay,
    bigram_exact: bool,
    post_trigram_exact: bool,
    page_id: u64,
    title: String,
}

#[derive(Clone, Debug)]
struct TrigramCount {
    occurrences: usize,
    error_sentences: usize,
}

#[derive(Default)]
struct StageAccumulator {
    top1_error_sentences: usize,
    character_different_error_sentences: usize,
    same_text_but_different_path_sentences: usize,
    character_diff_hunks: usize,
    character_edit_distance: usize,
    character_denominator: usize,
    gold: HashMap<[WordLabel; 3], TrigramCount>,
    converted: HashMap<[WordLabel; 3], TrigramCount>,
    pairs: HashMap<TrigramPairKey, TrigramCount>,
    examples: Vec<DiffExample>,
}

#[derive(Clone, Copy, Debug)]
struct Hunk {
    gold_start: usize,
    gold_end: usize,
    converted_start: usize,
    converted_end: usize,
}

pub fn analyze(
    model: &Model,
    index: &Index,
    samples: &[Sentence],
    options: &Options,
    trainer: &Trainer,
) -> Result<ErrorTrigramReport> {
    let limits = Limits {
        words: options.max_words,
        nodes: options.max_nodes,
        edges: options.max_edges,
    };
    let paths = trainer.map(samples, |sentence| {
        let prepared = corpus::prepare_sentence(sentence, model, index, limits)?;
        let candidates = nbest(
            model,
            &prepared.graph,
            options.nbest,
            options.max_astar_expansions,
        )?;
        let reranked = rerank(model, &prepared.graph, &candidates)?;
        let first = candidates.first().ok_or("N-best returned no paths")?;
        let final_path = candidates
            .get(reranked.best)
            .ok_or("reranker selected a missing path")?;
        Ok(PathSet {
            gold: path_display(model, &prepared, &prepared.graph.gold),
            bigram: path_display(model, &prepared, &first.nodes),
            post_trigram: path_display(model, &prepared, &final_path.nodes),
            bigram_exact: first.nodes == prepared.graph.gold,
            post_trigram_exact: final_path.nodes == prepared.graph.gold,
            page_id: prepared.page_id,
            title: prepared.title,
        })
    })?;

    let mut bigram = StageAccumulator::default();
    let mut post_trigram = StageAccumulator::default();
    for paths in &paths {
        bigram.add_error(
            &paths.gold,
            &paths.bigram,
            paths.bigram_exact,
            paths.page_id,
            &paths.title,
        );
        post_trigram.add_error(
            &paths.gold,
            &paths.post_trigram,
            paths.post_trigram_exact,
            paths.page_id,
            &paths.title,
        );
    }

    Ok(ErrorTrigramReport {
        method: "Unicode-scalar LCS diff; pair each wrong-side trigram overlapping the changed span with the nearest aligned gold-side trigram; retain CIC for one changed token, or CII/III/IIC for spans of two or more tokens",
        character_unit: "Unicode scalar value (Rust char), not grapheme cluster",
        sentences: samples.len(),
        nbest: options.nbest,
        bigram_stage: bigram.finish(),
        post_trigram_stage: post_trigram.finish(),
    })
}

/// Write the top 100 paired gold/wrong trigram entries across both stages.
/// Counts are one observation per erroneous sentence, with duplicate windows
/// within the same sentence deduplicated.
pub fn write_paired_tsv(report: &ErrorTrigramReport, path: &Path) -> Result<usize> {
    #[derive(Default)]
    struct Counts {
        bigram_occurrences: usize,
        bigram_sentences: usize,
        post_occurrences: usize,
        post_sentences: usize,
    }

    let mut pairs: HashMap<TrigramPairKey, Counts> = HashMap::new();
    for (stage, rows) in [
        (0, &report.bigram_stage.paired_gold_vs_wrong_trigrams),
        (1, &report.post_trigram_stage.paired_gold_vs_wrong_trigrams),
    ] {
        for row in rows {
            let key = TrigramPairKey {
                wrong_pattern: row.wrong_pattern.clone(),
                gold: row.gold_trigram.clone(),
                wrong: row.wrong_trigram.clone(),
            };
            let counts = pairs.entry(key).or_default();
            if stage == 0 {
                counts.bigram_occurrences += row.occurrences;
                counts.bigram_sentences += row.error_sentences;
            } else {
                counts.post_occurrences += row.occurrences;
                counts.post_sentences += row.error_sentences;
            }
        }
    }

    let mut rows: Vec<_> = pairs.into_iter().collect();
    rows.sort_by(|(a_key, a), (b_key, b)| {
        (b.bigram_occurrences + b.post_occurrences)
            .cmp(&(a.bigram_occurrences + a.post_occurrences))
            .then_with(|| {
                (b.bigram_sentences + b.post_sentences)
                    .cmp(&(a.bigram_sentences + a.post_sentences))
            })
            .then_with(|| a_key.wrong.cmp(&b_key.wrong))
            .then_with(|| a_key.gold.cmp(&b_key.gold))
            .then_with(|| a_key.wrong_pattern.cmp(&b_key.wrong_pattern))
    });

    let mut tsv = String::from(
        "rank\twrong_pattern\tgold_trigram\twrong_trigram\tbigram_occurrences\tbigram_sentences\tpost_trigram_occurrences\tpost_trigram_sentences\ttotal_occurrences\n",
    );
    let row_count = rows.len().min(100);
    for (index, (key, counts)) in rows.into_iter().take(100).enumerate() {
        let _ = writeln!(
            tsv,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            index + 1,
            tsv_cell(&key.wrong_pattern),
            tsv_cell(&display_trigram(&key.gold)),
            tsv_cell(&display_trigram(&key.wrong)),
            counts.bigram_occurrences,
            counts.bigram_sentences,
            counts.post_occurrences,
            counts.post_sentences,
            counts.bigram_occurrences + counts.post_occurrences,
        );
    }
    std::fs::write(path, tsv)?;
    Ok(row_count)
}

fn display_trigram(words: &[WordLabel; 3]) -> String {
    words
        .iter()
        .map(|word| format!("{}[{}]{{{}}}", word.surface, word.word_id, word.pos))
        .collect::<Vec<_>>()
        .join(" + ")
}

fn tsv_cell(value: &str) -> String {
    value
        .replace('\t', "\\t")
        .replace('\r', "\\r")
        .replace('\n', "\\n")
}

impl StageAccumulator {
    fn add_error(
        &mut self,
        gold: &PathDisplay,
        converted: &PathDisplay,
        exact: bool,
        page_id: u64,
        title: &str,
    ) {
        self.character_edit_distance += levenshtein_chars(&gold.text, &converted.text);
        self.character_denominator += gold
            .text
            .chars()
            .count()
            .max(converted.text.chars().count());
        if exact {
            return;
        }
        self.top1_error_sentences += 1;
        let hunks = char_diff(&gold.text, &converted.text);
        if hunks.is_empty() {
            self.same_text_but_different_path_sentences += 1;
            return;
        }

        self.character_different_error_sentences += 1;
        self.character_diff_hunks += hunks.len();
        let mut gold_windows = BTreeSet::new();
        let mut converted_windows = BTreeSet::new();
        let mut sentence_pairs = HashSet::new();
        for hunk in &hunks {
            let local_gold_windows = trigram_starts(gold, hunk.gold_start, hunk.gold_end);
            gold_windows.extend(local_gold_windows.iter().copied());
            let gold_tokens = touched_token_indices(gold, hunk.gold_start, hunk.gold_end);
            let wrong_tokens =
                touched_token_indices(converted, hunk.converted_start, hunk.converted_end);
            for (wrong_start, pattern) in paired_error_trigram_starts(converted, &wrong_tokens) {
                converted_windows.insert(wrong_start);
                let best_gold = local_gold_windows.iter().min_by(|&&a, &&b| {
                    let a_anchor = relative_window_anchor(
                        gold,
                        a,
                        &gold_tokens,
                        hunk.gold_start,
                        hunk.gold_end,
                    );
                    let b_anchor = relative_window_anchor(
                        gold,
                        b,
                        &gold_tokens,
                        hunk.gold_start,
                        hunk.gold_end,
                    );
                    let wrong_anchor = relative_window_anchor(
                        converted,
                        wrong_start,
                        &wrong_tokens,
                        hunk.converted_start,
                        hunk.converted_end,
                    );
                    (a_anchor - wrong_anchor)
                        .abs()
                        .total_cmp(&(b_anchor - wrong_anchor).abs())
                        .then_with(|| a.cmp(&b))
                });
                if let Some(&gold_start) = best_gold {
                    sentence_pairs.insert(TrigramPairKey {
                        wrong_pattern: pattern,
                        gold: trigram_at(gold, gold_start),
                        wrong: trigram_at(converted, wrong_start),
                    });
                }
            }
        }
        add_trigrams(&mut self.gold, gold, &gold_windows);
        add_trigrams(&mut self.converted, converted, &converted_windows);
        for pair in sentence_pairs {
            let count = self.pairs.entry(pair).or_insert(TrigramCount {
                occurrences: 0,
                error_sentences: 0,
            });
            count.occurrences += 1;
            count.error_sentences += 1;
        }
        if self.examples.len() < MAX_DIFF_EXAMPLES {
            self.examples.push(DiffExample {
                page_id,
                title: title.to_owned(),
                gold_text: gold.text.clone(),
                converted_text: converted.text.clone(),
                char_diff: hunks
                    .iter()
                    .map(|hunk| CharDiffHunk {
                        gold_char_range: [hunk.gold_start, hunk.gold_end],
                        converted_char_range: [hunk.converted_start, hunk.converted_end],
                        gold_text: chars_in(&gold.text, hunk.gold_start, hunk.gold_end),
                        converted_text: chars_in(
                            &converted.text,
                            hunk.converted_start,
                            hunk.converted_end,
                        ),
                    })
                    .collect(),
            });
        }
    }

    fn finish(self) -> StageReport {
        let character_level_accuracy = if self.character_denominator == 0 {
            1.0
        } else {
            1.0 - self.character_edit_distance as f64 / self.character_denominator as f64
        };
        StageReport {
            top1_error_sentences: self.top1_error_sentences,
            character_different_error_sentences: self.character_different_error_sentences,
            same_text_but_different_path_sentences: self.same_text_but_different_path_sentences,
            character_diff_hunks: self.character_diff_hunks,
            character_edit_distance: self.character_edit_distance,
            character_denominator: self.character_denominator,
            character_level_accuracy,
            gold_side_trigrams: rank(self.gold),
            converted_side_trigrams: rank(self.converted),
            paired_gold_vs_wrong_trigrams: rank_pairs(self.pairs),
            examples: self.examples,
        }
    }
}

fn path_display(model: &Model, sample: &Prepared, nodes: &[usize]) -> PathDisplay {
    let mut text = String::new();
    let mut labels = Vec::with_capacity(nodes.len());
    let mut token_chars = Vec::with_capacity(nodes.len().saturating_sub(2));
    labels.push(word_label(
        model,
        BOS,
        model.words[BOS as usize].surface.clone(),
    ));
    for &node_index in nodes
        .iter()
        .filter(|&&node| node != 0 && node + 1 != sample.graph.nodes.len())
    {
        let node = &sample.graph.nodes[node_index];
        let word = &model.words[node.word as usize];
        let surface = if word.model_only {
            sample.reading[node.start..node.end].to_owned()
        } else {
            word.surface.clone()
        };
        let start = text.chars().count();
        text.push_str(&surface);
        let end = start + surface.chars().count();
        token_chars.push(TokenChars { start, end });
        labels.push(word_label(model, node.word, surface));
    }
    labels.push(word_label(
        model,
        EOS,
        model.words[EOS as usize].surface.clone(),
    ));
    PathDisplay {
        text,
        labels,
        token_chars,
    }
}

fn word_label(model: &Model, word_id: u32, surface: String) -> WordLabel {
    let word = &model.words[word_id as usize];
    WordLabel {
        word_id,
        surface,
        pos: model.pos_names[word.pos as usize].clone(),
    }
}

fn char_diff(gold: &str, converted: &str) -> Vec<Hunk> {
    let gold: Vec<char> = gold.chars().collect();
    let converted: Vec<char> = converted.chars().collect();
    let cells = (gold.len() + 1).saturating_mul(converted.len() + 1);
    if cells > MAX_LCS_CELLS {
        return coarse_diff(&gold, &converted);
    }

    let columns = converted.len() + 1;
    let mut lcs = vec![0_u32; cells];
    for i in (0..gold.len()).rev() {
        for j in (0..converted.len()).rev() {
            lcs[i * columns + j] = if gold[i] == converted[j] {
                lcs[(i + 1) * columns + j + 1] + 1
            } else {
                lcs[(i + 1) * columns + j].max(lcs[i * columns + j + 1])
            };
        }
    }

    let (mut i, mut j) = (0, 0);
    let mut active: Option<(usize, usize)> = None;
    let mut hunks = Vec::new();
    while i < gold.len() || j < converted.len() {
        if i < gold.len() && j < converted.len() && gold[i] == converted[j] {
            if let Some((gold_start, converted_start)) = active.take() {
                hunks.push(Hunk {
                    gold_start,
                    gold_end: i,
                    converted_start,
                    converted_end: j,
                });
            }
            i += 1;
            j += 1;
        } else if i < gold.len()
            && (j == converted.len() || lcs[(i + 1) * columns + j] >= lcs[i * columns + j + 1])
        {
            active.get_or_insert((i, j));
            i += 1;
        } else {
            active.get_or_insert((i, j));
            j += 1;
        }
    }
    if let Some((gold_start, converted_start)) = active {
        hunks.push(Hunk {
            gold_start,
            gold_end: i,
            converted_start,
            converted_end: j,
        });
    }
    hunks
}

fn coarse_diff(gold: &[char], converted: &[char]) -> Vec<Hunk> {
    let prefix = gold
        .iter()
        .zip(converted)
        .take_while(|(a, b)| a == b)
        .count();
    let suffix = gold[prefix..]
        .iter()
        .rev()
        .zip(converted[prefix..].iter().rev())
        .take_while(|(a, b)| a == b)
        .count();
    if prefix == gold.len() && prefix == converted.len() {
        return Vec::new();
    }
    vec![Hunk {
        gold_start: prefix,
        gold_end: gold.len().saturating_sub(suffix),
        converted_start: prefix,
        converted_end: converted.len().saturating_sub(suffix),
    }]
}

fn trigram_starts(path: &PathDisplay, start: usize, end: usize) -> BTreeSet<usize> {
    let touched = touched_token_indices(path, start, end);
    let mut starts = BTreeSet::new();
    let last_start = path.labels.len().saturating_sub(3);
    for token_index in touched {
        let label_index = token_index + 1; // Account for the leading BOS label.
        let first = label_index.saturating_sub(2);
        let last = label_index.min(last_start);
        if first <= last {
            starts.extend(first..=last);
        }
    }
    starts
}

fn touched_token_indices(path: &PathDisplay, start: usize, end: usize) -> Vec<usize> {
    let mut touched = Vec::new();
    if start < end {
        touched.extend(
            path.token_chars
                .iter()
                .enumerate()
                .filter(|(_, span)| span.start < end && span.end > start)
                .map(|(index, _)| index),
        );
    } else if !path.token_chars.is_empty() {
        // Pure insertion/deletion: anchor the empty side at the adjacent token(s).
        let boundary = start;
        for (index, span) in path.token_chars.iter().enumerate() {
            if span.start <= boundary && boundary < span.end {
                touched.push(index);
            }
            if span.end == boundary || span.start == boundary {
                touched.push(index);
            }
        }
        if touched.is_empty() {
            touched.push(if boundary == 0 {
                0
            } else {
                path.token_chars.len() - 1
            });
        }
    }
    touched.sort_unstable();
    touched.dedup();
    touched
}

fn paired_error_trigram_starts(path: &PathDisplay, wrong_tokens: &[usize]) -> Vec<(usize, String)> {
    if wrong_tokens.is_empty() {
        return Vec::new();
    }
    let wrong: HashSet<usize> = wrong_tokens.iter().copied().collect();
    let allowed: &[&str] = if wrong_tokens.len() == 1 {
        &["CIC"]
    } else {
        &["CII", "III", "IIC"]
    };
    (0..path.labels.len().saturating_sub(2))
        .filter_map(|start| {
            let pattern: String = (start..start + 3)
                .map(|label_index| {
                    if label_index > 0
                        && label_index + 1 < path.labels.len()
                        && wrong.contains(&(label_index - 1))
                    {
                        'I'
                    } else {
                        'C'
                    }
                })
                .collect();
            allowed
                .contains(&pattern.as_str())
                .then_some((start, pattern))
        })
        .collect()
}

fn relative_window_anchor(
    path: &PathDisplay,
    trigram_start: usize,
    touched: &[usize],
    hunk_start: usize,
    hunk_end: usize,
) -> f64 {
    let centers: Vec<f64> = touched
        .iter()
        .filter(|&&token_index| {
            let label_index = token_index + 1;
            (trigram_start..trigram_start + 3).contains(&label_index)
        })
        .map(|&token_index| {
            let span = &path.token_chars[token_index];
            (span.start + span.end) as f64 / 2.0
        })
        .collect();
    if centers.is_empty() {
        return 0.5;
    }
    let center = centers.iter().sum::<f64>() / centers.len() as f64;
    if hunk_end <= hunk_start {
        0.5
    } else {
        ((center - hunk_start as f64) / (hunk_end - hunk_start) as f64).clamp(0.0, 1.0)
    }
}

fn trigram_at(path: &PathDisplay, start: usize) -> [WordLabel; 3] {
    std::array::from_fn(|offset| path.labels[start + offset].clone())
}

fn add_trigrams(
    counts: &mut HashMap<[WordLabel; 3], TrigramCount>,
    path: &PathDisplay,
    starts: &BTreeSet<usize>,
) {
    let mut sentence_counts: HashMap<[WordLabel; 3], usize> = HashMap::new();
    for &start in starts {
        let trigram = [
            path.labels[start].clone(),
            path.labels[start + 1].clone(),
            path.labels[start + 2].clone(),
        ];
        *sentence_counts.entry(trigram).or_default() += 1;
    }
    for (trigram, occurrences) in sentence_counts {
        let count = counts.entry(trigram).or_insert(TrigramCount {
            occurrences: 0,
            error_sentences: 0,
        });
        count.occurrences += occurrences;
        count.error_sentences += 1;
    }
}

fn rank(counts: HashMap<[WordLabel; 3], TrigramCount>) -> Vec<RankedTrigram> {
    let mut rows: Vec<_> = counts.into_iter().collect();
    rows.sort_by(|(a_key, a_count), (b_key, b_count)| {
        b_count
            .occurrences
            .cmp(&a_count.occurrences)
            .then_with(|| b_count.error_sentences.cmp(&a_count.error_sentences))
            .then_with(|| a_key.cmp(b_key))
    });
    rows.into_iter()
        .enumerate()
        .map(|(index, (trigram, count))| RankedTrigram {
            rank: index + 1,
            trigram,
            occurrences: count.occurrences,
            error_sentences: count.error_sentences,
        })
        .collect()
}

fn rank_pairs(counts: HashMap<TrigramPairKey, TrigramCount>) -> Vec<RankedTrigramPair> {
    let mut rows: Vec<_> = counts.into_iter().collect();
    rows.sort_by(|(a_key, a_count), (b_key, b_count)| {
        b_count
            .occurrences
            .cmp(&a_count.occurrences)
            .then_with(|| b_count.error_sentences.cmp(&a_count.error_sentences))
            .then_with(|| a_key.wrong.cmp(&b_key.wrong))
            .then_with(|| a_key.gold.cmp(&b_key.gold))
            .then_with(|| a_key.wrong_pattern.cmp(&b_key.wrong_pattern))
    });
    rows.into_iter()
        .enumerate()
        .map(|(index, (pair, count))| RankedTrigramPair {
            rank: index + 1,
            wrong_pattern: pair.wrong_pattern,
            gold_trigram: pair.gold,
            wrong_trigram: pair.wrong,
            occurrences: count.occurrences,
            error_sentences: count.error_sentences,
        })
        .collect()
}

fn chars_in(text: &str, start: usize, end: usize) -> String {
    text.chars()
        .skip(start)
        .take(end.saturating_sub(start))
        .collect()
}

fn levenshtein_chars(left: &str, right: &str) -> usize {
    let left: Vec<char> = left.chars().collect();
    let right: Vec<char> = right.chars().collect();
    if left.is_empty() {
        return right.len();
    }
    if right.is_empty() {
        return left.len();
    }
    let (outer, inner) = if left.len() >= right.len() {
        (&left, &right)
    } else {
        (&right, &left)
    };
    let mut previous: Vec<usize> = (0..=inner.len()).collect();
    let mut current = vec![0; inner.len() + 1];
    for (i, outer_char) in outer.iter().enumerate() {
        current[0] = i + 1;
        for (j, inner_char) in inner.iter().enumerate() {
            let substitution = previous[j] + usize::from(outer_char != inner_char);
            let deletion = previous[j + 1] + 1;
            let insertion = current[j] + 1;
            current[j + 1] = substitution.min(deletion).min(insertion);
        }
        std::mem::swap(&mut previous, &mut current);
    }
    previous[inner.len()]
}

#[cfg(test)]
mod tests {
    use super::{
        PathDisplay, TokenChars, WordLabel, char_diff, chars_in, levenshtein_chars,
        paired_error_trigram_starts, trigram_starts,
    };

    #[test]
    fn character_diff_finds_replacement_and_insertion_ranges() {
        let replacement = char_diff("猫が走る", "犬が走る");
        assert_eq!(replacement.len(), 1);
        assert_eq!(
            chars_in(
                "猫が走る",
                replacement[0].gold_start,
                replacement[0].gold_end
            ),
            "猫"
        );
        assert_eq!(
            chars_in(
                "犬が走る",
                replacement[0].converted_start,
                replacement[0].converted_end
            ),
            "犬"
        );

        let insertion = char_diff("猫が走る", "猫が早く走る");
        assert_eq!(insertion.len(), 1);
        assert_eq!(insertion[0].gold_start, insertion[0].gold_end);
        assert_eq!(
            chars_in(
                "猫が早く走る",
                insertion[0].converted_start,
                insertion[0].converted_end
            ),
            "早く"
        );
    }

    #[test]
    fn identical_text_has_no_character_diff() {
        assert!(char_diff("同じ", "同じ").is_empty());
    }

    #[test]
    fn a_changed_token_selects_every_containing_word_trigram() {
        let labels = ["[BOS]", "a", "b", "c", "d", "[EOS]"]
            .into_iter()
            .zip(0_u32..)
            .map(|(surface, word_id)| WordLabel {
                word_id,
                surface: surface.to_owned(),
                pos: "test".to_owned(),
            })
            .collect();
        let path = PathDisplay {
            text: "abcd".to_owned(),
            labels,
            token_chars: (0..4)
                .map(|index| TokenChars {
                    start: index,
                    end: index + 1,
                })
                .collect(),
        };
        assert_eq!(trigram_starts(&path, 1, 2), [0, 1, 2].into_iter().collect());
    }

    #[test]
    fn paired_error_trigrams_keep_only_cic_or_multiword_error_patterns() {
        let labels = ["[BOS]", "a", "b", "c", "d", "[EOS]"]
            .into_iter()
            .zip(0_u32..)
            .map(|(surface, word_id)| WordLabel {
                word_id,
                surface: surface.to_owned(),
                pos: "test".to_owned(),
            })
            .collect();
        let path = PathDisplay {
            text: "abcd".to_owned(),
            labels,
            token_chars: (0..4)
                .map(|index| TokenChars {
                    start: index,
                    end: index + 1,
                })
                .collect(),
        };

        let single = paired_error_trigram_starts(&path, &[1]);
        assert_eq!(single, vec![(1, "CIC".to_owned())]);

        let two = paired_error_trigram_starts(&path, &[1, 2]);
        assert_eq!(
            two.iter()
                .map(|(_, pattern)| pattern.as_str())
                .collect::<Vec<_>>(),
            vec!["CII", "IIC"]
        );

        let three = paired_error_trigram_starts(&path, &[1, 2, 3]);
        assert_eq!(
            three
                .iter()
                .map(|(_, pattern)| pattern.as_str())
                .collect::<Vec<_>>(),
            vec!["CII", "III", "IIC"]
        );
    }

    #[test]
    fn character_accuracy_distance_uses_unicode_scalars() {
        assert_eq!(levenshtein_chars("猫犬", "猫"), 1);
        assert_eq!(levenshtein_chars("かな", "カナ"), 2);
    }
}
