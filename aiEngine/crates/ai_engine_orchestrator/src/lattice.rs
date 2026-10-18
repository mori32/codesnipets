//! Dictionary query uses the edit graph and produces a separate word lattice.

use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;
use std::sync::Arc;

use crate::edit_line::{EditLine, TextRange};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InputEdge {
    pub start: usize,
    pub end: usize,
    pub reading: String,
    /// Some(surface) binds both reading and surface; it is not a ranking hint.
    pub display: Option<String>,
    pub searchable: bool,
    pub source: Arc<TextRange>,
    pub source_slice: Range<usize>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InputLattice {
    end: usize,
    edges: Vec<InputEdge>,
    outgoing: Vec<Vec<usize>>,
}

impl InputLattice {
    /// # Errors
    /// Rejects backward, empty, or out-of-bounds edges.
    pub fn new(end: usize, edges: Vec<InputEdge>) -> Result<Self, String> {
        let mut outgoing = vec![Vec::new(); end + 1];
        for (id, edge) in edges.iter().enumerate() {
            if edge.start >= edge.end || edge.end > end || edge.reading.is_empty() {
                return Err("invalid input lattice edge".into());
            }
            outgoing[edge.start].push(id);
        }
        Ok(Self {
            end,
            edges,
            outgoing,
        })
    }
    #[must_use]
    pub fn end(&self) -> usize {
        self.end
    }
    #[must_use]
    pub fn edges(&self) -> &[InputEdge] {
        &self.edges
    }
    #[must_use]
    pub fn outgoing(&self, frame: usize) -> &[usize] {
        self.outgoing.get(frame).map_or(&[], Vec::as_slice)
    }
}

/// Dictionary score is a higher-is-better potential; future CRF readers can
/// supply signed floating-point scores without converting them to `MeCab` cost.
#[derive(Clone, Debug, PartialEq)]
pub struct QueryWord {
    pub reading: String,
    pub surface: String,
    pub score: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct OutputEdge {
    pub start: usize,
    pub end: usize,
    pub word: QueryWord,
    pub input_edges: Vec<usize>,
    pub fallback: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct OutputLattice {
    pub end: usize,
    pub edges: Vec<OutputEdge>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct OutputPath {
    pub edges: Vec<usize>,
    pub text: String,
    pub score: f64,
}

impl OutputLattice {
    /// Classical bounded N-best for the P2 prototype. CRF/P3 decoding can replace
    /// this without changing the edit-buffer/dictionary query contract.
    /// # Errors
    /// Rejects invalid edges, references, and nonfinite scores.
    pub fn nbest(&self, input: &InputLattice, limit: usize) -> Result<Vec<OutputPath>, String> {
        self.validate(input)?;
        if limit == 0 || self.end == 0 {
            return Ok(Vec::new());
        }
        let width = limit.saturating_mul(8).max(8);
        let mut outgoing = vec![Vec::new(); self.end + 1];
        for (id, edge) in self.edges.iter().enumerate() {
            outgoing[edge.start].push(id);
        }
        let mut paths = vec![Vec::<OutputPath>::new(); self.end + 1];
        paths[0].push(OutputPath {
            edges: Vec::new(),
            text: String::new(),
            score: 0.0,
        });
        for frame in 0..self.end {
            paths[frame].sort_by(|a, b| {
                b.score
                    .total_cmp(&a.score)
                    .then_with(|| a.edges.cmp(&b.edges))
            });
            paths[frame].truncate(width);
            let current = std::mem::take(&mut paths[frame]);
            for path in current {
                for &id in &outgoing[frame] {
                    let edge = &self.edges[id];
                    let mut next = path.clone();
                    next.edges.push(id);
                    next.text.push_str(&edge.word.surface);
                    next.score += edge.word.score;
                    if !next.score.is_finite() {
                        return Err("path score overflow".into());
                    }
                    paths[edge.end].push(next);
                    if paths[edge.end].len() > width * 2 {
                        paths[edge.end].sort_by(|a, b| {
                            b.score
                                .total_cmp(&a.score)
                                .then_with(|| a.edges.cmp(&b.edges))
                        });
                        paths[edge.end].truncate(width);
                    }
                }
            }
        }
        let mut result = std::mem::take(&mut paths[self.end]);
        result.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| a.edges.cmp(&b.edges))
        });
        let mut seen = BTreeSet::new();
        result.retain(|path| seen.insert(path.text.clone()));
        result.truncate(limit);
        Ok(result)
    }

    /// # Errors
    /// Validates that every output word follows real input links and respects bindings.
    pub fn validate(&self, input: &InputLattice) -> Result<(), String> {
        if self.end != input.end {
            return Err("output lattice frame mismatch".into());
        }
        for edge in &self.edges {
            if edge.start >= edge.end
                || edge.end > self.end
                || !edge.word.score.is_finite()
                || edge.word.surface.is_empty()
                || edge.input_edges.is_empty()
            {
                return Err("invalid output lattice word".into());
            }
            let mut frame = edge.start;
            let mut reading = String::new();
            for &id in &edge.input_edges {
                let source = input.edges.get(id).ok_or("invalid input edge reference")?;
                if source.start != frame {
                    return Err("output word crosses disconnected input".into());
                }
                if let Some(display) = &source.display {
                    if edge.input_edges.len() != 1 || edge.word.surface != *display {
                        return Err("output word changes a bound surface".into());
                    }
                }
                if !source.searchable
                    && (edge.input_edges.len() != 1 || edge.word.surface != source.reading)
                {
                    return Err("output word changes literal input".into());
                }
                frame = source.end;
                reading.push_str(&source.reading);
            }
            if frame != edge.end || reading != edge.word.reading {
                return Err("output word reading mismatch".into());
            }
        }
        Ok(())
    }
}

/// P2 supplies the buffer itself. The dictionary owns structural traversal and
/// produces word nodes; P2 never converts this call to a flat string query.
pub trait DictionaryQuery {
    /// # Errors
    /// Reports reader failure or an invalid word lattice.
    fn query(&self, input: &EditLine, context: &str) -> Result<OutputLattice, String>;
}

#[derive(Clone, Debug, Default)]
struct TrieNode {
    next: BTreeMap<char, usize>,
    words: Vec<QueryWord>,
}

/// Small in-memory reference reader for development. The production mmap reader
/// can implement `DictionaryQuery` without depending on this trie or its scores.
#[derive(Clone, Debug)]
pub struct MemoryDictionary {
    trie: Vec<TrieNode>,
}

impl MemoryDictionary {
    /// # Errors
    /// Rejects empty words and nonfinite potentials.
    pub fn new(words: impl IntoIterator<Item = QueryWord>) -> Result<Self, String> {
        let mut trie = vec![TrieNode::default()];
        for word in words {
            if word.reading.is_empty() || word.surface.is_empty() || !word.score.is_finite() {
                return Err("invalid dictionary word".into());
            }
            let mut node = 0;
            for ch in word.reading.chars() {
                node = if let Some(&next) = trie[node].next.get(&ch) {
                    next
                } else {
                    let next = trie.len();
                    trie.push(TrieNode::default());
                    trie[node].next.insert(ch, next);
                    next
                };
            }
            trie[node].words.push(word);
        }
        Ok(Self { trie })
    }

    /// Debug string entry point follows exactly the same structural query path.
    /// The text is a literal reading; romaji keyboard input uses `EditLine` instead.
    /// # Errors
    /// Returns the same errors as querying the generated input lattice.
    pub fn query_reading(&self, reading: &str) -> Result<OutputLattice, String> {
        let mut line = EditLine::default();
        line.replace(reading, crate::TextLayer::Phonetic);
        self.query(&line, "")
    }

    /// Walks alternatives and multi-character input links directly through the trie.
    /// # Errors
    /// Rejects excessive structural expansion, rather than flattening or dropping branches.
    pub fn query_lattice(&self, input: &InputLattice) -> Result<OutputLattice, String> {
        let mut output = OutputLattice {
            end: input.end,
            edges: Vec::new(),
        };
        let mut expansions = 0;
        for start in 0..input.end {
            let mut stack = vec![(start, 0, Vec::<usize>::new())];
            while let Some((frame, node, path)) = stack.pop() {
                for &id in input.outgoing(frame) {
                    expansions += 1;
                    if expansions > 1_000_000 {
                        return Err("dictionary lattice traversal limit exceeded".into());
                    }
                    let edge = &input.edges[id];
                    if !edge.searchable || (edge.display.is_some() && !path.is_empty()) {
                        continue;
                    }
                    let mut cursor = Some(node);
                    for ch in edge.reading.chars() {
                        cursor = cursor.and_then(|index| self.trie[index].next.get(&ch).copied());
                    }
                    let Some(cursor) = cursor else {
                        continue;
                    };
                    let mut next_path = path.clone();
                    next_path.push(id);
                    for word in &self.trie[cursor].words {
                        if edge
                            .display
                            .as_ref()
                            .is_none_or(|display| *display == word.surface)
                        {
                            output.edges.push(OutputEdge {
                                start,
                                end: edge.end,
                                word: word.clone(),
                                input_edges: next_path.clone(),
                                fallback: false,
                            });
                        }
                    }
                    if edge.display.is_none() && edge.end < input.end {
                        stack.push((edge.end, cursor, next_path));
                    }
                }
            }
        }
        // Preserve literal/unknown input and bound words missing from the reader.
        let single_matches: BTreeSet<_> = output
            .edges
            .iter()
            .filter(|word| word.input_edges.len() == 1)
            .map(|word| (word.input_edges[0], word.word.surface.clone()))
            .collect();
        for (id, edge) in input.edges.iter().enumerate() {
            let surface = edge.display.as_ref().unwrap_or(&edge.reading);
            if !single_matches.contains(&(id, surface.clone())) {
                output.edges.push(OutputEdge {
                    start: edge.start,
                    end: edge.end,
                    word: QueryWord {
                        reading: edge.reading.clone(),
                        surface: surface.clone(),
                        score: -1000.0,
                    },
                    input_edges: vec![id],
                    fallback: true,
                });
            }
        }
        output.validate(input)?;
        Ok(output)
    }
}

impl DictionaryQuery for MemoryDictionary {
    fn query(&self, input: &EditLine, _context: &str) -> Result<OutputLattice, String> {
        self.query_lattice(&input.input_lattice())
    }
}
