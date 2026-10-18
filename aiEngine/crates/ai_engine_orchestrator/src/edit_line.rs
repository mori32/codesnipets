//! Layered composition buffer, inspired by Ribbon's EditLine/subRange model.

use std::ops::Range;
use std::sync::Arc;

use crate::lattice::{InputEdge, InputLattice};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TextLayer {
    Raw,
    Phonetic,
    Converted,
}

/// An immutable node in the growing edit graph. Replacements keep their sources.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TextRange {
    pub text: String,
    pub layer: TextLayer,
    pub sources: Vec<Arc<TextRange>>,
    /// A visual fragment of an earlier node, used when editing inside a range.
    /// No one-to-one raw/phonetic alignment is invented for shi -> し.
    pub source_slice: Option<Range<usize>>,
}

impl TextRange {
    #[must_use]
    pub fn len(&self) -> usize {
        self.text.chars().count()
    }
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }

    fn query_reading(&self) -> String {
        if self.layer == TextLayer::Converted && !self.sources.is_empty() {
            self.sources
                .iter()
                .map(|source| source.query_reading())
                .collect()
        } else {
            self.text.clone()
        }
    }
}

/// The frontier is the visible composition. Earlier layers remain reachable
/// through source links; cursor and selection use visible Unicode scalar positions.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct EditLine {
    ranges: Vec<Arc<TextRange>>,
    caret: usize,
    selection_len: usize,
}

impl EditLine {
    #[must_use]
    pub fn ranges(&self) -> &[Arc<TextRange>] {
        &self.ranges
    }
    #[must_use]
    pub fn composition(&self) -> String {
        self.ranges.iter().map(|node| node.text.as_str()).collect()
    }
    #[must_use]
    pub fn len(&self) -> usize {
        self.ranges.iter().map(|node| node.len()).sum()
    }
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }
    #[must_use]
    pub fn caret(&self) -> (usize, usize) {
        (self.caret, self.selection_len)
    }

    /// # Errors
    /// Rejects a selection outside the visible composition.
    pub fn set_caret(&mut self, position: usize, length: usize) -> Result<(), String> {
        if position
            .checked_add(length)
            .is_none_or(|end| end > self.len())
        {
            return Err("selection outside edit line".into());
        }
        // A converted word has an exact reading/surface binding, but no invented
        // per-kanji reading alignment. Peel it before editing inside that word.
        let mut top = 0;
        for node in &self.ranges {
            let end = top + node.len();
            if node.layer == TextLayer::Converted
                && ((top < position && position < end)
                    || (top < position + length && position + length < end))
            {
                return Err("cancel conversion before editing inside a bound word".into());
            }
            top = end;
        }
        self.caret = position;
        self.selection_len = length;
        Ok(())
    }

    pub fn clear(&mut self) {
        self.ranges.clear();
        self.caret = 0;
        self.selection_len = 0;
    }

    /// Replaces the current visual selection, retaining the replaced nodes below it.
    pub fn replace(&mut self, text: &str, layer: TextLayer) {
        let start = self.split_at(self.caret);
        let end = self.split_at(self.caret + self.selection_len);
        let sources: Vec<_> = self.ranges.drain(start..end).collect();
        if !text.is_empty() {
            if layer == TextLayer::Raw {
                self.ranges.splice(
                    start..start,
                    text.chars().map(|ch| {
                        Arc::new(TextRange {
                            text: ch.to_string(),
                            layer,
                            sources: Vec::new(),
                            source_slice: None,
                        })
                    }),
                );
            } else {
                self.ranges.insert(
                    start,
                    Arc::new(TextRange {
                        text: text.into(),
                        layer,
                        sources,
                        source_slice: None,
                    }),
                );
            }
        }
        self.caret += text.chars().count();
        self.selection_len = 0;
    }

    /// Input characters are individual leaves at the bottom layer.
    pub fn insert_raw(&mut self, text: &str) {
        if self.selection_len > 0 {
            self.replace("", TextLayer::Raw);
        }
        for ch in text.chars() {
            self.replace(&ch.to_string(), TextLayer::Raw);
        }
    }

    pub fn backspace(&mut self) {
        if self
            .ranges
            .iter()
            .any(|node| node.layer == TextLayer::Converted)
        {
            self.cancel_conversion();
            return;
        }
        if self.selection_len == 0 && self.caret > 0 {
            self.caret -= 1;
            self.selection_len = 1;
        }
        self.replace("", TextLayer::Raw);
    }

    pub fn delete(&mut self) {
        if self
            .ranges
            .iter()
            .any(|node| node.layer == TextLayer::Converted)
        {
            self.cancel_conversion();
            return;
        }
        if self.selection_len == 0 && self.caret < self.len() {
            self.selection_len = 1;
        }
        self.replace("", TextLayer::Raw);
    }

    /// Peels converted nodes back to their phonetic sources. This restores the
    /// previous frontier, rather than rerunning the raw keystroke history.
    pub fn cancel_conversion(&mut self) {
        fn peel(node: &Arc<TextRange>) -> Vec<Arc<TextRange>> {
            if node.layer == TextLayer::Converted && !node.sources.is_empty() {
                node.sources.iter().flat_map(peel).collect()
            } else {
                vec![Arc::clone(node)]
            }
        }
        self.ranges = self.ranges.iter().flat_map(peel).collect();
        self.caret = self.len();
        self.selection_len = 0;
    }

    /// Converts only contiguous raw input immediately before the caret.
    /// Already phonetic text is a boundary and is never reinterpreted as romaji.
    pub fn normalize_before_caret(&mut self, finish: bool) {
        let end = self.split_at(self.caret);
        let mut start = end;
        while start > 0 && self.ranges[start - 1].layer == TextLayer::Raw {
            start -= 1;
        }
        if start == end {
            return;
        }
        let text: String = self.ranges[start..end]
            .iter()
            .map(|node| node.text.as_str())
            .collect();
        let top: usize = self.ranges[..start].iter().map(|node| node.len()).sum();
        let steps = crate::romaji::steps(&text, finish);
        let new_length = steps
            .iter()
            .map(|(length, converted)| {
                converted
                    .as_ref()
                    .map_or(*length, |value| value.chars().count())
            })
            .sum::<usize>();
        let mut new_ranges = Vec::new();
        let mut source_offset = 0;
        for (raw_length, converted) in steps {
            // Raw leaves are one character each; a split fragment is also scalar sized.
            let sources =
                self.ranges[start + source_offset..start + source_offset + raw_length].to_vec();
            source_offset += raw_length;
            if let Some(converted) = converted {
                new_ranges.push(Arc::new(TextRange {
                    text: converted,
                    layer: TextLayer::Phonetic,
                    sources,
                    source_slice: None,
                }));
            } else {
                new_ranges.extend(sources);
            }
        }
        self.ranges.splice(start..end, new_ranges);
        // Recompute from the converted raw prefix, leaving the suffix untouched.
        self.caret = top + new_length;
        self.selection_len = 0;
    }

    pub(crate) fn katakana_display(&mut self) {
        for node in &mut self.ranges {
            if node.layer != TextLayer::Phonetic {
                continue;
            }
            let text: String = node
                .text
                .chars()
                .map(|ch| {
                    let code = u32::from(ch);
                    if (0x3041..=0x3096).contains(&code) {
                        char::from_u32(code + 0x60).unwrap_or(ch)
                    } else {
                        ch
                    }
                })
                .collect();
            if text != node.text {
                *node = Arc::new(TextRange {
                    text,
                    layer: TextLayer::Phonetic,
                    sources: vec![Arc::clone(node)],
                    source_slice: None,
                });
            }
        }
    }

    /// A structural projection for dictionary traversal. The buffer remains the
    /// query object; no joined reading string is used as the dictionary interface.
    ///
    /// # Panics
    /// Panics only if the internal forward-edge invariant is violated.
    #[must_use]
    pub fn input_lattice(&self) -> InputLattice {
        let mut edges = Vec::new();
        let mut position = 0;
        for node in &self.ranges {
            if node.layer == TextLayer::Converted {
                let reading = node.query_reading();
                let end = position + reading.chars().count();
                edges.push(InputEdge {
                    start: position,
                    end,
                    reading,
                    display: Some(node.text.clone()),
                    source: Arc::clone(node),
                    source_slice: 0..node.len(),
                    searchable: true,
                });
                position = end;
                continue;
            }
            for (offset, ch) in node.text.chars().enumerate() {
                edges.push(InputEdge {
                    start: position,
                    end: position + 1,
                    reading: ch.to_string(),
                    display: None,
                    source: Arc::clone(node),
                    source_slice: offset..offset + 1,
                    searchable: node.layer == TextLayer::Phonetic,
                });
                position += 1;
            }
        }
        InputLattice::new(position, edges).expect("edit line creates forward edges")
    }

    /// Replaces the entire phonetic frontier with selected dictionary words.
    /// Each word retains exactly its source span, including long/short segmentation.
    /// # Errors
    /// Rejects gaps, reversed spans, and mismatches with the current phonetic text.
    pub fn apply_words(&mut self, words: &[(Range<usize>, String, String)]) -> Result<(), String> {
        let mut next = self.clone();
        next.cancel_conversion();
        let text = next.composition();
        let mut previous = 0;
        for (span, _surface, reading) in words {
            if span.start != previous
                || span.end <= span.start
                || span.end > next.len()
                || text
                    .chars()
                    .skip(span.start)
                    .take(span.len())
                    .collect::<String>()
                    != *reading
            {
                return Err("word path does not match edit line".into());
            }
            previous = span.end;
        }
        if previous != next.len() {
            return Err("word path does not cover edit line".into());
        }
        for (span, surface, _reading) in words.iter().rev() {
            next.set_caret(span.start, span.len())?;
            next.replace(surface, TextLayer::Converted);
        }
        next.set_caret(next.len(), 0)?;
        *self = next;
        Ok(())
    }

    fn split_at(&mut self, position: usize) -> usize {
        let mut top = 0;
        for index in 0..self.ranges.len() {
            let node = Arc::clone(&self.ranges[index]);
            if top == position {
                return index;
            }
            let end = top + node.len();
            if position < end {
                let offset = position - top;
                let left = fragment(&node, 0..offset);
                let right = fragment(&node, offset..node.len());
                self.ranges.splice(index..=index, [left, right]);
                return index + 1;
            }
            top = end;
        }
        self.ranges.len()
    }
}

fn fragment(source: &Arc<TextRange>, span: Range<usize>) -> Arc<TextRange> {
    Arc::new(TextRange {
        text: source
            .text
            .chars()
            .skip(span.start)
            .take(span.len())
            .collect(),
        layer: source.layer,
        sources: vec![Arc::clone(source)],
        source_slice: Some(span),
    })
}
