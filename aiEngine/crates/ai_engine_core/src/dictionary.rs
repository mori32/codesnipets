//! Built-in and user dictionary support.

use std::collections::HashMap;
use std::fmt::Write as FmtWrite;
use std::fs;
use std::io;
use std::path::Path;

use crate::normalize::to_hiragana;
use crate::static_dictionary::{ConnectionMatrix, StaticDictionary};

/// A dictionary entry. Larger priority and frequency make an entry more likely.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Word {
    pub reading: String,
    pub surface: String,
    pub priority: i32,
    pub frequency: u32,
    pub left_id: u32,
    pub right_id: u32,
    pub cost: i32,
}

impl Word {
    fn new(reading: &str, surface: &str, priority: i32) -> Self {
        Self {
            reading: reading.to_owned(),
            surface: surface.to_owned(),
            priority,
            frequency: 0,
            left_id: 0,
            right_id: 0,
            cost: 0,
        }
    }
}

/// A compact dictionary container. User entries are kept separately so that
/// persisting the user dictionary never writes the built-in lexicon.
#[derive(Clone, Debug)]
pub struct Dictionary {
    builtins: Vec<Word>,
    external_words: Vec<Word>,
    user_words: Vec<Word>,
    index: HashMap<char, Vec<Word>>,
    connection: Option<ConnectionMatrix>,
}

impl Default for Dictionary {
    fn default() -> Self {
        Self::new()
    }
}

impl Dictionary {
    /// Creates a dictionary with the small Phase 1 built-in lexicon.
    #[must_use]
    pub fn new() -> Self {
        let mut dictionary = Self {
            builtins: builtin_words(),
            external_words: Vec::new(),
            user_words: Vec::new(),
            index: HashMap::new(),
            connection: None,
        };
        dictionary.rebuild_index();
        dictionary
    }

    pub(crate) fn matching_words(&self, input: &[char], start: usize) -> Vec<Word> {
        let Some(first) = input.get(start) else {
            return Vec::new();
        };
        self.index
            .get(first)
            .into_iter()
            .flat_map(|words| words.iter())
            .filter(|word| starts_with(input, start, &word.reading))
            .cloned()
            .collect()
    }

    /// Adds or replaces a user dictionary entry.
    pub fn add_user_word(&mut self, reading: &str, surface: &str, priority: i32) {
        let reading = to_hiragana(reading);
        if let Some(word) = self
            .user_words
            .iter_mut()
            .find(|word| word.reading == reading && word.surface == surface)
        {
            word.priority = priority;
            self.rebuild_index();
            return;
        }
        self.user_words.push(Word::new(&reading, surface, priority));
        self.rebuild_index();
    }

    /// Loads an external dictionary TSV without including it in user learning.
    ///
    /// # Errors
    ///
    /// Returns an I/O error for unreadable files or malformed rows.
    pub fn load_dictionary(&mut self, path: impl AsRef<Path>) -> io::Result<()> {
        self.external_words.extend(parse_tsv(path)?);
        self.rebuild_index();
        Ok(())
    }

    /// Loads the provisional P5 binary dictionary without adding entries to
    /// user learning data.
    ///
    /// The current format is decoded into memory. The public contract is kept
    /// separate so the storage can later be replaced by mmap and a trie.
    ///
    /// # Errors
    ///
    /// Returns an I/O or invalid-data error when the binary dictionary cannot
    /// be loaded or validated.
    pub fn load_static_dictionary(&mut self, path: impl AsRef<Path>) -> io::Result<()> {
        let dictionary = StaticDictionary::load(path)?;
        self.external_words
            .extend(dictionary.words().iter().cloned());
        self.connection =
            (dictionary.connection().dimension() > 0).then(|| dictionary.connection().clone());
        self.rebuild_index();
        Ok(())
    }

    pub(crate) fn connection_cost(&self, right_id: u32, left_id: u32) -> i32 {
        self.connection
            .as_ref()
            .and_then(|matrix| matrix.cost(right_id, left_id))
            .unwrap_or(0)
    }

    pub(crate) fn record_selection(&mut self, word: &Word) {
        if let Some(user_word) = self.user_words.iter_mut().find(|candidate| {
            candidate.reading == word.reading && candidate.surface == word.surface
        }) {
            user_word.frequency = user_word.frequency.saturating_add(1);
            self.rebuild_index();
            return;
        }
        let mut learned = word.clone();
        learned.frequency = 1;
        learned.priority = 0;
        learned.left_id = 0;
        learned.right_id = 0;
        learned.cost = 0;
        self.user_words.push(learned);
        self.rebuild_index();
    }

    /// Saves user entries as UTF-8 TSV: reading, surface, priority, frequency.
    ///
    /// # Errors
    ///
    /// Returns an I/O error when the destination cannot be created or written.
    pub fn save_user_dictionary(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let mut data = String::from("# aiEngine user dictionary v1\n");
        for word in &self.user_words {
            let _ = writeln!(
                data,
                "{}\t{}\t{}\t{}",
                word.reading, word.surface, word.priority, word.frequency
            );
        }
        fs::write(path, data)
    }

    /// Loads user entries from UTF-8 TSV. Missing files are handled by the CLI.
    ///
    /// # Errors
    ///
    /// Returns an I/O error for unreadable files or malformed rows.
    pub fn load_user_dictionary(&mut self, path: impl AsRef<Path>) -> io::Result<()> {
        for word in parse_tsv(path)? {
            if let Some(existing) = self.user_words.iter_mut().find(|existing| {
                existing.reading == word.reading && existing.surface == word.surface
            }) {
                *existing = word;
            } else {
                self.user_words.push(word);
            }
        }
        self.rebuild_index();
        Ok(())
    }

    fn rebuild_index(&mut self) {
        self.index.clear();
        for word in self
            .builtins
            .iter()
            .chain(self.external_words.iter())
            .chain(self.user_words.iter())
        {
            if let Some(first) = word.reading.chars().next() {
                self.index.entry(first).or_default().push(word.clone());
            }
        }
    }
}

pub(crate) fn parse_tsv(path: impl AsRef<Path>) -> io::Result<Vec<Word>> {
    let data = fs::read_to_string(path)?;
    let mut words = Vec::new();
    for (line_number, line) in data.lines().enumerate() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields: Vec<_> = line.split('\t').collect();
        if fields.len() != 4 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid dictionary row at line {}", line_number + 1),
            ));
        }
        let priority = fields[2].parse::<i32>().map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid priority at line {}: {error}", line_number + 1),
            )
        })?;
        let frequency = fields[3].parse::<u32>().map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid frequency at line {}: {error}", line_number + 1),
            )
        })?;
        words.push(Word {
            reading: to_hiragana(fields[0]),
            surface: fields[1].to_owned(),
            priority,
            frequency,
            left_id: 0,
            right_id: 0,
            cost: 0,
        });
    }
    Ok(words)
}

fn starts_with(input: &[char], start: usize, reading: &str) -> bool {
    let word: Vec<char> = reading.chars().collect();
    input
        .get(start..start.saturating_add(word.len()))
        .is_some_and(|slice| slice == word.as_slice())
}

fn builtin_words() -> Vec<Word> {
    [
        ("こんにちは", "今日は", 5),
        ("こんにちは", "こんにちは", 1),
        ("せかい", "世界", 20),
        ("にほん", "日本", 30),
        ("にほんご", "日本語", 50),
        ("ご", "語", 4),
        ("へんかん", "変換", 25),
        ("かな", "仮名", 15),
        ("かんじ", "漢字", 25),
        ("かなかんじ", "かな漢字", 20),
        ("かなかんじへんかん", "かな漢字変換", 80),
        ("とうきょう", "東京", 30),
        ("だいがく", "大学", 20),
        ("わたし", "私", 15),
        ("わたし", "わたし", 2),
        ("は", "は", 10),
        ("を", "を", 10),
        ("に", "に", 10),
        ("へ", "へ", 10),
        ("です", "です", 10),
        ("ます", "ます", 10),
        ("ありがとう", "有難う", 8),
        ("ありがとう", "ありがとう", 12),
    ]
    .into_iter()
    .map(|(reading, surface, priority)| Word::new(reading, surface, priority))
    .collect()
}
