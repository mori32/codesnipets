//! Provisional offline static-dictionary format.
//!
//! The format is intentionally dependency-free and easy to replace. It is
//! loaded into memory today; a later P5 implementation can keep the same
//! compile/load contract while changing the runtime storage to mmap and a
//! double-array trie.

use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use crate::Word;
use crate::dictionary::parse_tsv;

const MAGIC_V1: &[u8; 8] = b"AIDICT\0\x01";
const MAGIC_V2: &[u8; 8] = b"AIDICT\0\x02";

/// Dense row-major connection-cost matrix.
///
/// `cost(right_id, left_id)` follows Mozc's convention: the previous word's
/// right context selects the row and the next word's left context selects the
/// column.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ConnectionMatrix {
    dimension: u32,
    costs: Vec<i32>,
}

impl ConnectionMatrix {
    /// Creates a validated square matrix from row-major costs.
    ///
    /// # Errors
    ///
    /// Returns an invalid-data error when the cost count is not exactly the
    /// square of `dimension`.
    pub fn new(dimension: u32, costs: Vec<i32>) -> io::Result<Self> {
        let dimension_usize = usize::try_from(dimension)
            .map_err(|_| invalid_data("connection matrix dimension is invalid"))?;
        let expected = dimension_usize
            .checked_mul(dimension_usize)
            .ok_or_else(|| invalid_data("connection matrix size overflow"))?;
        if costs.len() != expected {
            return Err(invalid_data("connection matrix is not square"));
        }
        Ok(Self { dimension, costs })
    }

    /// Loads a plain Mozc `connection_single_column.txt` file.
    ///
    /// # Errors
    ///
    /// Returns an I/O or invalid-data error when the file is unreadable,
    /// contains a non-integer value, or does not contain a square matrix.
    pub fn from_text_file(path: impl AsRef<Path>) -> io::Result<Self> {
        let data = fs::read_to_string(path)?;
        let mut numbers = Vec::new();
        for raw_line in data.lines() {
            let line = raw_line.trim_start_matches('\u{feff}');
            let line = line.split_once('#').map_or(line, |(line, _)| line);
            for value in line.split_whitespace() {
                numbers.push(
                    value.parse::<i32>().map_err(|_| {
                        invalid_data("connection matrix contains a non-integer value")
                    })?,
                );
            }
        }
        let (dimension, costs) = numbers
            .split_first()
            .ok_or_else(|| invalid_data("connection matrix is empty"))?;
        let dimension = *dimension;
        if dimension < 0 {
            return Err(invalid_data("connection matrix dimension is negative"));
        }
        let dimension = u32::try_from(dimension)
            .map_err(|_| invalid_data("connection matrix dimension is invalid"))?;
        Self::new(dimension, costs.to_vec())
    }

    #[must_use]
    pub const fn dimension(&self) -> u32 {
        self.dimension
    }

    #[must_use]
    pub fn cost(&self, right_id: u32, left_id: u32) -> Option<i32> {
        if right_id >= self.dimension || left_id >= self.dimension {
            return None;
        }
        let dimension = usize::try_from(self.dimension).ok()?;
        let right = usize::try_from(right_id).ok()?;
        let left = usize::try_from(left_id).ok()?;
        self.costs
            .get(right.checked_mul(dimension)?.checked_add(left)?)
            .copied()
    }

    #[must_use]
    pub fn costs(&self) -> &[i32] {
        &self.costs
    }
}

/// A validated static dictionary loaded from the provisional binary format.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StaticDictionary {
    words: Vec<Word>,
    connection: ConnectionMatrix,
}

/// Counts produced by the provisional Mozc text importer.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MozcImportStats {
    pub files: usize,
    pub rows: usize,
    pub unique_entries: usize,
    pub matrix_dimension: u32,
}

impl StaticDictionary {
    /// Compiles a UTF-8 four-column TSV dictionary into a binary dictionary.
    ///
    /// The TSV columns are `reading`, `surface`, `priority`, and `frequency`.
    /// The resulting file contains no user-learning data beyond the frequency
    /// values explicitly present in the source TSV.
    ///
    /// # Errors
    ///
    /// Returns an I/O or validation error when the source cannot be read or a
    /// field cannot be represented by the provisional format.
    pub fn compile_tsv(
        source_path: impl AsRef<Path>,
        destination_path: impl AsRef<Path>,
    ) -> io::Result<()> {
        let words = parse_tsv(source_path)?;
        let data = encode(&words, &ConnectionMatrix::default())?;
        fs::write(destination_path, data)
    }

    /// Compiles `dictionary00.txt` through `dictionary99.txt` files in a
    /// Mozc `dictionary_oss` directory into the provisional P5 format.
    ///
    /// Mozc rows have the form `reading<TAB>left_id<TAB>right_id<TAB>cost<TAB>surface`.
    /// The context IDs and lexical cost are stored in the P5 entry. The core
    /// converts the lower-is-better cost into its higher-is-better score when
    /// evaluating a path.
    ///
    /// # Errors
    ///
    /// Returns an I/O or invalid-data error when the directory is unreadable,
    /// no dictionary files are found, or a row is malformed.
    pub fn compile_mozc_directory(
        source_directory: impl AsRef<Path>,
        destination_path: impl AsRef<Path>,
    ) -> io::Result<MozcImportStats> {
        Self::compile_mozc_directory_impl(
            source_directory.as_ref(),
            None,
            destination_path.as_ref(),
        )
    }

    /// Compiles Mozc dictionary text with an explicitly decompressed
    /// `connection_single_column.txt` path.
    ///
    /// # Errors
    ///
    /// Returns an I/O or invalid-data error when the source, connection
    /// matrix, or destination cannot be read or written.
    pub fn compile_mozc_directory_with_connection(
        source_directory: impl AsRef<Path>,
        connection_path: impl AsRef<Path>,
        destination_path: impl AsRef<Path>,
    ) -> io::Result<MozcImportStats> {
        Self::compile_mozc_directory_impl(
            source_directory.as_ref(),
            Some(connection_path.as_ref()),
            destination_path.as_ref(),
        )
    }

    fn compile_mozc_directory_impl(
        source_directory: &Path,
        connection_path: Option<&Path>,
        destination_path: &Path,
    ) -> io::Result<MozcImportStats> {
        let paths = mozc_dictionary_paths(source_directory)?;
        let connection = connection_path.map_or_else(
            || load_mozc_connection(source_directory),
            ConnectionMatrix::from_text_file,
        )?;
        let mut entries = BTreeMap::new();
        let mut stats = MozcImportStats {
            files: paths.len(),
            matrix_dimension: connection.dimension(),
            ..MozcImportStats::default()
        };
        for path in paths {
            parse_mozc_file(&path, &mut entries, &mut stats.rows)?;
        }
        stats.unique_entries = entries.len();
        let words = entries.into_values().collect::<Vec<_>>();
        fs::write(destination_path, encode(&words, &connection)?)?;
        Ok(stats)
    }

    /// Loads and validates a compiled static dictionary.
    ///
    /// # Errors
    ///
    /// Returns an I/O or invalid-data error for a truncated, incompatible, or
    /// invalid dictionary file.
    pub fn load(path: impl AsRef<Path>) -> io::Result<Self> {
        let data = fs::read(path)?;
        decode(&data)
    }

    /// Returns the entries in file order.
    #[must_use]
    pub fn words(&self) -> &[Word] {
        &self.words
    }

    /// Returns the connection matrix embedded in the dictionary.
    #[must_use]
    pub const fn connection(&self) -> &ConnectionMatrix {
        &self.connection
    }
}

fn load_mozc_connection(source_directory: &Path) -> io::Result<ConnectionMatrix> {
    let text_path = source_directory.join("connection_single_column.txt");
    if text_path.is_file() {
        return ConnectionMatrix::from_text_file(text_path);
    }
    let deflate_path = source_directory.join("connection.deflate");
    if deflate_path.is_file() {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            format!(
                "{} is compressed; decompress it to connection_single_column.txt first",
                deflate_path.display()
            ),
        ));
    }
    Ok(ConnectionMatrix::default())
}

fn mozc_dictionary_paths(source_directory: impl AsRef<Path>) -> io::Result<Vec<PathBuf>> {
    let mut paths = fs::read_dir(source_directory)?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| {
            let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
                return false;
            };
            let Some(index) = name
                .strip_prefix("dictionary")
                .and_then(|name| name.strip_suffix(".txt"))
            else {
                return false;
            };
            index.len() == 2 && index.chars().all(|character| character.is_ascii_digit())
        })
        .collect::<Vec<_>>();
    paths.sort();
    if paths.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            "no dictionaryNN.txt files found",
        ));
    }
    Ok(paths)
}

fn parse_mozc_file(
    path: &Path,
    entries: &mut BTreeMap<(String, String, u32, u32), Word>,
    row_count: &mut usize,
) -> io::Result<()> {
    let data = fs::read_to_string(path)?;
    for (line_number, raw_line) in data.lines().enumerate() {
        let line = raw_line.trim_start_matches('\u{feff}');
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields = line.split('\t').collect::<Vec<_>>();
        if fields.len() < 5 {
            return Err(invalid_mozc_row(
                path,
                line_number + 1,
                "expected five TSV fields",
            ));
        }
        let left_id = fields[1].parse::<u32>().map_err(|_| {
            invalid_mozc_row(path, line_number + 1, "left context ID is not an integer")
        })?;
        let right_id = fields[2].parse::<u32>().map_err(|_| {
            invalid_mozc_row(path, line_number + 1, "right context ID is not an integer")
        })?;
        let cost = fields[3]
            .parse::<i32>()
            .map_err(|_| invalid_mozc_row(path, line_number + 1, "word cost is not an integer"))?;
        let reading = crate::normalize::to_hiragana(fields[0]);
        let surface = fields[4].to_owned();
        if reading.is_empty() || surface.is_empty() {
            return Err(invalid_mozc_row(
                path,
                line_number + 1,
                "reading or surface is empty",
            ));
        }
        *row_count = row_count.saturating_add(1);
        let word = Word {
            reading: reading.clone(),
            surface: surface.clone(),
            priority: 0,
            frequency: 0,
            left_id,
            right_id,
            cost,
        };
        entries
            .entry((reading, surface, left_id, right_id))
            .and_modify(|existing| {
                if word.cost < existing.cost {
                    *existing = word.clone();
                }
            })
            .or_insert(word);
    }
    Ok(())
}

fn invalid_mozc_row(path: &Path, line_number: usize, message: &str) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        format!("{}:{}: {message}", path.display(), line_number),
    )
}

fn encode(words: &[Word], connection: &ConnectionMatrix) -> io::Result<Vec<u8>> {
    let count = u32::try_from(words.len()).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "static dictionary has too many entries",
        )
    })?;
    let mut data = Vec::new();
    data.extend_from_slice(MAGIC_V2);
    data.extend_from_slice(&count.to_le_bytes());
    data.extend_from_slice(&connection.dimension().to_le_bytes());

    for word in words {
        let reading = word.reading.as_bytes();
        let surface = word.surface.as_bytes();
        let reading_len = u32::try_from(reading.len()).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "static dictionary reading is too long",
            )
        })?;
        let surface_len = u32::try_from(surface.len()).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "static dictionary surface is too long",
            )
        })?;
        data.extend_from_slice(&reading_len.to_le_bytes());
        data.extend_from_slice(&surface_len.to_le_bytes());
        data.extend_from_slice(&word.priority.to_le_bytes());
        data.extend_from_slice(&word.frequency.to_le_bytes());
        data.extend_from_slice(&word.left_id.to_le_bytes());
        data.extend_from_slice(&word.right_id.to_le_bytes());
        data.extend_from_slice(&word.cost.to_le_bytes());
        data.extend_from_slice(reading);
        data.extend_from_slice(surface);
    }

    for cost in connection.costs() {
        data.extend_from_slice(&cost.to_le_bytes());
    }

    Ok(data)
}

fn decode(data: &[u8]) -> io::Result<StaticDictionary> {
    let mut cursor = Cursor::new(data);
    let magic = cursor.take(MAGIC_V2.len())?;
    if magic == MAGIC_V1 {
        return decode_v1(&mut cursor);
    }
    if magic != MAGIC_V2 {
        return Err(invalid_data("invalid static dictionary magic or version"));
    }
    let count = cursor.u32()?;
    let dimension = cursor.u32()?;
    let count = usize::try_from(count).map_err(|_| invalid_data("invalid entry count"))?;
    if count > cursor.remaining() / 28 {
        return Err(invalid_data(
            "entry count exceeds remaining dictionary data",
        ));
    }
    let mut words = Vec::with_capacity(count);

    for _ in 0..count {
        let reading_len =
            usize::try_from(cursor.u32()?).map_err(|_| invalid_data("invalid reading length"))?;
        let surface_len =
            usize::try_from(cursor.u32()?).map_err(|_| invalid_data("invalid surface length"))?;
        let priority = cursor.i32()?;
        let frequency = cursor.u32()?;
        let left_id = cursor.u32()?;
        let right_id = cursor.u32()?;
        let cost = cursor.i32()?;
        let reading = String::from_utf8(cursor.take(reading_len)?.to_vec())
            .map_err(|_| invalid_data("reading is not valid UTF-8"))?;
        let surface = String::from_utf8(cursor.take(surface_len)?.to_vec())
            .map_err(|_| invalid_data("surface is not valid UTF-8"))?;
        if reading.is_empty() || surface.is_empty() {
            return Err(invalid_data("static dictionary entries cannot be empty"));
        }
        words.push(Word {
            reading,
            surface,
            priority,
            frequency,
            left_id,
            right_id,
            cost,
        });
    }

    let dimension_usize = usize::try_from(dimension)
        .map_err(|_| invalid_data("connection matrix dimension is invalid"))?;
    let matrix_count = dimension_usize
        .checked_mul(dimension_usize)
        .ok_or_else(|| invalid_data("connection matrix size overflow"))?;
    if matrix_count > cursor.remaining() / 4 {
        return Err(invalid_data("truncated connection matrix"));
    }
    let mut costs = Vec::with_capacity(matrix_count);
    for _ in 0..matrix_count {
        costs.push(cursor.i32()?);
    }
    let connection = ConnectionMatrix::new(dimension, costs)?;

    if cursor.remaining() != 0 {
        return Err(invalid_data("static dictionary has trailing bytes"));
    }
    Ok(StaticDictionary { words, connection })
}

fn decode_v1(cursor: &mut Cursor<'_>) -> io::Result<StaticDictionary> {
    let count = cursor.u32()?;
    let count = usize::try_from(count).map_err(|_| invalid_data("invalid entry count"))?;
    if count > cursor.remaining() / 16 {
        return Err(invalid_data(
            "entry count exceeds remaining dictionary data",
        ));
    }
    let mut words = Vec::with_capacity(count);
    for _ in 0..count {
        let reading_len =
            usize::try_from(cursor.u32()?).map_err(|_| invalid_data("invalid reading length"))?;
        let surface_len =
            usize::try_from(cursor.u32()?).map_err(|_| invalid_data("invalid surface length"))?;
        let priority = cursor.i32()?;
        let frequency = cursor.u32()?;
        let reading = String::from_utf8(cursor.take(reading_len)?.to_vec())
            .map_err(|_| invalid_data("reading is not valid UTF-8"))?;
        let surface = String::from_utf8(cursor.take(surface_len)?.to_vec())
            .map_err(|_| invalid_data("surface is not valid UTF-8"))?;
        if reading.is_empty() || surface.is_empty() {
            return Err(invalid_data("static dictionary entries cannot be empty"));
        }
        words.push(Word {
            reading,
            surface,
            priority,
            frequency,
            left_id: 0,
            right_id: 0,
            cost: 0,
        });
    }
    if cursor.remaining() != 0 {
        return Err(invalid_data("static dictionary has trailing bytes"));
    }
    Ok(StaticDictionary {
        words,
        connection: ConnectionMatrix::default(),
    })
}

fn invalid_data(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

struct Cursor<'a> {
    data: &'a [u8],
    offset: usize,
}

impl<'a> Cursor<'a> {
    const fn new(data: &'a [u8]) -> Self {
        Self { data, offset: 0 }
    }

    fn take(&mut self, length: usize) -> io::Result<&'a [u8]> {
        let end = self
            .offset
            .checked_add(length)
            .ok_or_else(|| invalid_data("static dictionary offset overflow"))?;
        let result = self
            .data
            .get(self.offset..end)
            .ok_or_else(|| invalid_data("truncated static dictionary"))?;
        self.offset = end;
        Ok(result)
    }

    fn u32(&mut self) -> io::Result<u32> {
        let bytes = self.take(4)?;
        Ok(u32::from_le_bytes(bytes.try_into().map_err(|_| {
            invalid_data("invalid u32 in static dictionary")
        })?))
    }

    fn i32(&mut self) -> io::Result<i32> {
        let bytes = self.take(4)?;
        Ok(i32::from_le_bytes(bytes.try_into().map_err(|_| {
            invalid_data("invalid i32 in static dictionary")
        })?))
    }

    const fn remaining(&self) -> usize {
        self.data.len() - self.offset
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::StaticDictionary;

    #[test]
    fn compiles_and_loads_tsv_round_trip() {
        let prefix = format!("ai-engine-static-dictionary-{}", std::process::id());
        let source = std::env::temp_dir().join(format!("{prefix}.tsv"));
        let binary = std::env::temp_dir().join(format!("{prefix}.aidict"));
        fs::write(&source, "とうきょう\t東京\t30\t2\n").expect("write source");

        StaticDictionary::compile_tsv(&source, &binary).expect("compile dictionary");
        let dictionary = StaticDictionary::load(&binary).expect("load dictionary");
        assert_eq!(dictionary.words().len(), 1);
        assert_eq!(dictionary.words()[0].surface, "東京");
        assert_eq!(dictionary.words()[0].frequency, 2);

        fs::remove_file(source).expect("remove source");
        fs::remove_file(binary).expect("remove binary");
    }

    #[test]
    fn rejects_trailing_bytes() {
        let mut data = Vec::from(*b"AIDICT\0\x01");
        data.extend_from_slice(&0_u32.to_le_bytes());
        data.push(1);
        let path = std::env::temp_dir().join(format!(
            "ai-engine-static-dictionary-invalid-{}.aidict",
            std::process::id()
        ));
        fs::write(&path, data).expect("write invalid dictionary");

        let error = StaticDictionary::load(&path).expect_err("trailing bytes must fail");
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        fs::remove_file(path).expect("remove invalid dictionary");
    }

    #[test]
    fn imports_mozc_dictionary_rows_and_keeps_lowest_cost() {
        let directory =
            std::env::temp_dir().join(format!("ai-engine-mozc-import-{}", std::process::id()));
        let binary = std::env::temp_dir().join(format!(
            "ai-engine-mozc-import-{}.aidict",
            std::process::id()
        ));
        fs::create_dir_all(&directory).expect("create source directory");
        fs::write(
            directory.join("dictionary00.txt"),
            "# comment\nとうきょう\t1\t0\t9000\t東京\nとうきょう\t1\t0\t8000\t東京\nとうきょう\t2\t0\t7000\t東京\nぷろとたいぷ\t0\t1\t7000\tプロトタイプ\n",
        )
        .expect("write Mozc source");
        fs::write(
            directory.join("connection_single_column.txt"),
            "2\n0\n10\n20\n30\n",
        )
        .expect("write connection matrix");

        let stats = StaticDictionary::compile_mozc_directory(&directory, &binary)
            .expect("compile Mozc dictionary");
        assert_eq!(stats.files, 1);
        assert_eq!(stats.rows, 4);
        assert_eq!(stats.unique_entries, 3);
        assert_eq!(stats.matrix_dimension, 2);
        let dictionary = StaticDictionary::load(&binary).expect("load compiled dictionary");
        assert_eq!(dictionary.words().len(), 3);
        assert_eq!(dictionary.connection().cost(1, 0), Some(20));
        let tokyo = dictionary
            .words()
            .iter()
            .find(|word| word.surface == "東京")
            .expect("Tokyo entry");
        assert_eq!(tokyo.priority, 0);
        assert_eq!(tokyo.left_id, 1);
        assert_eq!(tokyo.right_id, 0);
        assert_eq!(tokyo.cost, 8000);

        fs::remove_file(binary).expect("remove binary");
        fs::remove_dir_all(directory).expect("remove source directory");
    }

    #[test]
    fn connection_cost_is_used_by_core_path_scoring() {
        let directory =
            std::env::temp_dir().join(format!("ai-engine-mozc-connection-{}", std::process::id()));
        let binary = std::env::temp_dir().join(format!(
            "ai-engine-mozc-connection-{}.aidict",
            std::process::id()
        ));
        fs::create_dir_all(&directory).expect("create source directory");
        fs::write(
            directory.join("dictionary00.txt"),
            "あ\t1\t1\t10\t阿\nい\t0\t0\t20\t伊\n",
        )
        .expect("write Mozc source");
        fs::write(
            directory.join("connection_single_column.txt"),
            "2\n0\n0\n30\n0\n",
        )
        .expect("write connection matrix");
        StaticDictionary::compile_mozc_directory(&directory, &binary)
            .expect("compile Mozc dictionary");

        let mut engine = crate::KanaKanjiEngine::new();
        engine
            .load_static_dictionary(&binary)
            .expect("load static dictionary");
        let candidates = engine.convert("あい", 1);
        assert_eq!(candidates[0].text, "阿伊");
        assert_eq!(candidates[0].score, -58);

        fs::remove_file(binary).expect("remove binary");
        fs::remove_dir_all(directory).expect("remove source directory");
    }
}
