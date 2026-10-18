//! Initial aiEngine-specific static dictionary container and dumper.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::error::Error;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use encoding_rs::{EUC_JP, Encoding};

type BoxError = Box<dyn Error>;

const MAGIC: &[u8; 8] = b"AISDICT\0";
const VERSION: u32 = 7;
const FIXED_HEADER_BYTES: usize = 72;
const DIRECTORY_ENTRY_BYTES: usize = 48;
const HEADER_GUID: [u8; 16] = *b"AIENG-SDICT-V3!!";

const POS_NAMES: [u8; 16] = *b"POS-NAMES-V2!!!!";
const VOCABULARY: [u8; 16] = *b"VOCABULARY-V2!!!";
const STRING_INDEX: [u8; 16] = *b"STRING-INDEX-V2!";
const QUANTIZATION: [u8; 16] = *b"QUANTIZATION-V2!";
const POS_BIGRAM: [u8; 16] = *b"POS-BIGRAM-V3!!!";
const WORD_BIGRAM: [u8; 16] = *b"WORD-BIGRAM-V3!!";
const TRIGRAM: [u8; 16] = *b"TRIGRAM-V3!!!!!!";
const MODEL_ONLY_SECTION: &[u8; 4] = b"MOS4";
const MODEL_ONLY_FOOTER: &[u8; 4] = b"MOF4";
pub(super) const DEFAULT_NGRAM_CUTOFF: u64 = 10;

const QUANT_WORD_UNIGRAM_IN_POS: u8 = 1;
const QUANT_POS_BIGRAM: u8 = 2;
const QUANT_WORD_BIGRAM: u8 = 3;
const QUANT_BIGRAM_BACKOFF: u8 = 4;
const QUANT_TRIGRAM: u8 = 5;
const PREFIX_CACHE_HEADER_TAG: u8 = 0x40;
const PREDICTION_PHRASE_TAG_BASE: u8 = 0x60;
const WORD_RECORD_TAG: u8 = 0x80;
const DISPLAY_REFERENCE_TAG_BASE: u8 = 0xa0;
const REQUIRED_MECAB_HEADER: [&str; 13] = [
    "surface",
    "left_id",
    "right_id",
    "cost",
    "pos1",
    "pos2",
    "pos3",
    "pos4",
    "pos5",
    "pos6",
    "base",
    "reading",
    "pronunciation",
];
const DEFAULT_POS_WORD_BIGRAM_LIMIT: usize = 500_000;
const DEFAULT_WORD_POS_BIGRAM_LIMIT: usize = 500_000;
const DEFAULT_AFFIX_POS_LIMIT: usize = 100;
const EXCLUDED_AFFIX_SURFACES: [&str; 5] = ["号", "時", "日", "数", "級"];

type CompiledWordTable = (
    Vec<Word>,
    Vec<String>,
    ProbabilityCodebook,
    Vec<Option<usize>>,
);

#[derive(Clone, Debug)]
#[allow(clippy::struct_field_names)]
struct Word {
    source_id: usize,
    word_id: u32,
    surface: String,
    reading: String,
    display_leaf: u32,
    reading_leaf: u32,
    pos: String,
    pos_id: u32,
    bigram_block_index: u32,
    backoff_alpha_code: u8,
    unigram_count: u64,
    word_unigram_in_pos_raw: f64,
    word_unigram_in_pos: f64,
    word_unigram_in_pos_code: u8,
    flags: u8,
}

#[derive(Clone, Debug)]
struct Phrase {
    ids: Vec<u32>,
    reading: String,
    unigram: f64,
    vocab_offset: u32,
}

#[derive(Clone, Debug, Default)]
struct KeyGroup {
    key: Vec<u8>,
    reading_words: Vec<usize>,
    display_words: Vec<usize>,
    phrase_ids: Vec<u32>,
    cache_targets: Vec<CacheTarget>,
    cache_target_offsets: Vec<u32>,
    prefix: Vec<u8>,
    tail: Vec<u8>,
    leaf_node: u32,
    vocab_offset: u32,
}

#[derive(Clone, Copy, Debug)]
struct BaseCheck {
    base: i32,
    check: u32,
}

impl Default for BaseCheck {
    fn default() -> Self {
        Self {
            base: 0,
            check: u32::MAX,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct FreeLink {
    previous: u32,
    next: u32,
}

#[derive(Clone, Debug)]
struct DoubleArray {
    cells: Vec<BaseCheck>,
    group_leaves: Vec<u32>,
}

#[derive(Clone, Debug)]
struct Block {
    guid: [u8; 16],
    offset: u64,
    bytes: Vec<u8>,
    item_count: u64,
}

#[derive(Clone, Debug)]
struct BlockInfo {
    guid: [u8; 16],
    offset: usize,
    size: usize,
    item_count: u64,
    checksum: u64,
}

#[derive(Clone, Debug)]
struct PosBigramEntry {
    source_pos: u32,
    target_pos: u32,
    probability: f64,
    code: u8,
}

#[derive(Clone, Debug)]
struct WordBigramEntry {
    source_word: u32,
    target_word: u32,
    count: u64,
    probability: f64,
    code: u8,
}

#[derive(Clone, Debug)]
struct WordBigramContext {
    source_word: u32,
    first_edge: u32,
    edge_count: u32,
    alpha: f64,
    alpha_code: u8,
}

#[derive(Clone, Debug)]
struct TrigramRecord {
    parent_edge: u32,
    slot: u32,
    probability: f64,
    code: u8,
}

#[derive(Clone, Debug)]
struct TrigramBlock {
    parent_edge: u32,
    continuation_count: u32,
    record_start: usize,
    record_count: usize,
}

#[derive(Clone, Debug)]
struct DecodedTrigram {
    source_word: u32,
    middle_word: u32,
    target_word: u32,
    parent_edge: u32,
    slot: u32,
    probability: f64,
    code: u8,
    codec: u8,
}

#[derive(Clone, Debug)]
struct NgramTables {
    pos_entries: Vec<PosBigramEntry>,
    word_contexts: Vec<WordBigramContext>,
    word_entries: Vec<WordBigramEntry>,
    trigram_records: Vec<TrigramRecord>,
    trigram_blocks: Vec<TrigramBlock>,
    codebooks: Vec<ProbabilityCodebook>,
}

type ReadingWordIndex = BTreeMap<String, Vec<u32>>;
type PhraseIndex = BTreeMap<String, Vec<u32>>;
type PrefixIndex = BTreeMap<String, Vec<CacheTarget>>;
type ResolvedPrefixIndex = BTreeMap<String, Vec<PredictionTarget>>;
type ParsedContainer = ([u8; 16], [u8; 16], Vec<BlockInfo>);

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct PhraseKey {
    ids: Vec<u32>,
    unigram_bits: u64,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum CandidateTarget {
    Word(usize),
    Phrase(PhraseKey),
}

#[derive(Clone, Debug)]
struct RankedCandidateTarget {
    target: CandidateTarget,
    unigram: f64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum CacheTarget {
    Word(usize),
    Phrase(u32),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PredictionTarget {
    Word(u32),
    Phrase(u32),
}

#[derive(Default)]
struct DecodedVocabulary {
    words: Vec<Word>,
    reading_words: ReadingWordIndex,
    phrases: Vec<Phrase>,
    prefixes: ResolvedPrefixIndex,
}

#[allow(clippy::too_many_lines)]
pub(super) fn compile(
    input_dir: &Path,
    destination: &Path,
    system_dictionary: &Path,
    user_additions: &Path,
    ngram_cutoff: u64,
) -> Result<(), BoxError> {
    if ngram_cutoff == 0 {
        return Err("ngram cutoff must be at least 1".into());
    }
    let vocab_path = input_dir.join("vocab.tsv");
    let phrase_path = input_dir.join("prediction-phrase.txt");
    let required_inputs = [
        "vocab.tsv",
        "1gram.by-id.tsv",
        "2gram.by-id.tsv",
        "2gram.tsv",
        "3gram.tsv",
        "prediction-phrase.txt",
    ]
    .map(|name| input_dir.join(name));
    if required_inputs.iter().any(|path| !path.is_file()) {
        return Err(format!(
            "compile-sdict expects vocab.tsv, 1gram.by-id.tsv, 2gram.by-id.tsv, 2gram.tsv, 3gram.tsv, and prediction-phrase.txt under {}",
            input_dir.display()
        )
        .into());
    }

    if !system_dictionary.is_dir() || !user_additions.is_dir() {
        return Err(
            "compile-sdict requires an existing MeCab dicdir and user-addition source directory"
                .into(),
        );
    }
    let (mut words, pos_names, unigram_codebook, corpus_source_ids) =
        compile_word_table_from_sources(
            &vocab_path,
            &input_dir.join("1gram.by-id.tsv"),
            Some(system_dictionary),
            Some(user_additions),
        )?;

    let (mut phrases, phrase_reading, prefix_cache) =
        read_phrases(&phrase_path, &words, &corpus_source_ids, 100_000, 5)?;
    let mut key_groups = make_key_groups(&words, &phrase_reading, &prefix_cache)?;
    let mut string_index = DoubleArray::build(&mut key_groups)?;
    let (mut vocabulary, source_word_ids) =
        encode_vocabulary_groups(&mut key_groups, &mut words, &mut phrases)?;
    let corpus_word_ids = corpus_source_ids
        .iter()
        .map(|source_id| {
            source_id.and_then(|source_id| source_word_ids.get(source_id).copied().flatten())
        })
        .collect::<Vec<_>>();
    remap_phrase_words(&mut phrases, &corpus_word_ids)?;
    patch_vocabulary_phrase_words(&mut vocabulary, &phrases)?;
    string_index.bind_vocab_offsets(&key_groups)?;
    let ngrams = build_ngram_tables(input_dir, &corpus_word_ids, &words, ngram_cutoff)?;
    patch_vocabulary_bigram_refs(&mut vocabulary, &mut words, &ngrams.word_contexts)?;
    let mut all_codebooks = vec![unigram_codebook];
    all_codebooks.extend(ngrams.codebooks.iter().cloned());

    let payloads = vec![
        (
            POS_NAMES,
            encode_pos_names(&pos_names)?,
            pos_names.len() as u64,
        ),
        (
            VOCABULARY,
            vocabulary,
            u64::try_from(
                words
                    .iter()
                    .filter(|word| word.word_id != u32::MAX)
                    .count()
                    .checked_add(phrases.len())
                    .ok_or("vocabulary item count overflow")?,
            )?,
        ),
        (
            STRING_INDEX,
            string_index.encode()?,
            string_index.cells.len() as u64,
        ),
        (
            POS_BIGRAM,
            encode_pos_bigrams(&ngrams.pos_entries, pos_names.len())?,
            u64::try_from(
                pos_names
                    .len()
                    .checked_mul(pos_names.len())
                    .ok_or("POS bigram matrix size overflow")?,
            )?,
        ),
        (
            WORD_BIGRAM,
            encode_word_bigrams(&ngrams.word_contexts, &ngrams.word_entries)?,
            ngrams.word_entries.len() as u64,
        ),
        (
            TRIGRAM,
            encode_trigrams(
                &ngrams.trigram_records,
                &ngrams.trigram_blocks,
                ngrams.word_entries.len(),
            )?,
            ngrams.trigram_records.len() as u64,
        ),
        (
            QUANTIZATION,
            encode_codebooks(&all_codebooks)?,
            all_codebooks.len() as u64,
        ),
    ];
    write_container(destination, payloads)?;
    Ok(())
}

#[allow(clippy::too_many_lines)]
pub(super) fn dump(source: &Path, output_dir: &Path) -> Result<(), BoxError> {
    let bytes = fs::read(source)?;
    let (dict_uuid, build_uuid, blocks) = parse_container(&bytes)?;
    if output_dir.exists() && fs::read_dir(output_dir)?.next().is_some() {
        return Err(format!(
            "dumper output directory is not empty; refusing to overwrite: {}",
            output_dir.display()
        )
        .into());
    }
    let block_map = blocks
        .iter()
        .map(|block| (block.guid, block_slice(&bytes, block)))
        .collect::<BTreeMap<_, _>>();
    let pos_names = decode_pos_names(required_block(&block_map, &POS_NAMES)?)?;
    let codebooks = decode_codebooks(required_block(&block_map, &QUANTIZATION)?)?;
    let unigram_codebook = codebooks
        .get(&QUANT_WORD_UNIGRAM_IN_POS)
        .ok_or("word unigram-in-POS codebook is missing")?;
    let vocabulary_bytes = required_block(&block_map, &VOCABULARY)?;
    let string_index = DoubleArray::decode(required_block(&block_map, &STRING_INDEX)?)?;
    let decoded_vocab = decode_vocabulary(
        vocabulary_bytes,
        &pos_names,
        unigram_codebook,
        &string_index,
    )?;
    let words_by_id = decoded_vocab
        .words
        .iter()
        .cloned()
        .map(|word| (word.word_id, word))
        .collect::<BTreeMap<_, _>>();
    let pos_bigram_codebook = codebooks
        .get(&QUANT_POS_BIGRAM)
        .ok_or("POS bigram codebook is missing")?;
    let word_bigram_codebook = codebooks
        .get(&QUANT_WORD_BIGRAM)
        .ok_or("word bigram codebook is missing")?;
    let backoff_codebook = codebooks
        .get(&QUANT_BIGRAM_BACKOFF)
        .ok_or("bigram backoff codebook is missing")?;
    let trigram_codebook = codebooks
        .get(&QUANT_TRIGRAM)
        .ok_or("trigram codebook is missing")?;
    let pos_bigrams = decode_pos_bigrams(
        required_block(&block_map, &POS_BIGRAM)?,
        &pos_names,
        pos_bigram_codebook,
    )?;
    let (word_contexts, word_bigrams) = decode_word_bigrams(
        required_block(&block_map, &WORD_BIGRAM)?,
        &words_by_id,
        word_bigram_codebook,
        backoff_codebook,
    )?;
    let trigrams = decode_trigrams(
        required_block(&block_map, &TRIGRAM)?,
        &word_contexts,
        &word_bigrams,
        &words_by_id,
        trigram_codebook,
    )?;
    for (guid, actual) in [
        (POS_NAMES, pos_names.len()),
        (
            VOCABULARY,
            decoded_vocab
                .words
                .len()
                .checked_add(decoded_vocab.phrases.len())
                .ok_or("vocabulary item count overflow")?,
        ),
        (STRING_INDEX, string_index.cells.len()),
        (
            POS_BIGRAM,
            pos_names
                .len()
                .checked_mul(pos_names.len())
                .ok_or("POS bigram matrix size overflow")?,
        ),
        (WORD_BIGRAM, word_bigrams.len()),
        (TRIGRAM, trigrams.len()),
        (QUANTIZATION, codebooks.len()),
    ] {
        verify_item_count(&blocks, &guid, actual)?;
    }

    fs::create_dir_all(output_dir)?;
    let mut header = BufWriter::new(File::create_new(output_dir.join("header.tsv"))?);
    writeln!(header, "key\tvalue")?;
    writeln!(header, "format_guid\t{}", format_guid(&HEADER_GUID))?;
    writeln!(header, "version\t{VERSION}")?;
    writeln!(header, "dictionary_uuid\t{}", format_guid(&dict_uuid))?;
    writeln!(header, "build_uuid\t{}", format_guid(&build_uuid))?;

    let mut block_file = BufWriter::new(File::create_new(output_dir.join("blocks.tsv"))?);
    writeln!(
        block_file,
        "block_guid\toffset\tsize\titem_count\tchecksum_fnv1a64"
    )?;
    for block in &blocks {
        writeln!(
            block_file,
            "{}\t{}\t{}\t{}\t{:016x}",
            format_guid(&block.guid),
            block.offset,
            block.size,
            block.item_count,
            block.checksum
        )?;
    }
    header.flush()?;
    block_file.flush()?;

    write_pos_dump(output_dir, &pos_names)?;
    write_vocab_dump(output_dir, &decoded_vocab.words, &pos_names)?;
    write_pos_bigram_dump(output_dir, &pos_bigrams, &pos_names)?;
    write_word_bigram_dump(output_dir, &word_contexts, &word_bigrams, &words_by_id)?;
    write_trigram_dump(output_dir, &trigrams, &words_by_id)?;
    write_reading_dump(output_dir, &decoded_vocab.reading_words)?;
    write_phrase_dump(output_dir, &decoded_vocab.phrases, &words_by_id)?;
    write_prefix_dump(
        output_dir,
        &decoded_vocab.prefixes,
        &decoded_vocab.phrases,
        &words_by_id,
    )?;
    Ok(())
}

#[allow(clippy::too_many_lines)]
fn compile_word_table_from_sources(
    vocabulary_path: &Path,
    unigram_path: &Path,
    system_dictionary: Option<&Path>,
    user_additions: Option<&Path>,
) -> Result<CompiledWordTable, BoxError> {
    let corpus_tokens = read_vocabulary(vocabulary_path)?;
    let corpus_counts = read_unigram_counts(unigram_path, corpus_tokens.len())?;
    let mut seeds = BTreeMap::<WordKey, WordSeed>::new();
    if let (Some(system_dictionary), Some(user_additions)) = (system_dictionary, user_additions) {
        read_dictionary_sources(system_dictionary, user_additions, &mut seeds)?;
    } else {
        // Small unit fixtures may use the corpus vocabulary as their explicit
        // dictionary source. Production CLI calls always supply both sources.
        for (id, token) in corpus_tokens.iter().enumerate().skip(3) {
            let (surface, reading, pos, flags) = decode_token(token, u32::try_from(id)?);
            let key = if flags & 2 != 0 {
                WordKey::model_only(&pos)
            } else {
                WordKey::new(surface, reading, pos)
            };
            seeds.entry(key.clone()).or_insert(WordSeed { key, flags });
        }
    }

    let mut corpus_keys = Vec::with_capacity(corpus_tokens.len());
    for (id, token) in corpus_tokens.iter().enumerate() {
        let key = match id {
            0 => None,
            1 => Some(WordKey::model_only("[BOS]")),
            2 => Some(WordKey::model_only("[EOS]")),
            _ => {
                let (surface, reading, pos, flags) = decode_token(token, u32::try_from(id)?);
                let key = if flags & 2 != 0 {
                    WordKey::model_only(&pos)
                } else {
                    WordKey::new(surface, reading, pos)
                };
                if !seeds.contains_key(&key) {
                    // Preserve a corpus-observed inflected/synthetic form that
                    // the source CSVs do not enumerate, while keeping the full
                    // MeCab and user-dictionary sources as the vocabulary base.
                    seeds.insert(
                        key.clone(),
                        WordSeed {
                            key: key.clone(),
                            flags,
                        },
                    );
                }
                Some(key)
            }
        };
        if let Some(key) = &key {
            let flags = if key.surface.is_empty() { 2 } else { 0 };
            seeds.entry(key.clone()).or_insert_with(|| WordSeed {
                key: key.clone(),
                flags,
            });
        }
        corpus_keys.push(key);
    }

    let source_index = seeds
        .keys()
        .cloned()
        .enumerate()
        .map(|(index, key)| (key, index))
        .collect::<BTreeMap<_, _>>();
    let mut observed_counts = vec![0_u64; seeds.len()];
    let mut corpus_source_ids = Vec::with_capacity(corpus_keys.len());
    for (id, key) in corpus_keys.into_iter().enumerate() {
        let Some(key) = key else {
            corpus_source_ids.push(None);
            continue;
        };
        let index = *source_index
            .get(&key)
            .ok_or("corpus token was not added to source dictionary")?;
        observed_counts[index] = observed_counts[index]
            .checked_add(corpus_counts[id])
            .ok_or("corpus unigram aggregation overflow")?;
        corpus_source_ids.push(Some(index));
    }

    let mut words = Vec::with_capacity(seeds.len());
    for (source_id, seed) in seeds.into_values().enumerate() {
        let count = observed_counts[source_id];
        let weighted_count = if count == 0 {
            1
        } else {
            count
                .checked_mul(10)
                .ok_or("10x corpus unigram count overflow")?
        };
        words.push(Word {
            source_id,
            word_id: u32::MAX,
            surface: seed.key.surface,
            reading: seed.key.reading,
            display_leaf: u32::MAX,
            reading_leaf: u32::MAX,
            pos: seed.key.pos,
            pos_id: 0,
            bigram_block_index: 0,
            backoff_alpha_code: 0,
            unigram_count: weighted_count,
            word_unigram_in_pos_raw: 0.0,
            word_unigram_in_pos: 0.0,
            word_unigram_in_pos_code: 0,
            flags: seed.flags,
        });
    }

    let pos_names = words
        .iter()
        .filter(|word| word.flags & 1 == 0)
        .map(|word| word.pos.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    for word in &mut words {
        if word.flags & 1 == 0 {
            word.pos_id = resolve_pos_id(&pos_names, &word.pos)
                .ok_or("compiler lost a POS name while assigning dictionary-local IDs")?;
        }
    }

    let mut pos_unigram_counts = vec![0_u64; pos_names.len() + 1];
    for word in &words {
        if word.flags & 1 == 0 {
            let total = pos_unigram_counts
                .get_mut(word.pos_id as usize)
                .ok_or("word POS ID is out of range while aggregating counts")?;
            *total = total
                .checked_add(word.unigram_count)
                .ok_or("POS unigram count overflow")?;
        }
    }
    let mut probabilities = Vec::with_capacity(words.len());
    for word in &mut words {
        if word.flags & 1 != 0 {
            continue;
        }
        let pos_total = pos_unigram_counts[word.pos_id as usize];
        word.word_unigram_in_pos_raw = count_as_f64(word.unigram_count) / count_as_f64(pos_total);
        word.word_unigram_in_pos = word.word_unigram_in_pos_raw;
        probabilities.push(word.word_unigram_in_pos_raw);
    }
    let (codebook, _) = ProbabilityCodebook::fit(QUANT_WORD_UNIGRAM_IN_POS, &probabilities)?;
    for word in &mut words {
        if word.word_unigram_in_pos > 0.0 {
            word.word_unigram_in_pos_code = codebook.encode(word.word_unigram_in_pos)?;
            word.word_unigram_in_pos = codebook.decode(word.word_unigram_in_pos_code)?;
        }
    }
    Ok((words, pos_names, codebook, corpus_source_ids))
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct WordKey {
    surface: String,
    reading: String,
    pos: String,
    token_key: String,
}

impl WordKey {
    fn new(surface: String, reading: String, pos: String) -> Self {
        let token_key = format!("{surface}\u{1f}{reading}\u{1f}{pos}");
        Self {
            surface,
            reading,
            pos,
            token_key,
        }
    }

    fn model_only(pos: &str) -> Self {
        Self::new(String::new(), String::new(), pos.to_owned())
    }
}

#[derive(Clone, Debug)]
struct WordSeed {
    key: WordKey,
    flags: u8,
}

fn read_dictionary_sources(
    system_dictionary: &Path,
    user_additions: &Path,
    output: &mut BTreeMap<WordKey, WordSeed>,
) -> Result<(), BoxError> {
    let encoding = dictionary_encoding(system_dictionary)?;
    let mut system_csvs = fs::read_dir(system_dictionary)?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "csv"))
        .collect::<Vec<_>>();
    system_csvs.sort();
    if system_csvs.is_empty() {
        return Err("MeCab system dictionary has no source CSV files".into());
    }
    let mut row_number = 0_u64;
    for path in system_csvs {
        let bytes = fs::read(&path)?;
        let (text, had_errors) = encoding.decode_without_bom_handling(&bytes);
        if had_errors {
            return Err(format!(
                "{} contains invalid {} bytes",
                path.display(),
                encoding.name()
            )
            .into());
        }
        for line in text.lines().filter(|line| !line.is_empty()) {
            row_number += 1;
            let fields = parse_csv_row(line)
                .map_err(|error| format!("{} row {row_number}: {error}", path.display()))?;
            add_mecab_source_row(output, &fields)
                .map_err(|error| format!("{} row {row_number}: {error}", path.display()))?;
        }
    }
    if row_number == 0 {
        return Err("MeCab system dictionary source CSVs were empty".into());
    }

    let mut addition_files = fs::read_dir(user_additions)?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| {
                    name.ends_with("-addition.csv") || name.ends_with("-new-words.txt")
                })
        })
        .collect::<Vec<_>>();
    addition_files.sort();
    if addition_files.is_empty() {
        return Err(
            "no user dictionary source files (*-addition.csv or *-new-words.txt) were found".into(),
        );
    }
    for path in addition_files {
        let bytes = fs::read(&path)?;
        let text = std::str::from_utf8(&bytes)?;
        let is_headered_csv = path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.ends_with("-addition.csv"));
        let mut lines = text.lines();
        let header = if is_headered_csv {
            Some(
                parse_csv_row(lines.next().ok_or("empty user addition CSV")?)
                    .map_err(|error| format!("{} header: {error}", path.display()))?,
            )
        } else {
            None
        };
        if let Some(header) = &header {
            validate_addition_header(header)?;
        }
        for (line_index, line) in lines.enumerate() {
            if line.is_empty() {
                continue;
            }
            let fields = parse_csv_row(line)
                .map_err(|error| format!("{} row {}: {error}", path.display(), line_index + 2))?;
            let fields = if let Some(header) = &header {
                reorder_addition_fields(header, &fields)?
            } else {
                fields
            };
            add_mecab_source_row(output, &fields)
                .map_err(|error| format!("{} row {}: {error}", path.display(), line_index + 2))?;
        }
    }
    Ok(())
}

fn dictionary_encoding(system_dictionary: &Path) -> Result<&'static Encoding, BoxError> {
    let config = fs::read(system_dictionary.join("dicrc"))?;
    let text = String::from_utf8_lossy(&config);
    let label = text
        .lines()
        .filter_map(|line| line.split_once('='))
        .find_map(|(key, value)| {
            key.trim()
                .eq_ignore_ascii_case("config-charset")
                .then(|| value.trim())
        })
        .ok_or("MeCab dicrc does not declare config-charset")?;
    if label.eq_ignore_ascii_case("EUC-JP") || label.eq_ignore_ascii_case("EUCJP") {
        return Ok(EUC_JP);
    }
    Encoding::for_label(label.as_bytes())
        .ok_or_else(|| format!("unsupported MeCab dictionary charset: {label}").into())
}

fn parse_csv_row(line: &str) -> Result<Vec<String>, BoxError> {
    let line = line.strip_prefix('\u{feff}').unwrap_or(line);
    let bytes = line.as_bytes();
    let mut fields = Vec::new();
    let mut field = Vec::new();
    let mut offset = 0;
    let mut quoted = false;
    let mut was_quoted = false;
    while offset < bytes.len() {
        let byte = bytes[offset];
        if quoted {
            if byte == b'"' {
                if bytes.get(offset + 1) == Some(&b'"') {
                    field.push(b'"');
                    offset += 2;
                } else {
                    quoted = false;
                    offset += 1;
                }
            } else {
                field.push(byte);
                offset += 1;
            }
        } else if byte == b',' {
            fields.push(String::from_utf8(std::mem::take(&mut field))?);
            was_quoted = false;
            offset += 1;
        } else if byte == b'"' {
            if !field.is_empty() || was_quoted {
                return Err("quote occurs inside an unquoted CSV field".into());
            }
            quoted = true;
            was_quoted = true;
            offset += 1;
        } else {
            if was_quoted {
                return Err("characters occur after a quoted CSV field".into());
            }
            field.push(byte);
            offset += 1;
        }
    }
    if quoted {
        return Err("unterminated quoted CSV field".into());
    }
    fields.push(String::from_utf8(field)?);
    Ok(fields)
}

fn validate_addition_header(header: &[String]) -> Result<(), BoxError> {
    if header.len() != REQUIRED_MECAB_HEADER.len()
        || REQUIRED_MECAB_HEADER
            .iter()
            .any(|column| !header.iter().any(|actual| actual == column))
    {
        return Err("user addition CSV header must contain the 13 MeCab columns".into());
    }
    Ok(())
}

fn reorder_addition_fields(header: &[String], fields: &[String]) -> Result<Vec<String>, BoxError> {
    if fields.len() != header.len() {
        return Err("user addition row does not match header width".into());
    }
    REQUIRED_MECAB_HEADER
        .iter()
        .map(|column| {
            let position = header
                .iter()
                .position(|name| name == column)
                .ok_or("addition header column disappeared")?;
            fields
                .get(position)
                .cloned()
                .ok_or_else(|| "addition field outside row".into())
        })
        .collect()
}

fn add_mecab_source_row(
    output: &mut BTreeMap<WordKey, WordSeed>,
    fields: &[String],
) -> Result<(), BoxError> {
    if fields.len() != 13 {
        return Err(format!("expected 13 MeCab columns, got {}", fields.len()).into());
    }
    if fields[0].is_empty() {
        return Err("MeCab source has an empty surface".into());
    }
    let pos = format!("{},*", fields[4..10].join(","));
    let reading = to_hiragana(&fields[11]);
    let surface = &fields[0];
    if pos == "記号,単漢字,*,*,*,*,*" {
        if surface.chars().count() != 1 {
            return Err("single-kanji source must have exactly one Unicode scalar".into());
        }
        // Keep the explicit internal POS and reading, including compatibility
        // ideographs and supplementary-plane kanji that generic classification
        // would otherwise collapse into an English or numeric model-only word.
        let key = WordKey::new(surface.clone(), reading, pos);
        output
            .entry(key.clone())
            .or_insert(WordSeed { key, flags: 0 });
        return Ok(());
    }
    let mut segment = String::new();
    for ch in surface.chars() {
        if is_source_symbol(ch) {
            if !segment.is_empty() {
                add_normalized_source_segment(output, &segment, &pos, &reading);
                segment.clear();
            }
            add_symbol_seed(output, ch);
        } else {
            segment.push(ch);
        }
    }
    if !segment.is_empty() {
        add_normalized_source_segment(output, &segment, &pos, &reading);
    }
    Ok(())
}

fn add_normalized_source_segment(
    output: &mut BTreeMap<WordKey, WordSeed>,
    surface: &str,
    pos: &str,
    reading: &str,
) {
    let normalized_surface = normalize_source_surface(surface);
    let key = if is_source_number(surface) {
        WordKey::model_only("名詞,数,*,*,*,*,*")
    } else if is_source_english(surface) {
        WordKey::model_only("英単語")
    } else {
        WordKey::new(
            normalized_surface.clone(),
            reading.to_owned(),
            model_pos_for_word(&normalized_surface, pos),
        )
    };
    let flags = u8::from(key.surface.is_empty()) * 2;
    output.entry(key.clone()).or_insert(WordSeed { key, flags });
}

fn add_symbol_seed(output: &mut BTreeMap<WordKey, WordSeed>, symbol: char) {
    let surface = normalize_source_surface(&symbol.to_string());
    let pos = "記号,一般,*,*,*,*,*";
    let key = WordKey::new(surface.clone(), surface.clone(), pos.to_owned());
    output
        .entry(key.clone())
        .or_insert(WordSeed { key, flags: 0 });
}

fn is_source_symbol(ch: char) -> bool {
    !ch.is_whitespace() && !ch.is_alphanumeric() && ch != 'ー'
}

fn is_source_number(surface: &str) -> bool {
    !surface.is_empty() && surface.chars().all(char::is_numeric)
}

fn is_source_english(surface: &str) -> bool {
    let mut has_letter = false;
    for ch in surface.chars() {
        if ch.is_alphabetic() && !(('あ'..='ヿ').contains(&ch) || ('㐀'..='鿿').contains(&ch)) {
            has_letter = true;
        } else if !ch.is_numeric() {
            return false;
        }
    }
    has_letter
}

pub(super) fn normalize_source_surface(surface: &str) -> String {
    surface
        .chars()
        .map(|ch| match ch {
            ' ' => '\u{3000}',
            '0'..='9' | '!'..='~' if !ch.is_ascii_alphabetic() => {
                char::from_u32(ch as u32 + 0xfee0).unwrap_or(ch)
            }
            _ => ch,
        })
        .collect()
}

fn read_vocabulary(path: &Path) -> Result<Vec<String>, BoxError> {
    let mut result = Vec::new();
    let mut reader = BufReader::new(File::open(path)?);
    let mut line = String::new();
    while reader.read_line(&mut line)? != 0 {
        let trimmed = line.trim_end_matches(['\r', '\n']);
        let (id_text, token) = trimmed.split_once('\t').ok_or("malformed vocab.tsv row")?;
        let id = id_text.parse::<usize>()?;
        if id != result.len() {
            return Err("vocab.tsv IDs must be contiguous and start at zero".into());
        }
        result.push(token.to_owned());
        line.clear();
    }
    if result.len() < 3 || result[0] != "[NULL]" || result[1] != "[BOS]" || result[2] != "[EOS]" {
        return Err("vocab.tsv reserved IDs must be NULL=0, BOS=1, EOS=2".into());
    }
    Ok(result)
}

fn read_unigram_counts(path: &Path, word_count: usize) -> Result<Vec<u64>, BoxError> {
    let mut counts = vec![0_u64; word_count];
    let mut seen = vec![false; word_count];
    if !path.is_file() {
        return Ok(counts);
    }
    let mut reader = BufReader::new(File::open(path)?);
    let mut line = String::new();
    let mut first = true;
    while reader.read_line(&mut line)? != 0 {
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if first && trimmed == "id\tcount" {
            first = false;
            line.clear();
            continue;
        }
        first = false;
        let (id_text, count_text) = trimmed.split_once('\t').ok_or("malformed 1gram row")?;
        let id = id_text.parse::<usize>()?;
        let count = count_text.parse::<u64>()?;
        let slot = counts.get_mut(id).ok_or("1gram ID is outside vocabulary")?;
        if seen[id] {
            return Err("duplicate ID in 1gram.by-id.tsv".into());
        }
        *slot = count;
        seen[id] = true;
        line.clear();
    }
    Ok(counts)
}

#[allow(clippy::too_many_lines)]
fn build_ngram_tables(
    input_dir: &Path,
    corpus_word_ids: &[Option<u32>],
    words: &[Word],
    ngram_cutoff: u64,
) -> Result<NgramTables, BoxError> {
    let words_by_id = words
        .iter()
        .filter(|word| word.word_id != u32::MAX)
        .map(|word| (word.word_id, word))
        .collect::<BTreeMap<_, _>>();
    let mut pos_pair_counts = BTreeMap::<(u32, u32), u64>::new();
    let mut pos_context_counts = BTreeMap::<u32, u64>::new();
    let mut word_outgoing_counts = BTreeMap::<u32, u64>::new();
    let no_selected_parent_pairs = HashSet::new();
    let mut no_selected_parent_counts = HashMap::new();
    let mut no_reading_length_counts = [0_u64; 81];
    let mut no_reading_length_contexts = [0_u64; 9];
    read_raw_bigram_counts(
        &input_dir.join("2gram.by-id.tsv"),
        corpus_word_ids,
        &words_by_id,
        &mut pos_pair_counts,
        &mut pos_context_counts,
        &mut word_outgoing_counts,
        &no_selected_parent_pairs,
        &mut no_selected_parent_counts,
        None,
        &mut no_reading_length_counts,
        &mut no_reading_length_contexts,
        None,
        None,
    )?;

    let mut pos_entries = pos_pair_counts
        .into_iter()
        .map(|((source_pos, target_pos), count)| {
            let denominator = *pos_context_counts
                .get(&source_pos)
                .ok_or("POS bigram source has no outgoing transitions")?;
            if denominator == 0 || count > denominator {
                return Err("invalid POS bigram count distribution".into());
            }
            Ok(PosBigramEntry {
                source_pos,
                target_pos,
                probability: count_as_f64(count) / count_as_f64(denominator),
                code: 0,
            })
        })
        .collect::<Result<Vec<_>, BoxError>>()?;
    let pos_probabilities = pos_entries
        .iter()
        .map(|entry| entry.probability)
        .collect::<Vec<_>>();
    let (pos_codebook, pos_codes) = ProbabilityCodebook::fit(QUANT_POS_BIGRAM, &pos_probabilities)?;
    for (entry, code) in pos_entries.iter_mut().zip(pos_codes) {
        entry.code = code;
    }
    let pos_probabilities = pos_entries
        .iter()
        .map(|entry| ((entry.source_pos, entry.target_pos), entry.probability))
        .collect::<BTreeMap<_, _>>();

    let mut word_entries = read_retained_bigrams(&input_dir.join("2gram.tsv"), corpus_word_ids)?;
    word_entries.sort_by_key(|(source, target, _)| (*source, *target));
    let mut merged_bigrams = Vec::<(u32, u32, u64)>::with_capacity(word_entries.len());
    for (source, target, count) in word_entries {
        if let Some(last) = merged_bigrams.last_mut() {
            if last.0 == source && last.1 == target {
                last.2 = last.2.checked_add(count).ok_or("bigram count overflow")?;
                continue;
            }
        }
        merged_bigrams.push((source, target, count));
    }
    merged_bigrams.retain(|(_, _, count)| *count >= ngram_cutoff);

    let mut word_entries = Vec::with_capacity(merged_bigrams.len());
    let mut contexts = Vec::<WordBigramContext>::new();
    let mut index = 0;
    while index < merged_bigrams.len() {
        let source = merged_bigrams[index].0;
        let first_edge = u32::try_from(word_entries.len())?;
        let start = index;
        while index < merged_bigrams.len() && merged_bigrams[index].0 == source {
            let (source_word, target_word, count) = merged_bigrams[index];
            let denominator = *word_outgoing_counts.get(&source_word).ok_or_else(|| {
                format!("word bigram source {source_word} has no corpus transitions")
            })?;
            if denominator == 0 || count > denominator {
                return Err(
                    format!("invalid word bigram count for {source_word}->{target_word}").into(),
                );
            }
            word_entries.push(WordBigramEntry {
                source_word,
                target_word,
                count,
                probability: count_as_f64(count) / count_as_f64(denominator),
                code: 0,
            });
            index += 1;
        }
        contexts.push(WordBigramContext {
            source_word: source,
            first_edge,
            edge_count: u32::try_from(index - start)?,
            alpha: 0.0,
            alpha_code: 0,
        });
    }

    let word_probabilities = word_entries
        .iter()
        .map(|entry| entry.probability)
        .collect::<Vec<_>>();
    let (word_codebook, word_codes) =
        ProbabilityCodebook::fit(QUANT_WORD_BIGRAM, &word_probabilities)?;
    for (entry, code) in word_entries.iter_mut().zip(word_codes) {
        entry.code = code;
    }

    for context in &mut contexts {
        let source = words_by_id
            .get(&context.source_word)
            .ok_or("word bigram context references missing vocabulary word")?;
        let mut explicit_word_mass = 0.0;
        let mut covered_pos_mass = 0.0;
        let start = usize::try_from(context.first_edge)?;
        let end = start
            .checked_add(usize::try_from(context.edge_count)?)
            .ok_or("word context edge range overflow")?;
        for entry in &word_entries[start..end] {
            explicit_word_mass += count_as_f64(entry.count)
                / count_as_f64(word_outgoing_counts[&context.source_word]);
            let target = words_by_id
                .get(&entry.target_word)
                .ok_or("word bigram target references missing vocabulary word")?;
            let pos_mass = *pos_probabilities
                .get(&(source.pos_id, target.pos_id))
                .unwrap_or(&0.0);
            covered_pos_mass += pos_mass * target.word_unigram_in_pos_raw;
        }
        let numerator = 1.0 - explicit_word_mass;
        let denominator = 1.0 - covered_pos_mass;
        context.alpha = if numerator <= 1.0e-12 {
            0.0
        } else if denominator <= 1.0e-12 {
            return Err(format!(
                "backoff alpha for word {} has positive residual mass but no uncovered POS support",
                context.source_word
            )
            .into());
        } else {
            numerator / denominator
        };
        if !context.alpha.is_finite() || context.alpha < 0.0 {
            return Err("computed invalid bigram backoff alpha".into());
        }
    }
    let alpha_values = contexts
        .iter()
        .map(|context| context.alpha)
        .collect::<Vec<_>>();
    let (backoff_codebook, alpha_codes) =
        ProbabilityCodebook::fit(QUANT_BIGRAM_BACKOFF, &alpha_values)?;
    for (context, code) in contexts.iter_mut().zip(alpha_codes) {
        context.alpha_code = code;
    }

    sort_word_bigram_blocks(&mut contexts, &mut word_entries)?;

    let mut pair_to_edge = HashMap::<(u32, u32), u32>::with_capacity(word_entries.len());
    let mut word_context_ranges = HashMap::<u32, (u32, u32)>::with_capacity(contexts.len());
    for context in &contexts {
        word_context_ranges.insert(
            context.source_word,
            (context.first_edge, context.edge_count),
        );
        let start = usize::try_from(context.first_edge)?;
        let end = start + usize::try_from(context.edge_count)?;
        for (edge_index, edge) in word_entries.iter().enumerate().take(end).skip(start) {
            pair_to_edge.insert(
                (edge.source_word, edge.target_word),
                u32::try_from(edge_index)?,
            );
        }
    }
    let raw_trigrams =
        read_retained_trigrams(&input_dir.join("3gram.tsv"), corpus_word_ids, ngram_cutoff)?;
    let mut trigram_records = Vec::with_capacity(raw_trigrams.len());
    for (source, middle, target, count) in raw_trigrams {
        let parent_edge = *pair_to_edge.get(&(source, middle)).ok_or_else(|| {
            format!("retained trigram {source}/{middle}/{target} has no retained A/B bigram")
        })?;
        let continuation_edge = *pair_to_edge.get(&(middle, target)).ok_or_else(|| {
            format!("retained trigram {source}/{middle}/{target} has no retained B/C bigram")
        })?;
        let (middle_first_edge, middle_edge_count) = *word_context_ranges
            .get(&middle)
            .ok_or("retained trigram middle word has no outgoing bigram context")?;
        let middle_end = middle_first_edge
            .checked_add(middle_edge_count)
            .ok_or("middle-word bigram range overflow")?;
        if continuation_edge < middle_first_edge || continuation_edge >= middle_end {
            return Err("trigram B/C edge does not belong to B/* context".into());
        }
        let slot = continuation_edge - middle_first_edge;
        let parent_count = word_entries
            .get(usize::try_from(parent_edge)?)
            .ok_or("trigram parent edge is out of range")?
            .count;
        if count > parent_count {
            return Err("trigram count exceeds its A/B parent bigram count".into());
        }
        trigram_records.push(TrigramRecord {
            parent_edge,
            slot,
            probability: count_as_f64(count) / count_as_f64(parent_count),
            code: 0,
        });
    }
    trigram_records.sort_by_key(|record| (record.parent_edge, record.slot));
    if trigram_records
        .windows(2)
        .any(|pair| pair[0].parent_edge == pair[1].parent_edge && pair[0].slot == pair[1].slot)
    {
        return Err("duplicate trigram continuation after corpus-ID remapping".into());
    }
    let trigram_probabilities = trigram_records
        .iter()
        .map(|record| record.probability)
        .collect::<Vec<_>>();
    let (trigram_codebook, trigram_codes) =
        ProbabilityCodebook::fit(QUANT_TRIGRAM, &trigram_probabilities)?;
    for (record, code) in trigram_records.iter_mut().zip(trigram_codes) {
        record.code = code;
    }
    let trigram_blocks = make_trigram_blocks(&trigram_records, &word_entries, &contexts)?;

    Ok(NgramTables {
        pos_entries,
        word_contexts: contexts,
        word_entries,
        trigram_records,
        trigram_blocks,
        codebooks: vec![
            pos_codebook,
            word_codebook,
            backoff_codebook,
            trigram_codebook,
        ],
    })
}

fn sort_word_bigram_blocks(
    contexts: &mut Vec<WordBigramContext>,
    entries: &mut Vec<WordBigramEntry>,
) -> Result<(), BoxError> {
    let mut blocks = Vec::with_capacity(contexts.len());
    for context in contexts.iter() {
        let start = usize::try_from(context.first_edge)?;
        let end = start
            .checked_add(usize::try_from(context.edge_count)?)
            .ok_or("word bigram block range overflow")?;
        let selected = entries
            .get(start..end)
            .ok_or("word bigram context range is outside the edge table")?;
        if selected.is_empty()
            || selected
                .windows(2)
                .any(|pair| pair[0].target_word >= pair[1].target_word)
        {
            return Err(
                "word bigram targets must be unique and ascending within each block".into(),
            );
        }
        blocks.push((
            selected
                .last()
                .ok_or("empty word bigram block")?
                .target_word,
            context.source_word,
            context.clone(),
            selected.to_vec(),
        ));
    }
    blocks.sort_by(|left, right| right.0.cmp(&left.0).then_with(|| left.1.cmp(&right.1)));

    let mut sorted_contexts = Vec::with_capacity(blocks.len());
    let mut sorted_entries = Vec::with_capacity(entries.len());
    for (_, _, mut context, selected) in blocks {
        context.first_edge = u32::try_from(sorted_entries.len())?;
        sorted_entries.extend(selected);
        sorted_contexts.push(context);
    }
    if sorted_entries.len() != entries.len() {
        return Err("word bigram block sort lost edge entries".into());
    }
    *contexts = sorted_contexts;
    *entries = sorted_entries;
    Ok(())
}

pub(super) fn training_seeds(
    corpus: &Path,
    system: &Path,
    additions: &Path,
) -> Result<crate::TrainingSeeds, BoxError> {
    training_seeds_with_bigrams(corpus, system, additions, 0).map(|(seeds, _)| seeds)
}

pub(super) fn training_seeds_with_bigrams(
    corpus: &Path,
    system: &Path,
    additions: &Path,
    bigram_limit: usize,
) -> Result<(crate::TrainingSeeds, Vec<crate::TrainingBigram>), BoxError> {
    let (seeds, bigrams, _) =
        training_seeds_with_ngrams(corpus, system, additions, bigram_limit, 0)?;
    Ok((seeds, bigrams))
}

#[allow(clippy::too_many_lines)]
pub(super) fn training_seeds_with_ngrams(
    corpus: &Path,
    system: &Path,
    additions: &Path,
    bigram_limit: usize,
    trigram_limit: usize,
) -> Result<crate::TrainingNgrams, BoxError> {
    let result = training_seed_builder(
        corpus,
        system,
        additions,
        bigram_limit,
        trigram_limit,
        false,
    )?;
    Ok((result.seeds, result.bigrams, result.trigrams))
}

pub(super) fn training_seeds_with_crf_features(
    corpus: &Path,
    system: &Path,
    additions: &Path,
    bigram_limit: usize,
    trigram_limit: usize,
) -> Result<crate::TrainingCrfNgrams, BoxError> {
    training_seed_builder(corpus, system, additions, bigram_limit, trigram_limit, true)
}

#[allow(clippy::too_many_lines)]
fn training_seed_builder(
    corpus: &Path,
    system: &Path,
    additions: &Path,
    bigram_limit: usize,
    trigram_limit: usize,
    collect_crf_features: bool,
) -> Result<crate::TrainingCrfNgrams, BoxError> {
    let (mut words, mut pos_names, _, source_ids) = compile_word_table_from_sources(
        &corpus.join("vocab.tsv"),
        &corpus.join("1gram.by-id.tsv"),
        Some(system),
        Some(additions),
    )?;
    let affix_pos_selections = if collect_crf_features {
        promote_top_affixes(&mut words, &mut pos_names, DEFAULT_AFFIX_POS_LIMIT)?
    } else {
        Vec::new()
    };
    for (index, word) in words.iter_mut().enumerate() {
        word.word_id = u32::try_from(index)?;
    }
    let ids = source_ids
        .iter()
        .map(|id| id.map(u32::try_from).transpose())
        .collect::<Result<Vec<_>, _>>()?;
    let by_id = words.iter().map(|word| (word.word_id, word)).collect();
    let retained_trigrams = if trigram_limit == 0 {
        Vec::new()
    } else {
        let mut retained =
            read_retained_trigrams(&corpus.join("3gram.tsv"), &ids, DEFAULT_NGRAM_CUTOFF)?;
        retained.sort_unstable_by(|a, b| {
            b.3.cmp(&a.3)
                .then_with(|| (a.0, a.1, a.2).cmp(&(b.0, b.1, b.2)))
        });
        retained.truncate(trigram_limit);
        retained
    };
    let retained_parent_pairs = retained_trigrams
        .iter()
        .map(|&(first, middle, _, _)| (first, middle))
        .collect::<HashSet<_>>();
    let mut retained_parent_counts = HashMap::with_capacity(retained_parent_pairs.len());
    let mut pairs = BTreeMap::new();
    let mut totals = BTreeMap::new();
    let mut word_outgoing = BTreeMap::new();
    let mut pos_word_counts = HashMap::<u64, u64>::new();
    let mut word_pos_counts = HashMap::<u64, u64>::new();
    let reading_lengths = words
        .iter()
        .map(|word| {
            (
                word.word_id,
                reading_length_bucket(&word.reading, &word.pos),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let mut reading_length_bigram_counts = [0_u64; 81];
    let mut reading_length_bigram_context_counts = [0_u64; 9];
    read_raw_bigram_counts(
        &corpus.join("2gram.by-id.tsv"),
        &ids,
        &by_id,
        &mut pairs,
        &mut totals,
        &mut word_outgoing,
        &retained_parent_pairs,
        &mut retained_parent_counts,
        collect_crf_features.then_some(&reading_lengths),
        &mut reading_length_bigram_counts,
        &mut reading_length_bigram_context_counts,
        collect_crf_features.then_some(&mut pos_word_counts),
        collect_crf_features.then_some(&mut word_pos_counts),
    )?;
    let initial_bigrams = if bigram_limit == 0 {
        Vec::new()
    } else {
        let mut retained = read_retained_bigrams(&corpus.join("2gram.tsv"), &ids)?;
        retained.sort_unstable_by_key(|&(source, target, _)| (source, target));
        let mut merged: Vec<(u32, u32, u64)> = Vec::with_capacity(retained.len());
        for (source, target, count) in retained {
            if let Some(last) = merged.last_mut().filter(|x| (x.0, x.1) == (source, target)) {
                last.2 = last
                    .2
                    .checked_add(count)
                    .ok_or("retained bigram count overflow")?;
            } else {
                merged.push((source, target, count));
            }
        }
        merged.sort_unstable_by(|a, b| b.2.cmp(&a.2).then_with(|| (a.0, a.1).cmp(&(b.0, b.1))));
        merged.truncate(bigram_limit);
        merged
            .into_iter()
            .map(|(source, target, count)| {
                let outgoing_count = word_outgoing
                    .get(&source)
                    .copied()
                    .ok_or("retained bigram has no raw outgoing count")?;
                if count > outgoing_count || outgoing_count == 0 {
                    return Err("retained bigram count exceeds raw outgoing count".into());
                }
                Ok(crate::TrainingBigram {
                    source,
                    target,
                    count,
                    outgoing_count,
                })
            })
            .collect::<Result<Vec<_>, BoxError>>()?
    };
    let initial_trigrams = retained_trigrams
        .into_iter()
        .map(|(first, middle, target, count)| {
            let parent_count = retained_parent_counts
                .get(&(first, middle))
                .copied()
                .ok_or("retained trigram has no raw parent bigram")?;
            if count > parent_count || parent_count == 0 {
                return Err("retained trigram count exceeds its raw parent bigram".into());
            }
            Ok(crate::TrainingTrigram {
                first,
                middle,
                target,
                count,
                parent_count,
            })
        })
        .collect::<Result<Vec<_>, BoxError>>()?;
    let pos_word_bigrams =
        retain_top_pos_word_bigrams(pos_word_counts, DEFAULT_POS_WORD_BIGRAM_LIMIT);
    let word_pos_bigrams = retain_top_word_pos_bigrams(
        word_pos_counts,
        &word_outgoing,
        DEFAULT_WORD_POS_BIGRAM_LIMIT,
    )?;
    let (pos_trigrams, reading_length_trigram_counts, reading_length_trigram_context_counts) =
        if collect_crf_features {
            let counts = read_full_trigram_feature_counts(
                &corpus.join("3gram.by-id.tsv"),
                &ids,
                &reading_lengths,
            )?;
            (
                Vec::new(),
                counts.reading_length_trigrams,
                counts.reading_length_contexts,
            )
        } else {
            (Vec::new(), [0_u64; 729], [0_u64; 81])
        };
    let mut pos_context_counts = vec![0_u64; pos_names.len() + 1];
    for (pos, count) in totals {
        let slot = pos_context_counts
            .get_mut(pos as usize)
            .ok_or("POS context ID is outside the seed POS table")?;
        *slot = count;
    }
    let seeds = crate::TrainingSeeds {
        words: words
            .into_iter()
            .map(|word| crate::TrainingWord {
                surface: word.surface,
                reading: word.reading,
                pos_id: word.pos_id,
                log_unigram: word.word_unigram_in_pos_raw.ln(),
                model_only: word.flags & 2 != 0,
            })
            .collect(),
        pos_names,
        pos_bigrams: pairs
            .into_iter()
            .map(|((a, b), count)| {
                let denominator = pos_context_counts[a as usize];
                (a, b, (count_as_f64(count) / count_as_f64(denominator)).ln())
            })
            .collect(),
    };
    Ok(crate::TrainingCrfNgrams {
        seeds,
        bigrams: initial_bigrams,
        trigrams: initial_trigrams,
        pos_word_bigrams,
        word_pos_bigrams,
        affix_pos_selections,
        feature_counts: crate::TrainingFeatureCounts {
            pos_context_counts,
            pos_trigrams,
            include_pos_trigrams: false,
            reading_length_bigram_counts,
            reading_length_bigram_context_counts,
            reading_length_trigram_counts,
            reading_length_trigram_context_counts,
        },
    })
}

fn affix_kind(surface: &str, pos: &str) -> Option<&'static str> {
    if EXCLUDED_AFFIX_SURFACES.contains(&surface) {
        return None;
    }
    let fields = pos.split(',').collect::<Vec<_>>();
    // The experiment refines only noun affixes. Verb suffixes already have
    // informative POS tags and counters (助数詞/数詞) are not the target.
    if (fields.first().copied() == Some("接頭詞") && fields.contains(&"名詞接続"))
        || (fields.first().copied() == Some("名詞")
            && fields
                .iter()
                .any(|field| matches!(*field, "接頭詞" | "接頭語")))
    {
        Some("接頭語")
    } else if fields.first().copied() == Some("名詞")
        && fields
            .iter()
            .any(|field| matches!(*field, "接尾" | "接尾語"))
        && !fields
            .iter()
            .any(|field| matches!(*field, "助数詞" | "数詞" | "数"))
    {
        Some("接尾語")
    } else {
        None
    }
}

/// Refine the 100 highest raw-unigram noun-affix entries with distinct POS IDs.
/// Each new ID contains its complete old POS plus that lexeme's affix identity;
/// no existing POS classes are merged. Then recompute P(word | POS).
fn promote_top_affixes(
    words: &mut [Word],
    pos_names: &mut Vec<String>,
    limit: usize,
) -> Result<Vec<crate::TrainingAffixPosSelection>, BoxError> {
    let mut ranked = words
        .iter()
        .enumerate()
        .filter(|(_, word)| word.flags & 1 == 0 && affix_kind(&word.surface, &word.pos).is_some())
        .map(|(index, word)| {
            (
                index,
                word.unigram_count,
                word.surface.clone(),
                word.reading.clone(),
                word.pos.clone(),
                affix_kind(&word.surface, &word.pos).expect("filtered affix"),
            )
        })
        .collect::<Vec<_>>();
    ranked.sort_unstable_by(|a, b| {
        b.1.cmp(&a.1)
            .then_with(|| a.2.cmp(&b.2))
            .then_with(|| a.3.cmp(&b.3))
            .then_with(|| a.4.cmp(&b.4))
    });
    ranked.truncate(limit);
    let selections = ranked
        .iter()
        .map(
            |(_, weighted_count, surface, reading, original_pos, assigned_pos)| {
                crate::TrainingAffixPosSelection {
                    surface: surface.clone(),
                    reading: reading.clone(),
                    original_pos: original_pos.clone(),
                    assigned_pos: format!(
                        "{original_pos},CRF頻出{assigned_pos}:{surface}/{reading}"
                    ),
                    unigram_count: *weighted_count / 10,
                }
            },
        )
        .collect::<Vec<_>>();
    for (index, _, surface, reading, original_pos, assigned_pos) in ranked {
        // A lexeme-specific sub-POS is deliberate: generic shared prefix/suffix
        // labels would collapse distinctions the user asked to preserve.
        words[index].pos = format!("{original_pos},CRF頻出{assigned_pos}:{surface}/{reading}");
    }

    *pos_names = words
        .iter()
        .filter(|word| word.flags & 1 == 0)
        .map(|word| word.pos.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    for word in words.iter_mut().filter(|word| word.flags & 1 == 0) {
        word.pos_id =
            resolve_pos_id(pos_names, &word.pos).ok_or("failed to assign promoted affix POS ID")?;
    }
    let mut pos_totals = vec![0_u64; pos_names.len() + 1];
    for word in words.iter().filter(|word| word.flags & 1 == 0) {
        let total = pos_totals
            .get_mut(word.pos_id as usize)
            .ok_or("promoted affix POS ID is out of range")?;
        *total = total
            .checked_add(word.unigram_count)
            .ok_or("promoted affix POS unigram count overflow")?;
    }
    for word in words.iter_mut().filter(|word| word.flags & 1 == 0) {
        let denominator = pos_totals[word.pos_id as usize];
        if denominator == 0 {
            return Err("promoted affix POS has no unigram mass".into());
        }
        word.word_unigram_in_pos_raw = count_as_f64(word.unigram_count) / count_as_f64(denominator);
        word.word_unigram_in_pos = word.word_unigram_in_pos_raw;
    }
    Ok(selections)
}

fn retain_top_pos_word_bigrams(
    counts: HashMap<u64, u64>,
    limit: usize,
) -> Vec<crate::TrainingPosWordBigram> {
    let mut ranked = counts.into_iter().collect::<Vec<_>>();
    ranked.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    ranked.truncate(limit);
    ranked
        .into_iter()
        .map(|(key, count)| crate::TrainingPosWordBigram {
            source_pos: u32::try_from(key >> 32).expect("POS ID fits u32"),
            target_word: u32::try_from(key & u64::from(u32::MAX)).expect("word ID fits u32"),
            count,
        })
        .collect()
}

fn retain_top_word_pos_bigrams(
    counts: HashMap<u64, u64>,
    outgoing: &BTreeMap<u32, u64>,
    limit: usize,
) -> Result<Vec<crate::TrainingWordPosBigram>, BoxError> {
    let mut ranked = counts.into_iter().collect::<Vec<_>>();
    ranked.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    ranked.truncate(limit);
    ranked
        .into_iter()
        .map(|(key, count)| {
            let source_word = u32::try_from(key >> 32)?;
            let target_pos = u32::try_from(key & u64::from(u32::MAX))?;
            let outgoing_count = outgoing
                .get(&source_word)
                .copied()
                .ok_or("retained word-POS bigram has no raw outgoing count")?;
            if count > outgoing_count || outgoing_count == 0 {
                return Err("retained word-POS count exceeds raw outgoing count".into());
            }
            Ok(crate::TrainingWordPosBigram {
                source_word,
                target_pos,
                count,
                outgoing_count,
            })
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_lines)]
#[allow(clippy::cast_precision_loss)]
fn read_raw_bigram_counts(
    path: &Path,
    corpus_word_ids: &[Option<u32>],
    words_by_id: &BTreeMap<u32, &Word>,
    pos_pair_counts: &mut BTreeMap<(u32, u32), u64>,
    pos_context_counts: &mut BTreeMap<u32, u64>,
    word_outgoing_counts: &mut BTreeMap<u32, u64>,
    selected_parent_pairs: &HashSet<(u32, u32)>,
    selected_parent_counts: &mut HashMap<(u32, u32), u64>,
    reading_lengths: Option<&BTreeMap<u32, u8>>,
    reading_length_counts: &mut [u64; 81],
    reading_length_contexts: &mut [u64; 9],
    mut pos_word_counts: Option<&mut HashMap<u64, u64>>,
    mut word_pos_counts: Option<&mut HashMap<u64, u64>>,
) -> Result<(), BoxError> {
    let total_bytes = fs::metadata(path)?.len();
    let report_interval = 256_u64 * 1024 * 1024;
    let mut next_report = report_interval;
    let mut bytes_read = 0_u64;
    let mut reader = BufReader::new(File::open(path)?);
    let mut line = String::new();
    let mut first = true;
    while reader.read_line(&mut line)? != 0 {
        bytes_read = bytes_read.saturating_add(u64::try_from(line.len())?);
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if first {
            first = false;
            if trimmed == "id1\tid2\tcount" {
                line.clear();
                continue;
            }
        }
        let mut fields = trimmed.split('\t');
        let first_id = fields
            .next()
            .ok_or("malformed 2gram.by-id row")?
            .parse::<usize>()?;
        let second_id = fields
            .next()
            .ok_or("malformed 2gram.by-id row")?
            .parse::<usize>()?;
        let count = fields
            .next()
            .ok_or("malformed 2gram.by-id row")?
            .parse::<u64>()?;
        if fields.next().is_some() || count == 0 {
            return Err("malformed or zero-count 2gram.by-id row".into());
        }
        let source = corpus_word_ids
            .get(first_id)
            .copied()
            .flatten()
            .ok_or("2gram source ID has no dictionary word ID")?;
        let target = corpus_word_ids
            .get(second_id)
            .copied()
            .flatten()
            .ok_or("2gram target ID has no dictionary word ID")?;
        let source_word = words_by_id
            .get(&source)
            .ok_or("2gram source references missing vocabulary word")?;
        let target_word = words_by_id
            .get(&target)
            .ok_or("2gram target references missing vocabulary word")?;
        checked_add_map(
            pos_pair_counts,
            (source_word.pos_id, target_word.pos_id),
            count,
        )?;
        checked_add_map(pos_context_counts, source_word.pos_id, count)?;
        checked_add_map(word_outgoing_counts, source, count)?;
        if let Some(counts) = pos_word_counts.as_deref_mut() {
            let key = (u64::from(source_word.pos_id) << 32) | u64::from(target);
            checked_add_hash_map(counts, key, count)?;
        }
        if let Some(counts) = word_pos_counts.as_deref_mut() {
            let key = (u64::from(source) << 32) | u64::from(target_word.pos_id);
            checked_add_hash_map(counts, key, count)?;
        }
        let pair = (source, target);
        if selected_parent_pairs.contains(&pair) {
            checked_add_hash_map(selected_parent_counts, pair, count)?;
        }
        if let Some(reading_lengths) = reading_lengths {
            let source_length = usize::from(
                *reading_lengths
                    .get(&source)
                    .ok_or("missing source reading length")?,
            );
            let target_length = usize::from(
                *reading_lengths
                    .get(&target)
                    .ok_or("missing target reading length")?,
            );
            checked_add_array(
                reading_length_counts,
                source_length * 9 + target_length,
                count,
            )?;
            checked_add_array(reading_length_contexts, source_length, count)?;
        }
        line.clear();
        if bytes_read >= next_report {
            let percent = if total_bytes == 0 {
                100.0
            } else {
                bytes_read as f64 * 100.0 / total_bytes as f64
            };
            eprintln!(
                "crf-feature-scan file={} bytes={}/{} ({percent:.1}%)",
                path.display(),
                bytes_read.min(total_bytes),
                total_bytes
            );
            next_report = next_report.saturating_add(report_interval);
        }
    }
    Ok(())
}

fn reading_length_bucket(reading: &str, pos: &str) -> u8 {
    if reading.is_empty() || pos.starts_with("記号,") || pos == "記号" {
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

fn checked_add_array<const N: usize>(
    array: &mut [u64; N],
    index: usize,
    count: u64,
) -> Result<(), BoxError> {
    let slot = array
        .get_mut(index)
        .ok_or("ngram bucket index is out of range")?;
    *slot = slot
        .checked_add(count)
        .ok_or("reading-length ngram count overflow")?;
    Ok(())
}

struct FullTrigramFeatureCounts {
    reading_length_trigrams: [u64; 729],
    reading_length_contexts: [u64; 81],
}

#[allow(clippy::cast_precision_loss)]
fn read_full_trigram_feature_counts(
    path: &Path,
    corpus_word_ids: &[Option<u32>],
    reading_lengths: &BTreeMap<u32, u8>,
) -> Result<FullTrigramFeatureCounts, BoxError> {
    // This experiment drops POS trigrams entirely but retains the exhaustive
    // 9^3 reading-length statistics from every source trigram row.
    let mut reading_length_counts = [0_u64; 729];
    let mut reading_length_contexts = [0_u64; 81];
    let total_bytes = fs::metadata(path)?.len();
    let report_interval = 256_u64 * 1024 * 1024;
    let mut next_report = report_interval;
    let mut bytes_read = 0_u64;
    let mut reader = BufReader::new(File::open(path)?);
    let mut line = String::new();
    let mut first = true;
    while reader.read_line(&mut line)? != 0 {
        bytes_read = bytes_read.saturating_add(u64::try_from(line.len())?);
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if first {
            first = false;
            if trimmed == "id1\tid2\tid3\tcount" {
                line.clear();
                continue;
            }
        }
        let mut fields = trimmed.split('\t');
        let first_id = fields
            .next()
            .ok_or("malformed 3gram.by-id row")?
            .parse::<usize>()?;
        let middle_id = fields
            .next()
            .ok_or("malformed 3gram.by-id row")?
            .parse::<usize>()?;
        let target_id = fields
            .next()
            .ok_or("malformed 3gram.by-id row")?
            .parse::<usize>()?;
        let count = fields
            .next()
            .ok_or("malformed 3gram.by-id row")?
            .parse::<u64>()?;
        if fields.next().is_some() || count == 0 {
            return Err("malformed or zero-count 3gram.by-id row".into());
        }
        let first = corpus_word_ids
            .get(first_id)
            .copied()
            .flatten()
            .ok_or("3gram first ID has no dictionary word ID")?;
        let middle = corpus_word_ids
            .get(middle_id)
            .copied()
            .flatten()
            .ok_or("3gram middle ID has no dictionary word ID")?;
        let target = corpus_word_ids
            .get(target_id)
            .copied()
            .flatten()
            .ok_or("3gram target ID has no dictionary word ID")?;
        let length_a = usize::from(
            *reading_lengths
                .get(&first)
                .ok_or("missing first reading length")?,
        );
        let length_b = usize::from(
            *reading_lengths
                .get(&middle)
                .ok_or("missing middle reading length")?,
        );
        let length_c = usize::from(
            *reading_lengths
                .get(&target)
                .ok_or("missing target reading length")?,
        );
        checked_add_array(
            &mut reading_length_counts,
            (length_a * 9 + length_b) * 9 + length_c,
            count,
        )?;
        checked_add_array(&mut reading_length_contexts, length_a * 9 + length_b, count)?;
        line.clear();
        if bytes_read >= next_report {
            let percent = if total_bytes == 0 {
                100.0
            } else {
                bytes_read as f64 * 100.0 / total_bytes as f64
            };
            eprintln!(
                "crf-feature-scan file={} bytes={}/{} ({percent:.1}%) pos_trigrams=disabled",
                path.display(),
                bytes_read.min(total_bytes),
                total_bytes
            );
            next_report = next_report.saturating_add(report_interval);
        }
    }
    Ok(FullTrigramFeatureCounts {
        reading_length_trigrams: reading_length_counts,
        reading_length_contexts,
    })
}

fn read_retained_bigrams(
    path: &Path,
    corpus_word_ids: &[Option<u32>],
) -> Result<Vec<(u32, u32, u64)>, BoxError> {
    let mut output = Vec::new();
    let mut reader = BufReader::new(File::open(path)?);
    let mut line = String::new();
    let mut first = true;
    while reader.read_line(&mut line)? != 0 {
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if first {
            first = false;
            if trimmed == "id1\tid2\tcount" {
                line.clear();
                continue;
            }
        }
        let mut fields = trimmed.split('\t');
        let source_id = fields
            .next()
            .ok_or("malformed 2gram row")?
            .parse::<usize>()?;
        let target_id = fields
            .next()
            .ok_or("malformed 2gram row")?
            .parse::<usize>()?;
        let count = fields.next().ok_or("malformed 2gram row")?.parse::<u64>()?;
        if fields.next().is_some() {
            return Err("malformed 2gram row".into());
        }
        output.push((
            corpus_word_ids
                .get(source_id)
                .copied()
                .flatten()
                .ok_or("2gram source ID has no dictionary word ID")?,
            corpus_word_ids
                .get(target_id)
                .copied()
                .flatten()
                .ok_or("2gram target ID has no dictionary word ID")?,
            count,
        ));
        line.clear();
    }
    Ok(output)
}

fn read_retained_trigrams(
    path: &Path,
    corpus_word_ids: &[Option<u32>],
    ngram_cutoff: u64,
) -> Result<Vec<(u32, u32, u32, u64)>, BoxError> {
    let mut output = Vec::new();
    let mut reader = BufReader::new(File::open(path)?);
    let mut line = String::new();
    let mut first = true;
    while reader.read_line(&mut line)? != 0 {
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if first {
            first = false;
            if trimmed == "id1\tid2\tid3\tcount" {
                line.clear();
                continue;
            }
        }
        let mut fields = trimmed.split('\t');
        let source_id = fields
            .next()
            .ok_or("malformed 3gram row")?
            .parse::<usize>()?;
        let middle_id = fields
            .next()
            .ok_or("malformed 3gram row")?
            .parse::<usize>()?;
        let target_id = fields
            .next()
            .ok_or("malformed 3gram row")?
            .parse::<usize>()?;
        let count = fields.next().ok_or("malformed 3gram row")?.parse::<u64>()?;
        if fields.next().is_some() {
            return Err("malformed 3gram row".into());
        }
        output.push((
            corpus_word_ids
                .get(source_id)
                .copied()
                .flatten()
                .ok_or("3gram first ID has no dictionary word ID")?,
            corpus_word_ids
                .get(middle_id)
                .copied()
                .flatten()
                .ok_or("3gram middle ID has no dictionary word ID")?,
            corpus_word_ids
                .get(target_id)
                .copied()
                .flatten()
                .ok_or("3gram last ID has no dictionary word ID")?,
            count,
        ));
        line.clear();
    }
    output.sort_by_key(|(source, middle, target, _)| (*source, *middle, *target));
    let mut merged = Vec::<(u32, u32, u32, u64)>::with_capacity(output.len());
    for (source, middle, target, count) in output {
        if let Some(last) = merged.last_mut() {
            if (last.0, last.1, last.2) == (source, middle, target) {
                last.3 = last.3.checked_add(count).ok_or("trigram count overflow")?;
                continue;
            }
        }
        merged.push((source, middle, target, count));
    }
    merged.retain(|(_, _, _, count)| *count >= ngram_cutoff);
    Ok(merged)
}

fn checked_add_map<K: Ord>(map: &mut BTreeMap<K, u64>, key: K, count: u64) -> Result<(), BoxError> {
    let value = map.entry(key).or_default();
    *value = value.checked_add(count).ok_or("ngram count overflow")?;
    Ok(())
}

fn checked_add_hash_map<K: Eq + std::hash::Hash>(
    map: &mut HashMap<K, u64>,
    key: K,
    count: u64,
) -> Result<(), BoxError> {
    let value = map.entry(key).or_default();
    *value = value.checked_add(count).ok_or("ngram count overflow")?;
    Ok(())
}

fn make_trigram_blocks(
    records: &[TrigramRecord],
    edges: &[WordBigramEntry],
    contexts: &[WordBigramContext],
) -> Result<Vec<TrigramBlock>, BoxError> {
    let context_ranges = contexts
        .iter()
        .map(|context| (context.source_word, context.edge_count))
        .collect::<BTreeMap<_, _>>();
    let mut blocks = Vec::new();
    let mut start = 0;
    while start < records.len() {
        let parent_edge = records[start].parent_edge;
        let mut end = start + 1;
        while end < records.len() && records[end].parent_edge == parent_edge {
            end += 1;
        }
        let parent = edges
            .get(usize::try_from(parent_edge)?)
            .ok_or("trigram parent edge is out of range")?;
        let continuation_count = *context_ranges
            .get(&parent.target_word)
            .ok_or("trigram middle word has no bigram context")?;
        if records[start..end]
            .iter()
            .any(|record| record.slot >= continuation_count)
        {
            return Err("trigram slot exceeds its B/* continuation list".into());
        }
        blocks.push(TrigramBlock {
            parent_edge,
            continuation_count,
            record_start: start,
            record_count: end - start,
        });
        start = end;
    }
    Ok(blocks)
}

fn decode_token(encoded: &str, id: u32) -> (String, String, String, u8) {
    if id < 3 {
        return (encoded.to_owned(), String::new(), String::new(), 1);
    }
    if matches!(encoded, "未知語" | "英単語" | "名詞,数,*,*,*,*,*") {
        return (String::new(), String::new(), encoded.to_owned(), 2);
    }
    let Some((surface, feature)) = encoded.split_once(',') else {
        return (encoded.to_owned(), String::new(), encoded.to_owned(), 0);
    };
    let fields = feature.split(',').collect::<Vec<_>>();
    let pos = model_pos_for_word(surface, feature);
    let reading_field = if fields.len() >= 8 {
        fields.get(7).copied().unwrap_or_default()
    } else {
        ""
    };
    (surface.to_owned(), to_hiragana(reading_field), pos, 0)
}

/// Model POS keeps grammar attributes and lexicalizes particles by surface or
/// grouped sound, while auxiliaries use their normalized reading.
/// Case-particle compounds keep their MeCab subcategory and share a POS by
/// initial mora within that subcategory; sentence-final and adverbial
/// particles do likewise.
/// `MeCab` source features remain unchanged; this is an aiEngine model key.
pub(super) fn model_pos_for_word(surface: &str, feature: &str) -> String {
    let source_fields = feature.split(',').collect::<Vec<_>>();
    let fields = source_fields.iter().take(6).copied().collect::<Vec<_>>();
    if fields.len() < 6 {
        return feature.to_owned();
    }

    let case_particle_compound =
        fields[0] == "助詞" && fields[1] == "格助詞" && fields[2] == "連語";
    let sentence_final_particle = fields[0] == "助詞" && fields[1] == "終助詞";
    let adverbial_particle = fields[0] == "助詞" && fields[1] == "副助詞";
    let lexical_field = if case_particle_compound || sentence_final_particle || adverbial_particle {
        let reading = source_fields
            .get(7)
            .copied()
            .filter(|reading| !reading.is_empty() && *reading != "*")
            .unwrap_or(surface);
        initial_mora(reading)
    } else if fields[0] == "助動詞" {
        let reading = source_fields
            .get(7)
            .copied()
            .filter(|reading| !reading.is_empty() && *reading != "*")
            .unwrap_or(surface);
        to_hiragana(reading)
    } else if fields[0] == "助詞" {
        normalize_source_surface(surface)
    } else {
        "*".to_owned()
    };
    format!("{},{lexical_field}", fields.join(","))
}

fn initial_mora(value: &str) -> String {
    let hiragana = to_hiragana(value);
    let mut chars = hiragana.chars();
    let Some(first) = chars.next() else {
        return hiragana;
    };
    let mut mora = first.to_string();
    if let Some(next) = chars.next() {
        if matches!(next, 'ゃ' | 'ゅ' | 'ょ') {
            mora.push(next);
        }
    }
    mora
}

pub(super) fn to_hiragana(value: &str) -> String {
    value
        .chars()
        .map(|ch| match ch {
            '\u{30a1}'..='\u{30f6}' => char::from_u32(u32::from(ch) - 0x60).unwrap_or(ch),
            _ => ch,
        })
        .collect()
}

fn read_phrases(
    path: &Path,
    words: &[Word],
    corpus_source_ids: &[Option<usize>],
    phrase_limit: usize,
    per_prefix_limit: usize,
) -> Result<(Vec<Phrase>, PhraseIndex, PrefixIndex), BoxError> {
    let mut ranked_prefixes = BTreeMap::<String, Vec<RankedCandidateTarget>>::new();
    let mut reader = BufReader::new(File::open(path)?);
    let mut line = String::new();
    while reader.read_line(&mut line)? != 0 {
        let (ids, reading, unigram) = parse_prediction_candidate(&line, words, corpus_source_ids)?;
        let target = if ids.len() == 1 {
            CandidateTarget::Word(
                corpus_source_ids
                    .get(ids[0] as usize)
                    .copied()
                    .flatten()
                    .ok_or("single-word prediction candidate has no source word")?,
            )
        } else {
            CandidateTarget::Phrase(PhraseKey {
                ids,
                unigram_bits: unigram.to_bits(),
            })
        };
        for prefix in reading_prefixes(&reading) {
            insert_ranked_candidate(
                ranked_prefixes.entry(prefix).or_default(),
                RankedCandidateTarget {
                    target: target.clone(),
                    unigram,
                },
                per_prefix_limit,
            );
        }
        line.clear();
    }

    // The cache contains only the top-P(word) candidates, but complete-reading
    // lookup retains the normal phrase cutoff. Scan again so evicted cache
    // phrases do not occupy vocabulary space unnecessarily.
    let selected_cache_phrases = ranked_prefixes
        .values()
        .flat_map(|candidates| candidates.iter())
        .filter_map(|candidate| match &candidate.target {
            CandidateTarget::Phrase(key) => Some(key.clone()),
            CandidateTarget::Word(_) => None,
        })
        .collect::<BTreeSet<_>>();
    let mut phrases = Vec::new();
    let mut phrase_reading = BTreeMap::<String, Vec<u32>>::new();
    let mut phrase_ids = BTreeMap::<PhraseKey, u32>::new();
    let mut base_phrase_rows = 0_usize;
    let mut reader = BufReader::new(File::open(path)?);
    while reader.read_line(&mut line)? != 0 {
        let (ids, reading, unigram) = parse_prediction_candidate(&line, words, corpus_source_ids)?;
        if ids.len() == 1 {
            line.clear();
            continue;
        }
        let is_base_phrase = base_phrase_rows < phrase_limit;
        base_phrase_rows += 1;
        let key = PhraseKey {
            ids: ids.clone(),
            unigram_bits: unigram.to_bits(),
        };
        if (is_base_phrase || selected_cache_phrases.contains(&key))
            && let std::collections::btree_map::Entry::Vacant(entry) = phrase_ids.entry(key)
        {
            let phrase_id = u32::try_from(phrases.len())?;
            entry.insert(phrase_id);
            phrase_reading
                .entry(reading.clone())
                .or_default()
                .push(phrase_id);
            phrases.push(Phrase {
                ids,
                reading,
                unigram,
                vocab_offset: u32::MAX,
            });
        }
        line.clear();
    }

    let mut prefixes = BTreeMap::new();
    for (prefix, candidates) in ranked_prefixes {
        let mut targets = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            let target = match candidate.target {
                CandidateTarget::Word(source_id) => CacheTarget::Word(source_id),
                CandidateTarget::Phrase(key) => CacheTarget::Phrase(
                    *phrase_ids
                        .get(&key)
                        .ok_or("selected prediction phrase was not retained")?,
                ),
            };
            targets.push(target);
        }
        prefixes.insert(prefix, targets);
    }
    Ok((phrases, phrase_reading, prefixes))
}

fn parse_prediction_candidate(
    line: &str,
    words: &[Word],
    corpus_source_ids: &[Option<usize>],
) -> Result<(Vec<u32>, String, f64), BoxError> {
    let fields = line
        .trim_end_matches(['\r', '\n'])
        .split('\t')
        .collect::<Vec<_>>();
    if fields.len() < 2 {
        return Err("prediction candidate row must contain one to six IDs and P(w)".into());
    }
    let unigram = fields[fields.len() - 1].parse::<f64>()?;
    if !unigram.is_finite() || !(0.0..=1.0).contains(&unigram) || unigram == 0.0 {
        return Err("prediction candidate P(w) must be finite and in (0, 1]".into());
    }
    let ids = fields[..fields.len() - 1]
        .iter()
        .map(|field| field.parse::<u32>().map_err(Into::into))
        .collect::<Result<Vec<_>, BoxError>>()?;
    if !(1..=6).contains(&ids.len()) {
        return Err("prediction candidate must contain between one and six word IDs".into());
    }
    let mut reading = String::new();
    for id in &ids {
        let source_id = corpus_source_ids
            .get(*id as usize)
            .copied()
            .flatten()
            .ok_or("prediction candidate references a non-word corpus ID")?;
        let word = words
            .get(source_id)
            .ok_or("prediction candidate word ID outside source dictionary")?;
        if word.flags != 0 || word.reading.is_empty() {
            return Err("prediction candidate contains a non-indexable word".into());
        }
        reading.push_str(&word.reading);
    }
    Ok((ids, reading, unigram))
}

fn reading_prefixes(reading: &str) -> Vec<String> {
    let chars = reading.chars().collect::<Vec<_>>();
    (1..=chars.len().min(3))
        .map(|prefix_len| chars[..prefix_len].iter().collect::<String>())
        .filter(|prefix| prefix.chars().all(is_hiragana))
        .collect()
}

fn insert_ranked_candidate(
    bucket: &mut Vec<RankedCandidateTarget>,
    candidate: RankedCandidateTarget,
    limit: usize,
) {
    if let Some(previous) = bucket
        .iter()
        .find(|previous| previous.target == candidate.target)
        && previous.unigram >= candidate.unigram
    {
        return;
    }
    bucket.retain(|previous| previous.target != candidate.target);
    bucket.push(candidate);
    bucket.sort_by(|left, right| {
        right
            .unigram
            .total_cmp(&left.unigram)
            .then_with(|| left.target.cmp(&right.target))
    });
    bucket.truncate(limit);
}

fn is_hiragana(ch: char) -> bool {
    matches!(ch, '\u{3041}'..='\u{3096}')
}

fn make_key_groups(
    words: &[Word],
    phrase_reading: &BTreeMap<String, Vec<u32>>,
    prefix_cache: &PrefixIndex,
) -> Result<Vec<KeyGroup>, BoxError> {
    let mut groups = BTreeMap::<Vec<u8>, KeyGroup>::new();
    for (word_index, word) in words.iter().enumerate() {
        if word.flags != 0 {
            continue;
        }
        if word.surface.is_empty() || word.reading.is_empty() {
            return Err(format!(
                "vocab source ID {} has no indexable surface or reading",
                word.source_id
            )
            .into());
        }
        let reading_key = super::dict_char::encode(&word.reading)?;
        let surface_key = super::dict_char::encode(&word.surface)?;
        groups
            .entry(reading_key.clone())
            .or_insert_with(|| KeyGroup {
                key: reading_key.clone(),
                ..KeyGroup::default()
            })
            .reading_words
            .push(word_index);
        groups
            .entry(surface_key.clone())
            .or_insert_with(|| KeyGroup {
                key: surface_key,
                ..KeyGroup::default()
            })
            .display_words
            .push(word_index);
    }
    for (reading, phrase_ids) in phrase_reading {
        let key = super::dict_char::encode(reading)?;
        let group = groups.entry(key.clone()).or_insert_with(|| KeyGroup {
            key,
            ..KeyGroup::default()
        });
        group.phrase_ids.extend(phrase_ids.iter().copied());
    }
    for (prefix, cache_targets) in prefix_cache {
        let key = super::dict_char::encode(prefix)?;
        let group = groups.entry(key.clone()).or_insert_with(|| KeyGroup {
            key,
            ..KeyGroup::default()
        });
        group.cache_targets.extend(cache_targets.iter().cloned());
    }
    for group in groups.values_mut() {
        group.reading_words.sort_by(|left, right| {
            let lhs = &words[*left];
            let rhs = &words[*right];
            lhs.pos_id
                .cmp(&rhs.pos_id)
                .then_with(|| rhs.word_unigram_in_pos.total_cmp(&lhs.word_unigram_in_pos))
                .then_with(|| lhs.surface.cmp(&rhs.surface))
        });
        group.display_words.sort_by(|left, right| {
            let lhs = &words[*left];
            let rhs = &words[*right];
            lhs.pos_id
                .cmp(&rhs.pos_id)
                .then_with(|| rhs.word_unigram_in_pos.total_cmp(&lhs.word_unigram_in_pos))
                .then_with(|| lhs.reading.cmp(&rhs.reading))
        });
        group.phrase_ids.sort_unstable();
        group.phrase_ids.dedup();
        group.cache_targets.dedup();
        if group.cache_targets.len() > 5 {
            return Err("prediction prefix cache exceeds five targets".into());
        }
    }

    let mut ordered = groups.into_values().collect::<Vec<_>>();
    for index in 0..ordered.len() {
        let previous_common = index.checked_sub(1).map_or(0, |previous| {
            common_prefix_len(&ordered[previous].key, &ordered[index].key)
        });
        let next_common = ordered
            .get(index + 1)
            .map_or(0, |next| common_prefix_len(&ordered[index].key, &next.key));
        let unique_prefix_len = previous_common.max(next_common).saturating_add(1);
        let max_tail_prefix_len = ordered[index].key.len().saturating_sub(32);
        let prefix_len = unique_prefix_len
            .max(max_tail_prefix_len)
            .min(ordered[index].key.len());
        ordered[index].prefix = ordered[index].key[..prefix_len].to_vec();
        ordered[index].tail = ordered[index].key[prefix_len..].to_vec();
        if ordered[index].tail.len() > 32 {
            return Err("double-array tail exceeds 32 encoded bytes".into());
        }
    }
    let mut seen_prefixes = BTreeSet::new();
    if ordered
        .iter()
        .any(|group| !seen_prefixes.insert(group.prefix.clone()))
    {
        return Err("unique-prefix selection produced a duplicate index key".into());
    }
    Ok(ordered)
}

fn common_prefix_len(left: &[u8], right: &[u8]) -> usize {
    left.iter()
        .zip(right)
        .take_while(|(lhs, rhs)| lhs == rhs)
        .count()
}

impl DoubleArray {
    #[allow(clippy::too_many_lines)] // Keep the range-placement state machine together.
    fn build(groups: &mut [KeyGroup]) -> Result<Self, BoxError> {
        if groups.is_empty() {
            return Ok(Self {
                cells: vec![BaseCheck { base: 0, check: 0 }],
                group_leaves: Vec::new(),
            });
        }
        let mut entries = groups
            .iter()
            .enumerate()
            .map(|(group_index, group)| {
                let mut key = group.prefix.clone();
                key.push(0);
                (key, group_index)
            })
            .collect::<Vec<_>>();
        entries.sort_by(|left, right| left.0.cmp(&right.0));
        if entries.windows(2).any(|pair| pair[0].0 == pair[1].0) {
            return Err("double-array terminal prefixes are not unique".into());
        }

        let mut builder = DoubleArrayBuilder::new();
        let mut pending = BTreeSet::<(usize, usize, usize, usize, usize)>::new();
        pending.insert((entries.len(), 0, entries.len(), 0, 0));
        let mut leaves = vec![u32::MAX; groups.len()];

        while let Some(range) = pending.pop_first() {
            let (_, top, end, char_index, parent) = range;
            let first_char = usize::from(entries[top].0[char_index]);
            let mut free_index = builder.first_free()?;
            while free_index < first_char {
                free_index = builder.next_free(free_index)?;
            }
            let mut base = free_index - first_char;

            loop {
                if base == 0 || builder.base_used(base) {
                    free_index = builder.next_free(free_index)?;
                    base = free_index
                        .checked_sub(first_char)
                        .ok_or("double-array base underflow")?;
                    continue;
                }
                let mut collision = false;
                for entry in &entries[top..end] {
                    let child = base
                        .checked_add(usize::from(entry.0[char_index]))
                        .ok_or("double-array child index overflow")?;
                    if child < builder.cells.len() && builder.used(child) {
                        collision = true;
                        break;
                    }
                }
                if !collision {
                    break;
                }
                free_index = builder.next_free(free_index)?;
                base = free_index
                    .checked_sub(first_char)
                    .ok_or("double-array base underflow")?;
            }

            let max_char = entries[top..end]
                .iter()
                .map(|entry| usize::from(entry.0[char_index]))
                .max()
                .ok_or("empty double-array range")?;
            let last = base
                .checked_add(max_char)
                .and_then(|index| index.checked_add(0xff))
                .ok_or("double-array size overflow")?;
            builder.ensure_capacity(last)?;
            builder.mark_base(base)?;
            builder.cells[parent].base = i32::try_from(base)?;

            let mut entry_top = top;
            while entry_top < end {
                let target_char = entries[entry_top].0[char_index];
                let mut entry_end = entry_top + 1;
                while entry_end < end && entries[entry_end].0[char_index] == target_char {
                    entry_end += 1;
                }
                let child = base
                    .checked_add(usize::from(target_char))
                    .ok_or("double-array child index overflow")?;
                builder.mark_used(child)?;
                builder.cells[child].check = u32::try_from(parent)?;

                let is_single_entry = entry_top + 1 == entry_end;
                let next_is_end = entries[entry_top]
                    .0
                    .get(char_index + 1)
                    .copied()
                    .unwrap_or(0)
                    == 0;
                if is_single_entry && (target_char == 0 || next_is_end) {
                    let group_index = entries[entry_top].1;
                    builder.cells[child].base = -i32::try_from(group_index + 1)?;
                    leaves[group_index] = u32::try_from(child)?;
                } else {
                    pending.insert((
                        entry_end - entry_top,
                        entry_top,
                        entry_end,
                        char_index + 1,
                        child,
                    ));
                }
                entry_top = entry_end;
            }
        }

        if leaves.contains(&u32::MAX) {
            return Err("double-array build left an unassigned key leaf".into());
        }
        let last = builder
            .max_base
            .checked_add(0xff)
            .ok_or("double-array padding overflow")?;
        builder.ensure_capacity(last)?;
        for (group, leaf) in groups.iter_mut().zip(leaves) {
            group.leaf_node = leaf;
        }
        Ok(Self {
            cells: builder.cells,
            group_leaves: groups.iter().map(|group| group.leaf_node).collect(),
        })
    }

    fn bind_vocab_offsets(&mut self, groups: &[KeyGroup]) -> Result<(), BoxError> {
        if groups.len() != self.group_leaves.len() {
            return Err("double-array key count differs from vocabulary groups".into());
        }
        for (group, leaf) in groups.iter().zip(&self.group_leaves) {
            let offset = i32::try_from(group.vocab_offset)
                .map_err(|_| "vocabulary block exceeds signed 32-bit leaf offsets")?;
            let cell = self
                .cells
                .get_mut(*leaf as usize)
                .ok_or("double-array leaf is outside node array")?;
            cell.base = offset
                .checked_add(1)
                .and_then(i32::checked_neg)
                .ok_or("vocabulary leaf offset overflow")?;
        }
        Ok(())
    }

    fn encode(&self) -> Result<Vec<u8>, BoxError> {
        let mut out = Vec::with_capacity(4 + self.cells.len() * 8);
        put_u32(&mut out, u32::try_from(self.cells.len())?);
        for cell in &self.cells {
            out.extend_from_slice(&cell.base.to_le_bytes());
            put_u32(&mut out, cell.check);
        }
        Ok(out)
    }

    fn decode(bytes: &[u8]) -> Result<Self, BoxError> {
        let mut input = Decoder::new(bytes);
        let count = usize::try_from(input.u32()?)?;
        if count == 0 || count > input.bytes.len().saturating_sub(input.offset) / 8 {
            return Err("invalid double-array node count".into());
        }
        let mut cells = Vec::with_capacity(count);
        for _ in 0..count {
            let base = i32::from_le_bytes(input.take(4)?.try_into()?);
            let check = input.u32()?;
            cells.push(BaseCheck { base, check });
        }
        input.finish()?;
        if cells[0].base < 0 || cells[0].check != 0 {
            return Err("invalid double-array root node".into());
        }
        for (index, cell) in cells.iter().enumerate().skip(1) {
            if cell.check != u32::MAX && cell.check as usize >= cells.len() {
                return Err("double-array parent index is outside node array".into());
            }
            if cell.base < 0 && cell.check == u32::MAX {
                return Err("double-array terminal has no parent".into());
            }
            if cell.check != u32::MAX && cell.check as usize == index {
                return Err("double-array node is its own parent".into());
            }
        }
        Ok(Self {
            cells,
            group_leaves: Vec::new(),
        })
    }

    fn leaves(&self) -> Result<Vec<(u32, u32)>, BoxError> {
        let mut result = Vec::new();
        for (index, cell) in self.cells.iter().enumerate() {
            if cell.base < 0 {
                result.push((
                    u32::try_from(index)?,
                    u32::try_from(-i64::from(cell.base) - 1)?,
                ));
            }
        }
        Ok(result)
    }

    fn lookup_leaf(&self, key: &[u8], vocabulary: &[u8]) -> Option<u32> {
        let mut node = 0_usize;
        let mut offset = 0_usize;
        loop {
            let cell = *self.cells.get(node)?;
            if cell.base < 0 {
                if tail_matches(vocabulary, cell.base, &key[offset..]) {
                    return u32::try_from(node).ok();
                }
                return None;
            }
            if offset == key.len() {
                let child = usize::try_from(cell.base).ok()?;
                let terminal = self.cells.get(child)?;
                if terminal.check as usize == node
                    && terminal.base < 0
                    && tail_matches(vocabulary, terminal.base, &[])
                {
                    return u32::try_from(child).ok();
                }
                return None;
            }
            let child = usize::try_from(cell.base)
                .ok()?
                .checked_add(usize::from(key[offset]))?;
            let child_cell = self.cells.get(child)?;
            if child_cell.check as usize != node {
                return None;
            }
            node = child;
            offset += 1;
            if self.cells[node].base < 0 {
                if tail_matches(vocabulary, self.cells[node].base, &key[offset..]) {
                    return u32::try_from(node).ok();
                }
                return None;
            }
        }
    }

    fn key_for_leaf(&self, leaf: u32, vocabulary: &[u8]) -> Result<Vec<u8>, BoxError> {
        let leaf_index = usize::try_from(leaf)?;
        let leaf_cell = self
            .cells
            .get(leaf_index)
            .ok_or("double-array leaf outside node array")?;
        if leaf_cell.base >= 0 {
            return Err("requested double-array node is not a terminal".into());
        }
        let mut reversed = Vec::new();
        let mut node = leaf_index;
        while node != 0 {
            let parent = self.cells[node].check as usize;
            let parent_cell = self
                .cells
                .get(parent)
                .ok_or("double-array parent outside node array")?;
            if parent_cell.base < 0 {
                return Err("double-array path crosses a terminal".into());
            }
            let byte = i64::try_from(node)? - i64::from(parent_cell.base);
            if !(0..=255).contains(&byte) {
                return Err("double-array edge byte is out of range".into());
            }
            if byte != 0 {
                reversed.push(u8::try_from(byte)?);
            }
            node = parent;
        }
        reversed.reverse();
        let group_offset = usize::try_from(-i64::from(leaf_cell.base) - 1)?;
        reversed.extend_from_slice(tail_at(vocabulary, group_offset)?);
        Ok(reversed)
    }
}

struct DoubleArrayBuilder {
    cells: Vec<BaseCheck>,
    free_links: Vec<FreeLink>,
    used_cells: Vec<bool>,
    base_cells: Vec<bool>,
    free_top: u32,
    free_end: u32,
    max_base: usize,
}

impl DoubleArrayBuilder {
    const NONE: u32 = u32::MAX;

    fn new() -> Self {
        let mut builder = Self {
            cells: vec![BaseCheck::default()],
            free_links: vec![FreeLink {
                previous: Self::NONE,
                next: Self::NONE,
            }],
            used_cells: vec![false],
            base_cells: vec![false],
            free_top: 0,
            free_end: 0,
            max_base: 0,
        };
        builder.mark_used(0).expect("root cell exists");
        builder.mark_base(0).expect("root cell exists");
        builder.cells[0].check = 0;
        builder
    }

    fn used(&self, index: usize) -> bool {
        self.used_cells.get(index).copied().unwrap_or(false)
    }

    fn base_used(&self, index: usize) -> bool {
        self.base_cells.get(index).copied().unwrap_or(false)
    }

    fn first_free(&mut self) -> Result<usize, BoxError> {
        if self.free_top == Self::NONE {
            let old_len = self.cells.len();
            self.ensure_capacity(old_len)?;
        }
        usize::try_from(self.free_top).map_err(Into::into)
    }

    fn next_free(&mut self, current: usize) -> Result<usize, BoxError> {
        let next = self.free_links[current].next;
        if next != Self::NONE {
            return usize::try_from(next).map_err(Into::into);
        }
        let old_len = self.cells.len();
        self.ensure_capacity(old_len)?;
        let grown = self.free_links[current].next;
        if grown == Self::NONE {
            return Err("double-array free list failed to grow".into());
        }
        usize::try_from(grown).map_err(Into::into)
    }

    fn ensure_capacity(&mut self, last_index: usize) -> Result<(), BoxError> {
        if last_index < self.cells.len() {
            return Ok(());
        }
        let old_len = self.cells.len();
        let new_len = last_index
            .checked_add(1)
            .ok_or("double-array size overflow")?;
        u32::try_from(new_len).map_err(|_| "double-array exceeds u32 node range")?;
        self.cells.resize(new_len, BaseCheck::default());
        self.used_cells.resize(new_len, false);
        self.base_cells.resize(new_len, false);
        self.free_links.resize(
            new_len,
            FreeLink {
                previous: Self::NONE,
                next: Self::NONE,
            },
        );
        for index in old_len..new_len {
            let index_u32 = u32::try_from(index)?;
            self.free_links[index].previous = self.free_end;
            self.free_links[index].next = Self::NONE;
            if self.free_end == Self::NONE {
                self.free_top = index_u32;
            } else {
                self.free_links[self.free_end as usize].next = index_u32;
            }
            self.free_end = index_u32;
        }
        Ok(())
    }

    fn mark_used(&mut self, index: usize) -> Result<(), BoxError> {
        if index >= self.cells.len() {
            self.ensure_capacity(index)?;
        }
        if self.used_cells[index] {
            return Err("double-array attempted to reuse an occupied cell".into());
        }
        let link = self.free_links[index];
        if link.previous == Self::NONE {
            self.free_top = link.next;
        } else {
            self.free_links[link.previous as usize].next = link.next;
        }
        if link.next == Self::NONE {
            self.free_end = link.previous;
        } else {
            self.free_links[link.next as usize].previous = link.previous;
        }
        self.free_links[index] = FreeLink {
            previous: Self::NONE,
            next: Self::NONE,
        };
        self.used_cells[index] = true;
        Ok(())
    }

    fn mark_base(&mut self, index: usize) -> Result<(), BoxError> {
        if index >= self.base_cells.len() {
            self.ensure_capacity(index)?;
        }
        self.base_cells[index] = true;
        self.max_base = self.max_base.max(index);
        Ok(())
    }
}

fn tail_at(vocabulary: &[u8], group_offset: usize) -> Result<&[u8], BoxError> {
    let tag = *vocabulary
        .get(group_offset)
        .ok_or("vocabulary key group offset is outside block")?;
    if tag >> 5 != 1 {
        return Ok(&[]);
    }
    let len = usize::from(tag & 0x1f) + 1;
    if len > 32 {
        return Err("vocabulary tail record exceeds 32 bytes".into());
    }
    let start = group_offset.checked_add(1).ok_or("tail offset overflow")?;
    let end = start.checked_add(len).ok_or("tail size overflow")?;
    Ok(vocabulary
        .get(start..end)
        .ok_or("truncated vocabulary tail record")?)
}

fn tail_matches(vocabulary: &[u8], encoded_leaf_value: i32, remaining: &[u8]) -> bool {
    let Some(offset) = usize::try_from(-i64::from(encoded_leaf_value) - 1).ok() else {
        return false;
    };
    tail_at(vocabulary, offset).is_ok_and(|tail| tail == remaining)
}

fn remap_phrase_words(
    phrases: &mut [Phrase],
    source_word_ids: &[Option<u32>],
) -> Result<(), BoxError> {
    for phrase in phrases {
        for word_id in &mut phrase.ids {
            let mapped = source_word_ids
                .get(*word_id as usize)
                .ok_or("phrase source word ID is outside vocabulary")?
                .ok_or("phrase refers to a token without a vocabulary word record")?;
            *word_id = mapped;
        }
    }
    Ok(())
}

fn patch_vocabulary_phrase_words(
    vocabulary: &mut [u8],
    phrases: &[Phrase],
) -> Result<(), BoxError> {
    for phrase in phrases {
        let offset = usize::try_from(phrase.vocab_offset)?;
        let expected_tag = PREDICTION_PHRASE_TAG_BASE
            | u8::try_from(phrase.ids.len().checked_sub(1).ok_or("empty phrase")?)?;
        if vocabulary.get(offset) != Some(&expected_tag) {
            return Err("phrase record offset does not point to its encoded record".into());
        }
        let ids_start = offset.checked_add(1).ok_or("phrase ID offset overflow")?;
        for (index, word_id) in phrase.ids.iter().enumerate() {
            let start = ids_start
                .checked_add(index.checked_mul(4).ok_or("phrase ID offset overflow")?)
                .ok_or("phrase ID offset overflow")?;
            let end = start.checked_add(4).ok_or("phrase ID offset overflow")?;
            vocabulary
                .get_mut(start..end)
                .ok_or("phrase word ID is outside vocabulary block")?
                .copy_from_slice(&word_id.to_le_bytes());
        }
    }
    Ok(())
}

#[allow(clippy::too_many_lines)]
fn encode_vocabulary_groups(
    groups: &mut [KeyGroup],
    words: &mut [Word],
    phrases: &mut [Phrase],
) -> Result<(Vec<u8>, Vec<Option<u32>>), BoxError> {
    let leaf_by_key = groups
        .iter()
        .map(|group| (group.key.clone(), group.leaf_node))
        .collect::<BTreeMap<_, _>>();
    for word in words.iter_mut().filter(|word| word.flags == 0) {
        word.reading_leaf = *leaf_by_key
            .get(&super::dict_char::encode(&word.reading)?)
            .ok_or("word reading is missing from shared string index")?;
        word.display_leaf = *leaf_by_key
            .get(&super::dict_char::encode(&word.surface)?)
            .ok_or("word surface is missing from shared string index")?;
    }

    let mut source_word_ids = vec![None; words.len()];
    let mut next_offset = 1_usize;
    for group in groups.iter_mut() {
        group.vocab_offset = u32::try_from(next_offset)?;
        next_offset = next_offset
            .checked_add(if group.tail.is_empty() {
                0
            } else {
                1 + group.tail.len()
            })
            .ok_or("vocabulary block size overflow")?;
        if !group.cache_targets.is_empty() {
            next_offset = next_offset
                .checked_add(2 + group.cache_targets.len() * 4)
                .ok_or("vocabulary block size overflow")?;
        }
        for phrase_id in &group.phrase_ids {
            let phrase = phrases
                .get_mut(*phrase_id as usize)
                .ok_or("key group references an unknown phrase ID")?;
            if !(2..=6).contains(&phrase.ids.len()) {
                return Err("prediction phrase must contain between two and six word IDs".into());
            }
            phrase.vocab_offset = u32::try_from(next_offset)?;
            next_offset = next_offset
                .checked_add(1 + phrase.ids.len() * 4 + 8)
                .ok_or("vocabulary block size overflow")?;
        }
        for word_index in &group.reading_words {
            let word = words
                .get_mut(*word_index)
                .ok_or("reading word index outside input vocabulary")?;
            word.word_id = u32::try_from(next_offset)?;
            source_word_ids[word.source_id] = Some(word.word_id);
            next_offset = next_offset
                .checked_add(14)
                .ok_or("vocabulary block size overflow")?;
        }
        for chunk in group.display_words.chunks(32) {
            next_offset = next_offset
                .checked_add(1 + chunk.len() * 4)
                .ok_or("vocabulary block size overflow")?;
        }
        next_offset = next_offset
            .checked_add(
                if group.reading_words.is_empty() && group.phrase_ids.is_empty() {
                    1
                } else {
                    5
                },
            )
            .ok_or("vocabulary block size overflow")?;
    }
    for group in groups.iter_mut() {
        group.cache_target_offsets = group
            .cache_targets
            .iter()
            .map(|target| match target {
                CacheTarget::Word(source_id) => {
                    let word = words
                        .get(*source_id)
                        .ok_or("prediction cache references an unknown source word")?;
                    if word.flags != 0 || word.word_id == u32::MAX {
                        return Err("prediction cache word has no indexed vocabulary record".into());
                    }
                    Ok(word.word_id)
                }
                CacheTarget::Phrase(phrase_id) => {
                    let phrase = phrases
                        .get(*phrase_id as usize)
                        .ok_or("prediction cache references an unknown phrase ID")?;
                    if phrase.vocab_offset == u32::MAX {
                        return Err("prediction cache phrase has no vocabulary record".into());
                    }
                    Ok(phrase.vocab_offset)
                }
            })
            .collect::<Result<Vec<_>, BoxError>>()?;
    }
    let model_only_count = words.iter().filter(|word| word.flags & 2 != 0).count();
    let model_section_offset = next_offset;
    next_offset = next_offset
        .checked_add(8)
        .and_then(|offset| offset.checked_add(model_only_count.checked_mul(10)?))
        .and_then(|offset| offset.checked_add(8))
        .ok_or("model-only vocabulary section size overflow")?;
    let mut model_word_offset = model_section_offset + 8;
    for word in words.iter_mut().filter(|word| word.flags & 2 != 0) {
        word.word_id = u32::try_from(model_word_offset)?;
        source_word_ids[word.source_id] = Some(word.word_id);
        model_word_offset = model_word_offset
            .checked_add(10)
            .ok_or("model-only word offset overflow")?;
    }
    if next_offset > i32::MAX as usize {
        return Err("vocabulary block exceeds signed 32-bit leaf offset range".into());
    }

    let mut output = Vec::with_capacity(next_offset);
    output.push(0); // offset zero is not a valid word or key-group ID
    for group in groups {
        if output.len() != group.vocab_offset as usize {
            return Err("vocabulary group sizing pass is inconsistent".into());
        }
        if !group.tail.is_empty() {
            if group.tail.len() > 32 {
                return Err("vocabulary tail exceeds 32 encoded bytes".into());
            }
            output.push(0x20 | u8::try_from(group.tail.len() - 1)?);
            output.extend_from_slice(&group.tail);
        }
        if !group.cache_target_offsets.is_empty() {
            if group.cache_targets.len() > 5 {
                return Err("prediction prefix cache exceeds five targets".into());
            }
            output.push(PREFIX_CACHE_HEADER_TAG);
            if group.cache_target_offsets.len() != group.cache_targets.len() {
                return Err("prediction cache target offset count is inconsistent".into());
            }
            output.push(u8::try_from(group.cache_target_offsets.len())?);
            for target_offset in &group.cache_target_offsets {
                put_u32(&mut output, *target_offset);
            }
        }
        for phrase_id in &group.phrase_ids {
            let phrase = phrases
                .get(*phrase_id as usize)
                .ok_or("key group references an unknown phrase ID")?;
            if output.len() != phrase.vocab_offset as usize {
                return Err("phrase record sizing pass is inconsistent".into());
            }
            output.push(
                PREDICTION_PHRASE_TAG_BASE
                    | u8::try_from(phrase.ids.len().checked_sub(1).ok_or("empty phrase")?)?,
            );
            for id in &phrase.ids {
                put_u32(&mut output, *id);
            }
            put_f64(&mut output, phrase.unigram);
        }
        for word_index in &group.reading_words {
            let word = &words[*word_index];
            output.push(WORD_RECORD_TAG);
            put_u32(&mut output, word.display_leaf);
            put_u32(&mut output, word.pos_id);
            output.push(word.word_unigram_in_pos_code);
            put_u24(&mut output, 0)?;
            output.push(0);
        }
        for chunk in group.display_words.chunks(32) {
            output.push(DISPLAY_REFERENCE_TAG_BASE | u8::try_from(chunk.len() - 1)?);
            for word_index in chunk {
                put_u32(&mut output, words[*word_index].word_id);
            }
        }
        let has_reading = !group.reading_words.is_empty() || !group.phrase_ids.is_empty();
        output.push(0xe0 | u8::from(has_reading));
        if has_reading {
            put_u32(&mut output, group.leaf_node);
        }
    }
    if output.len() != model_section_offset {
        return Err("model-only vocabulary section is not adjacent to key groups".into());
    }
    output.extend_from_slice(MODEL_ONLY_SECTION);
    put_u32(&mut output, u32::try_from(model_only_count)?);
    for word in words.iter().filter(|word| word.flags & 2 != 0) {
        if output.len() != word.word_id as usize {
            return Err("model-only word ID differs from its record offset".into());
        }
        output.push(0x80); // Non-indexed model symbol record.
        put_u32(&mut output, word.pos_id);
        output.push(word.word_unigram_in_pos_code);
        put_u24(&mut output, 0)?;
        output.push(0);
    }
    put_u32(&mut output, u32::try_from(model_section_offset)?);
    output.extend_from_slice(MODEL_ONLY_FOOTER);
    if output.len() != next_offset {
        return Err("vocabulary serialization differs from sizing pass".into());
    }
    Ok((output, source_word_ids))
}

fn patch_vocabulary_bigram_refs(
    vocabulary: &mut [u8],
    words: &mut [Word],
    contexts: &[WordBigramContext],
) -> Result<(), BoxError> {
    let word_indices = words
        .iter()
        .enumerate()
        .map(|(index, word)| (word.word_id, index))
        .collect::<BTreeMap<_, _>>();
    for context in contexts {
        let word_index = *word_indices
            .get(&context.source_word)
            .ok_or("word bigram context has no vocabulary word record")?;
        let word = &mut words[word_index];
        if word.bigram_block_index != 0 {
            return Err("duplicate word bigram block for source word".into());
        }
        let one_based_index = context
            .first_edge
            .checked_add(1)
            .ok_or("word bigram block index overflow")?;
        let pointer_offset = usize::try_from(word.word_id)?
            .checked_add(if word.flags & 2 != 0 { 6 } else { 10 })
            .ok_or("vocabulary bigram pointer offset overflow")?;
        let pointer = vocabulary
            .get_mut(pointer_offset..pointer_offset + 4)
            .ok_or("vocabulary bigram pointer is outside vocabulary block")?;
        pointer[..3].copy_from_slice(&one_based_index.to_le_bytes()[..3]);
        pointer[3] = context.alpha_code;
        word.bigram_block_index = one_based_index;
        word.backoff_alpha_code = context.alpha_code;
    }
    Ok(())
}

fn encode_pos_names(values: &[String]) -> Result<Vec<u8>, BoxError> {
    let mut out = Vec::new();
    put_u32(&mut out, u32::try_from(values.len())?);
    for value in values {
        put_string(&mut out, value)?;
    }
    Ok(out)
}

/// Resolves a user's stored POS feature string against one loaded dictionary's local table.
pub(super) fn resolve_pos_id(pos_names: &[String], feature: &str) -> Option<u32> {
    pos_names
        .binary_search_by(|candidate| candidate.as_str().cmp(feature))
        .ok()
        .and_then(|id| u32::try_from(id + 1).ok())
}

#[derive(Clone, Debug)]
struct ProbabilityCodebook {
    model_id: u8,
    log_values: Vec<f32>,
}

// Frequency ratios and k-means weighting need floating-point arithmetic; counts
// above 2^53 lose integer precision, which is acceptable for these probabilities.
#[allow(clippy::cast_precision_loss)]
fn count_as_f64(count: u64) -> f64 {
    count as f64
}

// The codebook intentionally stores logarithms at f32 precision.
#[allow(clippy::cast_possible_truncation)]
fn log_probability_as_f32(value: f64) -> f32 {
    value as f32
}

impl ProbabilityCodebook {
    #[allow(clippy::too_many_lines)] // The deterministic k-means repair loop is one algorithm.
    fn fit(model_id: u8, probabilities: &[f64]) -> Result<(Self, Vec<u8>), BoxError> {
        let mut logs = probabilities
            .iter()
            .copied()
            .filter(|value| value.is_finite() && *value > 0.0)
            .map(f64::ln)
            .collect::<Vec<_>>();
        if logs.is_empty() {
            return Ok((
                Self {
                    model_id,
                    log_values: Vec::new(),
                },
                vec![0; probabilities.len()],
            ));
        }
        logs.sort_by(f64::total_cmp);
        let mut distinct = Vec::<(f64, u64)>::new();
        for value in &logs {
            if let Some((last, weight)) = distinct.last_mut() {
                if last.total_cmp(value).is_eq() {
                    *weight = weight
                        .checked_add(1)
                        .ok_or("codebook input weight overflow")?;
                    continue;
                }
            }
            distinct.push((*value, 1));
        }
        let cluster_count = distinct.len().min(255);
        let mut centers = (0..cluster_count)
            .map(|index| {
                let distinct_index = if cluster_count == 1 {
                    0
                } else {
                    index * (distinct.len() - 1) / (cluster_count - 1)
                };
                distinct[distinct_index].0
            })
            .collect::<Vec<_>>();

        for _ in 0..64 {
            let mut sums = vec![0.0_f64; centers.len()];
            let mut weights = vec![0_u64; centers.len()];
            for (value, weight) in &distinct {
                let slot = nearest_center(centers.as_slice(), *value);
                sums[slot] += value * count_as_f64(*weight);
                weights[slot] = weights[slot]
                    .checked_add(*weight)
                    .ok_or("codebook cluster weight overflow")?;
            }
            let mut changed = false;
            for index in 0..centers.len() {
                if weights[index] != 0 {
                    let next = sums[index] / count_as_f64(weights[index]);
                    changed |= (next - centers[index]).abs() > 1.0e-12;
                    centers[index] = next;
                }
            }
            // Empty/collapsed clusters are repaired deterministically by moving
            // the empty center to the most poorly represented distinct value.
            for empty in 0..weights.len() {
                if weights[empty] != 0 {
                    continue;
                }
                let candidate = distinct
                    .iter()
                    .map(|(value, weight)| {
                        let closest = centers
                            .iter()
                            .enumerate()
                            .filter(|(index, _)| *index != empty)
                            .map(|(_, center)| (value - center).abs())
                            .fold(f64::INFINITY, f64::min);
                        (closest * closest * count_as_f64(*weight), *value)
                    })
                    .max_by(|left, right| left.0.total_cmp(&right.0))
                    .map(|(_, value)| value);
                if let Some(candidate) = candidate {
                    centers[empty] = candidate;
                    changed = true;
                }
            }
            centers.sort_by(f64::total_cmp);
            if !changed {
                break;
            }
        }

        let mut log_values = centers
            .into_iter()
            .map(log_probability_as_f32)
            .filter(|value| value.is_finite())
            .collect::<Vec<_>>();
        log_values.sort_by(f32::total_cmp);
        log_values.dedup_by(|left, right| left.to_bits() == right.to_bits());
        if log_values.is_empty() {
            return Err("probability codebook collapsed to no representable values".into());
        }
        let codebook = Self {
            model_id,
            log_values,
        };
        let codes = probabilities
            .iter()
            .map(|probability| {
                if !probability.is_finite() || *probability <= 0.0 {
                    Ok(0)
                } else {
                    codebook.encode(*probability)
                }
            })
            .collect::<Result<Vec<_>, BoxError>>()?;
        Ok((codebook, codes))
    }

    fn encode(&self, probability: f64) -> Result<u8, BoxError> {
        if !probability.is_finite() || probability <= 0.0 {
            return Err("cannot quantize nonpositive or nonfinite probability".into());
        }
        let log_value = log_probability_as_f32(probability.ln());
        let index = self
            .log_values
            .partition_point(|center| *center < log_value);
        let chosen = match (index.checked_sub(1), self.log_values.get(index)) {
            (Some(left), Some(right))
                if (log_value - self.log_values[left]).abs() <= (*right - log_value).abs() =>
            {
                left
            }
            (Some(left), None) => left,
            (_, Some(_)) => index,
            (None, None) => return Err("empty probability codebook".into()),
        };
        u8::try_from(chosen + 1).map_err(Into::into)
    }

    fn decode(&self, code: u8) -> Result<f64, BoxError> {
        if code == 0 {
            return Ok(0.0);
        }
        let log_value = *self
            .log_values
            .get(usize::from(code - 1))
            .ok_or("probability code is outside its codebook")?;
        Ok(f64::from(log_value).exp())
    }
}

fn nearest_center(centers: &[f64], value: f64) -> usize {
    let index = centers.partition_point(|center| *center < value);
    match (index.checked_sub(1), centers.get(index)) {
        (Some(left), Some(right)) if (value - centers[left]).abs() <= (right - value).abs() => left,
        (Some(left), None) => left,
        (_, Some(_)) => index,
        (None, None) => 0,
    }
}

fn encode_codebooks(codebooks: &[ProbabilityCodebook]) -> Result<Vec<u8>, BoxError> {
    let mut out = Vec::new();
    put_u32(&mut out, u32::try_from(codebooks.len())?);
    for codebook in codebooks {
        out.push(codebook.model_id);
        put_u16(&mut out, u16::try_from(codebook.log_values.len())?);
        for value in &codebook.log_values {
            out.extend_from_slice(&value.to_le_bytes());
        }
    }
    Ok(out)
}

fn encode_pos_bigrams(entries: &[PosBigramEntry], pos_count: usize) -> Result<Vec<u8>, BoxError> {
    let cell_count = pos_count
        .checked_mul(pos_count)
        .ok_or("POS bigram matrix size overflow")?;
    let mut output = vec![0_u8; cell_count];
    for entry in entries {
        if entry.source_pos == 0
            || entry.target_pos == 0
            || usize::try_from(entry.source_pos)? > pos_count
            || usize::try_from(entry.target_pos)? > pos_count
            || entry.code == 0
        {
            return Err("invalid POS bigram cell".into());
        }
        let row = usize::try_from(entry.source_pos - 1)?;
        let column = usize::try_from(entry.target_pos - 1)?;
        let cell = row
            .checked_mul(pos_count)
            .and_then(|base| base.checked_add(column))
            .ok_or("POS bigram matrix index overflow")?;
        let slot = output
            .get_mut(cell)
            .ok_or("POS bigram cell is outside the dense matrix")?;
        if *slot != 0 {
            return Err("duplicate POS bigram cell".into());
        }
        *slot = entry.code;
    }
    Ok(output)
}

fn encode_word_bigrams(
    contexts: &[WordBigramContext],
    entries: &[WordBigramEntry],
) -> Result<Vec<u8>, BoxError> {
    let mut output = Vec::new();
    let mut expected_first_edge = 0_usize;
    let mut previous_block_max = None;
    for context in contexts {
        let start = usize::try_from(context.first_edge)?;
        let end = start
            .checked_add(usize::try_from(context.edge_count)?)
            .ok_or("word bigram context range overflow")?;
        if start != expected_first_edge || context.edge_count == 0 {
            return Err("word bigram blocks do not form a contiguous edge table".into());
        }
        let selected = entries
            .get(start..end)
            .ok_or("word bigram context is outside the edge table")?;
        if selected
            .iter()
            .any(|entry| entry.source_word != context.source_word)
            || selected
                .windows(2)
                .any(|pair| pair[0].target_word >= pair[1].target_word)
        {
            return Err("word bigram block must contain its source and ascending targets".into());
        }
        let block_max = selected
            .last()
            .ok_or("empty word bigram block")?
            .target_word;
        if previous_block_max.is_some_and(|previous| previous < block_max) {
            return Err("word bigram blocks are not sorted by final target ID descending".into());
        }
        previous_block_max = Some(block_max);
        expected_first_edge = end;
    }
    if expected_first_edge != entries.len() {
        return Err("word bigram blocks do not cover all edges".into());
    }
    put_u32(&mut output, u32::try_from(entries.len())?);
    for entry in entries {
        put_u24(&mut output, entry.target_word)?;
        output.push(entry.code);
    }
    Ok(output)
}

fn encode_trigrams(
    records: &[TrigramRecord],
    blocks: &[TrigramBlock],
    bigram_edge_count: usize,
) -> Result<Vec<u8>, BoxError> {
    let mut parent_bitmap = vec![0_u64; bigram_edge_count.div_ceil(64)];
    let mut block_payload = Vec::new();
    let mut payload_starts = Vec::with_capacity(blocks.len());
    let mut previous_parent = None;
    for block in blocks {
        if block.parent_edge as usize >= bigram_edge_count
            || previous_parent.is_some_and(|previous| previous >= block.parent_edge)
        {
            return Err("trigram parent edges must be strictly increasing and in range".into());
        }
        previous_parent = Some(block.parent_edge);
        let bitmap_index = usize::try_from(block.parent_edge)?;
        parent_bitmap[bitmap_index / 64] |= 1_u64 << (bitmap_index % 64);
        let selected = records
            .get(block.record_start..block.record_start + block.record_count)
            .ok_or("trigram block record range is invalid")?;
        if selected.len() != block.record_count || selected.is_empty() {
            return Err("trigram block has no records".into());
        }
        let mut dense = vec![0_u8; usize::try_from(block.continuation_count)?];
        let mut sparse = Vec::new();
        let slot_width = trigram_slot_index_width(block.continuation_count);
        let mut last_slot = None;
        for record in selected {
            if record.slot >= block.continuation_count
                || last_slot.is_some_and(|previous| previous >= record.slot)
                || record.code == 0
            {
                return Err("invalid trigram slot ordering or probability code".into());
            }
            last_slot = Some(record.slot);
            dense[usize::try_from(record.slot)?] = record.code;
            put_trigram_slot(&mut sparse, record.slot, slot_width)?;
            sparse.push(record.code);
        }
        let (codec, encoded) = if sparse.len() < dense.len() {
            (1_u8, sparse)
        } else {
            (0_u8, dense)
        };
        payload_starts.push(u32::try_from(block_payload.len())?);
        block_payload.extend_from_slice(&encoded);
        let expected_len = if codec == 0 {
            usize::try_from(block.continuation_count)?
        } else {
            block
                .record_count
                .checked_mul(slot_width + 1)
                .ok_or("sparse trigram block size overflow")?
        };
        if encoded.len() != expected_len {
            return Err("trigram block encoding has an inconsistent byte length".into());
        }
    }
    let rank_group_count = bigram_edge_count.div_ceil(256);
    let mut ranks = Vec::with_capacity(rank_group_count + 1);
    let mut block_index = 0;
    for group in 0..rank_group_count {
        ranks.push(u32::try_from(block_index)?);
        let limit = (group + 1) * 256;
        while block_index < blocks.len() && (blocks[block_index].parent_edge as usize) < limit {
            block_index += 1;
        }
    }
    ranks.push(u32::try_from(block_index)?);

    let mut payload_start_bitmap = vec![0_u64; block_payload.len().div_ceil(64)];
    for start in &payload_starts {
        let index = usize::try_from(*start)?;
        payload_start_bitmap[index / 64] |= 1_u64 << (index % 64);
    }
    let payload_start_samples = payload_starts
        .iter()
        .step_by(256)
        .copied()
        .collect::<Vec<_>>();

    let mut output = Vec::new();
    put_u32(&mut output, u32::try_from(bigram_edge_count)?);
    put_u32(&mut output, u32::try_from(parent_bitmap.len())?);
    put_u32(&mut output, u32::try_from(ranks.len())?);
    put_u32(&mut output, u32::try_from(blocks.len())?);
    put_u32(&mut output, u32::try_from(block_payload.len())?);
    put_u32(&mut output, u32::try_from(payload_start_bitmap.len())?);
    put_u32(&mut output, u32::try_from(payload_start_samples.len())?);
    for word in parent_bitmap {
        put_u64(&mut output, word);
    }
    for rank in ranks {
        put_u32(&mut output, rank);
    }
    for word in payload_start_bitmap {
        put_u64(&mut output, word);
    }
    for sample in payload_start_samples {
        put_u32(&mut output, sample);
    }
    output.extend_from_slice(&block_payload);
    Ok(output)
}

fn trigram_slot_index_width(continuation_count: u32) -> usize {
    if continuation_count <= u32::from(u8::MAX) + 1 {
        1
    } else if continuation_count <= u32::from(u16::MAX) + 1 {
        2
    } else {
        4
    }
}

fn put_trigram_slot(output: &mut Vec<u8>, slot: u32, width: usize) -> Result<(), BoxError> {
    match width {
        1 => output.push(u8::try_from(slot)?),
        2 => put_u16(output, u16::try_from(slot)?),
        4 => put_u32(output, slot),
        _ => return Err("unsupported trigram slot index width".into()),
    }
    Ok(())
}

fn decode_codebooks(bytes: &[u8]) -> Result<BTreeMap<u8, ProbabilityCodebook>, BoxError> {
    let mut input = Decoder::new(bytes);
    let count = usize::try_from(input.u32()?)?;
    let mut codebooks = BTreeMap::new();
    for _ in 0..count {
        let model_id = input.u8()?;
        let value_count = usize::from(input.u16()?);
        if value_count > 255 {
            return Err("invalid probability codebook length".into());
        }
        let mut log_values = Vec::with_capacity(value_count);
        for _ in 0..value_count {
            log_values.push(f32::from_le_bytes(input.take(4)?.try_into()?));
        }
        if log_values.iter().any(|value| !value.is_finite())
            || log_values.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(
                "probability codebook values must be finite and strictly increasing".into(),
            );
        }
        if codebooks
            .insert(
                model_id,
                ProbabilityCodebook {
                    model_id,
                    log_values,
                },
            )
            .is_some()
        {
            return Err("duplicate probability codebook model ID".into());
        }
    }
    input.finish()?;
    Ok(codebooks)
}

fn decode_pos_bigrams(
    bytes: &[u8],
    pos_names: &[String],
    codebook: &ProbabilityCodebook,
) -> Result<Vec<PosBigramEntry>, BoxError> {
    let pos_count = pos_names.len();
    let expected_size = pos_count
        .checked_mul(pos_count)
        .ok_or("POS bigram matrix size overflow")?;
    if bytes.len() != expected_size {
        return Err("POS bigram block has the wrong dense matrix size".into());
    }
    let mut entries = Vec::new();
    for (source_index, row) in bytes.chunks_exact(pos_count).enumerate() {
        for (target_index, code) in row.iter().copied().enumerate() {
            if code == 0 {
                continue;
            }
            entries.push(PosBigramEntry {
                source_pos: u32::try_from(source_index + 1)?,
                target_pos: u32::try_from(target_index + 1)?,
                probability: codebook.decode(code)?,
                code,
            });
        }
    }
    Ok(entries)
}

#[allow(clippy::too_many_lines)]
fn decode_word_bigrams(
    bytes: &[u8],
    words: &BTreeMap<u32, Word>,
    word_codebook: &ProbabilityCodebook,
    backoff_codebook: &ProbabilityCodebook,
) -> Result<(Vec<WordBigramContext>, Vec<WordBigramEntry>), BoxError> {
    let mut input = Decoder::new(bytes);
    let edge_count = usize::try_from(input.u32()?)?;
    let expected_size = 4_usize
        .checked_add(
            edge_count
                .checked_mul(4)
                .ok_or("word bigram edge bytes overflow")?,
        )
        .ok_or("word bigram block size overflow")?;
    if expected_size != bytes.len() {
        return Err("word bigram block has the wrong edge table size".into());
    }
    let mut entries = Vec::with_capacity(edge_count);
    for _ in 0..edge_count {
        let target_word = input.u24()?;
        let code = input.u8()?;
        if !words.contains_key(&target_word) || code == 0 {
            return Err("invalid word bigram target or probability code".into());
        }
        entries.push(WordBigramEntry {
            source_word: u32::MAX,
            target_word,
            count: 0,
            probability: word_codebook.decode(code)?,
            code,
        });
    }
    input.finish()?;

    let mut starts = words
        .values()
        .filter(|word| word.bigram_block_index != 0)
        .map(|word| {
            Ok((
                usize::try_from(word.bigram_block_index - 1)?,
                word.word_id,
                word.backoff_alpha_code,
            ))
        })
        .collect::<Result<Vec<_>, BoxError>>()?;
    starts.sort_by_key(|(start, _, _)| *start);
    if edge_count == 0 {
        if !starts.is_empty() {
            return Err("vocabulary points to an empty word bigram table".into());
        }
        return Ok((Vec::new(), entries));
    }
    if starts.first().is_none_or(|(start, _, _)| *start != 0) {
        return Err("first word bigram block must start at edge zero".into());
    }
    if starts.windows(2).any(|pair| pair[0].0 >= pair[1].0)
        || starts.iter().any(|(start, _, _)| *start >= edge_count)
    {
        return Err("word vocabulary has duplicate or out-of-range bigram pointers".into());
    }

    let mut contexts = Vec::with_capacity(starts.len());
    let mut previous_block_max = None;
    for (block_index, (start, source_word, alpha_code)) in starts.iter().copied().enumerate() {
        let mut end = start + 1;
        while end < entries.len() && entries[end].target_word > entries[end - 1].target_word {
            end += 1;
        }
        if starts
            .get(block_index + 1)
            .is_some_and(|next| next.0 != end)
        {
            return Err("word bigram pointer disagrees with target-ID block boundary".into());
        }
        let selected = &mut entries[start..end];
        if selected
            .windows(2)
            .any(|pair| pair[0].target_word >= pair[1].target_word)
        {
            return Err("word bigram targets are not strictly ascending within a block".into());
        }
        let block_max = selected
            .last()
            .ok_or("word bigram pointer selects an empty block")?
            .target_word;
        if previous_block_max.is_some_and(|previous| previous < block_max) {
            return Err("word bigram blocks are not sorted by final target ID descending".into());
        }
        previous_block_max = Some(block_max);
        for entry in selected {
            entry.source_word = source_word;
        }
        let edge_count = u32::try_from(end - start)?;
        contexts.push(WordBigramContext {
            source_word,
            first_edge: u32::try_from(start)?,
            edge_count,
            alpha: backoff_codebook.decode(alpha_code)?,
            alpha_code,
        });
    }
    if contexts.last().is_none_or(|context| {
        usize::try_from(context.first_edge).unwrap_or(usize::MAX)
            + usize::try_from(context.edge_count).unwrap_or(usize::MAX)
            != edge_count
    }) {
        return Err("word bigram target-ID boundaries do not cover the edge table".into());
    }
    Ok((contexts, entries))
}

#[allow(clippy::too_many_lines)]
fn decode_trigrams(
    bytes: &[u8],
    contexts: &[WordBigramContext],
    edges: &[WordBigramEntry],
    words: &BTreeMap<u32, Word>,
    codebook: &ProbabilityCodebook,
) -> Result<Vec<DecodedTrigram>, BoxError> {
    let mut input = Decoder::new(bytes);
    let bigram_edge_count = usize::try_from(input.u32()?)?;
    let parent_bitmap_word_count = usize::try_from(input.u32()?)?;
    let parent_rank_count = usize::try_from(input.u32()?)?;
    let parent_count = usize::try_from(input.u32()?)?;
    let payload_len = usize::try_from(input.u32()?)?;
    let payload_start_bitmap_word_count = usize::try_from(input.u32()?)?;
    let payload_start_sample_count = usize::try_from(input.u32()?)?;
    if bigram_edge_count != edges.len()
        || parent_bitmap_word_count != bigram_edge_count.div_ceil(64)
    {
        return Err("trigram block has an invalid bigram bitmap extent".into());
    }
    if parent_rank_count != bigram_edge_count.div_ceil(256) + 1 {
        return Err("trigram rank directory has an invalid length".into());
    }
    if payload_start_bitmap_word_count != payload_len.div_ceil(64)
        || payload_start_sample_count != parent_count.div_ceil(256)
    {
        return Err("trigram payload succinct index has an invalid extent".into());
    }
    let mut parent_bitmap = Vec::with_capacity(parent_bitmap_word_count);
    for _ in 0..parent_bitmap_word_count {
        parent_bitmap.push(input.u64()?);
    }
    if bigram_edge_count % 64 != 0
        && parent_bitmap
            .last()
            .is_some_and(|last| *last >> (bigram_edge_count % 64) != 0)
    {
        return Err("trigram bitmap has bits beyond the bigram edge table".into());
    }
    let parent_ranks = (0..parent_rank_count)
        .map(|_| input.u32())
        .collect::<Result<Vec<_>, _>>()?;
    let parents = bitmap_parent_edges(&parent_bitmap, bigram_edge_count)?;
    if parents.len() != parent_count {
        return Err("trigram parent bitmap and count disagree".into());
    }
    let mut recomputed_rank = 0_u32;
    let mut parent_cursor = 0;
    for (group, rank) in parent_ranks.iter().enumerate() {
        if *rank != recomputed_rank {
            return Err("trigram rank directory contains an invalid rank".into());
        }
        if group + 1 < parent_rank_count {
            let end = ((group + 1) * 256).min(bigram_edge_count);
            while parent_cursor < parents.len() && usize::try_from(parents[parent_cursor])? < end {
                parent_cursor += 1;
            }
            recomputed_rank = u32::try_from(parent_cursor)?;
        }
    }
    let mut payload_start_bitmap = Vec::with_capacity(payload_start_bitmap_word_count);
    for _ in 0..payload_start_bitmap_word_count {
        payload_start_bitmap.push(input.u64()?);
    }
    if payload_len % 64 != 0
        && payload_start_bitmap
            .last()
            .is_some_and(|last| *last >> (payload_len % 64) != 0)
    {
        return Err("trigram payload-start bitmap has bits beyond its payload".into());
    }
    let payload_start_samples = (0..payload_start_sample_count)
        .map(|_| input.u32())
        .collect::<Result<Vec<_>, _>>()?;
    let payload = input.take(payload_len)?;
    input.finish()?;
    let payload_starts = bitmap_parent_edges(&payload_start_bitmap, payload_len)?;
    if payload_starts.len() != parent_count
        || (parent_count > 0 && payload_starts.first() != Some(&0))
        || payload_starts
            .iter()
            .step_by(256)
            .copied()
            .ne(payload_start_samples.iter().copied())
    {
        return Err("trigram payload-start succinct index is inconsistent".into());
    }
    if (parent_count == 0) != (payload_len == 0)
        || payload_starts
            .last()
            .is_some_and(|start| usize::try_from(*start).map_or(true, |value| value >= payload_len))
    {
        return Err("trigram payload-start vector does not delimit its payload".into());
    }

    let contexts_by_word = contexts
        .iter()
        .map(|context| (context.source_word, context))
        .collect::<BTreeMap<_, _>>();
    let mut output = Vec::new();
    for (block_index, parent_edge) in parents.into_iter().enumerate() {
        let offset = usize::try_from(payload_starts[block_index])?;
        let end = payload_starts
            .get(block_index + 1)
            .map_or(Ok(payload_len), |next| usize::try_from(*next))?;
        if offset >= end || end > payload.len() {
            return Err("trigram block start vector has an invalid boundary".into());
        }
        let parent = edges
            .get(usize::try_from(parent_edge)?)
            .ok_or("trigram parent edge is out of range")?;
        let middle_context = contexts_by_word
            .get(&parent.target_word)
            .ok_or("trigram block middle word has no outgoing context")?;
        let continuation_count = middle_context.edge_count;
        let encoded = &payload[offset..end];
        let slot_width = trigram_slot_index_width(continuation_count);
        let dense_len = usize::try_from(continuation_count)?;
        let sparse_record_width = slot_width + 1;
        let (codec, record_count) = if encoded.len() == dense_len {
            (0_u8, encoded.iter().filter(|code| **code != 0).count())
        } else if encoded.len() < dense_len && encoded.len() % sparse_record_width == 0 {
            (1_u8, encoded.len() / sparse_record_width)
        } else {
            return Err(
                "trigram block length does not identify a valid sparse or dense form".into(),
            );
        };
        if record_count == 0 {
            return Err("trigram block cannot be empty".into());
        }
        let mut slots = Vec::<(u32, u8)>::with_capacity(record_count);
        if codec == 0 {
            for (slot, code) in encoded.iter().copied().enumerate() {
                if code != 0 {
                    slots.push((u32::try_from(slot)?, code));
                }
            }
        } else {
            let mut sparse = Decoder::new(encoded);
            let mut previous_slot = None;
            for _ in 0..record_count {
                let slot = match slot_width {
                    1 => u32::from(sparse.u8()?),
                    2 => u32::from(sparse.u16()?),
                    4 => sparse.u32()?,
                    _ => return Err("unsupported trigram slot index width".into()),
                };
                let code = sparse.u8()?;
                if slot >= continuation_count
                    || code == 0
                    || previous_slot.is_some_and(|previous| previous >= slot)
                {
                    return Err("invalid sparse trigram slot".into());
                }
                previous_slot = Some(slot);
                slots.push((slot, code));
            }
            sparse.finish()?;
        }
        if slots.len() != record_count {
            return Err("trigram block record count disagrees with its payload".into());
        }
        for (slot, code) in slots {
            let edge_index = middle_context
                .first_edge
                .checked_add(slot)
                .ok_or("trigram slot index overflow")?;
            let continuation = edges
                .get(usize::try_from(edge_index)?)
                .ok_or("trigram slot points outside the B/* bigram row")?;
            if continuation.source_word != parent.target_word
                || !words.contains_key(&continuation.target_word)
            {
                return Err("trigram slot points to an invalid B/C edge".into());
            }
            output.push(DecodedTrigram {
                source_word: parent.source_word,
                middle_word: parent.target_word,
                target_word: continuation.target_word,
                parent_edge,
                slot,
                probability: codebook.decode(code)?,
                code,
                codec,
            });
        }
    }
    Ok(output)
}

fn bitmap_parent_edges(bitmap: &[u64], edge_count: usize) -> Result<Vec<u32>, BoxError> {
    let mut result = Vec::new();
    for (word_index, word) in bitmap.iter().copied().enumerate() {
        let mut remaining = word;
        while remaining != 0 {
            let bit = remaining.trailing_zeros();
            let index = word_index
                .checked_mul(64)
                .and_then(|base| base.checked_add(usize::try_from(bit).ok()?))
                .ok_or("trigram bitmap index overflow")?;
            if index >= edge_count {
                return Err("trigram bitmap points past the bigram edge table".into());
            }
            result.push(u32::try_from(index)?);
            remaining &= remaining - 1;
        }
    }
    Ok(result)
}

fn write_container(
    destination: &Path,
    payloads: Vec<([u8; 16], Vec<u8>, u64)>,
) -> Result<(), BoxError> {
    let block_count = payloads.len();
    let header_len = FIXED_HEADER_BYTES
        .checked_add(
            DIRECTORY_ENTRY_BYTES
                .checked_mul(block_count)
                .ok_or("header size overflow")?,
        )
        .ok_or("header size overflow")?;
    let mut next_offset = align8(header_len);
    let mut blocks = Vec::with_capacity(block_count);
    for (guid, bytes, item_count) in payloads {
        let offset = next_offset;
        next_offset = align8(
            offset
                .checked_add(bytes.len())
                .ok_or("file size overflow")?,
        );
        blocks.push(Block {
            guid,
            offset: u64::try_from(offset)?,
            bytes,
            item_count,
        });
    }

    let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let dict_uuid = make_uuid(now ^ u128::from(std::process::id()));
    let build_uuid =
        make_uuid(now.rotate_left(31) ^ 0xa1e_2026_u128 ^ u128::from(std::process::id()));
    let mut output = Vec::with_capacity(next_offset);
    output.extend_from_slice(MAGIC);
    output.extend_from_slice(&HEADER_GUID);
    put_u32(&mut output, VERSION);
    put_u32(&mut output, 0);
    output.extend_from_slice(&dict_uuid);
    output.extend_from_slice(&build_uuid);
    put_u32(&mut output, u32::try_from(block_count)?);
    put_u32(&mut output, u32::try_from(DIRECTORY_ENTRY_BYTES)?);
    for block in &blocks {
        output.extend_from_slice(&block.guid);
        put_u64(&mut output, block.offset);
        put_u64(&mut output, u64::try_from(block.bytes.len())?);
        put_u64(&mut output, block.item_count);
        put_u64(&mut output, fnv1a64(&block.bytes));
    }
    output.resize(align8(output.len()), 0);
    for block in blocks {
        let offset = usize::try_from(block.offset)?;
        output.resize(offset, 0);
        output.extend_from_slice(&block.bytes);
        output.resize(align8(output.len()), 0);
    }
    if let Some(parent) = destination
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    let mut file = File::create_new(destination)?;
    file.write_all(&output)?;
    file.sync_all()?;
    Ok(())
}

fn parse_container(bytes: &[u8]) -> Result<ParsedContainer, BoxError> {
    let mut input = Decoder::new(bytes);
    if input.take(8)? != MAGIC {
        return Err("not an aiEngine sdict file".into());
    }
    if input.take(16)? != HEADER_GUID {
        return Err("unknown aiEngine sdict format GUID".into());
    }
    if input.u32()? != VERSION {
        return Err("unsupported aiEngine sdict version".into());
    }
    let _flags = input.u32()?;
    let dict_uuid = input.array16()?;
    let build_uuid = input.array16()?;
    let block_count = usize::try_from(input.u32()?)?;
    if usize::try_from(input.u32()?)? != DIRECTORY_ENTRY_BYTES {
        return Err("invalid block directory entry size".into());
    }
    let min_header_len = FIXED_HEADER_BYTES
        .checked_add(
            DIRECTORY_ENTRY_BYTES
                .checked_mul(block_count)
                .ok_or("directory overflow")?,
        )
        .ok_or("directory overflow")?;
    if min_header_len > bytes.len() {
        return Err("truncated block directory".into());
    }
    let mut blocks = Vec::with_capacity(block_count);
    let mut seen = BTreeSet::new();
    for _ in 0..block_count {
        let guid = input.array16()?;
        let offset = usize::try_from(input.u64()?)?;
        let size = usize::try_from(input.u64()?)?;
        let item_count = input.u64()?;
        let checksum = input.u64()?;
        let end = offset.checked_add(size).ok_or("block range overflow")?;
        if offset < align8(min_header_len) || offset % 8 != 0 || end > bytes.len() {
            return Err("block points outside file or overlaps header".into());
        }
        if !seen.insert(guid) {
            return Err("duplicate block GUID".into());
        }
        if ![
            POS_NAMES,
            VOCABULARY,
            STRING_INDEX,
            QUANTIZATION,
            POS_BIGRAM,
            WORD_BIGRAM,
            TRIGRAM,
        ]
        .contains(&guid)
        {
            return Err(format!("unknown block GUID {}", format_guid(&guid)).into());
        }
        if fnv1a64(&bytes[offset..end]) != checksum {
            return Err(format!("checksum mismatch for block {}", format_guid(&guid)).into());
        }
        blocks.push(BlockInfo {
            guid,
            offset,
            size,
            item_count,
            checksum,
        });
    }
    let mut by_offset = blocks.clone();
    by_offset.sort_by_key(|block| block.offset);
    for pair in by_offset.windows(2) {
        if pair[0]
            .offset
            .checked_add(pair[0].size)
            .is_none_or(|end| end > pair[1].offset)
        {
            return Err("overlapping blocks".into());
        }
    }
    for required in [
        POS_NAMES,
        VOCABULARY,
        STRING_INDEX,
        QUANTIZATION,
        POS_BIGRAM,
        WORD_BIGRAM,
        TRIGRAM,
    ] {
        if !seen.contains(&required) {
            return Err(format!("required block {} is missing", format_guid(&required)).into());
        }
    }
    Ok((dict_uuid, build_uuid, blocks))
}

fn block_slice<'a>(bytes: &'a [u8], block: &BlockInfo) -> &'a [u8] {
    &bytes[block.offset..block.offset + block.size]
}

fn verify_item_count(blocks: &[BlockInfo], guid: &[u8; 16], actual: usize) -> Result<(), BoxError> {
    let block = blocks
        .iter()
        .find(|block| &block.guid == guid)
        .ok_or_else(|| format!("missing block {}", format_guid(guid)))?;
    if block.item_count != u64::try_from(actual)? {
        return Err(format!(
            "block {} item_count mismatch: header={}, decoded={actual}",
            format_guid(guid),
            block.item_count
        )
        .into());
    }
    Ok(())
}

fn required_block<'a>(
    blocks: &'a BTreeMap<[u8; 16], &'a [u8]>,
    guid: &[u8; 16],
) -> Result<&'a [u8], BoxError> {
    blocks
        .get(guid)
        .copied()
        .ok_or_else(|| format!("missing block {}", format_guid(guid)).into())
}

fn decode_pos_names(bytes: &[u8]) -> Result<Vec<String>, BoxError> {
    let mut input = Decoder::new(bytes);
    let count = usize::try_from(input.u32()?)?;
    let mut result = Vec::with_capacity(count);
    for _ in 0..count {
        result.push(input.string()?);
    }
    input.finish()?;
    if result.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err("POS_NAMES must be sorted and contain unique feature strings".into());
    }
    Ok(result)
}

#[derive(Clone, Debug)]
struct PendingWordRecord {
    word_id: u32,
    display_leaf: u32,
    pos_id: u32,
    unigram_code: u8,
    bigram_block_index: u32,
    backoff_alpha_code: u8,
    reading_leaf: u32,
    flags: u8,
}

#[derive(Default)]
struct ParsedKeyGroup {
    words: Vec<PendingWordRecord>,
    reading_word_ids: Option<Vec<u32>>,
    display_word_ids: Vec<u32>,
    phrases: Vec<Phrase>,
    cache_targets: Option<Vec<u32>>,
}

#[allow(clippy::too_many_lines)]
fn decode_vocabulary(
    bytes: &[u8],
    pos_names: &[String],
    unigram_codebook: &ProbabilityCodebook,
    string_index: &DoubleArray,
) -> Result<DecodedVocabulary, BoxError> {
    if bytes.first() != Some(&0) {
        return Err("vocabulary byte offset zero must remain reserved".into());
    }
    if bytes.len() < 16 || bytes.get(bytes.len() - 4..bytes.len()) != Some(MODEL_ONLY_FOOTER) {
        return Err("vocabulary model-only footer is missing".into());
    }
    let footer_start = bytes.len() - 8;
    let model_section_offset = usize::try_from(u32::from_le_bytes(
        bytes[footer_start..footer_start + 4].try_into()?,
    ))?;
    if model_section_offset < 1 || model_section_offset > footer_start {
        return Err("vocabulary model-only section offset is invalid".into());
    }
    let mut leaves = string_index.leaves()?;
    leaves.sort_by_key(|(_, offset)| *offset);
    if leaves.first().is_some_and(|(_, offset)| *offset != 1) {
        return Err("first vocabulary key group must begin at offset one".into());
    }
    if leaves.is_empty() && model_section_offset != 1 {
        return Err("vocabulary has data but the string index has no key groups".into());
    }

    let mut key_text_by_leaf = BTreeMap::<u32, String>::new();
    let mut leaf_by_key_text = BTreeMap::<String, u32>::new();
    let mut pending_words = Vec::<PendingWordRecord>::new();
    let mut reading_ids_by_leaf = BTreeMap::<u32, Vec<u32>>::new();
    let mut display_ids_by_leaf = BTreeMap::<u32, Vec<u32>>::new();
    let mut phrases = Vec::<Phrase>::new();
    let mut cache_targets_by_prefix = BTreeMap::<String, Vec<u32>>::new();
    let mut expected_group_offset = 1_usize;

    for (group_index, (leaf, raw_group_offset)) in leaves.iter().copied().enumerate() {
        let group_offset = usize::try_from(raw_group_offset)?;
        if group_offset != expected_group_offset {
            return Err("vocabulary key groups are not contiguous by byte offset".into());
        }
        let encoded_key = string_index.key_for_leaf(leaf, bytes)?;
        if string_index.lookup_leaf(&encoded_key, bytes) != Some(leaf) {
            return Err("double-array key does not resolve back to its terminal leaf".into());
        }
        let key = super::dict_char::decode(&encoded_key)?;
        if key.is_empty() {
            return Err("empty dictionary key is not supported".into());
        }
        if leaf_by_key_text.insert(key.clone(), leaf).is_some() {
            return Err("two double-array leaves decode to the same key".into());
        }
        key_text_by_leaf.insert(leaf, key.clone());

        let group_end = leaves
            .get(group_index + 1)
            .map(|(_, offset)| usize::try_from(*offset))
            .transpose()?
            .unwrap_or(model_section_offset);
        if group_end <= group_offset || group_end > model_section_offset {
            return Err("invalid vocabulary key-group byte range".into());
        }
        let parsed = decode_key_group(
            bytes,
            group_offset,
            group_end,
            leaf,
            &key,
            pos_names.len(),
            unigram_codebook,
        )?;
        pending_words.extend(parsed.words);
        if let Some(word_ids) = parsed.reading_word_ids {
            reading_ids_by_leaf.insert(leaf, word_ids);
        }
        if !parsed.display_word_ids.is_empty() {
            display_ids_by_leaf.insert(leaf, parsed.display_word_ids);
        }
        phrases.extend(parsed.phrases);
        if let Some(targets) = parsed.cache_targets {
            if cache_targets_by_prefix.insert(key, targets).is_some() {
                return Err("duplicate prediction cache key".into());
            }
        }
        expected_group_offset = group_end;
    }
    if expected_group_offset != model_section_offset {
        return Err("unindexed bytes remain before model-only vocabulary section".into());
    }
    pending_words.extend(decode_model_only_section(
        bytes,
        model_section_offset,
        footer_start,
        pos_names.len(),
        unigram_codebook,
    )?);
    let (words, reading_words) = assemble_vocabulary(
        pending_words,
        reading_ids_by_leaf,
        &display_ids_by_leaf,
        &key_text_by_leaf,
        pos_names,
        unigram_codebook,
    )?;
    let words_by_id = words
        .iter()
        .map(|word| (word.word_id, word))
        .collect::<BTreeMap<_, _>>();
    let mut phrase_by_offset = BTreeMap::<u32, &Phrase>::new();
    for phrase in &phrases {
        let mut reading = String::new();
        for id in &phrase.ids {
            let word = words_by_id
                .get(id)
                .ok_or("phrase points outside the vocabulary word records")?;
            if word.reading.is_empty() {
                return Err("phrase contains a word without an indexable reading".into());
            }
            reading.push_str(&word.reading);
        }
        if reading != phrase.reading {
            return Err("phrase reading differs from its word-ID sequence".into());
        }
        if phrase_by_offset
            .insert(phrase.vocab_offset, phrase)
            .is_some()
        {
            return Err("duplicate phrase record offset".into());
        }
    }
    let mut prefixes = BTreeMap::<String, Vec<PredictionTarget>>::new();
    for (prefix, targets) in cache_targets_by_prefix {
        let prefix_len = prefix.chars().count();
        if !(1..=3).contains(&prefix_len) || !prefix.chars().all(is_hiragana) {
            return Err("prediction cache key must be one to three hiragana characters".into());
        }
        let mut resolved = Vec::with_capacity(targets.len());
        for target in targets {
            resolved.push(resolve_prediction_target(
                target,
                &prefix,
                &words_by_id,
                &phrase_by_offset,
            )?);
        }
        if resolved.is_empty() || resolved.len() > 5 {
            return Err("prediction cache must contain between one and five targets".into());
        }
        if prefixes.insert(prefix, resolved).is_some() {
            return Err("duplicate prediction cache prefix".into());
        }
    }
    Ok(DecodedVocabulary {
        words,
        reading_words,
        phrases,
        prefixes,
    })
}

fn resolve_prediction_target(
    target_offset: u32,
    prefix: &str,
    words: &BTreeMap<u32, &Word>,
    phrases: &BTreeMap<u32, &Phrase>,
) -> Result<PredictionTarget, BoxError> {
    let (target, reading) = if let Some(word) = words.get(&target_offset) {
        if word.flags & 2 != 0 {
            return Err("prediction cache cannot point to a model-only word".into());
        }
        (PredictionTarget::Word(target_offset), word.reading.as_str())
    } else if let Some(phrase) = phrases.get(&target_offset) {
        (
            PredictionTarget::Phrase(target_offset),
            phrase.reading.as_str(),
        )
    } else {
        return Err("prediction cache target is neither a word nor phrase record".into());
    };
    if !reading.starts_with(prefix) {
        return Err("prediction cache target reading does not match its key".into());
    }
    Ok(target)
}

#[allow(clippy::too_many_lines)]
fn decode_key_group(
    vocabulary: &[u8],
    group_offset: usize,
    group_end: usize,
    leaf: u32,
    key: &str,
    pos_count: usize,
    unigram_codebook: &ProbabilityCodebook,
) -> Result<ParsedKeyGroup, BoxError> {
    let group_bytes = &vocabulary[group_offset..group_end];
    let mut input = Decoder::new(group_bytes);
    if group_bytes.first().is_some_and(|tag| tag >> 5 == 1) {
        let tag = input.u8()?;
        let tail_len = usize::from(tag & 0x1f) + 1;
        let encoded_tail = input.take(tail_len)?;
        if tail_at(vocabulary, group_offset)? != encoded_tail {
            return Err("inline tail record disagrees with the string-index leaf".into());
        }
    }

    let mut parsed = ParsedKeyGroup::default();
    let mut last_record_rank = 0_u8;
    let mut saw_cache_header = false;
    let (has_reading, reading_leaf) = loop {
        let record_offset = group_offset
            .checked_add(input.offset)
            .ok_or("vocabulary record offset overflow")?;
        let tag = input.u8()?;
        match tag {
            PREFIX_CACHE_HEADER_TAG => {
                if last_record_rank > 1 || saw_cache_header {
                    return Err("prediction cache header is duplicated or out of order".into());
                }
                let prefix_len = key.chars().count();
                if !(1..=3).contains(&prefix_len) || !key.chars().all(is_hiragana) {
                    return Err("prediction cache header appears on a non-prefix key".into());
                }
                last_record_rank = 1;
                saw_cache_header = true;
                let count = usize::from(input.u8()?);
                if !(1..=5).contains(&count) {
                    return Err("prediction cache must contain between one and five targets".into());
                }
                let mut targets = Vec::with_capacity(count);
                for _ in 0..count {
                    targets.push(input.u32()?);
                }
                parsed.cache_targets = Some(targets);
            }
            _ if tag >> 5 == PREDICTION_PHRASE_TAG_BASE >> 5 => {
                if last_record_rank > 2 {
                    return Err("prediction phrase record is out of order".into());
                }
                last_record_rank = 2;
                let word_count = usize::from(tag & 0x1f) + 1;
                if !(2..=6).contains(&word_count) {
                    return Err("prediction phrase must contain between two and six words".into());
                }
                let mut ids = Vec::with_capacity(word_count);
                for _ in 0..word_count {
                    ids.push(input.u32()?);
                }
                let unigram = input.f64()?;
                if !unigram.is_finite() || !(0.0..=1.0).contains(&unigram) || unigram == 0.0 {
                    return Err("invalid prediction phrase unigram probability".into());
                }
                parsed.phrases.push(Phrase {
                    ids,
                    reading: key.to_owned(),
                    unigram,
                    vocab_offset: u32::try_from(record_offset)?,
                });
            }
            WORD_RECORD_TAG => {
                if last_record_rank > 3 {
                    return Err("word record is out of order".into());
                }
                last_record_rank = 3;
                let display_leaf = input.u32()?;
                let pos_id = input.u32()?;
                let unigram_code = input.u8()?;
                let bigram_block_index = input.u24()?;
                let backoff_alpha_code = input.u8()?;
                if pos_id == 0 || usize::try_from(pos_id)? > pos_count {
                    return Err("vocabulary POS ID is out of range".into());
                }
                if bigram_block_index == 0 && backoff_alpha_code != 0 {
                    return Err("word without a bigram block has a nonzero backoff code".into());
                }
                if unigram_code != 0 {
                    unigram_codebook.decode(unigram_code)?;
                }
                let word_id = u32::try_from(record_offset)?;
                parsed.words.push(PendingWordRecord {
                    word_id,
                    display_leaf,
                    pos_id,
                    unigram_code,
                    bigram_block_index,
                    backoff_alpha_code,
                    reading_leaf: leaf,
                    flags: 0,
                });
                parsed
                    .reading_word_ids
                    .get_or_insert_with(Vec::new)
                    .push(word_id);
            }
            _ if tag >> 5 == DISPLAY_REFERENCE_TAG_BASE >> 5 => {
                if last_record_rank > 4 {
                    return Err("display reference record is out of order".into());
                }
                last_record_rank = 4;
                let count = usize::from(tag & 0x1f) + 1;
                for _ in 0..count {
                    parsed.display_word_ids.push(input.u32()?);
                }
            }
            0xe0 | 0xe1 => {
                let has_reading = tag & 1 != 0;
                let reading_leaf = if has_reading {
                    Some(input.u32()?)
                } else {
                    None
                };
                input.finish()?;
                break (has_reading, reading_leaf);
            }
            _ if tag >> 5 == 1 => {
                return Err("inline tail record may only appear at group start".into());
            }
            _ => return Err(format!("unknown vocabulary record tag 0x{tag:02x}").into()),
        }
    };
    if has_reading {
        if reading_leaf != Some(leaf) || (parsed.words.is_empty() && parsed.phrases.is_empty()) {
            return Err(
                "reading key group must terminate with its own leaf and reading records".into(),
            );
        }
    } else if !parsed.words.is_empty() || !parsed.phrases.is_empty() {
        return Err("reading records exist in a group without a reading leaf".into());
    } else {
        parsed.reading_word_ids = None;
    }
    if let Some(first_word) = parsed.words.first() {
        if reading_leaf_from_word_offset(vocabulary, first_word.word_id)? != leaf {
            return Err("word body does not lead to its key-group reading terminator".into());
        }
    }
    Ok(parsed)
}

fn reading_leaf_from_word_offset(vocabulary: &[u8], word_offset: u32) -> Result<u32, BoxError> {
    let start = usize::try_from(word_offset)?;
    let mut input = Decoder::new(
        vocabulary
            .get(start..)
            .ok_or("word ID is outside the vocabulary block")?,
    );
    loop {
        let tag = input.u8()?;
        match tag {
            WORD_RECORD_TAG => {
                input.take(13)?;
            }
            _ if tag >> 5 == DISPLAY_REFERENCE_TAG_BASE >> 5 => {
                let count = usize::from(tag & 0x1f) + 1;
                input.take(
                    count
                        .checked_mul(4)
                        .ok_or("display reference size overflow")?,
                )?;
            }
            0xe1 => return input.u32(),
            0xe0 => return Err("word body reaches a non-reading key-group terminator".into()),
            _ => {
                return Err(format!(
                    "unexpected record 0x{tag:02x} between word body and reading terminator"
                )
                .into());
            }
        }
    }
}

fn decode_model_only_section(
    vocabulary: &[u8],
    section_offset: usize,
    footer_start: usize,
    pos_count: usize,
    unigram_codebook: &ProbabilityCodebook,
) -> Result<Vec<PendingWordRecord>, BoxError> {
    let section = vocabulary
        .get(section_offset..footer_start)
        .ok_or("model-only vocabulary section is outside vocabulary block")?;
    let mut input = Decoder::new(section);
    if input.take(4)? != MODEL_ONLY_SECTION {
        return Err("model-only vocabulary section marker is invalid".into());
    }
    let count = usize::try_from(input.u32()?)?;
    let mut words = Vec::with_capacity(count);
    for _ in 0..count {
        let word_id = u32::try_from(section_offset + input.offset)?;
        if input.u8()? != 0x80 {
            return Err("invalid model-only vocabulary record tag".into());
        }
        let pos_id = input.u32()?;
        if pos_id == 0 || usize::try_from(pos_id)? > pos_count {
            return Err("model-only vocabulary POS ID is out of range".into());
        }
        let unigram_code = input.u8()?;
        let bigram_block_index = input.u24()?;
        let backoff_alpha_code = input.u8()?;
        if unigram_code == 0 {
            return Err("model-only vocabulary word must have a unigram code".into());
        }
        if bigram_block_index == 0 && backoff_alpha_code != 0 {
            return Err("model-only word without a bigram block has a nonzero backoff code".into());
        }
        unigram_codebook.decode(unigram_code)?;
        words.push(PendingWordRecord {
            word_id,
            display_leaf: u32::MAX,
            pos_id,
            unigram_code,
            bigram_block_index,
            backoff_alpha_code,
            reading_leaf: u32::MAX,
            flags: 2,
        });
    }
    input.finish()?;
    Ok(words)
}

fn assemble_vocabulary(
    pending_words: Vec<PendingWordRecord>,
    reading_ids_by_leaf: BTreeMap<u32, Vec<u32>>,
    display_ids_by_leaf: &BTreeMap<u32, Vec<u32>>,
    key_text_by_leaf: &BTreeMap<u32, String>,
    pos_names: &[String],
    unigram_codebook: &ProbabilityCodebook,
) -> Result<(Vec<Word>, ReadingWordIndex), BoxError> {
    let mut words = BTreeMap::<u32, Word>::new();
    for record in pending_words {
        let pos = record
            .pos_id
            .checked_sub(1)
            .and_then(|id| pos_names.get(id as usize))
            .ok_or("vocabulary POS ID is out of range")?
            .clone();
        let probability = unigram_codebook.decode(record.unigram_code)?;
        let word = Word {
            source_id: usize::MAX,
            word_id: record.word_id,
            surface: String::new(),
            reading: String::new(),
            display_leaf: record.display_leaf,
            reading_leaf: record.reading_leaf,
            pos,
            pos_id: record.pos_id,
            bigram_block_index: record.bigram_block_index,
            backoff_alpha_code: record.backoff_alpha_code,
            unigram_count: 0,
            word_unigram_in_pos_raw: probability,
            word_unigram_in_pos: probability,
            word_unigram_in_pos_code: record.unigram_code,
            flags: record.flags,
        };
        if words.insert(record.word_id, word).is_some() {
            return Err("duplicate vocabulary word record offset".into());
        }
    }

    validate_display_references(&words, display_ids_by_leaf)?;
    for word in words.values_mut() {
        if word.flags & 2 != 0 {
            continue;
        }
        word.surface = key_text_by_leaf
            .get(&word.display_leaf)
            .ok_or("word display leaf is not present in the string index")?
            .clone();
        word.reading = key_text_by_leaf
            .get(&word.reading_leaf)
            .ok_or("word reading leaf is not present in the string index")?
            .clone();
    }

    let mut reading_words = BTreeMap::new();
    for (leaf, ids) in reading_ids_by_leaf {
        let reading = key_text_by_leaf
            .get(&leaf)
            .ok_or("reading index leaf is not present in the string index")?
            .clone();
        for id in &ids {
            if words
                .get(id)
                .is_none_or(|word| word.reading_leaf != leaf || word.reading != reading)
            {
                return Err("reading group contains a word with a different reading".into());
            }
        }
        if reading_words.insert(reading, ids).is_some() {
            return Err("duplicate reading key".into());
        }
    }
    Ok((words.into_values().collect(), reading_words))
}

fn validate_display_references(
    words: &BTreeMap<u32, Word>,
    display_ids_by_leaf: &BTreeMap<u32, Vec<u32>>,
) -> Result<(), BoxError> {
    let mut referenced_words = BTreeSet::<u32>::new();
    for (display_leaf, word_ids) in display_ids_by_leaf {
        if word_ids.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err("display index repeats a word ID".into());
        }
        for word_id in word_ids {
            let word = words
                .get(word_id)
                .ok_or("display index points outside the vocabulary word records")?;
            if word.flags & 2 != 0 || word.display_leaf != *display_leaf {
                return Err("display index points to a word with a different surface leaf".into());
            }
            if !referenced_words.insert(*word_id) {
                return Err("word record appears more than once in display indexes".into());
            }
        }
    }
    let expected_indexed_word_count = words.values().filter(|word| word.flags & 2 == 0).count();
    if referenced_words.len() != expected_indexed_word_count {
        return Err("one or more words are missing from display indexes".into());
    }
    Ok(())
}

fn write_pos_dump(output_dir: &Path, pos_names: &[String]) -> Result<(), BoxError> {
    let mut out = BufWriter::new(File::create_new(output_dir.join("pos.tsv"))?);
    writeln!(out, "pos_id\tmecab_feature")?;
    for (index, name) in pos_names.iter().enumerate() {
        writeln!(out, "{}\t{}", index + 1, escape_tsv(name))?;
    }
    out.flush()?;
    Ok(())
}

fn write_vocab_dump(
    output_dir: &Path,
    words: &[Word],
    pos_names: &[String],
) -> Result<(), BoxError> {
    let mut out = BufWriter::new(File::create_new(output_dir.join("vocabulary.tsv"))?);
    writeln!(
        out,
        "word_id\tsurface\treading\tpos_id\tmecab_feature\tP_word_given_POS\tcode\tflags"
    )?;
    for word in words {
        let pos = if word.flags & 1 != 0 {
            String::new()
        } else {
            pos_names
                .get(word.pos_id.checked_sub(1).ok_or("POS ID 0 is reserved")? as usize)
                .ok_or("word POS ID out of range while dumping")?
                .clone()
        };
        writeln!(
            out,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            word.word_id,
            escape_tsv(&word.surface),
            escape_tsv(&word.reading),
            if word.flags & 1 == 0 {
                word.pos_id.to_string()
            } else {
                String::new()
            },
            escape_tsv(&pos),
            word.word_unigram_in_pos,
            word.word_unigram_in_pos_code,
            word.flags
        )?;
    }
    out.flush()?;
    Ok(())
}

fn write_pos_bigram_dump(
    output_dir: &Path,
    entries: &[PosBigramEntry],
    pos_names: &[String],
) -> Result<(), BoxError> {
    let mut out = BufWriter::new(File::create_new(output_dir.join("pos-bigram.tsv"))?);
    writeln!(
        out,
        "source_pos_id\tsource_mecab_feature\ttarget_pos_id\ttarget_mecab_feature\tprobability\tcode"
    )?;
    for entry in entries {
        writeln!(
            out,
            "{}\t{}\t{}\t{}\t{:.12e}\t{}",
            entry.source_pos,
            pos_names
                .get(
                    entry
                        .source_pos
                        .checked_sub(1)
                        .ok_or("POS ID 0 is reserved")? as usize,
                )
                .ok_or("source POS out of range while dumping bigram")?,
            entry.target_pos,
            pos_names
                .get(
                    entry
                        .target_pos
                        .checked_sub(1)
                        .ok_or("POS ID 0 is reserved")? as usize,
                )
                .ok_or("target POS out of range while dumping bigram")?,
            entry.probability,
            entry.code
        )?;
    }
    out.flush()?;
    Ok(())
}

fn write_word_bigram_dump(
    output_dir: &Path,
    contexts: &[WordBigramContext],
    entries: &[WordBigramEntry],
    words: &BTreeMap<u32, Word>,
) -> Result<(), BoxError> {
    let mut bigrams = BufWriter::new(File::create_new(output_dir.join("word-bigram.tsv"))?);
    writeln!(
        bigrams,
        "source_word_id\ttarget_word_id\tsource_surface\ttarget_surface\tprobability\tcode"
    )?;
    for entry in entries {
        let source = words
            .get(&entry.source_word)
            .ok_or("word bigram source missing while dumping")?;
        let target = words
            .get(&entry.target_word)
            .ok_or("word bigram target missing while dumping")?;
        writeln!(
            bigrams,
            "{}\t{}\t{}\t{}\t{:.12e}\t{}",
            entry.source_word,
            entry.target_word,
            source.surface,
            target.surface,
            entry.probability,
            entry.code
        )?;
    }
    bigrams.flush()?;
    let mut backoffs = BufWriter::new(File::create_new(
        output_dir.join("word-bigram-backoff.tsv"),
    )?);
    writeln!(backoffs, "source_word_id\tsource_surface\talpha\tcode")?;
    for context in contexts {
        let source = words
            .get(&context.source_word)
            .ok_or("backoff source missing while dumping")?;
        writeln!(
            backoffs,
            "{}\t{}\t{:.12e}\t{}",
            context.source_word, source.surface, context.alpha, context.alpha_code
        )?;
    }
    backoffs.flush()?;
    Ok(())
}

fn write_trigram_dump(
    output_dir: &Path,
    entries: &[DecodedTrigram],
    words: &BTreeMap<u32, Word>,
) -> Result<(), BoxError> {
    let mut out = BufWriter::new(File::create_new(output_dir.join("trigram.tsv"))?);
    writeln!(
        out,
        "source_word_id\tmiddle_word_id\ttarget_word_id\tsource_surface\tmiddle_surface\ttarget_surface\tprobability\tcode\tparent_bigram_edge\tB_row_slot\tencoding"
    )?;
    for entry in entries {
        let source = words
            .get(&entry.source_word)
            .ok_or("trigram source missing while dumping")?;
        let middle = words
            .get(&entry.middle_word)
            .ok_or("trigram middle word missing while dumping")?;
        let target = words
            .get(&entry.target_word)
            .ok_or("trigram target missing while dumping")?;
        writeln!(
            out,
            "{}\t{}\t{}\t{}\t{}\t{}\t{:.12e}\t{}\t{}\t{}\t{}",
            entry.source_word,
            entry.middle_word,
            entry.target_word,
            source.surface,
            middle.surface,
            target.surface,
            entry.probability,
            entry.code,
            entry.parent_edge,
            entry.slot,
            if entry.codec == 0 { "dense" } else { "sparse" }
        )?;
    }
    out.flush()?;
    Ok(())
}

fn write_reading_dump(
    output_dir: &Path,
    index: &BTreeMap<String, Vec<u32>>,
) -> Result<(), BoxError> {
    let mut out = BufWriter::new(File::create_new(output_dir.join("reading-word-index.tsv"))?);
    writeln!(out, "reading\tword_ids")?;
    for (reading, ids) in index {
        writeln!(out, "{}\t{}", escape_tsv(reading), join_ids(ids))?;
    }
    out.flush()?;
    Ok(())
}

fn write_phrase_dump(
    output_dir: &Path,
    phrases: &[Phrase],
    words: &BTreeMap<u32, Word>,
) -> Result<(), BoxError> {
    let mut out = BufWriter::new(File::create_new(output_dir.join("prediction-phrases.tsv"))?);
    writeln!(out, "phrase_id\treading\tword_ids\tdisplay\tunigram")?;
    for phrase in phrases {
        let display = phrase
            .ids
            .iter()
            .map(|word_id| {
                words
                    .get(word_id)
                    .map(|word| word.surface.as_str())
                    .ok_or("phrase dump references a missing word")
            })
            .collect::<Result<Vec<_>, _>>()?
            .join("/");
        writeln!(
            out,
            "{}\t{}\t{}\t{}\t{:.12e}",
            phrase.vocab_offset,
            escape_tsv(&phrase.reading),
            join_ids(&phrase.ids),
            escape_tsv(&display),
            phrase.unigram
        )?;
    }
    out.flush()?;
    Ok(())
}

fn write_prefix_dump(
    output_dir: &Path,
    prefixes: &ResolvedPrefixIndex,
    phrases: &[Phrase],
    words: &BTreeMap<u32, Word>,
) -> Result<(), BoxError> {
    let mut out = BufWriter::new(File::create_new(output_dir.join("prediction-prefix.tsv"))?);
    writeln!(
        out,
        "prefix\trank\ttarget_type\ttarget_id\tword_ids\tdisplay\treading\tstored_probability_type\tstored_probability"
    )?;
    let phrases_by_offset = phrases
        .iter()
        .map(|phrase| (phrase.vocab_offset, phrase))
        .collect::<BTreeMap<_, _>>();
    for (prefix, entries) in prefixes {
        for (rank, target) in entries.iter().enumerate() {
            let (target_type, target_id, ids, reading, probability_type, probability, display) =
                match target {
                    PredictionTarget::Word(id) => {
                        let word = words
                            .get(id)
                            .ok_or("prefix cache word target is missing while dumping")?;
                        (
                            "word",
                            *id,
                            vec![*id],
                            word.reading.as_str(),
                            "word_unigram_in_pos",
                            word.word_unigram_in_pos,
                            word.surface.clone(),
                        )
                    }
                    PredictionTarget::Phrase(id) => {
                        let phrase = phrases_by_offset
                            .get(id)
                            .ok_or("prefix cache phrase target is missing while dumping")?;
                        let display = phrase
                            .ids
                            .iter()
                            .map(|word_id| {
                                words
                                    .get(word_id)
                                    .map(|word| word.surface.as_str())
                                    .ok_or("prefix dump references a missing word")
                            })
                            .collect::<Result<Vec<_>, _>>()?
                            .join("/");
                        (
                            "phrase",
                            *id,
                            phrase.ids.clone(),
                            phrase.reading.as_str(),
                            "phrase_unigram",
                            phrase.unigram,
                            display,
                        )
                    }
                };
            writeln!(
                out,
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.12e}",
                escape_tsv(prefix),
                rank + 1,
                target_type,
                target_id,
                join_ids(&ids),
                escape_tsv(&display),
                escape_tsv(reading),
                probability_type,
                probability
            )?;
        }
    }
    out.flush()?;
    Ok(())
}

fn join_ids(ids: &[u32]) -> String {
    ids.iter().map(u32::to_string).collect::<Vec<_>>().join(" ")
}

fn escape_tsv(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('\t', "\\t")
        .replace('\r', "\\r")
        .replace('\n', "\\n")
}

fn put_string(out: &mut Vec<u8>, value: &str) -> Result<(), BoxError> {
    put_u32(out, u32::try_from(value.len())?);
    out.extend_from_slice(value.as_bytes());
    Ok(())
}

fn put_u16(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_u24(out: &mut Vec<u8>, value: u32) -> Result<(), BoxError> {
    if value > 0x00ff_ffff {
        return Err("word ID exceeds the 24-bit vocabulary-offset range".into());
    }
    out.extend_from_slice(&value.to_le_bytes()[..3]);
    Ok(())
}

fn put_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_f64(out: &mut Vec<u8>, value: f64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn align8(value: usize) -> usize {
    (value + 7) & !7
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

fn make_uuid(seed: u128) -> [u8; 16] {
    let mut bytes = seed.to_le_bytes();
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    bytes
}

fn format_guid(guid: &[u8; 16]) -> String {
    format!(
        "{}-{}-{}-{}-{}",
        hex(&guid[0..4]),
        hex(&guid[4..6]),
        hex(&guid[6..8]),
        hex(&guid[8..10]),
        hex(&guid[10..16])
    )
}

fn hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut result = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        result.push(char::from(HEX[usize::from(byte >> 4)]));
        result.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    result
}

struct Decoder<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Decoder<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8], BoxError> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or("decode offset overflow")?;
        let slice = self.bytes.get(self.offset..end).ok_or("truncated block")?;
        self.offset = end;
        Ok(slice)
    }

    fn array16(&mut self) -> Result<[u8; 16], BoxError> {
        Ok(self.take(16)?.try_into()?)
    }

    fn u8(&mut self) -> Result<u8, BoxError> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, BoxError> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into()?))
    }

    fn u24(&mut self) -> Result<u32, BoxError> {
        let bytes = self.take(3)?;
        Ok(u32::from(bytes[0]) | (u32::from(bytes[1]) << 8) | (u32::from(bytes[2]) << 16))
    }

    fn u32(&mut self) -> Result<u32, BoxError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into()?))
    }

    fn u64(&mut self) -> Result<u64, BoxError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into()?))
    }

    fn f64(&mut self) -> Result<f64, BoxError> {
        Ok(f64::from_le_bytes(self.take(8)?.try_into()?))
    }

    fn string(&mut self) -> Result<String, BoxError> {
        let len = usize::try_from(self.u32()?)?;
        Ok(std::str::from_utf8(self.take(len)?)?.to_owned())
    }

    fn finish(&self) -> Result<(), BoxError> {
        if self.offset == self.bytes.len() {
            Ok(())
        } else {
            Err("unexpected trailing bytes in block".into())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_ID: AtomicU64 = AtomicU64::new(0);

    fn temp_dir() -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "ai-engine-sdict-test-{}-{}",
            std::process::id(),
            TEST_ID.fetch_add(1, Ordering::Relaxed)
        ))
    }

    #[test]
    fn feature_maps_to_dictionary_local_pos_and_hiragana_reading() {
        let word = decode_token("庭,名詞,一般,*,*,*,*,庭,ニワ,ニワ", 3);
        assert_eq!(word.0, "庭");
        assert_eq!(word.1, "にわ");
        assert_eq!(word.2, "名詞,一般,*,*,*,*,*");
    }

    #[test]
    fn particles_group_selected_subcategories_by_initial_mora() {
        let ga = decode_token("が,助詞,格助詞,一般,*,*,*,が,ガ,ガ", 3);
        let wo = decode_token("を,助詞,格助詞,一般,*,*,*,を,ヲ,ヲ", 4);
        assert_eq!(ga.2, "助詞,格助詞,一般,*,*,*,が");
        assert_eq!(wo.2, "助詞,格助詞,一般,*,*,*,を");
        assert_ne!(ga.2, wo.2);

        let ni = decode_token("に,助詞,格助詞,一般,*,*,*,に,ニ,ニ", 5);
        let ni_yotte = decode_token("によって,助詞,格助詞,連語,*,*,*,よる,ニヨッテ,ニヨッテ", 6);
        let ni_yori = decode_token("により,助詞,格助詞,連語,*,*,*,よる,ニヨリ,ニヨリ", 7);
        assert_eq!(ni.2, "助詞,格助詞,一般,*,*,*,に");
        assert_eq!(ni_yotte.2, "助詞,格助詞,連語,*,*,*,に");
        assert_eq!(ni_yori.2, ni_yotte.2);
        assert_ne!(ni_yotte.2, ni.2);

        let to = decode_token("と,助詞,格助詞,一般,*,*,*,と,ト,ト", 8);
        let toshite = decode_token("として,助詞,格助詞,連語,*,*,*,とする,トシテ,トシテ", 9);
        assert_eq!(toshite.2, "助詞,格助詞,連語,*,*,*,と");
        assert_ne!(toshite.2, to.2);

        let ha = decode_token("は,助詞,係助詞,*,*,*,*,は,ハ,ワ", 10);
        let mo = decode_token("も,助詞,係助詞,*,*,*,*,も,モ,モ", 11);
        assert_ne!(ha.2, mo.2);

        let ne = decode_token("ね,助詞,終助詞,*,*,*,*,ね,ネ,ネ", 12);
        let nee = decode_token("ねぇ,助詞,終助詞,*,*,*,*,ねえ,ネェ,ネェ", 13);
        let nee_long = decode_token("ねー,助詞,終助詞,*,*,*,*,ねえ,ネー,ネー", 14);
        assert_eq!(ne.2, "助詞,終助詞,*,*,*,*,ね");
        assert_eq!(nee.2, ne.2);
        assert_eq!(nee_long.2, ne.2);

        let kana = decode_token("かな,助詞,終助詞,*,*,*,*,かな,カナ,カナ", 15);
        assert_ne!(ne.2, kana.2);

        let bakari = decode_token("ばかり,助詞,副助詞,*,*,*,*,ばかり,バカリ,バカリ", 18);
        let bakkari = decode_token(
            "ばっかり,助詞,副助詞,*,*,*,*,ばっかり,バッカリ,バッカリ",
            19,
        );
        let hodo = decode_token("ほど,助詞,副助詞,*,*,*,*,ほど,ホド,ホド", 20);
        let kanji_hodo = decode_token("程,助詞,副助詞,*,*,*,*,程,ホド,ホド", 21);
        let made = decode_token("まで,助詞,副助詞,*,*,*,*,まで,マデ,マデ", 22);
        assert_eq!(bakari.2, "助詞,副助詞,*,*,*,*,ば");
        assert_eq!(bakkari.2, bakari.2);
        assert_eq!(hodo.2, kanji_hodo.2);
        assert_ne!(hodo.2, made.2);

        // A mixed MeCab subcategory is not folded into pure 副助詞.
        let mixed = decode_token("か,助詞,副助詞／並立助詞／終助詞,*,*,*,*,か,カ,カ", 23);
        assert_eq!(mixed.2, "助詞,副助詞／並立助詞／終助詞,*,*,*,*,か");

        // Voiced/non-voiced surfaces used to share this complete grammar key.
        let ta = decode_token("た,助動詞,*,*,*,特殊・タ,基本形,た,タ,タ", 16);
        let da = decode_token("だ,助動詞,*,*,*,特殊・タ,基本形,た,ダ,ダ", 17);
        assert_eq!(ta.2, "助動詞,*,*,*,特殊・タ,基本形,た");
        assert_eq!(da.2, "助動詞,*,*,*,特殊・タ,基本形,だ");
        assert_ne!(ta.2, da.2);
    }

    #[test]
    fn lexicalized_model_pos_preserves_grammar_and_is_idempotent() {
        let case_particle = model_pos_for_word("が", "助詞,格助詞,一般,*,*,*,*");
        let conjunctive_particle = model_pos_for_word("が", "助詞,接続助詞,*,*,*,*,*");
        assert_ne!(case_particle, conjunctive_particle);
        assert_eq!(case_particle.split(',').count(), 7);
        assert_eq!(model_pos_for_word("が", &case_particle), case_particle);
        assert_eq!(
            model_pos_for_word("庭", "名詞,一般,*,*,*,*,庭,ニワ,ニワ"),
            "名詞,一般,*,*,*,*,*"
        );
    }

    #[test]
    fn auxiliary_orthographic_variants_share_pos_by_reading() {
        let hiragana_gozaru = decode_token(
            "ござる,助動詞,*,*,*,五段・ラ行特殊,基本形,ござる,ゴザル,ゴザル",
            3,
        );
        let kanji_gozaru = decode_token(
            "御座る,助動詞,*,*,*,五段・ラ行特殊,基本形,ござる,ゴザル,ゴザル",
            4,
        );
        assert_eq!(
            hiragana_gozaru.2,
            "助動詞,*,*,*,五段・ラ行特殊,基本形,ござる"
        );
        assert_eq!(kanji_gozaru.2, hiragana_gozaru.2);

        let hiragana_gotoku = decode_token(
            "ごとく,助動詞,*,*,*,文語・ゴトシ,連用形,ごとし,ゴトク,ゴトク",
            5,
        );
        let kanji_gotoku = decode_token(
            "如く,助動詞,*,*,*,文語・ゴトシ,連用形,ごとし,ゴトク,ゴトク",
            6,
        );
        assert_eq!(hiragana_gotoku.2, "助動詞,*,*,*,文語・ゴトシ,連用形,ごとく");
        assert_eq!(kanji_gotoku.2, hiragana_gotoku.2);

        let ta = decode_token("た,助動詞,*,*,*,特殊・タ,基本形,た,タ,タ", 7);
        let da = decode_token("だ,助動詞,*,*,*,特殊・タ,基本形,た,ダ,ダ", 8);
        assert_ne!(ta.2, da.2, "different readings remain separate POS values");
    }

    #[test]
    fn dictionary_sources_and_corpus_use_the_same_lexicalized_model_pos() {
        let mut seeds = BTreeMap::new();
        for (id, csv) in [
            (3, "が,148,148,100,助詞,格助詞,一般,*,*,*,が,ガ,ガ"),
            (4, "を,148,148,100,助詞,格助詞,一般,*,*,*,を,ヲ,ヲ"),
            (5, "だ,473,473,100,助動詞,*,*,*,特殊・タ,基本形,た,ダ,ダ"),
        ] {
            let fields = parse_csv_row(csv).expect("MeCab source row");
            add_mecab_source_row(&mut seeds, &fields).expect("source word");
            let corpus_token = format!("{},{}", fields[0], fields[4..].join(","));
            let (surface, reading, pos, _) = decode_token(&corpus_token, id);
            assert!(seeds.contains_key(&WordKey::new(surface, reading, pos)));
        }
        assert_eq!(seeds.len(), 3);
    }

    #[test]
    fn single_kanji_source_preserves_explicit_pos_surface_and_multiple_readings() {
        let mut seeds = BTreeMap::new();
        for (surface, reading, expected_reading) in [
            ("漢", "カン", "かん"),
            ("﨑", "サキ", "さき"),
            ("𠮷", "キチ", "きち"),
            ("𠮷", "キツ", "きつ"),
            ("〇", "レイ", "れい"),
        ] {
            let csv =
                format!("{surface},5,5,1818,記号,単漢字,*,*,*,*,{surface},{reading},{reading}");
            let fields = parse_csv_row(&csv).expect("single-kanji source row");
            add_mecab_source_row(&mut seeds, &fields).expect("single-kanji source word");
            assert!(seeds.contains_key(&WordKey::new(
                surface.to_owned(),
                expected_reading.to_owned(),
                "記号,単漢字,*,*,*,*,*".to_owned(),
            )));
        }
        assert_eq!(seeds.len(), 5);
        let invalid = parse_csv_row("漢字,5,5,1818,記号,単漢字,*,*,*,*,漢字,カンジ,カンジ")
            .expect("multi-character source row");
        assert!(add_mecab_source_row(&mut seeds, &invalid).is_err());

        // Ordinary sources still use the existing English/numeric normalization.
        for csv in [
            "A,5,5,1818,名詞,一般,*,*,*,*,A,エー,エー",
            "123,5,5,1818,名詞,数,*,*,*,*,123,ヒャクニジュウサン,ヒャクニジュウサン",
        ] {
            add_mecab_source_row(&mut seeds, &parse_csv_row(csv).expect("ordinary row"))
                .expect("ordinary source word");
        }
        assert!(seeds.contains_key(&WordKey::model_only("英単語")));
        assert!(seeds.contains_key(&WordKey::model_only("名詞,数,*,*,*,*,*")));
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn functional_word_pos_keeps_unigram_and_transition_counts_separate() {
        let dir = temp_dir();
        let system = dir.join("system");
        let additions = dir.join("additions");
        fs::create_dir_all(&system).expect("system fixture directory");
        fs::create_dir_all(&additions).expect("addition fixture directory");
        fs::write(system.join("dicrc"), "config-charset = UTF-8\n")
            .expect("system fixture charset");
        fs::write(
            additions.join("20261003-test-new-words.txt"),
            "が,148,148,100,助詞,格助詞,一般,*,*,*,が,ガ,ガ\n",
        )
        .expect("user addition fixture");
        fs::write(
            system.join("functional.csv"),
            "が,148,148,100,助詞,格助詞,一般,*,*,*,が,ガ,ガ\nを,148,148,100,助詞,格助詞,一般,*,*,*,を,ヲ,ヲ\nた,473,473,100,助動詞,*,*,*,特殊・タ,基本形,た,タ,タ\nだ,473,473,100,助動詞,*,*,*,特殊・タ,基本形,た,ダ,ダ\n",
        )
        .expect("functional word sources");
        let vocab = dir.join("vocab.tsv");
        fs::write(
            &vocab,
            "0\t[NULL]\n1\t[BOS]\n2\t[EOS]\n3\tが,助詞,格助詞,一般,*,*,*,が,ガ,ガ\n4\tを,助詞,格助詞,一般,*,*,*,を,ヲ,ヲ\n5\tた,助動詞,*,*,*,特殊・タ,基本形,た,タ,タ\n6\tだ,助動詞,*,*,*,特殊・タ,基本形,た,ダ,ダ\n",
        )
        .expect("corpus vocabulary");
        let unigram = dir.join("1gram.by-id.tsv");
        fs::write(
            &unigram,
            "id\tcount\n1\t20\n2\t20\n3\t12\n4\t8\n5\t12\n6\t8\n",
        )
        .expect("corpus unigram counts");
        let (mut words, pos_names, _, corpus_ids) =
            compile_word_table_from_sources(&vocab, &unigram, Some(&system), Some(&additions))
                .expect("compile functional word table");
        assert_eq!(words.len(), 6, "source/corpus records are not duplicated");
        for (index, word) in words.iter_mut().enumerate() {
            // Transient test IDs; the production encoder assigns vocabulary offsets.
            word.word_id = u32::try_from(index).expect("fixture word ID");
            assert!((word.word_unigram_in_pos_raw - 1.0).abs() < f64::EPSILON);
        }
        let pos = |surface: &str| {
            words
                .iter()
                .find(|word| word.surface == surface)
                .expect("functional word")
                .pos_id
        };
        assert_ne!(pos("が"), pos("を"));
        assert_ne!(pos("た"), pos("だ"));
        assert_eq!(
            resolve_pos_id(
                &pos_names,
                &model_pos_for_word("を", "助詞,格助詞,一般,*,*,*,*")
            ),
            Some(pos("を"))
        );
        assert_eq!(resolve_pos_id(&pos_names, "助詞,格助詞,一般,*,*,*,*"), None);

        let bigram = dir.join("2gram.by-id.tsv");
        fs::write(
            &bigram,
            "id1\tid2\tcount\n1\t3\t12\n1\t4\t8\n3\t5\t12\n4\t6\t8\n5\t2\t12\n6\t2\t8\n",
        )
        .expect("corpus bigram counts");
        let corpus_word_ids = corpus_ids
            .iter()
            .map(|id| id.map(|id| u32::try_from(id).expect("fixture mapped ID")))
            .collect::<Vec<_>>();
        let words_by_id = words.iter().map(|word| (word.word_id, word)).collect();
        let mut pairs = BTreeMap::new();
        let mut pos_contexts = BTreeMap::new();
        let mut word_contexts = BTreeMap::new();
        let no_selected_parent_pairs = HashSet::new();
        let mut no_selected_parent_counts = HashMap::new();
        read_raw_bigram_counts(
            &bigram,
            &corpus_word_ids,
            &words_by_id,
            &mut pairs,
            &mut pos_contexts,
            &mut word_contexts,
            &no_selected_parent_pairs,
            &mut no_selected_parent_counts,
            None,
            &mut [0; 81],
            &mut [0; 9],
            None,
            None,
        )
        .expect("separate functional word POS transitions");
        assert_eq!(pairs.get(&(pos("が"), pos("た"))), Some(&12));
        assert_eq!(pairs.get(&(pos("を"), pos("だ"))), Some(&8));
        assert!(!pairs.contains_key(&(pos("が"), pos("だ"))));
        // The trainer must use the raw count-8 pair too, independent of cutoff.
        fs::write(dir.join("2gram.tsv"), "id1\tid2\tcount\n").expect("empty cutoff bigram fixture");
        let training = training_seeds(&dir, &system, &additions).expect("raw trainer seeds");
        let bos_pos = resolve_pos_id(&training.pos_names, "[BOS]").expect("BOS POS");
        let score = |target| {
            training
                .pos_bigrams
                .iter()
                .find(|&&(a, b, _)| a == bos_pos && b == target)
                .expect("uncut POS transition")
                .2
        };
        assert!((score(pos("が")) - 0.6_f64.ln()).abs() < 1e-12);
        assert!((score(pos("を")) - 0.4_f64.ln()).abs() < 1e-12);
        fs::write(
            dir.join("2gram.tsv"),
            "id1\tid2\tcount\n1\t3\t12\n1\t4\t8\n3\t5\t12\n4\t6\t8\n5\t2\t12\n6\t2\t8\n",
        )
        .expect("retained word bigram fixture");
        let (_, initial_bigrams) = training_seeds_with_bigrams(&dir, &system, &additions, 2)
            .expect("count-ranked initial bigrams");
        assert_eq!(initial_bigrams.len(), 2);
        assert!(initial_bigrams.iter().all(|entry| entry.count == 12));
        assert!(
            initial_bigrams
                .iter()
                .all(|entry| entry.outgoing_count >= entry.count)
        );

        let trigram_rows =
            "id1\tid2\tid3\tcount\n1\t3\t5\t12\n1\t4\t6\t8\n3\t5\t2\t12\n4\t6\t2\t8\n";
        fs::write(dir.join("3gram.tsv"), trigram_rows).expect("frequency trigram fixture");
        fs::write(dir.join("3gram.by-id.tsv"), trigram_rows).expect("raw trigram fixture");
        let crf = training_seeds_with_crf_features(&dir, &system, &additions, 2, 4)
            .expect("CRF feature seeds");
        assert_eq!(crf.bigrams.len(), 2);
        assert_eq!(
            crf.trigrams.len(),
            2,
            "count-8 rows are below the word-trigram cutoff"
        );
        assert!(crf.feature_counts.pos_trigrams.is_empty());
        assert!(!crf.feature_counts.include_pos_trigrams);
        assert_eq!(crf.feature_counts.reading_length_bigram_counts[1], 20);
        assert_eq!(crf.feature_counts.reading_length_bigram_counts[10], 20);
        assert_eq!(crf.feature_counts.reading_length_bigram_counts[9], 20);
        assert_eq!(crf.feature_counts.reading_length_trigram_counts[10], 20);
        assert_eq!(crf.feature_counts.reading_length_trigram_counts[90], 20);
        fs::remove_dir_all(dir).expect("cleanup functional word fixture");
    }

    #[test]
    fn user_pos_string_resolves_independently_for_each_dictionary() {
        let first = vec!["名詞,一般,*,*,*,*,*".to_owned()];
        let second = vec![
            "動詞,自立,*,*,*,*,*".to_owned(),
            "名詞,一般,*,*,*,*,*".to_owned(),
        ];
        assert_eq!(resolve_pos_id(&first, "名詞,一般,*,*,*,*,*"), Some(1));
        assert_eq!(resolve_pos_id(&second, "名詞,一般,*,*,*,*,*"), Some(2));
        assert_eq!(resolve_pos_id(&first, "存在しない品詞"), None);
    }

    #[test]
    fn top_100_noun_affixes_are_refined_after_exclusions_without_merging_pos() {
        let original_pos = "名詞,接尾,一般,*,*,*,*";
        let mut words = Vec::new();
        for id in 0..105_u32 {
            let mut word = test_word(id);
            word.surface = format!("suffix-{id}");
            word.reading = format!("reading-{id}");
            word.pos = original_pos.to_owned();
            word.unigram_count = u64::from(1_000 - id) * 10;
            word.word_unigram_in_pos_raw = 1.0;
            word.word_unigram_in_pos = 1.0;
            words.push(word);
        }
        for (id, surface) in ["号", "時", "日", "数", "級"].into_iter().enumerate() {
            let mut word = test_word(200 + u32::try_from(id).expect("small fixture"));
            word.surface = surface.to_owned();
            word.reading = format!("reading-{surface}");
            word.pos = original_pos.to_owned();
            word.unigram_count = 100_000_000;
            words.push(word);
        }
        let mut counter = test_word(210);
        counter.surface = "個".to_owned();
        counter.reading = "こ".to_owned();
        counter.pos = "名詞,接尾,助数詞,*,*,*,*".to_owned();
        counter.unigram_count = 100_000_000;
        words.push(counter);

        let mut pos_names = vec![original_pos.to_owned()];
        let selections = promote_top_affixes(&mut words, &mut pos_names, 100)
            .expect("refine selected affix POS values");

        assert_eq!(selections.len(), 100);
        assert_eq!(selections[0].surface, "suffix-0");
        assert_eq!(selections[0].unigram_count, 1_000);
        assert_eq!(selections[99].surface, "suffix-99");
        assert_eq!(selections[99].unigram_count, 901);
        assert!(selections.iter().all(|selection| {
            selection.assigned_pos.starts_with(original_pos)
                && !EXCLUDED_AFFIX_SURFACES.contains(&selection.surface.as_str())
                && selection.surface != "個"
        }));
        assert_eq!(pos_names.len(), 102, "only the selected 100 POS are split");
        assert!(
            words
                .iter()
                .filter(
                    |word| EXCLUDED_AFFIX_SURFACES.contains(&word.surface.as_str())
                        || word.surface == "個"
                )
                .all(|word| word.pos == original_pos || word.pos.starts_with("名詞,接尾,助数詞"))
        );
    }

    #[test]
    fn prefix_cache_target_is_resolved_by_following_the_record_offset() {
        let mut word = test_word(41);
        word.reading = "にわ".to_owned();
        let words = [(41, word)].into_iter().collect::<BTreeMap<_, _>>();
        let phrase = Phrase {
            ids: vec![41, 42],
            reading: "にわいぬ".to_owned(),
            unigram: 0.25,
            vocab_offset: 99,
        };
        let phrases = [(99, &phrase)].into_iter().collect::<BTreeMap<_, _>>();
        let word_refs = words
            .iter()
            .map(|(id, word)| (*id, word))
            .collect::<BTreeMap<_, _>>();

        assert_eq!(
            resolve_prediction_target(41, "に", &word_refs, &phrases).expect("resolve word target"),
            PredictionTarget::Word(41)
        );
        assert_eq!(
            resolve_prediction_target(99, "に", &word_refs, &phrases)
                .expect("resolve phrase target"),
            PredictionTarget::Phrase(99)
        );
        assert!(resolve_prediction_target(99, "い", &word_refs, &phrases).is_err());
        assert!(resolve_prediction_target(77, "に", &word_refs, &phrases).is_err());
    }

    #[test]
    fn cutoff_external_phrases_are_retained_when_a_prefix_cache_needs_them() {
        let words = ["あ", "い", "う", "え"]
            .into_iter()
            .enumerate()
            .map(|(source_id, reading)| {
                let mut word = test_word(u32::try_from(source_id).expect("small word id"));
                word.source_id = source_id;
                word.surface = reading.to_owned();
                word.reading = reading.to_owned();
                word
            })
            .collect::<Vec<_>>();
        let source_ids = vec![None, None, None, Some(0), Some(1), Some(2), Some(3)];
        let dir = temp_dir();
        fs::create_dir_all(&dir).expect("create phrase fixture directory");
        let phrase_path = dir.join("phrases.tsv");
        fs::write(&phrase_path, "3\t4\t0.9\n5\t6\t0.8\n3\t0.8\n3\t6\t0.7\n")
            .expect("write candidate fixture");

        let (phrases, reading_index, prefixes) =
            read_phrases(&phrase_path, &words, &source_ids, 1, 2)
                .expect("read phrases beyond base cutoff for cache");
        assert_eq!(phrases.len(), 3);
        assert_eq!(reading_index.len(), 3);
        assert_eq!(
            prefixes.get("う").expect("cache-only prefix"),
            &[CacheTarget::Phrase(1)]
        );
        assert_eq!(
            prefixes.get("あ").expect("base prefix and cache-only"),
            &[CacheTarget::Phrase(0), CacheTarget::Word(0)]
        );
        fs::remove_dir_all(dir).expect("remove phrase fixture directory");
    }

    #[test]
    fn prediction_cache_header_is_only_allowed_on_short_hiragana_keys() {
        let group = [0, PREFIX_CACHE_HEADER_TAG, 1, 17, 0, 0, 0, 0xe0];
        let parsed = decode_key_group(
            &group,
            1,
            group.len(),
            4,
            "に",
            0,
            &ProbabilityCodebook {
                model_id: QUANT_WORD_UNIGRAM_IN_POS,
                log_values: Vec::new(),
            },
        )
        .expect("parse prefix cache group");
        assert_eq!(parsed.cache_targets, Some(vec![17]));

        assert!(
            decode_key_group(
                &group,
                1,
                group.len(),
                4,
                "庭",
                0,
                &ProbabilityCodebook {
                    model_id: QUANT_WORD_UNIGRAM_IN_POS,
                    log_values: Vec::new(),
                },
            )
            .is_err()
        );
    }

    #[test]
    fn pos_bigram_is_dense_and_reserves_id_zero() {
        let names = vec![
            "動詞,自立,*,*,*,*,*".to_owned(),
            "名詞,一般,*,*,*,*,*".to_owned(),
        ];
        let entries = vec![
            PosBigramEntry {
                source_pos: 1,
                target_pos: 2,
                probability: 0.5,
                code: 1,
            },
            PosBigramEntry {
                source_pos: 2,
                target_pos: 1,
                probability: 0.25,
                code: 2,
            },
        ];
        let bytes = encode_pos_bigrams(&entries, names.len()).expect("encode POS matrix");
        assert_eq!(bytes, [0, 1, 2, 0]);

        let codebook = ProbabilityCodebook {
            model_id: QUANT_POS_BIGRAM,
            log_values: vec![0.5_f32.ln(), 0.25_f32.ln()],
        };
        let decoded = decode_pos_bigrams(&bytes, &names, &codebook).expect("decode POS matrix");
        assert_eq!(decoded.len(), 2);
        assert_eq!((decoded[0].source_pos, decoded[0].target_pos), (1, 2));
        assert_eq!((decoded[1].source_pos, decoded[1].target_pos), (2, 1));

        let id_zero = [PosBigramEntry {
            source_pos: 0,
            target_pos: 1,
            probability: 1.0,
            code: 1,
        }];
        assert!(encode_pos_bigrams(&id_zero, names.len()).is_err());
    }

    #[test]
    fn word_bigram_word_id_uses_three_bytes() {
        let mut bytes = Vec::new();
        put_u24(&mut bytes, 0x00ab_cdef).expect("encode 24-bit word ID");
        assert_eq!(bytes, [0xef, 0xcd, 0xab]);
        assert_eq!(
            Decoder::new(&bytes).u24().expect("decode 24-bit word ID"),
            0x00ab_cdef
        );
        assert!(put_u24(&mut Vec::new(), 0x0100_0000).is_err());
    }

    #[test]
    fn trigram_uses_dense_or_sparse_b_and_c_slots_and_round_trips() {
        let codebook = ProbabilityCodebook {
            model_id: QUANT_TRIGRAM,
            log_values: vec![log_probability_as_f32(0.5_f64.ln())],
        };
        let words = [10, 20, 30, 40, 50]
            .into_iter()
            .map(|word_id| (word_id, test_word(word_id)))
            .collect::<BTreeMap<_, _>>();
        let sparse_edges = vec![
            test_bigram(10, 20),
            test_bigram(20, 30),
            test_bigram(20, 40),
            test_bigram(20, 50),
        ];
        let sparse_contexts = vec![test_context(10, 0, 1), test_context(20, 1, 3)];
        let sparse_records = vec![test_trigram(0, 0)];
        let sparse_blocks = make_trigram_blocks(&sparse_records, &sparse_edges, &sparse_contexts)
            .expect("sparse block metadata");
        let sparse_bytes = encode_trigrams(&sparse_records, &sparse_blocks, sparse_edges.len())
            .expect("encode sparse trigram");
        let sparse_decoded = decode_trigrams(
            &sparse_bytes,
            &sparse_contexts,
            &sparse_edges,
            &words,
            &codebook,
        )
        .expect("decode sparse trigram");
        assert_eq!(sparse_decoded.len(), 1);
        assert_eq!(sparse_decoded[0].codec, 1);
        assert_eq!(sparse_decoded[0].target_word, 30);

        let dense_edges = vec![test_bigram(10, 20), test_bigram(20, 30)];
        let dense_contexts = vec![test_context(10, 0, 1), test_context(20, 1, 1)];
        let dense_records = vec![test_trigram(0, 0)];
        let dense_blocks = make_trigram_blocks(&dense_records, &dense_edges, &dense_contexts)
            .expect("dense block metadata");
        let dense_bytes = encode_trigrams(&dense_records, &dense_blocks, dense_edges.len())
            .expect("encode dense trigram");
        let dense_decoded = decode_trigrams(
            &dense_bytes,
            &dense_contexts,
            &dense_edges,
            &words,
            &codebook,
        )
        .expect("decode dense trigram");
        assert_eq!(dense_decoded.len(), 1);
        assert_eq!(dense_decoded[0].codec, 0);
        assert_eq!(dense_decoded[0].target_word, 30);
    }

    #[test]
    fn trigram_slot_index_width_tracks_bigram_continuation_count() {
        assert_eq!(trigram_slot_index_width(256), 1);
        assert_eq!(trigram_slot_index_width(257), 2);
        assert_eq!(trigram_slot_index_width(65_536), 2);
        assert_eq!(trigram_slot_index_width(65_537), 4);

        let mut encoded = Vec::new();
        put_trigram_slot(&mut encoded, 299, 2).expect("encode u16 trigram slot");
        assert_eq!(encoded, 299_u16.to_le_bytes());
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn container_round_trip_checks_blocks_and_dump_outputs_records() {
        let dir = temp_dir();
        fs::create_dir_all(&dir).expect("test directory");
        let input = dir.join("input");
        fs::create_dir_all(&input).expect("input dir");
        let system_dictionary = input.join("ipadic-source");
        let user_additions = input.join("user-additions");
        fs::create_dir_all(&system_dictionary).expect("system dictionary source dir");
        fs::create_dir_all(&user_additions).expect("user additions source dir");
        fs::write(system_dictionary.join("dicrc"), "config-charset = UTF-8\n")
            .expect("system dicrc");
        fs::write(
            system_dictionary.join("Noun.csv"),
            "庭,1285,1285,5543,名詞,一般,*,*,*,*,庭,ニワ,ニワ\n犬,1285,1285,5543,名詞,一般,*,*,*,*,犬,イヌ,イヌ\nあ,1285,1285,5543,名詞,一般,*,*,*,*,あ,ア,ア\n",
        )
        .expect("system noun source");
        fs::write(
            user_additions.join("20261002-test-new-words.txt"),
            "未出,1285,1285,5000,名詞,一般,*,*,*,*,未出,ミシュツ,ミシュツ\n",
        )
        .expect("user addition source");
        fs::write(
            input.join("vocab.tsv"),
            "0\t[NULL]\n1\t[BOS]\n2\t[EOS]\n3\t庭,名詞,一般,*,*,*,*,庭,ニワ,ニワ\n4\t犬,名詞,一般,*,*,*,*,犬,イヌ,イヌ\n5\tあ,名詞,一般,*,*,*,*,あ,ア,ア\n",
        )
        .expect("vocab fixture");
        fs::write(
            input.join("1gram.by-id.tsv"),
            "id\tcount\n1\t12\n2\t12\n3\t10\n4\t10\n5\t2\n",
        )
        .expect("unigram fixture");
        fs::write(
            input.join("2gram.by-id.tsv"),
            "id1\tid2\tcount\n1\t3\t10\n1\t5\t2\n3\t4\t10\n4\t2\t10\n5\t2\t2\n",
        )
        .expect("full bigram fixture");
        fs::write(
            input.join("2gram.tsv"),
            "id1\tid2\tcount\n1\t3\t10\n3\t4\t10\n4\t2\t10\n",
        )
        .expect("retained bigram fixture");
        fs::write(
            input.join("3gram.tsv"),
            "id1\tid2\tid3\tcount\n1\t3\t4\t10\n3\t4\t2\t10\n",
        )
        .expect("retained trigram fixture");
        fs::write(
            input.join("prediction-phrase.txt"),
            "3\t4\t0.25\n4\t3\t0.125\n3\t0.217391304348\n",
        )
        .expect("prediction candidate fixture");
        let sdict = dir.join("fixture.sdict");
        compile(
            &input,
            &sdict,
            &system_dictionary,
            &user_additions,
            DEFAULT_NGRAM_CUTOFF,
        )
        .expect("compile fixture");
        let dictionary_bytes = fs::read(&sdict).expect("read compiled fixture");
        let (_, _, blocks) = parse_container(&dictionary_bytes).expect("parse v7 fixture");
        assert_eq!(blocks.len(), 7, "prediction uses no standalone blocks");
        let block_map = blocks
            .iter()
            .map(|block| (block.guid, block_slice(&dictionary_bytes, block)))
            .collect::<BTreeMap<_, _>>();
        let vocabulary_bytes = required_block(&block_map, &VOCABULARY).expect("vocabulary block");
        let string_index = DoubleArray::decode(
            required_block(&block_map, &STRING_INDEX).expect("string index block"),
        )
        .expect("decode string index");
        let prefix_key = super::super::dict_char::encode("に").expect("encode prefix key");
        let prefix_leaf = string_index
            .lookup_leaf(&prefix_key, vocabulary_bytes)
            .expect("prefix cache key leaf");
        let prefix_group_offset = string_index
            .leaves()
            .expect("enumerate leaves")
            .into_iter()
            .find(|(leaf, _)| *leaf == prefix_leaf)
            .map(|(_, offset)| usize::try_from(offset).expect("group offset"))
            .expect("prefix group offset");
        let tail = tail_at(vocabulary_bytes, prefix_group_offset).expect("prefix tail");
        let cache_header_offset =
            prefix_group_offset + if tail.is_empty() { 0 } else { 1 + tail.len() };
        let mut cache_input = Decoder::new(&vocabulary_bytes[cache_header_offset..]);
        assert_eq!(
            cache_input.u8().expect("cache header tag"),
            PREFIX_CACHE_HEADER_TAG
        );
        let cached_target_count = usize::from(cache_input.u8().expect("cache target count"));
        assert_eq!(cached_target_count, 2);
        let phrase_target = cache_input.u32().expect("cache phrase target offset");
        assert!((0x61..=0x65).contains(&vocabulary_bytes[phrase_target as usize]));
        let word_target = cache_input.u32().expect("cache word target offset");
        assert_eq!(vocabulary_bytes[word_target as usize], WORD_RECORD_TAG);

        let dump_dir = dir.join("dump");
        dump(&sdict, &dump_dir).expect("dump fixture");
        let pos = fs::read_to_string(dump_dir.join("pos.tsv")).expect("POS dump");
        assert!(pos.contains("名詞,一般,*,*,*,*,*"));
        let phrases =
            fs::read_to_string(dump_dir.join("prediction-phrases.tsv")).expect("phrase dump");
        assert!(phrases.contains("にわいぬ"));
        let prefixes =
            fs::read_to_string(dump_dir.join("prediction-prefix.tsv")).expect("prefix dump");
        assert!(prefixes.contains("にわ\t1\t"));
        assert!(prefixes.contains("\t庭/犬\tにわいぬ\t"));
        assert!(prefixes.contains("にわ\t2\tword\t"));
        assert!(prefixes.contains("\t庭\tにわ\tword_unigram_in_pos\t"));
        let vocabulary =
            fs::read_to_string(dump_dir.join("vocabulary.tsv")).expect("vocabulary dump");
        assert!(vocabulary.contains("\t庭\tにわ\t"));
        assert!(vocabulary.contains("\tあ\tあ\t"));
        assert!(vocabulary.contains("\t未出\tみしゅつ\t"));
        let garden_probability = vocabulary
            .lines()
            .find(|line| line.split('\t').nth(1) == Some("庭"))
            .and_then(|line| line.split('\t').nth(5))
            .and_then(|value| value.parse::<f64>().ok())
            .expect("observed corpus word probability");
        let unseen_probability = vocabulary
            .lines()
            .find(|line| line.split('\t').nth(1) == Some("未出"))
            .and_then(|line| line.split('\t').nth(5))
            .and_then(|value| value.parse::<f64>().ok())
            .expect("unseen lexicon word probability");
        assert!(garden_probability > unseen_probability);
        assert!(unseen_probability > 0.0);
        let observed_to_unseen_ratio = garden_probability / unseen_probability;
        assert!((95.0..=105.0).contains(&observed_to_unseen_ratio));
        let pos_bigrams =
            fs::read_to_string(dump_dir.join("pos-bigram.tsv")).expect("POS bigram dump");
        assert!(pos_bigrams.lines().count() > 1);
        let word_bigrams =
            fs::read_to_string(dump_dir.join("word-bigram.tsv")).expect("word bigram dump");
        assert!(word_bigrams.contains("\t庭\t犬\t"));
        let backoffs =
            fs::read_to_string(dump_dir.join("word-bigram-backoff.tsv")).expect("backoff dump");
        assert!(backoffs.contains("\t庭\t"));
        let trigrams = fs::read_to_string(dump_dir.join("trigram.tsv")).expect("trigram dump");
        assert!(trigrams.contains("\t庭\t犬\t"));

        let cutoff_sdict = dir.join("fixture-cutoff11.sdict");
        compile(
            &input,
            &cutoff_sdict,
            &system_dictionary,
            &user_additions,
            11,
        )
        .expect("compile cutoff-11 fixture");
        let cutoff_dump_dir = dir.join("dump-cutoff11");
        dump(&cutoff_sdict, &cutoff_dump_dir).expect("dump cutoff-11 fixture");
        let cutoff_pos_bigrams = fs::read_to_string(cutoff_dump_dir.join("pos-bigram.tsv"))
            .expect("cutoff POS bigram dump");
        assert_eq!(cutoff_pos_bigrams, pos_bigrams);
        assert_eq!(
            fs::read_to_string(cutoff_dump_dir.join("word-bigram.tsv"))
                .expect("cutoff word bigram dump")
                .lines()
                .count(),
            1
        );
        assert_eq!(
            fs::read_to_string(cutoff_dump_dir.join("trigram.tsv"))
                .expect("cutoff trigram dump")
                .lines()
                .count(),
            1
        );
        let readings =
            fs::read_to_string(dump_dir.join("reading-word-index.tsv")).expect("reading dump");
        assert!(readings.contains("あ\t"));
        fs::remove_dir_all(dir).expect("cleanup fixture");
    }

    fn test_word(word_id: u32) -> Word {
        Word {
            source_id: 0,
            word_id,
            surface: format!("w{word_id}"),
            reading: String::new(),
            display_leaf: 0,
            reading_leaf: 0,
            pos: String::new(),
            pos_id: 0,
            bigram_block_index: 0,
            backoff_alpha_code: 0,
            unigram_count: 0,
            word_unigram_in_pos_raw: 0.0,
            word_unigram_in_pos: 0.0,
            word_unigram_in_pos_code: 0,
            flags: 0,
        }
    }

    fn test_bigram(source_word: u32, target_word: u32) -> WordBigramEntry {
        WordBigramEntry {
            source_word,
            target_word,
            count: 1,
            probability: 0.5,
            code: 1,
        }
    }

    fn test_context(source_word: u32, first_edge: u32, edge_count: u32) -> WordBigramContext {
        WordBigramContext {
            source_word,
            first_edge,
            edge_count,
            alpha: 0.0,
            alpha_code: 0,
        }
    }

    fn test_trigram(parent_edge: u32, slot: u32) -> TrigramRecord {
        TrigramRecord {
            parent_edge,
            slot,
            probability: 0.5,
            code: 1,
        }
    }

    #[test]
    fn double_array_uses_inline_vocab_tail_and_restores_original_keys() {
        let source_keys = [
            "あ",
            "あい",
            "あいう",
            "日本語",
            "にわ",
            "にわとり",
            "ああああああああああああああああああああああああああああああああああああああああ",
        ];
        let mut groups = source_keys
            .iter()
            .map(|key| KeyGroup {
                key: super::super::dict_char::encode(key).expect("encode key"),
                ..KeyGroup::default()
            })
            .collect::<Vec<_>>();
        groups.sort_by(|left, right| left.key.cmp(&right.key));
        for index in 0..groups.len() {
            let previous = index.checked_sub(1).map_or(0, |prev| {
                common_prefix_len(&groups[prev].key, &groups[index].key)
            });
            let next = groups
                .get(index + 1)
                .map_or(0, |next| common_prefix_len(&groups[index].key, &next.key));
            let unique = previous.max(next).saturating_add(1);
            let minimum_prefix = groups[index].key.len().saturating_sub(32);
            let prefix_length = unique.max(minimum_prefix).min(groups[index].key.len());
            groups[index].prefix = groups[index].key[..prefix_length].to_vec();
            groups[index].tail = groups[index].key[prefix_length..].to_vec();
        }

        let mut index = DoubleArray::build(&mut groups).expect("build double array");
        let mut vocabulary = vec![0];
        for group in &mut groups {
            group.vocab_offset = u32::try_from(vocabulary.len()).expect("group offset");
            if !group.tail.is_empty() {
                vocabulary.push(0x20 | u8::try_from(group.tail.len() - 1).expect("tail length"));
                vocabulary.extend_from_slice(&group.tail);
            }
            vocabulary.push(0xe0);
        }
        index.bind_vocab_offsets(&groups).expect("bind offsets");
        let restored =
            DoubleArray::decode(&index.encode().expect("encode index")).expect("decode index");
        for group in &groups {
            assert_eq!(
                restored.lookup_leaf(&group.key, &vocabulary),
                Some(group.leaf_node)
            );
            assert_eq!(
                restored
                    .key_for_leaf(group.leaf_node, &vocabulary)
                    .expect("restore key"),
                group.key
            );
            assert!(group.tail.len() <= 32);
        }
        assert!(restored.lookup_leaf(&[0xff, 0xff], &vocabulary).is_none());
    }
}
