//! One-pass normalized page-block cache for random-order epochs over bz2 input.
//! Pages stay intact; no lattices or morphological-analysis output are cached.

use std::collections::HashSet;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use bzip2::write::BzEncoder;
use serde::{Deserialize, Serialize};

use crate::Result;
use crate::corpus::{self, Index, Sentence};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CachedSentence {
    pub record: usize,
    pub sentence: Sentence,
}

#[derive(Clone, Copy, Debug)]
pub struct BuildConfig {
    pub prefix_records: usize,
    pub skip_records: usize,
    pub training_sentences: usize,
    pub max_words: usize,
    pub pages_per_block: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Block {
    pub file: String,
    pub pages: usize,
    pub sentences: usize,
    pub source_first_record: usize,
    pub source_last_record: usize,
    pub compressed_bytes: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PageBlocks {
    pub directory: PathBuf,
    pub pages_per_block: usize,
    pub blocks: Vec<Block>,
    pub training_sentences: usize,
}

type BlockWriter = BzEncoder<BufWriter<File>>;

pub struct Builder {
    directory: PathBuf,
    pages_per_block: usize,
    pages: HashSet<u64>,
    previous_page: Option<u64>,
    current: Option<(BlockWriter, Block)>,
    blocks: Vec<Block>,
    sentences: usize,
    started: std::time::Instant,
}
impl std::fmt::Debug for Builder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Builder")
            .field("directory", &self.directory)
            .field("pages_per_block", &self.pages_per_block)
            .field("pages", &self.pages.len())
            .field("sentences", &self.sentences)
            .field("blocks", &self.blocks)
            .finish_non_exhaustive()
    }
}

#[derive(Serialize)]
struct BorrowedSentence<'a> {
    record: usize,
    sentence: &'a Sentence,
}

impl Builder {
    pub fn new(directory: &Path, pages_per_block: usize) -> Result<Self> {
        if pages_per_block == 0 {
            return Err("page block size must be positive".into());
        }
        fs::create_dir(directory)?;
        Ok(Self {
            directory: directory.to_owned(),
            pages_per_block,
            pages: HashSet::new(),
            previous_page: None,
            current: None,
            blocks: Vec::new(),
            sentences: 0,
            started: std::time::Instant::now(),
        })
    }

    /// Observe *all* source pages, including held-out/rejected records, before
    /// selection. Never split a page or define page blocks by sentence counts.
    pub fn observe(&mut self, page: u64, record: usize) -> Result<()> {
        if self.previous_page != Some(page) {
            if !self.pages.insert(page) {
                return Err(
                    "page records are not contiguous; cannot form intact page blocks".into(),
                );
            }
            self.previous_page = Some(page);
            if self
                .current
                .as_ref()
                .is_some_and(|(_, b)| b.pages == self.pages_per_block)
            {
                let (writer, block) = self.current.take().expect("active block");
                self.blocks.push(finish(&self.directory, writer, block)?);
            }
            if self.current.is_none() {
                self.current = Some(new_block(&self.directory, self.blocks.len(), record)?);
            }
            self.current.as_mut().expect("active block").1.pages += 1;
        }
        self.current
            .as_mut()
            .ok_or("missing page block")?
            .1
            .source_last_record = record;
        Ok(())
    }

    pub fn add(&mut self, record: usize, sentence: &Sentence) -> Result<()> {
        if self.previous_page != Some(sentence.page_id) {
            return Err("cached sentence does not belong to observed page".into());
        }
        let (writer, block) = self.current.as_mut().ok_or("missing page block")?;
        serde_json::to_writer(&mut *writer, &BorrowedSentence { record, sentence })?;
        writeln!(writer)?;
        self.sentences += 1;
        block.sentences += 1;
        if self.sentences % 10_000 == 0 {
            eprintln!(
                "page_cache pages={} blocks={} sentences={} elapsed={:.1}s",
                self.pages.len(),
                self.blocks.len() + 1,
                self.sentences,
                self.started.elapsed().as_secs_f64()
            );
        }
        Ok(())
    }

    pub fn finish(mut self) -> Result<PageBlocks> {
        if let Some((writer, block)) = self.current.take() {
            self.blocks.push(finish(&self.directory, writer, block)?);
        }
        let result = PageBlocks {
            directory: self.directory,
            pages_per_block: self.pages_per_block,
            blocks: self.blocks,
            training_sentences: self.sentences,
        };
        let mut manifest = BufWriter::new(
            OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(result.directory.join("manifest.json"))?,
        );
        serde_json::to_writer_pretty(&mut manifest, &result)?;
        writeln!(manifest)?;
        manifest.flush()?;
        eprintln!(
            "page_cache completed pages={} blocks={} sentences={} elapsed={:.1}s",
            self.pages.len(),
            result.blocks.len(),
            result.training_sentences,
            self.started.elapsed().as_secs_f64()
        );
        Ok(result)
    }
}

fn new_block(directory: &Path, number: usize, record: usize) -> Result<(BlockWriter, Block)> {
    let file = format!("block-{number:05}.normalized.jsonl.bz2");
    let writer = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(directory.join(&file))?;
    Ok((
        BzEncoder::new(BufWriter::new(writer), bzip2::Compression::fast()),
        Block {
            file,
            pages: 0,
            sentences: 0,
            source_first_record: record,
            source_last_record: record,
            compressed_bytes: 0,
        },
    ))
}

fn finish(directory: &Path, writer: BlockWriter, mut block: Block) -> Result<Block> {
    writer.finish()?.flush()?;
    block.compressed_bytes = fs::metadata(directory.join(&block.file))?.len();
    Ok(block)
}

impl PageBlocks {
    /// Reuse the exact normalized cache of a stopped run. Manifest paths never
    /// override the caller's directory; each block must be a single file name.
    pub fn load(directory: &Path) -> Result<Self> {
        let mut saved: Self = serde_json::from_reader(std::io::BufReader::new(File::open(
            directory.join("manifest.json"),
        )?))?;
        directory.clone_into(&mut saved.directory);
        if saved.pages_per_block == 0 || saved.blocks.is_empty() {
            return Err("invalid page cache manifest".into());
        }
        let mut total = 0_usize;
        let mut previous = 0;
        for block in &saved.blocks {
            let mut parts = Path::new(&block.file).components();
            if !matches!(parts.next(), Some(std::path::Component::Normal(_)))
                || parts.next().is_some()
                || block.pages == 0
                || block.pages > saved.pages_per_block
                || block.source_first_record <= previous
                || block.source_last_record < block.source_first_record
                || fs::metadata(directory.join(&block.file))?.len() != block.compressed_bytes
            {
                return Err("page cache manifest/file mismatch".into());
            }
            previous = block.source_last_record;
            total = total
                .checked_add(block.sentences)
                .ok_or("page cache count overflow")?;
        }
        if total != saved.training_sentences {
            return Err("page cache total mismatch".into());
        }
        Ok(saved)
    }

    /// Build only the already-selected source prefix. IDs and normalized gold
    /// belong to the current vocabulary; this per-run cache is not cross-model input.
    pub fn build(input: &Path, directory: &Path, index: &Index, c: BuildConfig) -> Result<Self> {
        if c.pages_per_block == 0 || c.training_sentences == 0 {
            return Err("page block size and training limit must be positive".into());
        }
        let mut builder = Builder::new(directory, c.pages_per_block)?;
        let mut reader = corpus::open(input)?;
        let mut line = String::new();
        let mut sentences = 0;
        for number in 1..=c.prefix_records {
            let record = corpus::read_record(&mut *reader, &mut line, number)?
                .ok_or("source ended before selected prefix")?;
            if number <= c.skip_records {
                continue;
            }
            builder.observe(record.page_id, number)?;
            if corpus::validation_page(record.page_id) || sentences >= c.training_sentences {
                continue;
            }
            let Ok(sentence) = corpus::normalize(record, index, c.max_words) else {
                continue;
            };
            builder.add(number, &sentence)?;
            sentences += 1;
        }
        if sentences != c.training_sentences {
            return Err("training selection changed while constructing page blocks".into());
        }
        builder.finish()
    }

    pub fn open(&self, block: usize) -> Result<Box<dyn BufRead>> {
        let info = self
            .blocks
            .get(block)
            .ok_or("page block index out of range")?;
        corpus::open(&self.directory.join(&info.file))
    }

    /// Independent deterministic order for each epoch; the complete permutation
    /// is recorded in metrics, so cache selection and thread count don't affect it.
    #[must_use]
    pub fn order(&self, seed: u64, epoch: usize) -> Vec<usize> {
        let mut order = (0..self.blocks.len()).collect::<Vec<_>>();
        let mut random = seed ^ (epoch as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        if random == 0 {
            random = 1;
        }
        corpus::shuffle(&mut order, &mut random);
        order
    }
}

pub fn read_cached(reader: &mut dyn BufRead, line: &mut String) -> Result<Option<CachedSentence>> {
    line.clear();
    if reader.take((8 << 20) + 1).read_line(line)? == 0 {
        return Ok(None);
    }
    if line.len() > 8 << 20 {
        return Err("cached sentence exceeds 8 MiB safety limit".into());
    }
    Ok(Some(serde_json::from_str(line)?))
}
