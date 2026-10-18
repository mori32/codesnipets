//! Resume an interrupted page-block epoch without repeating committed batches.
use std::collections::BTreeMap;
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};

use ai_engine_crf::{Result, blocks, corpus::Sentence, output::Fingerprint};
use serde::de::DeserializeOwned;
use serde_json::Value;

use super::{InputReport, MissedRecord, Options, Pass, TrainingStop};

pub(super) fn read<T: DeserializeOwned>(path: &Path) -> Result<T> {
    Ok(serde_json::from_reader(BufReader::new(File::open(path)?))?)
}

pub(super) struct Resume {
    pub directory: PathBuf,
    pub page_cache: PathBuf,
    pub stop: TrainingStop,
    pub processed: usize,
    pub previous_seconds: f64,
    batches: usize,
    first_loss: f64,
    primary_margin_active_sentences: usize,
    tri_loss: f64,
    tri_count: usize,
    missed: usize,
    top1_hits: usize,
    top1_evaluated: usize,
    gold_augmented: usize,
    post_trigram_top1_hits: usize,
    post_trigram_top1_evaluated: usize,
    missed_records: Vec<MissedRecord>,
    expected_fingerprint: Value,
}

fn integer(value: &Value, key: &str) -> Result<usize> {
    Ok(usize::try_from(
        value[key]
            .as_u64()
            .ok_or_else(|| format!("missing resume field {key}"))?,
    )?)
}

impl Resume {
    pub fn resolve_checkpoint(path: &Path) -> Result<PathBuf> {
        if path.join("COMMITTED").is_file()
            && path.join("training-stop.json").is_file()
            && path.join("stopped-model.json").is_file()
        {
            return Ok(std::fs::canonicalize(path)?);
        }
        if path.join("training-stop.json").is_file() && path.join("stopped-model.json").is_file() {
            return Ok(std::fs::canonicalize(path)?);
        }
        let recovery = if path.file_name().is_some_and(|name| name == "recovery") {
            path.to_path_buf()
        } else {
            path.join("recovery")
        };
        let mut candidates = std::fs::read_dir(&recovery)?
            .filter_map(std::result::Result::ok)
            .map(|entry| entry.path())
            .filter(|candidate| {
                candidate.is_dir()
                    && candidate.join("COMMITTED").is_file()
                    && candidate.join("training-stop.json").is_file()
                    && candidate.join("stopped-model.json").is_file()
            })
            .collect::<Vec<_>>();
        candidates.sort();
        candidates.pop().ok_or_else(|| {
            format!(
                "no complete recovery checkpoint under {}",
                recovery.display()
            )
            .into()
        })
    }

    pub fn load(directory: &Path, o: &Options) -> Result<Self> {
        let directory = Self::resolve_checkpoint(directory)?;
        let old: Options = read(&directory.join("config.json"))?;
        let stop: TrainingStop = read(&directory.join("training-stop.json"))?;
        if old.seed != o.seed
            || (old.batch_size != o.batch_size && !o.allow_resume_batch_size_change)
            || old.shuffle_buffer != o.shuffle_buffer
            || old.page_block_size != o.page_block_size
            || old.nbest != o.nbest
            || old.gold_path_augmentation != o.gold_path_augmentation
            || old.max_words != o.max_words
            || old.learning_rate.to_bits() != o.learning_rate.to_bits()
            || old.l2.to_bits() != o.l2.to_bits()
            || old.primary_nbest_margin != o.primary_nbest_margin
            || old.full_corpus != o.full_corpus
            || old.missed_feedback != o.missed_feedback
            || old.test_prefix_sentences != o.test_prefix_sentences
            || std::fs::canonicalize(&old.input)? != std::fs::canonicalize(&o.input)?
            || stop.epoch == 0
            || stop.epoch > o.epochs
            || stop.active_page_block.is_none()
            || stop.shuffle_state == 0
        {
            return Err("resume settings/state mismatch (seed, batch size unless explicitly allowed, window, cache, N, corpus, learning rate/L2)".into());
        }
        if old.batch_size != o.batch_size {
            eprintln!(
                "resume_batch_size_changed old={} new={} pending_window_rebatched=true; synchronous gradients are averaged over the new minibatch, so the optimization trajectory changes",
                old.batch_size, o.batch_size
            );
        }
        let partial: Value = read(&directory.join("partial-training.json"))?;
        let processed = integer(&partial, "training_sentences_processed")?;
        if integer(&partial, "epoch")? != stop.epoch {
            return Err("resume epoch mismatch".into());
        }
        let metrics: Value = read(&directory.join("metrics.json"))?;
        let epoch = metrics
            .as_array()
            .ok_or("invalid prior metrics")?
            .iter()
            .find(|m| m["epoch"].as_u64() == Some(stop.epoch as u64))
            .ok_or("missing stopped epoch metrics")?;
        if integer(epoch, "training_sentences_processed")? != processed {
            return Err("resume progress mismatch".into());
        }
        let missed = integer(epoch, "training_gold_missing_nbest")?;
        let raw_hits = processed
            .checked_sub(missed)
            .ok_or("invalid resume gold count")?;
        let tri_count = partial["rerank_sentences"]
            .as_u64()
            .map(usize::try_from)
            .transpose()?
            .unwrap_or(raw_hits);
        let gold_augmented = partial["gold_augmented_sentences"]
            .as_u64()
            .map(usize::try_from)
            .transpose()?
            .unwrap_or(0);
        let primary_margin_active_sentences = partial["primary_nbest_margin_active_sentences"]
            .as_u64()
            .map(usize::try_from)
            .transpose()?
            .unwrap_or(0);
        let top1_hits = partial["training_preupdate_top1_gold_hits"]
            .as_u64()
            .map(usize::try_from)
            .transpose()?
            .unwrap_or(0);
        let top1_evaluated = partial["training_preupdate_top1_evaluated_sentences"]
            .as_u64()
            .map(usize::try_from)
            .transpose()?
            .unwrap_or(0);
        let post_trigram_top1_hits = partial["training_post_trigram_top1_gold_hits"]
            .as_u64()
            .map(usize::try_from)
            .transpose()?
            .unwrap_or(0);
        let post_trigram_top1_evaluated = partial["training_post_trigram_top1_evaluated_sentences"]
            .as_u64()
            .map(usize::try_from)
            .transpose()?
            .unwrap_or(0);
        let missed_records = if let Some(path) = partial["missed_records_file"].as_str() {
            let path = Path::new(path);
            let path = if path.is_absolute() {
                path.to_path_buf()
            } else {
                directory.join(path)
            };
            super::read_missed_records(&path)?
        } else {
            Vec::new()
        };
        if top1_hits > top1_evaluated || top1_evaluated > processed {
            return Err("invalid resumed pre-update top-1 counters".into());
        }
        #[allow(clippy::cast_precision_loss)]
        let first_loss = partial["first_loss_sum"].as_f64().unwrap_or_else(|| {
            epoch["mean_update_first_nll"].as_f64().unwrap_or(0.0) * processed as f64
        });
        #[allow(clippy::cast_precision_loss)]
        let tri_loss = partial["rerank_loss_sum"].as_f64().unwrap_or_else(|| {
            epoch["mean_update_rerank_nll"].as_f64().unwrap_or(0.0) * tri_count as f64
        });
        let fingerprint_path = directory.join("resume-fingerprint.json");
        let expected_fingerprint = if fingerprint_path.is_file() {
            read(&fingerprint_path)?
        } else {
            let verification: Value = read(&directory.join("verification.json"))?;
            verification["fingerprint"].clone()
        };
        if expected_fingerprint.is_null() {
            return Err("missing resume model fingerprint".into());
        }
        let info_path = directory.join("resume-info.json");
        let page_cache = if info_path.is_file() {
            let info: Value = read(&info_path)?;
            PathBuf::from(
                info["page_cache_directory"]
                    .as_str()
                    .ok_or("missing reused cache location")?,
            )
        } else {
            directory.join("page-blocks")
        };
        Ok(Self {
            directory,
            page_cache,
            stop,
            processed,
            first_loss,
            primary_margin_active_sentences,
            tri_loss,
            tri_count,
            missed,
            top1_hits,
            top1_evaluated,
            gold_augmented,
            post_trigram_top1_hits,
            post_trigram_top1_evaluated,
            missed_records,
            previous_seconds: partial["training_update_seconds"]
                .as_f64()
                .ok_or("missing prior update time")?,
            batches: integer(&partial, "training_batches")?,
            expected_fingerprint,
        })
    }

    pub fn validate_model(&self, actual: &Fingerprint) -> Result<()> {
        if serde_json::to_value(actual)? != self.expected_fingerprint {
            return Err("resume checkpoint does not match the saved stop fingerprint".into());
        }
        Ok(())
    }

    pub fn samples(&self) -> Result<(Vec<Sentence>, Vec<Sentence>, InputReport)> {
        let train: Vec<Sentence> = read(&self.directory.join("training-probe.json"))?;
        let dev: Vec<Sentence> = read(&self.directory.join("test-sample.json"))?;
        if train.is_empty() || dev.is_empty() {
            return Err("empty resume evaluation sample".into());
        }
        Ok((train, dev, read(&self.directory.join("input-report.json"))?))
    }

    pub fn position(&self, blocks: &blocks::PageBlocks) -> Result<usize> {
        let mut order = self.stop.page_block_order.clone();
        order.sort_unstable();
        if order != (0..blocks.blocks.len()).collect::<Vec<_>>() {
            return Err("invalid resume block permutation".into());
        }
        self.stop
            .page_block_order
            .iter()
            .position(|&n| Some(n) == self.stop.active_page_block)
            .ok_or_else(|| "resume block not in saved permutation".into())
    }

    pub fn validate_prefix(
        &self,
        blocks: &blocks::PageBlocks,
        missed_input: Option<&[MissedRecord]>,
    ) -> Result<()> {
        let position = self.position(blocks)?;
        let block = self.stop.page_block_order[position];
        let mut pending = BTreeMap::new();
        for entry in &self.stop.pending_window {
            if entry.record > self.stop.source_records_read
                || pending.insert(entry.record, entry).is_some()
            {
                return Err("invalid duplicate/out-of-range pending resume record".into());
            }
        }
        let mut reader = blocks.open(block)?;
        let mut line = String::new();
        let mut prefix = 0_usize;
        let mut last = None;
        while let Some(entry) = blocks::read_cached(&mut *reader, &mut line)? {
            if entry.record > self.stop.source_records_read {
                break;
            }
            last = Some(entry.record);
            if let Some(misses) = missed_input {
                if let Ok(miss_index) =
                    misses.binary_search_by_key(&entry.record, |miss| miss.record_id)
                {
                    if misses[miss_index].page_id != entry.sentence.page_id {
                        return Err(
                            "missed resume record page ID differs from original cache".into()
                        );
                    }
                    prefix += 1;
                }
            } else {
                prefix += 1;
            }
            if let Some(expected) = pending.remove(&entry.record) {
                if serde_json::to_value(&entry)? != serde_json::to_value(expected)? {
                    return Err("pending resume sentence differs from original cache".into());
                }
            }
        }
        let completed = if let Some(misses) = missed_input {
            self.stop.page_block_order[..position]
                .iter()
                .try_fold(0_usize, |n, &b| {
                    let info = &blocks.blocks[b];
                    let count = misses
                        .iter()
                        .filter(|miss| {
                            (info.source_first_record..=info.source_last_record)
                                .contains(&miss.record_id)
                        })
                        .count();
                    n.checked_add(count).ok_or("resume sentence total overflow")
                })?
        } else {
            self.stop.page_block_order[..position]
                .iter()
                .try_fold(0_usize, |n, &b| {
                    n.checked_add(blocks.blocks[b].sentences)
                        .ok_or("resume sentence total overflow")
                })?
        };
        if !pending.is_empty()
            || last != Some(self.stop.source_records_read)
            || completed
                .checked_add(prefix)
                .and_then(|n| n.checked_sub(self.stop.pending_window.len()))
                != Some(self.processed)
        {
            return Err("resume cache cursor/committed count mismatch".into());
        }
        Ok(())
    }

    pub fn restore(&self, pass: &mut Pass, state: &mut u64) {
        pass.processed = self.processed;
        pass.selected = self.processed + self.stop.pending_window.len();
        pass.batches = self.batches;
        pass.first_loss = self.first_loss;
        pass.primary_nbest_margin_active_sentences = self.primary_margin_active_sentences;
        pass.tri_loss = self.tri_loss;
        pass.tri_count = self.tri_count;
        pass.missed = self.missed;
        pass.top1_hits = self.top1_hits;
        pass.top1_evaluated = self.top1_evaluated;
        pass.post_trigram_top1_hits = self.post_trigram_top1_hits;
        pass.post_trigram_top1_evaluated = self.post_trigram_top1_evaluated;
        pass.missed_records.clone_from(&self.missed_records);
        pass.gold_augmented = self.gold_augmented;
        pass.block_order.clone_from(&self.stop.page_block_order);
        *state = self.stop.shuffle_state;
        eprintln!(
            "resuming epoch={} committed_sentences={} batches={} pending={} active_block={:?}",
            self.stop.epoch,
            self.processed,
            self.batches,
            self.stop.pending_window.len(),
            self.stop.active_page_block
        );
    }
}
