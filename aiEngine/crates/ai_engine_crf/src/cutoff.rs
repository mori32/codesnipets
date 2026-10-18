//! Explicit magnitude cutoff of learned deltas, not initial statistical scores.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::time::Instant;

use serde::Serialize;

use crate::sparse::{Key, SparseMap};
use crate::{Feature, Model, Parameters, ReadingLengthTriple, Result, Triple};

#[derive(Clone, Copy, Debug, Serialize)]
pub struct Policy {
    pub minimum_delta: f64,
    pub maximum_delta: f64,
    /// Target retained-table size; room for the retry's new keys is still needed.
    pub target_fraction: f64,
}
impl Policy {
    pub fn validate(self) -> Result<()> {
        if !self.minimum_delta.is_finite()
            || self.minimum_delta <= 0.0
            || !self.maximum_delta.is_finite()
            || self.maximum_delta < self.minimum_delta
            || !self.target_fraction.is_finite()
            || self.target_fraction <= 0.0
            || self.target_fraction >= 1.0
        {
            return Err("invalid delta cutoff policy".into());
        }
        Ok(())
    }
    fn thresholds(self) -> Vec<f64> {
        let mut thresholds = vec![self.minimum_delta];
        while *thresholds.last().unwrap() < self.maximum_delta {
            thresholds.push((thresholds.last().unwrap() * 10.0).min(self.maximum_delta));
        }
        thresholds
    }
}

#[derive(Debug, Default, Serialize)]
pub struct Family {
    pub entries_before: usize,
    pub entries_after: usize,
    pub removed: usize,
    pub maximum_removed_abs_delta: f64,
    pub sum_removed_abs_delta: f64,
}

#[derive(Debug, Serialize)]
pub struct Report {
    pub policy: Policy,
    pub selected_delta: f64,
    pub table_bytes_before: usize,
    pub table_bytes_after: usize,
    pub retained_minimum_table_bytes: usize,
    pub target_table_bytes: usize,
    pub target_reached: bool,
    pub retry_capacity_reached: bool,
    pub removed_total: usize,
    pub families: BTreeMap<String, Family>,
    pub selection_seconds: f64,
    pub retain_seconds: f64,
    pub total_seconds: f64,
}

pub const WORD_BIGRAM_COUNT_TRIGGER: usize = 50_000_000;
pub const WORD_TRIGRAM_COUNT_TRIGGER: usize = 100_000_000;

#[derive(Debug, Serialize)]
pub struct CountCapEvent {
    pub family: String,
    pub trigger_entries: usize,
    pub entries_before: usize,
    pub entries_after: usize,
    pub selected_minimum_abs_delta: f64,
    pub temporary_selection_bytes: usize,
}

fn retain_largest<K: Key>(
    table: &mut SparseMap<K>,
    scale: f64,
    family: &str,
    trigger: usize,
    keep: usize,
    selected: impl Fn(K) -> bool,
) -> Option<CountCapEvent> {
    let entries_before = table.iter().filter(|(key, _)| selected(*key)).count();
    if entries_before <= trigger || keep == 0 || keep >= entries_before {
        return None;
    }
    let mut ranked = table
        .iter_raw()
        .filter(|(packed, _)| selected(K::unpack(*packed)))
        .map(|(packed, raw)| ((raw * scale).abs(), packed))
        .collect::<Vec<_>>();
    let rank = |a: &(f64, K::Packed), b: &(f64, K::Packed)| {
        b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1))
    };
    let (_, threshold, _) = ranked.select_nth_unstable_by(keep - 1, rank);
    let threshold = *threshold;
    let temporary_selection_bytes = ranked.capacity() * std::mem::size_of::<(f64, K::Packed)>();
    drop(ranked);
    table.retain(|key, raw| {
        if !selected(key) {
            return true;
        }
        rank(&((raw * scale).abs(), key.pack()), &threshold) != Ordering::Greater
    });
    let entries_after = table.iter().filter(|(key, _)| selected(*key)).count();
    Some(CountCapEvent {
        family: family.to_owned(),
        trigger_entries: trigger,
        entries_before,
        entries_after,
        selected_minimum_abs_delta: threshold.0,
        temporary_selection_bytes,
    })
}

/// Enforce generous family-specific count caps before large sparse tables
/// dominate memory locality. When a family exceeds its trigger, retain its
/// largest-magnitude half; selection is exact and ties use packed-key order.
pub fn apply_count_caps(model: &mut Model) -> Vec<CountCapEvent> {
    let mut events = Vec::new();
    if let Some(event) = retain_largest(
        &mut model.first.values,
        model.first.scale,
        "word-bigram",
        WORD_BIGRAM_COUNT_TRIGGER,
        WORD_BIGRAM_COUNT_TRIGGER / 2,
        |feature| matches!(feature, Feature::Bigram(..)),
    ) {
        events.push(event);
    }
    if let Some(event) = retain_largest(
        &mut model.third.values,
        model.third.scale,
        "word-trigram",
        WORD_TRIGRAM_COUNT_TRIGGER,
        WORD_TRIGRAM_COUNT_TRIGGER / 2,
        |_| true,
    ) {
        events.push(event);
    }
    events
}

fn family(feature: Feature) -> usize {
    match feature {
        Feature::Word(_) => 0,
        Feature::Pos(..) => 1,
        Feature::Bigram(..) => 2,
        Feature::PosWord(..) => 3,
        Feature::WordPos(..) => 4,
        Feature::Script(_) => 5,
        Feature::ScriptBigram(..) => 6,
        Feature::ReadingLengthBigram(..) => 7,
    }
}

fn histogram<K: Key>(parameters: &Parameters<K>, thresholds: &[f64]) -> Result<Vec<usize>> {
    if !parameters.scale.is_finite() || parameters.scale <= 0.0 {
        return Err("invalid parameter scale for delta cutoff".into());
    }
    let mut bands = vec![0_usize; thresholds.len() + 1];
    for (_, raw) in parameters.values.iter() {
        let abs_delta = (raw * parameters.scale).abs();
        if !abs_delta.is_finite() {
            return Err("nonfinite cutoff delta".into());
        }
        bands[thresholds.partition_point(|&t| abs_delta >= t)] += 1;
    }
    Ok(bands)
}

fn retry_fits(model: &Model, kept: [usize; 4], additional: [usize; 4], budget: usize) -> bool {
    let maps = [
        (
            model.first.values.capacity(),
            model.first.values.projected_bytes(0),
        ),
        (
            model.third.values.capacity(),
            model.third.values.projected_bytes(0),
        ),
        (
            model.third_pos.values.capacity(),
            model.third_pos.values.projected_bytes(0),
        ),
        (
            model.third_reading_length.values.capacity(),
            model.third_reading_length.values.projected_bytes(0),
        ),
    ];
    let needed_entries = [
        kept[0].saturating_add(additional[0]),
        kept[1].saturating_add(additional[1]),
        kept[2].saturating_add(additional[2]),
        kept[3].saturating_add(additional[3]),
    ];
    let needed_bytes = [
        SparseMap::<Feature>::minimum_bytes(needed_entries[0]),
        SparseMap::<Triple>::minimum_bytes(needed_entries[1]),
        SparseMap::<Triple>::minimum_bytes(needed_entries[2]),
        SparseMap::<ReadingLengthTriple>::minimum_bytes(needed_entries[3]),
    ];
    maps.iter()
        .zip(needed_entries)
        .all(|(&(capacity, _), needed)| capacity == 0 || needed <= capacity)
        && maps
            .iter()
            .zip(needed_bytes)
            .map(|(&(capacity, bytes), needed)| if capacity == 0 { needed } else { bytes })
            .sum::<usize>()
            <= budget
}

/// Two linear scans and at most one table-sized allocation at a time. Histogram
/// bands choose the smallest configured threshold meeting the capacity target.
/// Surviving raw bits/scales and all seed metadata remain unchanged. Absent
/// deltas become zero and may be learned again by a later gradient.
pub fn apply(model: &mut Model, policy: Policy, budget: usize) -> Result<Report> {
    apply_with_reservation(model, policy, budget, [0; 4], false)
}

/// Cut weak deltas while preserving allocated buckets so the pending batch can
/// reuse the freed slots rather than allocating replacement hash tables.
pub fn apply_for_retry(
    model: &mut Model,
    policy: Policy,
    budget: usize,
    additional: [usize; 4],
) -> Result<Report> {
    apply_with_reservation(model, policy, budget, additional, true)
}

#[allow(clippy::too_many_lines)]
fn apply_with_reservation(
    model: &mut Model,
    policy: Policy,
    budget: usize,
    additional: [usize; 4],
    reuse_buckets: bool,
) -> Result<Report> {
    policy.validate()?;
    let started = Instant::now();
    let thresholds = policy.thresholds();
    let bands = [
        histogram(&model.first, &thresholds)?,
        histogram(&model.third, &thresholds)?,
        histogram(&model.third_pos, &thresholds)?,
        histogram(&model.third_reading_length, &thresholds)?,
    ];
    let before = model.first.values.projected_bytes(0)
        + model.third.values.projected_bytes(0)
        + model.third_pos.values.projected_bytes(0)
        + model.third_reading_length.values.projected_bytes(0);
    // A loaded model already over budget must be physically compacted before
    // retry. Normal full-corpus pressure happens before growth commits, so its
    // current allocation is within budget and can reuse its buckets in place.
    let reuse_buckets = reuse_buckets && before <= budget;
    let target = (budget as f64 * policy.target_fraction) as usize;
    let mut removed = [0; 4];
    let mut selected = *thresholds.last().unwrap();
    for (i, &threshold) in thresholds.iter().enumerate() {
        for family in 0..4 {
            removed[family] += bands[family][i];
        }
        let kept = [
            model.first.values.len() - removed[0],
            model.third.values.len() - removed[1],
            model.third_pos.values.len() - removed[2],
            model.third_reading_length.values.len() - removed[3],
        ];
        let minimum = SparseMap::<Feature>::minimum_bytes(kept[0])
            + SparseMap::<Triple>::minimum_bytes(kept[1])
            + SparseMap::<Triple>::minimum_bytes(kept[2])
            + SparseMap::<ReadingLengthTriple>::minimum_bytes(kept[3]);
        let fits = if reuse_buckets {
            retry_fits(model, kept, additional, budget)
        } else {
            SparseMap::<Feature>::minimum_bytes(kept[0].saturating_add(additional[0]))
                + SparseMap::<Triple>::minimum_bytes(kept[1].saturating_add(additional[1]))
                + SparseMap::<Triple>::minimum_bytes(kept[2].saturating_add(additional[2]))
                + SparseMap::<ReadingLengthTriple>::minimum_bytes(
                    kept[3].saturating_add(additional[3]),
                )
                <= budget
        };
        if minimum <= target && fits {
            selected = threshold;
            break;
        }
    }
    let selection_seconds = started.elapsed().as_secs_f64();
    eprintln!(
        "delta_cutoff selected={selected:e} table_bytes_before={before} target_bytes={target}"
    );
    let mut stats: [Family; 11] = std::array::from_fn(|_| Family::default());
    let mut keep = |index: usize, delta: f64| {
        let stat = &mut stats[index];
        stat.entries_before += 1;
        if delta.abs() < selected {
            stat.removed += 1;
            stat.maximum_removed_abs_delta = stat.maximum_removed_abs_delta.max(delta.abs());
            stat.sum_removed_abs_delta += delta.abs();
            false
        } else {
            stat.entries_after += 1;
            true
        }
    };
    model
        .first
        .values
        .retain(|key, raw| keep(family(key), raw * model.first.scale));
    model
        .third
        .values
        .retain(|_, raw| keep(8, raw * model.third.scale));
    model
        .third_pos
        .values
        .retain(|_, raw| keep(9, raw * model.third_pos.scale));
    model
        .third_reading_length
        .values
        .retain(|_, raw| keep(10, raw * model.third_reading_length.scale));
    if !reuse_buckets {
        // The public explicit compaction path is retained for callers/tests that
        // need a physically smaller table. Full-corpus pressure recovery uses
        // apply_for_retry, which reuses existing capacity and avoids a second
        // table-sized allocation.
        let first_release =
            model.first.values.projected_bytes(0).saturating_sub(
                SparseMap::<Feature>::minimum_bytes(model.first.values.len()),
            );
        let third_release = model
            .third
            .values
            .projected_bytes(0)
            .saturating_sub(SparseMap::<Triple>::minimum_bytes(model.third.values.len()));
        if first_release >= third_release {
            model.first.values.compact()?;
            model.third.values.compact()?;
        } else {
            model.third.values.compact()?;
            model.first.values.compact()?;
        }
        model.third_pos.values.compact()?;
        model.third_reading_length.values.compact()?;
    }
    let after = model.first.values.projected_bytes(0)
        + model.third.values.projected_bytes(0)
        + model.third_pos.values.projected_bytes(0)
        + model.third_reading_length.values.projected_bytes(0);
    let retained_minimum = SparseMap::<Feature>::minimum_bytes(model.first.values.len())
        + SparseMap::<Triple>::minimum_bytes(model.third.values.len())
        + SparseMap::<Triple>::minimum_bytes(model.third_pos.values.len())
        + SparseMap::<ReadingLengthTriple>::minimum_bytes(model.third_reading_length.values.len());
    let retry_capacity_reached =
        if reuse_buckets {
            retry_fits(
                model,
                [
                    model.first.values.len(),
                    model.third.values.len(),
                    model.third_pos.values.len(),
                    model.third_reading_length.values.len(),
                ],
                additional,
                budget,
            )
        } else {
            SparseMap::<Feature>::minimum_bytes(
                model.first.values.len().saturating_add(additional[0]),
            ) + SparseMap::<Triple>::minimum_bytes(
                model.third.values.len().saturating_add(additional[1]),
            ) + SparseMap::<Triple>::minimum_bytes(
                model.third_pos.values.len().saturating_add(additional[2]),
            ) + SparseMap::<ReadingLengthTriple>::minimum_bytes(
                model
                    .third_reading_length
                    .values
                    .len()
                    .saturating_add(additional[3]),
            ) <= budget
        };
    let total_seconds = started.elapsed().as_secs_f64();
    Ok(Report {
        policy,
        selected_delta: selected,
        table_bytes_before: before,
        table_bytes_after: after,
        retained_minimum_table_bytes: retained_minimum,
        target_table_bytes: target,
        target_reached: retained_minimum <= target,
        retry_capacity_reached,
        removed_total: stats.iter().map(|s| s.removed).sum(),
        families: [
            "word",
            "pos-bigram",
            "word-bigram",
            "pos-word-bigram",
            "word-pos-bigram",
            "surface-unigram",
            "surface-bigram",
            "reading-length-bigram",
            "word-trigram",
            "pos-trigram",
            "reading-length-trigram",
        ]
        .into_iter()
        .zip(stats)
        .map(|(name, stat)| (name.to_owned(), stat))
        .collect(),
        selection_seconds,
        retain_seconds: total_seconds - selection_seconds,
        total_seconds,
    })
}

#[cfg(test)]
mod count_cap_tests {
    use super::retain_largest;
    use crate::Feature;
    use crate::sparse::SparseMap;

    #[test]
    fn count_cap_retains_largest_family_entries_without_pruning_other_features() {
        let mut table = SparseMap::default();
        for id in 0..8 {
            table.insert(Feature::Bigram(id, 99), f64::from(id));
        }
        table.insert(Feature::PosWord(8, 3), 0.001);
        let report = retain_largest(&mut table, 1.0, "word-bigram", 6, 3, |feature| {
            matches!(feature, Feature::Bigram(..))
        })
        .expect("the trigger was exceeded");
        assert_eq!(report.entries_before, 8);
        assert_eq!(report.entries_after, 3);
        assert!(table.get(&Feature::PosWord(8, 3)).is_some());
        for id in 5..8 {
            assert!(table.get(&Feature::Bigram(id, 99)).is_some());
        }
        for id in 0..5 {
            assert!(table.get(&Feature::Bigram(id, 99)).is_none());
        }
    }
}
