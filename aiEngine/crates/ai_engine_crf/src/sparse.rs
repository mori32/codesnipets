//! Compact hash tables for correction weights; optional cutoff is explicit.
use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::{BuildHasherDefault, Hash, Hasher};
use std::marker::PhantomData;

use crate::{Feature, PosTriple, Triple};

pub const MAX_ID: u32 = 0x00ff_ffff;

#[derive(Clone, Debug, Default)]
pub struct IntegerHasher(u64);
impl Hasher for IntegerHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for chunk in bytes.chunks(8) {
            let mut value = [0; 8];
            value[..chunk.len()].copy_from_slice(chunk);
            self.write_u64(u64::from_le_bytes(value));
        }
    }
    fn write_u64(&mut self, value: u64) {
        let mut mixed = value
            .wrapping_add(0x9e37_79b9_7f4a_7c15)
            .wrapping_add(self.0);
        mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        self.0 = mixed ^ (mixed >> 31);
    }
    fn write_u32(&mut self, value: u32) {
        self.write_u64(u64::from(value));
    }
}

pub trait Key: Copy + Ord + Debug {
    type Packed: Copy + Eq + Ord + Hash + Debug;
    fn pack(self) -> Self::Packed;
    fn unpack(value: Self::Packed) -> Self;
}

impl Key for Feature {
    type Packed = u64;
    fn pack(self) -> u64 {
        let (tag, a, b): (u64, u32, u32) = match self {
            Self::Word(w) => (1, 0, w),
            Self::Pos(a, b) => (2, a, b),
            Self::Bigram(a, b) => (3, a, b),
            Self::Script(s) => (4, 0, u32::from(s)),
            Self::ScriptBigram(a, b) => (5, u32::from(a), u32::from(b)),
            Self::ReadingLengthBigram(a, b) => (6, u32::from(a), u32::from(b)),
            Self::PosWord(a, b) => (7, a, b),
            Self::WordPos(a, b) => (8, a, b),
        };
        assert!(
            a <= MAX_ID && b <= MAX_ID,
            "training ID exceeds 24-bit key capacity"
        );
        (tag << 56) | (u64::from(a) << 24) | u64::from(b)
    }
    fn unpack(value: u64) -> Self {
        let a = u32::try_from((value >> 24) & u64::from(MAX_ID)).expect("24-bit ID");
        let b = u32::try_from(value & u64::from(MAX_ID)).expect("24-bit ID");
        match value >> 56 {
            1 => Self::Word(b),
            2 => Self::Pos(a, b),
            3 => Self::Bigram(a, b),
            4 => Self::Script(u8::try_from(b).expect("script ID")),
            5 => Self::ScriptBigram(
                u8::try_from(a).expect("script ID"),
                u8::try_from(b).expect("script ID"),
            ),
            6 => Self::ReadingLengthBigram(
                u8::try_from(a).expect("reading-length ID"),
                u8::try_from(b).expect("reading-length ID"),
            ),
            7 => Self::PosWord(a, b),
            8 => Self::WordPos(a, b),
            _ => unreachable!("only packed feature keys are stored"),
        }
    }
}

impl Key for Triple {
    type Packed = Triple;
    fn pack(self) -> Triple {
        self
    }
    fn unpack(value: Triple) -> Triple {
        value
    }
}

impl Key for [u8; 3] {
    type Packed = u32;
    fn pack(self) -> u32 {
        assert!(self.into_iter().all(|length| length < 9));
        (u32::from(self[0]) << 16) | (u32::from(self[1]) << 8) | u32::from(self[2])
    }
    fn unpack(value: u32) -> Self {
        let result = [
            u8::try_from((value >> 16) & 0xff).expect("reading length"),
            u8::try_from((value >> 8) & 0xff).expect("reading length"),
            u8::try_from(value & 0xff).expect("reading length"),
        ];
        assert!(result.into_iter().all(|length| length < 9));
        result
    }
}

impl Key for PosTriple {
    type Packed = PosTriple;
    fn pack(self) -> PosTriple {
        self
    }
    fn unpack(value: PosTriple) -> Self {
        value
    }
}

#[derive(Clone, Debug)]
pub struct SparseMap<K: Key> {
    table: HashMap<K::Packed, f64, BuildHasherDefault<IntegerHasher>>,
    marker: PhantomData<K>,
}
impl<K: Key> Default for SparseMap<K> {
    fn default() -> Self {
        Self {
            table: HashMap::default(),
            marker: PhantomData,
        }
    }
}
impl<K: Key> PartialEq for SparseMap<K> {
    fn eq(&self, other: &Self) -> bool {
        self.table == other.table
    }
}
impl<K: Key> FromIterator<(K, f64)> for SparseMap<K> {
    fn from_iter<T: IntoIterator<Item = (K, f64)>>(values: T) -> Self {
        let mut result = Self::default();
        for (key, value) in values {
            result.insert(key, value);
        }
        result
    }
}
impl<K: Key> SparseMap<K> {
    #[must_use]
    pub fn len(&self) -> usize {
        self.table.len()
    }
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.table.capacity()
    }
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.table.is_empty()
    }
    pub fn get(&self, key: &K) -> Option<&f64> {
        self.table.get(&key.pack())
    }
    pub fn contains_key(&self, key: &K) -> bool {
        self.table.contains_key(&key.pack())
    }
    pub fn insert(&mut self, key: K, value: f64) -> Option<f64> {
        self.table.insert(key.pack(), value)
    }
    pub fn remove(&mut self, key: &K) -> Option<f64> {
        self.table.remove(&key.pack())
    }
    pub fn entry_value(&mut self, key: K) -> &mut f64 {
        self.table.entry(key.pack()).or_default()
    }
    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut f64> {
        self.table.values_mut()
    }
    pub fn keys(&self) -> impl Iterator<Item = K> + '_ {
        self.table.keys().copied().map(K::unpack)
    }
    pub fn iter(&self) -> impl Iterator<Item = (K, f64)> + '_ {
        self.table.iter().map(|(&k, &v)| (K::unpack(k), v))
    }
    /// Iterate without allocating or sorting a second copy of the sparse table.
    pub fn iter_raw(&self) -> impl Iterator<Item = (K::Packed, f64)> + '_ {
        self.table.iter().map(|(&key, &value)| (key, value))
    }
    #[must_use]
    pub fn sorted_raw(&self) -> Vec<(K::Packed, f64)> {
        let mut result = self.table.iter().map(|(&k, &v)| (k, v)).collect::<Vec<_>>();
        result.sort_unstable_by_key(|p| p.0);
        result
    }
    pub fn reserve(&mut self, additional: usize) -> crate::Result<()> {
        self.table.try_reserve(additional)?;
        Ok(())
    }
    pub fn retain(&mut self, mut keep: impl FnMut(K, f64) -> bool) {
        self.table
            .retain(|&key, value| keep(K::unpack(key), *value));
    }
    /// Release unused buckets without a second full list of entries. Allocation
    /// is fallible; the retained weights remain intact if allocation fails.
    pub fn compact(&mut self) -> crate::Result<()> {
        if Self::minimum_bytes(self.len()) >= self.projected_bytes(0) {
            return Ok(());
        }
        let mut compact = HashMap::default();
        compact.try_reserve(self.len())?;
        compact.extend(self.table.iter().map(|(&key, &value)| (key, value)));
        self.table = compact;
        Ok(())
    }
    /// Bucket/control estimate for a fresh table containing this many entries.
    #[must_use]
    pub fn minimum_bytes(entries: usize) -> usize {
        if entries == 0 {
            return 0;
        }
        let buckets = entries
            .saturating_mul(8)
            .div_ceil(7)
            .max(4)
            .checked_next_power_of_two()
            .unwrap_or(usize::MAX);
        buckets
            .saturating_mul(std::mem::size_of::<(K::Packed, f64)>() + 1)
            .saturating_add(16)
    }
    /// Estimated allocated bucket/control bytes, not process RSS/allocator overhead.
    #[must_use]
    pub fn projected_bytes(&self, additional: usize) -> usize {
        let desired = self.len().saturating_add(additional);
        let buckets = if desired <= self.table.capacity() {
            if self.table.capacity() == 0 {
                0
            } else {
                self.table.capacity().next_power_of_two()
            }
        } else {
            desired
                .saturating_mul(8)
                .div_ceil(7)
                .max(4)
                .checked_next_power_of_two()
                .unwrap_or(usize::MAX)
        };
        buckets
            .saturating_mul(std::mem::size_of::<(K::Packed, f64)>() + 1)
            .saturating_add(if buckets == 0 { 0 } else { 16 })
    }
    /// Order-independent diagnostic of all raw key/value bits (not cryptographic).
    #[must_use]
    pub fn fingerprint(&self) -> (u64, u64) {
        let mut xor = 0;
        let mut sum: u64 = 0;
        for (&key, &value) in &self.table {
            let mut h = IntegerHasher::default();
            key.hash(&mut h);
            value.to_bits().hash(&mut h);
            xor ^= h.finish();
            sum = sum.wrapping_add(h.finish());
        }
        (xor, sum)
    }
}
