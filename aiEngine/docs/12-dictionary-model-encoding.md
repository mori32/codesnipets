# Dictionary model encoding notes

規範的な意味仕様は [`05-static-dictionary.md`](05-static-dictionary.md)、実装段階は [`07-dictionary-compiler-dumper.md`](07-dictionary-compiler-dumper.md) を参照。この文書では確率量子化とcompact trigram blockの実装詳細を整理する。

## Log-probability quantization

### Input and code meaning

- input is a positive probability `p`; quantizer input is `ln(p)`.
- Each probability family has its own independent codebook: word unigram within POS, word bigram, bigram backoff, trigram, and POS bigram. Do not share centers across these tables.
- Code `0` is reserved for absent/unused values; ordinary probabilities use the remaining 8-bit indices `1..255` (up to 255 learned log centers per table).
- codebook entries are the representative log values used by decoder reconstruction.
- codebook entries and decision thresholds are sorted strictly ascending. Exact midpoint ties choose the lower code.
- nonfinite inputs, `p <= 0`, invalid codebook lengths, and code 0 used as an ordinary value are rejected.

### K-means and empty/colliding clusters

Log probabilities can be highly dispersed, so a bare Lloyd iteration is insufficient: several initial centers can converge to the same populated region while other clusters remain empty. The compiler must explicitly track occupied clusters, empty slots, and collisions after converting representatives to the on-disk float precision.

Use sorted input and deterministic initialization. During each iteration:

1. Assign each input to its nearest current representative; ties resolve to the lower code.
2. Recompute each occupied center as the mean of its assigned log values.
3. If fewer than the requested number of clusters are occupied, fill the vacant centers instead of leaving duplicate array slots. Prefer a populated center with available neighboring separation; place a new center a small relative fraction (initial default `1/10,000`) inside an adjacent gap, then sort centers and rebuild boundaries. The inserted empty center must not be mistaken for an observed sample cluster.
4. If adjacent centers still compare equal after float serialization, repeat collision repair using the next available distinct gap. If no representable gap exists, reduce the effective code count and remap assignments; never emit ambiguous duplicate representatives.
5. Stop after the configured iteration limit or when squared center movement falls below the configured threshold. Output must be deterministic for identical input and parameters.

The collision repair preserves ordered search ranges without pretending that an empty cluster has observed probability mass. After final assignment, empty centers may be used as codebook boundaries/representatives only if their decode value is unambiguous and strictly ordered; metadata records the effective code count. Tests include one tight dense region plus a long sparse tail, repeated identical probabilities, fewer distinct inputs than requested levels, float32 collisions, and endpoint clusters.

Quality reports include input count, distinct log-value count, effective levels, occupied levels, maximum/RMS absolute log error, maximum relative probability error, and any merged level count. Compare the compact result against unquantized scores before changing the model.

## Trigram slot mapping

For each bigram `A/B`, trigram continuations `A/B/C` map into the `B/*` bigram row. Let `M` be the number of retained bigrams whose left word is `B`, and `K` the number of retained trigram continuations for `A/B`. The succinct index is used only to locate this `A/B` trigram block from the global word-bigram edge address; it does not encode `B/*` slot positions.

- Dense representation: `M` one-byte quantized slots; absent continuation is code 0.
- Sparse representation: sorted `(slot_index, code)` pairs. Encode each zero-based slot in a fixed-width little-endian integer selected by `M` (1 byte for `M <= 256`, 2 bytes for `M <= 65,536`, otherwise 4 bytes), followed by one quantized byte. Dense costs `M` bytes; sparse costs `K * (slot_width + 1)` bytes. Choose sparse only when strictly smaller; ties use dense. Codec is inferred from the block length and `M`; no per-block header or descriptor is stored.
- The selected payload is the smaller of those two block representations. A dense block costs `M`, not `K`, bytes; therefore total trigram payload is not guaranteed to be bounded by the number of trigram records.
- Same count cutoff is applied to bigram and trigram; validate every retained trigram's `A/B` and `B/C` parents exist.
- The block-address index marks each `A/B` parent in the global sorted word-bigram edge table. A rank directory sampled every 256 bigram edges obtains the parent block ordinal. A second succinct start vector marks each trigram block's first byte in the concatenated block payload; select samples every 256 blocks resolve the block start address. The block end is the next start (or payload end for the last block). These vectors serve only the bigram-address-to-trigram-block lookup.

The trigram probability is `count(A/B/C)/count(A/B)`. No trigram backoff alpha is stored; runtime interpolation with bigram remains a separate scoring operation. Each probability family has a separate codebook, including the word unigram-in-POS, POS bigram, word bigram, word-bigram alpha, and trigram tables. These format-v7 byte layouts are exercised by compiler/dumper round-trip fixtures.

## VOCABULARY prediction records (format v7)

Phrase and short-prefix prediction data live in `VOCABULARY` key groups; there are no separate binary phrase, phrase-reading, or prefix-cache blocks. Each group serializes optional sections in this order:

```text
[TAIL_STRING][PREDICTION_CACHE_HEADER + count + target offsets]
[PREDICTION phrase records...][WORD records...][DISPLAY_REF records...][TERMINATOR]
```

`PREDICTION_CACHE_HEADER` (`0x40`) is present only on one-, two-, or three-hiragana prefix keys. Its `count:u8` is followed by up to five `target_offset:u32` values. Each offset points directly into the same vocabulary record stream. Follow the target and inspect its record tag to determine whether it is a word or a phrase; the cache does not redundantly encode the target kind. The phrase-finding input contains one-word and 2–6-word candidates. Prefix cache targets are selected from that shared candidate set and ordered by unigram probability: raw corpus `P(word)` for a word, phrase `P(w)` for a phrase. The one-word probability is used only while building the cache and is not added to the dictionary; the WORD record continues to store only `P(word|POS)` for conversion.

A phrase record uses tag `0x60 | (word_count - 1)` for 2–6 words, followed by `word_id:u32[word_count]` and `P(w):f64`. Its record byte offset is its phrase ID. Its reading is the owning key group’s reading key, so no phrase-reading block or per-phrase reading leaf is stored. A word record uses tag `0x80`; its byte offset is the word ID. Display references use tag `0xa0 | (count - 1)`. `0xe1` terminates a reading-bearing key group and stores its reading leaf; `0xe0` terminates other groups.

The compiler retains the normal top-100,000 multiword phrase set and additionally stores any phrase records required by prefix caches. Word candidates point directly to their existing WORD records; phrases are stored once and can be referenced from multiple caches. The dumper reconstructs phrase readings from the key group and validates them against the concatenated readings of their constituent word IDs. It resolves each cache target by following the offset, accepting either a word record or a phrase record. `prediction-prefix.tsv` reports the target's stored probability (`P(word|POS)` for words and `P(w)` for phrases); this is distinct from the transient `P(word)` used to order word targets in the cache.
