# aiEngine

Rust で実装する、ローカル LLM を候補評価器として活用するかな漢字変換エンジンです。

設計方針は [docs/02-design.md](docs/02-design.md) を参照してください。

## 開発の準備

Rust stable toolchain を導入後、リポジトリ直下で以下を実行します。

```powershell
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

## Phase 1 の CLI

```powershell
cargo run -p ai_engine_cli -- nihongohenkan
cargo run -p ai_engine_cli -- にほんごへんかん --candidates 5
cargo run -p ai_engine_cli -- toukyou --select 0
```

入力はローマ字・ひらがな・カタカナを受け付けます。候補を選択すると、既定の `user_dictionary.tsv` に学習結果を保存します。別の辞書を使う場合は `--user-dict PATH` を指定してください。

Phase 1 では、依存なしの小さな組み込み辞書、入力正規化、ビーム幅付き Top-N ラティス探索、ユーザー辞書の保存・学習までを実装しています。大規模辞書、LLM 評価器、Windows TSF は後続フェーズです。

## Phase 2 の辞書とベンチマーク

外部辞書は UTF-8 TSV（`reading<TAB>surface<TAB>priority<TAB>frequency`）で指定できます。サンプル辞書は [dictionaries/phase2.tsv](dictionaries/phase2.tsv) にあります。

```powershell
cargo run -p ai_engine_cli -- ぷろとたいぷ --dictionary dictionaries\phase2.tsv
cargo run -p ai_engine_bench --release
```

辞書検索は読みの先頭文字インデックスを使い、候補生成時に全単語を走査しません。ベンチマークは依存なしの簡易測定で、平均・p50・p95 を表示します。辞書サイズを拡大する際は、この測定を基準にメモリマップやダブル配列トライ導入の効果を比較します。

## Phase 3 の評価器境界

`ai_engine_llm` は、モデルランタイムに依存しない `ContextScorer` を提供します。現在は `NoopScorer` と `FixedScorer` を実装し、古典候補の再順位付け、同期・非同期パイプライン、世代管理を検証しています。

```rust
use ai_engine_llm::{ConversionPipeline, FixedScorer};

let scorer = FixedScorer::new(vec![0.0]);
let pipeline = ConversionPipeline::with_default_engine(scorer, 1.0);
let ranked = pipeline.convert("にほんごへんかん", "前の文脈", 1)?;
```

`AsyncScorer` は入力ごとに世代番号を発行します。`AsyncConversionPipeline` は古典候補を入力更新時に即時作成し、最新世代だけの評価結果を再順位付けへ適用します。新しい入力を受け取ったら古い推論結果は破棄します。次の段階で、この境界に小型ローカルモデルの同一深度 log-prob 評価を接続します。

## Phase 5 の暫定 static-dictionary

P5 の試験用として、UTF-8 TSV を依存なしの `.aidict` バイナリへ変換できます。現在は読み込み時にメモリ展開する暫定形式です。

```powershell
cargo run -p ai_engine_dict -- compile dictionaries\phase2.tsv dictionaries\phase5.aidict
cargo run -p ai_engine_cli -- ぷろとたいぷ --static-dictionary dictionaries\phase5.aidict

# Mozc OSS dictionary_oss を使う場合
cargo run -p ai_engine_dict -- compile-mozc path\to\mozc\src\data\dictionary_oss dictionaries\mozc-oss.aidict
cargo run -p ai_engine_cli -- とうきょう --static-dictionary dictionaries\mozc-oss.aidict
```

Mozc importer は単語の `left_id` / `right_id` / `cost` と接続行列を `.aidict` に保存します。`connection.deflate` しかない公式ツリーでは、`build_tools/zlib_util.py` と Python を自動検出して一時展開します。詳細は [docs/05-static-dictionary.md](docs/05-static-dictionary.md) を参照してください。正式版では mmap と double-array trie に置き換えます。
