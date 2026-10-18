# aiEngine

Rust で実装する、ローカル LLM を候補評価器として活用するかな漢字変換エンジンです。

現在のワークスペースには、変換エンジン・LLM連携・辞書コンパイラ・CRF学習と評価のソースを残しています。コーパスの取得・加工・補正・集計に限定したツールと対応するテスト・作業資料は、今回の統合対象から外しています。学習用のコーパス・追加語データはローカル入力として扱い、Gitの統合対象から外しています。

設計方針は [docs/02-design.md](docs/02-design.md) を参照してください。

大容量データの配置、作業用フォルダの `README.md`、およびリポジトリ内に置ける生成物の例外は [docs/18-workspace-layout.md](docs/18-workspace-layout.md) を参照してください。

移動できない既存データとして固定許可されているリポジトリ内の場所は `corpus\wiki-ja\chunked`、`corpus\wiki-ja\clean-split`、`corpus\wiki-ja\tagged-mecab-initial`、`dictionary\additions`、`dictionary\build`、`dictionary\candidates` のみです。これら以外のデータ置き場やrun履歴をリポジトリ内に増やさず、新規の中間データ・成果物は外部作業フォルダへ置きます。

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

`ai_engine_llm` は、モデルランタイムに依存しない `ContextScorer` を提供します。`NoopScorer` と `FixedScorer` で古典候補の再順位付け、同期・非同期パイプライン、世代管理を検証でき、`llama-cpp` feature では GGUF-backed `LlamaCppScorer` を接続できます。

```rust
use ai_engine_llm::{ConversionPipeline, FixedScorer};

let scorer = FixedScorer::new(vec![0.0]);
let pipeline = ConversionPipeline::with_default_engine(scorer, 1.0);
let ranked = pipeline.convert("にほんごへんかん", "前の文脈", 1)?;
```

`AsyncScorer` は入力ごとに世代番号を発行します。`AsyncConversionPipeline` は古典候補を入力更新時に即時作成し、最新世代だけの評価結果を再順位付けへ適用します。新しい入力を受け取ったら古い推論結果は破棄します。次の段階で、この境界に小型ローカルモデルの同一深度 log-prob 評価を接続します。

### llama.cpp の実モデル評価

`llama-cpp` feature を有効にすると、GGUF モデルを C++ ブリッジ経由でロードし、全候補の最短 token 深度までに揃えた log-probability の総和を `ContextScorer` として取得できます。閉じた文・分節の比較（`--sentence`）では、最短深度で打ち切らず、全 token + EOS を length penalty なしで合算します。llama.cpp はリポジトリへ含めず、`third_party/` 以下へセットアップスクリプトで取得・ビルドします。

Qiita のリランキング例は、期待値付きの [examples/qiita-reranker-cases.tsv](examples/qiita-reranker-cases.tsv) に再現可能なテストケースとして収録しています。

```powershell
cargo run -p ai_engine_eval --features llama-cpp -- `
  --model tests\fixtures\GGUF\Qwen2.5-1.5B-Instruct-Q8_0.gguf `
  --sentence-cases examples\qiita-reranker-cases.tsv
```

EOS を付けない open 終端・共通最短 token 深度の比較は、`--sentence-cases-open` を使います。

Greedy next-word 生成のケースは [examples/next-word-cases.tsv](examples/next-word-cases.tsv:1) にあります。

```powershell
cargo run -p ai_engine_eval --features llama-cpp -- `
  --model tests\fixtures\GGUF\qwen2.5-0.5b-instruct-q8_0.gguf `
  --generation-cases examples\next-word-cases.tsv
```

英数字や Markdown 記号を候補から除外し、かな・漢字・日本語句読点に制限するには
`--japanese-only` を追加します。これは意味や文法を制限するものではなく、次に選べる
token piece の文字種だけを制限します。

```powershell
cargo run -p ai_engine_eval --features llama-cpp -- `
  --model tests\fixtures\GGUF\qwen2.5-0.5b-instruct-q8_0.gguf `
  --generation-cases examples\next-word-cases.tsv `
  --japanese-only
```

長期のユーザー確定履歴、入力ヒストリ、そして現在の文脈による上書きを比較する
`learning-cases` もあります。各行は `group<TAB>context<TAB>candidate<TAB>expected` です。

```powershell
cargo run -p ai_engine_eval --features llama-cpp -- `
  --model tests\fixtures\GGUF\Qwen2.5-1.5B-Instruct-Q8_0.gguf `
  --learning-cases examples\learning-cases.tsv
```

テスト用の Qwen2.5-0.5B-Instruct Q8_0 は、明示的なセットアップスクリプトで `tests/fixtures/GGUF/` へ取得できます。モデルは大きいため、通常の Cargo ビルドではダウンロードしません。

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup-gguf.ps1
```

```powershell
# llama.cpp のソース取得、CMake ビルド、インストールを行う
powershell -ExecutionPolicy Bypass -File scripts\setup-llama-cpp.ps1

# 生成されたローカル環境変数を現在の PowerShell に読み込む
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
. .\third_party\llama.cpp-env.ps1

cargo run -p ai_engine_cli --features llama-cpp -- `
  --model tests\fixtures\GGUF\qwen2.5-0.5b-instruct-q8_0.gguf --context "庭で犬を" かう `
  --dictionary examples\context-words.tsv --candidates 2
```

セットアップスクリプトは `vswhere.exe` で Visual Studio の MSVC を検出し、`VsDevCmd.bat` の x64 開発環境を自動的に初期化します。通常の PowerShell の `PATH` に `cl.exe` がなくても、Visual Studio と C++ workload がインストール済みなら実行できます。
既定の CMake generator は `Ninja`（MSVC をコンパイラとして使用）で、GitHub Actions とローカルのビルド経路を一致させています。別の generator が必要な場合だけ `-Generator` で上書きします。

同じモデルで `--context "店で犬を"` と比較できます。モデルが利用できない場合は、従来どおり古典候補だけで起動できます。ネイティブ bridge のビルドは `ai_engine_llm` の `llama-cpp` feature を有効にした場合だけ行われます。

### 評価機

`ai_engine_eval` は、同じ辞書・候補生成経路に対して classical ranking と GGUF model ranking を同じ評価データで比較します。top-1 accuracy、MRR、平均/p50/p95 latency を表示し、`--report` を指定するとケースごとの順位を TSV に保存します。

```powershell
# 古典辞書だけの基準値
cargo run -p ai_engine_eval -- `
  --cases examples\context-cases.tsv `
  --dictionary examples\context-words.tsv `
  --candidates 2 `
  --report target\classical-eval.tsv

# llama.cpp + GGUF の実モデル評価
cargo run -p ai_engine_eval --features llama-cpp -- `
  --cases examples\context-cases.tsv `
  --dictionary examples\context-words.tsv `
  --model tests\fixtures\GGUF\qwen2.5-0.5b-instruct-q8_0.gguf `
  --candidates 2 `
  --score-weight 1.0 `
  --report target\qwen-eval.tsv
```

データセットの形式は `context<TAB>reading<TAB>expected surface` です。評価結果は候補を対象アプリへ commit せず、モデル・重み・辞書の比較だけに使用します。

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
