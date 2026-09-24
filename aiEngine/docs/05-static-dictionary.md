# P5: static-dictionary 設計（暫定実装）

## 1. 目的

P5 は、実行時に変更しない辞書を提供する。P4 のユーザー学習データとは分離し、オフラインで生成した辞書を読み込み専用で利用する。

現段階では P5 の経路を試せることを優先し、依存なしの簡易バイナリ形式を実装している。これは正式な大規模辞書形式ではなく、後から mmap と double-array trie に置き換えるための暫定版である。

## 2. 現在の構成

```text
UTF-8 TSV
  │
  └─ ai_engine_dict compile
       │
       ▼
  .aidict（AIDICT v2）
       │
       └─ StaticDictionary::load
            │ 検証・メモリ展開
            ▼
       ai_engine_core::Dictionary
            │
            └─ 先頭ひらがなインデックスで候補探索
```

`StaticDictionary` はファイルの形式検証を担当し、`Dictionary::load_static_dictionary` が core の検索辞書へ取り込む。静的辞書から読み込んだエントリはユーザー辞書として保存されない。

## 2.1 Mozc importer（暫定）

Mozc OSS の `dictionary_oss` には `dictionary00.txt` などのファイルがあり、各行は次の形式である。

```text
reading<TAB>left_id<TAB>right_id<TAB>cost<TAB>surface
```

`ai_engine_dict compile-mozc` は `dictionary00.txt` から `dictionary99.txt` を名前順に読み込み、`(reading, surface, left_id, right_id)` が同じ行だけを重複排除して `.aidict` を生成する。同一文脈の重複時は Mozc cost が最小の行を残す。文脈 ID が異なる同一表記は、接続候補として別エントリのまま保存する。

現在の暫定版では `left_id` / `right_id` / `cost` と接続行列を保存する。`connection_single_column.txt` がある場合はそれを読み込み、公式ツリーの `connection.deflate` しかない場合は `build_tools/zlib_util.py` とインストール済み Python を使って CLI が一時展開してから保存する。Python が使えない場合は、先に `connection_single_column.txt` を用意する。

## 3. TSV 入力仕様

入力は次の UTF-8 TSV とする。

```text
reading<TAB>surface<TAB>priority<TAB>frequency
```

- `reading`: ローマ字・カタカナをひらがなへ正規化して格納
- `surface`: 候補として表示する文字列
- `priority`: 初期優先度。大きいほど優先
- `frequency`: 配布時点の補助頻度。ユーザー選択では更新しない
- 空行と `#` で始まる行は無視

現在の例は [dictionaries/phase2.tsv](../dictionaries/phase2.tsv) を利用できる。

## 4. 暫定バイナリ形式

ファイルは次の little-endian 形式である。

```text
magic       8 bytes   "AIDICT\0\x02"
entry_count u32
matrix_size u32
entries:
  reading_len  u32
  surface_len  u32
  priority     i32
  frequency    u32
  left_id      u32
  right_id     u32
  cost         i32
  reading      UTF-8 bytes
  surface      UTF-8 bytes
connection matrix:
  matrix_size * matrix_size i32 values (row-major)
```

ロード時に以下を検証する。

- magic/version が一致すること
- 各レコードがファイル末尾を越えないこと
- reading と surface が UTF-8 で空でないこと
- レコード後に余分なバイトがないこと

現行版は全エントリを `Vec<Word>` に展開し、core が先頭文字 `HashMap` を構築する。そのため、大規模辞書向けの最終形ではない。

## 5. 使い方

TSV をコンパイルする。

```powershell
cargo run -p ai_engine_dict -- compile `
  dictionaries\phase2.tsv `
  dictionaries\phase5.aidict
```

コンパイル済み辞書を CLI で使う。

```powershell
cargo run -p ai_engine_cli -- ぷろとたいぷ `
  --static-dictionary dictionaries\phase5.aidict
```

Mozc のソース辞書を使う場合は、Mozc リポジトリの `src/data/dictionary_oss` を指定する。

```powershell
cargo run -p ai_engine_dict -- compile-mozc `
  path\to\mozc\src\data\dictionary_oss `
  dictionaries\mozc-oss.aidict
cargo run -p ai_engine_cli -- とうきょう `
  --static-dictionary dictionaries\mozc-oss.aidict
```

現在の試験用出力は `プロトタイプ` を候補として返す。

## 6. API 境界

```rust
use ai_engine_core::{KanaKanjiEngine, StaticDictionary};

StaticDictionary::compile_tsv("source.tsv", "dictionary.aidict")?;

let mut engine = KanaKanjiEngine::new();
engine.load_static_dictionary("dictionary.aidict")?;
let candidates = engine.convert("ぷろとたいぷ", 5);
```

P3 は `KanaKanjiEngine::load_static_dictionary` だけを利用し、バイナリ形式の詳細を知らない。P5 の実装を mmap に差し替えても、この呼び出し境界を維持する。

## 7. 暫定版の制約

- 実行時にファイル全体をメモリへコピーする
- 検索構造は先頭文字インデックスであり、double-array trie ではない
- 辞書の署名、圧縮、差分更新、バージョン移行は未実装
- `frequency` は配布時の値であり、P4 の学習スナップショットとは分離する
- 形式の互換性は magic/version の一致までしか保証しない

## 8. 次の置き換え

1. 採用辞書のライセンスと変換元を確定する
2. TSV から品詞・左文脈 ID・右文脈 ID・接続コストを生成する
3. Rust API からも `connection.deflate` を直接扱えるようにする
4. 読み検索を double-array trie または daachorse へ置き換える
5. ファイルを mmap の読み込み専用スナップショットにする
6. checksum、署名、原子的な差し替え、ロールバックを追加する
7. 辞書サイズ別にロード時間、常駐メモリ、p50/p95 を測定する

正式版へ移行するまでは、`phase5.aidict` を実験用の生成物として扱い、配布辞書やライセンス済みデータとはみなさない。
