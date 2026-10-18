# ADR-0001: ローカル言語モデルの推論ランタイムに llama.cpp を採用する

- Status: Accepted
- Date: 2026-09-26
- Scope: P3 の LLM 評価、type-ahead prediction、next-word prediction

## Context

現在の ai_engine_llm は ContextScorer、同期・非同期パイプライン、古典候補へのフォールバックまで実装している。一方、実際のモデルランタイムは未接続である。

モデルの用途は次の三つである。

1. 古典かな漢字変換が作った候補の文脈再順位付け
2. 入力途中の type-ahead prediction
3. 確定語の後に続く next-word prediction

必要な機能は、オフライン動作、低レイテンシ、CPU fallback、量子化モデル、raw logits、KV cache、候補制約付き生成である。

## Decision

第一候補として llama.cpp を採用する。製品経路では OpenAI 互換 HTTP サーバーを経由せず、llama.cpp の C API を P1-b の常駐 host、または Rust 側の薄い FFI adapter から直接呼ぶ。

Rust の既存 API は維持する。

- ai_engine_core: 辞書探索と古典候補生成
- ai_engine_llm: ContextScorer、Onnx などではなく LlamaCppScorer、非同期 generation 管理
- myime_host.exe: モデルロード、推論、キャンセル、IPC の所有者
- P1-a TSF DLL: 入力コンテキストと候補操作だけを host へ渡す

モデルが未配置、ロード失敗、推論タイムアウトの場合は、既存の古典候補へ降格する。

## Scoring contract

### かな漢字変換

古典エンジンが作った候補集合をモデルへ渡す。入力途中の open prefix では、候補ごとの token 長の違いによる偏りを避けるため、全候補の最短 token 深度 `K_min` までだけ logits から log-softmax を計算し、その共通深度の log-probability を総和して返す。`K_min` より後ろの候補部分は古典コストに任せる。

IME の分節・単語を確定可能な閉じた入力単位として比較する評価経路では、この制約を適用しない。候補全体の log-probability を最後まで合算し、最後にモデルの EOS の log-probability を加算する。EOS は長さ合わせの padding ではなく、分節がその位置で終了する確率を表すためである。length penalty は適用しない。

    combined_score = classical_score + alpha * normalized_candidate_logprob

モデルの自由生成を確定経路に使わない。確定文字列は static/dynamic dictionary 由来の候補に限定する。

### type-ahead prediction

入力途中の読みと前文脈から、辞書に存在する次の語候補を列挙し、llama.cpp の next-token logits で順位付けする。候補数、token 深度、推論時間を制限し、入力更新ごとに generation を更新する。

### next-word prediction

確定済み文脈から次の語を予測する。初期版は辞書候補を制約付きで順位付けする。辞書外の自由生成は実験 API に隔離し、ユーザー操作なしに対象アプリへ commit しない。

## Why llama.cpp

llama.cpp は GGUF の量子化モデルを CPU/GPU で実行でき、生成に必要な KV cache と sampler を同じ C/C++ ランタイムで扱える。C API の batch には logits 出力指定があり、context ごとの logits を取得できるため、候補再順位付けと独自の raw-logit 評価に直接対応できる。

このプロジェクトはすでに C++ の TSF/host と Rust の C-ABI 境界を想定している。そのため llama.cpp が C++ であることは追加の設計負債になりにくく、推論を out-of-process の myime_host.exe に置けば対象アプリのクラッシュやモデル停止から分離できる。

開発時だけ llama-server の OpenAI 互換 API を使うことは許可する。ただし HTTP の log-probability は検証用に限定し、製品経路は直接 C API とする。

## Alternatives

### ONNX Runtime / ONNX Runtime GenAI

Execution Provider と Windows/WinML の統合は強い。しかし、モデルを ONNX へ変換し、tokenizer、KV cache、出力 logits、量子化形式を配布単位として管理する必要がある。GenAI API も現時点では preview であり、今回の raw-logit 評価と GGUF の試作速度では llama.cpp を優先する。

将来、ONNX モデルを社内標準にする場合や WinML の hardware routing が必要になった場合に第二 adapter として追加する。

### Candle

Rust-native で依存を Rust に寄せられる点は魅力だが、モデルごとの transformer 実装、tokenizer、KV cache、量子化、GPU backend をプロジェクト側で保守する範囲が大きい。純 Rust の研究用実装には向くが、最初の製品 runtime には採用しない。

### ローカル OpenAI 互換 HTTP サーバー

llama-server を含む HTTP 経路は CLI と結合テストに便利である。一方、常駐 IME の入力経路ではプロセス・HTTP・JSON の遅延と、raw logits/KV cache API の非標準性が問題になるため、製品経路には使わない。

## Model package contract

モデルの必須配布物は GGUF 1 ファイルとする。GGUF には llama.cpp が推論に必要とする重み、語彙、tokenizer、モデル設定が含まれるため、今回の candidate scoring の実行に `tokenizer.json` や `config.json` を別途要求しない。

検証・運用用の sidecar metadata は任意で、モデルディレクトリまたはアーカイブに追加できる。

    model.gguf             # 必須
    sha256                 # 任意だが配布時は推奨
    model.json             # 任意: context_length / vocabulary_size / supported_tasks

sidecar の `supported_tasks` は candidate_scoring、type_ahead、next_word を区別する。sidecar がない場合でも、ロード時に GGUF 内の tokenizer、語彙数、コンテキスト長、量子化形式を読み取り、実行時設定との整合性を検証する。

## Architecture

    ai_engine_core
      └─ 古典候補、辞書、ラティス
            │
            └─ ai_engine_llm::ContextScorer
                  └─ LlamaCppScorer
                        ├─ llama_model
                        ├─ llama_context
                        ├─ tokenizer
                        ├─ logits / log-prob evaluator
                        └─ KV cache

LlamaCppScorer は候補生成や TSF を直接知らない。AsyncConversionPipeline の generation 管理、タイムアウト、フォールバックを利用する。

推論プロセスの所有権は段階的に次のようにする。

1. CLI/ベンチでは Rust FFI adapter から直接ロードする
2. TSF 接続時は myime_host.exe がモデルと llama_context を所有する
3. P1-a は IPC 経由で候補要求を送り、推論を対象アプリ内で実行しない

## Rollout

1. llama.cpp の固定リビジョンを vendor/build artifact として決める。
2. 小型 GGUF モデルを CPU でロードし、1 回の next-token logits を取得する smoke test を追加する。
3. 候補 token の共通深度における log-prob 総和を LlamaCppScorer として実装する。
4. 閉じた分節・文の比較では、全候補 token + EOS の log-prob 総和を実装する。
4. ConversionPipeline と AsyncConversionPipeline に接続し、古典のみとの順位差、p50/p95 遅延、失敗時降格を計測する。
5. static/dynamic dictionary と結合した type-ahead prediction を追加する。
6. KV cache を利用した next-word prediction と辞書制約を追加する。
7. Windows の CPU/GPU backend 配布と host IPC を検証する。

## Consequences

### Positive

- raw logits、log-probability、KV cache、生成を一つの runtime で扱える。
- GGUF 量子化モデルを使って CPU-only 構成から始められる。
- 既存の C++ host と Rust C-ABI 設計に自然に接続できる。
- HTTP を常駐入力経路から外し、候補評価の遅延を制御できる。
- モデル未配置時の古典 fallback を維持できる。

### Negative

- C++ runtime と native backend DLL のビルド・配布管理が必要になる。
- llama.cpp の API と GGUF metadata の固定リビジョン管理が必要になる。
- 日本語 tokenizer とモデル品質は別途評価が必要になる。
- Rust から raw C API を安全に扱う薄い FFI 層を実装する必要がある。

## Acceptance criteria

- モデル未配置時も古典かな漢字変換は動作する。
- 候補評価は入力更新をブロックせず、古い generation の結果を破棄する。
- CPU 実行で候補評価と next-token logits の再現可能なテストがある。
- type-ahead と next-word prediction は候補数・深度・時間上限を超えない。
- raw model output を対象アプリへ自動 commit しない。
- モデル、tokenizer、llama.cpp revision、quantization、backend のバージョンを記録できる。

## References

- llama.cpp repository: https://github.com/ggml-org/llama.cpp
- llama.cpp C API header: https://github.com/ggml-org/llama.cpp/blob/master/include/llama.h
- llama.cpp server: https://github.com/ggml-org/llama.cpp/tree/master/tools/server
- Candle repository: https://github.com/huggingface/candle
- ONNX Runtime Generate API: https://onnxruntime.ai/docs/genai/
