# P2: IME オーケストレータ設計

## 1. 役割

P2 は OS から受け取った抽象入力イベントを、P3（変換エンジン）、P4（動的辞書）、P5（静的辞書）へ配分するクロスプラットフォームのコントローラーである。

P2 は単なるヘッダーや C-ABI の薄い箱ではない。ローマ字入力、キーマップ、セッション状態、確定・キャンセル、サブモジュールの初期化と終了を一つの状態機械として管理する。

```text
P1 の抽象イベント
       │
       ▼
P2 Controller (Rust)
  ├─ 入力モード / セッション状態
  ├─ ローマ字かな変換
  ├─ キー・機能マッピング
  ├─ P3/P4/P5 のセットアップと配線
  └─ UI向け action の生成
       │
       ├─ composition / candidates / commit / prediction
       ▼
P1 の表示・対象アプリへの反映
```

## 2. P2 が所有する状態

セッションごとに `ImeSession` を作り、別セッションの入力や候補を混ぜない。

```text
SessionState {
  mode: Direct | Hiragana | Katakana | FullWidthAscii,
  phase: Direct | Composing | Converting | Predicting,
  raw_romaji: String,
  reading: String,
  committed_context: String,
  candidates: Vec<CandidateRef>,
  selected_index: usize,
  generation: u64,
  caret: CaretInfo,
}
```

P2 が保持するのは現在の編集状態と候補 ID であり、静的辞書や LLM の内部状態ではない。P4 が保持する履歴、P5 が保持する mmap 辞書、P3 が保持する推論キャッシュはそれぞれの所有者に任せる。

## 3. 抽象イベントと action

### 入力イベント

```text
KeyDown { key, modifiers, timestamp }
KeyUp { key, modifiers, timestamp }
TextInput { text }
FocusIn { context_id, caret }
FocusOut
CaretMoved { caret }
CandidateSelected { candidate_id }
Tick { now }
```

P1 は VK コードや TSF の型を P2 のキー種別へ変換する。P2 は Windows、macOS、Linux の仮想キーコードを知らない。

### 出力 action

```text
Noop
UpdateComposition { text, selection }
ShowCandidates { candidates, caret, generation }
HideCandidates
Commit { text }
SetMode { mode }
ShowGhost { generation, text }
HideGhost
RequestRedraw
Error { recoverable, message_code }
```

action は純粋なデータであり、P2 自身はウィンドウを作らない。P1 は action を TSF 操作と WebView2 メッセージへ変換する。

## 4. 状態遷移

```text
Direct
  └─ 文字キー → Composing
Composing
  ├─ 文字キー → reading 更新 / UpdateComposition
  ├─ Backspace → 1 文字削除 / 空なら Direct
  ├─ Space → Converting / P3 の Top-N 要求
  ├─ Enter → Commit(reading または候補)
  └─ Esc → Direct / HideCandidates
Converting
  ├─ Space / PageUp / PageDown → 候補選択変更
  ├─ Enter → Commit(selected)
  ├─ Tab → Predicting または Ghost 確定
  ├─ 文字キー → 新しい Composing
  └─ Esc → Composing へ戻す
Predicting
  ├─ 文字キー → 予測をキャンセルして Composing
  ├─ Tab → Ghost を Commit
  └─ 入力停止 → P3 の非同期結果を世代番号で検証
```

状態遷移は決定論的にテストできる純粋関数として切り出す。タイマーやスレッドはイベントを生成するだけで、状態を直接変更しない。

## 5. ローマ字かな変換

ローマ字変換は P2 の責務である。P3 はひらがな列を入力として受け取る。

- 最長一致のローマ字テーブル
- `ん`、促音、拗音、`n` の未確定処理
- `si/shi`、`ti/chi` などの許容表記
- Backspace で編集可能な未確定ローマ字列
- カタカナ・全角英数・直接入力モード
- ユーザー設定で変更可能なキー配列

P2 のローマ字 parser は「入力途中」を表現できる必要がある。たとえば `n` の直後に次のキーが来るまで、`ん` と確定せず `n` を保留する。

## 6. キーマップ

初期の既定値は次の通りとする。

| キー | Composing | Converting | Predicting |
|---|---|---|---|
| Space | 変換開始 | 次候補 / 次文節 | 予測を隠す |
| Enter | 読みを確定 | 選択候補を確定 | 予測を確定 |
| Tab | 予測要求 | 予測要求 | ゴーストを確定 |
| Esc | 編集取消 | 変換取消 | 予測取消 |
| Backspace | 1 単位削除 | Composing へ戻る | 予測取消後に削除 |
| 矢印 | カーソル移動 | 候補移動 | 候補移動 |

キー設定は P1 の VK コードではなく、`Key::Space`、`Key::Enter` のような論理キーに対して行う。

## 7. サブモジュールのセットアップ

P2 のセッション生成時に以下を依存性注入する。

```text
ControllerConfig
  ├─ Engine (P3)
  ├─ DynamicStore (P4)
  ├─ StaticDictionary (P5)
  ├─ Keymap
  └─ PredictionPolicy
```

初期化失敗の扱いは次の通り。

- P5 失敗: 組み込み最小辞書で起動し、診断 action を記録
- P4 失敗: 学習なしで起動
- LLM 失敗: P3 の古典変換のみで起動
- P3 失敗: P2 は読みをそのまま `Commit` できる安全モードへ移行

P2 は部分的な機能低下を許容し、IME 全体を落とさない。

## 8. P2 の C-ABI

P1-b からは C-ABI の薄い入口を提供する。Rust の型は不透明ハンドルの内側に閉じ込める。

```c
typedef struct MyImeSession MyImeSession;

MyImeSession* myime_create_session(const char* config_path);
void myime_destroy_session(MyImeSession* session);
int32_t myime_process_event(
    MyImeSession* session,
    const uint8_t* event_json,
    size_t event_len,
    uint8_t* action_json,
    size_t action_capacity,
    size_t* action_len);
```

P2 の C-ABI は P1-b 以外からも利用できるよう、将来の macOS/Linux アダプターを意識して設計する。JSON は境界の表現形式であり、P2 内部では型付き enum と構造体へ直ちに変換する。

## 9. 並行性

- 1 セッションの通常キー処理は直列化する。
- P3 の古典変換は同期でもよいが、P3 の LLM 評価・予測は非同期にする。
- P2 は `generation` を発行し、古い結果を捨てる。
- P4 の書き込みは入力処理をブロックしない書き込みキューへ送る。
- P5 の辞書スナップショットは読み取り専用で共有する。

## 10. 検証

- ローマ字 parser の表形式テスト
- Backspace、Esc、Enter、Space の状態遷移テスト
- 同一 session 内の generation 単調増加テスト
- P3/P4/P5 の初期化失敗時のフォールバックテスト
- P1 を使わない CLI／テストランナーからの action 再現テスト
