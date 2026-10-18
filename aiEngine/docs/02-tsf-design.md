# P1: Windows TSF フロントエンド設計

## 1. 目的と境界

P1 は Windows 固有の入力・表示・プロセス連携だけを担当する。かな漢字変換、ローマ字処理、辞書探索、LLM 推論は P1 に置かない。

P1 は次の 2 プロセスで構成する。

```text
対象アプリ (Word / Chrome / メモ帳 ...)
        │  TSF が DLL をロード
        ▼
P1-a: myime_tsf.dll       インプロセス、C++
        │  COM LocalServer32 / IPC（プロセス間）
        ▼
P1-b: myime_host.exe      常駐ローカルサーバー、C++
        ├─ WebView2 候補ウィンドウ
        └─ P2: Rust controller DLL (C-ABI)
```

P1-a は対象アプリのプロセス内で最小限に動かす。P1-b は WebView2、Win32 ウィンドウ、P2 のロード、COM ローカルサーバーを所有する。WebView2 を P1-a にロードしてはならない。

### 用語の固定

- **P1-a は in-process**: `myime_tsf.dll` は TSF によって対象アプリのプロセスへロードされ、そのプロセス内で COM/TIP として実行される。
- **P1-b は out-of-process**: `myime_host.exe` は対象アプリとは別プロセスで動作する。
- **Out-of-Process COM は P1-a と P1-b の通信方式**であり、P1-a の実行形態を指す言葉ではない。

したがって、P1-a の責務は TSF コールバック、対象コンテキスト操作、キャレット座標取得、P1-b への短い要求の送信に限定する。重い処理、WebView2、辞書・モデルのロードは P1-b 側へ置く。

## 2. P1-a: TSF DLL

### 担当すること

- `ITfTextInputProcessor` と関連する TSF インターフェースの実装
- キーイベントの `OnTestKeyDown` / `OnKeyDown` / `OnKeyUp` 処理
- composition の開始・更新・確定・キャンセル
- `ITfContextView::GetTextExt` によるキャレット矩形取得
- P1-b の COM オブジェクトへの接続・再接続
- P1-b から受け取った確定文字列の対象コンテキストへの反映

### 担当しないこと

- 辞書やモデルのロード
- JSON の候補生成
- WebView2 や候補ウィンドウの生成
- ネットワークアクセス
- 長時間ブロックする処理

TSF のコールバックは短時間で戻る必要がある。P1-b が停止している場合は、入力を壊さず直接入力へフォールバックし、再接続を次のフォーカスまたはキーイベントで試みる。

## 3. P1-b: 常駐 EXE

P1-b は Out-of-Process COM の LocalServer32 として登録する。P1-a は次のように `CoCreateInstance` するだけでよく、起動・再起動管理を自前で行わない。

```text
TSF DLL → CoCreateInstance(CLSID_MyImeServer, CLSCTX_LOCAL_SERVER)
       → Windows COM が未起動なら myime_host.exe を起動
       → COM proxy を返す
```

### 担当すること

- COM ローカルサーバーの STA/MTA 初期化とクラスファクトリ
- セッション ID と P1-a 接続の管理
- P2 Rust DLL の `LoadLibrary` / `FreeLibrary`
- C-ABI を通じた P2 呼び出し
- WebView2 の非表示ウォームアップと候補ウィンドウ管理
- 複数セッション・複数アプリからの要求の分離
- P1-a への候補選択・確定・再描画指示

P1-b は P2 を C-ABI の不透明ハンドルとして扱う。P2 の Rust 型、`Vec`、`String`、Rust のアロケータを P1-b に露出させない。

## 4. COM インターフェース案

実際の IDL は P1 のプロトタイプ時に確定するが、意味的な契約は次の通りとする。

```text
IMyImeServer::OpenSession(session_id, client_info) -> session_handle
IMyImeServer::ProcessKey(session_handle, key_event, caret_rect) -> action
IMyImeServer::UpdateContext(session_handle, committed_text) -> void
IMyImeServer::SelectCandidate(session_handle, candidate_id) -> action
IMyImeServer::CloseSession(session_handle) -> void
```

`key_event` は仮想キーコード、修飾キー、キー状態、タイムスタンプを含む。`caret_rect` はスクリーン座標、DPI、モニター識別子を含む。候補の表記や JSON は P2 の結果として P1-b が受け取り、WebView2 に渡す。

### COM 境界の原則

- 文字列は BSTR または UTF-16 の長さ付きバッファで扱う。
- 配列は `(pointer, count)` または SAFEARRAY のどちらかに統一する。
- 呼び出し元・解放側を IDL に明記する。
- COM 例外をプロセス外へ漏らさず HRESULT に変換する。
- セッションを跨いだポインタを保持しない。

## 5. P2 Rust DLL 呼び出し

P1-b から P2 へは、初期段階では以下の最小 C-ABI を想定する。

```c
typedef struct MyImeSession MyImeSession;

MyImeSession* myime_create_session(const char* config_path);
void myime_destroy_session(MyImeSession* session);

int32_t myime_process_key(
    MyImeSession* session,
    uint32_t vk_code,
    uint32_t modifiers,
    const char* caret_json,
    char* out_json,
    size_t out_json_len);

void myime_free_buffer(char* buffer);
```

プロトタイプでは固定長出力バッファで開始し、候補数・JSON サイズが安定してからバッファ所有権の改善を行う。エラーは負の整数コードとし、詳細ログは P1-b 側の診断ログへ出す。

## 6. 候補ウィンドウ

候補ウィンドウは P1-b の WebView2 を使う。ウィンドウは次のスタイルを必須とする。

- `WS_EX_NOACTIVATE`
- `WS_EX_TOPMOST`
- `WS_EX_TOOLWINDOW`
- 表示は `SW_SHOWNOACTIVATE`
- `WM_MOUSEACTIVATE` は `MA_NOACTIVATE`

P1-a が取得したキャレット矩形を P1-b に渡し、P1-b が DPI とマルチモニターの座標変換、画面端での上下反転、候補ウィンドウのサイズ調整を行う。候補クリックは WebView2 から P1-b、P1-a、対象 TSF コンテキストの順に確定処理へ返す。

WebView2 には `PostWebMessageAsJson` で候補を渡す。候補 UI は入力処理の真実を持たず、表示と選択イベントの送信だけを行う。

## 7. ライフサイクル・障害時の挙動

1. TSF DLL がロードされる。
2. 必要時に P1-b を COM Activation する。
3. P1-b が P2 セッションを作成する。
4. キーイベントを P2 に送り、返却 action を処理する。
5. フォーカスアウト時に composition と候補窓を閉じる。
6. COM 切断時は P1-a が接続を破棄し、次回イベントで再接続する。

P1-b のクラッシュで対象アプリを巻き込まないことを最優先とする。P2 のエラー、辞書ロード失敗、LLM タイムアウトは候補なし・古典候補のみへ降格する。

## 8. 登録・配布

TSF TIP の登録は通常の Per-User COM 登録だけでは完結しないため、インストーラーは管理者権限を要求する。P1 の登録・アンインストールは Inno Setup 等の従来型インストーラーで扱う。

- TSF DLL / COM LocalServer / P2 DLL / WebView2 固定資産を配置
- CLSID、TIP、言語プロファイル、カテゴリを登録
- アンインストール時に逆順で解除
- 自己署名は開発用、配布は署名または GitHub Releases とする
- MSIX を第一配布経路にしない

## 9. P1 の検証順序

1. Microsoft の TSF サンプルをビルドし、素通りする TIP を確認する。
2. P1-a と P1-b の Out-of-Process COM 接続だけを確認する。
3. WebView2 なしで P2 の固定 action を表示する。
4. 非アクティブ候補窓とキャレット追従を確認する。
5. WebView2 を接続する。
6. 最後に実際の P2 Rust DLL と辞書・LLM を接続する。
