# P2 EditLine・辞書queryの初期実装

2026-10-03。`D:\Git\ribbonime\src\corelogic\inputmodel\EditLine.{h,cpp}`、`ja.LiteralConvert.cpp`、`EditFunctions.cpp`、`dictionary/Decoder.cpp`を参照し、ユーザーが指定した層構造・辞書query契約をRustで実装した。

## バッファの正本

`EditLine`は最表層のrange列と、表示上のcaret/選択範囲を保持する。各rangeは不変の`Arc<TextRange>`。`Raw`（入力文字）、`Phonetic`（かな）、`Converted`（漢字等）の層を持ち、置換前のrangeを`sources`に残す。生入力と表示の長さが同じという仮定を置かない。

例:

```text
表示 frontier:             市
                          │ sources
かな層:                   し
                          │ sources
入力層:                 s / h / i
```

`composition()`はfrontierの表記を順に取り出す。Backspaceは表示文字を削除するため、`shi → し`の後にBackspaceすると`sh`へ戻らず、`し`が消える。かなrangeの途中で分割する場合は、元rangeへの参照と`source_slice`を残し、たとえば`しゃ`の`し`に対して生入力の一文字対応を捏造しない。

ローマ字変換はcaret直前の連続したRaw範囲だけに作用する。変換済みかなの境界を越えてローマ字として再解釈しない。未完成の子音や末尾`n`はRawのまま表示し、確定操作でcaret前の末尾`n`を解決する。未完成の文字を捨てない。`nn`は後続が来るまでRawに保持し、後続の母音との組合せまたは明示確定で解決する。

漢字語には語全体の読み・表記の対応があるが、各漢字に個別の読みを割り当てない。漢字語の途中を選択しようとするとエラーを返す。まず変換を取消してかな層に戻し、その表示位置で編集する。カーソル/Backspace/Deleteのセッション操作も変換中は先にかな層へ戻す。

## 辞書queryとoutput lattice

```text
P2 EditLine（入力・かな・漢字の参照グラフ）
    │ DictionaryQuery::query(&EditLine, context)
    ▼
辞書reader: input lattice のリンクを辿って索引検索
    │ 単語、開始/終了reading frame、通過したinput edge ID
    ▼
OutputLattice
    │ N-best経路選択
    ▼
EditLine::apply_words → 新しいConverted frontier → composition
```

P2の辞書呼出し境界に読みの連結文字列を渡さない。readerがバッファから構造的なinput latticeを取得する。開発用`MemoryDictionary`はtrieの状態を入力リンクの接続に沿って進め、異なる単語長・単語境界・分岐入力・複数文字のreadingリンクに対応する。

frameは読み層のUnicode scalar位置。compositionのcaretもUnicode scalar位置だが、表記層の位置であるため、両者の数値を混同しない。`にほんご → 日本語`はreading frame 0..4、表示長3。WindowsとのUTF-16位置変換は今後のfrontend境界で扱う。結合文字/emojiのgrapheme単位編集は未実装。

漢字層があるrangeは`reading`と`display=Some(surface)`を持つ単一のbound入力リンクになる。たとえば`かう/飼う`を指定したqueryは`買う`を返さない。このboundリンクを自由な読みの一部として横断し、別の長い単語へ吸収することも禁止する。辞書に一致する組がなくても、その読み・表記をfallbackとして維持する。

`OutputLattice::validate`は、単語が実在する連続input edge列に対応し、読みと表記拘束を守ることを検査する。`apply_words`は、選択したword pathが読み全体を覆うことを変更前に検証し、各単語の読み範囲を元として漢字層を作る。変換取消ではsourceを辿ってかなfrontierを復元する。候補巡回のたびに生ローマ字を再解釈する方式にはしない。

`query_reading(&str)`はデバッグ用。文字列からPhoneticの`EditLine`を作り、input latticeを生成する同じquery経路を呼ぶ。別の文字列検索実装は設けない。

## セッションと外部処理

- `FocusIn/FocusOut`、入力、選択、カーソル、変換、候補巡回、確定、取消を型付きeventで処理する。
- 候補IDはsession ID・generation・indexの組。古い表示や別sessionの候補選択を採用しない。
- 予測は`RequestPrediction`をworkerへ渡し、`PredictionReady`で結果を受ける。入力更新・context更新・取消・focus喪失では古い結果を無効にする。予測結果だけではcommitしない。
- `LearnSelection`は辞書語をP4の書込queueへ渡すためのaction。P2のキー処理ではファイルへ保存しない。
- readerエラー時はかなのcompositionを保持して確定可能にする。未知文字・未完成ローマ字はfallbackで保持する。

現段階の`MemoryDictionary`とbounded N-bestは構造契約の検証用。word単体の`f64` potentialを加算し、beam幅を制限する。既存coreの連接行列、P5 mmap/byte-offset ID、CRFの6種スコア、LLMの非同期再順位付けはまだ接続していない。ユーザー変更可能なキーマップ/ローマ字table、入力停止timer、正式JSON/C-ABIも後続工程。

## 検証と実行

他担当と共有する既定targetを避け、独立したbuild出力で確認する。

```powershell
cargo fmt -p ai_engine_orchestrator --check
cargo clippy -p ai_engine_orchestrator --all-targets --target-dir target/orchestrator-check -- -D warnings
cargo test -p ai_engine_orchestrator --target-dir target/orchestrator-check
cargo run -p ai_engine_orchestrator --example replay --target-dir target/orchestrator-check -- nihongohenkan
```

23テストで、三層のsource保持、表示文字単位の削除、途中挿入、選択置換、Unicode位置、変換取消、候補確定、世代/session分離、辞書障害、予測取消、入力分岐、異なる語境界、bound query、bound語と未変換読みの混在、デバッグquery経路一致、壊れたoutput拒否を検証した。replayは小さなfixture辞書で入力→composition→候補→確定を実行する。大規模辞書の性能やTSFの操作感を評価した結果ではない。
