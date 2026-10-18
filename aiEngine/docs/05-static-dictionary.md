# P5: aiEngine 静的辞書

## 1. 目的と構成

P5では、かな漢字変換と入力中predictionが読み取り専用で利用する静的辞書を生成する。コンパイラ、dumper、runtime readerをRustで実装する。バイナリ形式はaiEngine用に新規定義し、外部辞書形式とのバイナリ互換を目標にしない。

実装状況: Rust compiler/dumper v7は、MeCab標準辞書とuser辞書ソースを母集団にした語彙、byte-offset word ID、共通double-array、VOCABULARY内のprediction word/phrase候補・prefix cache、POS bigram、word bigram/backoff、succinct index付きtrigram block、各確率表の独立量子化とdumpまで実装済み。実行時readerは未実装。

```text
MeCab語彙 / コーパスID列 / n-gram / 単語・phrase prediction候補
              │
              ▼
     aiEngine dictionary compiler
       ├─ ローカルPOS ID採番
       ├─ byte-offset word ID付きvocabulary作成
       ├─ 読み索引・確率モデル作成
       └─ GUID付きblockを出力
              │
        .sdict + dumper
              │
       （今後）mmap read-only reader
```

既存の暫定 `AIDICT v2` importerは残すが、新形式と混同しない。新形式は専用magic、format GUID、versionを持つ。

## 2. IDと文字列の契約

### 2.1 Word ID

- `word_id` は`VOCABULARY` block内にあるword recordのbyte offsetである。別のID→record変換表は置かない。
- word recordは表記・読み文字列を持たず、表記keyのdouble-array leaf、POS ID、`P(word | POS)` の量子化codeを保持する。
- 生の単語countと品詞別合計はコンパイル時の正規化にのみ使い、辞書へ格納しない。変換モデル用の独立した `P(word)` テーブルも持たない。
- `word_id` はword bodyを直接指す。表記はbody内のsurface leafから復元し、読みはword body位置から同じkey groupの終端recordまで進み、そこにあるreading leafから復元する。品詞文字列はPOS IDから得る。
- n-gramとprediction phrase候補はword ID列を保存する。単語候補は既存WORD recordを参照する。

### 2.2 POS ID

このシステムでいう品詞（model POS）は、`名詞,一般,*,*,*,*,*` のようなMeCabの文法属性を保持した文字列を基本とする。大分類だけに縮約しない。助詞・助動詞は最後のフィールドで細分化するが、格助詞・連語、終助詞、副助詞には下記の先頭音グルーピングを適用する。

- コンパイル時に、そのsdict内で使うPOS文字列集合からローカルPOS IDを割り当てる。
- sdictは `pos_id → model POS文字列` の表を必ず保存する。dumpの `mecab_feature` 列にもこのmodel POSを出す。
- POS IDの数値は辞書間で固定しない。同じ文字列でもsdictごとに異なるIDでよい。
- 異なるsdictのPOS ID数値を比較・コピーしてはならない。辞書間の同一性は文字列で判定する。
- ユーザー辞書はPOS IDではなく文法feature文字列を保持する。ロード時、単語の表層形と文法featureから下記と同じmodel POSを作り、対象sdictのPOS文字列表からローカルIDへ解決する。助詞・助動詞を細分化前のPOSへフォールバックさせない。
- 対応文字列が存在しない場合に、数値IDを推測して割り当ててはならない。ロードエラーまたは明示的な未解決扱いとする。
- compilerの `vocab.tsv` adapterとMeCab/user辞書ソースの取り込みには、同じmodel POS生成規則を適用する。生のMeCab featureから文法属性6項目を取り、model POSの7項目目は通常 `*` とする。通常の助詞では表層形を正規化して7項目目に使う。`助詞,格助詞,連語` はMeCabの下位分類を保ったまま、同じ連語内で読みの先頭モーラごとに7項目目をまとめる（例: `によって`・`により` は `助詞,格助詞,連語,*,*,*,に`）。通常の `助詞,格助詞,一般` とは別POSのままとする。`助詞,副助詞` と終助詞もそれぞれの下位分類内で読みの先頭モーラごとにまとめる（例: `ばかり`・`ばっかり` は `助詞,副助詞,*,*,*,*,ば`、`ね`・`ねぇ`・`ねー` は `助詞,終助詞,*,*,*,*,ね`）。助動詞では読みをひらがな化して7項目目に使うため、文法属性と読みが同じ漢字・ひらがな表記は同じmodel POSになる（例: `ござる`・`御座る`、`ごとく`・`如く`）。先頭モーラは読みをひらがな化して求め、「きゃ」等の拗音は合わせて扱うが、「ねぇ」のような小書き母音は次のモーラとして扱う。これはPOSのグルーピングであり、MeCab tokenの分割・結合やMeCab辞書は変更しない。混合下位分類のPOSは純粋な副助詞などへ併合しない。フィールドは追加せず、7項目のまま保つ。その他の品詞では単語固有の原形・読み・発音をPOS IDの一部にしない。読みは読み項目から取り出し、ひらがなへ正規化する。

例:

```text
助詞,格助詞,一般,*,*,*,が
助詞,格助詞,一般,*,*,*,を
助詞,格助詞,一般,*,*,*,に  # 通常の格助詞「に」
助詞,格助詞,連語,*,*,*,に  # によって、により等
助詞,格助詞,連語,*,*,*,と  # として、という等
助詞,副助詞,*,*,*,*,ば  # ばかり、ばっかり等
助詞,係助詞,*,*,*,*,は
助詞,係助詞,*,*,*,*,も
助詞,終助詞,*,*,*,*,ね  # ね、ねぇ、ねー等
助動詞,*,*,*,特殊・タ,基本形,た
助動詞,*,*,*,特殊・タ,基本形,だ
```

文法属性は残すため、格助詞の「が」と接続助詞の「が」は別POSになる。同じ文法属性でも、通常の助詞「が」と「を」は別POSになる。一方、助動詞の「ござる」と「御座る」のように文法属性・読みが同じ表記違いは同じPOSになる。「た」と「だ」は読みが異なるため別POSのまま。格助詞・連語は連語の分類を保ったまま先頭モーラでまとめ、通常の格助詞とは混ぜない。終助詞と副助詞も各下位分類内で先頭モーラごとにまとめる。混合下位分類のPOSはそのまま保つ。これにより、対象語の表層形ごとにPOSが増えるのを避ける。MeCabそのものの辞書CSV、左右連接ID、コーパスword IDは変更しない。POS keyを変更したモデルでは、POS表・POS bigramと、依存する学習済みCRFモデルを同じPOS規則で再生成する。

頻出機能語の接続を個別に調整する場合は、この最後のフィールドを使う。通常の助詞は表層形ごと、助動詞は文法属性と読みごと、格助詞・連語、終助詞、副助詞はそれぞれの下位分類を保って読みの先頭モーラごとにPOS化する。他の機能語を対象に加えるときは語・文法属性を明示的に選ぶ。単なる頻度閾値で全ての高頻度語を独立POS化するものではない。文字種はこれとは別の特徴軸として扱う。

#### 単漢字辞書ソース（2026-10-15）

単漢字辞書ソースの文法属性は `記号,単漢字,*,*,*,*`、model POSは `記号,単漢字,*,*,*,*,*`。原表記と読みを保持し、互換漢字・補助平面の漢字を通常の英語・数のPOS-only語彙へ集約しない。ソース取り込みは明示的な内部品詞を優先する。CSV上の左右接続ID・単語コストは、使用IPADICの `記号,一般` の実在行を参照する。

Mozc + Unihanによる13列TXTソース、固定版・件数・再生成手順は単漢字辞書ソース（今回の統合対象外のデータ作成資料）に記録する。

#### 読み仮名の内部品詞（2026-10-13追加仕様）

読み注釈として使われる仮名列には、aiEngine内部品詞 `記号,読み仮名` を割り当てる。例えば `令和（れいわ）` の `れいわ` は注釈区間全体で1 tokenとし、括弧と本文の `令和` は別tokenとして保持する。

| 項目 | 定義 |
|---|---|
| 文法属性6項目 | `記号,読み仮名,*,*,*,*` |
| model POS（7項目） | `記号,読み仮名,*,*,*,*,*` |
| 左接続ID | 使用する辞書の `記号,一般` と同じID |
| 右接続ID | 使用する辞書の `記号,一般` と同じID |

接続IDを共有しても、品詞文字列とローカルPOS IDは `記号,読み仮名` として独立させる。読み仮名の表記ごとにPOSを細分化しない。

このtokenは補正時の保護単位とする。マルチparserの単語境界投票でも、内部の再分割、周囲のtokenとの結合、通常品詞への上書きを補正対象から外す。括弧内の仮名列というだけで確定せず、読み注釈として扱う範囲を特定してからこの品詞を付ける。

ここでは品詞仕様を定義する。読み仮名検出・rg/sed工程（今回の統合対象外のデータ作成資料） は、2026-10-13にchunk015へ初回適用した後、001〜016へ番号順に展開した。途中で日付・別名・学名の誤検出条件を絞り、旧条件の001の91区間・015の160区間を元セルへ復元した。修正版runは1,022,812文・1,042,895区間を新規補正し、全変更行照合済み。元行不一致・置換拒否3,052行は保留。既存タグは保持する。独立した言語的正解率は未測定。補正器での保護と接続ID割当ての実装は別工程である。

### 2.3 予約token

コーパス語彙の予約IDは `NULL=0, BOS=1, EOS=2`。これらは通常wordと区別し、辞書表示語としては扱わない。POSを要するモデルでの予約tokenのPOS割り当ては、専用の予約feature文字列を使う。

## 3. 読み検索と高速性

表記と読みは同一のdouble-array key空間で検索する。index key集合は語彙の全surface集合とreading集合の和集合とし、同じ文字列ならleafも共有する。読みkey groupにはword record群を置き、各word recordのsurface-leaf参照から表記を復元する。surface key groupにはword ID参照群を置き、word recordが属するreading key groupから逆引き読みを復元する。surfaceとreadingが同一キーのwordは、同じgroupにword recordとsurface参照の両方を持つ。語彙recordにsurface/display文字列やreading文字列は格納しない。

索引キーはUTF-8やUTF-16のバイト列を直接使わず、コンパイラとreaderの双方で同じ内部文字コードへ変換する。Unicode文字列をUTF-16 code unitごとに次の順序で符号化する。ひらがなは1文字1 byteとなる。

| 入力 | 内部表現 |
| --- | --- |
| U+3041〜U+3096 | `codepoint - 0x3040` の1 byte（`0x01`〜`0x56`） |
| `、。・ー〜「」{|}` | `0x57`〜`0x60`（順に割り当て） |
| U+0021〜U+0060 | `codepoint + 0x40` の1 byte |
| U+3001〜U+30FF（上記の個別割当を除く） | `0xAA`, low byte |
| U+0001〜U+00FF | `0xAB`, low byte |
| low byteが`0x00`のcode unit | `0xAC`, high byte |
| U+4E00〜U+9FFF（上記のlow-byte-zeroを除く） | `(codepoint - 0x4E00 + 0xAE00) >> 8`, low byte |
| その他のBMP code unit | `0xAD`, high byte, low byte |

索引キー終端の`0x00`は文字列終端であり、入力U+0000は許可しない。コンパイラ・検索・leaf→root復元は同じ符号化表を共有し、変換後のbyte列をdouble-arrayのキーとして扱う。tail recordに格納するsuffixも、この内部byte列のsuffixである。

各keyの一意prefixより後ろのsuffixは、別blockやdedup tableにせず、そのterminalが指す`VOCABULARY` block内へtail-string special recordとして直接置く。terminalのindex値は同block内のkey group offsetを指す。検索時はprefix pathの後にそのkey groupのtailを照合し、文字列復元時はleaf→rootのprefixにtail bytesを接続する。suffixの別ID・offset表など追加のtail索引は持たない。

読み索引は、現在の試作で速いdouble-array構築を基準性能とし、構築・検索を遅くする変更を入れない。より高速な方式を提案する場合は、同一fixture・同一マシンでbuild時間、file size、lookup p50/p95を測定し、既存方式以上であることを示してから採用する。最初の実装では構築順序や空きslot探索を不用意に単純な全走査へ置き換えない。

tail recordは確定済みのkey圧縮方式である。tag 1 byteの上位3 bitをrecord ID、下位5 bitを `tail_length - 1` とし、その直後へtail bytesだけを置く（長さ1〜32 byte）。別のtail length・ID・offset表は追加しない。辞書の正しさ・高速性を確認するため、compilerの出力をdumperで展開し、入力語彙・索引先・ID列と照合できるようにする。

## 4. Prediction phrase

prediction候補（単語・複数語phrase）は入力途中の候補比較専用であり、POS bigram、word bigram、trigram、変換用連語スコアを加算しない。phrase recordと短いprefix用cacheはどちらも`VOCABULARY`内の見出しkey groupに格納する。prefix cacheは既存WORD recordまたはphrase recordへのbyte-offset参照を持ち、word/phrase本体をcacheへ複製しない。同じ候補が複数prefixに現れれば、複数cacheから同一recordを参照する。

### 4.1 単語・phrase候補

候補抽出は単語（1 word）と複数語phrase（2〜6 words）を同じ母集団に含め、`score(w) = P(w) × H_forward(w) × H_backward(w)` 降順に上位1,000,000件を選ぶ。前後entropyはbit単位のShannon entropy。単語候補のword IDは既存のWORD recordを指すため、単語専用のPrediction recordは作らない。phrase候補だけをphrase ID、word ID列、連結reading、unigram `P(w)` とともに復元可能にする。phraseの比較順位スコアは選択時だけ使い、記録するphrase確率は `P(w)` のみとする。

`P(w) = count(w) / total_count[length(w)]`。単語候補はコーパスの生1-gram count、複数語phrase候補は該当ID列の観測countを使う。単語・phraseとも前後方向countのminや、短いphraseからの残差減算は行わない。prefix cache内の候補順は抽出scoreではなく `P(w)` 降順とし、ここで単語の `P(word)` はcacheを作る間だけ使う。辞書には追加保存せず、WORD recordには従来どおり変換用 `P(word|POS)` のみを格納する。

### 4.2 索引と上限

Prediction用に次の索引を作る。

1. **完全reading検索**: `STRING_INDEX`でreading keyの見出しkey groupへ進み、group内の`PREDICTION` phrase recordsとWORD recordsを読む。phraseごとのreading索引blockは作らない。
2. **ひらがなprefix cache**: ひらがな1文字、2文字、3文字からなる各prefixキーの見出しkey groupに、最大5件の単語またはphrase record参照を格納する。候補順位は単語 `P(word)` / phrase `P(w)` の共通比較で降順にする。参照先はword recordまたはphrase recordであり、cache entry自体には種別を重ねて書かず、参照先recordのtagを読んで判別する。
3. **通常phrase一覧**: 全reading key groupに置く複数語phrase record全体を抽出順位順に最大100,000件選ぶ。この上限は最小countではなくphrase record数上限で、単語WORD recordは数えない。

「各prefixにつき5件」は実際に候補に現れるprefixキーごとの上限で、全prefix合計5件ではない。prefix cacheが通常一覧の上限外にあるphraseを参照する場合、そのphrase recordも`VOCABULARY`へ追加し、複数prefixや完全reading索引から共有する。このため、cache参照先を含むphrase record総数は100,000件を超える場合がある。`PREDICTION_CACHE_HEADER`は1〜3文字ひらがなprefixのkey groupにのみ存在する。単語cache候補は既存WORD recordを直接参照する。`prediction-prefix.tsv`の`stored_probability`は参照先に保存されている値を示し、単語では`P(word|POS)`であってcache順位に使った一時的な`P(word)`ではない。

### 4.3 phrase境界・候補フィルター

- 単語・phrase候補にはPOS-only token（数字、英単語、未知語など）や記号tokenを含めない。ただしn-gram・entropy統計からは除かない。
- 句読点・記号を含む候補は削除する。
- 句頭の助詞・助動詞・接尾・動詞非自立は句頭候補から読み飛ばす。
- 句末の通常助詞は読み飛ばすが、終助詞は残す。
- 形容動詞連体形「な」、動詞未然形、連用タ接続は句末から読み飛ばす。一般の動詞連用形は活用形だけを理由に除外しない。
- `名詞,サ変接続` の直後にある `動詞,自立,...,サ変・スル` から始まる用例は、その用例をphrase count・前後entropyから除外する。他の左文脈で同じ動詞が現れた用例は保持する。
- 境界tokenを読み飛ばして同一word ID列へ縮約した候補が重複する場合、最大の順位scoreを持つ候補を1件採用し、その候補の `P(w)` を引き継ぐ。

## 5. 確率モデル

変換スコアはprediction phraseとは別に保持する。

### 5.1 POS bigramとword unigram

基本分解は次のとおり。

```text
P(word_i | word_{i-1})
  = POS-bigram/backoff model × word-unigram-within-POS
```

POS bigramは完全なMeCab feature文字列から割り当てたローカルPOS IDを使う。word unigramはそのPOS内での確率。両者を一つのカウントへ混ぜない。

`P(word|POS)` の重みは、MeCab標準辞書CSVと登録済みuser辞書ソースを全件母集団として作る。コーパスに出現した語は生unigram countの10倍、ソース辞書にはあるがコーパスに出現しなかった語はcount=1とする。その後、完全なMeCab featureごとに合計して正規化する。したがって、未出現の辞書語も確率0にはならず、コーパス出現語はコーパス頻度を反映した重みを持つ。この10倍はword-unigram-in-POS専用で、POS bigramやword bigram/trigramのコーパスcountには適用しない。コーパスtokenがソースCSVに個別行として現れない活用形などは、ID列とngramを欠落させないため観測形として補助語彙に加えるが、ソース辞書の未出現語彙は全て残す。

### 5.2 Word bigram backoff

文脈word `a` について、明示的に保持する後続word集合を `S(a)` とする。backoff基底分布は

```text
q_a(x) = P(POS_x | POS_a) × P(x | POS_x)
```

である。word bigramを持たない後続語へPOS backoff質量を再配分する係数は

```text
alpha(a) = (1 - Σ_x P(x | a))
           / (1 - Σ_{x ∈ S(a)} q_a(x))
```

とし、`x ∉ S(a)` では `P(x | a) = alpha(a) × q_a(x)` を使う。明示word bigramとPOS backoff分布が覆う後続語の重複質量は分母で補正する。

### 5.3 Trigram

trigramはbigramへbackoffせず、指定された線形補間を使う。`lambda`等の値は別途モデル仕様で確定し、backoff alphaは持たない。

trigram値は1 byte量子化。基準bigramは `B/*` の後続語列で、trigram `A/B/C` のslotはbigram `B/C` の `B/*` 内index。`A/B` ごとにtrigram blockを持つ。

- `M = |B/*|`、`N_AB = |A/B/*|` とする。各AB blockについて、slot幅は`M <= 256`なら1 byte、`M <= 65,536`なら2 byte、それを超える場合は4 byteとする。
- sparseは各trigramについて固定幅slot indexと1-byte probability codeを格納する。denseは`B/*` slot順のprobability-code byte配列を持ち、非該当slotを0とする。
- `N_AB * (slot_width + 1)` と`M`を比較し、sparseがdenseより小さい場合のみsparseを選ぶ。同byte数ならdenseとする。従って、`M=256`なら`N_AB=128`、`M=65,536`なら`N_AB=65,536/3`付近が切替境界となる。境界は各AB blockの実際の`M`と`N_AB`から判定する。denseは`M` byteなので、trigram件数`N_AB` byte以下になる保証はない。
- succinct indexはbigram addressから`A/B` trigram blockの開始・終端を得る用途だけに使う。global word-bigram edge座標のparent bitvector/rankと、trigram payload上のblock開始bitvector/select sampleからblock addressを得る。`B/*` slot位置の表現には使わない。
- bigram/trigramに同じcount cutoffを適用するため、保持trigramの`A/B`と`B/C` bigramが欠落しないことをbuild時に検証する。

## 6. Log確率の1-byte量子化

確率 `p > 0` を `log(p)` に変換して量子化する。code `0` は未格納・非該当などの特殊値とし、通常の確率codeには使わない。量子化codebookは各blockまたは互換スケール単位で明示的に保存する。

単純なpure k-meansだけでは、空clusterや同一値への収束により代表値が重なり、同じ復元値・重複境界が発生し得る。そこで次を必須とする。

1. 入力をlog空間でソートし、重み付きクラスタリングする。
2. 空clusterまたはfloat保存後に重複する代表値を検出する。
3. 重複がある場合、クラスタ数が利用可能な異なる入力値数を超えないようにしつつ、within-cluster誤差の大きいclusterを分割して空きを埋める。分割点は隣接する異なる入力値の間に置き、同じ値を無理に別clusterへ分離しない。
4. 代表値を保存精度へ丸めた後に再度重複検査し、重複codeを統合・再割当する。codebookは厳密な昇順とし、検索境界は隣接代表値の中点から一意に作る。
5. 同距離の入力は常に低いcodeへ割り当てる。0 sentinelを含むcode、境界、復元値をround-trip testする。

実装は収束回数・停止条件を固定し、buildが同じ入力に対して決定的であること。空cluster補完には固定絶対epsilonではなく、隣接する代表値間の幅に対する小さな相対差を使う。最終品質は元log値と復元log値の誤差分布、code使用数、重複code数で評価する。

## 7. ファイル形式とblock識別

ファイル形式は新規設計とし、headerと各blockの意味を固定GUIDで識別する。辞書全体はformat GUID・version・dictionary UUID・build UUIDを持つ。block GUIDは種類を示す安定IDであり、dictionary-local POS IDとは異なる。

初期block一覧:

| Block | 内容 |
|---|---|
| `POS_NAMES` | POS IDから完全なMeCab feature文字列への対応 |
| `VOCABULARY` | byte-offset word IDのword recordとkey group record列 |
| `STRING_INDEX` | surfaceとreadingのunion keyから`VOCABULARY`内key group offsetへの共通double-array |
| `QUANTIZATION` | word unigram-in-POS、POS bigram、word bigram、bigram backoff、trigramの独立log確率codebook |
| `POS_BIGRAM` | 完全MeCab feature POS ID間の条件付き遷移確率 |
| `WORD_BIGRAM` | source wordごとの明示後続word列、条件付き確率、backoff alpha |
| `TRIGRAM` | 1-byte量子化値。bigram addressからA/B blockをsuccinct indexで引き、B/* slotにdenseまたはsparse格納 |

現format v7は上記7 blockをcompiler/dumperで読み書きする。phrase/prefix用の独立blockはなく、各見出しkey group内のrecordとして保存する。POS ID 0は予約し、実POSは辞書ごとに1から採番する。POS bigramは実POS数×実POS数の密な1-byte行列とし、セルの計算時にはPOS IDから1を引く。POS/word n-gram確率は未加重のコーパスcountから計算し、`P(word|POS)`のみ、コーパス出現countを10倍、辞書にのみ存在する語を1回として正規化する。

format v7のformat GUIDとblock GUIDは以下で固定する。prediction phrase/cacheは`VOCABULARY`内のrecordなので、独立block GUIDを持たない。

| 種類 | GUID |
|---|---|
| format | `4149454e-472d-5344-4943-542d56332121` |
| POS_NAMES | `504f532d-4e41-4d45-532d-563221212121` |
| VOCABULARY | `564f4341-4255-4c41-5259-2d5632212121` |
| STRING_INDEX | `53545249-4e47-2d49-4e44-45582d563221` |
| QUANTIZATION | `5155414e-5449-5a41-5449-4f4e2d563221` |
| POS_BIGRAM | `504f532d-4249-4752-414d-2d5633212121` |
| WORD_BIGRAM | `574f5244-2d42-4947-5241-4d2d56332121` |
| TRIGRAM | `54524947-5241-4d2d-5633-212121212121` |

ファイルはmagic `AISDICT\0`、format GUID、version=7、flags、dictionary UUID、build UUID、block数、directory entry幅から始まる。header固定部は72 byte、directory entryは48 byte（block GUID 16 byte、offset/size/item_count/checksum各8 byte）。blockは8 byte alignment。すべてlittle-endian、可変UTF-8文字列はu32 byte length prefixを持つ。checksumはblock payload単位のFNV-1a 64-bitで、破損検出用であり暗号学的認証ではない。

format v7のrecordは次のとおり。

- `POS_NAMES`: `count:u32`, 連続するPOS ID順の `string`。
- `STRING_INDEX`: `count:u32` と `count` 個の `base:i32, check:u32`。key byte pathはsurface/readingを兼ねる。leafの負の `base` は `-(key_group_offset + 1)`。非terminal nodeの `base` は次node配列のbase index。`check` はparent node index。検索キーには内部コード化したkey終端 `0x00` を使う。
- `VOCABULARY`: byte-offset-addressed record stream。offset 0は予約し、各key group内を`[TAIL_STRING?][PREDICTION_CACHE?][PREDICTION*][WORD*][DISPLAY_REF*][TERMINATOR]`の順に置く。各部は必要な場合だけ存在する。block directoryの`item_count`はword body数とphrase record数の合計で、tail/cache/display/terminatorは含めない。`TAIL_STRING`はtag `0x20 | (length-1)`と1〜32 byteのtail bytes。prefix cacheを持つ1〜3文字ひらがな見出しに限り、`PREDICTION_CACHE_HEADER` tag `0x40`、続けて`count:u8 (1〜5)`、`target_offset:u32[count]`を置く。target offsetはVOCABULARY内のwordまたはphrase recordを直接指し、参照先のtagを読むまで種別は決まらない。
- `PREDICTION` phrase recordはtag `0x60 | (word_count-1)`（2〜6語では`0x61`〜`0x65`）、`word_id:u32[word_count]`、phrase unigram `P(w):f64`。recordのbyte offsetがphrase ID。phrase readingは所属key groupの見出しから得る。構成word IDから表記と各word readingを引き、reading連結を検証する。通常phraseは上位100,000件までとし、prefix cacheが参照するcutoff外phraseも共有recordとして追加格納する。単語予測候補はphrase recordを作らず、既存WORD recordをcacheから参照する。
- `WORD` record（tag `0x80`）は `display_leaf:u32, pos_id:u32, P(word|POS)_code:u8, bigram_block_index:u24, backoff_alpha_code:u8`。bigram indexはedge ordinal+1で、0は明示bigram blockなし。alpha codeはblock参照と同じword recordに保持する。そのレコード開始byte offsetがword ID。reading-key groupの終端`0xe1`は`reading_leaf:u32`を持ち、word IDからrecordをたどって末尾まで読むことでreading keyを逆引きする。
- `DISPLAY_REF` record（tag `0xa0 | (count-1)`）は1〜32個のword IDを保持する。
- `TERMINATOR`はtag `0xe0` または `0xe1`。`0xe1`はreading/phrase key groupで`reading_leaf:u32`を続ける。surfaceとreadingの文字列本体は保持しない。確率は`QUANTIZATION` block内の専用codebookから復元する。
- `POS_BIGRAM`: 実POS数を`N`とし、`N×N`セルをsource POS ID順・target POS ID順に並べた`N²` byte配列。POS ID 0は予約され、行列のID 1は添字0に対応する。code 0は未観測セル。
- `WORD_BIGRAM`: `edge_count:u32`に続けて全edgeを格納し、各edgeは`target_word:u24, probability_code:u8`。source word ID、block offset、edge count、alphaはedge列に重複保存しない。source wordのVOCABULARY recordが`bigram_block_index`とalpha codeを持つ。各A* block内はtarget ID昇順、block同士は最後のtarget ID降順。現在blockの末尾で次target ID `<=` 現target IDとなる位置を次block開始とし、各blockの長さはその境界から復元する。word IDはVOCABULARY内byte offsetで、現行語彙サイズが24-bit範囲内であることをbuild時に検証する。alpha code 0はalpha=0。
- `TRIGRAM`: global headerは`bigram_edge_count:u32, parent_bitmap_word_count:u32, parent_rank_count:u32, parent_count:u32, payload_len:u32, payload_start_bitmap_word_count:u32, payload_start_sample_count:u32`。A/B parent bitvector `u64[]`と256 edgeごとのrank `u32[]`でbigram addressからblock ordinalを引く。payload byte address上のstart bitvector `u64[]`と256 blockごとのselect sample `u32[]`からblock開始addressを求める。このsuccinct indexの役割はbigram addressからtrigram blockの開始位置へ移動することだけで、B/* slotを引くためのものではない。trigram block自体にはheader/descriptorがなく、block終端は次の開始address（最後はpayload末尾）。`M=|B/*|`、`N_AB=|A/B/*|`。codecを別保存せずblock長から判定し、denseは`M` slotのprobability code配列、sparseは`N_AB`件の`(slot index, probability_code)`列。index幅はMに応じて決まる。
- `QUANTIZATION`: `codebook_count:u32`; 各tableは `model_id:u8, center_count:u16, log_probability:f32[]`。model IDは1=`P(word|POS)`、2=POS bigram、3=word bigram、4=backoff alpha、5=trigram。code 0は特殊値、code 1〜Nは代表値配列のindex+1。

現在のdumperはoffset/size overflow、header/blocks overlap、重複または未知block GUID、checksum不一致、範囲外leaf/word/POS参照、壊れたrecord/文字コード、quant code・bitmap/rank・sparse slotの不整合を拒否する。runtime readerは別工程。

## 8. Compiler / dumperの要求

Compilerは現在の高速な読み索引構築を退行させない。性能変更はbenchmarkで同等以上を確認する。巨大TSVを不必要に全件RAMへ読み込まず、ソート済み入力のstream処理や外部sortを使える境界を保つ。

Dumperは最低限、以下を人が監査できる形で表示する。

- header・format/version・dictionary/build UUID・全block GUID/offset/size/count
- POS IDとfeature文字列の対応
- word ID・表記・読み・POS IDとPOS文字列
- union double-arrayのnode配列、terminal→vocabulary offset
- vocab内tail record、leaf→rootで復元したsurface/reading、およびkeyからword/phrase ID列への対応
- phrase ID・word ID列・表記列・読み・P(w)
- prefixごとの最大5件の単語・phrase候補
- 確率量子化code・復元値・codebook

小さなfixtureをcompilerで作成し、dumperで完全に再展開する往復試験を必須とする。破損block、範囲外ID、重複GUID、POS未解決、prefix上限違反、壊れたquantization tableは拒否する。

## 9. 実装順序

1. GUID付きheader/block directory、POS_NAMES、VOCABULARYのwrite/read/dumpと往復fixture。
2. POS文字列から辞書ローカルIDへの解決APIと、順序を変えた2辞書でのテスト。
3. key group内phrase本体・共通STRING_INDEXによるreading検索・1/2/3文字prefix cache（各5件、通常phrase一覧100,000件）。
4. 共通STRING_INDEXのreading lookup性能baseline。
5. POS bigram、word unigram、word bigram backoff。
6. robustなlog確率量子化と復元誤差テスト。
7. compact trigram encodingとsparse/denseサイズ検証。

個々のblock仕様・GUID一覧・CLIの確定値は、仕様を変更するときにversioned format fixtureを更新し、compilerとdumperの両方で検証する。
