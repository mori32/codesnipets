# CRFによる辞書スコア補正

> **データ配置:** 新規CRF runのcache、学習補正、checkpoint、評価結果などのデータは [作業データ配置ルール](18-workspace-layout.md) に従い、`E:\lmwork-int` / `E:\lmwork-target` に置く。以下に残る `target/` パスや過去runの配置は履歴であり、新規runへ流用しない。リポジトリ内の既存データ許可リストは同文書に記載し、実行可能コードのビルド生成物以外の置き場を増やさない。

## 目的と範囲

`ai_engine_crf` は、読みから作った単語分割ラティスとMeCab解析済みの擬似正解列を使い、辞書へ格納するための浮動小数点スコアを学習するオフラインRustツールである。LLM、GGUF、ローカルサーバー、MeCabの整数costによる再解析は使わない。

## 次回chunk003複合素性実験の仕様（2026-10-07）

この節が初期データ生成と新規runの正本仕様である。以下の旧run記録に異なる素性や値が書かれている場合、履歴としては残すが、この節を優先する。

- **一次Viterbi**は word-unigram-in-POS、POS-bigram、POS-WORD、WORD-POS、word-bigram補正、読み長bigram補正を加算する。
- **N-best再順位付け**は一次経路スコアへ word-trigram、読み長trigramの補正を加算する。POS-trigramは使わない。gold path augmentationは学習時にN-best外の正解pathを三次側候補へ加える既存オプションであり、一次Viterbiや評価候補には加えない。
- 初期word-trigram 3,000,000組の値は固定し、各word-trigramキー固有のCRF deltaを0からN-best損失で学ぶ。共有係数 `alpha` は使用しない。読み長bigram/trigramも、キー別の学習deltaを0から学ぶ。
- 名詞の接頭語・接尾語から、raw unigram頻度順top 100語を選び、その語彙項目だけに語彙素ごとの新しいPOS IDを割り当てる。`号・時・日・数・級` と数字・助数詞相当、既に対象外の動詞接尾語は候補にしない。POSの統合・併合はせず、既存POSは保ち、選択語だけ細分化する。
- POS-WORDおよびWORD-POSの初期bigramはそれぞれ頻度上位500,000組までとし、POS-bigramの初期確率をbase、観測pair確率をempirical側として `t=0.1` で独立に線形補間したlog-score差を格納する。POS-trigramは無効。
- 大きな有限スコアへの暴走を早期発見できるよう、評価イベントごとに一次特徴、word-bigram、word-trigram、POS/readlen trigramの最大絶対deltaと、word-trigram学習キー数を記録する。値には上限クリップを加えない。非有限値は更新時にエラーとする。正則化は補正を0へ縮めるが、係数が小さければ有限値でも大きくなり得るため、ログを確認する。
- 素性ファミリー間の追加backoffや相互補間は行わない。POS-WORD/WORD-POSのbaseはPOS-bigram、word-bigramのbaseはPOS-bigram × word-unigram-in-POS、word-trigram/read-lengthのbaseと重みは従来の初期モデル規則に従う。
- word-bigram/CROSS bigram経験分布の混合重みは経験側 t=0.1（対応するbase側0.9）、word-trigramは t=0.1、読み長bigram/trigramは t=0.15。式は `delta = log((1-t) * P0 + t * P1) - log(P0)`。
- 初期word-bigramは頻度上位1,500,000組、初期word-trigramは頻度上位3,000,000組。word-trigramの P0 は word-bigram補正を含めず、POS-bigram × target word unigramだけから作る。
- POS-trigramは生成しない。読み長bigram/trigramは9クラスの全81/729セルを保持する。
- 読み長のクラスは 0〜7 と 8以上の9種類。ひらがな化した読みに対して「ゃ・ゅ・ょ」を数えず、「ー・っ」は数える。記号と読みのない単語は長さ0。読み長bigram 81セル、trigram 729セルは、観測の有無にかかわらず全件を保持・出力する。
- 経験値が存在しない組、および頻度cutoffで初期テーブルに採らなかった組は、経験側確率を P0 と同じに置く。従って式から補正値は厳密に0となる。全候補に一律 `log(1-t)` を足すことはない。
- 表記種別 unigram/bigram は今回のモデルから除外する。固定testの30分レポートでは、文完全一致率に加えて文字レベル一致率も出す。

初期モデル生成コマンド、成果物、READMEと再読込検証は [CRF初期モデル生成](19-crf-initial-model.md) を参照。

標準学習は「一次ラティスCRF + N-best内のtrigram補正」であり、二次ラティス全体のCRFではない。`--primary-nbest-margin` または `--primary-nbest-listwise` を指定する実験では、一次もN-best候補ベースへ切り替わる。共有モデルを読み取り専用で参照し、バッチの各文について一次勾配とN-best/trigram勾配を**同じ未更新モデル状態**から計算する。両方の評価・勾配計算が終わったあと、一回の並列joinを経て、一次・trigram補正をまとめて更新する。一次更新を挟む中間同期はしない。`--batch-size 1` もこの同時評価方式であり、オンライン逐次更新の `train_online` とは区別する。一次モデルを全コーパスで学習し終えてからtrigramを学習する二段階方式ではない。

### 一次・二次ともN-best listwise損失にする実験

`--primary-nbest-listwise` を指定すると、一次の全ラティスforward-backwardを行わず、raw N-best候補の一次スコア上で `logsumexp(score(candidate)) - score(gold)` を計算する。損失の候補集合にはraw N-bestを使い、goldが候補外なら一次損失の計算に限りgold pathを追加する。したがってN-best含有率は追加前のraw候補で測り、augmentationで水増ししない。gold-path augmentationオプションは従来どおりtrigram側にもgoldを追加する設定であり、一次listwise側でgoldを含める要件とは別である。

一次listwise勾配は候補softmax下の一次素性期待値からgold素性を引く。word-bigram素性はbatch内で勾配を合算し、1/batch-sizeでは割らない。word-unigram、POS-bigram、読み長bigramなどの他の一次素性はbatch全体で平均する。二次側は既存のtrigram listwise損失を維持し、word-trigram勾配をbatch内で合算、POS-trigramと読み長trigramを有効候補文で平均する。両方の勾配は同じ更新前モデルから計算し、並列join後に一度だけ適用する。この実験フラグは同期最適化trainer専用で、評価専用mode、reference kernel、Hogwildとは併用できない。

### raw N-best包含を狙う一次margin損失（実験用）

`--primary-nbest-margin M` を指定した実験では、一次の全ラティスCRF NLLを置き換え、raw N-best全経路とgoldのペアワイズhingeを使う。評価・通常runの既定値は従来どおり全ラティスNLLであり、このオプションを付けたrunだけが切り替わる。

1. 未更新の同一モデルからraw N-bestを一度だけ生成し、正解pathが含まれるか調べる。
2. raw N-best中のgold以外の各候補 `p` に対し、`L_p=max(0, M + score(p) - score(gold))` を計算し、文の損失はこれらの**総和**とする。goldがraw N-best内にある場合も、goldより高スコアの誤り候補があれば一次更新する。
3. 正のhingeを持つ各候補について、一次勾配へ `features(p)-features(gold)` を加算する。従って誤り候補固有の素性には負方向、正解固有の素性には正方向の更新が入り、共有素性は差し引きで相殺される。候補がgoldを十分下回ればその候補のhingeは非活性になる。
4. 今回の試験は `M=0` とする。これはraw N-best候補内でgoldをtop-1にする目的であり、全ラティスの他経路やraw N-bestに現れない経路を直接補正するものではない。gold-path augmentationを有効にしても、gold追加先は従来どおりtrigram候補だけで、raw N-best含有率へ加算しない。trigram側はraw N-best（goldが外なら追加goldを含む）に対する独立したlistwise損失を計算し、同じbatch join後に一次とまとめて更新する。

`M` は加算log-score単位で、0以上を指定できる。今回の`M=0`では「誤り候補をgoldよりmargin分だけ下げ切る」ことを要求せず、goldを上回る候補にだけ修正圧力がかかる。margin modeでは一次のforward-backward計算を省略してN-bestを共有する。一次のPOS/unigram/読み長などは従来どおりバッチ全体で平均し、word-bigram勾配はバッチ内で合計する。固定テストのbigram top-1、raw N-best含有率、trigram後top-1は学習損失とは独立した同じ評価指標として残す。

### raw N-best全pairwise更新のchunk003実験

以下は2026-10-06に実行した過去runの記録であり、現行listwise実験の目的関数ではない。

2026-10-06の新規試験ではmargin `M=0` を指定し、raw N-best内のgold以外すべてに対するpairwise hingeを使う。正解が候補内にある場合も、goldより高スコアの誤候補は更新対象となる。word-bigramとword-trigramの勾配は16文batch内で**合計し、1/16しない**。その他の一次特徴とPOS/readlen trigramは実装定義の有効文数で平均する。L2は0。一次bigramとN-best trigramは、同じ未更新モデルから計算してbatch後にまとめて反映する。

実行設定は16スレッド、batch 16、N=128、固定5,000文テスト、最大2epoch、epoch 2はepoch 1のpost-trigram top-1 missのみ、学習率0.03、補正表24,576 MiB、delta cutoff `1e-4`〜`1e-2`、gold-path augmentationとmissed feedback有効。初期重みは `E:\\lmwork-target\\crf-initial-model\\model.json` から毎回新規に開始する。前runから再利用するのは正規化済みページcacheと固定test sampleだけであり、学習済みparameter/checkpointは引き継がない。

この目的は、raw N-best候補内でgoldより高い誤候補の差分を直接押し下げる。候補外の経路や、すでにgoldより低い候補に追加の押し下げを要求しない。評価時は固定test上のbigram top-1、raw N=128 gold cover、post-trigram top-1を別々に報告する。最終モデル保存・再読込後、同じ固定sampleをevaluation-onlyで再評価し、必要なら文字差分に重なるgold/predicted word trigram頻度分析も実行する。

### chunk003 primary-listwise run (2026-10-06)

前回pairwise runの学習値は引き継がず、`E:\lmwork-target\crf-initial-model\model.json` から新規開始する。再利用するのは `E:\lmwork-target\crf-wiki003-n128-word-trigram-learning-20261006-run02` の正規化済みpage-block cacheと固定5,000文test sampleだけ。全chunk003、16 thread、batch16、N=128、2 epoch上限、epoch 2はepoch 1 post-trigram final top-1 missesだけ、gold-path augmentation、missed feedback、学習率0.03、L2=0、学習補正表24,576 MiB、delta cutoff `1e-4`〜`1e-2`、page block 1,000、shuffle window 256。

一次・二次は各文の同じ更新前modelからN-best listwise勾配を計算して一括更新する。一次候補softmaxはraw N-bestとgold path（N-best外なら一次損失計算だけに追加）を使い、二次はraw N-bestとgold-path augmentationで追加されたgoldを使う。word-bigram/word-trigram勾配はbatch内合算、word-unigram/POS-bi/読み長-biおよびPOS-tri/読み長-triは実装所定の平均。固定testではbigram top-1、raw N=128 cover、trigram後top-1をそれぞれ報告し、処理文/秒も直近区間差分つきで記録する。前runの学習値/checkpointは参照しない。

今回の試験入力:

```text
corpus/wiki-ja/tagged-mecab-initial/jawiki-corpus-003.mecab-initial.jsonl.bz2
```

### chunk003 POS細分化 + POS-WORD/WORD-POS run（2026-10-07）

前のlistwise学習checkpointは引き継がず、初期統計から作り直した新モデルでchunk003のみを学習する。語彙・POS IDが変わるため、旧runの正規化済みpage-block cacheは使わず、同じchunk003 JSONLを再走査する。固定testの5,000文は旧runと同じprefix定義で、新model上で採り直す。

初期モデルはword-bigram 1,500,000、word-trigram 3,000,000、POS-WORD/WORD-POS各500,000、read-length bi/tri全セルを生成する。raw unigram頻度top100の対象名詞接頭語・接尾語を個別POSに細分化し、POSを併合しない。surface `号・時・日・数・級` と数字・助数詞相当などは選択から除外する。POS-trigramは外す。交差bigramはPOS-bigram基準に経験側重み0.1で補正する。

学習設定は前回のfull chunk003 primary-N-best-listwise runと同一: 16 threads、batch16、N=128、2 epoch上限、epoch2はepoch1最終post-trigram top-1 missのみ、primary N-best listwise + trigram listwise、word-bi/tri勾配batch内合算、他一次特徴とread-length trigramは既存の有効文数平均、学習率0.03、L2=0、gold augmentation、missed feedback、24,576 MiB学習補正表予算、delta cutoff `1e-4`〜`1e-2`。学習delta件数cutoffはword-bigramが50,000,000件、word-trigramが100,000,000件を超えたら各表の大きい絶対値を残す半減処理を行う。

30分ごとの固定5,000文評価は、処理文数・文/秒・bigram top-1・raw N-best gold含有率・post-trigram top-1に加え、bigram段階とpost-trigram段階の文字レベル一致率を記録する。まず両方の追加素性を含めてchunk003を完走し、固定testで初期modelからの効果を確認する。その後、同じtest sampleを使ったfeature ablationでread-length bi/trigramなどの内訳を調べる。初期POS細分化とcross-bigramを独立ablationに分けない。

run output/workは `E:\\lmwork-target\\crf-wiki003-n128-primary-listwise-affix100-crossbi-20261007-run01` と `E:\\lmwork-int\\crf-wiki003-n128-primary-listwise-affix100-crossbi-20261007-run01` を使う。初期モデル作成先は `E:\\lmwork-target\\crf-initial-model-affix100-crossbi-20261007-run01\\model`、workは `E:\\lmwork-int\\crf-initial-model-affix100-crossbi-20261007-run01\\work`。

入力全体は7,811,350レコードである。初回検証は学習1,000文・検証200文、その後の規模試験は学習10,000文・検証200文まで。2026-10-03には別途、8スレッドで全chunkをEOFまで読む実行を開始した（下記「全100,000ページ実行」）。開始と完了は区別する。標準の学習文数上限は1,000文のままとし、全件実行は `--full-corpus` で明示する。N-bestの標準値は256。

## 旧runのスコア一覧（現行の指定は上記仕様）

すべて自然対数に基づく `f64` の加算スコアで、大きいほどよい。補正後の値は正規化された確率のlogとは限らず、正値も許される。「cost」の符号をMeCabに合わせて整数化することはしない。

|対象|初期値|学習値|
|---|---|---|
|POS接続|`ln P(POS_b | POS_a)`|POSペア別の加算補正|
|単語コスト|`ln P(word | POS)`|単語別の加算補正|
|word-bigram|通常は0 entry。実験では静的bigram上位N件を対数補正として初期化|ラティスに現れた単語ペアの正負の学習補正|
|word-trigram|静的初期値（chunk003は上位300万件）|キー別word-trigram deltaをN-best上で学習|
|POS/readlen trigram|静的POS/readlen初期値|N-best上のPOS/readlen三つ組にキー別学習補正|
|表記unigram|全クラス0|表記文字種クラス別の補正|
|表記bigram|全クラスペア0|前後の表記文字種クラスの接続補正|

表記素性は単語の**表記**から分類する。読みを使って全単語をひらがな扱いにはしない。クラスはBOS、EOS、ひらがな、カタカナ、漢字、記号、英字、数字、混在、その他の10種類。漢字と送り仮名の「買う」は混在クラスになる。一語に複数クラスを重複加算する設計ではなく、排他的な一クラスである。

POSは辞書コンパイラと同じ生成関数を共有する。通常はMeCab文法属性6項目 + `*`、助詞・助動詞は7項目目を正規化した表記に置き換える。助詞「が」「を」などをまとめない。POS IDはこのモデル内のローカルIDで、文字列との対応を保存する。ID 0は予約する。

読み注釈の追加品詞 `記号,読み仮名` はmodel POS `記号,読み仮名,*,*,*,*,*` として独立させる。左右接続IDは `記号,一般` と共有するが、モデルのPOS IDは別に採番する。定義・tokenの保護規則・実装状態は [品詞定義](05-static-dictionary.md#読み仮名の内部品詞2026-10-13追加仕様) を参照する。

## 初期値の構築

語彙の母集団はMeCab標準辞書CSVと `dictionary/additions` 内の登録済みCSV/TXTである。コーパスの語彙表だけに限定しない。既存compilerの語彙正規化・ソース重複除去・POS細分化をライブラリとして共有し、コーパスで観測した活用形も取り込む。

- 単語はコーパス出現countを10倍、未出現語はcount 1としてPOS内で正規化する。
- POS接続は **`2gram.by-id.tsv` 全件**を読み、POS別countを集計する。word-bigramの11 cutoff後のファイルを使わない。
- 未観測POS接続は初期確率 `1e-8` のlogを使う。`--missing-pos-probability` で変更可能。この床値は欠損に有限ポテンシャルを与えるためで、行全体を改めて確率正規化したものではない。
- k-meansを経た8-bit復号値ではなく、量子化前の実数値を取得する。
- 通常の初期統計ではword-bigramを使わず0 entryで開始する。比較実験では静的辞書のcount上位N件について、`P0=P(POS_b|POS_a)P(word_b|POS_b)`、`P1=P(word_b|word_a)` として、`log(δP0+(1−δ)P1)−log(P0)` をimmutableな初期word-bigram補正にする。これは加算CRFで線形補間を表す対数比であり、正・負どちらの値も保持する。学習deltaとは別テーブルで、L2や容量cutoffの対象にしない。
- 静的trigram初期補正を指定した実験では、count上位N件だけ `log((1-λ)P(C|B)+λP(C|A,B)) - log P(C|B)` をimmutableな初期値として格納する。`P(C|B)` は同じモデルのPOS-bigram × word-unigram-in-POSと静的word-bigram補正を合わせた一次スコア、`P(C|A,B)=count(ABC)/count(AB)`。保持しない三つ組は0補正のままとし、全候補へ `log(1-λ)` を一律加えることはしない。学習可能なtrigram deltaはこれと別Mapで0から学び、L2も0 deltaへ適用する。
- chunk003の初期値trigram実験では、静的辞書の頻度上位3,000,000組を使う。trigram補間の経験値側の重みλは指示値が未指定のため暫定0.1、bigram backoff側0.9とし、`config.json` / `verification.json` に記録する。初期bigramは既存比較に合わせて1,500,000件、POS/unigram baseline重み0.9。

数・英単語・未知語のPOS-onlyクラスも学習語彙に含める。未知語にMeCabが自動で推定した未確認の名詞POSを採用しない。記号は全角正規化した1文字単位の語に分ける。

## 読みラティスと擬似正解

入力JSONLには `page_id`, `title`, `sentence`, `tokens` が必要。各tokenは `surface`, `feature`, `reading`, `unknown` を持ち、任意の `segment_index` の既定値は0。

1. 既知語の読みをひらがな化し、連結して読み入力を作る。
2. 全ソース語彙の読み索引で、読みの全開始位置を検索する。
3. 同音異表記だけでなく、分割境界が違う長短の単語もラティスへ入れる。
4. 元token列を辞書ソースの表記・読み・POSで照合し、正解パスとして指定する。
5. BOS/EOSを含め、同じ読み位置で接続可能な単語ノードをすべて結ぶ。

通常は正解をN-bestへ追加しない。`--gold-path-augmentation` を付けた学習では、まず通常のN-bestを生成してraw hitを記録し、正解が集合外の場合に限り、既知のgold pathをtrigram学習用候補へ一つ追加する。一次勾配、推論候補、heldout評価には追加しない。したがってN-best hit率は常に**追加前の生N-best**について測る。正解語がソースにない場合にも、その語をその場で発明せず入力棄却として記録する。

未知語・英字・数には正しい日本語読みが付いていない場合がある。今回の実装はその部分だけリテラル表記の固定区間として保持し、POS-only語を置く。読みを捏造せず、その区間内の未知語の分割や読み候補の選択は今回の学習対象にしない。周辺の既知語は通常の分割競合を保つ。固定区間数を `input-report.json` に記録する。

未知語にも実際の表記の `C1` / `C2` を適用する。たとえばカタカナ未知語の次に漢字語とひらがな語が競合すれば、表記bigramで周辺候補の選択を学習できる。ただし固定区間の `C1` は全パス共通なので、その区間だけによる勾配は相殺される。既知語で学んだ表記スコアも未知語へ適用されるが、未知語の分割・長さ自体を改善する学習には、将来その代替ノードが必要である。

長すぎる文、読みのない既知語、ソースにない語、複数 `segment_index` を含む未分割文は、正規化段階で理由と件数を報告してサンプルから外す。一方、採用済み文のラティス上限超過・A*上限超過は実行を失敗させる。壊れたJSON/UTF-8/圧縮データもエラーとし、1 JSONLレコードには8 MiBの読み取り上限を設ける。入力に合わせて正解パスを黙って捨てたり、候補を上位の同音語だけに削ったりしない。

## 損失と更新手順

一次スコアは次式とする。すべての和に必要なBOS/EOS接続を含める。

```text
S1(y) = Σ U(word_i)
      + Σ T(POS_(i-1), POS_i)
      + Σ R2(word_(i-1), word_i)
      + Σ C1(script_i)
      + Σ C2(script_(i-1), script_i)

L1 = log Σ_(全ラティスパス y) exp(S1(y)) - S1(gold)
∂L1/∂θ_f = E_(p(y|reading))[count_f(y)] - count_f(gold)
```

トポロジカル順のforward-backwardをlog空間で実行し、全パスを正規化する。正解にある素性は相対的に上がり、競合にある素性は下がる。未登録のwordペアも、この勾配が非ゼロになればパラメータとして登録する。単に正解bigramの頻度を足す処理ではない。

一次勾配と並行して、逆向きViterbiのsuffix最大値をA*の上界としてN-bestを生成する。N-bestはそのバッチ開始時点の一次スコアで作り、再順位付けには同時点の一次スコアと既存trigram補正を使う。A*上界にはtrigramを混ぜない。上限超過時はエラーとし、ビーム近似に黙って切り替えない。

```text
S3(y) = S1_snapshot(y) + Σ R3(word_(i-2), word_(i-1), word_i)
L3 = log Σ_(候補 y ∈ C_N) exp(S3(y)) - S3(gold)
```

`C_N` は一次スコアsnapshotから得たraw N-best。学習時gold augmentationが有効で、goldがraw N-bestになければ、gold path一つだけを加えた集合になる（最大N+1候補）。

この二番目の勾配はR3だけに適用する。この更新中だけS1を定数として扱い、次のバッチでは再び一次CRFを更新する。同じtrigramが一パスに複数回現れた場合は、その回数で加算する。BOS付きの最初のtrigramとEOS付きの最後のtrigramを含める。

標準設定では正解がN-bestにない文にはL3を定義しない。`--gold-path-augmentation` 時はraw N-bestの後ろへ正解gold pathを加えてL3を評価する。このときもraw hit/missを別に数え、`training_gold_missing_nbest` と検証の `nbest_gold_recall` へ反映する。候補拡張により学習可能なtrigram文が増える一方、それは元N-best内の包含率が上がったことを意味しない。

一次・trigramの勾配計算とN-best再順位付けが全バッチ文で終わるまで、どちらのモデル補正も変更しない。各workerは一文内で一次勾配、N-best生成、gold条件付き拡張、trigram勾配を順に計算する。結果を入力順に集約し、一次は全バッチ文、trigramはL3が定義された文でそれぞれ平均し、両補正を最後に適用する。これにより一次更新後にN-best用workerが再同期する段階をなくす。各更新は同じ学習率・L2設定を使うが、それぞれ独立した特徴テーブルへ一回ずつ適用される。

## 最適化と評価

- 同一chunk内を元のページ順で最大1,000ページずつに分け、各epochでブロック順をseed付きshuffleする。ブロック内にも最大256文の窓内shuffleを適用する。全学習文をRAMに保持した全体shuffleではない。
- `--batch-size` の文数の平均勾配で同期更新するSGD。標準16文。スレッド数とは独立に指定できる。
- epoch 1の学習率は指定値、epoch eは `rate / (1 + 0.2*(e-1))`。
- L2は初期スコアからの**補正値**に適用する。初期log P自体を0へ縮めない。共通scaleにより、そのバッチに登場しなかった登録済み補正も減衰させる。一次はバッチごとに一回、R3は正解を含むtrigram候補集合が一文以上あるバッチで一回減衰させる。
- すべての疎な学習deltaは0を中心にL2正則化する。word-trigramを含め共有係数や絶対値上限クリップは使わない。更新値が非有限になった場合はそのbatchをエラー停止する。
- 30分のwall-clock間隔でも固定5,000文テストを再評価し、`training-curve.jsonl` とstderrへ記録する。`validation_bigram_top1_accuracy` はtrigramを適用する前、`validation_post_trigram_top1_accuracy` は再順位付け後の最終top-1、`validation_bigram_nbest_gold_cover_rate` はtrigram前のN-best正解包含率。word-trigram学習キー数と、各疎deltaの最大絶対値も同じイベントに含める。評価に時間がかかる場合、次の30分タイマーは評価完了時から数える。
- top-1誤りは、bigram段階とtrigram再順位付け後を別々に分類する。予測とgoldのtoken span `(start,end)` が違えば「境界違い」、spanが一致して表記が違えば「同音異義語・表記選択違い」、spanと表記は一致するがword ID列が違えば「同表記での語ID/POS違い」とする。各3カテゴリはtop-1誤り文数を排他的に分け、`metrics.json` / `evaluation.json` と固定heldoutの `training-curve.jsonl` に保存する。N-best gold coverは別の候補生成指標として維持する。

### 2026-10-06 word-trigram個別学習run

前runの学習済みcheckpointや途中復旧状態は使わず、固定初期モデルからコーパス先頭へ戻して新規実行する。word-trigramは初期上位300万件を固定初期値として保ち、N-best勾配から個別deltaを学習する。N-best数は128。一次word-bigramもN-bestに依存しない全ラティス勾配で引き続き学習する。

- 出力: `E:\lmwork-target\crf-wiki003-n128-word-trigram-learning-20261006-run02`
- 作業データ: `E:\lmwork-int\crf-wiki003-n128-word-trigram-learning-20261006-run02`
- 設定: 8内部スレッド、同期batch16、N=128、full chunk003、最大2epoch、固定test先頭5,000文、epoch 2はepoch 1の再順位付け後top-1 missだけ、gold path augmentation、学習補正表上限24,576 MiB、delta cutoffは下限1e-4 / 最大1e-2。
- sourceは `E:\lmwork-target\crf-initial-model\model.json`（静的word-bigram 150万、静的word-trigram 300万を含む）と元のtagged corpus。alphaフィールドがある旧形式を読む場合も値は無視し、保存する新形式にはalphaを含めない。
- 損失は正解候補の特徴量とN-best競合候補の期待特徴量の差。誤候補だけに出るbigram/trigramは正の勾配となり、勾配降下でそのスコアが下がる。正解だけに出る特徴は逆向きに補正され、両者を比較する基準として使う。
- ページIDの固定hashにより約20%のページを検証側にする。同じページの文を学習・検証両方へ入れない。
- 指標は一次NLL、一次1-best完全一致、N-best正解包含率、trigram再順位付け完全一致。word ID・境界・POSを区別する完全一致と、表記単語列だけの一致を分ける。
- N-best損失は正解が集合内にある文だけの平均。集合外の文を勝手に損失0として平均へ入れない。最終正解率の分母には全検証文を含める。
- 学習側の評価は最大 `--max-eval-sentences` 文を毎epoch再評価する。標準256文。通常の小規模試験は先頭から、`--full-corpus` では対象全体からseed固定のreservoir samplingで選ぶ。`metrics.json` の `train.sentences` はこの評価用文数であり、更新した文数は `training_sentences_processed` に別記する。
- run02の進捗中バイナリはtop-1誤り分類を出さないため、固定5,000文の分類値は学習完了後に保存済み `test-sample.json` と最終 `model.json` を使ったevaluation-only再評価で追加する。学習自体は分類のために停止・再起動しない。

### 実験的なscc/Hogwild更新

通常の同期SGDとは別に `--hogwild` を付けると、固定スレッドpoolがshuffle窓から文を一つずつ取り、各workerがその文のラティスを作成・評価した後、共有 `scc::HashMap` へ一次・trigram補正を即時反映する。各勾配の非ゼロ素性について、該当する値は更新ごとに実際に書き換える。`update_sync` によるマップ内の同期更新を使うため、同一キーへの同時更新を落とさず、Rustの安全なAPI内で競合を処理する。これは更新を省く最適化ではなく、全表ロックやバッチ同期を避ける更新方式である。各文のグラフを破棄してから次の文へ進み、窓の終了時だけjoinして進捗集計・評価・checkpoint snapshotを行う。

これは更新方式だけの実験で、目的関数を変更しない。一次は元の全ラティスforward-backward期待素性−gold勾配を用い、raw N-best集合に依存しない。N-bestはtrigram補正の学習だけに使い、gold augmentationもそのtrigram候補集合だけに適用する。一次補正へN-best素性差分を混ぜない。各文のN-best pathはその文のedge scoreを集めた後に確定し、同じedge-score配列でtrigram補正する。他workerによる同期更新は実行中にも反映され得るため、一文内の全edgeが同一時刻の完全なモデルsnapshotを参照する保証はない。これはHogwildの近似である。なお、全テーブルを毎文コピーしてはいないが、勾配に現れた各パラメータの更新自体は省略していない。L2のglobal scaleだけを遅延適用し、未アクセスの値の全件書き換えを避ける。

1文ごとの勾配の大きさを元の `batch-size` 同期更新に合わせるため、一次・trigram双方に `learning_rate / batch_size` を適用する。したがって `batch-size` はHogwildでは同時に処理する勾配batchではなく、学習率の基準とshuffle窓内の設定である。各shuffle窓の完了時に累積top-1率とraw N-best gold包含率をstderrへ出す。`training-curve.jsonl` は設定された評価間隔とepoch終端でheldout指標を記録する。

安全のため現時点の `--hogwild` は、新規run・最大10,000学習文限定で、`--full-corpus`、resume、delta cutoffとの併用を拒否する。exact resume、delta剪定、RAM全体の上限保証はないため、10,000文を越える本学習には使わない。特に `--max-parameter-mib` は通常SGD側の確保前検査であり、Hogwild側の実メモリ上限としては未実装である。

2026-10-04の試験は静的word-bigram初期補正上位150万件（δ=0.9）、N=32、gold-path augmentationあり、同じ初期モデル・seed・8 thread・batch-size 16・10,000学習文・2,000 heldout・delta cutoffなしで同期版およびPapaya版と比較した。3方式とも同じMeCab擬似正解、同じ初期checkpoint、同じ固定heldout標本を使う。

|指標|同期版|Papaya Hogwild|scc Hogwild|
|---|---:|---:|---:|
|ベンチマーク全体時間|77.70 s|290.54 s|130.24 s|
|学習更新時間|46.39 s|224.42 s|48.92 s|
|peak working set|811 MiB|4,290 MiB|3,575 MiB|
|heldout top-1|60.65%|60.65%|60.65%|
|heldout raw N-best hit|90.00%|90.00%|90.00%|
|学習中 raw N-best hit|85.52%|85.53%|85.53%|
|学習中 top-1|56.69%|56.70%|56.71%|
|補正テーブル|311 MB|1,167 MB|1,167 MB|

3方式ともcheckpoint roundtripは成功し、heldout top-1/N-best率は同じだった。scc版の学習更新はPapaya比で約4.6倍速く、全体時間は約2.2倍速く、peak working setは約17%小さい。同期版と比べた更新時間は約5%増だが、全体時間とピークメモリは大きい。8 threadで同じキーを競合更新する追加テストでは16,000回の加算が欠落しないことも確認した。これはデータ競合・更新落ちを避ける実装確認であり、Hogwild特有の文間更新順序の近似までなくすものではない。MeCab擬似正解の小規模試験であり、性能向上とは結論しない。

この10,000文runで学習された補正テーブルは約1.17 GB、checkpointは約1.83 GBだった。Hogwildでは `--max-parameter-mib` が強制されず、正確なresume・delta cutoffもない。したがってこのベンチは並列コンテナの選定に限り、1.6M文の一epoch完走可能性やメモリ安全性を示さない。全件runを開始する前に、実運用の24 GiB上限・保存/再開条件を満たす同期学習側のスケール試験を行う。

**評価の限界:** 正解列はMeCab由来の擬似正解であり、人手で確認した変換の正解ではない。また今回の初期POS/単語統計は全コーパス由来なので、検証ページも初期値の集計に含まれる。ページ分離はCRFの補正学習だけに適用される。今回の数値は「コードと補正学習の動作検証」であり、完全な未見データ性能や人間の変換意図の正解率とは主張しない。

## 実行方法

リポジトリのルートで実行する。外部C++ランタイムやLLMモデルのビルドは不要。bz2はRustライブラリから直接ストリーム解凍する。MeCab出力の再生成や保存はしないが、ページブロックをランダム順に読むために、正規化済みの軽量な正解spanを圧縮キャッシュに保存する。ラティスは最大一バッチ分を構築・更新・破棄する。

```powershell
cd D:\Git\codesnipets\aiEngine
cargo test -p ai_engine_crf -p ai_engine_dict
cargo clippy -p ai_engine_crf -p ai_engine_dict --all-targets -- -D warnings
cargo build --release -p ai_engine_crf

.\target\release\ai_engine_crf.exe `
  --input corpus/wiki-ja/tagged-mecab-initial/jawiki-corpus-003.mecab-initial.jsonl.bz2 `
  --corpus-statistics target/dictionary-build-userdic-20261003-run01/global `
  --system-dictionary 'C:\Program Files\MeCab\dic\ipadic' `
  --user-additions dictionary/additions `
  --output target/crf-wiki003-run-new `
  --epochs 1 --max-sentences 10000 --max-validation-sentences 200 `
  --nbest 256 --max-records 200000 --seed 20261003 `
  --shuffle-buffer 256 --max-eval-sentences 1000 `
  --max-parameter-mib 1024 `
  --threads 16 --batch-size 16 --page-block-size 1000
```

既存の出力がある非空ディレクトリは拒否する。毎回新しい出力先を指定する。`--help` で全オプションを表示する。主要安全上限は `--max-words 48`, `--max-nodes 20000`, `--max-edges 1000000`, `--max-astar-expansions 500000`。`--max-records` は採用文数ではなく読み取りレコード上限である。

同じ学習済みモデルを読む場合は `--model-in <出力先>/model.json` を指定する。この場合、初期統計の再読み込みは不要。`--epochs 0` で学習せず評価・再出力できる。ただし別コーパスの語彙を勝手に追加する機能や、連続する複数chunkを一回のepochで学習する運用までは今回実装しない。

既存の正規化済み固定標本だけをN-best評価する場合は、`--evaluation-sample-in <run>/test-sample.json --epochs 0 --model-in <initial-model.json>` を指定する。原コーパスの再走査や再正規化をせず、同じword ID列・同じheldout標本でNの値を比較できる。この経路のN-best hit率はgold augmentationなしのraw候補について測る。`evaluation.json` の `metrics.seconds` はwall time、`nbest_rerank_worker_seconds` は各文workerの所要時間合計（8並列なら壁時計時間とは一致しない）。

`--threads` の標準値は利用可能な論理CPU数、最大16。`--batch-size` は標準16で、`--shuffle-buffer >= --batch-size` が必要。窓やページブロックの末尾は実際に残った文数で平均する。末尾の文を捨てない。逐次実行は `--threads 1 --batch-size 1 --page-block-size 0`。ただしこの場合も一次・trigram勾配は更新前の同じモデルから計算してから適用する。

### ストリーム処理とメモリ管理

最初の走査で対象範囲・採用件数を決め、軽量な正解spanだけの学習評価用サンプルと検証文を保持する。ページキャッシュはこの走査中に作り、同じ原入力をキャッシュ作成のためにもう一周は読まない。各epochではキャッシュを読む。正規化済みの最大 `--shuffle-buffer` 文を窓内shuffleし、最大 `--batch-size` 文のラティスを構築する。語彙・読み索引と学習済みパラメータは一組だけを共通で保持する。評価も同じ内部スレッドpoolで文ごとに並列化し、集計順は固定する。

補正値の格納はtreeから整数キーのHashMapへ変更した。一次素性のキーは種類と24-bit IDを `u64` に詰め、trigramは3つのIDを保持する。キー本体で同一性を確認するHashMapであり、ハッシュ値の自力比較や、衝突時の素性統合は行わない。学習値は `f64` のまま保持し、計算時の量子化・小さい補正の剪定はしない。

`--max-parameter-mib` は**学習可能な補正テーブル**のバケット容量に対する安全上限で、標準512 MiB。immutableな静的bigram初期補正テーブルは別に保持されるため、この値には含まれない。一次・trigram勾配の計算後、更新前に、そのバッチが実際に新規登録しようとするdistinct素性数から必要容量を検査する。gold augmentation時に追加されたgold pathのtrigramも勾配に含む。上限を超える場合はバッチ全体を更新せず、直前まで確定したモデルと `training-stop.json` を出力し、終了コード1で停止する。未処理窓・入力位置・乱数状態・ブロック順と現在のブロックも記録する。特徴を黙って捨てて処理を続けない。

この上限はプロセス全体のRAM上限ではない。語彙・読み索引・バッチのラティスと勾配・集約配列・更新を取り消すための変更キーの退避・rehash中の新旧バケット・保存/再読込時の一時配列などは別途メモリを使う。HashMapの容量は段階的に増えるため、512 MiB設定でも、使用容量322 MiBから次の拡張594 MiBが必要になる地点で停止する場合がある。

N-best/A*エラーなどでどちらかの勾配計算が失敗した場合、まだどちらの補正も適用していない。容量確保・更新段階の異常もrollbackし、一次だけ・trigramだけが適用された半端なバッチを残さない。普段は変更対象キーと共通scaleだけを退避し、全モデルを複製しない。L2の極小scaleを全テーブルへ反映する稀な再正規化時だけ、該当補正テーブルを退避する。

容量上限だけでなく、ラティス構築・学習エラーで止まる場合も `training-stop.json` と `stopped-model.json` を、後続の評価より先に保存する。評価自体が失敗しても直前まで確定した更新は失わない。キャッシュ読込などepoch全体のIOエラー時には `emergency-stop.json` / `emergency-model.json` を保存する。OSによる強制終了・電源断・書込先故障まで救済する仕組みではない。未処理文を黙って飛ばして「全件完了」としない。

保存時は語彙/初期値を借用し、補正Mapを直接JSONへstreamする。チェックポイント配列順と補正TSV行順は未規定で、読込側はIDキーで扱う。大きなMapの整列コピーは作らない。word-bigram/word-trigram TSVもMapから逐次出力し、静的初期bigramとの重複はキーlookupで統合する。往復検証は元モデルと索引を破棄してから再読込し、キー/値/初期値の診断用fingerprintと検証NLL・正解率を照合する。fingerprintは暗号学的検証ではない。

`--checkpoint-every 1` ならepochごとのcheckpointも保存する。標準0では最終モデルだけを保存する。現在の `--model-in` は重みを引き継ぐwarm startであり、停止窓・epoch内位置・学習率状態を自動で復元する**完全な中断再開は未実装**。`training-stop.json` の保存だけで自動再開できるとは扱わない。

### ベンチマーク方法

Windowsプロセスのpeak working setと500 ms間隔のprivate bytesを、学習・出力・再読込まで計測する。

```powershell
.\scripts\benchmark-crf.ps1 `
  -OutputDirectory target/crf-wiki003-benchmark-new `
  -Sentences 10000 -NBest 256 -Epochs 1 `
  -ShuffleBuffer 256 -MaxParameterMiB 1024 `
  -Threads 16 -BatchSize 16 -PageBlockSize 1000
```

学習出力の隣に `.benchmark.json`, `.stdout.log`, `.stderr.log` を保存する。既存の出力先やログは上書きしない。時間比較ではCPUを競合させずに順番に測る。

初期語彙を更新している別作業と比較試験が重なる場合、実行間でword IDや初期確率が変わる。`--save-initial-model`（benchmarkスクリプトでは `-SaveInitialModel`）で入力直後の `initial-model.json` を保存し、以後は全試験を `--model-in`（`-ModelIn`）で同じファイルから始める。スレッド数だけを比較するときはバッチ文数・入力範囲・seed・ページブロックサイズも固定する。

### 1,000ページブロックのランダム順処理

100,000ページの元chunkファイルはそのまま残す。現在のCLIは一回につき一つのchunkを入力する。ページの出現順で1,000ページずつ区切るので、全100,000ページを選択した場合は約100ブロックになる。最後の端数も保持し、一つのページを二つのブロックへ分断しない。ページ数は文数でもpage IDの数値範囲でもなく、原入力のページの切り替わりで数える。検証ページ・棄却文のページも境界のページ数に含める。

初回のストリーム走査で学習対象だけを `<output>/page-blocks/block-NNNNN.normalized.jsonl.bz2` に保存する。内容は元レコード番号・page ID・タイトル・本文・連結読み・正解span（開始/終了位置、word ID、文字種、固定区間フラグ）。MeCab featureの再出力、全候補ノード、ラティスedge、勾配は格納しない。元chunkを全展開してRAMへ載せる処理もない。キャッシュは現在のモデルのword IDに依存し、この実行内のepoch間で再利用する。別モデル・別実行のキャッシュを無検証で共有する機能は実装しない。

各epochの順序はseedとepoch番号から決め、`metrics.json` の `page_block_order` にブロック番号の順列を記録する。ブロック内の窓もshuffleし、ブロック末尾で窓を処理し切ってから次のブロックへ進む。同一ページのレコードが非連続に再登場する入力は、ページを誤って分割するのを避けるためエラーにする。

`--max-sentences` / `--max-records` を低く設定した試験では、選択した先頭範囲だけをキャッシュする。1,000文が1,000ページになるわけではなく、選択範囲が1,000ページ未満なら一つの部分ブロックになる。全chunkをランダム化する場合は `--full-corpus` を指定する。これは文数/レコード数上限を無制限にし、EOFまで走査する（`--skip-records` とは併用できない）。`--page-block-size 0` ではキャッシュとページ順shuffleを無効にし、元入力の再走査方式になる。

### 全100,000ページ実行・テストコーパスのルール

2026-10-03開始の入力はchunk 003全体。学習対象を先頭数千文に切らず、EOFまで走査する。ただし従来の入力品質条件（最大48語、読み・辞書対応・segment条件など）は維持する。全ページを読むことと、全レコードを無条件で正解として学習することは違う。棄却理由・件数は `input-report.json` に残す。

分割ルール:

1. `corpus::validation_page(page_id)` の固定hashで約20%のページをテスト側、それ以外を学習側にする。chunkやスレッド数・shuffle seedに依存させない。同じpage IDの文は必ず同じ側へ入る。
2. ページID集合を `corpus-split.json` に保存し、集合が交差しないことを確認する。元レコード番号もキャッシュに保存する。
3. テスト側で正規化できた**全レコード**を `test-corpus.normalized.jsonl.bz2` に保存する。テスト側には一次/R3の勾配計算・重み更新・学習窓shuffleを実行しない。
4. 毎epochの指標計算は、全テスト側から均一な文単位reservoir samplingで選んだ固定2,000文を使う。全テスト側を毎回評価した値とは呼ばない。学習側の評価用probeも全学習対象から固定1,000文を採る。両方とも独立の固定乱数を使い、学習順の乱数状態を消費しない。
5. 評価用の `test-sample.json` / `training-probe.json` を保存する。メモリに保持するのはこれらの固定サイズ標本、最大256文の学習窓、最大16文のラティスと勾配であり、781万文全部ではない。

これは**CRF更新のheld-out検証用コーパス**である。初期統計はテストページを含む全コーパスから作成され、正解もMeCabの擬似正解。過去の同chunkでの検証や今後の調整にも使用するため、完全に未使用の最終テストセットとは主張しない。本番の未見性能を評価するには、初期統計と独立した別コーパスを必要とする。

実行条件は8 thread、16文batch、N=256、1,000ページblock、窓256文、1 epoch、学習率0.03、L2 0.001。学習中の素性削除・量子化・beamへの切替はしない。補正テーブル上限16,384 MiBを設け、上限到達時には停止を報告する。全件完了がこの上限内で可能かは実測で判断する。

```powershell
cargo build --release -p ai_engine_crf
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\start-crf-full-corpus.ps1
```

起動スクリプトは既存の初期モデルを固定して読み、今回のexeも出力先の隣へコピーして固定する。別作業による辞書CSV更新・再ビルドに引きずられない。初期モデルとexeのSHA-256、起動時空きRAM、runner PID、ログの場所を `.launch.json` に記録する。runnerを非表示の独立プロセスで起動するため、起動元のターミナルを閉じても続く。PCのスリープ/終了には依存する。

現在の有効な実行先は `target/crf-wiki003-full-threads8-20261003-run02`、開始15:10（JST）。run01はPowerShellの絶対パス結合エラーで学習開始前に失敗した記録であり、修正後のrun02とは区別する。初期モデルは `target/crf-wiki003-fixedseed-init-threads16-1k-20261003/initial-model.json`（SHA-256 `bca4af747df1a76425f410a9fdf961e9fe64e84ba5ef3628fae46a8a307ca983`）。この節の執筆時点では全件完了していない。

進捗は出力先の隣の `.stderr.log`、RAM/CPU/時間計測は `.runner.stdout.log` を見る。終了時に `.benchmark.json` が生成される。正常終了はexit code 0と `verification.json` の `status=completed`、処理文数を併せて確認する。メモリ上限・ラティス/A*エラー時には未完了であり、モデル保存があっても完了とはしない。現在はepoch途中からの完全再開を自動で行わない。

## 出力と辞書コンパイラへの受け渡し

|ファイル|内容|
|---|---|
|`config.json`|実行パラメータと入力パス|
|`input-report.json`|採用文数・ページ数、固定区間数、棄却理由・例|
|`metrics.json`|epoch 0（未学習）から各epochまでの学習/検証指標|
|`verification.json`|checkpoint往復検証、時間、実行範囲、評価の限界|
|`model.json`|初期値・全補正・共通scaleを保持したlossless checkpoint|
|`initial-model.json`|同条件比較用の実行開始時モデル。指定時だけ生成|
|`page-blocks/manifest.json`|各ページブロックのページ数・採用文数・元レコード範囲・圧縮サイズ|
|`page-blocks/block-NNNNN.normalized.jsonl.bz2`|epoch間で再利用する正規化済み正解span。学習対象だけ|
|`corpus-split.json`|全件実行のページID分割・テスト標本ルール・評価の限界|
|`test-corpus.normalized.jsonl.bz2`|全件実行のテスト側で採用された全正解span。学習へ入れない|
|`test-sample.json`, `training-probe.json`|全chunkから採った固定サイズの指標計算用標本|
|`checkpoint-epoch-NNNN.json`|指定間隔のepoch終了時checkpoint。設定時だけ生成|
|`training-stop.json`, `stopped-model.json`, `partial-training.json`|容量/ラティス/学習エラー時の停止位置・未処理窓・確定済みモデル|
|`emergency-stop.json`, `emergency-model.json`|epochのIO等による緊急停止情報・確定済みモデル|
|`vocabulary.jsonl`, `pos.tsv`|学習ローカルword/POS IDと文字列対応|
|`word-cost.tsv`|word ID、初期log P、delta、合計score|
|`pos-bigram.tsv`|全POS接続の初期log P、delta、合計score|
|`word-bigram.tsv`|word IDペアと加算補正score。未記載は0|
|`word-trigram.tsv`|word ID三つ組と加算補正score。未記載は0|
|`surface-unigram.tsv`, `surface-bigram.tsv`|表記クラスと加算補正score|

確率モデルのword-bigram→POS-bigramのbackoff alphaは使わず、6種類のポテンシャルを加算する。R3も確率的backoffではなく再順位付けへの加算である。

これらは将来のCRF対応コンパイラの入力境界となる。**既存の確率辞書コンパイラへそのまま渡すことはできない。** 既存codecは正規化確率やbackoff・8-bit log Pを前提としており、今回のsigned correctionは別の意味を持つ。量子化・小さい絶対補正のcutoff・学習word IDからvocabulary record offsetへの再配置・trigram親の保持・表記table格納は、コンパイラ側の後続実装で扱う。今回のツールは `.sdict` や `user.dic` を上書きしない。

## 正しさのテスト

小ラティスの全パスを列挙した値と照合する。

- 全パスlog Z、全edge marginal、ノード内の流量保存。
- 一次側5種類すべてとtrigramの勾配を中心差分で照合。
- 単語分割の異なるパスを含め、Viterbi+A*のN-bestを全列挙順位と照合。
- 一次更新後のN-best再生成、N-bestへ正解を強制注入しないこと。
- 初期wordペア・tripleが空であること、正負両方の補正が発生すること。
- 同じ最後のbigramでは区別できない二つの文脈をtrigramで学習できること。
- BOS/EOS、反復trigram、大きい負スコア、未出現パラメータのL2。
- 未知語の未確認POS不採用、数字・英単語、記号単位、助詞の細分POS。
- 連結bz2 memberと日本語UTF-8、壊れた圧縮入力の拒否。
- checkpoint保存/復元後のスコア一致と、6種類のテキスト出力。
- 診断を省いた逐次更新が、元の一文更新と補正値・L2 scaleまでbit一致すること。
- 表記unigramの固定未知語区間での相殺と、未知語から競合する周辺語への表記bigram勾配。
- メモリ上限で未完了文の更新・素性削除を行わないこと、整数キーの往復と容量見積り。
- 複数shuffle窓の処理、学習評価用サンプル数と更新文数の分離、epochごとのcheckpoint。

CLI自身も出力checkpointを再ロードし、検証NLL・最終正解率が変化しないことを確認してから完了する。

## 初回実行結果（旧・全サンプル保持方式）

実測結果は `target/crf-wiki003-20261003-run01/metrics.json`、入力棄却の内訳は同ディレクトリの `input-report.json` に記録した。2026-10-03実行、release build、乱数seed 20261003、N-best 32。

- 語彙423,448（NULL/BOS/EOSを含む）、POS 977（予約ID 0を除く）。
- 入力先頭1,246レコードを読み、36ページの学習1,000文、4ページの検証200文を採用。
- 棄却は48語超等の文26件、読みのない既知語1件。検証上限が埋まった後など、採用不要のレコードはこれとは別に読み飛ばしている。
- 固定した未知語・数字・英字区間は合計979 token。
- 初期統計読み込み16.7秒、サンプル解凍・ラティス構築1.28秒、5epoch・評価・出力・再ロード検証を含む合計約282秒。

|epoch|学習・一次NLL|学習・再順位付け完全一致|検証・一次NLL|検証・一次完全一致|検証・再順位付け完全一致|検証・N-best正解包含率|
|---|---:|---:|---:|---:|---:|---:|
|0|3.665482|44.6%|3.044910|55.0%|55.0%|90.5%|
|1|2.422036|67.9%|2.664483|61.0%|63.0%|91.5%|
|2|2.169999|70.6%|2.569192|62.5%|63.0%|86.5%|
|3|2.038032|72.2%|2.557073|63.5%|64.0%|86.5%|
|4|1.949969|73.6%|2.537973|63.5%|64.0%|92.0%|
|5|1.871765|74.5%|2.516429|63.0%|63.5%|86.0%|

最終出力はepoch 5のモデルであり、検証結果を見て最高値のepochを選び直していない。損失は改善するが、N-best包含率は単調ではなく最終86%。R3は候補外の14%を救えない。これは一次候補集合とtrigram学習の間にある実測上の制約であり、今後N-best数や一次モデルの汎化を検討する際の指標になる。

最終補正登録数は単語27,942、POS接続40,777、word-bigram 2,676,760、word-trigram 42,731、表記unigram 6、表記bigram 48。全ラティスの競合ペアにも期待値勾配が付くため、1,000文の正解列にあるペア数よりword-bigramが多い。非常に小さい補正も保持しており、コンパイラでの絶対値cutoff前の生の学習結果である。checkpoint約209 MB、6種類のscore TSVとID対応などを含め出力全体は約461 MB。配布用辞書のサイズではない。

この初回の数値は全学習文のshuffleと診断用の一文内再評価を含む旧方式である。以下の窓内shuffle方式とは学習順序が異なるため、モデルの結果をbit一致させる比較には使わない。

## ストリーム方式の規模試験（2026-10-03）

すべてrelease build、seed 20261003、窓256文、1epoch、N-best 256、学習評価用サンプル1,000文・検証200文。初期語彙と統計は初回と同じ。全工程時間とメモリは辞書初期値読込・学習・評価・全TSV出力・checkpoint再読込検証を含む。

|学習指定文数|実際の更新文数|補正容量上限|更新時間|全工程時間|補正テーブル容量|peak working set|結果|
|---:|---:|---:|---:|---:|---:|---:|---|
|1,000|1,000|512 MiB|12.42秒|49.76秒|80.50 MiB|609.78 MiB|完了|
|10,000|6,591|512 MiB|124.95秒|175.32秒|322.00 MiB|885.05 MiB|次の容量拡張前に停止|
|10,000|10,000|1,024 MiB|202.89秒|257.66秒|644.00 MiB|1,391.57 MiB|完了|

512 MiB試験の次文は594 MiBのバケット容量が必要と判定され、重み更新前に停止した。停止モデルの保存・再読込検証も通った。1,024 MiB試験のsampled private bytes最大値は1,716.65 MiBであり、補正テーブル容量644 MiBだけがメモリ使用量ではない。

10,000文試験では先頭14,627レコードから160ページの学習文と4ページの検証文を選択した。正規化棄却は長さ制限等596件、読みなし20件、語彙ソース照合失敗21件。一文の最大ラティスは4,815ノード・189,733 edge。途中でラティスを保持し続ける処理はない。

|更新文数|word-bigram補正数|word-trigram補正数|検証・一次NLL|検証・再順位付け完全一致|検証・N-best正解包含率|
|---:|---:|---:|---:|---:|---:|
|未学習|0|0|3.044910|55.0%|93.5%|
|1,000|2,587,437|189,758|2.693770|63.0%|94.0%|
|6,591（容量停止）|14,516,274|1,347,944|2.609778|61.5%|95.5%|
|10,000|19,752,366|2,027,842|2.668667|59.5%|95.5%|

学習文数を増やすだけで固定検証200文の精度が単調に上がる結果ではなかった。検証は先頭の4ページに偏り、統計初期値にも同ページが含まれる。これを実用変換の精度比較とは扱わない。

実行記録:

- `target/crf-wiki003-stream-1k-n256-20261003`
- `target/crf-wiki003-stream-10k-n256-20261003`（容量停止）
- `target/crf-wiki003-stream-10k-n256-1024m-20261003`（完了）

### N-best 32と256の同条件比較

学習1,000文・窓256文・1epochをN=32でも実行した。一次パラメータのfingerprintと検証一次NLLは両実行で一致し、R3だけがNの影響を受ける構造を確認した。

|N|更新時間|trigram補正数|検証・一次NLL|検証・再順位付け完全一致|検証・N-best正解包含率|
|---:|---:|---:|---:|---:|---:|
|32|7.26秒|35,515|2.693770|63.0%|91.5%|
|256|12.42秒|189,758|2.693770|63.0%|94.0%|

候補包含率は改善したが、この1epoch試験の再順位付け完全一致は同じだった。N=32側の出力は `target/crf-wiki003-stream-1k-n32-20261003`。

### 現在のスケール上の限界

全ラティス保持は解消したが、一次CRFの期待値勾配は正解にない競合wordペアにも付く。補正の累積が10,000文で約1,975万wordペアまで増えており、781万文をこのまま処理できるとは確認していない。コンパイラ出力時のcutoffだけでは、学習中のメモリ増加を解決しない。学習時の小補正削除・素性選択・外部記憶のどれを採るかは、勾配と精度への影響を別途検証する必要がある。今回は数値を維持し、上限時に停止する方式までとした。

速度も文の曖昧性とテーブル増大に依存する。1,000文から10,000文への単純な一定時間外挿や、8並列なら必ず8倍速になる見積りはしない。

## 内部スレッド並列の実装

`parallel::Trainer` は、一プロセス内で存続する専用のRayon thread poolを使う。起動時に `--threads` 個のworkerを作り、文ごと・バッチごとにOSスレッドを作り直さない。グローバルpoolは変更しない。スレッド数指定とpool内での処理実行は [Rayon ThreadPool](https://docs.rs/rayon/1.12.0/rayon/struct.ThreadPool.html) のAPIを使用する。

実装したのは**共通モデルを使う同期mini-batch**。ここで「同期」とはバッチ境界でのみモデル更新を確定する意味であり、一次計算後にN-best計算前の中間同期は置かない:

1. 同じモデルと読み索引を読み取り専用で共有し、バッチの各文のラティスを並列構築する。
2. 各workerが一文について、同じ未更新モデルからforward-backward/一次勾配を計算する。
3. 同じworker内で、その未更新モデルからN-bestを生成し、現行R3を加えて再順位付けし、trigram勾配を計算する。`--gold-path-augmentation` 時はraw N-bestにgoldがない場合だけ、trigram学習候補にgoldを一つ足す。
4. 全文の両勾配が計算されて一回joinした後、文順・素性順を固定して集約する。一次勾配は**バッチ内の実文数**で平均する。trigram勾配は損失が定義された（gold入りの）学習候補数で平均する。
5. 一次補正とtrigram補正をそれぞれ一度更新する。trigram対象が0文ならR3更新・L2減衰をしない。gold拡張時の「対象文」には、raw N-best missからgoldを追加した文も入る。

通常の同期モードでは二種類の勾配とtrigram候補評価を終えるまでモデルを不変に保つ。join後の集約・更新とworkerの読み取りが重ならない。Rustの共有参照/排他的参照で区切り、workerごとのモデルコピーや各文ごとのモデルlockは使わない。Hogwildは既述の実験オプションに限って使用する。正常時の同期集約は文の完了順ではなく入力順であり、スレッド数を変えても同条件の学習値は変わらない。異常時のエラーも文順で選ぶ。

モデル8個や16個を独立学習して後から合算する方式ではない。独立学習は各モデルのラティス分布・N-best・L2状態が異なり、合算/平均しても現在の同期学習と同じ結果にはならない。

バッチ文数を変えると、スレッド数だけを変える場合とは違い、学習手順そのものが変わる。16文の平均勾配・L2一回は、一文更新16回と同じではない。標準学習率0.03をバッチ数に合わせて自動16倍する処理は入れていない。学習率・epoch数の調整と精度比較は別途必要であり、並列化が精度向上を保証するとは扱わない。

検証済み事項:

- 一文batchの一次/trigram勾配が、同じ未更新モデルから独立に計算した参照値と重み・L2 scaleまでbit一致する。
- 同じバッチを1 / 2 / 8 / 16スレッドで処理すると重み・L2 scaleまでbit一致する。
- 一次・trigram両方の平均勾配を同一の更新前モデルから計算し、両方を最後に適用した重みが逐次計算の参照値と一致する。
- A*上限超過と補正メモリ上限でバッチを部分更新しない。稀なL2再正規化時のrollbackも検証する。
- gold拡張なしではN-bestに正解がない文をR3 loss/gradient/L2平均に混ぜない。拡張ありではgoldを一つだけ追加し、raw hit率は追加前の候補で数える。
- 16workerの実スレッドIDが異なり、結果の返却順は入力順になる。
- 1,001ページ×2文が1,000ページ/2,000文と1ページ/2文に分かれ、ページが分断されない。
- 100ブロックの順列がseed付きで再現でき、epochごとに生成される。
- CLIでも複数ページブロック、窓の端数、epoch checkpoint、圧縮入力、保存/再読込を通す。

unit testは31件、CLI integration testは1件、既存辞書の回帰testは20件、合計52件。両crateの全targetに対してClippy `-D warnings` が通る。追加したreservoir testは容量・seed再現性・先頭だけに偏らないことを確認し、CLIでも全件指定が低い文数/レコード上限を解除してEOFまで読み、テスト側全文を保存しながら標本数は制限されることを確認する。

### 内部並列の実測（2026-10-03）

CPUはRyzen 7 5800X、物理8 core / 論理16 thread。初期モデルを `target/crf-wiki003-fixedseed-init-threads16-1k-20261003/initial-model.json` に固定し、比較試験では全てこれを読む。語彙423,453、POS 977。CSVから実行ごとに初期値を再生成した予備試験は、その間に登録語彙が更新されたためbit一致の判定には使わない。

学習1,000文・検証200文、N=256、16文batch、窓256文、1epoch。複数ブロックの実行を確認するため、この小試験だけはページブロックを10ページに設定した。原入力40ページから4ブロックを作り、順序は `2,1,0,3`。ブロックの端数も含め64バッチ、学習更新1,000文を完了した。

|内部スレッド|更新時間|全工程時間|peak working set|
|---:|---:|---:|---:|
|1|12.40秒|33.24秒|356.20 MiB|
|8|5.89秒|13.89秒|487.24 MiB|
|16|5.91秒|13.35秒|629.18 MiB|

全工程はモデル読込・評価・score TSV出力・checkpoint再読込を含む。更新時間は1→8/16で約2.1倍速になったが、8→16の増速はこの小試験では観測しなかった。スレッド数とともに各workerの作業領域も増えるので、16 threadが常に最速とは主張しない。

3実行の `model.json` はファイル全体のSHA-256が一致した:

```text
31d76d839ff8eb69b8c6a78c1d1215cbb751adcc9ae032534cc818e465343b45
```

検証NLLも `2.9575549953919347`、再順位付け完全一致55.0%、N-best包含率93.5%で一致した。平均勾配の16文batchは逐次SGDと学習率/更新回数が異なるため、旧一文更新の63%と同じ精度になるとは扱わない。今回の目的は並列計算・更新・再現性と速度の検証であり、学習率の最適化ではない。

再実行記録は `target/crf-wiki003-fixedseed-threads{1,8,16}-batch16-1k-20261003`。1,000文の正規化済みブロックキャッシュは全4ファイルで約82 KBだった。キャッシュ容量はMeCabの全feature出力や全ラティスのサイズではない。

続けて16 thread、16文batch、N=256、ページブロック1,000、補正上限1,024 MiBで10,000文・625バッチを完了した。更新約94秒、保存/評価/再読込を含む全工程約117秒、peak working set約1.30 GiB。検証NLLは約2.87916、再順位付け完全一致58.5%、N-best包含率94.5%。モデルの保存/再読込検証も通った。

出力は `target/crf-wiki003-fixedseed-threads16-batch16-10k-20261003`。この採用範囲には原入力193ページしかないため、1,000ページ設定では一つの部分ブロックとなる。100,000ページ全体のランダム順学習を実行した結果ではない。キャッシュは1,254,588 bytes（約1.20 MiB）。ページ1,000単位の境界・全ブロックの順列は別途unit/CLI testで確認している。

## AI補正コーパスの試験入力（2026-10-03）

初期MeCabタグの既知語も含めて主AIが1,000文をレビューし、文脈に基づく読み・境界・品詞を直接補正する手順を タグ付きコーパスのAI補正手順（今回の統合対象外のデータ作成資料） に保存した。標本から415文の採用候補、うち未確定の英数字・数詞読み合成を含まない190文の読み→表記候補を作成した。全1,000文を正解化したことにはせず、除外・保留理由も保存する。

保存先は `target/tagged-corpus-main-review-015-1000-20261003/`。AIが確定したtokenはunknown=falseとし、旧語彙への存在判定をbaseline_lexicon_match、元のMeCab未知語フラグをsource_unknownに分ける。旧語彙にない確定語の115キーも補助語彙として出力した。学習へ渡す場合はこの語彙を含めて正解パスを表現できるか確認し、MeCabが再現できないという理由で補正コーパスを元の誤解析へ戻さない。この節はデータ補正試験の記録であり、上記の既存CRF試験がこのAI補正入力で実行済みという意味ではない。

## 計算方式を固めるための効率化（2026-10-03〜04）

現段階は仮の参照コーパスで、効率よく計算する方法とメモリ管理を検証する段階である。以下のNLL・一致率は計算の回帰チェック用の参考値であり、実用IMEの精度評価ではない。初期統計にもこのコーパスが含まれ、MeCabの仮タグを正解としているため、真の未知データに対する性能を示さない。学習率・epoch数の最適化、小補正cutoff、量子化、素性選択は今回変更しない。

### 実装した変更

- worker内の期待値勾配は、文字列ではなくpacked IDキーのハッシュ表で集計し、終了時にキー順へソートする。参照のedge/path順で加算するため、浮動小数点の計算順を維持する。
- worker間の勾配は、ソート済み列のk-way mergeで集約する。heapには各文一つだけcursorを入れ、同じキーでは文の入力順で加算する。集約用の大規模ハッシュ表や再ソートは不要。処理済みの勾配列は直ちに解放する。
- forward/backward、スコア、勾配ハッシュ表、trigram count、A*作業領域は専用workerごとに再利用する。学習で未使用の全edge marginal列を作らない。モデルは共有参照であり、worker別モデルは作らない。
- 大きいラティスでは単語項をnode単位、表記項をクラス単位で計算し、edge間で再利用する。POS・単語ペア項はedgeごとに計算する。局所スコアは毎回現在のモデルから再構築し、モデル更新をまたいで古い値を保持しない。
- A*の探索途中では全経路Vecを分岐ごとにコピーせず、親prefixを指すarenaを使う。完成したN-bestだけnode列を復元する。boundとnode列の辞書順によるtie-breakは参照実装と同じ。探索上限でエラーにする方針も維持し、beam pruningは追加しない。
- 新しい一次モデルから生成したN-bestと同じモデル借用の間にR3を計算する。trigramの出現数を一度だけ計算し、スコアと勾配の両方に使う。外部から渡されたPathのキャッシュ値を信頼せず再計算する公開`rerank`には変更を加えない。
- rollback用の旧値読込結果を新規キー数の算出にも使い、巨大な補正表への重複lookupを減らす。rollbackのキーもpacked IDにする。L2 scaleとバッチの原子性は維持する。
- checkpoint読込はJSON配列を逐次ハッシュ表へ直接投入する。全重みVecと完成ハッシュ表の二重保持をなくす。重複・24-bit超過ID・不正なfeature・非有限値は拒否する。保存JSONの形式・並び・数値は変更しない。
- TSV出力ではソート済みのraw値にscaleを掛け、同じ重みをハッシュ表から再度引かない。

`metrics.json` にラティス準備時間と5区間のwall timeを追加した。並列区間はworker CPU時間の合計ではなく、その区間全体の経過時間である。`training_update_seconds` にはキャッシュ読込・shuffle・進捗出力等も含むため、区間合計と完全には一致しない。

公開の`first_order` / `nbest` / `rerank`は参照実装として残し、`--reference-kernels`でそれらを使って学習できる。ただしバッチ集約・rollback・保存/読込は共通の新実装であり、旧実行ファイル全体を再現するflagではない。完全な旧版比較には保存済みの旧実行ファイルを使った。

### 同じ初期モデル・1万文での実測

Ryzen 7 5800X、release build、8 thread、batch 16、N-best 256、窓256文、ページブロック1,000、1epoch、補正上限1,024 MiB。初期値は既存の固定`initial-model.json`。学習10,000文・625batch、学習評価1,000文・検証200文を全実行で固定した。

|方式|学習更新時間|全工程時間|
|---|---:|---:|
|旧実行ファイル|95.33秒|119.47秒|
|旧アルゴリズム＋区間計測|99.17秒|122.95秒|
|packed勾配・ハッシュ集約|80.22秒|104.89秒|
|A* arena・局所スコア再利用追加|54.99秒|79.35秒|
|旧値lookup再利用・直接checkpoint読込追加|51.06秒|73.20秒|
|最終：ソート済み勾配のmerge追加|45.49秒|68.09秒|

最終の更新時間は旧実行ファイル比約2.10倍速、約52%短縮。各段階は単回測定で、OS・他作業の影響を含む。1万文での計測値を全コーパスの所要時間へ単純外挿しない。

|区間|旧アルゴリズム＋計測|最終|
|---|---:|---:|
|一次forward/backward・勾配（並列）|21.68秒|11.82秒|
|一次勾配集約（直列）|24.51秒|3.15秒|
|容量確保・rollback準備・一次重み更新（直列）|16.98秒|20.31秒|
|N-best生成・R3勾配（並列）|27.19秒|6.32秒|
|R3勾配集約・更新（直列）|0.69秒|0.42秒|
|ラティス準備（並列）|2.35秒|1.72秒|

一次重み更新を含む直列区間が次の主要ボトルネックになった。全区間が改善したわけではなく、intrinsics/SIMDを先に追加するより、重複lookup・集約・データ配置を見直した効果が大きかった。今回unsafeや近似指数関数は導入していない。

全段階で出力`model.json`のSHA-256が一致した:

```text
c15c2d64c48ee2fd61e196b2c08614fa5bd18b9a9d9f12d20afc0aef552f2399
```

一次/R3の平均更新NLL、全素性数、評価NLL・N-best包含率・最終一致率も一致した。保存/再読込のfingerprint検証も通った。

1万文後の補正テーブルは675,282,976 bytes（644.00 MiB）で変更前と同じ。checkpoint JSONは966,292,766 bytes（約0.97 GB）。検証一次NLLは3.04490994→2.87916025、最終完全一致55.0%→58.5%、N-best正解包含率93.5%→94.5%。補正登録数は単語82,992、POS接続86,160、word-bigram 15,570,122、word-trigram 1,926,656、表記unigram 4、表記bigram 53。

作業領域の再利用は解放/再確保の時間を減らす一方で、workerに容量を保持する。**補正テーブルの恒常容量を削減した変更ではない**。プロセスpeakとsampled private peakは各`.benchmark.json`へ記録し、容量削減を一律に達成したとは扱わない。

最終1万文試験のpeak working setは1,398,046,720 bytes（約1.30 GiB）、sampled private peakは1,445,310,464 bytes（約1.35 GiB）。旧実行ファイルのpeak working set約1.28 GiBより僅かに大きく、全体メモリの削減とは主張しない。直接checkpoint読込で二重Vec保持はなくなったが、HashMapの拡張時・ソート・worker作業領域・allocatorを含むpeakは別に計測する必要がある。

実行記録:

- `target/crf-perf-original-8t-10k-20261003`
- `target/crf-perf-profile-8t-10k-20261003`
- `target/crf-perf-packed-8t-10k-20261003`
- `target/crf-perf-arena-8t-10k-20261003`
- `target/crf-perf-final-8t-10k-20261004`
- `target/crf-perf-merged-8t-10k-20261004`（今回の採用方式）

再実行例（未使用の出力名を指定）:

```powershell
cargo build --release -p ai_engine_crf
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/benchmark-crf.ps1 `
  -OutputDirectory target/crf-perf-next `
  -ModelIn target/crf-wiki003-fixedseed-init-threads16-1k-20261003/initial-model.json `
  -Sentences 10000 -Threads 8 -BatchSize 16 -NBest 256 -MaxParameterMiB 1024
```

`-ReferenceKernels`で参照kernelを選べる。計測scriptは実行ファイルSHA-256と終了コードも保存する。Windows PowerShellで終了コードがnullになる旧scriptの問題を修正し、取得できない場合を成功扱いしない。過去のbenchmark記録は書き換えない。

効率化後の回帰testはCRF unit 38件、CLI integration 1件、既存辞書20件、合計59件。両crate全targetのClippy `-D warnings`を通した。勾配・重みのbit一致、可変分割・同点A*順位、更新後の局所スコア再構築、反復trigram、加算の桁落ち・signed zero、checkpoint不正キーを追加検証した。

### 大規模試験の保存済み状態（効率化前）

`target/crf-wiki003-full-threads8-20261003-run02`は原入力7,811,350レコード・99,970ページを読み、学習候補5,971,600文、heldout候補1,456,842文を作った。100ブロックをランダム順に学習したが、**2,431,933文・152,015batchで16 GiBの補正容量上限に達して停止**した。全件学習完了ではない。

|項目|停止時の値|
|---|---:|
|word-bigram補正|432,739,357件|
|word-trigram補正|234,760,818件|
|単語補正 / POS接続補正|260,454 / 300,372件|
|補正テーブル確保容量|15,837,691,936 bytes（14.75 GiB）|
|次バッチの必要容量見積り|22,548,578,336 bytes（上限17,179,869,184 bytes）|
|checkpoint `model.json`|33,286,633,405 bytes（33.29 GB / 31.00 GiB）|
|peak working set（旧方式）|約27.88 GiB|
|sampled private peak（旧方式）|約31.73 GiB|
|学習更新時間 / 全工程時間|約7.21時間 / 約7.69時間|

検証は全heldoutから固定seedで抽出した2,000文。効率化試験の先頭200文とは別の標本であり、以下の50.75%と上の58.5%を精度の上下として比較しない。

|指標|初期|停止時|
|---|---:|---:|
|検証・一次NLL（小さい方がよい）|3.766506|3.379723|
|検証・一次完全一致|44.90%|50.05%|
|検証・再順位付け完全一致|44.90%|50.75%|
|検証・N-best正解包含率|86.40%|87.90%|

停止モデルの保存/再読込検証は通っている。既存の`model.json`・`stopped-model.json`・停止位置ログは保全した。今回の効率化で全件ジョブを自動再開したり、cutoffを導入したり、上限を32 GiBへ変えたりしていない。停止位置情報は残るが、`--model-in`によるwarm startを厳密な停止位置からのresumeとは扱わない。

数億の競合ペアを無削除で保持する問題は、今回の約2倍速化とは独立して残っている。次のスケール試験では、一次重み更新の並列化/配置改善と、補正登録数をどう制御するかを分けて検証する。仮コーパスの精度を上げるためだけの調整は、この計算方式検証とは別に扱う。

## 停止位置からの再開（2026-10-04）

再開要求に合わせて`--resume-from STOPPED_RUN_DIR`を実装した。上の「厳密なresumeとは扱わない」は効率化完了時点の旧仕様であり、現在は以下の条件を満たすページブロック学習を再開できる。キャッシュなしの逐次入力や、停止位置を保存できなかったIOエラーからの厳密なresumeは対象外。

1. `stopped-model.json`を読み、保存済みfingerprintと重み・metadataの一致を確認する。新しい辞書ソースや初期統計から作り直さない。
2. 元の正規化済み`page-blocks/manifest.json`・各blockを再利用する。元XML/MeCabコーパスの再読込・再解析・再正規化はしない。manifestの件数・ファイルサイズ・ファイル名を検査する。
3. 元のページブロック順序、active block、元レコードの読込位置、shuffle後の未処理窓、shuffle RNGを引き継ぐ。
4. active blockの既読prefixと未処理窓を照合し、累積の確定済み文数が一致することを更新前に確認する。
5. 未処理窓は**再shuffleせず**元のbatch境界のまま計算する。その窓の後から通常のshuffle/バッチ処理を続ける。既に完了したblock・確定済みbatchは再学習しない。
6. 評価用`training-probe.json`・`test-sample.json`もそのまま再利用する。`resume-info.json`に元runと共用cacheの場所を保存するため、再度容量停止したrunからの再開も同じcacheを参照できる。

seed、batchサイズ、shuffle窓、ページブロックサイズ、N-best数、学習率/L2、max words、入力コーパス/選択方式が異なる場合は拒否する。スレッド数・メモリ上限・ラティス/A*安全上限は変更できる。出力先は新しい空directoryを指定し、元のモデルやcacheは上書きしない。

#### batchサイズを明示変更して再開する場合

通常のresumeではbatchサイズも一致させる。意図的に変える場合に限り、`--allow-resume-batch-size-change`を付けると、保存済みの未処理shuffle窓を新しいbatchサイズで分割して再開する。確定済みモデル・文カーソル・ページcache・shuffle順序/RNGはチェックポイントから復元するが、未処理窓のbatch境界と以後の勾配集約が変わるため、学習軌跡は元のbatch設定を続けた場合と一致しない。

同期学習では、各文の勾配をキーごとに合計し、実際に有効だったミニバッチ文数で割ってから更新する。したがって`--batch-size 4`なら各文の寄与は通常1/4（末尾の部分batchでは実際の文数で除算）であり、trigram側は正解を含むN-best勾配がある文の数で別途平均する。gold-path augmentationが有効な通常ケースではtrigram側も最大4文平均となる。この変更は「同じbatch16平均を単に4文ずつに切る」操作ではなく、4文ごとにモデルを更新するため、その後の評価と勾配も変化する。

この例外は明示opt-inであり、seed、shuffle窓、page block、N、学習率/L2、コーパス等の他のresume整合性検査は引き続き維持する。新しい空の出力・作業フォルダを指定し、checkpointと元runの成果物は保持する。

新しい停止ログにはモデルfingerprintと損失のraw累積値も保存する。旧ログでは損失の累積値がないため、平均NLL×文数から診断値だけを復元する場合がある。これは集計表示に限る丸め差であり、保存重み・L2 scale・勾配計算・batch順序には影響しない。

再開の回帰テストでは、4文窓をshuffleした後の2番目のbatchで停止し、残り3文の窓から再開する。停止しなかった実行の`model.json`とファイル全体が一致し、確定済みbatchを重複しないことを確認した。窓の内容を改変した場合は、追加更新前に拒否することも検証した。CRF unit 38件、CLI integration 2件、辞書20件、合計60件とClippyを通した。

今回の大規模再開は、2,431,933文の確定済みモデル、42番目のblock（ID 17）、shuffle済み未処理256文から開始する。内部8 thread、N=256、batch 16、窓256、1epochは維持。16 GiBでは次の容量拡張を許可できないため、補正テーブル上限を**24 GiB**とした。プロセス全体のメモリ上限とは異なり、拡張・ソート・作業領域で一時的に増える。cutoffは入れず、次の容量上限に達したら再び確定済みモデルを保存して停止する。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/benchmark-crf.ps1 `
  -OutputDirectory target/crf-wiki003-resume-threads8-20261004-run01 `
  -ResumeFrom target/crf-wiki003-full-threads8-20261003-run02 `
  -FullCorpus -Epochs 1 -Threads 8 -BatchSize 16 -NBest 256 `
  -ShuffleBuffer 256 -PageBlockSize 1000 `
  -ValidationSentences 2000 -EvaluationSentences 1000 -MaxParameterMiB 24576
```

途中の進捗は同名の`.stderr.log`、終了時のプロセス計測は`.benchmark.json`へ保存する。再開runの文数・batch数・更新時間は前run分を含む累積で表示し、区間別wall timeは再開後に計測した分だけ。再開機能によってパラメータ増大の問題を解消したわけではない。

### 24 GiBでの再開結果

`target/crf-wiki003-resume-threads8-20261004-run01`は、保存済み2,431,933文から450,225文を追加し、累計**2,883,158文・180,220 batch**で再び容量停止した。49番目の処理block（ID 49）内の未処理窓とモデルを保存した。全件学習完了ではない。

|項目|値|
|---|---:|
|word-bigram / word-trigram補正|469,186,472 / 267,791,149件|
|word unigram-in-POS / POS bigram補正|265,428 / 308,233件|
|補正テーブル確保容量|22,548,578,336 bytes（21 GiB）|
|次の拡張の必要容量|31,675,383,840 bytes（約29.5 GiB）|
|保存モデル|36,693,896,358 bytes（36.69 GB）|
|今回区間の更新時間|約2,781秒|
|今回区間の全工程時間|約3,898秒|
|peak working set / sampled private peak|約28.45 / 28.56 GiB|
|検証NLL（再開前 → 停止時）|3.379723 → 3.332984|
|検証・再順位付け完全一致|50.75% → 51.95%|

## delta cutoffと容量限界での継続（2026-10-04）

初期統計値ではなく、学習で増えた補正`delta`の絶対値で削除する。対象は**POS bigram、word unigram-in-POS、word bigram、word trigram、表記種別unigram/bigram**の6系統。単語初期`log P(word|POS)`、初期POS接続`log P(POS_b|POS_a)`、語彙・POS ID・キャッシュは変更しない。

- `--delta-cutoff 0`は従来どおり無削除。`1e-4`等の正の値で容量圧迫時のcutoffを有効にする。
- `--delta-cutoff-max 1e-2`は許可する最大閾値。閾値を下限から10倍ずつ広げた帯域（今回は`1e-4, 1e-3, 1e-2`）の小さなヒストグラムを作り、再確保後のテーブルが上限の75%以下になる最小の帯域を選ぶ。どれでも到達しなければ最大閾値までに限定する。
- 判定は`abs(raw_weight * L2_scale) < threshold`。正負にかかわらず対象になり、閾値と等しい値は残す。生存値のraw bitとscaleは変えない。
- 容量チェックはbatch更新前に行う。圧迫時には確定済みモデルへcutoffし、**同じbatchを一度だけ再計算**する。文を飛ばさず、L2や勾配を二重に適用しない。後続batchでも圧迫が再発したらこの手順を使う。
- 全件学習時は、削除後のハッシュテーブルを再確保・縮小せず、空いたbucketを次の同一batchのretryで再利用する。cutoff閾値は、学習補正budgetの75%以下にする最小帯域に加え、そのbatchの非ゼロ勾配キー上限を既存capacityへ収める最小帯域として選ぶ。モデル全体を複製したり、旧・新テーブルを二重に常駐させたりしない。実確保bytesは高水位のままでも、retained entry数が減り、次batchが既存capacityへ入る。空きslotsでは足りない場合、上限`1e-2`内でのcutoff後に同じbatchを一度だけretryし、なお失敗すれば位置を保存して停止する。HashMapへ予約する際にallocatorが拒否した場合も同じ回復経路を使う。24 GiBはプロセスRSS上限ではない。
- 削除したキーを恒久的に禁止しない。以後の勾配で再び補正値を登録できる。
- 最大閾値でも同じbatchが収まらない場合は、無限再試行せず従来の`stopped-model.json`と厳密な未処理位置を保存する。上限や最大閾値を勝手に増やさない。

各cutoffの前後に、同じ固定heldout標本を**更新を挟まず**評価し、`delta-cutoff.jsonl`に即時追記する。学習再開後の改善とcutoff自体の影響を混同しない。ログは実施位置、採用閾値、6系統別の削除/残存件数、削除したdeltaの最大絶対値と絶対値合計、確保bytesの前後、処理秒数、前後NLL・完全一致・N-best包含率を持つ。`metrics.json`にも当該runのイベントを保存し、`verification.json`に有効なpolicyと実施回数を記録する。初期補正なしのrunでも、最初の容量圧迫まではcutoffしない。

回帰検証は合計63件（CRF 40、CLI 3、辞書20）。L2を反映した正負・閾値境界、初期統計の保全、raw bitの保全、容量縮小、削除キーの再登録、最小帯域選択・最大閾値厳守を確認した。CLIでは131,072件の微小補正を保持したモデルを1 MiB上限で学習し、容量圧迫→cutoff→同じbatchの再計算→全8文の完了を確認した。その`model.json`は事前に同じcutoffをして学習した参照モデルと全byte一致した。無効時と、強い補正だけで収まらない場合の停止・保存も検証した。両crate全targetのClippy `-D warnings`を通した。

実データの続行例（元run・モデル・共用cacheを保全し、未使用の出力先へ保存）:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/benchmark-crf.ps1 `
  -OutputDirectory target/crf-wiki003-cutoff-threads8-20261004-run01 `
  -ResumeFrom target/crf-wiki003-resume-threads8-20261004-run01 `
  -FullCorpus -Epochs 1 -Threads 8 -BatchSize 16 -NBest 256 `
  -ShuffleBuffer 256 -PageBlockSize 1000 `
  -ValidationSentences 2000 -EvaluationSentences 1000 -MaxParameterMiB 24576 `
  -DeltaCutoff 0.0001 -DeltaCutoffMax 0.01
```

継続確認のheartbeatは、このチャットに紐づく「CRF長時間学習の継続確認」（ID `crf`、10分間隔）。稼働中の学習を重複起動しない。既知の同じ停止状態を繰り返し再起動せず、意味のある状態変化・失敗・判断が必要な場合を知らせる。ローカルファイルへアクセスするためPCとアプリは稼働状態にしておく。ユーザーが停止を求めたら継続設定も一時停止する。

### 実データでの最初の容量回復結果

上記`crf-wiki003-cutoff-threads8-20261004-run01`で、モデルの保存fingerprint・active cache prefix・未処理窓の一致を確認した後、累計2,883,158文の停止batch（元record 4,157,417）にて容量圧迫を再現し、cutoff後に同じbatchを計算し直して継続できた。採用閾値は**下限の`1e-4`**であり、`1e-3`や`1e-2`へ広げる必要はなかった。

|補正系統|cutoff前|cutoff後|
|---|---:|---:|
|word unigram-in-POS|265,428|130,530|
|POS bigram|308,233|36,665|
|word bigram|469,186,472|3,762,115|
|word trigram|267,791,149|6,633,433|
|表記unigram / bigram|5 / 57|4 / 55|
|合計|737,551,344|10,562,802|

削除は726,988,542件。初期統計と語彙は削除していない。補正テーブルの確保容量は**22,548,578,336 bytes（21 GiB）→352,321,568 bytes（336 MiB＋32 bytes）**。cutoffの選択走査2.80秒、削除・再確保7.95秒、合計10.75秒だった。再開後の勾配では新しいdeltaを再登録するため、容量はこの336 MiBのまま固定されるわけではない。

固定heldout 2,000文をcutoff前後に連続評価した結果:

|指標|cutoff前|cutoff後|
|---|---:|---:|
|一次NLL|3.3329840803|3.3329943239|
|R3 NLL（正解包含文）|1.7642495885|1.7642687671|
|一次完全一致|51.45%|51.45%|
|再順位付け完全一致|51.95%|51.95%|
|N-best正解包含率|88.05%|88.05%|
|一次表記一致 / 再順位付け表記一致|53.30% / 53.85%|53.30% / 53.85%|

この標本では集計正解率は変化せず、NLLは約`1.0e-5`増加した。個々の全候補順位の一致や別コーパスでの無影響まで保証するものではない。評価時点は同じで、cutoff間に学習更新は挟んでいない。詳細は当該runの`delta-cutoff.jsonl`に保存した。cutoff後に累計2,888,502文までの更新を確認した時点では、ジョブは引き続き実行中であり、全件完了とは扱わない。

## 静的word-bigram初期補正の実験（2026-10-04）

全件学習runを停止し、初期値の比較実験へ切り替えた。停止時プロセスは強制終了が必要で、直前の厳密な再開checkpointは累計2,883,158文。停止run・そのモデル・ページcacheは保持し、新しい初期値モデルではwarm startしない。

静的辞書の`2gram.tsv`はcount降順、count 10以上で5,855,220組。実験ではこの表をword IDへ対応づけ、同一IDペアをcount合算した後、count上位1,500,000組を選ぶ。同点はword ID昇順。分母`outgoing_count(a)`はcutoff後のbigram表ではなく、`2gram.by-id.tsv`全件から集計する。

各選択ペア`a→b`について:

```text
P0(a,b) = P(POS_b | POS_a) * P(word_b | POS_b)
P1(a,b) = count(a,b) / outgoing_count(a)
initial_delta(a,b) = log(δ*P0(a,b) + (1-δ)*P1(a,b)) - log(P0(a,b))
```

したがって一次モデルの当該遷移スコアは`log(P0)+initial_delta`となり、確率領域で線形補間した値のlogに等しい。`initial_delta`は正負どちらもあり得る。今回の暫定設定は`δ=0.9`（POS/unigram基準90%、統計bigram10%）で、初期値にする組数1,500,000とは独立の設定である。静的初期補正はimmutableテーブルに保持し、学習deltaのL2縮小・delta cutoff・24GiB学習補正上限から分離する。trigram初期値は0 entry / 0 scoreのまま。

N-best正解包含率の推移は新規runの`training-curve.jsonl`に記録する。`processed_training_sentences=0`の初期モデル評価後、学習済み文数が25万増えるごとに、同じページ分離で作った固定2,000文heldoutをN=256で評価する。heldoutの`validation_nbest_gold_hit_rate`と、訓練中に生成されたN-bestの累積`training_nbest_gold_hit_rate`を別フィールドに記録し、混同しない。評価時刻はstderrにも出す。正解ヒット率の定義は「gold token/path列が生成N-best内に含まれた文数 / 評価文数」。

進捗stderrと`training-curve.jsonl`では、訓練中の一次Viterbi top-1正解率（`training_preupdate_top1_gold_hit_rate`）と、raw N-best正解包含率を累積件数・分母付きで表示する。top-1はtrigram再順位付け前の先頭経路、N-best包含はgold追加前の候補集合で測る。定期heldout評価では一次top-1、trigram再順位付け後top-1、raw N-best包含率もそれぞれ分けて表示・保存する。gold augmentationが有効でもraw N-best包含率の分子には加算しない。

実行パラメータは既存runと同じく内部8スレッド、batch 16、N-best 256、shuffle窓256、ページblock 1,000、1 epoch、cutoff下限`1e-4`・最大`1e-2`、補正テーブル上限24GiB。新runは強制停止した旧runをresumeせず、初期モデルと入力評価標本を新規生成する。

### N=32・gold path追加と更新同期の評価（2026-10-04）

ユーザー指示により、全エッジ補正値を保持する追加キャッシュは作らず、既存の文ごとのオンデマンド採点を使用する。各文workerが未更新モデルから一次勾配を計算し、続けてraw N-best生成・trigram評価を行う。raw N-bestにgoldがなければ、学習時のみ正解path一つをtrigram候補へ追加する。バッチの全workerが返った一回のjoin後に、一次・trigram勾配をそれぞれ平均して両補正を更新する。一次更新と二次計算の間に全workerを同期しない。評価・推論はgoldを追加せず、生N-bestを使う。

N=32とN=256の速度・包含率は、同じ静的bigram 150万件入り`initial-model.json`、同じ実行前の固定heldout 2,000文、同じ8-thread release binary（同一SHA-256）で比較した。評価専用モードは既存の正規化済み`test-sample.json`を直接読むため、コーパスの再走査や再正規化を含まない。

|raw N|gold hit|miss|評価wall time|N-best/rerank worker時間合計|peak working set|
|---:|---:|---:|---:|---:|---:|
|32|88.35% (1,767/2,000)|233|1.119秒|4.269 worker秒|243.4 MiB|
|256|93.00% (1,860/2,000)|140|3.711秒|23.532 worker秒|463.8 MiB|

この標本ではN=32はN=256より**3.32倍速く**、peak working setは約220 MiB小さい一方、gold hitは4.65ポイント（93文）低い。worker時間は文worker内で測ったCPU所要時間の合計で、8並列時のwall timeではない。評価wall timeにはラティス準備、一次NLL、N-best生成と再順位付けが含まれる。gold augmentationはこの評価経路では適用していない。

続けて、同じ初期モデルからN=32、8 thread、16文batch、1,000学習文のgold-augmentation学習を完了した:

|項目|実測|
|---|---:|
|raw training N-best hit|900/1,000 (90.00%)|
|raw N-bestから外れgold pathを追加|100文|
|trigram lossを計算した文|1,000/1,000|
|バッチ数|63|
|学習更新時間|2.277秒|
|一次＋N-best/trigramの並列計算wall合計|1.139秒|
|評価後のraw heldout N=32 hit|89.95% (1,799/2,000)|
|全工程 / peak working set|16.86秒 / 337.1 MiB|

この1,000文試験は実装・経路の確認であり、gold augmentationが精度を改善したという結論ではない。学習後のheldout再順位付け完全一致は60.05%で、一次と同じだった。hit率は常にgold追加**前**のraw N-best候補で数える。評価onlyの`evaluation.json`にはwall timeとworker時間合計を別々に保存し、学習runの`metrics.json`にもN、raw hit、追加gold数、並列wall/worker時間を記録する。

再現用コマンド例（出力先は毎回新規にする）:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/benchmark-crf.ps1 `
  -OutputDirectory target/crf-wiki003-static-bi1500k-d090-goldpath-n32-eval-new `
  -ModelIn target/crf-wiki003-static-bi1500k-d090-threads8-20261004-run01/initial-model.json `
  -EvaluationSampleIn target/crf-wiki003-static-bi1500k-d090-threads8-20261004-run01/test-sample.json `
  -Epochs 0 -NBest 32 -Threads 8 -GoldPathAugmentation
```

`-GoldPathAugmentation`は学習runでのみ有効。評価モードでは設定にかかわらずgoldを挿入しない。対応するソーステストはCRF unit 42件、CLI 4件で全件通過し、`cargo clippy -p ai_engine_crf --all-targets -- -D warnings`も通過した。

### gold path追加が学習品質を上げるかの対照実験（2026-10-04）

同じ初期モデルから、N=32・1epochでgold path追加なし／ありを比較した。両runともseed `20261003`、先頭から選ぶ同じ入力範囲、同じ2,000文のpage-heldout検証セット、同じ8 thread・batch 16を使い、runは並列起動していない。初期評価の指標は全て一致しており、比較可能な対照になっている。どちらも入力はMeCab解析をgoldとする疑似教師データで、人手で正解確認したコーパスではない。

|学習文数|gold追加|raw訓練N-best hit|gold不在／追加|heldout raw hit (2,000)|heldout一次正解率|heldout trigram再順位付け正解率|heldout正解包含時の再順位付けNLL|
|---:|---|---:|---:|---:|---:|---:|---:|
|10,000|なし|8,552 (85.52%)|1,448 / 0|90.00% (1,800/2,000)|60.65%|60.65%|1.16559582|
|10,000|あり|8,552 (85.52%)|1,448 / 1,448|90.00% (1,800/2,000)|60.65%|60.65%|1.16564197|
|100,000|なし|89,357 (89.357%)|10,643 / 0|90.10% (1,802/2,000)|60.40%|60.40%|1.17108157|
|100,000|あり|89,357 (89.357%)|10,643 / 10,643|90.10% (1,802/2,000)|60.40%|60.40%|1.17110927|

100,000文runでは、gold path追加によって学習時にtrigram目的関数を計算できる文を10,643文増やした。学習済みword-trigram補正のunique feature数は3,483,700から3,978,556（+494,856）へ、補正テーブルは2,386,559,008 bytesから2,491,416,608 bytes（+100 MiB）へ増加した。checkpoint roundtripは両runで成功。処理時間は全工程546.1秒（なし）対542.6秒（あり）、ピークworking setは3.59 GiB対3.69 GiBで、速度差は測定ゆらぎの範囲。

raw N-best hit率は挿入前の候補に対して測るため、trigram補正の成果ではなく適用可能性の上限・条件である。全2,000文を分母とする正解率だけでは、gold不在による固定失敗が混ざり、候補内での改善と悪化が相殺される可能性がある。そこで評価を拡張し、goldがraw N-bestに含まれる文に限った一次／再順位付けtop-1正解率、およびtrigramがtop-1正解を改善／悪化させた文数を測った。結果は次の通り:

|学習文数|gold追加|gold-in-N文|一次top-1正解|trigram後top-1正解|改善／悪化|gold-in-N条件付きtop-1正解率|gold-in-N条件付き再順位付けNLL|
|---:|---|---:|---:|---:|---:|---:|---:|
|10,000|なし|1,800|1,213|1,213|0 / 0|67.39% → 67.39%|1.16559582|
|10,000|あり|1,800|1,213|1,213|0 / 0|67.39% → 67.39%|1.16564197|
|100,000|なし|1,802|1,208|1,208|0 / 0|67.04% → 67.04%|1.17108157|
|100,000|あり|1,802|1,208|1,208|0 / 0|67.04% → 67.04%|1.17110927|

従って、この同一検証標本ではtrigram再順位付け自体がgold-in-Nのどの文でもtop-1を改善も悪化もさせず、gold path追加によるtop-1改善も観測されなかった。NLL差もごく小さい悪化（100,000文で+0.0000277）である。これはN-best hitの高低とは別に、今回のモデル・学習量・評価標本ではtrigramがgoldを先頭へ押し上げる効果を確認できなかった、という結論である。候補内のtop-1以外のrank移動は今回記録していない。

再評価では各学習済みcheckpointと同じSHA-256の固定2,000文`test-sample.json`を使い、gold augmentationを無効にした評価only modeでraw N-bestを再生成した。evaluation結果は以下に保存:

- `target/crf-wiki003-static-bi1500k-d090-nogold-n32-train10k-rankdiag-20261004-run01/evaluation.json`
- `target/crf-wiki003-static-bi1500k-d090-goldpath-n32-train10k-rankdiag-20261004-run01/evaluation.json`
- `target/crf-wiki003-static-bi1500k-d090-nogold-n32-train100k-rankdiag-20261004-run01/evaluation.json`
- `target/crf-wiki003-static-bi1500k-d090-goldpath-n32-train100k-rankdiag-20261004-run01/evaluation.json`

評価実装は`gold_present_sentences`、条件付きtop-1 accuracy、`rerank_top1_helped_sentences`、`rerank_top1_harmed_sentences`を`metrics.json` / `evaluation.json`に記録する。raw N-best hit率は独立のcoverage指標として残す。

比較run:

- `target/crf-wiki003-static-bi1500k-d090-nogold-n32-train10k-20261004-run01`
- `target/crf-wiki003-static-bi1500k-d090-goldpath-n32-train10k-20261004-run01`
- `target/crf-wiki003-static-bi1500k-d090-nogold-n32-train100k-20261004-run01`
- `target/crf-wiki003-static-bi1500k-d090-goldpath-n32-train100k-20261004-run01`

## word-bigram / word-trigram の強い更新実験（2026-10-06）

前runの学習時間に対して固定テストの改善が小さかったため、次の新規runでは初期統計へ delta を引き戻す L2 を無効化し、疎な単語n-gramの更新を強める。旧runのcheckpointからは再開せず、初期CRFモデルから開始する。これは収束や品質改善を保証する変更ではなく、効果と過学習を測るための明示的な実験条件である。

- 内部16スレッド、同期mini-batch 16、N=128、chunk003全件、最大2epoch。
- 初期静的word-bigram上位150万件・word-trigram上位300万件を含む同一初期モデルから開始。word-trigramの個別deltaも学習する。
- **word-bigramとword-trigramの勾配はバッチ内で加算し、batch size 16で割らない。** 一次のPOS/unigram等、およびPOS/readlen trigramは従来どおり、実際に勾配を出した文数で平均する。
- `L2=0`。従って未更新weightを初期統計側へ縮めず、学習した補正をそのまま累積する。パラメータ表現は引き続き静的初期score + 学習deltaだが、L2によるdelta縮小は発生しない。
- 他の条件（5,000文の固定テストprefix、missed-feedback、gold-path augmentation、cutoff・容量上限、入力・seed・候補生成）はrun configに記録した値を使う。

新規出力は`E:\lmwork-target\crf-wiki003-n128-word-trigram-learning-20261006-run04-batch16-sum-word-bi-tri-l2off`、作業cache/checkpointは`E:\lmwork-int\crf-wiki003-n128-word-trigram-learning-20261006-run04-batch16-sum-word-bi-tri-l2off`に置く。旧runと初期モデルは保持する。レポートには処理文数/総数、前回からの文数差、文/秒とその差、固定5,000文テストのbigram top-1・raw N=128 gold cover・trigram後top-1とそれぞれの前回差を併記し、評価が更新されていない場合はその時刻と件数を明示する。

入力準備は、run02の同一タグ付き入力・同一初期model-inで作られたpage-block cacheと固定テスト5,000文を再利用する。`--prepared-data-in`は正規化済み学習文・評価標本だけを供給し、学習済みmodel/checkpointを読む経路ではない。実行側が入力、初期モデル、max-words、missing-POS確率、page-blockサイズ、seed、固定prefix、training-probe容量を一致確認する。これにより約15分の入力再走査を避け、モデル重みと更新状態は初期モデルから開始する。

## primary-listwise 2epoch完了値と素性ablation（2026-10-06）

比較用に、2epochまで完了・保存検証済みの旧run `crf-wiki003-n128-primary-listwise-sum-bi-tri-l2off-20261006-run01` の結果を記録する。これは後続のPOS細分化・交差bigram追加runとは別の実験であり、後続runの初期値やcheckpointには流用しない。

- 入力はchunk003全件。固定testは先頭の正規化済み5,000文、N=128。
- epoch 1は全7,411,443学習文を処理。epoch 2はepoch 1最終モデルでtrigram後top-1を外した文のみを対象にし、2,200,200文を処理して完了した。
- 学習目的はprimary raw-N-best listwise + trigram listwise。word-bigram/word-trigram勾配はbatch 16内で合算し、1/16しない。初期値からのwarm startやepoch 1 checkpoint再利用ではなく、このrunの指定初期モデルから開始した。
- 保存先は`E:\lmwork-target\crf-wiki003-n128-primary-listwise-sum-bi-tri-l2off-20261006-run01\model`。`verification.json`のstatusは`completed`。

epoch-endの固定test結果:

|時点|固定test文数|bigram top-1|raw N=128正解含有率|trigram後 top-1|
|---|---:|---:|---:|---:|
|epoch 1終了（全学習文）|5,000|63.54%|91.80%|66.70%|
|epoch 2終了（miss-only 2,200,200文）|5,000|61.96%|92.06%|64.38%|

epoch 2終了時のN-best包含率はepoch 1終了時より0.26ポイント上がった一方、bigram top-1は1.58ポイント、trigram後top-1は2.32ポイント低かった。固定test上では、miss-only epoch 2をepoch 1より良い最終top-1として扱わない。

### 完了モデルへの推論時ablation

以下はepoch 2完了モデルと**同じ固定5,000文test**に対する推論時の素性除去であり、除去条件で再学習した比較ではない。`without_*`では評価時に指定された初期scoreと学習deltaの両方を除く。文字正解率はUnicode scalar値単位のLCS編集距離から算出する。tagged corpusはMeCabによる疑似正解で、人手で全件検証したground truthではない。

|評価条件|bigram top-1|raw N=128正解含有率|trigram後top-1|文字正解率 bigram段階 → trigram後|trigram後top-1の全素性との差|
|---|---:|---:|---:|---:|---:|
|全素性（基準）|61.96%|92.06%|64.38%|95.8719% → 96.3011%|—|
|POS-trigramを除去|61.96%|92.06%|65.04%|95.8719% → 96.3250%|+0.66 pp|
|読み長bigramを除去（trigramは保持）|59.18%|90.04%|63.88%|95.3799% → 96.0318%|−0.50 pp|
|読み長trigramを除去（bigramは保持）|61.96%|92.06%|61.84%|95.8719% → 95.8171%|−2.54 pp|
|読み長bigram・trigramを両方除去|59.18%|90.04%|64.66%|95.3799% → 96.2221%|+0.28 pp|

この保存済みモデルでは、POS-trigramを除くと固定testのtrigram後top-1が0.66ポイント上がった。読み長trigramのみを除くと2.54ポイント下がった。読み長bigramのみを除くとbigram段階とN-best包含率が下がり、最終top-1も0.50ポイント低下した。一方、読み長bi/triを両方除くと最終top-1は0.28ポイント上がるが、文字正解率は基準より0.079ポイント低い。したがって、単語top-1だけと文字一致を同一の結論として扱わない。

この既存評価には「POS-trigramと読み長bigram・trigramを同時にすべて除去」した組合せはない。個別除去および読み長bi/triの同時除去から、その三者同時除去の効果を推定しない。

再確認用artifact:

- epoch-end曲線・学習結果・保存検証: `E:\lmwork-target\crf-wiki003-n128-primary-listwise-sum-bi-tri-l2off-20261006-run01\model\training-curve.jsonl`、`metrics.json`、`verification.json`
- POS-trigram、read-length trigram、read-length bi+triのablationと文字評価: `E:\lmwork-target\crf-wiki003-n128-primary-listwise-sum-bi-tri-l2off-20261006-run01-ablation-character-eval\model\feature-ablation-evaluation.json`
- read-length bigram単独除去: `E:\lmwork-target\crf-wiki003-n128-primary-listwise-sum-bi-tri-l2off-20261006-run01-readlen-bi-off-eval\model\feature-ablation-evaluation.json`

## count cutoff時の固定test差分（2026-10-07）

chunk003のPOS細分化 + POS-WORD/WORD-POS runでは、count cutoffが発生した処理文数と同じ地点で固定5,000文testの評価も記録されている。各cutoff後のtest値と、cutoff直前に保存されたtest評価との差分を以下に記録する。これはcutoff時点までの学習全体の推移を含む差分であり、cutoff処理だけの因果効果を分離した値ではない。純粋なcutoff影響を測るには、同一モデル状態でprune直前・直後の両方を評価する必要がある。

| cutoff family / 発生時の学習文数 | cutoff前→後の学習delta件数 | 比較する固定test評価（処理文数） | bigram top-1差 | raw N=128含有率差 | trigram後top-1差 | 文字正解率差（bigram段階 / trigram後） |
|---:|---:|---:|---:|---:|---:|---:|
| word-bigram / 4,000,001 | 51,550,111 → 25,000,000 | 3,867,740 → 4,000,001 | −0.60 pp | +0.28 pp | +0.16 pp | −0.0550 / +0.0536 pp |
| word-bigram / 6,250,003 | 50,388,440 → 25,000,000 | 6,000,012 → 6,250,003 | −0.04 pp | +0.10 pp | +0.12 pp | +0.0127 / +0.0020 pp |
| word-trigram / 7,250,011 | 117,333,648 → 50,000,000 | 7,238,747 → 7,250,011 | +0.60 pp | −0.02 pp | +0.10 pp | +0.0632 / +0.0079 pp |

word-trigram cutoffの`selected_minimum_abs_delta`は`0.0010309495864567811`。このcutoffと同じ処理文数の5,000文test値はbigram top-1 64.66%、raw N=128含有率92.84%、trigram後top-1 66.60%、文字正解率96.1515% / 96.4718%。直前に保存された評価（7,238,747文）との差は表のとおりで、評価間の学習も含むためcutoffだけの因果効果ではない。

同じrunのepoch 2では、500,012 miss-only文時点でword-trigram cutoffが再発し、113,110,786件から50,000,000件へ削減（`selected_minimum_abs_delta=0.0011313834652312447`）。cutoff前の直近test（epoch 2の375,805文）と、cutoff後の直近test（905,792文、各5,000文）を比べると、bigram top-1は63.14%→63.38%（+0.24 pp）、raw N=128含有率は92.90%→92.96%（+0.06 pp）、trigram後top-1は64.94%→65.06%（+0.12 pp）、文字正解率は96.0367%→96.1189%（+0.0822 pp） / 96.3654%→96.3616%（−0.0038 pp）。比較区間ではcutoff後に529,987 miss-only文を追加学習しているため、これらはcutoff単独の因果効果ではない。

epoch 2の1,250,008文でもword-bigramとword-trigramのcutoffが同じbatchで発生した。word-bigramは50,656,928件→25,000,000件（`selected_minimum_abs_delta=0.0003801430291070707`）、word-trigramは119,807,288件→50,000,000件（`selected_minimum_abs_delta=0.001224810535353342`）。cutoff前の直近test（905,792文）と、cutoff後の直近test（1,438,543文、各5,000文）では、bigram top-1 63.38%→62.42%（−0.96 pp）、raw N=128含有率92.96%→93.10%（+0.14 pp）、trigram後top-1 65.06%→64.74%（−0.32 pp）、文字正解率96.1189%→96.0329%（−0.0860 pp） / 96.3616%→96.3187%（−0.0429 pp）。この間にmiss-only学習が532,751文進んでおり、両cutoffの単独因果効果を分離した比較ではない。

このrunのepoch 2完了後に行う追加評価（同じ固定5,000文test・最終checkpoint/modelを使用）：

1. feature設定と実際のscoring pathを確認し、POS-trigramが評価時に有効かを確かめる。有効ならPOS-trigramを除外したablationを追加して測る。既に無効なら、その事実を確認記録し、無効化済みの値を基準として扱う。
2. POS-WORDとWORD-POSの両feature familyを評価時に同時に除外したablationを測る。各featureの初期scoreと学習deltaの両方を除外し、全featureの最終モデル評価と比較する。

両評価は再学習せず同じ最終modelと固定5,000文testを使う。top-1、raw N=128含有率、文字正解率（bigram段階・trigram後）を記録し、全feature条件との差を示す。

今後のcount cutoff報告では、`count-cutoff.jsonl`の`committed_sentences`と`training-curve.jsonl`の`processed_training_sentences`を照合し、cutoff直後の5,000文test値と直前評価との差（top-1、N-best含有、文字正解率）を併記する。評価が同じ文数で記録されていないcutoffについては、最寄りの評価を使った差分をcutoff単独効果と誤認せず、そのずれを明記する。

## POS-WORD / WORD-POS の学習delta平均（checkpoint、2026-10-07）

checkpoint `checkpoint-epoch-0001-batch-0000421914-1791317409242484800`（epoch 1、処理6,750,013文、`first_scale=1`）の`first_values`から、POS-WORD/WORD-POSの学習deltaを集計した。初期seedは各500,000組。`active平均`はcheckpointに実際に保存されている非ゼロ学習deltaだけの算術平均、`seed上の平均delta`は初期seed 500,000組を分母にし未更新組を0として算出、`seed上のcheckpoint平均score`は初期値 + 学習deltaの平均である。新規に学習されたseed外組は`seed上の平均`に含めない。

| family | 学習delta active件数 | seed内の学習delta件数 | active平均delta | active絶対値平均 | seed上の平均delta | seed上のcheckpoint平均score | 正 / 負delta件数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| POS-WORD | 8,771,398 | 491,837 / 500,000 | +0.00002131 | 0.0109401 | +0.0044123 | +0.1974725 | 1,504,772 / 7,266,626 |
| WORD-POS | 8,893,269 | 491,838 / 500,000 | +0.00002102 | 0.0114154 | +0.0064752 | +0.2662509 | 1,374,036 / 7,519,233 |

したがって、このcheckpointの保存済み学習deltaの算術平均は負ではなく、両familyとも僅かなプラス。ただしactive deltaの件数では負が多数であり、正負件数と値の大きさを含む平均は別の統計である。seed外の学習deltaまで含め、初期値0の新規組も分母に加えたcheckpoint score平均はPOS-WORD `+0.0110162`、WORD-POS `+0.0146128`。この値は6,750,013文時点の中間checkpointであり、最終モデルの平均ではない。

## POS-WORD/WORD-POS・POS-trigram ablation（2026-10-07）

対象run: `crf-wiki003-n128-primary-listwise-affix100-crossbi-20261007-run01`。学習は初期モデル `E:\lmwork-target\crf-initial-model-affix100-crossbi-20261007-run01\model\model.json` から実施し、既存学習checkpointは引き継いでいない。入力reportはchunk003の全走査（7,811,350 records read、7,411,443 eligible training sentences、5,000 fixed-test sentences）、full-corpus完了。epoch 1は7,411,443文、epoch 2はmiss-only 2,126,162文を処理。`verification.json` は`status=completed`、`full_corpus=true`、`checkpoint_roundtrip=passed`、16 threadsを記録し、runnerの終了コードは0。

epoch 2最終モデルを同一の固定5,000文・N=128で評価した。再評価のbaselineは学習curveのepoch-end値と一致した。文字正解率はUnicode scalar value単位のLCS編集距離によるmicro平均。

| 評価条件 | bigram top-1 | raw N=128含有率 | trigram後top-1 | 文字正解率（bigram段階 → trigram後） | baselineとの差（最終top-1 / 最終文字率） |
|---|---:|---:|---:|---:|---:|
| 全feature（baseline） | 63.04% | 92.92% | 64.34% | 95.9866% → 96.2618% | — |
| POS-trigram除外 | 63.04% | 92.92% | 64.34% | 95.9866% → 96.2618% | 0.00 / 0.0000 pp |
| POS-WORDとWORD-POSを両方除外 | 60.56% | 90.66% | 63.50% | 95.5919% → 96.1212% | −0.84 / −0.1406 pp |

POS-trigramについては、初期モデルmanifestが`pos_trigrams_enabled=false`、run中にこのflagをtrueへ切り替える代入はなく、CRF実装もflagがfalseならPOS-trigram feature列を生成せずscoreを0にする。runのmodel出力に`pos-trigram.tsv`もなく、今回の除外ablationも全feature baselineと完全一致した。したがって、このrunの推論ではPOS-trigramは既に無効であり、別途除外しても差はない。一方、training `metrics.json`の`feature_counts.pos-trigram`にはepoch 2で3,884,255件と記録されている。これは候補pathからPOS-trigram勾配を生成し疎表に保存した件数で、推論scoreに有効だったことを意味しない。学習側は無効featureのdeltaを保持する非対称な実装になっており、無効flag下ではそのdeltaは推論結果に寄与しない。

POS-WORD/WORD-POSは両方を同時に除き、初期scoreと学習deltaの双方を除いた。除去時にbigram top-1は−2.48 pp、raw N=128含有率は−2.26 pp、最終top-1は−0.84 pp、最終文字正解率は−0.1406 pp。従ってこの組合せは固定test上で有用だった。両familyを一括で除いた比較なので、個別の寄与は分離していない。

後処理artifact:

- 最終モデルと学習完了検証: `E:\lmwork-target\crf-wiki003-n128-primary-listwise-affix100-crossbi-20261007-run01\model\model.json`、`verification.json`、`training-curve.jsonl`
- 固定test入力の同一内容コピーおよびREADME: `E:\lmwork-int\crf-wiki003-n128-primary-listwise-affix100-crossbi-20261007-run01-posthoc-ablation\work\test-sample.json`
- baselineと各feature ablationの数値: `E:\lmwork-target\crf-wiki003-n128-primary-listwise-affix100-crossbi-20261007-run01-posthoc-ablation\model\eval\feature-ablation-evaluation.json`
- 評価は`epochs=0`のevaluation-only実行であり、モデル学習値は変更していない。
