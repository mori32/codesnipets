# Dictionary compiler / dumper 実装計画

この文書は [`05-static-dictionary.md`](05-static-dictionary.md) の仕様を実装可能な作業単位へ分ける。形式はaiEngine用に新規設計し、他形式とのバイナリ互換を受け入れ条件にしない。format v7でPOS/語彙、共通double-arrayとinline tail、VOCABULARY key group内のprediction phrase/cache、POS/word bigram、backoff alpha、compact trigram、独立codebookのcompiler/dumperを実装する。runtime readerは未完了。

## 入出力

CRF補正学習の別ツールは [`14-crf-training.md`](14-crf-training.md) を参照。そこで出力するsigned potential TSVは現在の確率モデルcompilerへの直接入力ではなく、CRF対応codec・ID再配置のための後続入力仕様である。

目標のfull compilerでは以下を入力する。

- contiguous IDの `vocab.tsv`（`word_id<TAB>token`）
- 1/2/3/8-gramのID列とcount
- 単語候補とphrase候補のword ID列・unigram確率。候補抽出行はscore降順、prefix cacheはunigram確率降順
- 必要に応じてMeCab/Mozc語彙と接続・確率データ

format v7 CLIの入力は `vocab.tsv`、`1gram.by-id.tsv`、`2gram.by-id.tsv`、count cutoff済みの `2gram.tsv` と `3gram.tsv`、`prediction-phrase.txt`、MeCab標準辞書CSV、user addition CSV/TXT。`2gram.by-id.tsv` は全行をストリームしPOS遷移・文脈countの分母を作る。8-gramはコンパイル入力にせず、phrase抽出までの中間データである。

出力はバイナリ `.sdict` とする。`dumper`はバイナリを読み、header、block、POS、word、読み索引、prediction、量子化tableをTSVまたはJSONLで復元表示する。

## 実装段階

### S1: format skeleton

- [x] magic、format GUID、version、dictionary UUID、build UUID
- [x] 固定GUIDのblock directory（offset/size/item_count/checksum）
- [x] 範囲検査を行うdumper
- [x] 小fixtureのwrite/read/dump往復試験

### S2: POSと語彙

- [x] POS文字列を辞書内で決定的にID化し、POS_NAMES blockへ文字列を保存
- [x] word IDをvocabulary block内のword record開始byte offsetにする
- [x] 表記・読みleaf、POS ID、量子化した `P(word|POS)` を保存する。原count・POS合計はbuild時だけ使い、辞書へ格納しない。独立した `P(word)` も保存しない。
- [x] 異なるPOS table順で同じfeature文字列が異なるローカルIDへ解決されることをテスト
- [x] 助詞・助動詞は文法属性6項目を保持し、最後の7項目目の `*` を単語表記へ置き換えてmodel POSを単語ごとに細分化する。MeCab/user辞書ソースと既存corpus tokenに同じ生成規則を適用し、POS bigram・POS内unigram・backoffをこのPOSで計算する

### S3: phraseとprediction key-group records

- [x] phrase recordを完全reading見出しのVOCABULARY key group内に格納し、phrase record offsetから検索可能にする
- [x] 1/2/3文字のひらがなprefix key groupだけにcache headerを置き、最大5個のword/phrase record offsetを保存する
- [x] 通常phrase上位100,000件を格納し、単語cacheは既存WORD recordを参照、cacheが参照するcutoff外phraseは共有recordとして追加格納する
- [x] phraseは各word IDから表記・読み・POSを復元できること
- [x] prefix候補5件以下、cache targetの型解決、phrase読みに一致するword ID列をdumper round-tripで検査

### S4: 表記・読みdouble-arrayとvocabulary tail record

- [x] surfaceとreadingのunion key集合からdouble-arrayを構築
- [x] 一意prefix以降はterminalが指すvocabulary key group内にtail recordとして直接格納する。tail ID表・tail poolは追加しない
- [x] word IDからword bodyへ直接進み、display leafとkey-group terminatorのreading leafから表記・読みを復元する
- [x] prefix/tail境界、prefix関係のkey、内部文字コード、未知key拒否、compile→dump往復のfixtureを追加
- [x] 実データでformat v7をcompileし、dumperで完全復元。出力は92,196,288 byte。build/dump時間の性能比較は別途行う
- [ ] 既存C++ compilerとのbuild時間・file sizeを比較し、runtime lookup p50/p95を測定
- [ ] 測定後も性能を維持する。単純な全候補走査へ置き換えない

### S5: 確率block

- [x] POS bigram、word bigram/backoff blockを実装
- [x] POS内word unigram `P(word|POS)` を専用log確率codebookで8bit量子化（0は特殊値）
- [x] POS bigram、word bigram、bigram backoff、trigramは各々独立したcodebookで8bit量子化する。
- 量子化後の確率復元誤差、コードの重複、単調な探索境界をfixtureで検査

### S6: trigram block

- [x] bigram `B/*` のslotをtrigram `A/B/*` の基準配列にする
- [x] sparse/denseの実byte数を比較し小さい方を選ぶ
- [x] bigram addressからtrigram blockの開始addressを引くsuccinct index（parent bitvector/rankとblock-start bitvector/select sample）を保存。slot indexの表現には使わず、block自体にheader/descriptorは持たせない
- [x] decode後のword ID/slot/probability対応をfixtureで照合

## CLI案

```text
ai-engine-dict compile-sdict <input-dir> <output.sdict> --system-dictionary <MeCab-dicdir> --user-additions <addition-dir>
  [--ngram-cutoff <count>]
ai-engine-dict dump-sdict <input.sdict> <output-dir>
```

`--ngram-cutoff` の既定値は10で、明示word-bigram/trigramに同じ閾値を適用する。POS bigramとword-bigram確率の分母は、cutoff済み表ではなく全件の `2gram.by-id.tsv` から計算する。CLIは上記の引数形式で実装済み。既存の `compile` / `compile-mozc` は別機能として維持する。

## 必須テスト

- 壊れたheader、未知の必須block、重複block GUID、範囲外offset/sizeの拒否
- word IDの欠落・重複・範囲外参照の拒否
- POS ID→文字列往復、辞書別ID採番、user dictionaryの文字列解決
- phrase recordがkey groupにあり、語構成word ID列の読み連結が見出しと一致する
- prefix headerは1〜3文字ひらがなのkey groupだけに存在し、最大5個のword/phrase target offsetを正しく解決する
- prefix targetがword recordを指すケースとphrase recordを指すケースの両方を検査する
- quantization codebookの重複代表値解消とround-trip
- compiler→dumper→再構成データの完全一致
