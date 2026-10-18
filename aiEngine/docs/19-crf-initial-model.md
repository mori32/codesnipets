# CRF初期モデルの生成

## 入力と出力

生成コードは `crates/ai_engine_crf/src/bin/build-initial-model.rs`。語彙入力はMeCab IPADICと `dictionary/additions`、統計は `E:\lmwork-target\wiki-ja-wordid-ngrams`。固定global `vocab`とID順の `1/2/3gram.by-id.tsv` が原統計IDを対応づけ、頻度順 `2gram.tsv` / `3gram.tsv` から初期word-bi/triを選ぶ。

```powershell
cargo run --release -p ai_engine_crf --bin build-initial-model -- `
  --output E:\lmwork-target\crf-initial-model-affix100-crossbi-20261007-run01\model `
  --work E:\lmwork-int\crf-initial-model-affix100-crossbi-20261007-run01\work
```

生成先とwork先が存在する場合は、上書き・再利用せず停止する。両方にREADMEを作成する。データは `aiEngine` 外に置き、`aiEngine\\target` は実行可能コードのビルド生成物だけに使う。

## 今回の初期モデル

log-score上の補間補正は `delta = log((1-t) * P0 + t * P1) - log(P0)`。経験値のないeventでは `P1=P0` として補正0になる。POSの併合は行わず、選択した語彙項目のPOSを新しいIDに細分化する。

|要素|今回の作り方|
|---|---|
|POS bigram|raw `2gram.by-id.tsv` 全体から新POS IDで集計|
|word unigram-in-POS|辞書語彙 + corpus unigramを新POS IDで再正規化|
|word bigram|count上位1,500,000、POS-bigram × word-unigram-in-POS基準、経験側 `t=0.1`|
|word trigram|count上位3,000,000、従来と同じPOS/unigram基準と `t=0.1`。word-bigramへbackoffしない|
|POS-WORD bigram|count上位500,000、POS-bigramをbase、経験側 `t=0.1`|
|WORD-POS bigram|count上位500,000、POS-bigramをbase、経験側 `t=0.1`|
|POS trigram|無効、生成しない|
|reading-length bigram/trigram|9クラスの全81/729セル。従来の初期化 `t=0.15` を維持し、biからtriへのbackoffなし|

候補はraw unigram頻度降順から選ぶ**名詞接頭語・名詞接尾語の合計top100**。`号・時・日・数・級`、数字・助数詞系、対象外の動詞接尾語は選択しない。各選択語は元の読み・元POSと組み合わせた個別のrefined POS IDを得る。同じ既存POSカテゴリの他の語を変えず、既存POS同士も統合しない。

統計由来のword-ID bigram/trigramは原corpus語彙IDからMeCab語彙IDへ正しく変換する。細分化後は、選択済みsurface + normalized reading + original POSのtokenだけがrefined POSへ対応づく。CRFのcorpus normalizerにも同じ対応を適用し、初期モデル語彙と学習文のPOS IDが一致する。

## 成果物と検証

|ファイル|内容|
|---|---|
|`model.json`|初期CRFモデル。学習時は `--model-in` で読む|
|`vocabulary.jsonl`, `pos.tsv`|model-local word/POS IDとsurface/reading/feature対応|
|`pos-token.tsv`|POS-WORD / WORD-POS交差素性用の合成POS token ID|
|`affix-pos-selection.tsv`|選ばれた接頭語・接尾語の順位、raw count、元POS、新POS|
|`word-cost.tsv`, `pos-bigram.tsv`|word unigram-in-POSおよびPOS bigram|
|`word-bigram.tsv`, `word-trigram.tsv`|初期word-bi/tri log-score補正|
|`pos-word-bigram.tsv`, `word-pos-bigram.tsv`|初期交差bigram補正。各最大500,000行|
|`reading-length-bigram.tsv`, `reading-length-trigram.tsv`|全81/729 read-lengthセル|
|`manifest.json`|設定、実際件数、file bytes、経過時間、roundtrip fingerprint|

POS-trigram表と表記種別n-gram表は出力しない。モデル保存後に同じ `model.json` を再読込してfingerprint一致を検証し、manifest statusがcompleteにならなければ学習runを開始しない。READMEには各保存先のソース、生成内容、次の利用先、再生成・削除可否を記録する。

初期モデルの後続学習は、新しい語彙/POS IDに合わせてchunk003入力を再正規化する。旧runの学習済み重み、checkpoint、正規化page-block cacheは流用しない。固定test 5,000文も同じchunk003の受理文prefixから新モデル用に生成する。
