# 作業データ配置ルール

## 独立したソースの配置

|目的|場所|
|---|---|
|変換エンジン・入力編集・実行時LLM評価|`D:GitcodesnipetsaiEngine`|
|crawl・コーパス処理・辞書作成・CRF計算|`D:Gitcorpus-management`|
|大容量の作業途中データ|`E:lmwork-int<目的フォルダ>`|
|検証済み成果物・次工程へ渡すデータ|`E:lmwork-target<目的フォルダ>`|

corpus-managementに常設するのはCRF学習に必要な17ソース。crawl・補正・集計などの周辺コードはMD仕様から必要時に作り、作業領域に置く。

両リポジトリは最終的にも別実装として管理する。ソースを共有するpath/Git依存を作らず、計算済みデータの形式で接続する。

実行可能コードのビルド出力は各リポジトリの `target` や専用CMake build出力に置ける。コーパス、ダンプ、辞書データ、ID列、ngram、学習checkpoint、分析DB、巨大ログの新規生成先は `lmwork-int` / `lmwork-target` とする。

## 既存の正本データ

2026-10-18の明示的な移動指示により、既存の `dictionary/additions`、`dictionary/build`、POS参照資料は `D:Gitcorpus-managementdictionary` へ移した。既存データはローカル専用・Git対象外で保持し、ここに別runのcacheや複製を増やさない。既存の追加語の履歴は不用意に削除・再生成しない。

aiEngine内の `corpus/wiki-ja/chunked`、`clean-split`、`tagged-mecab-initial` などの旧junctionは履歴として残る。存在しないリンク先もあるため、新規runでは入力パスを明示する。外部のコーパス本体は今回移動していない。

## READMEの記録

`E:lmwork-int` と `E:lmwork-target` の1階層下の目的フォルダにREADMEを置く。目的・状態、入力の絶対パスと由来、生成物、コマンドと版、次の利用先、削除可否を記録する。重要な入力・出力にはサイズとSHA-256を記録する。まだ作成していない出力を生成済みと記載しない。深い階層へのREADME追加は必須にしない。

## 大容量作業領域の検索・保持

`E:lmwork-int`、`E:lmwork-target` とその目的フォルダ全体を、明示的な許可なく再帰列挙・検索しない。README、manifest、ledgerなど既知の小さな管理ファイルを優先し、本文が必要なら対象ファイル・検索語・読み取り範囲を限定する。圧縮コーパスを丸ごと展開しない。

既存データをこの規則だけで移動・削除しない。使用中のプロセス、次工程の参照、再開に必要なcheckpoint、正本かどうかを確認する。明示的に移動するときはコピー後にサイズとSHA-256を確認してから切り替える。目的フォルダをまたぐ一括削除を行わない。

過去の実行記録に残る旧パスは履歴として扱う。新規runでは現在の配置と明示的な入力・出力先を使う。
