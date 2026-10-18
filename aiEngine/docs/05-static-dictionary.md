# 計算済み辞書・モデルの利用

aiEngineは変換の実行側です。crawl、辞書作成、ngram集計、CRF計算は独立リポジトリ [corpus-management](https://github.com/mori32/corpus-management) が担当します。

両者のソースを共有するcrateやpath/Git依存を設けず、計算済みunigram / bigram / trigram / 8gram、語彙・品詞ID対応、辞書・モデルのデータ形式で接続します。将来の実装もこの責務分離を維持します。

現在のランタイムは暫定 `.aidict` のreaderを備えます。`dictionaries/phase5.aidict` は小さな固定fixtureです。学習seed・CRF計算はcorpus-management側へ移動しました。周辺の辞書writerやcompiler/dumperは同リポジトリのMD仕様から必要時に作成し、常設しません。エンジン内の旧writerは既存unit testの小さなfixture生成専用で、通常ビルドの公開APIには含めません。

`.sdict`、各ngram表、CRF出力を正式ランタイムへ読み込む処理は未実装です。形式・値の種類・ID対応・版を定めてから、エンジン所有のreaderを実装します。
