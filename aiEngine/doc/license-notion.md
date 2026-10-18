# 商用IME辞書のライセンス・表示に関する検討メモ

記録日: 2026-10-14（日本時間）

現在の作成方法、使用資源の版、一次資料で確認した条件、および暫定的な表示方針を記録する。製品用辞書に対する最終的な権利判定は未完了であり、確定事項と推奨する対応を区別する。独自の生成手法の詳細は記載・公表しない。

## 商用利用についての最小限の結論

最も重要な確認事項は、作成した製品辞書を商用IMEへ収録し、有償で提供できるかである。

確認済みの公開辞書資源については、適用される条件を満たした商用利用・販売が可能である。IPADICのCOPYINGは利用・複製・配布を許諾しており、非営利限定の条件を設けていない。今回の現代書き言葉UniDic、SudachiDict、NEologdにも商用利用を禁止する条件はない。WikipediaのCC BY-SA 4.0も商用利用を認める。したがって、これらの公開資源を使ったという理由だけで製品辞書の商用提供が禁止されることはない。必要な表示・条件を満たすことと、製品を有償で提供することは両立する。

**確認したIFELanguageの公式資料には、出力の再利用・商用利用を禁止する記載はありません。** Windows 11の標準ライセンス条項にも、作成した成果物へWindowsのライセンスを継承させる規定は確認できませんでした。[IFELanguage公式資料](https://learn.microsoft.com/ja-jp/windows/win32/api/msime/nn-msime-ifelanguage)、[Windowsの条項](https://www.microsoft.com/content/dam/microsoft/usetm/documents/windows/11/oem-%28pre-installed%29/Useterms_OEM_Windows_11_Japanese.pdf)

今回説明された、一般的な語・読み・品詞と独自の統計値を収録する使い方なら、**該当するIPA等の条件を守って商用提供できる、という見立てです。** 先ほどの「未確認」は、別途Microsoftの許諾が必要であるかのように読める、過剰な留保でした。

これは完成した全データの権利調査を終えたという意味ではない。IFEDictionaryによる辞書データの直接取得・再配布については、後述する別の条件がある。

出典: [IPADIC COPYING](https://github.com/taku910/mecab/blob/master/mecab-ipadic/COPYING)、[UniDicの商用利用](https://clrd.ninjal.ac.jp/unidic/commerce_use.html)、[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)、[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)、[Windows 11ライセンス条項（2024年4月更新）](https://www.microsoft.com/content/dam/microsoft/usetm/documents/windows/11/oem-%28pre-installed%29/Useterms_OEM_Windows_11_Japanese.pdf)、[IFELanguage公式資料](https://learn.microsoft.com/ja-jp/windows/win32/api/msime/nn-msime-ifelanguage)。使用版と個別条件は下記に記録する。

## 現時点の整理

- 元辞書のバイナリを配布する場合だけでなく、保護される辞書内容を別形式で複製・改変して配布する場合も、元辞書の条件が関係する。形式を変更したことや正規APIで取得したことだけで条件が消えるわけではない。
- 一方、解析に使用しただけで、すべての解析結果・統計値に元辞書のライセンスが自動的に適用されるわけではない。
- 一般的な単語・読み・品詞・活用等の言語情報や事実・数値と、著作権で保護される表現、辞書・データベースの創作的な情報選択・構成を区別する。
- 大量のWikipediaを解析し、元辞書のエントリを相当程度再現できるような利用では、「開発時に解析器を使っただけ」と一括して結論づけない。ただし、量や一致率だけで著作権上の複製・翻案が確定するわけでもない。
- 「IPADIC由来が約1/3」という割合は利用者の見積もりであり、未測定。その割合をもって表示義務の有無を判断する基準は確認できていない。
- IPADIC、今回のUniDic、SudachiDict、NEologdには商用利用可能な条件がある。保護される元データを再配布する場合の条件を守ることと、独自の手法・製品全体を公開することは別である。
- 現在の説明では、製品に収録する一般的な言語情報・統計値に、原辞書の創作的な表現やデータベースの選択・構成が引き継がれたことを確認していない。IPADICを含め、「由来する情報がある」「解析に使用した」という事実だけでライセンス表示を必須とはしない。
- 利用者の認識では、IPADICはベースとして引き継いでいる一方、UniDic・SudachiDict・NEologdの元の語彙集合をダンプして製品に採用しているわけではない。この役割の違いを前提に、IPADICのCOPYINGを残す案と、ほかの資源の任意の利用クレジットを区別する。ただし「IPADICの一般的な言語情報だけの利用でも必須」と確定したわけではない。
- Windows、Codex、開発時にだけ使用した解析器の名称を、無難という理由で製品のライセンス欄に列挙しない。開発環境の記録、任意の謝辞、配布物に必要なライセンス表示は別である。

「すべての資源のライセンス全文を必ず同梱する」「解析結果だから元辞書の条件は関係しない」という一律の説明は採用しない。保護対象を利用しているかを先に検討し、法的な表示義務、条件を満たすための実務上の対応、任意の利用クレジットを分けて考える。

## 「保護される内容」と単語セット

普通の単語が辞書に載ったことで、その単語自体が辞書の著作者に独占されるわけではない。一般的な読み・品詞・基本形・活用などの言語情報、出現数・確率等の事実・数値も、それだけで創作的な表現になるわけではない。文化庁は、単なる事実・データ・アイデア・ありふれた表現は著作物に当たらず、著作権者等の許諾なく利用できると説明している。

| 対象の例 | 著作権上の基本的な考え方 |
|---|---|
| 「東京」という一般的な語と「とうきょう」という読み | その語・読み自体を辞書制作者が独占することにはならない。 |
| 一般的な品詞・活用・基本形の情報 | 思想・分類の考え方や言語情報と、具体的な創作的表現を区別する。 |
| 独自に集計したunigram・bigramの回数・確率 | 数値自体が元辞書の創作的な表現であるとは通常考えない。 |
| 辞書独自の語釈、説明文、創作的な例文 | その具体的な文章表現は保護され得る。 |
| 独自の選定方針で選ばれた語の集合、独自の体系的な構成 | 選択・構成に創作性がある場合、集合・データベースとして保護され得る。単語一つ一つが独占されるという意味ではない。 |
| 辞書を検索・解析するソフトウェアのコード | 単語・事実情報とは別の著作物として扱う。 |

「そこに載せた単語セット」も、ただの語の羅列なら必ず保護されるというものではない。辞書という名前やライセンス表示の存在だけで、すべての単語・情報が保護対象になるとは推定しない。

今回の想定どおり、Wikipediaから独自に得た語彙を対象に一般的な言語情報を付け、独自の統計を作るのであれば、原辞書の保護される内容を利用していないという見立てには根拠がある。これは収録情報と語彙選択についての現在の説明に基づく評価であり、将来の完成した製品データ全体を比較検証した結果ではない。

出典: [文化庁・著作物に当たるか](https://www.bunka.go.jp/seisaku/chosakuken/seidokaisetsu/chosakukensha_fumei/)、[文化庁・著作権テキスト令和8年度版](https://www.bunka.go.jp/seisaku/chosakuken/seidokaisetsu/pdf/94383901_01.pdf)、[CC公式FAQ・データベースの保護対象](https://creativecommons.org/faq/)。

## 作成・収録予定のデータ

日本語版Wikipediaを言語資料とし、複数の形態素解析資源の出力を独自に処理してタグ付きコーパスを作成している。タグ付きコーパスは内部の解析・補正・集計専用であり、配布しない。本メモの検討対象は、商用IMEに収録する単語と必要な言語属性、unigram・bigram等の統計値・学習パラメータである。

使用するfeatureはIPADIC形式の1始まりで1～7番目のフィールドと読みである。

| フィールド | 内容 |
|---|---|
| 1 | 品詞大分類 |
| 2～4 | 品詞細分類1～3 |
| 5 | 活用型 |
| 6 | 活用形 |
| 7 | 基本形 |
| 読み | IPADIC形式では8番目に相当する読み |

品詞体系・フィールド構成の採用と、元辞書の保護されるエントリ集合の再利用を混同しない。IPA系の体系を採用したという事実だけでCOPYINGの同梱が必須とは断定できない。

## 確認できた使用版

最新公開版ではなく、手元の作業に使用した資源を記録する。

| 資源 | 版 | 条件・一次資料 |
|---|---|---|
| MeCab本体 | 0.996 | GPL/LGPL/BSDから選択可能。[COPYING](https://github.com/taku910/mecab/blob/master/mecab/COPYING) |
| IPADIC | 2.7.0-20070801 | NAIST・ICOTの条件。[COPYING](https://github.com/taku910/mecab/blob/master/mecab-ipadic/COPYING) |
| UniDic（現代書き言葉） | 3.1.0 | GPL/LGPL/BSDから選択可能。BSDを選ぶ方針。[商用利用](https://clrd.ninjal.ac.jp/unidic/commerce_use.html)、[FAQ](https://clrd.ninjal.ac.jp/unidic/faq.html) |
| Sudachi Rust版（sudachi.rs） | 0.6.11 | Apache License 2.0。[LICENSE](https://github.com/WorksApplications/sudachi.rs/blob/v0.6.11/LICENSE) |
| SudachiDict | core、V0形式、20260428 | Apache License 2.0と付属LEGALに記載された由来資源の表示。[使用版のLEGAL](https://github.com/WorksApplications/SudachiDict/blob/v20260428/LEGAL) |
| Vibrato | 0.5.2 | MITまたはApache License 2.0から選択可能。MITを選ぶ方針。[ライセンス情報](https://github.com/daac-tools/vibrato) |
| mecab-ipadic-NEologd | 0.0.7-20200820 | 追加部分のApache License 2.0と基礎IPADICの条件。[COPYING](https://github.com/neologd/mecab-ipadic-neologd/blob/master/COPYING) |

MeCab本体0.996は実行ファイルのバージョン出力で確認した。実行ファイルとIPADICのsys.dicのSHA-256は、初期タグ付きコーパスchunk001の生成manifestに記録された値と一致した。IPADICの版はインストール済みconfigure.inで確認した。

ほかの版は、生成manifest、取得元source.json、導入manifest、Cargo.toml等を照合した。

## Wikipedia原資料

- 原ファイル: F:/lmwork-fixed/jawiki-latest-pages-articles.xml.bz2
- ファイル名: jawiki-latest-pages-articles.xml.bz2
- ファイル更新日時: 2026-09-27 19:01:53（日本時間）
- サイズ: 4,730,918,124 bytes
- 先頭のXML情報: xml:lang=ja、dbname=jawiki、generator=MediaWiki 1.47.0-wmf.17
- 確認時には先頭64KiBだけを展開した。ダンプ全体の走査・ハッシュ計算は行っていない。
- ファイル更新日は確認できたが、公式のダンプ生成日・日付付きリリース識別子は未確定。「2026年9月27日版」と断定せず「ファイル更新日: 2026年9月27日」と表記する。

現在のWikimedia利用規約では本文の基本ライセンスはCC BY-SA 4.0。個別記事の引用等に関する例外・追加表示をダンプ全体について監査したものではない。

製品辞書に記事の創作的な表現を含めず、事実・統計を収録する利用だけなら、CC BY-SAの継承・表示義務が自動的に生じるとは考えない。

出典: [Wikimedia利用規約・第7節](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use)、[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/legalcode.ja)、[CC公式FAQ](https://creativecommons.org/faq/)。

## 辞書ごとの判断と表示

### IPADIC

元辞書のバイナリを使わず、保護される辞書内容を独自の形式にして配布する場合も、元内容の複製・改変である限りCOPYINGの条件が関係する。IPADICのCOPYINGは、原形または改変されたコピーに著作権表示と所定の条項を保持することを定めている。

IPADICの保護されるエントリ集合・選択・構成等を実際に引き継ぐ部分については、COPYING全文を残す。バイナリから独自のデータ形式に変えても、この条件は形式だけを理由には消えない。一方、引き継いだのが一般的な言語情報だけなら、原辞書のライセンスが当然に適用されるとはいえない。今回の「ベースとして引き継いだ」がどちらに当たるかの法的判定は未確定であり、「IPA由来だから必須」とはしない。

MeCab本体のBSD等のライセンスと、IPADIC辞書のNAIST・ICOTの条件は別である。製品辞書全体を「MeCabのBSDライセンス」と表示しない。

### UniDic

使用中の現代書き言葉UniDic 3.1.0は商用利用可能なGPL/LGPL/BSDの選択式ライセンス。ほかの時代・種類のUniDicには異なる条件のものがあるため、この判断を全UniDicへ一般化しない。

公式FAQは、使用した辞書名・種類・版の記載について説明する。また、商用成果物では解析結果の占める割合や寄与度により判断が分かれる場合の相談も案内している。今回の商用IME辞書がどの扱いになるかを確定したものではない。

そのため、UniDicの利用辞書名・版の記載は公式の案内として考慮する。FAQの利用表示の案内と、著作権上の元辞書の複製・再配布に対する条件は区別する。一般的な言語情報を確認するために使用したことだけで、製品辞書全体をBSDにするものではない。保護される元辞書内容の再配布として扱う場合は、使用版のBSDおよびAUTHORSを保持する。

### SudachiDict

使用版はcore、V0形式、20260428。配布物はApache License 2.0で提供され、LEGALにはUniDic・NEologd等の由来が記載されている。

解析に使っただけで、独自に集計した統計辞書全体がApache License 2.0になるわけではない。保護される元内容を複製・改変して再配布する部分があれば、LICENSE-2.0.txtと関連するLEGALの表示、変更したファイルの表示等を保持する。

### NEologd

使用版はmecab-ipadic-NEologd 0.0.7-20200820。追加部分のApache License 2.0と、基礎となるIPADICの条件を分けて確認する。

NEologdはIPADICを基礎に構築される。SudachiDictにもUniDic・NEologd由来の部分があるため、解析器別の採用割合と、元の言語資源の由来は一致するとは限らない。

手元のE:/lmwork-target/vibrato-neologd/v0.5.2-neologd-0.0.7-20200820/COPYING-NEOLOGDは、内容を確認するとNAIST・ICOTのIPADIC条項だった。ファイル名だけを根拠に、これをNEologdのApache License 2.0全文として扱わない。必要な配布資料を揃える場合は、NEologdの追加部分のCOPYING・Apacheライセンス本文と、IPADICの条件をそれぞれ確認する。

## 「ライセンス継承」の意味

元資源の条件を守ることと、独自の製品全体へ同じライセンスを適用することは別である。

今回のIPADIC、BSDを選択したUniDic、Apache License 2.0のSudachiDict・NEologdについて、元内容の再配布条件を満たすことを理由に、独自の生成アルゴリズム・ソースコード・製品辞書全体を同じライセンスで公開する必要があるとは確認していない。Apache License 2.0第4条は、元条件を守ったうえで独自変更部分等に別の条件を設定することを認める。

変更の事実の表示が必要な場合でも、具体的な生成手法の説明とは区別する。

出典: [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)、使用版のIPADIC COPYINGおよびUniDic BSD。

## Windows / IFELanguage（内部記録）

IFELanguageはMicrosoftの公式資料で、Microsoft IMEを用いた言語処理サービスとして説明されている。Windowsの機能を公開APIから利用していることと、内部でMicrosoft IMEのサービスを使っていることは両立する。

2026年10月14日に、IFELanguageとGetJMorphResultの公式資料、および公開されているWindows 11の標準ライセンス条項（最終更新: 2024年4月）を確認した。IFELanguageの資料は変換・形態素解析等のサービスと呼び出し仕様を説明しており、出力の再利用・商用利用を禁止する記載はない。Windowsの標準条項にも、通常の変換・逆変換結果がMicrosoftに帰属する、一般的な語・読み等の出力の商用再利用を禁止する、または成果物にWindowsのライセンスを適用する条項は見つからなかった。この確認は公開標準条項についてのものであり、個別の契約・追加条項がある場合はその対象範囲で扱う。

一般的な語・読み・品詞等をIFELanguageの解析・逆変換結果として得て独自の統計辞書を作るという今回の説明について、Microsoft IMEを使用したことだけを理由に商用利用不可とする根拠は確認できない。公開APIによる逆変換を、名称だけでWindows条項のリバースエンジニアリング禁止に該当すると断定しない。大量の照会をしたという事実だけで、元辞書の保護される選択・構成の複製が確定するわけでもない。

一方、別のAPIであるIFEDictionaryの公式資料には、当該APIの利用条件として、Microsoft IMEに含まれる辞書データをダンプ・複製・配布しないこと、およびMicrosoft IMEを既に持つ利用者向けのアプリケーション開発に用いることが明記されている。IFEDictionaryは辞書エントリの取得等を提供する。本件はIFELanguageを用いた解析・逆変換という利用者の説明を前提に評価しており、この別APIの条件を根拠なくIFELanguageのすべての出力へ適用しない。実際にIFEDictionaryによる辞書データの取得・再配布を行う場合は、その具体的な条件を検討する。

製品へのMicrosoft IMEクレジットやWindowsのEULA全文の同梱を求める条件も、今回確認した通常の解析・逆変換利用については見つかっていない。Windowsという開発環境の名称も製品のライセンス欄へ列挙しない。元の保護される辞書内容を実際に複製しているかは、一般的な言語情報の利用と区別して判断する。

出典: [IFELanguage公式資料](https://learn.microsoft.com/ja-jp/windows/win32/api/msime/nn-msime-ifelanguage)、[GetJMorphResult公式資料](https://learn.microsoft.com/en-us/windows/win32/api/msime/nf-msime-ifelanguage-getjmorphresult)、[IFEDictionary公式資料](https://learn.microsoft.com/ja-jp/windows/win32/api/msime/nn-msime-ifedictionary)、[Windows 11ライセンス条項（2024年4月更新）](https://www.microsoft.com/content/dam/microsoft/usetm/documents/windows/11/oem-%28pre-installed%29/Useterms_OEM_Windows_11_Japanese.pdf)。

## Codexによる補正・判定（内部記録）

単語の補正や読み・品詞等の判定にCodexを利用している。Codexが知っている一般的な日本語情報を補正に使ったことによって、その単語・読み等へOpenAIの著作権が付くわけではない。一般的な言語知識と、既存の具体的な創作的表現の引用・複製を区別する。

OpenAIのTerms of UseのContent節、および事業向けServices Agreement第4.1条は、利用者とOpenAIの間では、法令で認められる範囲で出力は利用者に帰属し、OpenAIが有する出力の権利があれば利用者へ譲渡すると定める。これが、事実情報に新たな著作権を作り出すという意味ではない。また、第三者の権利が自動的に解消されるという意味でもない。

今回確認した条件から、Codexを補正・判定の支援に使っただけで「開発環境: Codex」やOpenAIの著作権表示を製品ライセンス欄に載せる義務は確認していない。現在の製品表記案へCodexの行は入れない。

AI利用の開示は、著作権・ライセンス表示とは別の問題である。OpenAIのSharing & Publication Policyには、SNS・実演での生成内容の共有やAPIと共作した文章の出版等についてAI利用の開示に関する説明がある。またTerms of UseはAI出力を人間だけで生成したものと偽ることを禁止する。これらを無視して「どのような公開方法でもAI利用の開示は不要」とはしない。一方、これらの規定から今回のIME辞書のライセンス欄へCodexの名前を列挙する義務があるとは確認していない。

出典: [OpenAI Terms of Use](https://openai.com/policies/terms-of-use/)、[OpenAI Services Agreement](https://openai.com/policies/services-agreement/)、[Sharing & Publication Policy](https://openai.com/policies/sharing-publication-policy/)。

## ライセンス表示・謝辞・開発記録の区別

| 表示の種類 | 現在の方針 |
|---|---|
| 配布物に必要なライセンス表示 | 実際に配布する著作物・保護されるデータについて、必要性が確認されたものを記載する。 |
| 使用資源への謝辞・出典クレジット | 別の謝辞・資料欄へ任意に記載できる。公式の利用表示の案内がある資源はそれも考慮する。 |
| Windows、Codex、解析器等の開発環境・作業用ツールの記録 | この内部メモには残すが、使用したという理由だけで製品ライセンス欄へ列挙しない。 |

「無難だから開発環境をすべてライセンスに載せる」という方針は採用しない。任意の謝辞を付けないという選択を、直ちにライセンス違反と扱わない。

## 辞書配布時の同梱ファイル

配布用のひな形は[license.txt](license.txt)に作成した。製品辞書の利用条件は未設定の記入欄とし、IPADIC 2.7.0-20070801のCOPYING全文を第2項へ収録した。WindowsやCodex等の開発環境の一覧は含めていない。

COPYINGはインストール済みIPADICの原ファイルを参照し、公式リポジトリの原文と改行を正規化して一致することを確認した。末尾の原文に含まれる「÷÷」も保持している。

必要なライセンス表示は、辞書ファイルと同じ配布ZIP・配布フォルダにlicense.txtを同梱する形でまとめられる。ファイル名そのものではなく、必要な内容が読める形で受領者へ渡ることが重要である。

IPADICのCOPYINGを保持する方針の場合は、IPADIC 2.7.0-20070801の著作権表示、所定の条項、NAIST・ICOTの免責条項を含むCOPYING全文を省略せず収録する。名称とURLだけの記載で全文を代替しない。

製品辞書の利用条件、第三者由来部分の条件、任意の謝辞は、1つのlicense.txtにまとめてよい。「分けて記載する」とは同じファイル内で見出し・対象範囲を明確にする意味であり、これらを別ファイルに分割することを求めるものではない。

license.txtでは、以下を項目ごとに区別する。

1. 製品辞書について設定する利用条件。
2. 該当する第三者由来部分の対象・資源名・版と、それに必要な正式な著作権表示・ライセンス本文等。
3. 記載することを選んだ任意の謝辞・使用資源クレジット。

製品辞書全体をIPADIC等の原資源と同じライセンスにするという表示は、そうした提供方針を決めていない限り行わない。WindowsやCodex等の開発環境の一覧は含めない。

この同梱方法は表示・条項の渡し方であり、個々の資源について表示義務があるかの判定を変更するものではない。

## 任意の言語資源クレジットの文案

以下は使用資源への謝辞・出典を記載することを選んだ場合の文案であり、製品ライセンスそのものではない。すべての行を法的に必須と判定したものではない。開発環境・作業用ツールや独自の生成手法は列挙しない。

> 本製品の辞書作成にあたり、日本語版Wikipediaおよび下記の言語資源を参考資料として利用しました。
>
> 使用したWikipedia資料: jawiki-latest-pages-articles.xml.bz2（ファイル更新日: 2026年9月27日）。原資料の著作者は、日本語版Wikipediaの各記事の執筆者です。Wikipedia本文の基本ライセンスはCC BY-SA 4.0です。
>
> 使用した言語資源: IPADIC 2.7.0-20070801、UniDic（現代書き言葉）3.1.0、SudachiDict core（V0形式）20260428、mecab-ipadic-NEologd 0.0.7-20200820。

この文案に、各資源への出典リンクを付ける。元資源の保護される内容を配布対象へ含める場合は、このクレジットだけで済ませず、その部分に必要な著作権表示、ライセンス本文、免責条項、追加表示を別紙等で保持する。元資料のライセンス名を示すことと、製品辞書全体の提供ライセンスを指定することを混同しない。

## 単漢字辞書ソースの追加（2026-10-15）

Mozcの単漢字データ（commit `1f3c17119514d6df107358322a307ee2003cd818`）を基本に、Unicode 18.0.0のUnihan `kJapanese` で補完した。単漢字辞書ソースは115,815行、51,663文字。品詞は `記号,単漢字,*,*,*,*`、model POSは `記号,単漢字,*,*,*,*,*`。詳細は単漢字辞書ソース（今回の統合対象外のデータ作成資料）に記録した。

このソースは公開データを取得・抽出・変換して作ったため、該当部分の原条件・著作権表示を保持する。入力版・取得URL・SHA-256は `E:/lmwork-target/single-kanji-dictionary/source-manifest.json`、出典対照は同フォルダのentry-origins.tsvに記録し、成果物と同じフォルダのlicense.txtにMozc LICENSE、Unicode License v3、参照IPADICのCOPYING全文を収録した。

製品用の[license.txtひな形](license.txt)にも、単漢字データのMozc由来部分とUnihan由来部分を区別して追加した。Mozcの一般部分のBSD条件とUnicode License v3の全文を収録し、製品辞書の独自の利用条件は未設定の記入欄を維持した。WindowsやCodex等の開発環境一覧は追加していない。

## 今後の確認点

1. 製品用辞書の実際の収録列・語彙集合・生成経路を、中間コーパスと分けて確定する。
2. 原辞書から引き継いだ一般的な事実情報と、保護されるエントリの選択・構成等を区別する。文字列の一致率だけを法的判断の代わりにしない。
3. 4辞書へ同じ判断基準を適用する。IPADICをベースにしたという呼称や他辞書の採用割合だけで、表示が必要な資源を決めない。
4. UniDic公式FAQにおける解析結果・商用成果物の扱いを確認する。Microsoftについては、IFELanguageの解析・逆変換とIFEDictionaryによる辞書データ取得を生成経路で区別する。一般的な言語情報の出力に、根拠のない追加許諾や開発環境名の表示を要求しない。
5. 配布資料を作る段階で、使用版の元ファイルから正式な著作権表示・利用条件を取得する。記憶で著作権者や年を補完しない。
