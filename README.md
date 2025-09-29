## **概要**

言語モデルの蒸留手法であるTAIDを文埋め込みにも適用して、効率的な知識蒸留を行う

## **検討している損失関数**

$t_i$に対するモデルmの埋め込みを$x_i^m$とする場合

T: 教師モデル, S: 生徒モデル, N: batch_size

M: 生徒モデルの埋め込み次元数を教師に揃えるための行列（`torch.linear(S.dim, T.dim)`で初期化してる）

### MSE

単純に同じ文についての生徒と教師の表現のMSE

参考：https://github.com/caskcsg/sentemb/blob/main/DistilCSE/kd_distillation.py

### KLD

生徒の類似度行列、教師の類似度行列それぞれにSoftmaxした確率分布をKLダイバージェンスで近づける

イメージ…？：$\sum_{i=i}^{N}(KL(softmax(sim(x_i^S,x_i^S)),softmax(sim(x_i^T,x_i^T))$

### CKD

生徒と教師の埋め込みの間で類似度行列を計算し、同じ文を埋め込んだものの類似度が高くなるようにCrossEntropyで訓練

イメージ↓

|  | $x_1^T$ | $x_2^T$ | $x_3^T$ | $x_4^T$ | $x_5^T$ | … |
| --- | --- | --- | --- | --- | --- | --- |
| $x_1^S$ | 1 |  |  |  |  |  |
| $x_2^S$ |  | 1 |  |  |  |  |
| $x_3^S$ |  |  | 1 |  |  |  |
| $x_4^S$ |  |  |  | 1 |  |  |
| $x_5^S$ |  |  |  |  | 1 |  |
| … |  |  |  |  |  |  |

通常のunsup-simcse ↓

|  | $x_1^S$ | $x_2^S$ | $x_3^S$ | $x_4^S$ | $x_5^S$ | … |
| --- | --- | --- | --- | --- | --- | --- |
| $x_1^S$ | 1 |  |  |  |  |  |
| $x_2^S$ |  | 1 |  |  |  |  |
| $x_3^S$ |  |  | 1 |  |  |  |
| $x_4^S$ |  |  |  | 1 |  |  |
| $x_5^S$ |  |  |  |  | 1 |  |
| … |  |  |  |  |  |  |

メモリーバンク（過去のバッチに含まれる教師の埋め込みを最大m件、バッチ内負例として訓練に利用する）も実装している

参考：https://github.com/caskcsg/sentemb/blob/main/DistilCSE/ckd_contrastive.py

### J&S

Cosine lossとsimilarity_lossとtriplet_lossの重み付き和

Cosine loss

    生徒と教師の埋め込みの類似度についてF.cosine_embedding_lossで訓練（同じ文をエンコードした埋め込みのCos simが高くなり、他が低くなるように）

similarity_loss

    生徒のin-batch類似度と教師のin-batch類似度が似るようにmseで訓練

triplet_loss

    生徒のin-batch類似度と教師のin-batch類似度を利用し、教師がより似ているとしたペアの類似度が高くなるようにマージン付きで訓練

参考：https://github.com/NovaSearch-Team/RAG-Retrieval/blob/master/rag_retrieval/train/embedding/model_distill.py

### Distill

ditill_lossとcse_lossの足し算

distil_loss

    生徒の埋め込みと教師の埋め込みのクロスエントロピー

cse_loss

    生徒と教師の埋め込みの類似度について、同じ文の埋め込みが最も高い類似度になるようにクロスエントロピー

参考：https://github.com/Jiahao004/DistillCSE/blob/main/distillcse/models_distill_calibrate.py

### RKD

第１項と２項を反対にした（逆向きに近づけるようにした）KLダイバージェンス

極端に平坦な分布にはならないが、どこか特定の値だけ飛び抜けた分布を作ってしまいがちになるらしい

### TAID

時間係数tを導入し、訓練が進むにつれ教師モデルに近しい分布を出力させるようにする

tは線形ではなくて、train lossの減り具合によって調整される

本家は言語モデルの出力確率を利用して、KLダイバージェンスで訓練を行っているが、今回は様々な特徴量/損失関数で試してみる

## positiveを使うかどうか

use-posオプションで変更可能

Anc-AncではなくAnc-Posについての埋め込みを利用する

## negativeを使うかどうか

use-negオプションで変更可能

※use-pos=Trueでないとエラーになります。

データごとに事前に用意した負例を利用する。より多くのメモリを使うことに注意。


## 評価指標
`tasks.yaml`参照

訓練中に行われる評価は以下

eng

    # clustering
    - StackExchangeClustering.v2
    # Retrieval
    - HotpotQAHardNegatives
    # STS
    - STSBenchmark
    - SICK-R
    # Summarize
    - SummEvalSummarization.V2

jpn
    
    # Classification
    - "AmazonReviewsClassification"
    # STS
    - "JSICK"
    - "JSTS"

## 生徒モデルと教師モデル

**生徒**: "nomic-ai/modernbert-embed-base-unsupervised"

**教師**："Qwen/Qwen3-Embedding-0.6B" "Qwen/Qwen3-Embedding-4B"

## 訓練データ

### 日本語
#### triplet

`cl-nagoya/ruri-dataset-v2-pt`から、いくつかの不適切そうな？データセットを排除して利用
主に簡単すぎる言い換えデータ、多すぎるWikiドメインを削った
合計1Mになるように割合そのまま減らした

排除したデータ↓
```
        "wordnet-ja-synonyms",
        "wordnet-ja-same-synsets",
        "word-ja-definitions",
        "ihyoki",
        "jawiki-paragraphs1",
        "jawiki-hyperlinks",
        "wiki-summary-title2text",
        "wiki-summary-text2text",
        "wiki-para-text2text",
        "wiki-para-title2text",
        "wiki-qa",
```

### 英語
#### triplet
`sentence-transformers/embedding-training-data`からいくつか排除
主に似たようなデータがたくさんあるものを削った
各データMAX1Mに削って、そこから合計1Mになるように割合そのまま減らした

```
        "altlex",
        "WikiAnswers",
        "S2ORC_citations_titles",
        "specter_train_triples",
        "yahoo_answers_question_answer",
        "yahoo_answers_title_question",
        "cnn_dailymail_splitted",
        "S2ORC_citations_abstracts",

        "msmarco-triples",
        "quora_duplicates",
        "quora_duplicates_triplets",
        "PAQ_pairs",
        "flickr30k_captions"
```
#### gte
mGTEの論文に書かれている英語の訓練セットを再現

各データと元データは以下
```
- microsoft/ms_marcoから拝借
    - MS MARCO
- bgeの公開訓練データから拝借
    - Natural Questions (NQ)
    - TriviaQA 
    - HotpotQA
    - SQuAD
    - AllNLI from SimCSE
- MTEBのtrainデータを拝借
    - FEVER
```
- 教師埋め込みの精度を確保するため、教師モデルによるエンコード時にはInstructionを付けてみた
- posが複数ある場合は先頭のみ利用、データは全部で0.8M件となった

## 訓練設定

最大系列長：512

学習率：1e-04

各種損失にくっついている温度などのハイパラ：元実装を参考にそのまま

スケジューラ：cosine -> WSDにしてみた

同時にWarm up ratioを0.1から0.05に