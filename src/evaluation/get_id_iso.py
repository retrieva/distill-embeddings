"""
適当なモデルに対して、simple_wikiのデータセットを用いて固有次元とIsoScoreを計算するスクリプト
こちら(https://www.anlp.jp/proceedings/annual_meeting/2025/pdf_dir/A5-4.pdf)を参考に、TwoNNを利用して内在次元を計算
"""

from argparse import ArgumentParser

import pandas as pd
import skdim
from datasets import load_from_disk
from IsoScore import IsoScore

from src.utils import PROMPT_MAP, load_model

# 本来はMTEB全部のプロンプトで測った方がいいけど、一旦これぐらいにしておく
parser = ArgumentParser(description="Calculate IsoScore for a set of texts.")
parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-Embedding-0.6B", help="Path to the model name.")
args = parser.parse_args()

wiki = load_from_disk("data/anly-wiki/en")
texts = wiki["text"]
print("text len:", len(texts))
model = load_model(model_name=args.model_name, return_output_folder=False)
model.eval()
result = []
for prompt_name, prompt in PROMPT_MAP.items():
    embeddings = model.encode(
        texts, convert_to_tensor=True, prompt=prompt, max_length=512, batch_size=16, truncate_dim=None
    ).to("cpu")
    # --- 2. TwoNNモデルを初期化 ---
    twonn = skdim.id.TwoNN()
    # --- 3. モデルをデータに適合させ、内在次元を推定 ---
    twonn.fit(embeddings)
    # --- 4. 結果の表示 ---
    intrinsic_dimension_twonn = twonn.dimension_

    print(args.model_name, prompt_name, "IsoScore: ", IsoScore(embeddings))
    print(f"元の次元数: {embeddings.shape[1]}, 内在次元: {intrinsic_dimension_twonn}")
    result.append({
        "prompt_name": prompt_name,
        "IsoScore": IsoScore(embeddings),
        "original_dimension": embeddings.shape[1],
        "intrinsic_dimension": intrinsic_dimension_twonn,
    })

result = pd.DataFrame(result)
print(result)
