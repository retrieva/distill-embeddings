from IsoScore.IsoScore import *
import pandas as pd
from sentence_transformers import SentenceTransformer
from argparse import ArgumentParser
import skdim

# 本来はMTEB全部のプロンプトで測った方がいいけど、一旦これぐらいにしておく
parser = ArgumentParser(description="Calculate IsoScore for a set of texts.")
parser.add_argument("--model_name", type=str,default="Qwen/Qwen3-Embedding-0.6B", help="Path to the model name.")
parser.add_argument("--prompt_name", type=str,default=None, help="Name of the prompt to use.")

args = parser.parse_args()

if args.prompt_name is None:
    prompt = "none"
elif args.prompt_name == "retrieval":
    prompt = "Given a question, retrieve passages that answer the question"
elif args.prompt_name == "sts":
    prompt = "Retrieve semantically similar text"
elif args.prompt_name == "classification":
    prompt = "Given a text, classify its topic"
else:
    raise ValueError(f"Unknown prompt name: {args.prompt_name}")


simple_wiki = pd.read_json("data/triplet-eng/SimpleWiki.jsonl",lines=True,orient="records")
texts = simple_wiki.sample(10000,random_state=42)["anc"].unique().tolist()
print("text len: ",len(texts))
model = SentenceTransformer(args.model_name)
embeddings = model.encode(texts, convert_to_tensor=True, prompt=prompt).to("cpu")

# --- 2. TwoNNモデルを初期化 ---
twonn = skdim.id.TwoNN()
# --- 3. モデルをデータに適合させ、内在次元を推定 ---
twonn.fit(embeddings)
# --- 4. 結果の表示 ---
intrinsic_dimension_twonn = twonn.dimension_

print(args.model_name, args.prompt_name, "IsoScore: ",IsoScore(embeddings))
print(f"元の次元数: {embeddings.shape[1]}, 内在次元: {intrinsic_dimension_twonn}")
