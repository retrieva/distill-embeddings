from IsoScore.IsoScore import *
import pandas as pd
from sentence_transformers import SentenceTransformer
from argparse import ArgumentParser

parser = ArgumentParser(description="Calculate IsoScore for a set of texts.")
parser.add_argument("--model_name", type=str,required=True, help="Path to the model name.")
parser.add_argument("--prompt_name", type=str,required=True, help="Name of the prompt to use.")


args = parser.parse_args()
simple_wiki = pd.read_json("data/triplet-eng/SimpleWiki.jsonl",lines=True,orient="records")
texts = simple_wiki.sample(10000,random_state=42)["anc"].unique().tolist()
print(len(texts))
model = SentenceTransformer(args.model_name)
embeddings = model.encode(texts, convert_to_tensor=True, prompt_name=args.prompt_name).to("cpu")
print(embeddings)
print(args.model_name, args.prompt_name, IsoScore(embeddings))
