from pathlib import Path

import pandas as pd

base_dir = Path("output/result/nomic-ai_modernbert-embed-base-unsupervised/Qwen_Qwen3-Embedding-4B/794554")
result_file_name = "eng_on_train_end_tasks_scores.csv"

files = list(base_dir.rglob(result_file_name))
agg_results = []
for file in files:
    result = pd.read_csv(file)
    result["model_path"] = str(file.parent.name)
    # Move model_path column to the first position
    cols = result.columns.tolist()
    cols = ["model_path"] + [col for col in cols if col != "model_path"]
    result = result[cols]
    agg_results.append(result)
agg_results = pd.concat(agg_results)
agg_results.to_csv(base_dir / f"all_{result_file_name}", index=False)
print(base_dir / f"all_{result_file_name}")
