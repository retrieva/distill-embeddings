import pandas as pd
import yaml
import mteb
from mteb.leaderboard.table import create_tables

models = [
    "Qwen/Qwen3-Embedding-0.6B",
            "Qwen/Qwen3-Embedding-4B",
            "Qwen/Qwen3-Embedding-8B",
          "nomic-ai/modernbert-embed-base",
          "nomic-ai/modernbert-embed-base-unsupervised",
          "Alibaba-NLP/gte-modernbert-base",
          "lightonai/modernbert-embed-large-unsupervised",
          "lightonai/modernbert-embed-large"]
with open("tasks.yaml", 'r') as file:
    tasks = yaml.safe_load(file)
    tasks = tasks["eng"]["on_eval_tasks"]
scores = mteb.load_results(models=models,tasks=tasks)
# Convert scores into long format
scores_long = scores.get_scores(format="long")

# Convert scores into leaderboard tables
summary_gr_df, per_task_gr_df = create_tables(scores_long=scores_long)

# Convert Gradio DataFrames to Pandas
summary_df = pd.DataFrame(
    summary_gr_df.value["data"], columns=summary_gr_df.value["headers"]
)
per_task_df = pd.DataFrame(
    per_task_gr_df.value["data"], columns=per_task_gr_df.value["headers"]
)
scores = pd.concat([summary_df, per_task_df], axis=1)
scores.to_csv(f"output/loaded_mteb_score.csv", index=False)