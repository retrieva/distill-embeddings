#!/bin/sh
#PJM -L rscgrp=b-batch
#PJM -L gpu=1
#PJM -L elapse=12:00:00
#PJM -j
#PJM -o logs/0903/02.log

module load cuda cudnn nccl gcc

nvidia-smi
uv sync
model_paths=(
"output/result/nomic-ai_modernbert-embed-base-unsupervised/Qwen_Qwen3-Embedding-4B/794554/gte_e3_bs128_wsd0.0001_taid-js_w-pos_prefix/checkpoints/epoch=02.ckpt"
"output/result/nomic-ai_modernbert-embed-base-unsupervised/Qwen_Qwen3-Embedding-4B/794554/gte_e3_bs128_wsd5e-05_ckd_w-pos_prefix/checkpoints/epoch=02.ckpt"
"output/result/nomic-ai_modernbert-embed-base-unsupervised/Qwen_Qwen3-Embedding-4B/794554/gte_e3_bs128_wsd5e-05_infocse_w-pos_prefix/checkpoints/epoch=02.ckpt"
"output/result/nomic-ai_modernbert-embed-base-unsupervised/Qwen_Qwen3-Embedding-4B/794554/gte_e3_bs128_wsd5e-05_js_w-pos_prefix/checkpoints/epoch=02.ckpt"
"output/result/nomic-ai_modernbert-embed-base-unsupervised/Qwen_Qwen3-Embedding-4B/794554/gte_e3_bs128_wsd5e-05_kld_w-pos_prefix/checkpoints/epoch=02.ckpt"
)
for model_path in ${model_paths[@]}; do
    uv run python -m src.evaluation.run_mteb \
        --model_name $model_path \
        --batch_size 128 \
        --benchmark_name "on_train_end_tasks"
done

# # 0.【重要】まずドライラン（テスト実行）で何が転送されるか確認
# # DRY_RUN="--dry-run"
# DRY_RUN="" # 本番実行時はこちらを有効に

# # 1. 転送元の基準となるディレクトリに移動
# cd /home/pj25000118/ku50001638/distill-embeddings/

# # 2. 検索を開始するサブディレクトリのパス
# #    epoch=02.ckpt ファイルが含まれる親ディレクトリを指定します。
# SEARCH_DIR="output/result/nomic-ai_modernbert-embed-base-unsupervised/Qwen_Qwen3-Embedding-4B/794554"

# # 3. 転送先の情報
# DEST_INFO="chihiro_yano@172.16.3.17:/data/yano/distill-embeddings/"

# # 4. SSHの接続情報
# SSH_OPT="-e 'ssh -J chihiro_yano@gate.retrieva.jp'"

# # 5. findでファイルリストを作成し、rsyncで転送を実行
# find "$SEARCH_DIR" -name "eng_on_train_end_tasks.csv" -print0 | rsync -aR --progress $DRY_RUN $SSH_OPT --files-from=- --from0 . "$DEST_INFO"



