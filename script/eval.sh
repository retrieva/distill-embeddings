
# models=("/home/chihiro_yano/work/distill-embeddings/output/cl-nagoya_ruri-v3-pt-30m/Qwen_Qwen3-Embedding-4B/sample_10BT_1000/e3_bs16_lr5e-05_taid-ckd/distillation/2zqc9nu4/checkpoints/last.ckpt")
# models=("/home/chihiro_yano/work/distill-embeddings/output/cl-nagoya_ruri-v3-pt-30m/Qwen_Qwen3-Embedding-4B/sample_10BT_1000/e3_bs16_lr5e-05_taid-kld/distillation/dwp09prg/checkpoints/last.ckpt")
# models=("/home/chihiro_yano/work/distill-embeddings/output/cl-nagoya_ruri-v3-pt-30m/Qwen_Qwen3-Embedding-4B/sample_10BT_1000/e3_bs16_lr5e-05_taid-mse/distillation/rygs174q/checkpoints/last.ckpt")
models=("cl-nagoya/ruri-v3-pt-30m" "Qwen/Qwen3-Embedding-4B")
# models=("cl-nagoya/ruri-v3-30m" "cl-nagoya/ruri-v3-70m" "cl-nagoya/ruri-v3-130m" "cl-nagoya/ruri-v3-310m")
# models=("Qwen/Qwen3-Embedding-0.6B" "Qwen/Qwen3-Embedding-4B" "Qwen/Qwen3-Embedding-8B")

for model_name in "${models[@]}"; do
    uv run python eval.py \
        --model_name "$model_name" \
        --batch_size 64 \
        --num_workers 4
done