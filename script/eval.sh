# models=("cl-nagoya/ruri-v3-30m" "cl-nagoya/ruri-v3-pt-130m" "cl-nagoya/ruri-v3-130m" "cl-nagoya/ruri-v3-pt-310m"  "cl-nagoya/ruri-v3-310m")
# models=("Qwen/Qwen3-Embedding-0.6B" "Qwen/Qwen3-Embedding-4B" "Qwen/Qwen3-Embedding-8B")
# models=("sbintuitions/modernbert-ja-30m" "cl-nagoya/ruri-v3-pt-30m" "sbintuitions/modernbert-ja-130m" "sbintuitions/modernbert-ja-310m")
# models=("sbintuitions/modernbert-ja-30m" "cl-nagoya/ruri-v3-pt-30m" "sbintuitions/modernbert-ja-130m" "sbintuitions/modernbert-ja-310m")
# models=("sbintuitions/modernbert-ja-30m" "cl-nagoya/ruri-v3-pt-30m" "Qwen/Qwen3-Embedding-4B" "sbintuitions/modernbert-ja-130m" "sbintuitions/modernbert-ja-310m")
models=("/home/chihiro_yano/work/distill-embeddings/output/cl-nagoya_ruri-v3-pt-30m/Qwen_Qwen3-Embedding-4B/sample_10BT_100000/e20_bs128_lr5e-05_ckd/distillation/6eblhwpf/checkpoints/last.ckpt"
"/home/chihiro_yano/work/distill-embeddings/output/cl-nagoya_ruri-v3-pt-30m/Qwen_Qwen3-Embedding-4B/sample_10BT_100000/e20_bs128_lr5e-05_kld/distillation/iliow746/checkpoints/last.ckpt" 
"/home/chihiro_yano/work/distill-embeddings/output/cl-nagoya_ruri-v3-pt-30m/Qwen_Qwen3-Embedding-4B/sample_10BT_100000/e20_bs128_lr5e-05_taid-ckd/distillation/rrxewbqo/checkpoints/last.ckpt" 
"/home/chihiro_yano/work/distill-embeddings/output/cl-nagoya_ruri-v3-pt-30m/Qwen_Qwen3-Embedding-4B/sample_10BT_100000/e20_bs128_lr5e-05_taid-kld/distillation/0rdkf0gi/checkpoints/last.ckpt" 
"/home/chihiro_yano/work/distill-embeddings/output/output/sbintuitions_modernbert-ja-30m/Qwen_Qwen3-Embedding-4B/sample_10BT_100000/e20_bs128_lr5e-05_ckd/distillation/2dm1j2nl/checkpoints/last.ckpt" 
"/home/chihiro_yano/work/distill-embeddings/output/output/sbintuitions_modernbert-ja-30m/Qwen_Qwen3-Embedding-4B/sample_10BT_100000/e20_bs128_lr5e-05_kld/distillation/sv9kw0vc/checkpoints/last.ckpt" 
"/home/chihiro_yano/work/distill-embeddings/output/output/sbintuitions_modernbert-ja-30m/Qwen_Qwen3-Embedding-4B/sample_10BT_100000/e20_bs128_lr5e-05_taid-ckd/distillation/4h5h87uz/checkpoints/last.ckpt" 
"/home/chihiro_yano/work/distill-embeddings/output/output/sbintuitions_modernbert-ja-30m/Qwen_Qwen3-Embedding-4B/sample_10BT_100000/e20_bs128_lr5e-05_taid-kld/distillation/wj6sv5n6/checkpoints/last.ckpt" 
"/home/chihiro_yano/work/distill-embeddings/output/output/sbintuitions_modernbert-ja-130m/Qwen_Qwen3-Embedding-4B/sample_10BT_100000/e20_bs128_lr5e-05_kld/distillation/uq4uvxxl/checkpoints/last.ckpt" 
"/home/chihiro_yano/work/distill-embeddings/output/output/sbintuitions_modernbert-ja-130m/Qwen_Qwen3-Embedding-4B/sample_10BT_100000/e20_bs128_lr5e-05_taid-kld/distillation/jmury2nz/checkpoints/last.ckpt"
)
for model_name in "${models[@]}"; do
    uv run python eval.py \
        --model_name "$model_name" \
        --batch_size 64 \
        --num_workers 4
done