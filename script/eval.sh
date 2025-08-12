# models=("cl-nagoya/ruri-v3-30m" "cl-nagoya/ruri-v3-70m" "cl-nagoya/ruri-v3-130m" "cl-nagoya/ruri-v3-310m")
# models=("Qwen/Qwen3-Embedding-0.6B" "Qwen/Qwen3-Embedding-4B" "Qwen/Qwen3-Embedding-8B")
models=("sbintuitions/modernbert-ja-30m" "cl-nagoya/ruri-v3-pt-30m" "sbintuitions/modernbert-ja-130m" "sbintuitions/modernbert-ja-310m")
# models=("sbintuitions/modernbert-ja-30m" "cl-nagoya/ruri-v3-pt-30m" "Qwen/Qwen3-Embedding-4B" "sbintuitions/modernbert-ja-130m" "sbintuitions/modernbert-ja-310m")
for model_name in "${models[@]}"; do
    uv run python eval.py \
        --model_name "$model_name" \
        --batch_size 64 \
        --num_workers 4
done