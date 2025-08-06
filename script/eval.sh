
models=("cl-nagoya/ruri-v3-pt-30m" "cl-nagoya/ruri-v3-pt-70m" "cl-nagoya/ruri-v3-pt-130m" "cl-nagoya/ruri-v3-pt-310m")
# models=("cl-nagoya/ruri-v3-30m" "cl-nagoya/ruri-v3-70m" "cl-nagoya/ruri-v3-130m" "cl-nagoya/ruri-v3-310m")
# models=("Qwen/Qwen3-Embedding-0.6B" "Qwen/Qwen3-Embedding-4B" "Qwen/Qwen3-Embedding-8B")

for model_name in "${models[@]}"; do
    uv run python eval.py \
        --model_name "$model_name" \
        --batch_size 64 \
        --num_workers 4
done