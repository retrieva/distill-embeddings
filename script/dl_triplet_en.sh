uv run hf download sentence-transformers/embedding-training-data \
  --repo-type dataset \
  --local-dir ./data/gte_plus-eng/triplet/raw
gzip -dvf ./data/gte_plus-eng/triplet/raw/*.gz