# mkdir data/triplet-en
# cd data/triplet-en
# hf download sentence-transformers/embedding-training-data \
#   --repo-type dataset \
#   --local-dir ./data/triplet-en
gzip -dvf ./data/triplet-en/*.gz