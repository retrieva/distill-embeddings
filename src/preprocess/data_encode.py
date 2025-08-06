from datasets import Dataset, load_dataset
import argparse
from sentence_transformers import SentenceTransformer
import random
from pathlib import Path
import torch
import numpy as np
import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

def split_sentences(sentences, max_length):
    result = []
    i = 0
    n = len(sentences)
    while i+2 < n:
        # 3以上max_length以下のランダムな長さ
        chunk_size = random.randint(3, min(max_length, n - i))
        result.append("\n".join(sentences[i:i + chunk_size]))
        i += chunk_size
    return result


def split_to_short_texts(texts, short_text_ratio):
    """
    Splits texts into shorter samples based on the specified ratio.
    """
    num_short = int(len(texts) * short_text_ratio)
    short_indices = set(random.sample(range(len(texts)), num_short))
    new_texts = []
    for idx, text in enumerate(texts):
        if idx in short_indices:
            sentences = text.split('\n')
            # Choose 1-5 sentences randomly, ensuring each chunk is at least 20 characters
            chunks = split_sentences(sentences, 10)
            filtered_chunks = []
            for chunk in chunks:
                if len(chunk) >= 20:
                    filtered_chunks.append(chunk)
                elif filtered_chunks:
                    # Merge with the previous chunk if it's too short
                    filtered_chunks[-1] += "\n" + chunk
                else:
                    # If it's the first chunk and too short, keep it anyway
                    filtered_chunks.append(chunk)
            new_texts.extend(filtered_chunks)
        else:
            new_texts.append(text)
    return new_texts


def sequential_sort(texts):
    return sorted(texts, key=len, reverse=True)

def stat_text_lengths(texts):
    lengths = [len(text) for text in texts]
    return {
        "count": int(len(lengths)),
        "min": int(np.min(lengths)),
        "max": int(np.max(lengths)),
        "mean": float(np.mean(lengths)),
        "median": float(np.median(lengths)),
        "std": float(np.std(lengths)),
        "25th_percentile": float(np.percentile(lengths, 25)),
        "75th_percentile": float(np.percentile(lengths, 75))
    }
def main(args):
    # Load a dataset from the Hugging Face Hub
    dataset = load_dataset(
        args.data_name,
        args.subset_name,
        split='train',
    )
    texts = list(dataset['text'][:args.sample_size]) if args.sample_size else list(dataset['text'])
    logger.info(f"Loaded dataset: {args.data_name} ({args.subset_name}) with {len(texts)} samples")
    if args.short_text_ratio > 0:
        texts = split_to_short_texts(texts, args.short_text_ratio)
    # 後ろの方にめっちゃ短いの集まってそうなので、削っとく
    drop_num = int(len(texts) * 0.01)
    texts = sequential_sort(texts)[:-drop_num]
    logger.info(f"Number of texts after splitting: {len(texts)}")
    # Print statistics about the distribution of text lengths
    stats = stat_text_lengths(texts)
    logger.info(f"Text length statistics: {stats}")
    long_texts = [text for text in texts if len(text) > args.threshold]
    short_texts = [text for text in texts if len(text) <= args.threshold]
    teacher_model = SentenceTransformer(args.teacher_model).bfloat16()
    logger.info(f"Loaded teacher model: {args.teacher_model}")

    with torch.no_grad():
        logger.info("-- Check OOM --")
        teacher_model.encode(
            long_texts[:2],
            show_progress_bar=True, 
            batch_size=args.long_batch_size,
            max_length=args.max_length,
            normalize_embeddings=True,
        )
        teacher_model.encode(
            short_texts[:32],
            show_progress_bar=True, 
            batch_size=args.short_batch_size,
            max_length=args.max_length,
            normalize_embeddings=True,
        )
        logger.info("-- Long Encode --")
        # Iterate through the dataset and print each sample
        long_teacher_features = teacher_model.encode(
            long_texts,
            show_progress_bar=True, 
            batch_size=args.long_batch_size,
            max_length=args.max_length,
            normalize_embeddings=True,
        )
        logger.info("-- Short Encode --")
        short_teacher_features = teacher_model.encode(
            short_texts,
            show_progress_bar=True, 
            batch_size=args.short_batch_size,
            max_length=args.max_length,
            normalize_embeddings=True,
            )
    output_dataset = Dataset.from_dict({
        'text': long_texts + short_texts,
        'teacher_features': np.concatenate([long_teacher_features, short_teacher_features]),
    })
    output_path = Path(args.output_dir) / f"{args.teacher_model.replace('/', '_')}_encoded" / f"{args.subset_name}_{args.sample_size if args.sample_size else 'full'}"
    output_path.mkdir(parents=True, exist_ok=True)
    output_dataset.save_to_disk(output_path)
    json.dump(stats, open(output_path / "stats.json", "w"), indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and print samples from a dataset.")
    parser.add_argument("--data_name", type=str, required=True, help="Name of the HF dataset")
    parser.add_argument("--subset_name", type=str, default=None, help="Name of the subset to load")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save the output directory")
    parser.add_argument("--teacher_model", type=str, required=True, help="Path to the teacher model")
    parser.add_argument("--sample_size", type=int, default=None, help="Number of samples to load")
    parser.add_argument("--long_batch_size", type=int, default=1, help="Batch size for processing long texts")
    parser.add_argument("--short_batch_size", type=int, default=32, help="Batch size for processing short texts")
    parser.add_argument("--max_length", type=int, default=None, help="Maximum length for tokenization")
    parser.add_argument("--threshold", type=int, default=4096, help="Threshold for distinguishing long and short texts 文字数なのに注意")
    parser.add_argument("--short_text_ratio", type=float, default=0.3, help="Ratio of short text samples")
    args = parser.parse_args()
    main(args)

