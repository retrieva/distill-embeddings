import numpy as np
import pickle
import os
import logging
from pathlib import Path
from tqdm import tqdm
import torch
from datasets import Dataset
from typing import Union, List

logger = logging.getLogger(__name__)


def get_available_gpus():
    """利用可能なGPU数を取得"""
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 0


def encode_with_checkpoint_multigpu(
    model, texts: List[str], batch_size: int, checkpoint_path: Union[str, Path], max_length: int = None, pool=None
) -> np.ndarray:
    """マルチGPU対応のチェックポイント機能付きエンコーディング"""
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # チェックポイントから復元
    all_features, start_batch = load_checkpoint(checkpoint_path)

    total_batches = (len(texts) + batch_size - 1) // batch_size

    if start_batch > 0:
        logger.info(f"Resuming from batch {start_batch}/{total_batches}")

    # 残りのテキストを取得
    remaining_texts = texts[start_batch * batch_size :]

    if remaining_texts:
        # SentenceTransformerのstart_multi_process_poolを使用
        batch_features = model.encode_multi_process(
            remaining_texts,
            pool=pool,
            batch_size=batch_size,
            max_length=max_length,
            normalize_embeddings=True,
            show_progress_bar=True,
            chunksize=1000,  # プロセス間でのチャンク サイズ
        )
        all_features.extend(batch_features)

    return np.array(all_features)


def setup_multiprocess_pool(model, target_devices=None):
    """マルチプロセスプールをセットアップ"""
    if target_devices is None:
        gpu_count = get_available_gpus()
        if gpu_count > 1:
            target_devices = [f"cuda:{i}" for i in range(gpu_count)]
        else:
            target_devices = ["cuda:0"] if torch.cuda.is_available() else ["cpu"]

    logger.info(f"Setting up multiprocess pool with devices: {target_devices}")
    pool = model.start_multi_process_pool(target_devices=target_devices)
    return pool


# 既存の関数は保持
def stat_text_lengths(texts):
    """テキスト長の統計情報を計算"""
    lengths = [len(text) for text in texts]
    return {
        "count": int(len(lengths)),
        "min": int(np.min(lengths)),
        "max": int(np.max(lengths)),
        "mean": float(np.mean(lengths)),
        "median": float(np.median(lengths)),
        "std": float(np.std(lengths)),
        "25th_percentile": float(np.percentile(lengths, 25)),
        "75th_percentile": float(np.percentile(lengths, 75)),
    }


def flatten_dataset_batch(batch, indices, subset, instruction: str = ""):
    """データセットをフラット化してanc/posを個別のレコードに"""
    results = {"text": [], "subset": [], "id": [], "column": [], "len": []}

    for i, idx in enumerate(indices):
        for column in ["anc", "pos"]:
            text = batch[column][i] if column in batch else ""
            if column == "anc" and instruction != "":
                text = f"Instruct:{instruction}\nQuery:{text}"
            results["text"].append(text)
            results["subset"].append(subset)
            results["id"].append(idx)
            results["column"].append(column)
            results["len"].append(len(text))

    return results


def save_checkpoint(features, batch_idx, checkpoint_path):
    """チェックポイントを保存"""
    checkpoint_data = {"features": features, "batch_idx": batch_idx}
    with open(checkpoint_path, "wb") as f:
        pickle.dump(checkpoint_data, f)
    logger.info(f"Checkpoint saved: batch {batch_idx}, features shape: {np.array(features).shape}")


def load_checkpoint(checkpoint_path):
    """チェックポイントを読み込み"""
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "rb") as f:
            data = pickle.load(f)
        logger.info(
            f"Checkpoint loaded: batch {data['batch_idx']}, features shape: {np.array(data['features']).shape}"
        )
        return data["features"], data["batch_idx"]
    return [], 0


def encode_with_checkpoint(
    model, texts: List[str], batch_size: int, checkpoint_path: Union[str, Path], max_length: int = None
) -> np.ndarray:
    """シングルGPU用のチェックポイント機能付きエンコーディング（既存）"""
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # チェックポイントから復元
    all_features, start_batch = load_checkpoint(checkpoint_path)

    total_batches = (len(texts) + batch_size - 1) // batch_size

    if start_batch > 0:
        logger.info(f"Resuming from batch {start_batch}/{total_batches}")

    for batch_idx in tqdm(
        range(start_batch, total_batches), desc="Encoding", initial=start_batch, total=total_batches
    ):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(texts))
        batch_texts = texts[start_idx:end_idx]

        batch_features = model.encode(
            batch_texts,
            batch_size=len(batch_texts),
            max_length=max_length,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        all_features.extend(batch_features)

        # 定期的にチェックポイント保存（1000バッチごと）
        if (batch_idx + 1) % 1000 == 0:
            save_checkpoint(all_features, batch_idx + 1, checkpoint_path)

    return np.array(all_features)


def save_split_dataset(dataset: Dataset, all_features: np.ndarray, output_path: Union[str, Path]) -> None:
    """データセットを埋め込みと他のデータに分けて保存"""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(output_path)
    np.save(output_path / "emb.npy", all_features)
    logger.info(f"Dataset saved to {output_path}")
    logger.info(f"Features saved to {output_path / 'emb.npy'}")


# def load_split_dataset(dataset_path: Union[str, Path]) -> Dataset:
#     """分割保存されたデータセットを読み込み"""
#     dataset_path = Path(dataset_path)
#     dataset = load_from_disk(dataset_path)
#     embeddings = np.load(dataset_path / "emb.npy",  mmap_mode='r')
#     logger.info(f"Total samples: {len(dataset)}, embeddings: {embeddings.shape}")
#     return


def process_encoded_dataset(
    long_texts: Dataset,
    short_texts: Dataset,
    long_features: Union[np.ndarray, torch.Tensor],
    short_features: Union[np.ndarray, torch.Tensor],
) -> tuple[np.ndarray, Dataset]:
    """エンコード結果を直接元のanc/pos形式に再構築（メモリ効率化）"""
    from datasets import concatenate_datasets

    # numpy配列に統一
    if isinstance(long_features, torch.Tensor):
        long_features = long_features.cpu().numpy()
    if isinstance(short_features, torch.Tensor):
        short_features = short_features.cpu().numpy()

    # サイズチェック
    assert len(long_texts) == len(long_features), f"Long data size mismatch: {len(long_texts)} vs {len(long_features)}"
    assert len(short_texts) == len(short_features), (
        f"Short data size mismatch: {len(short_texts)} vs {len(short_features)}"
    )

    # 全てのテキストデータと特徴量を結合
    all_texts = concatenate_datasets([long_texts, short_texts])
    all_features = np.vstack([long_features, short_features])

    # 結合後のサイズも確認
    assert len(all_texts) == len(all_features), f"Combined data size mismatch: {len(all_texts)} vs {len(all_features)}"

    all_texts = all_texts.add_column(name="emb_idx", column=[i for i in range(len(all_texts))])

    # IDでグループ化
    grouped = {}
    for i, example in enumerate(all_texts):
        key = (example["subset"], example["id"])
        if key not in grouped:
            grouped[key] = {}

        column = example["column"]
        grouped[key][column] = {"text": example["text"], "emb_idx": example["emb_idx"]}

    # 元の形式に再構築
    reconstructed_data = {"anc": [], "pos": [], "anc_emb_idx": [], "pos_emb_idx": [], "subset": [], "id": []}

    incomplete_pairs = 0

    for (subset, example_id), columns in grouped.items():
        if "anc" in columns and "pos" in columns:
            reconstructed_data["anc"].append(columns["anc"]["text"])
            reconstructed_data["pos"].append(columns["pos"]["text"])
            reconstructed_data["anc_emb_idx"].append(columns["anc"]["emb_idx"])
            reconstructed_data["pos_emb_idx"].append(columns["pos"]["emb_idx"])
            reconstructed_data["subset"].append(subset)
            reconstructed_data["id"].append(example_id)
        else:
            incomplete_pairs += 1

    if incomplete_pairs > 0:
        logger.warning(f"Found {incomplete_pairs} incomplete pairs (missing anc or pos)")

    return all_features, Dataset.from_dict(reconstructed_data)
