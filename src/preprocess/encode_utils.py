import numpy as np
import pickle
import os
import logging
from pathlib import Path
from tqdm import tqdm
import torch
from datasets import Dataset

logger = logging.getLogger(__name__)

def get_available_gpus():
    """利用可能なGPU数を取得"""
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 0

def encode_with_checkpoint_multigpu(model, texts, batch_size, checkpoint_path, max_length=None, pool=None):
    """マルチGPU対応のチェックポイント機能付きエンコーディング"""
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    # チェックポイントから復元
    all_features, start_batch = load_checkpoint(checkpoint_path)
    
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    if start_batch > 0:
        logger.info(f"Resuming from batch {start_batch}/{total_batches}")
    
    # 残りのテキストを取得
    remaining_texts = texts[start_batch * batch_size:]
    
    if remaining_texts:
        # SentenceTransformerのstart_multi_process_poolを使用
        batch_features = model.encode_multi_process(
            remaining_texts,
            pool=pool,
            batch_size=batch_size,
            max_length=max_length,
            normalize_embeddings=True,
            show_progress_bar=True,
            chunksize=1000  # プロセス間でのチャンク サイズ
        )
        all_features.extend(batch_features)
    
    return np.array(all_features)

def setup_multiprocess_pool(model, target_devices=None):
    """マルチプロセスプールをセットアップ"""
    if target_devices is None:
        gpu_count = get_available_gpus()
        if gpu_count > 1:
            target_devices = [f'cuda:{i}' for i in range(gpu_count)]
        else:
            target_devices = ['cuda:0'] if torch.cuda.is_available() else ['cpu']
    
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
        "75th_percentile": float(np.percentile(lengths, 75))
    }

def flatten_dataset_batch(batch, indices, subset):
    """データセットをフラット化してanc/posを個別のレコードに"""
    results = {"text": [], "subset": [], "id": [], "column": [], "len": []}
    
    for i, idx in enumerate(indices):
        for column in ["anc", "pos"]:
            text = batch[column][i] if column in batch else ""
            results["text"].append(text)
            results["subset"].append(subset)
            results["id"].append(idx)
            results["column"].append(column)
            results["len"].append(len(text))
    
    return results

def save_checkpoint(features, batch_idx, checkpoint_path):
    """チェックポイントを保存"""
    checkpoint_data = {
        'features': features,
        'batch_idx': batch_idx
    }
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    logger.info(f"Checkpoint saved: batch {batch_idx}, features shape: {np.array(features).shape}")

def load_checkpoint(checkpoint_path):
    """チェックポイントを読み込み"""
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            data = pickle.load(f)
        logger.info(f"Checkpoint loaded: batch {data['batch_idx']}, features shape: {np.array(data['features']).shape}")
        return data['features'], data['batch_idx']
    return [], 0

def encode_with_checkpoint(model, texts, batch_size, checkpoint_path, max_length=None):
    """シングルGPU用のチェックポイント機能付きエンコーディング（既存）"""
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    # チェックポイントから復元
    all_features, start_batch = load_checkpoint(checkpoint_path)
    
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    if start_batch > 0:
        logger.info(f"Resuming from batch {start_batch}/{total_batches}")
    
    for batch_idx in tqdm(range(start_batch, total_batches), desc="Encoding", initial=start_batch, total=total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(texts))
        batch_texts = texts[start_idx:end_idx]
        
        batch_features = model.encode(
            batch_texts,
            batch_size=len(batch_texts),
            max_length=max_length,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        all_features.extend(batch_features)

        # 定期的にチェックポイント保存（1000バッチごと）
        if (batch_idx + 1) % 1000 == 0:
            save_checkpoint(all_features, batch_idx + 1, checkpoint_path)
    
    return np.array(all_features)

def reconstruct_dataset(encoded_dataset):
    """エンコードされたデータセットを元のanc/pos形式に戻す"""
    # IDでグループ化
    grouped = {}
    for i, example in enumerate(encoded_dataset):
        key = (example["subset"], example["id"])
        if key not in grouped:
            grouped[key] = {}
        
        column = example["column"]
        grouped[key][column] = {
            "text": example["text"],
            "teacher_features": example["teacher_features"]
        }
    
    # 元の形式に再構築
    reconstructed_data = {
        "anc": [],
        "pos": [], 
        "anc_features": [],
        "pos_features": [],
        "subset": [],
        "id": []
    }
    
    for (subset, example_id), columns in grouped.items():
        if "anc" in columns and "pos" in columns:
            reconstructed_data["anc"].append(columns["anc"]["text"])
            reconstructed_data["pos"].append(columns["pos"]["text"])
            reconstructed_data["anc_features"].append(columns["anc"]["teacher_features"])
            reconstructed_data["pos_features"].append(columns["pos"]["teacher_features"])
            reconstructed_data["subset"].append(subset)
            reconstructed_data["id"].append(example_id)
    
    return Dataset.from_dict(reconstructed_data)

def save_split_dataset(dataset, output_path):
    """データセットを埋め込みと他のデータに分けて保存"""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 埋め込み以外のデータを抽出
    metadata = {
        key: dataset[key] for key in dataset.column_names 
        if not key.endswith('_features')
    }
    
    # 埋め込みデータを抽出
    embeddings = {}
    for key in dataset.column_names:
        if key.endswith('_features'):
            embeddings[key] = np.array(dataset[key])
    
    # メタデータをJSONLで保存（軽量）
    metadata_dataset = Dataset.from_dict(metadata)
    metadata_dataset.to_json(output_path / "metadata.jsonl")
    
    # 埋め込みをnumpyで保存（効率的）
    for key, features in embeddings.items():
        np.save(output_path / f"{key}.npy", features)
    
    # インデックス情報を保存（順番保証のため）
    index_info = {
        "total_samples": len(dataset),
        "embedding_keys": list(embeddings.keys()),
        "metadata_keys": list(metadata.keys())
    }
    
    import json
    with open(output_path / "index_info.json", "w") as f:
        json.dump(index_info, f, indent=2)
    
    logger.info(f"Dataset split and saved to {output_path}")
    logger.info(f"Metadata: {len(metadata)} columns, {len(dataset)} samples")
    logger.info(f"Embeddings: {len(embeddings)} arrays")

def load_split_dataset(dataset_path):
    """分割保存されたデータセットを読み込み"""
    dataset_path = Path(dataset_path)
    
    # インデックス情報を読み込み
    with open(dataset_path / "index_info.json", "r") as f:
        index_info = json.load(f)
    
    # メタデータを読み込み
    metadata_dataset = Dataset.from_json(dataset_path / "metadata.jsonl")
    metadata = metadata_dataset.to_dict()
    
    # 埋め込みを読み込み
    embeddings = {}
    for key in index_info["embedding_keys"]:
        embeddings[key] = np.load(dataset_path / f"{key}.npy").tolist()
    
    # データセットを再構築
    combined_dict = {**metadata, **embeddings}
    reconstructed_dataset = Dataset.from_dict(combined_dict)
    
    logger.info(f"Dataset loaded from {dataset_path}")
    logger.info(f"Total samples: {len(reconstructed_dataset)}")
    
    return reconstructed_dataset

def process_encoded_dataset(long_texts, short_texts, long_features, short_features):
    """エンコード結果をデータセットに変換し、再構築"""
    # add_columnの代わりにfrom_dictを使用
    long_dataset = Dataset.from_dict({
        **long_texts.to_dict(),
        'teacher_features': long_features.tolist() if isinstance(long_features, np.ndarray) else long_features
    })
    short_dataset = Dataset.from_dict({
        **short_texts.to_dict(),
        'teacher_features': short_features.tolist() if isinstance(short_features, np.ndarray) else short_features
    })
    
    from datasets import concatenate_datasets
    output_dataset = concatenate_datasets([long_dataset, short_dataset])
    
    # 元の形式に再構築
    return reconstruct_dataset(output_dataset)