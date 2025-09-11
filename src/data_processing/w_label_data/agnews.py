"""
AG News 用のオリジンデータ（newsSpace）を整形するスクリプト。

対応フォーマット:
- newsSpace（タブ区切り・本文に物理改行。列: source, url, title, none, category, content, label_id, ts, null）

出力:
 1行1JSON (UTF-8)
  {"text": "<title>\n\n<description>", "label": <label>}

デフォルトは newsSpace を読み、label は category 文字列を出力。
CSV の場合は label（数値）を出力。
"""

from __future__ import annotations

import argparse
import html as htmlmod
import json
try:
    from datasets import load_dataset, Dataset
except Exception:
    load_dataset = None
    Dataset = None
import os
import random
import re
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TextIO

# ========= I/O 基本 =========


def open_text(path: str) -> TextIO:
    # 読み取り中のデコードエラーで停止しないように置換
    return open(path, encoding="utf-8", errors="replace")


# ========= newsSpace パーサ =========


@dataclass(frozen=True)
class NewsRecord:
    source: str
    url: str
    title: str
    category: str
    content: str
    label_id: str | int
    timestamp: str


def parse_news_space(fp: TextIO) -> Iterator[NewsRecord]:
    """
    9カラムTSV（source, url, title, none, category, content, label_id, ts, null）。
    本文に物理改行があるため、タブ8個そろうまで行を連結して1レコードにする。
    """
    buf, tabc = [], 0
    for line in fp:
        buf.append(line)
        tabc += line.count("\t")
        if tabc >= 8:  # 9列=タブ8個
            record = "".join(buf)
            cols = record.split("\t", 8)
            if len(cols) != 9:
                buf, tabc = [], 0
                continue
            source, url, title, _none_field, category, content, label_id, ts, _null_field = cols
            # 本文の行継続（行末バックスラッシュ）を復元
            content = content.replace("\\\n", "\n")
            lbl = int(label_id) if label_id.isdigit() else label_id
            yield NewsRecord(
                source=source,
                url=url,
                title=title.strip(),
                category=category.strip(),
                content=content.strip(),
                label_id=lbl,
                timestamp=ts.strip(),
            )
            buf, tabc = [], 0


# ========= エクスポート =========


def export_newsspace_to_jsonl(in_path: str, out_path: str, limit: int | None = None, joiner: str = "\n\n") -> None:
    n_out = 0
    with open_text(in_path) as fp, open(out_path, "w", encoding="utf-8") as out:
        for i, rec in enumerate(parse_news_space(fp), 1):
            text = rec.title
            if rec.content:
                text = f"{rec.title}{joiner}{rec.content}"
            obj = {"text": text, "label": rec.category}
            out.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n_out += 1
            if limit is not None and n_out >= limit:
                break


# CSV は非対応に変更


# ========= Triplets 構築（text+label JSONL → anc/pos/neg） =========


def clean_text(raw: str) -> str:
    """
    軽量クリーニング:
    - HTMLエンティティのデコード
    - <img ...> を除去
    - <a ...>inner</a> は inner を残してタグ除去
    - その他の HTML タグを除去
    - URL文字列 (http(s)://, www., http___) を除去
    - タイムスタンプの除去（例: 2007-03-05 18:10:50, 2007/03/05 18:10）
    - 余分な空白を圧縮
    """
    if not raw:
        return ""
    s = htmlmod.unescape(raw)
    # <img ...>
    s = re.sub(r"(?is)<img[^>]*>", "", s)
    # <a ...>inner</a> -> inner
    s = re.sub(r"(?is)<a\s+[^>]*>(.*?)</a>", r"\1", s)
    # その他タグ
    s = re.sub(r"(?is)<[^>]+>", "", s)
    # URL除去
    s = re.sub(r"(?i)https?://\S+", "", s)
    s = re.sub(r"(?i)\bwww\.\S+", "", s)
    s = re.sub(r"(?i)http___\S+", "", s)
    # タイムスタンプ（ISOライク）除去: YYYY-MM-DD HH:MM(:SS)? or YYYY/MM/DD HH:MM(:SS)?
    s = re.sub(r"\b\d{4}[-/]\d{2}[-/]\d{2}[ T]\d{2}:\d{2}(?::\d{2})?\b", "", s)
    # 英語日付（例: Mon, 05 Mar 2007 18:10:50 GMT）簡易対応
    s = re.sub(
        r"\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun),?\s+\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}\s+\d{2}:\d{2}(?::\d{2})?(?:\s+[A-Z]{2,4})?\b",
        "",
        s,
        flags=re.IGNORECASE,
    )
    # 12時間表記（例: 6:44 PM）簡易対応
    s = re.sub(r"\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM)\b", "", s, flags=re.IGNORECASE)
    # 連続空白
    s = re.sub(r"\s+", " ", s).strip()
    return s


def is_valid_text(s: str, min_alpha: int = 10, min_length: int = 20) -> bool:
    if not s:
        return False
    t = s.strip().lower()
    if not t:
        return False
    if t == "none":
        return False
    # バックスラッシュやスラッシュのみ
    if set(t) <= {"\\", "/", "-", "_"}:
        return False
    alpha = sum(ch.isalpha() for ch in t)
    return alpha >= min_alpha and len(t) >= min_length


def build_triplets_from_jsonl(
    in_path: str,
    out_path: str,
    num_negs: int = 7,
    seed: int = 42,
    limit: int | None = None,
    min_alpha: int = 10,
    min_length: int = 20,
) -> None:
    rng = random.Random(seed)

    # 1st pass: 全行ロード＆クリーニングし、ラベル→テキスト全集合を構築
    label_to_texts: dict[object, list[str]] = {}
    labels: list[object] = []
    all_examples: list[tuple[str, object]] = []  # (text, label)
    print("[1/2] Loading & cleaning JSONL to build pools ...")
    t0 = time.time()
    total_lines = 0
    total_kept = 0
    with open(in_path, "rb") as f:
        while True:
            line = f.readline()
            if not line:
                break
            try:
                obj = json.loads(line.decode("utf-8", errors="replace"))
            except Exception:
                continue
            lab = obj.get("label")
            if lab not in label_to_texts:
                label_to_texts[lab] = []
                labels.append(lab)
            raw = (obj.get("text") or "").strip()
            txt = clean_text(raw)
            if is_valid_text(txt, min_alpha=min_alpha, min_length=min_length):
                label_to_texts[lab].append(txt)
                all_examples.append((txt, lab))
                total_kept += 1
            total_lines += 1

    elapsed = time.time() - t0
    print(
        f"[1/2] Loaded {total_lines} lines, kept {total_kept} valid in {elapsed:.1f}s"
        f" ({total_lines / max(elapsed, 1e-6):.1f}/s)"
    )
    # print("[1/2] Labels:")
    # for lb in labels:
    # n_valid = len(label_to_texts[lb])
    # print(f"  - {lb}: valid={n_valid}")

    if not labels:
        # 入力が空
        open(out_path, "w").close()
        return

    # 2nd pass: 各行を anc にして pos/neg をサンプリングして書き出し
    print(f"[2/2] Building triplets -> {out_path}")
    t1 = time.time()
    with open(out_path, "w", encoding="utf-8") as out:
        n_out = 0
        for anc_text, anc_label in all_examples:
            # pos: 同ラベルからランダム（可能なら anc と異なるテキスト）
            pool_same = label_to_texts.get(anc_label, [])
            pos_text = anc_text
            if pool_same:
                if len(pool_same) == 1:
                    pos_text = pool_same[0]
                else:
                    for _ in range(20):
                        cand = rng.choice(pool_same)
                        if cand != anc_text:
                            pos_text = cand
                            break

            # neg: 異ラベルから num_negs 件（valid pool のみ）
            neg_texts: list[str] = []
            other_labels = [lb for lb in labels if lb != anc_label and label_to_texts.get(lb)]
            if other_labels:
                seen_offsets: set[int] = set()
                while len(neg_texts) < num_negs:
                    lb = rng.choice(other_labels)  # ラベルをランダム選択
                    pool_other = label_to_texts.get(lb) or []
                    if not pool_other:
                        continue
                    # pool内のインデックスをキーにして重複防止
                    j = rng.randrange(len(pool_other))
                    if (lb, j) in seen_offsets:
                        continue
                    neg_text = pool_other[j]
                    if neg_text and neg_text != anc_text and neg_text != pos_text:
                        neg_texts.append(neg_text)
                        seen_offsets.add((lb, j))

            out.write(
                json.dumps(
                    {
                        "anc": anc_text,
                        "pos": pos_text,
                        "neg": neg_texts,
                        "label": anc_label,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            n_out += 1
            if n_out % 100000 == 0:
                el = time.time() - t1
                print(f"[2/2] wrote {n_out} triplets in {el / 60:.1f} min ({n_out / max(el, 1e-6):.1f}/s)")
            if limit is not None and n_out >= limit:
                break
    t2 = time.time()
    print(f"[done] Triplets: {n_out} lines -> {out_path} in {(t2 - t1) / 60:.1f} min")


def build_triplets_dataset_from_jsonl(
    in_path: str,
    out_dir: str,
    num_negs: int = 7,
    seed: int = 42,
    limit: int | None = None,
    min_alpha: int = 10,
    min_length: int = 20,
    num_proc: int | None = None,
):
    if load_dataset is None:
        raise RuntimeError("datasets ライブラリが見つかりません。pip install datasets してください。")

    num_proc = num_proc or max(os.cpu_count() or 1, 1)

    print("[HF] load_dataset(json) ...")
    t0 = time.time()
    ds = load_dataset("json", data_files=in_path, split="train")

    print("[HF] clean_text (map, num_proc=%d) ..." % num_proc)
    ds = ds.map(lambda ex: {"text": clean_text(ex.get("text", ""))}, num_proc=num_proc, desc="clean_text")

    print("[HF] filter valid (num_proc=%d) ..." % num_proc)
    ds = ds.filter(
        lambda ex: is_valid_text(ex.get("text", ""), min_alpha=min_alpha, min_length=min_length),
        num_proc=num_proc,
        desc="filter_valid",
    )

    # ラベル→テキスト全集合を構築
    print("[HF] build label pools ...")
    label_to_texts: dict[object, list[str]] = {}
    labels: list[object] = []
    for ex in ds:
        lab = ex.get("label")
        if lab not in label_to_texts:
            label_to_texts[lab] = []
            labels.append(lab)
        label_to_texts[lab].append(ex.get("text", ""))

    # anc 対象の制限（プールは全体のまま）
    if limit is not None:
        n = min(len(ds), limit)
        ds_anc = ds.select(range(n))
    else:
        ds_anc = ds

    def _map_one(ex, idx):
        anc_text = ex.get("text", "")
        anc_label = ex.get("label")
        rng = random.Random((seed + int(idx)) & 0xFFFFFFFF)

        # pos
        pos_text = anc_text
        pool_same = label_to_texts.get(anc_label, [])
        if pool_same:
            if len(pool_same) == 1:
                pos_text = pool_same[0]
            else:
                for _ in range(20):
                    cand = rng.choice(pool_same)
                    if cand != anc_text:
                        pos_text = cand
                        break

        # neg
        negs: list[str] = []
        other_labels = [lb for lb in labels if lb != anc_label and label_to_texts.get(lb)]
        seen: set[tuple] = set()
        if other_labels:
            while len(negs) < num_negs:
                lb = rng.choice(other_labels)
                pool_other = label_to_texts.get(lb) or []
                if not pool_other:
                    continue
                j = rng.randrange(len(pool_other))
                if (lb, j) in seen:
                    continue
                cand = pool_other[j]
                if cand and cand != anc_text and cand != pos_text:
                    negs.append(cand)
                    seen.add((lb, j))

        return {"anc": anc_text, "pos": pos_text, "neg": negs, "label": anc_label}

    print("[HF] map build_triplets (num_proc=%d) ..." % num_proc)
    ds_trip = ds_anc.map(
        _map_one,
        with_indices=True,
        num_proc=num_proc,
        remove_columns=ds_anc.column_names,
        desc="build_triplets",
    )

    print(f"[HF] save_to_disk -> {out_dir}")
    ds_trip.save_to_disk(out_dir)
    print(f"[HF] done in {(time.time()-t0)/60:.1f} min: {len(ds_trip)} examples")


def build_triplets_dataset_from_origin(
    in_path: str,
    out_dir: str,
    num_negs: int = 7,
    seed: int = 42,
    limit: int | None = None,
    min_alpha: int = 10,
    min_length: int = 20,
    num_proc: int | None = None,
):
    if Dataset is None:
        raise RuntimeError("datasets ライブラリが見つかりません。pip install datasets してください。")

    def _gen():
        kept = 0
        total = 0
        with open_text(in_path) as fp:
            for rec in parse_news_space(fp):
                total += 1
                title = clean_text(rec.title)
                desc = clean_text(rec.content)
                label = rec.category
                if is_valid_text(title, min_alpha=min_alpha, min_length=min_length) and is_valid_text(
                    desc, min_alpha=min_alpha, min_length=min_length
                ):
                    kept += 1
                    yield {"title": title, "description": desc, "label": label}
                    if limit is not None and kept >= limit:
                        return

    print("[HF] Building base dataset from origin (generator)...")
    t0 = time.time()
    ds_base = Dataset.from_generator(_gen)
    print(f"[HF] Base dataset rows: {len(ds_base)} built in {(time.time()-t0)/60:.1f} min")

    # プール（ラベル→description集）
    print("[HF] Building label->description pools ...")
    label_to_descs: dict[object, list[str]] = {}
    labels: list[object] = []
    for ex in ds_base:
        lab = ex["label"]
        if lab not in label_to_descs:
            label_to_descs[lab] = []
            labels.append(lab)
        label_to_descs[lab].append(ex["description"])

    num_proc = num_proc or max(os.cpu_count() or 1, 1)

    def _to_trip(ex, idx):
        anc_title = ex["title"]
        pos_desc = ex["description"]
        lab = ex["label"]
        rng = random.Random((seed + int(idx)) & 0xFFFFFFFF)

        # neg descs from other labels
        negs: list[str] = []
        other_labels = [lb for lb in labels if lb != lab and label_to_descs.get(lb)]
        seen: set[tuple] = set()
        if other_labels:
            while len(negs) < num_negs:
                lb = rng.choice(other_labels)
                pool = label_to_descs.get(lb) or []
                if not pool:
                    continue
                j = rng.randrange(len(pool))
                if (lb, j) in seen:
                    continue
                cand = pool[j]
                if cand and cand != pos_desc:
                    negs.append(cand)
                    seen.add((lb, j))
        return {"anc": anc_title, "pos": pos_desc, "neg": negs, "label": lab}

    print(f"[HF] Mapping to triplets (num_proc={num_proc}) ...")
    ds_trip = ds_base.map(
        _to_trip,
        with_indices=True,
        num_proc=num_proc,
        remove_columns=ds_base.column_names,
        desc="build_triplets",
    )

    print(f"[HF] save_to_disk -> {out_dir}")
    ds_trip.save_to_disk(out_dir)
    print(f"[HF] done in {(time.time()-t0)/60:.1f} min: {len(ds_trip)} examples")


# ========= CLI =========


def main():
    p = argparse.ArgumentParser(
        description="AG News origin を text+label の JSONL に整形 / 既存JSONLから triplets を生成"
    )
    p.add_argument("--input", default="data/w_label_data/agnews/newsSpace", help="入力ファイルパス（newsSpace）")
    p.add_argument("--output", default="data/w_label_data/agnews/agnews_text_label.jsonl", help="出力JSONLパス")
    p.add_argument("--limit", type=int, default=None, help="最大件数（デバッグ用）")
    # triplets 生成
    p.add_argument("--make-triplets", action="store_true", help="text+label JSONL から anc/pos/neg を構築")
    p.add_argument(
        "--triplets-from-jsonl",
        default=None,
        help="triplets 構築の入力JSONL（未指定なら --output を利用）",
    )
    p.add_argument(
        "--triplets-out",
        default="data/w_label_data/agnews/agnews_triplets.jsonl",
        help="triplets の出力先",
    )
    p.add_argument("--num-negs", type=int, default=7, help="neg の個数")
    p.add_argument("--triplet-limit", type=int, default=None, help="anc 件数の上限（デバッグ用）")
    p.add_argument("--seed", type=int, default=42, help="乱数シード（triplets 用）")
    p.add_argument("--min-length", type=int, default=20, help="テキスト最小長（全体フィルタ）")
    p.add_argument("--min-alpha", type=int, default=10, help="英字最小数（全体フィルタ）")
    p.add_argument("--joiner", default="\n\n", help="title と description の連結文字列")
    # HF datasets 出力
    p.add_argument("--make-triplets-ds", action="store_true", help="HF datasets 形式でtripletsを保存")
    p.add_argument(
        "--triplets-out-ds",
        default="data/w_label_data/agnews/agnews_triplets_hf",
        help="HF datasets の保存ディレクトリ",
    )
    p.add_argument("--num-proc", type=int, default=None, help="map/filter 並列数（デフォルト: CPUコア数）")
    args = p.parse_args()

    # JSONL中間（必要なら）
    export_newsspace_to_jsonl(args.input, args.output, limit=args.limit, joiner=args.joiner)

    if args.make_triplets:
        src = args.triplets_from_jsonl or args.output
        # 出力ディレクトリ作成
        os.makedirs(os.path.dirname(args.triplets_out) or ".", exist_ok=True)
        build_triplets_from_jsonl(
            src,
            args.triplets_out,
            num_negs=args.num_negs,
            seed=args.seed,
            limit=args.triplet_limit,
            min_alpha=args.min_alpha,
            min_length=args.min_length,
        )

    if args.make_triplets_ds:
        # origin（newsSpace/CSV）から直接 datasets を構築
        os.makedirs(args.triplets_out_ds, exist_ok=True)
        build_triplets_dataset_from_origin(
            in_path=args.input,
            out_dir=args.triplets_out_ds,
            num_negs=args.num_negs,
            seed=args.seed,
            limit=args.triplet_limit,
            min_alpha=args.min_alpha,
            min_length=args.min_length,
            num_proc=args.num_proc,
        )


if __name__ == "__main__":
    main()
