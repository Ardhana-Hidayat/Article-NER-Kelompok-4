"""
Step 3: Annotation Pipeline, Evaluation Examples, and Data Split (70/15/15)
- Weak-supervision pre-annotation using spaCy (en_core_web_sm)
- Exact character span alignment dengan validasi matematis offset:
    combined_text = f"{title}\\n\\n{body}"
    assert combined_text[start:end] == entity_text
- Skema 5 entitas: PERSON, ORG, GPE, DATE, EVENT
  (LOC -> GPE, TIME -> DATE sebagai alias)
- Deterministic 70/15/15 split by original_id (SEED=42)
- Saves JSONL splits dan split_metadata.json
"""
from pathlib import Path
import hashlib
import json
import os
import numpy as np  # type: ignore[import-untyped]
import pandas as pd
import spacy  # type: ignore[import-untyped]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_OUT = PROJECT_ROOT / "data"
SEED = 42

# Mapping label spaCy -> skema 5 entitas proyek
# LOC  -> GPE (lokasi geografis disamakan dengan geo-political entity)
# TIME -> DATE (waktu spesifik disamakan dengan tanggal)
SUPPORTED_LABELS = {
    "PERSON": "PERSON",
    "ORG":    "ORG",
    "GPE":    "GPE",
    "LOC":    "GPE",
    "DATE":   "DATE",
    "TIME":   "DATE",
    "EVENT":  "EVENT",
}


def get_data_dir() -> Path:
    """Resolve dataset directory."""
    if "NER_DATA_DIR" in os.environ and os.path.exists(os.environ["NER_DATA_DIR"]):
        return Path(os.environ["NER_DATA_DIR"])
    for cand in [
        PROJECT_ROOT / "use_cases" / "pnm_study_case" / "sample",
        PROJECT_ROOT / "Dataset",   # folder aktual di repo ini
        PROJECT_ROOT / "dataset",
    ]:
        if cand.exists() and list(cand.glob("*.csv")):
            return cand
    return PROJECT_ROOT / "Dataset"


def load_and_deduplicate(data_dir: Path) -> pd.DataFrame:
    """Load CSVs and perform strict article-level deduplication."""
    files = sorted(data_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    dfs = [pd.read_csv(f, sep=";", quotechar='"', encoding="utf-8", on_bad_lines="warn") for f in files]
    df = pd.concat(dfs, ignore_index=True)

    # Tahap 1: deduplikasi berdasarkan original_id
    df = df.drop_duplicates(subset=["original_id"], keep="first")

    # Tahap 2: deduplikasi berdasarkan MD5 hash dari (title + body)
    df["_hash"] = df.apply(
        lambda r: hashlib.md5(
            (str(r["title"]).strip() + str(r["body"]).strip()).encode("utf-8")
        ).hexdigest(),
        axis=1,
    )
    df = df.drop_duplicates(subset=["_hash"], keep="first").reset_index(drop=True)
    return df


def run_pipeline():
    DATA_OUT.mkdir(parents=True, exist_ok=True)

    # 1. Load dan deduplikasi dataset
    data_dir = get_data_dir()
    print(f"Loading CSVs from: {data_dir}")
    df = load_and_deduplicate(data_dir)
    print(f"Unique articles after deduplication: {len(df)}")

    # 2. Muat model spaCy (nonaktifkan komponen yang tidak diperlukan untuk NER)
    nlp = spacy.load(
        "en_core_web_sm",
        disable=["tagger", "parser", "attribute_ruler", "lemmatizer"],
    )
    nlp.max_length = 200_000
    print(f"spaCy pipeline: {nlp.pipe_names}")

    # 3. Siapkan teks gabungan dan metadata per artikel
    combined_texts = []
    rows_meta = []
    for idx, row in df.iterrows():
        title = str(row["title"]).strip()
        body = str(row["body"]).strip()
        # Skema combined_text yang disepakati untuk validasi offset
        combined = f"{title}\n\n{body}"
        combined_texts.append(combined)
        rows_meta.append({
            "doc_id": int(idx),
            "original_id": row["original_id"],
            "title": title,
            "body": body,
            "combined_text": combined,
            "language": row.get("language", "'en"),
        })

    # 4. Pre-anotasi menggunakan nlp.pipe (batch processing)
    print(f"\nAnnotating {len(combined_texts)} articles ...")
    records = []
    for i, doc in enumerate(nlp.pipe(combined_texts, batch_size=32)):
        combined_text = combined_texts[i]
        entities = []
        for ent in doc.ents:
            mapped = SUPPORTED_LABELS.get(ent.label_)
            if not mapped:
                continue
            start, end = ent.start_char, ent.end_char
            # Validasi matematis offset karakter:
            # combined_text[start:end] harus == entity_text
            if combined_text[start:end] == ent.text:
                entities.append({
                    "text":  ent.text,
                    "label": mapped,
                    "start": start,
                    "end":   end,
                })
        rec = rows_meta[i].copy()
        rec["entities"] = entities
        records.append(rec)

        # Progress log setiap 100 artikel
        if (i + 1) % 100 == 0 or (i + 1) == len(combined_texts):
            total_ents = sum(len(r["entities"]) for r in records)
            print(f"  [{i+1:>4}/{len(combined_texts)}]  spans so far: {total_ents}")

    # 5. Deterministic 70/15/15 split berdasarkan original_id
    np.random.seed(SEED)
    unique_ids = df["original_id"].unique().copy()
    np.random.shuffle(unique_ids)

    n = len(unique_ids)
    train_ids = set(unique_ids[:int(n * 0.70)])
    val_ids   = set(unique_ids[int(n * 0.70):int(n * 0.85)])

    for r in records:
        oid = r["original_id"]
        r["split"] = "train" if oid in train_ids else ("val" if oid in val_ids else "test")

    # 6. Simpan JSONL splits
    splits = {"train": [], "val": [], "test": []}
    for r in records:
        splits[r["split"]].append(r)

    for name, items in splits.items():
        path = DATA_OUT / f"annotations_{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        n_spans = sum(len(x["entities"]) for x in items)
        print(f"Saved {name} split: {len(items)} docs, {n_spans} spans -> {path}")

    # 7. Simpan split metadata
    meta = {
        "seed": SEED,
        "total_unique_articles": len(records),
        "train_docs":  len(splits["train"]),
        "val_docs":    len(splits["val"]),
        "test_docs":   len(splits["test"]),
        "train_spans": sum(len(x["entities"]) for x in splits["train"]),
        "val_spans":   sum(len(x["entities"]) for x in splits["val"]),
        "test_spans":  sum(len(x["entities"]) for x in splits["test"]),
        "total_spans": sum(len(r["entities"]) for r in records),
    }
    meta_path = DATA_OUT / "split_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved split metadata -> {meta_path}")

    # 8. Ringkasan akhir
    print("\n" + "=" * 50)
    print("  SELESAI - Pre-Annotation & Split Summary")
    print("=" * 50)
    print(f"  Total artikel unik : {meta['total_unique_articles']}")
    print(f"  Total spans        : {meta['total_spans']}")
    print(f"  Train (70%)        : {meta['train_docs']} docs, {meta['train_spans']} spans")
    print(f"  Val   (15%)        : {meta['val_docs']} docs, {meta['val_spans']} spans")
    print(f"  Test  (15%)        : {meta['test_docs']} docs, {meta['test_spans']} spans")
    print("=" * 50)


if __name__ == "__main__":
    run_pipeline()
