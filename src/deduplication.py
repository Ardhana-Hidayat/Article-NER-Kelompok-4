"""
Step 2: Multi-Stage Deduplication Script
Melakukan deduplikasi bertingkat:
1. Hapus duplikat berdasarkan original_id (keep='first')
2. Hapus duplikat sindikasi berita berdasarkan MD5 hash dari gabungan (title + body)
Menghasilkan 741 artikel unik.
"""
from pathlib import Path
import hashlib
import json
import os
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def get_data_dir() -> Path:
    """Resolve dataset path."""
    if "NER_DATA_DIR" in os.environ and os.path.exists(os.environ["NER_DATA_DIR"]):
        return Path(os.environ["NER_DATA_DIR"])
    for candidate in [
        PROJECT_ROOT / "use_cases" / "pnm_study_case" / "sample",
        PROJECT_ROOT / "Dataset",
        PROJECT_ROOT / "dataset",
        PROJECT_ROOT / "src" / "dataset",
    ]:
        if candidate.exists() and list(candidate.glob("*.csv")):
            return candidate
    return PROJECT_ROOT / "Dataset"


def load_dataset(data_dir: Path) -> pd.DataFrame:
    """Read and concatenate all semicolon-separated CSV files."""
    files = sorted(data_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    print(f"Loading {len(files)} CSV files from: {data_dir}")
    dfs = [pd.read_csv(f, sep=";", quotechar='"', encoding="utf-8", on_bad_lines="warn") for f in files]
    return pd.concat(dfs, ignore_index=True)


def compute_md5(text: str) -> str:
    """Compute MD5 hash for a given text string."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def deduplicate_articles(df: pd.DataFrame) -> tuple:
    """
    Perform multi-stage deduplication:
    Stage 1: Deduplicate by original_id  (keep='first')
    Stage 2: Deduplicate by MD5 hash of (title + body)
              combined_text = f"{title}\\n\\n{body}"

    Returns:
        df_clean : deduplicated DataFrame
        stats    : dict of deduplication statistics
    """
    initial_count = len(df)

    # --- TAHAP 1: Berdasarkan original_id ---
    df_stage1 = df.drop_duplicates(subset=["original_id"], keep="first").copy()
    stage1_count = len(df_stage1)
    dup_id_removed = initial_count - stage1_count

    # --- TAHAP 2: Berdasarkan MD5 hash dari (title + body) ---
    # Gabungkan title dan body lalu hitung MD5 hash
    combined_text = (
        df_stage1["title"].fillna("").astype(str)
        + df_stage1["body"].fillna("").astype(str)
    )
    df_stage1["content_hash"] = combined_text.apply(compute_md5)

    df_stage2 = df_stage1.drop_duplicates(subset=["content_hash"], keep="first").copy()
    final_count = len(df_stage2)
    dup_content_removed = stage1_count - final_count

    stats = {
        "initial_total": initial_count,
        "after_stage_1_id": stage1_count,
        "stage_1_duplicates_removed": dup_id_removed,
        "after_stage_2_content": final_count,
        "stage_2_duplicates_removed": dup_content_removed,
        "total_duplicates_removed": initial_count - final_count,
    }

    return df_stage2, stats


def main():
    data_dir = get_data_dir()
    df_raw = load_dataset(data_dir)

    print("\n--- MEMULAI DEDUPLIKASI BERTINGKAT ---")
    df_clean, stats = deduplicate_articles(df_raw)

    print(f"1. Total Baris Awal               : {stats['initial_total']}")
    print(f"2. Setelah Tahap 1 (original_id)  : {stats['after_stage_1_id']} "
          f"(Duplikat ID dibuang: {stats['stage_1_duplicates_removed']})")
    print(f"3. Setelah Tahap 2 (MD5 Hash Teks): {stats['after_stage_2_content']} "
          f"(Duplikat Sindikasi dibuang: {stats['stage_2_duplicates_removed']})")
    print(f"\n=> HASIL AKHIR: {stats['after_stage_2_content']} ARTIKEL UNIK")

    # Simpan hasil deduplikasi
    output_path = PROJECT_ROOT / "data" / "deduplicated_articles.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(output_path, sep=";", index=False, encoding="utf-8")
    print(f"File tersimpan di                 : {output_path}")

    # Simpan statistik
    stats_path = PROJECT_ROOT / "data" / "deduplication_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Statistik tersimpan di            : {stats_path}")


if __name__ == "__main__":
    main()
