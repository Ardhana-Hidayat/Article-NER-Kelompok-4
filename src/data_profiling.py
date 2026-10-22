"""
Step 1: Dataset Profiling Script
Loads raw CSV files, analyzes data quality, and outputs profiling metrics.
"""
from pathlib import Path
import glob
import json
import os
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def get_data_dir() -> Path:
    """Resolve dataset path from env or standard directory locations."""
    if "NER_DATA_DIR" in os.environ and os.path.exists(os.environ["NER_DATA_DIR"]):
        return Path(os.environ["NER_DATA_DIR"])
    for candidate in [
        PROJECT_ROOT / "use_cases" / "pnm_study_case" / "sample",
        PROJECT_ROOT.parent / "use_cases" / "pnm_study_case" / "sample",
        PROJECT_ROOT / "dataset",
        PROJECT_ROOT / "Dataset",
        PROJECT_ROOT / "src" / "dataset",
    ]:
        if candidate.exists() and list(candidate.glob("*.csv")):
            return candidate
    return PROJECT_ROOT / "dataset"


def load_dataset(data_dir: Path) -> pd.DataFrame:
    """Read and concatenate all semicolon-separated CSV files."""
    files = sorted(data_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    print(f"Loading {len(files)} CSV files from: {data_dir}")
    dfs = [pd.read_csv(f, sep=";", quotechar='"', encoding="utf-8", on_bad_lines="warn") for f in files]
    return pd.concat(dfs, ignore_index=True)


def profile_and_save(df: pd.DataFrame, out_path: Path) -> dict:
    """Generate profiling metrics and save summary to JSON."""
    metrics = {
        "total_rows": int(len(df)),
        "unique_original_ids": int(df["original_id"].nunique()),
        "duplicate_by_id": int(len(df) - df["original_id"].nunique()),
        "title_null": int(df["title"].isna().sum()),
        "title_empty": int((df["title"].fillna("").str.strip() == "").sum()),
        "body_null": int(df["body"].isna().sum()),
        "body_empty": int((df["body"].fillna("").str.strip() == "").sum()),
        "language_distribution": df["language"].value_counts(dropna=False).to_dict(),
        "entities_null": int(df["entities"].isna().sum()),
        "entities_notnull": int(df["entities"].notna().sum()),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n--- DATASET PROFILING SUMMARY ---")
    print(f"Total Rows Loaded     : {metrics['total_rows']}")
    print(f"Unique Articles (ID)  : {metrics['unique_original_ids']} (Duplicates: {metrics['duplicate_by_id']})")
    print(f"Missing Values        : Title={metrics['title_null']}, Body={metrics['body_null']}")
    print(f"Language Distribution : {metrics['language_distribution']}")
    print(f"Entities Column Null  : {metrics['entities_null']} / {metrics['total_rows']}")
    print(f"Results saved to      : {out_path}\n")
    return metrics


if __name__ == "__main__":
    data_dir = get_data_dir()
    df = load_dataset(data_dir)
    profile_and_save(df, PROJECT_ROOT / "data" / "profiling_results.json")
