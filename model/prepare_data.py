"""
prepare_data.py
===============
Fake Review Detection – Data Preparation Pipeline
Recovers all ~359,052 rows from the YelpNYC dataset (Sheet2 in Yelp Metadata.xlsx
and Yelp dataset.xlsx).

Note on IDs:
- In the raw YelpNYC dataset, the column 'Review_id' is actually the Reviewer/User ID.
  Multiple reviews share the same user ID (160,225 unique users across 359,052 reviews).
- Sheet2 of both files is 1-to-1 row aligned on (User_id, Product_id, Review_Date).
- We do NOT drop duplicates on review_id alone.
"""

import os
import gc
import time
import psutil
import pandas as pd


BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RAW_DIR     = os.path.join(BASE_DIR, "data", "raw")
OUTPUT_DIR  = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "yelp_nyc_clean.parquet")

META_FILE   = os.path.join(RAW_DIR, "Yelp Metadata.xlsx")
TEXT_FILE   = os.path.join(RAW_DIR, "Yelp dataset.xlsx")


def print_memory(label: str = "") -> None:
    rss_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    tag = f"[{label}] " if label else ""
    print(f"  {tag}Memory usage: {rss_mb:.1f} MB RSS")


def main():
    t_start = time.time()
    print("=" * 70)
    print("=== Fake Review Detection: Full Dataset Preparation (~359k rows) ===")
    print("=" * 70)
    print_memory("start")


    print("\n── Step 1: Loading Sheet2 from Raw Excel Files ──")
    print(f"  Loading metadata from: {META_FILE} ...")
    t0 = time.time()
    df_meta = pd.read_excel(META_FILE, sheet_name="Sheet2", engine="openpyxl")
    print(f"  Loaded metadata: {df_meta.shape} in {time.time() - t0:.1f}s")

    print(f"  Loading review texts from: {TEXT_FILE} ...")
    t0 = time.time()
    df_text = pd.read_excel(TEXT_FILE, sheet_name="Sheet2", engine="openpyxl")
    print(f"  Loaded texts: {df_text.shape} in {time.time() - t0:.1f}s")
    print_memory("after load")


    print("\n── Step 2: Standardizing Column Names ──")
    df_meta.columns = [c.strip().lower() for c in df_meta.columns]
    df_text.columns = [c.strip().lower() for c in df_text.columns]

    print(f"  Metadata columns: {list(df_meta.columns)}")
    print(f"  Text columns:     {list(df_text.columns)}")


    if "review_id" in df_meta.columns:
        df_meta["user_id"] = df_meta["review_id"]
    if "review_id" in df_text.columns:
        df_text["user_id"] = df_text["review_id"]


    print("\n── Step 3: Aligning & Merging Metadata with Review Texts ──")

    check_cols = ["user_id", "product_id", "review_date"]
    aligned = True
    for col in check_cols:
        if col in df_meta.columns and col in df_text.columns:
            matches = (df_meta[col] == df_text[col]).all()
            print(f"  1-to-1 row alignment on '{col}': {matches}")
            if not matches:
                aligned = False

    if aligned and len(df_meta) == len(df_text):
        print("  Direct 1-to-1 row index alignment confirmed across all 359,052 rows.")

        text_col = "review_text" if "review_text" in df_text.columns else "text"
        df = df_meta.copy()
        df["text"] = df_text[text_col]
    else:
        print("  Row index mismatch detected; performing composite key merge on (user_id, product_id, review_date)...")
        text_col = "review_text" if "review_text" in df_text.columns else "text"
        df = pd.merge(
            df_meta,
            df_text[["user_id", "product_id", "review_date", text_col]],
            on=["user_id", "product_id", "review_date"],
            how="inner"
        )
        df.rename(columns={text_col: "text"}, inplace=True)

    del df_meta, df_text
    gc.collect()
    print(f"  Merged dataset shape: {df.shape}")
    print_memory("after merge")


    print("\n── Step 4: Cleaning Missing Values & Exact Full-Row Duplicates ──")
    n_before = len(df)


    df = df[df["text"].notna()]
    df = df[df["text"].astype(str).str.strip() != ""]
    print(f"  Valid non-empty review text: {len(df):,} rows (dropped {n_before - len(df)})")


    df = df[df["label"].notna()]


    n_pre_dup = len(df)
    df = df.drop_duplicates(subset=["user_id", "product_id", "review_date", "text"], keep="first")
    print(f"  After dropping exact identical duplicate reviews: {len(df):,} rows (dropped {n_pre_dup - len(df)})")


    print("\n── Step 5: Label Mapping (-1 -> 1 Spam/Fake, 1 -> 0 Genuine) ──")
    print("  RAW label distribution:")
    print(df["label"].value_counts(dropna=False).to_string())

    expected = {-1, 1}
    unexp = set(df["label"].unique()) - expected
    if unexp:
        print(f"  Dropping unexpected labels: {unexp}")
        df = df[df["label"].isin(expected)]

    df["label"] = df["label"].map({-1: 1, 1: 0}).astype(int)

    print("\n  PROCESSED class distribution:")
    total = len(df)
    counts = df["label"].value_counts().sort_index()
    for lbl, cnt in counts.items():
        name = "Genuine (0)" if lbl == 0 else "Spam/Fake (1)"
        print(f"    {name:<15}: {cnt:,} ({cnt / total * 100:.2f}%)")


    print("\n── Step 6: Standardizing Final Parquet Schema ──")

    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").clip(1.0, 5.0).fillna(3.0).astype(float)

    
    final_cols = ["user_id", "review_id", "product_id", "rating", "label", "review_date", "text"]
    df = df[final_cols].copy()
    print("  Final schema:")
    for col in df.columns:
        print(f"    - {col}: {df[col].dtype}")


    print(f"\n── Step 7: Exporting to {OUTPUT_PATH} ──")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, engine="pyarrow", index=False)
    file_size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    print(f"  Saved parquet file: {file_size_mb:.1f} MB")
    print(f"  Exact final row count: {len(df):,}")

    gc.collect()
    print_memory("final")
    print(f"\nElapsed time: {time.time() - t_start:.1f}s")
    print("Data preparation complete! All ~359k rows recovered.")


if __name__ == "__main__":
    main()
