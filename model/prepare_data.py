"""
Fake Review Detection – Data Preparation
Reads ALL sheets from both Yelp Metadata.xlsx and Yelp dataset.xlsx,
merges them, cleans labels, and saves yelp_nyc_clean.parquet.
"""

import os
import gc
import time
import numpy as np
import pandas as pd
import psutil
import openpyxl  


BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
RAW_DIR    = os.path.join(BASE_DIR, "data", "raw")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "yelp_nyc_clean.parquet")
META_FILE  = os.path.join(RAW_DIR, "Yelp Metadata.xlsx")
TEXT_FILE  = os.path.join(RAW_DIR, "Yelp dataset.xlsx")


def memory_mb() -> float:
    proc = psutil.Process(os.getpid())
    return proc.memory_info().rss / 1024 ** 2


def print_memory(label: str = "") -> None:
    print(f"  [MEM{' – ' + label if label else ''}] RSS = {memory_mb():.1f} MB")


def banner(step: int, desc: str) -> None:
    print(f"\n── Step {step}: {desc} ──")


def find_col(columns, candidates):
    """Return first column name (case-insensitive) that matches any candidate."""
    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None



def main():
    t_start = time.time()
    print("=" * 55)
    print("  Fake Review Detection: Data Preparation")
    print("=" * 55)
    print_memory("start")


    banner(1, "Load ALL sheets from BOTH files")


    with pd.ExcelFile(META_FILE, engine="openpyxl") as xf:
        meta_sheets = xf.sheet_names
    print(f"Metadata sheets: {meta_sheets}")
    meta_frames = []
    for sheet in meta_sheets:
        df_s = pd.read_excel(META_FILE, sheet_name=sheet, engine="openpyxl")
        print(f"  Sheet '{sheet}': {df_s.shape}")
        meta_frames.append(df_s)
    metadata_df = pd.concat(meta_frames, ignore_index=True)
    print(f"Loaded {len(meta_sheets)} sheets from Metadata: {len(metadata_df)} rows")
    del meta_frames
    gc.collect()


    with pd.ExcelFile(TEXT_FILE, engine="openpyxl") as xf:
        text_sheets = xf.sheet_names
    print(f"Dataset sheets: {text_sheets}")
    text_frames = []
    for sheet in text_sheets:
        df_s = pd.read_excel(TEXT_FILE, sheet_name=sheet, engine="openpyxl")
        print(f"  Sheet '{sheet}': {df_s.shape}")
        text_frames.append(df_s)
    text_df = pd.concat(text_frames, ignore_index=True)
    print(f"Loaded {len(text_sheets)} sheets from Dataset: {len(text_df)} rows")
    del text_frames
    gc.collect()
    print_memory("after load")


    banner(2, "Inspect and standardise columns")


    metadata_df.columns = [c.lower().strip() for c in metadata_df.columns]
    text_df.columns     = [c.lower().strip() for c in text_df.columns]

    print("\nMetadata columns:", list(metadata_df.columns))
    print("Metadata dtypes:\n", metadata_df.dtypes)
    print("Metadata null counts:\n", metadata_df.isnull().sum())

    print("\nDataset columns:", list(text_df.columns))
    print("Dataset dtypes:\n", text_df.dtypes)
    print("Dataset null counts:\n", text_df.isnull().sum())


    id_candidates = ["review_id", "reviewid", "id"]
    meta_id_col = find_col(metadata_df.columns, id_candidates)
    text_id_col = find_col(text_df.columns, id_candidates)

    if meta_id_col is None:
        raise KeyError(f"Cannot find ID column in metadata. Columns: {list(metadata_df.columns)}")
    if text_id_col is None:
        raise KeyError(f"Cannot find ID column in text dataset. Columns: {list(text_df.columns)}")

    if meta_id_col != "review_id":
        metadata_df.rename(columns={meta_id_col: "review_id"}, inplace=True)
        print(f"Renamed metadata ID col '{meta_id_col}' → 'review_id'")
    if text_id_col != "review_id":
        text_df.rename(columns={text_id_col: "review_id"}, inplace=True)
        print(f"Renamed text ID col '{text_id_col}' → 'review_id'")

    # Identify and rename text column
    text_col = find_col(text_df.columns, ["review", "text", "review_text", "content"])
    if text_col is None:
        raise KeyError(f"Cannot find text column in dataset. Columns: {list(text_df.columns)}")
    if text_col != "text":
        text_df.rename(columns={text_col: "text"}, inplace=True)
        print(f"Renamed text col '{text_col}' → 'text'")

    # Identify and rename label column
    label_col = find_col(metadata_df.columns, ["label", "fake", "is_spam"])
    if label_col is None:
        raise KeyError(f"Cannot find label column in metadata. Columns: {list(metadata_df.columns)}")
    if label_col != "label":
        metadata_df.rename(columns={label_col: "label"}, inplace=True)
        print(f"Renamed label col '{label_col}' → 'label'")


    banner(3, "Merge metadata + text on review_id (inner join)")

    keep_meta_cols = list(metadata_df.columns)  
    print(f"Before merge: metadata={len(metadata_df)}, text={len(text_df)}")

    merged = pd.merge(
        metadata_df[keep_meta_cols],
        text_df[["review_id", "text"]],
        on="review_id",
        how="inner",
    )
    print(f"After inner merge: {len(merged)} rows")
    del metadata_df, text_df
    gc.collect()
    print_memory("after merge")


    banner(4, "Drop nulls and duplicates")

    n0 = len(merged)

    merged = merged[merged["text"].notna()]
    merged = merged[merged["text"].astype(str).str.strip() != ""]
    print(f"Dropped {n0 - len(merged)} rows with null/empty text. Remaining: {len(merged)}")

    n1 = len(merged)
    merged = merged[merged["label"].notna()]
    print(f"Dropped {n1 - len(merged)} rows with null label. Remaining: {len(merged)}")

    n2 = len(merged)
    merged = merged.drop_duplicates(subset="review_id", keep="first")
    print(f"Dropped {n2 - len(merged)} duplicate review_ids. Remaining: {len(merged)}")


    banner(5, "Label mapping (-1→1 Spam, 1→0 Genuine)")

    print("BEFORE mapping – raw label distribution:")
    print(merged["label"].value_counts(dropna=False).to_string())


    expected = {-1, 1}
    unique_labels = set(merged["label"].dropna().unique())
    unexpected = unique_labels - expected
    if unexpected:
        print(f"WARNING: unexpected label values {unexpected} – dropping those rows.")
        merged = merged[merged["label"].isin(expected)]
        print(f"  Remaining after drop: {len(merged)} rows")

    merged["label"] = merged["label"].map({-1: 1, 1: 0})

    print("AFTER mapping – class distribution:")
    dist = merged["label"].value_counts().sort_index()
    for lbl, cnt in dist.items():
        name = "Genuine" if lbl == 0 else "Spam"
        print(f"  {name} ({lbl}): {cnt}")


    banner(6, "Normalise columns and select final set")


    prod_col = find_col(merged.columns, ["product_id"])
    if prod_col is None:

        prod_col = next(
            (c for c in merged.columns if "product" in c.lower()), None
        )
    if prod_col and prod_col != "product_id":
        merged.rename(columns={prod_col: "product_id"}, inplace=True)
        print(f"Renamed '{prod_col}' → 'product_id'")
    elif prod_col is None:
        merged["product_id"] = 0
        print("product_id column not found; filled with 0")


    date_col = find_col(merged.columns, ["review_date"])
    if date_col is None:
        date_col = next(
            (c for c in merged.columns if "date" in c.lower()), None
        )
    if date_col and date_col != "review_date":
        merged.rename(columns={date_col: "review_date"}, inplace=True)
        print(f"Renamed '{date_col}' → 'review_date'")
    elif date_col is None:
        merged["review_date"] = pd.NaT
        print("review_date column not found; filled with NaT")


    rating_col = find_col(merged.columns, ["rating"])
    if rating_col is None:
        merged["rating"] = 3.0
        print("rating column not found; filled with 3.0")
    else:
        merged["rating"] = pd.to_numeric(merged[rating_col], errors="coerce")
        merged["rating"] = merged["rating"].clip(1.0, 5.0).fillna(3.0)


    final_cols = ["review_id", "product_id", "rating", "label", "review_date", "text"]
    for col in final_cols:
        if col not in merged.columns:
            if col == "review_date":
                merged[col] = pd.NaT
            elif col == "product_id":
                merged[col] = 0
            else:
                merged[col] = np.nan

    df = merged[final_cols].copy()
    del merged
    gc.collect()

    print(f"\nFinal shape: {df.shape}")
    print("Final column dtypes:")
    print(df.dtypes.to_string())


    banner(7, "Save parquet")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, engine="pyarrow", index=False)

    file_kb = os.path.getsize(OUTPUT_PATH) / 1024
    print(f"\nSaved: {OUTPUT_PATH}")
    print(f"Final shape: {df.shape}")
    print(f"File size: {file_kb:.1f} KB")

    print("\nFinal label distribution:")
    total = len(df)
    for lbl in sorted(df["label"].unique()):
        name = "Genuine" if lbl == 0 else "Spam"
        cnt  = int((df["label"] == lbl).sum())
        pct  = 100 * cnt / total if total else 0
        print(f"  {name} ({lbl}): {cnt} ({pct:.2f}%)")

    gc.collect()

    elapsed = time.time() - t_start
    print(f"\nTotal elapsed time: {elapsed:.1f}s")
    print_memory("end")
    print("\n Data preparation complete!")


if __name__ == "__main__":
    main()
