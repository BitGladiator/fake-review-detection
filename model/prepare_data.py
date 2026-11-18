"""
prepare_data.py
===============
Fake Review Detection — Data Preparation Pipeline
Loads raw Yelp metadata (Sheet1 & Sheet2) and review text, merges them,
cleans and maps labels, then saves a clean parquet file.

Run from the /model directory:
    python prepare_data.py
"""

import os
import gc
import time

import numpy as np
import pandas as pd
import psutil


print("=== Fake Review Detection: Data Preparation ===")


T0 = time.time()


BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RAW_DIR     = os.path.join(BASE_DIR, "data", "raw")
OUTPUT_DIR  = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "yelp_nyc_clean.parquet")

META_FILE   = os.path.join(RAW_DIR, "Yelp Metadata.xlsx")
TEXT_FILE   = os.path.join(RAW_DIR, "Yelp dataset.xlsx")



def print_memory(label: str = "") -> None:
    """Print current process RSS memory in MB."""
    rss_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    tag = f"[{label}] " if label else ""
    print(f"  {tag}Memory usage: {rss_mb:.1f} MB RSS")


print_memory("start")



_ID_CANDIDATES = ["Review_id", "review_id", "ReviewID", "id", "ID",
                  "review_ID", "REVIEW_ID"]

def find_id_column(df: pd.DataFrame, candidates=_ID_CANDIDATES) -> str | None:
    """
    Return the first column name from *candidates* that exists in *df*
    (case-insensitive).  Returns None if none found.
    """
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None



_TEXT_CANDIDATES = ["Review", "review", "Text", "text", "Review_text",
                    "review_text", "ReviewText", "content", "Content"]

def find_text_column(df: pd.DataFrame, candidates=_TEXT_CANDIDATES) -> str | None:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None



_LABEL_CANDIDATES = ["Label", "label", "LABEL", "Fake", "fake", "is_spam"]

def find_label_column(df: pd.DataFrame, candidates=_LABEL_CANDIDATES) -> str | None:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None





print("\n── Step 1: Loading Yelp Metadata.xlsx ──")

try:
    sheet1 = pd.read_excel(META_FILE, sheet_name="Sheet1", engine="openpyxl")
    print(f"  Sheet1 shape : {sheet1.shape}")
except Exception as exc:
    raise RuntimeError(f"Could not load Sheet1 from {META_FILE}: {exc}") from exc

try:
    sheet2 = pd.read_excel(META_FILE, sheet_name="Sheet2", engine="openpyxl")
    print(f"  Sheet2 shape : {sheet2.shape}")
except Exception as exc:
    raise RuntimeError(f"Could not load Sheet2 from {META_FILE}: {exc}") from exc

print_memory("after metadata load")





print("\n── Step 2: Loading Yelp dataset.xlsx ──")

try:
    text_df = pd.read_excel(TEXT_FILE, engine="openpyxl")   
    print(f"  Text dataset shape: {text_df.shape}")
except Exception as exc:
    raise RuntimeError(f"Could not load {TEXT_FILE}: {exc}") from exc

print_memory("after text load")





print("\n── Step 3: Inspection ──")

for name, df in [("Sheet1", sheet1), ("Sheet2", sheet2), ("TextDataset", text_df)]:
    print(f"\n  [{name}] columns  : {list(df.columns)}")
    print(f"  [{name}] dtypes   :\n{df.dtypes.to_string()}")
    print(f"  [{name}] head(3)  :\n{df.head(3).to_string()}")
    print(f"  [{name}] nulls    :\n{df.isnull().sum().to_string()}")


print("\n── Step 4: Concatenating metadata sheets ──")

cols1 = set(sheet1.columns)
cols2 = set(sheet2.columns)

if cols1 == cols2:
    print("  Columns match exactly — concatenating with ignore_index.")
    metadata = pd.concat([sheet1, sheet2], ignore_index=True)
else:
    print(f"  Column mismatch detected!")
    print(f"    Sheet1 only : {cols1 - cols2}")
    print(f"    Sheet2 only : {cols2 - cols1}")
    print("  Attempting case-insensitive fuzzy alignment …")


    s1_lower = {c.lower(): c for c in sheet1.columns}
    s2_lower = {c.lower(): c for c in sheet2.columns}
    common_lower = set(s1_lower) & set(s2_lower)

    if common_lower:

        rename2 = {s2_lower[k]: s1_lower[k] for k in common_lower}
        sheet2_aligned = sheet2.rename(columns=rename2)
        metadata = pd.concat([sheet1, sheet2_aligned], ignore_index=True)
        print(f"  Aligned on {len(common_lower)} common columns (fuzzy). "
              f"Non-matching columns filled with NaN.")
    else:
        print("  No common columns found even after fuzzy match — concatenating with outer join.")
        metadata = pd.concat([sheet1, sheet2], join="outer", ignore_index=True)

print(f"  Combined metadata shape: {metadata.shape}")

del sheet1, sheet2
gc.collect()
print_memory("after metadata concat")





print("\n── Step 5: Merging metadata with text dataset ──")

meta_id_col = find_id_column(metadata)
text_id_col = find_id_column(text_df)

if meta_id_col is None:
    print(f"  WARNING: Could not find ID column in metadata. Columns: {list(metadata.columns)}")
    raise ValueError("No recognisable Review-ID column in metadata.")
if text_id_col is None:
    print(f"  WARNING: Could not find ID column in text dataset. Columns: {list(text_df.columns)}")
    raise ValueError("No recognisable Review-ID column in text dataset.")

print(f"  Merging on metadata['{meta_id_col}'] ↔ text['{text_id_col}']")


if meta_id_col != "Review_id":
    metadata = metadata.rename(columns={meta_id_col: "Review_id"})
if text_id_col != "Review_id":
    text_df = text_df.rename(columns={text_id_col: "Review_id"})

df = pd.merge(metadata, text_df, on="Review_id", how="inner")
print(f"  Merged shape: {df.shape}")

del metadata, text_df
gc.collect()
print_memory("after merge")





print("\n── Step 6: Resolving text and label columns ──")

text_col  = find_text_column(df)
label_col = find_label_column(df)

if text_col is None:
    print(f"  Available columns: {list(df.columns)}")
    raise ValueError("No recognisable Review-text column found.")
if label_col is None:
    print(f"  Available columns: {list(df.columns)}")
    raise ValueError("No recognisable Label column found.")

print(f"  Text column  : '{text_col}'")
print(f"  Label column : '{label_col}'")

# Standardise names
if text_col != "Review":
    df = df.rename(columns={text_col: "Review"})
if label_col != "Label":
    df = df.rename(columns={label_col: "Label"})





print("\n── Step 7: Dropping null Review / Label rows ──")

before = len(df)
df = df.dropna(subset=["Review", "Label"])
after  = len(df)
print(f"  Dropped {before - after} rows with null Review or Label. Remaining: {after}")





print("\n── Step 8: Dropping duplicate Review_ids ──")

before = len(df)
df = df.drop_duplicates(subset=["Review_id"])
after  = len(df)
print(f"  Dropped {before - after} duplicate Review_id rows. Remaining: {after}")





print("\n── Step 9: Label mapping (-1 → 1 Spam, 1 → 0 Genuine) ──")

print("  Class distribution BEFORE mapping:")
print(df["Label"].value_counts().to_string())

label_map = {-1: 1, 1: 0}
unmapped = set(df["Label"].unique()) - set(label_map.keys())
if unmapped:
    print(f"  WARNING: Unexpected label values found: {unmapped} — they will become NaN.")

df["Label"] = df["Label"].map(label_map)

print("\n  Class distribution AFTER mapping (0=Genuine, 1=Spam):")
print(df["Label"].value_counts().to_string())


null_labels = df["Label"].isnull().sum()
if null_labels > 0:
    print(f"  Dropping {null_labels} rows with unmapped labels.")
    df = df.dropna(subset=["Label"])

df["Label"] = df["Label"].astype(np.int8)





print("\n── Step 10: Final dataset summary ──")
print(f"  Shape   : {df.shape}")
print(f"  Columns : {list(df.columns)}")
print("\n  Sample (5 rows):")
print(df.sample(min(5, len(df)), random_state=42).to_string())





print(f"\n── Step 11: Saving to {OUTPUT_PATH} ──")

os.makedirs(OUTPUT_DIR, exist_ok=True)
df.to_parquet(OUTPUT_PATH, engine="pyarrow", index=False)
print(f"  File size: {os.path.getsize(OUTPUT_PATH) / 1024:.1f} KB")

gc.collect()
print_memory("after save")





elapsed = time.time() - T0
print(f"\n  Total elapsed time: {elapsed:.2f}s")
print_memory("final")
print(f"\nData preparation complete! Saved to: {OUTPUT_PATH}")
