"""
IO Utilities — data_pipeline.utils.io
File I/O helpers for the per-asset data pipeline.
"""

import glob
import os
import pandas as pd

def check_ratio(df: pd.DataFrame, label_col: str = "label") -> None:
    counts = df[label_col].value_counts()
    total = len(df)
    label_0 = counts.get(0, 0)
    label_1 = counts.get(1, 0)
    print(f"Label 0: {label_0} ({label_0 / total:.2%})")
    print(f"Label 1: {label_1} ({label_1 / total:.2%})")
    print(f"Ratio 0:1 = {label_0 / label_1:.2f}" if label_1 > 0 else "No label-1 samples.")

def read_and_concat_csv(folder_path):
    csv_pattern = os.path.join(folder_path, "*.csv")
    csv_files = glob.glob(csv_pattern)

    df_list = []
    for file_path in csv_files:
        df = pd.read_csv(file_path)
        df_list.append(df)

    if df_list:
        return pd.concat(df_list, ignore_index=True)
    else:
        return pd.DataFrame()
    

def save_to_csv(df: pd.DataFrame, file_name: str, output_path: str) -> None: 
    os.makedirs(output_path, exist_ok=True) 
    full_path = os.path.join(output_path, file_name) 
    df.to_csv(full_path, index = False)

def data_info(df: pd.DataFrame, name: str = "DataFrame", n_head: int = 5) -> None:
    print(f"\n{'=' * 60}")
    print(f"{name}")
    print(f"{'=' * 60}")

    print(f"Shape: {df.shape}")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print(f"Duplicate rows: {df.duplicated().sum()}")

    print("\nMissing values:")
    missing = df.isna().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("No missing values.")
    else:
        print(missing.sort_values(ascending=False))

    print("\nData types:")
    print(df.dtypes)

    print("\nBasic info:")
    df.info()

    print(f"\nFirst {n_head} rows:")
    print(df.head(n_head))

    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols) > 0:
        print("\nNumeric summary:")
        print(df[numeric_cols].describe().T)

    categorical_cols = df.select_dtypes(exclude="number").columns
    if len(categorical_cols) > 0:
        print("\nCategorical summary:")
        print(df[categorical_cols].describe().T)
