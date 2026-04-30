#!/usr/bin/env python
# coding: utf-8

# %% [markdown]
# # CRISP-DM Phase 2: Data Understanding for CARE to Compare
#
# This notebook-style script profiles all three raw wind farms:
#
# - Wind Farm A
# - Wind Farm B
# - Wind Farm C
#
# It is designed for capstone reporting. It reads the raw dataset structure,
# normalizes metadata differences, profiles event files safely, and exports
# tables/charts to `reports/phase2_data_understanding`.
#
# The default mode samples the beginning of each event CSV so it can run on a
# laptop. For final exact row/status counts, set `FULL_EVENT_SCAN = True`.

# %%
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    display
except NameError:
    def display(obj):
        print(obj)

# %% [markdown]
# ## 1. Settings

# %%
FARM_NAMES = ["Wind Farm A", "Wind Farm B", "Wind Farm C"]

# Keep this False first. The raw CSVs are large, especially Wind Farm C.
FULL_EVENT_SCAN = False

# Used when FULL_EVENT_SCAN is False. This samples the first N rows from each
# event CSV for status/train-test/missing-value checks.
SAMPLE_ROWS_PER_EVENT = 2_000

# Set to an integer while debugging, for example 2. Use None for all events.
MAX_EVENTS_PER_FARM = None

CHUNK_SIZE = 250_000
EXPECTED_INTERVAL = pd.Timedelta(minutes=10)
NORMAL_STATUS = {0, 2}

METADATA_COLUMNS = {
    "time_stamp",
    "asset_id",
    "asset",
    "id",
    "train_test",
    "status_type_id",
    "status_type",
    "label",
    "event_id",
    "sequence_id",
}


def find_project_root() -> Path:
    candidates = []
    if "__file__" in globals():
        here = Path(__file__).resolve()
        candidates.extend([here.parent, *here.parents])
    candidates.extend([Path.cwd(), *Path.cwd().parents])

    for candidate in candidates:
        if (candidate / "Dataset" / "raw").exists():
            return candidate

    raise FileNotFoundError("Could not find project root containing Dataset/raw.")


PROJECT_ROOT = find_project_root()
RAW_ROOT = PROJECT_ROOT / "Dataset" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "phase2_data_understanding"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"

for directory in [OUTPUT_DIR, TABLE_DIR, FIGURE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

print("Project root:", PROJECT_ROOT)
print("Raw root:", RAW_ROOT)
print("Output dir:", OUTPUT_DIR)

# %% [markdown]
# ## 2. Helper Functions

# %%
def numeric_event_csvs(datasets_dir: Path) -> list[Path]:
    """Return only event CSV files such as 0.csv, 68.csv, 94.csv."""
    files = [
        path
        for path in datasets_dir.glob("*.csv")
        if path.stem.isdigit()
    ]
    files = sorted(files, key=lambda path: int(path.stem))
    if MAX_EVENTS_PER_FARM is not None:
        files = files[:MAX_EVENTS_PER_FARM]
    return files


def normalize_bool(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y"})
    )


def read_first_row(csv_path: Path, usecols: Iterable[str] | None = None) -> pd.DataFrame:
    try:
        return pd.read_csv(csv_path, sep=";", nrows=1, usecols=usecols)
    except ValueError:
        return pd.read_csv(csv_path, sep=";", nrows=1)


def read_last_nonempty_line(path: Path, block_size: int = 65_536) -> str:
    with path.open("rb") as handle:
        handle.seek(0, 2)
        position = handle.tell()
        buffer = b""

        while position > 0:
            read_size = min(block_size, position)
            position -= read_size
            handle.seek(position)
            buffer = handle.read(read_size) + buffer
            lines = buffer.splitlines()

            if len(lines) > 1 or position == 0:
                for line in reversed(lines):
                    if line.strip():
                        return line.decode("utf-8", errors="replace")

    return ""


def read_feature_description(farm_dir: Path) -> pd.DataFrame:
    feature_path = farm_dir / "feature_description.csv"
    df = pd.read_csv(feature_path, sep=";")
    df["is_angle_bool"] = normalize_bool(df.get("is_angle", pd.Series(False, index=df.index)))
    df["is_counter_bool"] = normalize_bool(df.get("is_counter", pd.Series(False, index=df.index)))
    return df


def read_event_info(farm_name: str, farm_dir: Path, event_csvs: list[Path]) -> pd.DataFrame:
    info = pd.read_csv(farm_dir / "event_info.csv", sep=";")

    if "asset" in info.columns and "asset_id" not in info.columns:
        info = info.rename(columns={"asset": "asset_id"})
    if "asset_id" not in info.columns:
        info["asset_id"] = pd.NA

    info["farm"] = farm_name
    info["event_id"] = pd.to_numeric(info["event_id"], errors="coerce").astype("Int64")

    for col in ["event_start", "event_end"]:
        if col in info.columns:
            info[col] = pd.to_datetime(info[col], errors="coerce")

    if {"event_start", "event_end"}.issubset(info.columns):
        duration = info["event_end"] - info["event_start"]
        info["event_duration_hours"] = duration.dt.total_seconds() / 3600.0

    # Wind Farm B metadata does not include asset_id. Recover it from each
    # event CSV so the report can still describe turbine-level distribution.
    asset_by_event = {}
    for csv_path in event_csvs:
        first = read_first_row(csv_path, usecols=["asset_id"])
        if "asset_id" in first.columns and not first.empty:
            asset_by_event[int(csv_path.stem)] = first.loc[0, "asset_id"]

    missing_asset = info["asset_id"].isna()
    if missing_asset.any():
        info.loc[missing_asset, "asset_id"] = (
            info.loc[missing_asset, "event_id"]
            .astype(int)
            .map(asset_by_event)
        )

    return info


def quick_file_summary(farm_name: str, csv_path: Path) -> dict:
    header = pd.read_csv(csv_path, sep=";", nrows=0).columns.tolist()
    first = read_first_row(csv_path)
    last_fields = read_last_nonempty_line(csv_path).split(";")
    index_by_col = {name: idx for idx, name in enumerate(header)}

    def last_value(column: str):
        idx = index_by_col.get(column)
        if idx is None or idx >= len(last_fields):
            return pd.NA
        return last_fields[idx]

    first_row = first.iloc[0] if not first.empty else pd.Series(dtype=object)
    feature_columns = [col for col in header if col not in METADATA_COLUMNS]

    first_time = pd.to_datetime(first_row.get("time_stamp", pd.NA), errors="coerce")
    last_time = pd.to_datetime(last_value("time_stamp"), errors="coerce")
    file_duration_hours = np.nan
    if pd.notna(first_time) and pd.notna(last_time):
        file_duration_hours = (last_time - first_time).total_seconds() / 3600.0

    return {
        "farm": farm_name,
        "event_id": int(csv_path.stem),
        "csv_path": str(csv_path.relative_to(PROJECT_ROOT)),
        "file_size_mb": csv_path.stat().st_size / (1024 * 1024),
        "csv_columns": len(header),
        "csv_feature_columns": len(feature_columns),
        "has_label_column": "label" in header,
        "first_time": first_time,
        "last_time": last_time,
        "file_duration_hours": file_duration_hours,
        "first_asset_id": first_row.get("asset_id", pd.NA),
        "first_train_test": first_row.get("train_test", pd.NA),
        "first_status_type_id": first_row.get("status_type_id", pd.NA),
    }


def update_counter_rows(
    rows: list[dict],
    farm: str,
    event_id: int,
    name: str,
    counter: Counter,
) -> None:
    for value, count in counter.items():
        rows.append(
            {
                "farm": farm,
                "event_id": event_id,
                "profile_source": "full_scan" if FULL_EVENT_SCAN else "sample",
                name: value,
                "row_count": int(count),
            }
        )


def profile_sample(csv_path: Path, farm_name: str, event_id: int) -> tuple[dict, list[dict], list[dict], list[dict], dict, dict]:
    df = pd.read_csv(csv_path, sep=";", nrows=SAMPLE_ROWS_PER_EVENT, low_memory=False)
    feature_cols = [col for col in df.columns if col not in METADATA_COLUMNS]

    row_count = len(df)
    train_counter = Counter()
    status_counter = Counter()
    label_counter = Counter()

    if "train_test" in df.columns:
        train_counter.update(df["train_test"].astype(str).str.lower().fillna("missing"))
    if "status_type_id" in df.columns:
        status_counter.update(pd.to_numeric(df["status_type_id"], errors="coerce").fillna(-1).astype(int))
    if "label" in df.columns:
        label_counter.update(pd.to_numeric(df["label"], errors="coerce").fillna(-1).astype(int))

    expected_gap_pct = np.nan
    top_gaps = ""
    if "time_stamp" in df.columns and row_count > 1:
        timestamps = pd.to_datetime(df["time_stamp"], errors="coerce")
        diffs = timestamps.diff().dropna()
        if len(diffs) > 0:
            expected_gap_pct = float((diffs == EXPECTED_INTERVAL).mean() * 100.0)
            top_gap_counts = diffs.astype(str).value_counts().head(3)
            top_gaps = " | ".join(f"{gap}: {count}" for gap, count in top_gap_counts.items())

    missing_by_feature = {}
    constant_feature_count = 0
    missing_cells = 0
    total_feature_cells = max(row_count * len(feature_cols), 1)
    if feature_cols:
        missing_counts = df[feature_cols].isna().sum()
        missing_cells = int(missing_counts.sum())
        missing_by_feature = {
            feature: int(count)
            for feature, count in missing_counts.items()
            if int(count) > 0
        }
        nunique = df[feature_cols].nunique(dropna=False)
        constant_feature_count = int((nunique <= 1).sum())
    observed_by_feature = {feature: row_count for feature in feature_cols}

    quality_row = {
        "farm": farm_name,
        "event_id": event_id,
        "profile_source": "sample",
        "rows_profiled": row_count,
        "sample_rows_requested": SAMPLE_ROWS_PER_EVENT,
        "feature_columns_profiled": len(feature_cols),
        "missing_feature_cells": missing_cells,
        "missing_feature_cell_pct": missing_cells / total_feature_cells * 100.0,
        "constant_feature_count_sample": constant_feature_count,
        "expected_10min_gap_pct_sample": expected_gap_pct,
        "top_time_gaps_sample": top_gaps,
    }

    train_rows = []
    status_rows = []
    label_rows = []
    update_counter_rows(train_rows, farm_name, event_id, "train_test", train_counter)
    update_counter_rows(status_rows, farm_name, event_id, "status_type_id", status_counter)
    update_counter_rows(label_rows, farm_name, event_id, "label", label_counter)
    return quality_row, train_rows, status_rows, label_rows, missing_by_feature, observed_by_feature


def profile_full_metadata(csv_path: Path, farm_name: str, event_id: int) -> tuple[dict, list[dict], list[dict], list[dict]]:
    header = pd.read_csv(csv_path, sep=";", nrows=0).columns.tolist()
    usecols = [
        col
        for col in ["time_stamp", "train_test", "status_type_id", "label"]
        if col in header
    ]

    train_counter = Counter()
    status_counter = Counter()
    label_counter = Counter()
    gap_counter = Counter()
    row_count = 0
    previous_time = pd.NaT
    expected_gap_count = 0
    total_gap_count = 0

    for chunk in pd.read_csv(csv_path, sep=";", usecols=usecols, chunksize=CHUNK_SIZE):
        row_count += len(chunk)
        if "train_test" in chunk.columns:
            train_counter.update(chunk["train_test"].astype(str).str.lower().fillna("missing"))
        if "status_type_id" in chunk.columns:
            status_counter.update(pd.to_numeric(chunk["status_type_id"], errors="coerce").fillna(-1).astype(int))
        if "label" in chunk.columns:
            label_counter.update(pd.to_numeric(chunk["label"], errors="coerce").fillna(-1).astype(int))

        if "time_stamp" in chunk.columns:
            timestamps = pd.to_datetime(chunk["time_stamp"], errors="coerce")
            diffs = timestamps.diff().dropna()
            if pd.notna(previous_time) and len(timestamps) > 0:
                first_gap = timestamps.iloc[0] - previous_time
                if pd.notna(first_gap):
                    diffs = pd.concat([pd.Series([first_gap]), diffs], ignore_index=True)
            if len(timestamps) > 0:
                previous_time = timestamps.iloc[-1]
            total_gap_count += len(diffs)
            expected_gap_count += int((diffs == EXPECTED_INTERVAL).sum())
            gap_counter.update(diffs.astype(str).value_counts().head(10).to_dict())

    expected_gap_pct = np.nan
    if total_gap_count > 0:
        expected_gap_pct = expected_gap_count / total_gap_count * 100.0

    quality_row = {
        "farm": farm_name,
        "event_id": event_id,
        "profile_source": "full_scan",
        "rows_profiled": row_count,
        "sample_rows_requested": np.nan,
        "feature_columns_profiled": np.nan,
        "missing_feature_cells": np.nan,
        "missing_feature_cell_pct": np.nan,
        "constant_feature_count_sample": np.nan,
        "expected_10min_gap_pct_sample": expected_gap_pct,
        "top_time_gaps_sample": " | ".join(
            f"{gap}: {count}" for gap, count in gap_counter.most_common(3)
        ),
    }

    train_rows = []
    status_rows = []
    label_rows = []
    update_counter_rows(train_rows, farm_name, event_id, "train_test", train_counter)
    update_counter_rows(status_rows, farm_name, event_id, "status_type_id", status_counter)
    update_counter_rows(label_rows, farm_name, event_id, "label", label_counter)
    return quality_row, train_rows, status_rows, label_rows


def write_csv(df: pd.DataFrame, filename: str) -> Path:
    path = TABLE_DIR / filename
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def markdown_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    shown = df.head(max_rows).copy()
    shown = shown.fillna("")
    columns = list(shown.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in shown.iterrows():
        values = [str(row[col]).replace("|", "/") for col in columns]
        lines.append("| " + " | ".join(values) + " |")
    if len(df) > max_rows:
        lines.append(f"\n_Only the first {max_rows} rows are shown._")
    return "\n".join(lines)


def save_current_figure(filename: str) -> Path:
    path = FIGURE_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    return path


def add_vertical_bar_labels(ax, fmt: str = "{:.0f}", padding_points: int = 3) -> None:
    """Add numeric labels above positive vertical bars."""
    ymin, ymax = ax.get_ylim()
    values = [
        patch.get_height()
        for patch in ax.patches
        if np.isfinite(patch.get_height()) and patch.get_height() > 0
    ]
    if not values:
        return

    for patch in ax.patches:
        height = patch.get_height()
        if not np.isfinite(height) or height <= 0:
            continue
        ax.annotate(
            fmt.format(height),
            xy=(patch.get_x() + patch.get_width() / 2, height),
            xytext=(0, padding_points),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    headroom = max(values) * 0.12
    ax.set_ylim(ymin, max(ymax, max(values) + headroom))


def add_horizontal_bar_labels(ax, fmt: str = "{:.0f}", padding_points: int = 4) -> None:
    """Add numeric labels to the right of positive horizontal bars."""
    xmin, xmax = ax.get_xlim()
    values = [
        patch.get_width()
        for patch in ax.patches
        if np.isfinite(patch.get_width()) and patch.get_width() > 0
    ]
    if not values:
        return

    for patch in ax.patches:
        width = patch.get_width()
        if not np.isfinite(width) or width <= 0:
            continue
        ax.annotate(
            fmt.format(width),
            xy=(width, patch.get_y() + patch.get_height() / 2),
            xytext=(padding_points, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=9,
        )

    headroom = max(values) * 0.12
    ax.set_xlim(xmin, max(xmax, max(values) + headroom))


def add_stacked_bar_labels(ax, fmt: str = "{:.0f}", min_label_height: float = 0.0) -> None:
    """Add numeric labels inside stacked bar segments and totals above bars."""
    totals = defaultdict(float)
    for patch in ax.patches:
        height = patch.get_height()
        if not np.isfinite(height) or height <= 0:
            continue

        center_x = patch.get_x() + patch.get_width() / 2
        totals[round(center_x, 6)] += height
        if height <= min_label_height:
            continue

        center_y = patch.get_y() + height / 2
        ax.annotate(
            fmt.format(height),
            xy=(center_x, center_y),
            ha="center",
            va="center",
            fontsize=8,
            color="black",
        )

    if not totals:
        return

    ymin, ymax = ax.get_ylim()
    max_total = max(totals.values())
    for center_x, total in totals.items():
        ax.annotate(
            fmt.format(total),
            xy=(center_x, total),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    ax.set_ylim(ymin, max(ymax, max_total * 1.12))


add_bar_value_labels = add_vertical_bar_labels


VIETNAMESE_FIGURE_NAMES = {
    "event_label_counts": "Số lượng event normal/anomaly theo wind farm",
    "event_count_by_asset": "Số lượng event theo từng turbine",
    "feature_dimensions": "Số lượng đặc trưng và nhóm đặc trưng đặc biệt",
    "train_test_distribution": "Phân bố train/prediction theo wind farm",
    "status_distribution": "Phân bố trạng thái vận hành theo wind farm",
    "top_missing_features": "Các đặc trưng thiếu dữ liệu nhiều nhất",
    "event_duration": "Phân bố thời lượng event theo wind farm",
}

# %% [markdown]
# ## 3. Build Inventory and Data Quality Tables

# %%
farm_overview_rows = []
event_inventory_rows = []
all_event_info = []
all_feature_descriptions = []
quality_rows = []
train_test_rows = []
status_rows = []
label_rows = []
missing_accumulator = defaultdict(int)
observed_accumulator = defaultdict(int)

for farm_name in FARM_NAMES:
    farm_dir = RAW_ROOT / farm_name
    datasets_dir = farm_dir / "datasets"
    event_csvs = numeric_event_csvs(datasets_dir)

    feature_desc = read_feature_description(farm_dir)
    event_info = read_event_info(farm_name, farm_dir, event_csvs)
    event_info["event_id_int"] = event_info["event_id"].astype(int)

    csv_by_event = {int(path.stem): path for path in event_csvs}
    quick_rows = []

    print(f"\nProfiling {farm_name}: {len(event_csvs)} event CSVs")
    for csv_path in event_csvs:
        event_id = int(csv_path.stem)
        quick = quick_file_summary(farm_name, csv_path)
        quick_rows.append(quick)

        if FULL_EVENT_SCAN:
            quality, train_rows, status_count_rows, label_count_rows = profile_full_metadata(
                csv_path, farm_name, event_id
            )
            missing_by_feature = {}
            observed_by_feature = {}
        else:
            quality, train_rows, status_count_rows, label_count_rows, missing_by_feature, observed_by_feature = profile_sample(
                csv_path, farm_name, event_id
            )

        quality_rows.append(quality)
        train_test_rows.extend(train_rows)
        status_rows.extend(status_count_rows)
        label_rows.extend(label_count_rows)

        for feature, missing_count in missing_by_feature.items():
            key = (farm_name, feature)
            missing_accumulator[key] += missing_count
        for feature, observed_count in observed_by_feature.items():
            key = (farm_name, feature)
            observed_accumulator[key] += observed_count

    quick_df = pd.DataFrame(quick_rows)
    info_merged = event_info.merge(
        quick_df,
        left_on=["farm", "event_id_int"],
        right_on=["farm", "event_id"],
        how="left",
        suffixes=("", "_file"),
    )

    info_merged["csv_exists"] = info_merged["event_id_int"].isin(set(csv_by_event))
    event_inventory_rows.extend(info_merged.to_dict("records"))
    all_event_info.append(event_info)

    feature_desc_export = feature_desc.copy()
    feature_desc_export["farm"] = farm_name
    all_feature_descriptions.append(feature_desc_export)

    label_counts = event_info["event_label"].value_counts().to_dict()
    asset_count = event_info["asset_id"].nunique(dropna=True)
    column_min = int(quick_df["csv_columns"].min()) if not quick_df.empty else np.nan
    column_max = int(quick_df["csv_columns"].max()) if not quick_df.empty else np.nan
    feature_column_min = int(quick_df["csv_feature_columns"].min()) if not quick_df.empty else np.nan
    feature_column_max = int(quick_df["csv_feature_columns"].max()) if not quick_df.empty else np.nan

    farm_overview_rows.append(
        {
            "farm": farm_name,
            "event_info_rows": len(event_info),
            "event_csv_files": len(event_csvs),
            "asset_count": asset_count,
            "anomaly_events": int(label_counts.get("anomaly", 0)),
            "normal_events": int(label_counts.get("normal", 0)),
            "feature_description_rows": len(feature_desc),
            "angle_sensor_rows": int(feature_desc["is_angle_bool"].sum()),
            "counter_sensor_rows": int(feature_desc["is_counter_bool"].sum()),
            "csv_columns_min": column_min,
            "csv_columns_max": column_max,
            "csv_feature_columns_min": feature_column_min,
            "csv_feature_columns_max": feature_column_max,
            "has_any_label_column": bool(quick_df["has_label_column"].any()) if not quick_df.empty else False,
        }
    )

farm_overview = pd.DataFrame(farm_overview_rows)
event_inventory = pd.DataFrame(event_inventory_rows)
event_info_all = pd.concat(all_event_info, ignore_index=True)
feature_description_all = pd.concat(all_feature_descriptions, ignore_index=True)
quality_summary = pd.DataFrame(quality_rows)
train_test_distribution = pd.DataFrame(train_test_rows)
status_distribution = pd.DataFrame(status_rows)
label_distribution = pd.DataFrame(label_rows)

missing_rows = []
for (farm, feature), missing_count in missing_accumulator.items():
    observed = observed_accumulator[(farm, feature)]
    missing_rows.append(
        {
            "farm": farm,
            "feature": feature,
            "missing_count_sample": missing_count,
            "observed_rows_sample": observed,
            "missing_pct_sample": missing_count / max(observed, 1) * 100.0,
        }
    )
missing_by_feature = pd.DataFrame(
    missing_rows,
    columns=[
        "farm",
        "feature",
        "missing_count_sample",
        "observed_rows_sample",
        "missing_pct_sample",
    ],
)
if not missing_by_feature.empty:
    missing_by_feature = missing_by_feature.sort_values(
        ["farm", "missing_pct_sample", "missing_count_sample"],
        ascending=[True, False, False],
    )

print("\nFarm overview")
display(farm_overview)

# %% [markdown]
# ## 4. Save Tables

# %%
paths = {
    "farm_overview": write_csv(farm_overview, "farm_overview.csv"),
    "event_inventory": write_csv(event_inventory, "event_inventory.csv"),
    "event_info_all": write_csv(event_info_all, "event_info_all.csv"),
    "feature_description_all": write_csv(feature_description_all, "feature_description_all.csv"),
    "quality_summary": write_csv(quality_summary, "quality_summary.csv"),
    "train_test_distribution": write_csv(train_test_distribution, "train_test_distribution.csv"),
    "status_distribution": write_csv(status_distribution, "status_distribution.csv"),
    "label_distribution": write_csv(label_distribution, "label_distribution.csv"),
    "missing_by_feature_sample": write_csv(missing_by_feature, "missing_by_feature_sample.csv"),
}

for name, path in paths.items():
    print(f"{name}: {path}")

# %% [markdown]
# ## 5. Visualizations for the Capstone Report

# %%
plt.rcParams.update({
    "figure.figsize": (9, 5),
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.family": "DejaVu Sans",
    "font.size": 10,
})

figure_paths = {}

# %%
label_counts = (
    event_info_all.groupby(["farm", "event_label"])
    .size()
    .unstack(fill_value=0)
    .reindex(FARM_NAMES)
)
ax = label_counts.plot(kind="bar", color=["#c44e52", "#4c72b0"])
plt.title("Số lượng event normal/anomaly theo wind farm")
plt.xlabel("Wind farm")
plt.ylabel("Số lượng event")
plt.xticks(rotation=0)
add_vertical_bar_labels(ax)
figure_paths["event_label_counts"] = save_current_figure("event_label_counts_by_farm.png")

# %%
asset_counts = (
    event_info_all.assign(asset_id=lambda df: df["asset_id"].astype(str))
    .groupby(["farm", "asset_id"])
    .size()
    .reset_index(name="event_count")
)
asset_counts["farm_asset"] = asset_counts["farm"] + " / asset " + asset_counts["asset_id"]
asset_counts = asset_counts.sort_values(["farm", "asset_id"])

plt.figure(figsize=(10, max(5, len(asset_counts) * 0.22)))
ax = plt.gca()
ax.barh(asset_counts["farm_asset"], asset_counts["event_count"], color="#55a868")
plt.title("Số lượng event theo từng turbine")
plt.xlabel("Số lượng event")
plt.ylabel("")
add_horizontal_bar_labels(ax)
figure_paths["event_count_by_asset"] = save_current_figure("event_count_by_asset.png")

# %%
dimension_cols = [
    "csv_feature_columns_max",
    "feature_description_rows",
    "angle_sensor_rows",
    "counter_sensor_rows",
]
ax = farm_overview.set_index("farm")[dimension_cols].plot(kind="bar")
plt.title("Số lượng đặc trưng và nhóm đặc trưng đặc biệt")
plt.xlabel("Wind farm")
plt.ylabel("Số lượng")
plt.xticks(rotation=0)
add_vertical_bar_labels(ax)
figure_paths["feature_dimensions"] = save_current_figure("feature_dimensions_by_farm.png")

# %%
if not train_test_distribution.empty:
    train_plot = (
        train_test_distribution.groupby(["farm", "train_test"])["row_count"]
        .sum()
        .unstack(fill_value=0)
        .reindex(FARM_NAMES)
    )
    ax = train_plot.plot(kind="bar")
    source_note = "quét toàn bộ dữ liệu" if FULL_EVENT_SCAN else f"mẫu {SAMPLE_ROWS_PER_EVENT} dòng đầu mỗi event"
    plt.title(f"Phân bố train/prediction theo wind farm ({source_note})")
    plt.xlabel("Wind farm")
    plt.ylabel("Số dòng")
    plt.xticks(rotation=0)
    add_vertical_bar_labels(ax)
    figure_paths["train_test_distribution"] = save_current_figure("train_test_distribution.png")

# %%
if not status_distribution.empty:
    status_plot = (
        status_distribution.groupby(["farm", "status_type_id"])["row_count"]
        .sum()
        .reset_index()
    )
    status_plot["status_type_id"] = status_plot["status_type_id"].astype(str)
    pivot = status_plot.pivot(index="farm", columns="status_type_id", values="row_count").fillna(0)
    pivot = pivot.reindex(FARM_NAMES)
    ax = pivot.plot(kind="bar", stacked=True)
    source_note = "quét toàn bộ dữ liệu" if FULL_EVENT_SCAN else f"mẫu {SAMPLE_ROWS_PER_EVENT} dòng đầu mỗi event"
    plt.title(f"Phân bố trạng thái vận hành theo wind farm ({source_note})")
    plt.xlabel("Wind farm")
    plt.ylabel("Số dòng")
    plt.xticks(rotation=0)
    min_readable_segment = max(float(pivot.sum(axis=1).max()) * 0.025, 500.0)
    add_stacked_bar_labels(ax, min_label_height=min_readable_segment)
    figure_paths["status_distribution"] = save_current_figure("status_distribution.png")

# %%
if not missing_by_feature.empty:
    top_missing = (
        missing_by_feature.sort_values("missing_pct_sample", ascending=False)
        .head(25)
        .copy()
    )
    top_missing["farm_feature"] = top_missing["farm"] + " / " + top_missing["feature"]
    plt.figure(figsize=(10, max(5, len(top_missing) * 0.28)))
    ax = plt.gca()
    ax.barh(top_missing["farm_feature"], top_missing["missing_pct_sample"], color="#8172b2")
    ax.invert_yaxis()
    plt.title(f"Các đặc trưng thiếu dữ liệu nhiều nhất (mẫu {SAMPLE_ROWS_PER_EVENT} dòng mỗi event)")
    plt.xlabel("Tỷ lệ thiếu trong dữ liệu đã profile (%)")
    plt.ylabel("")
    add_horizontal_bar_labels(ax, fmt="{:.2f}")
    figure_paths["top_missing_features"] = save_current_figure("top_missing_features_sample.png")

# %%
duration_data = [
    group["event_duration_hours"].dropna().to_numpy()
    for _, group in event_info_all.groupby("farm", sort=False)
]
duration_labels = [
    farm
    for farm, group in event_info_all.groupby("farm", sort=False)
    if group["event_duration_hours"].notna().any()
]
duration_data = [arr for arr in duration_data if len(arr) > 0]
if duration_data:
    plt.figure(figsize=(9, 5))
    plt.boxplot(duration_data, tick_labels=duration_labels, showmeans=True)
    plt.title("Phân bố thời lượng event theo wind farm")
    plt.xlabel("Wind farm")
    plt.ylabel("Thời lượng (giờ)")
    figure_paths["event_duration"] = save_current_figure("event_duration_boxplot.png")

for name, path in figure_paths.items():
    print(f"{name}: {path}")

# %% [markdown]
# ## 6. Generate Vietnamese Markdown Summary

# %%
overview_for_report = farm_overview[
    [
        "farm",
        "event_info_rows",
        "event_csv_files",
        "asset_count",
        "anomaly_events",
        "normal_events",
        "csv_feature_columns_min",
        "csv_feature_columns_max",
        "angle_sensor_rows",
        "counter_sensor_rows",
    ]
].copy()

quality_for_report = quality_summary.groupby("farm").agg(
    profiled_events=("event_id", "count"),
    profiled_rows=("rows_profiled", "sum"),
    mean_missing_feature_cell_pct=("missing_feature_cell_pct", "mean"),
    mean_expected_10min_gap_pct=("expected_10min_gap_pct_sample", "mean"),
).reset_index()

mode_note = (
    "Cac thong ke theo dong duoc tinh bang full scan tren toan bo CSV."
    if FULL_EVENT_SCAN
    else f"Cac thong ke theo dong dang duoc tinh tren mau {SAMPLE_ROWS_PER_EVENT} dong dau cua moi event CSV."
)

summary_lines = [
    "# Phase 2 - Data Understanding cho CARE to Compare",
    "",
    f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    "",
    "## Muc tieu",
    "",
    "Phase 2 tap trung hieu du lieu truoc khi tien xu ly va huan luyen mo hinh. "
    "Ba wind farm A, B va C duoc doc theo cung mot quy trinh: kiem tra metadata, "
    "event labels, asset distribution, schema CSV, status distribution, train/prediction split, "
    "missing values va cac dac diem feature nhu angle/counter.",
    "",
    "## Tong quan dataset",
    "",
    markdown_table(overview_for_report),
    "",
    "## Chat luong du lieu da profile",
    "",
    mode_note,
    "",
    markdown_table(quality_for_report.round(3)),
    "",
    "## Nhan xet nhanh cho bao cao",
    "",
    "- Moi file trong `datasets/*.csv` nen duoc xem la mot event time series rieng, khong duoc tao window vuot qua ranh gioi event.",
    "- `event_info.csv` la nguon nhan cap event chinh; `status_type_id` la trang thai van hanh theo timestamp va can loai khoi input feature neu no lam ro nhan.",
    "- Ba wind farm co so chieu feature khac nhau rat lon, nen giai doan modeling nen danh gia theo tung farm truoc khi thu nghiem gop lien farm.",
    "- `feature_description.csv` giup xac dinh angle/counter va la co so de mo ta y nghia feature trong capstone report.",
    "- Khi sang Data Preparation, moi window can duoc group theo `farm + asset_id + event_id/sequence_id` de tranh leakage.",
    "",
    "## Artifacts",
    "",
    "Tables:",
]

for name, path in paths.items():
    summary_lines.append(f"- `{name}`: `{path.relative_to(PROJECT_ROOT)}`")

summary_lines.extend(["", "Figures:"])
for name, path in figure_paths.items():
    chart_name = VIETNAMESE_FIGURE_NAMES.get(name, name)
    summary_lines.append(f"- {chart_name}: `{path.relative_to(PROJECT_ROOT)}`")

summary_path = OUTPUT_DIR / "phase2_data_understanding_summary.md"
summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

print("Summary:", summary_path)

# %% [markdown]
# ## 7. Suggested Next Checks Before Data Preparation
#
# - Review `tables/event_inventory.csv` for missing CSVs, unexpected columns,
#   and event duration outliers.
# - Review `figures/status_distribution.png` to understand how many sampled
#   rows are normal status (`0`, `2`) versus abnormal/service/downtime states.
# - Review `tables/missing_by_feature_sample.csv` and decide which feature
#   columns need imputation, removal, or farm-specific handling.
# - Lock the grouping rule for the next phase:
#   `farm + asset_id + event_id` in raw form, or
#   `farm + asset_id + sequence_id` after combining files.
