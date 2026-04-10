from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Default configuration ────────────────────────────────────────────────────
DATASET_DIR = Path(__file__).parent
OUTPUT_DIR = DATASET_DIR / "screening_results"
CSV_SEP = ";"
TARGET_COL = "label"

DROP_COLUMNS = {
    "time_stamp", "asset_id", "train_test",
    "status_type_id", "sequence_id", "id",
}

# Point-biserial feature-vs-target selection thresholds
DEFAULT_MIN_MEAN_ABS_PB_R = 0.05
DEFAULT_MIN_PB_SIGN_CONSISTENCY = 0.60
DEFAULT_MIN_PB_SIG_RATIO = 0.30
DEFAULT_ALPHA = 0.05

# Feature-feature redundancy threshold
DEFAULT_FF_THRESHOLD = 0.85


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1 — Load & validate a single dataset
# ═══════════════════════════════════════════════════════════════════════════

def encode_binary_target(target: pd.Series, dataset_name: str = "") -> pd.Series:
    """
    Ensure the target is binary and encode it to {0, 1} while preserving NaN.
    """
    target = target.copy()
    non_na = target.dropna()
    unique_vals = pd.unique(non_na)

    if len(unique_vals) == 0:
        return pd.to_numeric(target, errors="coerce")

    if len(unique_vals) > 2:
        raise ValueError(
            f"[{dataset_name}] Target column must be binary; found values: {sorted(map(str, unique_vals))}"
        )

    unique_set = set(unique_vals.tolist())
    if unique_set <= {0, 1}:
        return pd.to_numeric(target, errors="coerce")

    sorted_vals = sorted(unique_vals.tolist())
    mapping = {sorted_vals[0]: 0, sorted_vals[1]: 1}
    encoded = target.map(mapping)
    log.info("[%s] Encoded target values %s -> {0,1}", dataset_name, mapping)
    return pd.to_numeric(encoded, errors="coerce")


def load_dataset(
    fpath: str | Path,
    sep: str = CSV_SEP,
    target_col: str = TARGET_COL,
    drop_columns: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Load a CSV dataset and return (features_df, target_series, skipped_cols).

    Features are restricted to numerical columns not in `drop_columns`.
    Columns with zero variance are removed.
    """
    if drop_columns is None:
        drop_columns = DROP_COLUMNS

    df = pd.read_csv(fpath, sep=sep, low_memory=False)
    fname = Path(fpath).name

    if target_col not in df.columns:
        raise ValueError(f"[{fname}] Column '{target_col}' not found.")

    target = encode_binary_target(df[target_col], dataset_name=fname)

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [
        c for c in num_cols
        if c not in drop_columns and c != target_col
    ]

    features = df[feature_cols].copy()

    skipped: list[str] = []
    constant_mask = features.nunique(dropna=True) <= 1
    const_cols = constant_mask[constant_mask].index.tolist()
    if const_cols:
        log.debug("[%s] Dropping %d constant feature(s): %s", fname, len(const_cols), const_cols)
        features.drop(columns=const_cols, inplace=True)
        skipped.extend(const_cols)

    return features, target, skipped


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2 — Per-dataset feature-vs-target metrics
# Main metric: Point-biserial
# Robustness check: Spearman
# ═══════════════════════════════════════════════════════════════════════════

def feature_vs_target_metrics(
    features: pd.DataFrame,
    target: pd.Series,
    alpha: float = DEFAULT_ALPHA,
) -> pd.DataFrame:
    """
    Compute per-feature association with a binary target.

    Returns columns:
        pb_r, pb_p_value, pb_significant,
        sp_rho, sp_p_value, sp_significant,
        n_pairs
    """
    records = []

    for col in features.columns:
        mask = features[col].notna() & target.notna()
        feat = features.loc[mask, col]
        tgt = target.loc[mask]

        if len(feat) < 5 or feat.nunique(dropna=True) < 2 or tgt.nunique(dropna=True) < 2:
            records.append({
                "feature": col,
                "pb_r": np.nan,
                "pb_p_value": np.nan,
                "pb_significant": False,
                "sp_rho": np.nan,
                "sp_p_value": np.nan,
                "sp_significant": False,
                "n_pairs": int(len(feat)),
            })
            continue

        try:
            pb_r, pb_pval = stats.pointbiserialr(tgt.astype(int).values, feat.astype(float).values)
        except Exception:
            pb_r, pb_pval = np.nan, np.nan

        try:
            sp_rho, sp_pval = stats.spearmanr(feat.astype(float).values, tgt.astype(int).values)
        except Exception:
            sp_rho, sp_pval = np.nan, np.nan

        records.append({
            "feature": col,
            "pb_r": float(pb_r) if not np.isnan(pb_r) else np.nan,
            "pb_p_value": float(pb_pval) if not np.isnan(pb_pval) else np.nan,
            "pb_significant": bool(pb_pval < alpha) if not np.isnan(pb_pval) else False,
            "sp_rho": float(sp_rho) if not np.isnan(sp_rho) else np.nan,
            "sp_p_value": float(sp_pval) if not np.isnan(sp_pval) else np.nan,
            "sp_significant": bool(sp_pval < alpha) if not np.isnan(sp_pval) else False,
            "n_pairs": int(len(feat)),
        })

    return pd.DataFrame(records).set_index("feature")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3 — Aggregate across datasets
# ═══════════════════════════════════════════════════════════════════════════

def _sign_consistency(values: np.ndarray) -> float:
    values = values[~np.isnan(values)]
    signs = np.sign(values[values != 0])
    if len(signs) == 0:
        return 0.0
    dominant = max(np.sum(signs > 0), np.sum(signs < 0))
    return float(dominant / len(signs))


def aggregate_feature_target_results(
    per_dataset_results: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Aggregate point-biserial and Spearman summaries across datasets.
    Selection should be based on point-biserial columns.
    """
    all_features = sorted({feat for df in per_dataset_results.values() for feat in df.index})
    rows = []

    for feat in all_features:
        pb_vals, pb_sigs = [], []
        sp_vals, sp_sigs = [], []

        for ds_df in per_dataset_results.values():
            if feat not in ds_df.index:
                continue

            pb = ds_df.loc[feat, "pb_r"]
            sp = ds_df.loc[feat, "sp_rho"]
            pb_sig = ds_df.loc[feat, "pb_significant"]
            sp_sig = ds_df.loc[feat, "sp_significant"]

            if not np.isnan(pb):
                pb_vals.append(pb)
                pb_sigs.append(int(pb_sig))
            if not np.isnan(sp):
                sp_vals.append(sp)
                sp_sigs.append(int(sp_sig))

        if not pb_vals:
            continue

        pb_arr = np.array(pb_vals, dtype=float)
        sp_arr = np.array(sp_vals, dtype=float) if sp_vals else np.array([], dtype=float)

        row = {
            "feature": feat,
            "mean_pb_r": float(np.mean(pb_arr)),
            "median_pb_r": float(np.median(pb_arr)),
            "std_pb_r": float(np.std(pb_arr, ddof=1)) if len(pb_arr) > 1 else 0.0,
            "min_pb_r": float(np.min(pb_arr)),
            "max_pb_r": float(np.max(pb_arr)),
            "mean_abs_pb_r": float(np.mean(np.abs(pb_arr))),
            "median_abs_pb_r": float(np.median(np.abs(pb_arr))),
            "pb_sign_consistency": _sign_consistency(pb_arr),
            "pb_significant_count": int(np.sum(pb_sigs)),
            "pb_significant_ratio": float(np.mean(pb_sigs)) if pb_sigs else 0.0,
            "dataset_count": int(len(pb_arr)),
        }

        if len(sp_arr) > 0:
            row.update({
                "mean_sp_rho": float(np.mean(sp_arr)),
                "median_sp_rho": float(np.median(sp_arr)),
                "std_sp_rho": float(np.std(sp_arr, ddof=1)) if len(sp_arr) > 1 else 0.0,
                "min_sp_rho": float(np.min(sp_arr)),
                "max_sp_rho": float(np.max(sp_arr)),
                "mean_abs_sp_rho": float(np.mean(np.abs(sp_arr))),
                "median_abs_sp_rho": float(np.median(np.abs(sp_arr))),
                "sp_sign_consistency": _sign_consistency(sp_arr),
                "sp_significant_count": int(np.sum(sp_sigs)),
                "sp_significant_ratio": float(np.mean(sp_sigs)) if sp_sigs else 0.0,
            })
        else:
            row.update({
                "mean_sp_rho": np.nan,
                "median_sp_rho": np.nan,
                "std_sp_rho": np.nan,
                "min_sp_rho": np.nan,
                "max_sp_rho": np.nan,
                "mean_abs_sp_rho": np.nan,
                "median_abs_sp_rho": np.nan,
                "sp_sign_consistency": np.nan,
                "sp_significant_count": 0,
                "sp_significant_ratio": np.nan,
            })

        rows.append(row)

    agg = pd.DataFrame(rows).set_index("feature")
    return agg.sort_values("mean_abs_pb_r", ascending=False)


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4 — Select robust features by threshold
# ═══════════════════════════════════════════════════════════════════════════

def select_features(
    agg: pd.DataFrame,
    min_mean_abs_pb_r: float = DEFAULT_MIN_MEAN_ABS_PB_R,
    min_pb_sign_consistency: float = DEFAULT_MIN_PB_SIGN_CONSISTENCY,
    min_pb_sig_ratio: float = DEFAULT_MIN_PB_SIG_RATIO,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply thresholds on aggregated point-biserial metrics.
    """
    mask = (
        (agg["mean_abs_pb_r"] >= min_mean_abs_pb_r) &
        (agg["pb_sign_consistency"] >= min_pb_sign_consistency) &
        (agg["pb_significant_ratio"] >= min_pb_sig_ratio)
    )
    return agg[mask].copy(), agg[~mask].copy()


# ═══════════════════════════════════════════════════════════════════════════
# STEP 5 — Feature-feature Spearman + redundancy removal
# ═══════════════════════════════════════════════════════════════════════════

def build_feature_feature_corr(
    csv_files,
    feature_cols,
    target_col: str = TARGET_COL,
    sep: str = CSV_SEP,
    sample_per_file: int = 10_000,
    random_state: int = 42,
):
    dfs = []
    rng = np.random.default_rng(random_state)

    usecols = list(feature_cols) + [target_col]

    for fpath in csv_files:
        try:
            df = pd.read_csv(fpath, sep=sep, low_memory=False, usecols=usecols)
            df = df.dropna(how="all")
            df[target_col] = encode_binary_target(df[target_col], dataset_name=Path(fpath).name)
            df = df[df[target_col].isin([0, 1])]

            if df.empty or df[target_col].nunique(dropna=True) < 2:
                log.warning(
                    "Skipping %s for FF-corr: target column '%s' is not usable after encoding.",
                    Path(fpath).name,
                    target_col,
                )
                continue

            normal_df = df[df[target_col] == 0]
            anomaly_df = df[df[target_col] == 1]

            # keep all anomalies
            # sample only normals
            if len(normal_df) > sample_per_file:
                idx = rng.choice(len(normal_df), size=sample_per_file, replace=False)
                normal_df = normal_df.iloc[idx]

            sampled_df = pd.concat([normal_df, anomaly_df], ignore_index=True)
            sampled_df = sampled_df.drop(columns=[target_col])

            if sampled_df.empty:
                log.warning("Skipping %s for FF-corr: no rows remain after sampling.", Path(fpath).name)
                continue

            dfs.append(sampled_df)

        except Exception as exc:
            log.warning("Could not read %s for FF-corr: %s", Path(fpath).name, exc)

    if not dfs:
        return pd.DataFrame()

    pooled = pd.concat(dfs, ignore_index=True)[feature_cols]
    if pooled.empty:
        return pd.DataFrame()

    if pooled.shape[1] == 1:
        return pd.DataFrame([[1.0]], index=feature_cols, columns=feature_cols)

    corr_matrix, _ = stats.spearmanr(pooled.values, nan_policy="omit")

    return pd.DataFrame(corr_matrix, index=feature_cols, columns=feature_cols)


def remove_redundant_features(
    ff_corr: pd.DataFrame,
    agg_scores: pd.Series,
    threshold: float = DEFAULT_FF_THRESHOLD,
) -> tuple[list[str], list[str]]:
    """
    Greedy removal of redundant features.
    When two features have |rho| >= threshold, drop the one with the lower score.
    """
    features = list(ff_corr.columns)
    to_remove = set()

    for i, fi in enumerate(features):
        if fi in to_remove:
            continue
        for fj in features[i + 1:]:
            if fj in to_remove:
                continue
            if abs(ff_corr.loc[fi, fj]) >= threshold:
                score_i = agg_scores.get(fi, 0.0)
                score_j = agg_scores.get(fj, 0.0)
                loser = fj if score_i >= score_j else fi
                to_remove.add(loser)

    kept = [f for f in features if f not in to_remove]
    removed = [f for f in features if f in to_remove]
    return kept, removed


# ═══════════════════════════════════════════════════════════════════════════
# STEP 6 — Visualisations
# ═══════════════════════════════════════════════════════════════════════════

def plot_feature_importance_bar(
    agg: pd.DataFrame,
    selected_feats: list[str],
    out_path: str | Path,
    top_n: int = 40,
) -> None:
    """Horizontal bar chart of mean_abs_pb_r for aggregated features."""
    plot_df = agg.head(top_n)[["mean_abs_pb_r", "std_pb_r"]].copy().iloc[::-1]

    colors = ["#4CAF50" if f in selected_feats else "#F44336" for f in plot_df.index]

    fig, ax = plt.subplots(figsize=(12, max(6, len(plot_df) * 0.30)))
    ax.barh(
        plot_df.index,
        plot_df["mean_abs_pb_r"],
        xerr=plot_df["std_pb_r"],
        color=colors,
        edgecolor="white",
        linewidth=0.4,
        capsize=3,
        error_kw={"elinewidth": 0.8, "alpha": 0.7},
    )

    ax.set_xlabel("Mean |point-biserial r| vs target", fontsize=11)
    ax.set_title(
        f"Feature-Target Association (top {top_n})\n"
        "Point-biserial main metric | green = selected | red = rejected",
        fontsize=12,
        fontweight="bold",
    )
    ax.tick_params(axis="y", labelsize=7)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.axvline(0, color="black", linewidth=0.6)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved bar plot -> %s", out_path)


def plot_sign_consistency(
    agg: pd.DataFrame,
    out_path: str | Path,
    top_n: int = 40,
) -> None:
    """Bar chart of point-biserial sign consistency colored by mean direction."""
    plot_df = agg.head(top_n)[["pb_sign_consistency", "mean_pb_r"]].copy().iloc[::-1]
    colors = ["#1976D2" if r >= 0 else "#E53935" for r in plot_df["mean_pb_r"]]

    fig, ax = plt.subplots(figsize=(12, max(6, len(plot_df) * 0.30)))
    ax.barh(
        plot_df.index,
        plot_df["pb_sign_consistency"],
        color=colors,
        edgecolor="white",
        linewidth=0.4,
    )
    ax.axvline(0.5, color="grey", linewidth=0.8, linestyle="--", label="50 %")
    ax.set_xlabel("Point-biserial sign consistency", fontsize=11)
    ax.set_title(
        f"Sign Consistency (top {top_n})\nblue = positive | red = negative",
        fontsize=12,
        fontweight="bold",
    )
    ax.tick_params(axis="y", labelsize=7)
    ax.set_xlim(0, 1.05)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved sign-consistency plot -> %s", out_path)


def plot_ff_heatmap(
    ff_corr: pd.DataFrame,
    out_path: str | Path,
    max_features: int = 60,
) -> None:
    """Heatmap of the feature-feature Spearman correlation matrix."""
    if ff_corr.empty:
        log.warning("FF correlation matrix is empty; skipping heatmap.")
        return

    feats = ff_corr.columns.tolist()
    if len(feats) > max_features:
        feats = feats[:max_features]
        ff_corr = ff_corr.loc[feats, feats]

    n = len(feats)
    fig_size = max(8, n * 0.30)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    img = ax.imshow(ff_corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(img, ax=ax, shrink=0.8, label="Spearman rho")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(feats, rotation=90, fontsize=max(4, 8 - n // 15))
    ax.set_yticklabels(feats, fontsize=max(4, 8 - n // 15))
    ax.set_title("Feature-Feature Spearman Correlation (selected)", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved heatmap -> %s", out_path)


def plot_per_dataset_metric_box(
    per_dataset_results: dict[str, pd.DataFrame],
    selected_feats: list[str],
    out_path: str | Path,
    metric_col: str = "pb_r",
    x_label: str = "Point-biserial r vs target",
    title: str = "Cross-dataset distribution (selected features)",
) -> None:
    """Boxplot showing metric distribution for each selected feature across datasets."""
    data_for_plot: dict[str, list[float]] = {}
    for feat in selected_feats:
        vals = []
        for ds_df in per_dataset_results.values():
            if feat in ds_df.index:
                v = ds_df.loc[feat, metric_col]
                if not np.isnan(v):
                    vals.append(v)
        if vals:
            data_for_plot[feat] = vals

    if not data_for_plot:
        return

    sorted_feats = sorted(data_for_plot, key=lambda f: np.median(data_for_plot[f]))
    data_arrays = [data_for_plot[f] for f in sorted_feats]

    fig, ax = plt.subplots(figsize=(14, max(5, len(sorted_feats) * 0.35)))
    bp = ax.boxplot(
        data_arrays,
        vert=False,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.5},
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("#42A5F5")
        patch.set_alpha(0.7)

    ax.set_yticks(range(1, len(sorted_feats) + 1))
    ax.set_yticklabels(sorted_feats, fontsize=7)
    ax.axvline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved box plot -> %s", out_path)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════

def run_pipeline(
    dataset_dir: str | Path = DATASET_DIR,
    output_dir: str | Path = OUTPUT_DIR,
    csv_sep: str = CSV_SEP,
    target_col: str = TARGET_COL,
    drop_columns: set[str] | None = None,
    alpha: float = DEFAULT_ALPHA,
    min_mean_abs_pb_r: float = DEFAULT_MIN_MEAN_ABS_PB_R,
    min_pb_sign_consistency: float = DEFAULT_MIN_PB_SIGN_CONSISTENCY,
    min_pb_sig_ratio: float = DEFAULT_MIN_PB_SIG_RATIO,
    ff_threshold: float = DEFAULT_FF_THRESHOLD,
    sample_per_file: int = 10_000,
) -> dict:
    """
    End-to-end feature screening pipeline.

    Main feature-vs-target metric: point-biserial.
    Robustness check: Spearman.
    Feature-feature redundancy metric: Spearman.
    """
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if drop_columns is None:
        drop_columns = DROP_COLUMNS

    csv_files = sorted(dataset_dir.glob("*.csv"))
    if not csv_files:
        log.error("No CSV files found in: %s", dataset_dir)
        sys.exit(1)

    log.info("═" * 60)
    log.info("Found %d CSV file(s) in %s", len(csv_files), dataset_dir)
    log.info("Main metric: point-biserial | Robustness: Spearman | FF redundancy: Spearman")
    log.info("═" * 60)

    per_dataset_results: dict[str, pd.DataFrame] = {}
    skipped_datasets: dict[str, str] = {}
    all_feature_sets: list[set[str]] = []

    for fpath in csv_files:
        fname = fpath.name
        try:
            features, target, const_cols = load_dataset(
                fpath,
                sep=csv_sep,
                target_col=target_col,
                drop_columns=drop_columns,
            )
        except Exception as exc:
            log.warning("Skipping %s - load error: %s", fname, exc)
            skipped_datasets[fname] = f"load_error: {exc}"
            continue

        if target.dropna().nunique() < 2:
            log.warning("SKIP (constant target) -> %s | unique values: %s", fname, target.dropna().unique().tolist())
            skipped_datasets[fname] = "constant_target"
            continue

        log.info(
            "Processing %-18s | rows=%d | features=%d | anomaly_ratio=%.3f | dropped_constant_features=%d",
            fname,
            len(target),
            features.shape[1],
            float(target.mean(skipna=True)),
            len(const_cols),
        )

        ds_result = feature_vs_target_metrics(features, target, alpha=alpha)
        per_dataset_results[fname] = ds_result
        all_feature_sets.append(set(features.columns))

    if not per_dataset_results:
        log.error("No valid datasets after filtering. Exiting.")
        sys.exit(1)

    log.info("Valid datasets: %d / %d", len(per_dataset_results), len(csv_files))
    if skipped_datasets:
        log.info("Skipped datasets: %s", list(skipped_datasets.keys()))

    all_features_union = set.union(*all_feature_sets)
    log.info("Feature union across datasets: %d features", len(all_features_union))

    log.info("Aggregating feature-target results ...")
    agg = aggregate_feature_target_results(per_dataset_results)

    per_ds_rows = []
    for fname, ds_df in per_dataset_results.items():
        tmp = ds_df.reset_index()
        tmp.insert(0, "dataset", fname)
        per_ds_rows.append(tmp)
    pd.concat(per_ds_rows, ignore_index=True).to_csv(
        output_dir / "per_dataset_feature_target_metrics.csv", index=False
    )
    agg.to_csv(output_dir / "aggregated_feature_target_metrics.csv")
    log.info("Saved aggregated_feature_target_metrics.csv (%d features)", len(agg))

    selected_agg, rejected_agg = select_features(
        agg,
        min_mean_abs_pb_r=min_mean_abs_pb_r,
        min_pb_sign_consistency=min_pb_sign_consistency,
        min_pb_sig_ratio=min_pb_sig_ratio,
    )
    log.info("Feature selection: %d selected / %d rejected", len(selected_agg), len(rejected_agg))
    selected_agg.to_csv(output_dir / "selected_features.csv")
    rejected_agg.to_csv(output_dir / "rejected_features.csv")

    selected_feats = selected_agg.index.tolist()

    final_features: list[str] = selected_feats.copy()
    redundant_removed: list[str] = []
    ff_corr = pd.DataFrame()

    if len(selected_feats) >= 2:
        log.info("Building feature-feature correlation (%d features) ...", len(selected_feats))

        available = set(selected_feats)
        for fpath in csv_files:
            try:
                header = pd.read_csv(fpath, sep=csv_sep, nrows=0).columns.tolist()
                available &= set(header)
            except Exception:
                pass
        common_selected = [f for f in selected_feats if f in available]

        if len(common_selected) >= 2:
            ff_corr = build_feature_feature_corr(
                csv_files,
                common_selected,
                target_col=target_col,
                sep=csv_sep,
                sample_per_file=sample_per_file,
            )
            final_features, redundant_removed = remove_redundant_features(
                ff_corr,
                agg_scores=selected_agg["mean_abs_pb_r"],
                threshold=ff_threshold,
            )
            log.info(
                "Redundancy removal: kept %d, dropped %d (|rho| >= %.2f)",
                len(final_features), len(redundant_removed), ff_threshold,
            )
            ff_corr.to_csv(output_dir / "feature_feature_corr.csv")
            pd.DataFrame({"removed_feature": redundant_removed}).to_csv(
                output_dir / "redundant_removed.csv", index=False
            )

    final_df = agg.loc[[f for f in final_features if f in agg.index]].copy()
    final_df.to_csv(output_dir / "final_selected_features.csv")
    log.info("Final features: %d", len(final_df))

    if skipped_datasets:
        pd.DataFrame(
            [(k, v) for k, v in skipped_datasets.items()],
            columns=["dataset", "reason"],
        ).to_csv(output_dir / "skipped_datasets.csv", index=False)

    log.info("Generating plots ...")
    plot_feature_importance_bar(
        agg,
        selected_feats,
        out_path=output_dir / "bar_mean_abs_pb_r.png",
        top_n=min(50, len(agg)),
    )
    plot_sign_consistency(
        agg,
        out_path=output_dir / "bar_pb_sign_consistency.png",
        top_n=min(50, len(agg)),
    )
    plot_per_dataset_metric_box(
        per_dataset_results,
        selected_feats=final_features,
        out_path=output_dir / "boxplot_cross_dataset_pb_r.png",
        metric_col="pb_r",
        x_label="Point-biserial r vs target",
        title="Cross-dataset point-biserial distribution (selected features)",
    )
    plot_per_dataset_metric_box(
        per_dataset_results,
        selected_feats=final_features,
        out_path=output_dir / "boxplot_cross_dataset_spearman.png",
        metric_col="sp_rho",
        x_label="Spearman rho vs target",
        title="Cross-dataset Spearman distribution (selected features)",
    )

    if not ff_corr.empty and len(ff_corr) >= 2:
        final_in_corr = [f for f in final_features if f in ff_corr.index]
        if len(final_in_corr) >= 2:
            plot_ff_heatmap(
                ff_corr.loc[final_in_corr, final_in_corr],
                out_path=output_dir / "heatmap_ff_corr.png",
            )

    log.info("═" * 60)
    log.info("PIPELINE COMPLETE")
    log.info("  Datasets processed  : %d", len(per_dataset_results))
    log.info("  Datasets skipped    : %d", len(skipped_datasets))
    log.info("  Features in union   : %d", len(all_features_union))
    log.info("  Features selected   : %d", len(selected_feats))
    log.info("  Redundant removed   : %d", len(redundant_removed))
    log.info("  Final features      : %d", len(final_features))
    log.info("  Outputs saved to    : %s", output_dir)
    log.info("═" * 60)

    log.info("Final feature list:")
    for i, f in enumerate(sorted(final_features), 1):
        score = final_df.loc[f, "mean_abs_pb_r"] if f in final_df.index else float("nan")
        log.info("  %3d. %-40s  mean|pb_r|=%.4f", i, f, score)

    return {
        "agg": agg,
        "selected": selected_agg,
        "rejected": rejected_agg,
        "final_features": final_features,
        "redundant_removed": redundant_removed,
        "per_dataset_results": per_dataset_results,
        "ff_corr": ff_corr,
        "skipped_datasets": skipped_datasets,
    }


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-dataset SCADA feature screening pipeline (point-biserial main metric)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset-dir", default=str(DATASET_DIR), help="Folder with CSV files")
    p.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Output folder")
    p.add_argument("--sep", default=CSV_SEP, help="CSV delimiter")
    p.add_argument("--target", default=TARGET_COL, help="Target column name")
    p.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="p-value significance level")
    p.add_argument("--min-abs-pb-r", type=float, default=DEFAULT_MIN_MEAN_ABS_PB_R, help="Min mean |point-biserial r|")
    p.add_argument("--min-pb-sign-cons", type=float, default=DEFAULT_MIN_PB_SIGN_CONSISTENCY, help="Min point-biserial sign consistency")
    p.add_argument("--min-pb-sig-ratio", type=float, default=DEFAULT_MIN_PB_SIG_RATIO, help="Min significant-dataset ratio for point-biserial")
    p.add_argument("--ff-threshold", type=float, default=DEFAULT_FF_THRESHOLD, help="Feature-feature redundancy threshold")
    p.add_argument("--sample-per-file", type=int, default=10_000, help="Rows sampled per file for FF corr")
    p.add_argument("--verbose", "-v", action="store_true", help="Debug logging")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    run_pipeline(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        csv_sep=args.sep,
        target_col=args.target,
        alpha=args.alpha,
        min_mean_abs_pb_r=args.min_abs_pb_r,
        min_pb_sign_consistency=args.min_pb_sign_cons,
        min_pb_sig_ratio=args.min_pb_sig_ratio,
        ff_threshold=args.ff_threshold,
        sample_per_file=args.sample_per_file,
    )
