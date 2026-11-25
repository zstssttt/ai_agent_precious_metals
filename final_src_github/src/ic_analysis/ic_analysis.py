"""
Factor IC analysis and heatmap visualization.

Standalone script that reads a pre-computed factor wide table (factor_values_raw.csv)
and Au9995 closing prices, then:
  1. Computes Y variables (forward 1-10 day log-returns on Au9995)
  2. Computes rolling Pearson and Spearman IC / ICIR for each factor-Y pair
  3. Draws combined heatmaps with significance stars

Designed to be independent of the factor computation pipeline — any script that
produces a factor_values_raw.csv can feed into this analysis.
"""

import logging
import os
import warnings
from typing import Optional

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
matplotlib.rcParams["font.family"] = ["Arial Unicode MS"]
matplotlib.rcParams["axes.unicode_minus"] = False

logger = logging.getLogger(__name__)

# --- Constants ---
TRADING_DAYS_PER_YEAR = 252
IC_MIN_PERIODS = 180
ICIR_SIGNIFICANT = 0.5
ICIR_HIGHLY_SIGNIFICANT = 1.0
EPSILON = 1e-8
HEATMAP_FONT_SIZE = 8.5
DEFAULT_COLOR_RANGE = 0.3
Y_RANGE = range(1, 11)
Y_COLS = [f"Y_{n}" for n in Y_RANGE]


# ======================================================
# User configuration (modify only this block per run)
# ======================================================

# Pre-computed factor wide table: first column = date (index), remaining = factor values
FACTOR_CSV_PATH = os.environ.get(
    "FACTOR_CSV_PATH",
    "results/factor_values_top15.csv",
)

# Au9995 closing price CSV (same file used in the factor computation script)
AU9995_PATH = os.environ.get(
    "AU9995_PATH",
    "data/aligned/daily/Au9995_close.csv",
)

# Output directory for IC/ICIR CSVs and heatmap PNGs
OUTPUT_DIR = os.environ.get("IC_OUTPUT_DIR", "results")

# Rolling IC window (trading days)
ROLLING_IC_WINDOW = 180

# Cutoff date: rows beyond this date are discarded
CUTOFF_DATE = "2023-12-31"

# Time-period split: True = output full / pre-split / post-split results
USE_SPLIT = False
SPLIT_DATE = "2022-01-01"


# ======================================================
# Utility functions
# ======================================================

def _read_csv_any_encoding(path: str, **kwargs) -> pd.DataFrame:
    """Try utf-8 / gbk / gb2312 / latin1 encodings in sequence to read a CSV."""
    for enc in ("utf-8", "gbk", "gb2312", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except (UnicodeDecodeError, LookupError):
            continue
    raise ValueError(f"Cannot detect encoding: {path}")


def auto_read_csv(path: str, rename_price_to: str | None = None) -> pd.DataFrame:
    """Read a price CSV, auto-detect the date column as index, keep only the price column."""
    df = _read_csv_any_encoding(path)
    df.columns = df.columns.str.strip()

    date_col = next((c for c in df.columns if c.lower() == "date"), None)
    if date_col is None:
        raise ValueError(f"No date column found (expected 'date'): {path}")

    price_col = next((c for c in df.columns if c.lower() == "price"), None)
    if price_col is None:
        raise ValueError(f"No price column found (expected 'price'): {path}")

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    df = df[[price_col]]
    if rename_price_to:
        df = df.rename(columns={price_col: rename_price_to})
    return df


def _spearman_rolling_corr(
    factor_window: pd.Series,
    y_series: pd.Series,
) -> float:
    """Compute Spearman rank correlation within a rolling window.

    Factored out to module level so it is defined once (not redefined per loop
    iteration). The y_series is passed explicitly rather than captured via closure.
    """
    idx = factor_window.index
    yy = y_series.reindex(idx)
    mask = factor_window.notna() & yy.notna()
    if mask.sum() < 2:
        return np.nan
    r, _ = spearmanr(factor_window[mask], yy[mask])
    return r


# ======================================================
# IC analysis
# ======================================================

def build_annot_matrix(
    ic_df: pd.DataFrame, icir_df: pd.DataFrame
) -> pd.DataFrame:
    """Build annotation text matrix for heatmaps: IC value + significance stars."""
    annot = pd.DataFrame(index=ic_df.index, columns=ic_df.columns, dtype=object)
    for fc in ic_df.index:
        for yc in ic_df.columns:
            ic_val = ic_df.loc[fc, yc]
            icir_val = icir_df.loc[fc, yc]
            if pd.isna(ic_val):
                annot.loc[fc, yc] = ""
                continue
            text = f"{ic_val:.3f}"
            if not pd.isna(icir_val):
                if abs(icir_val) > ICIR_HIGHLY_SIGNIFICANT:
                    text += "**"
                elif abs(icir_val) > ICIR_SIGNIFICANT:
                    text += "*"
            annot.loc[fc, yc] = text
    return annot


def run_ic_analysis(
    sub_df: pd.DataFrame,
    factor_cols: list[str],
    label: str,
    out_dir: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute rolling IC / ICIR for all factor-Y pairs and save 4 CSVs.

    Args:
        sub_df: DataFrame with factor columns and Y_1..Y_10.
        factor_cols: List of factor column names.
        label: Period label ('full', 'pre2022', etc.). Empty string = no suffix.
        out_dir: Directory to write CSV output.

    Returns:
        (ic_pearson, icir_pearson, ic_spearman, icir_spearman) DataFrames.
    """
    suffix = f"_{label}" if label else ""
    total_pairs = len(factor_cols) * len(Y_COLS)
    done = 0

    results = {
        key: {fc: {} for fc in factor_cols}
        for key in ("ic_pearson", "icir_pearson", "ic_spearman", "icir_spearman")
    }

    for fc in factor_cols:
        fac_series = sub_df[fc] if fc in sub_df.columns else pd.Series(dtype=float)

        for yc in Y_COLS:
            y_series = sub_df[yc] if yc in sub_df.columns else pd.Series(dtype=float)

            valid_count = (fac_series.notna() & y_series.notna()).sum()
            if valid_count < ROLLING_IC_WINDOW:
                for key in results:
                    results[key][fc][yc] = np.nan
                done += 1
                continue

            # -- Pearson rolling IC --
            ic_p = fac_series.rolling(
                ROLLING_IC_WINDOW, min_periods=IC_MIN_PERIODS
            ).corr(y_series)
            mean_ic_p = ic_p.mean()
            std_ic_p = ic_p.std()
            results["ic_pearson"][fc][yc] = mean_ic_p
            results["icir_pearson"][fc][yc] = (
                mean_ic_p / std_ic_p if std_ic_p >= EPSILON else np.nan
            )

            # -- Spearman rolling IC --
            ic_s = fac_series.rolling(
                ROLLING_IC_WINDOW, min_periods=IC_MIN_PERIODS
            ).apply(
                lambda x, _y=y_series: _spearman_rolling_corr(x, _y),
                raw=False,
            )
            mean_ic_s = ic_s.mean()
            std_ic_s = ic_s.std()
            results["ic_spearman"][fc][yc] = mean_ic_s
            results["icir_spearman"][fc][yc] = (
                mean_ic_s / std_ic_s if std_ic_s >= EPSILON else np.nan
            )

            done += 1

        print(f"  [{label or 'full'}] [{done}/{total_pairs}] Factor done: {fc}")

    out_dfs = {}
    file_names = {
        "ic_pearson": "IC_pearson_matrix",
        "icir_pearson": "ICIR_pearson_matrix",
        "ic_spearman": "IC_spearman_matrix",
        "icir_spearman": "ICIR_spearman_matrix",
    }
    for key, fname in file_names.items():
        df = pd.DataFrame(results[key]).T.reindex(index=factor_cols, columns=Y_COLS)
        df.to_csv(os.path.join(out_dir, f"{fname}{suffix}.csv"))
        out_dfs[key] = df

    print(f"\n[{label or 'full'}] CSVs saved to: {out_dir}")

    for name, df in out_dfs.items():
        print(f"\n{name}{suffix} (first 5 rows):")
        print(df.head())

    return (
        out_dfs["ic_pearson"],
        out_dfs["icir_pearson"],
        out_dfs["ic_spearman"],
        out_dfs["icir_spearman"],
    )


# ======================================================
# Heatmap visualization
# ======================================================

def draw_and_save_heatmap(
    ic_p_df: pd.DataFrame,
    icir_p_df: pd.DataFrame,
    ic_s_df: pd.DataFrame,
    icir_s_df: pd.DataFrame,
    factor_cols: list[str],
    label: str,
    out_dir: str,
) -> None:
    """Draw Pearson + Spearman dual-panel IC heatmap and save as PNG."""
    suffix = f"_{label}" if label else ""

    annot_p = build_annot_matrix(ic_p_df, icir_p_df)
    annot_s = build_annot_matrix(ic_s_df, icir_s_df)

    max_abs = max(ic_p_df.abs().max().max(), ic_s_df.abs().max().max())
    if np.isnan(max_abs) or max_abs == 0:
        max_abs = DEFAULT_COLOR_RANGE
    vmax = max_abs * 1.1
    vmin = -vmax

    n_factors = len(factor_cols)
    subplot_height = max(4, n_factors * 0.6)
    fig_height = subplot_height * 2 + 2

    fig, axes = plt.subplots(2, 1, figsize=(14, fig_height),
                              gridspec_kw={"hspace": 0.5})
    cmap = matplotlib.colormaps["RdBu_r"]
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    def _draw(ax, ic_df, annot_df, title, colorbar_label):
        data = ic_df.values.astype(float)
        im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                text = annot_df.iloc[i, j]
                if text == "" or np.isnan(data[i, j]):
                    continue
                rgba = cmap(norm(data[i, j]))
                lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                ax.text(j, i, text, ha="center", va="center",
                        fontsize=HEATMAP_FONT_SIZE,
                        color="white" if lum < 0.5 else "black")

        ax.set_xticks(range(len(Y_COLS)))
        ax.set_xticklabels(Y_COLS, fontsize=10)
        ax.set_yticks(range(len(factor_cols)))
        ax.set_yticklabels(factor_cols, fontsize=10)
        ax.set_xlabel("Y (Forward N-day Log Return)", fontsize=10)
        ax.set_ylabel("Factor Name", fontsize=10)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
        ax.text(0.5, 1.01, f"* |IR|>{ICIR_SIGNIFICANT}, ** |IR|>{ICIR_HIGHLY_SIGNIFICANT}",
                transform=ax.transAxes, fontsize=9, ha="center", va="bottom")
        cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
        cbar.set_label(colorbar_label, fontsize=9)

    period_note = f" ({label})" if label else ""
    _draw(
        axes[0], ic_p_df, annot_p,
        f"Gold Factor{period_note} — Pearson IC ({ROLLING_IC_WINDOW}-day rolling)",
        "Pearson IC",
    )
    _draw(
        axes[1], ic_s_df, annot_s,
        f"Gold Factor{period_note} — Spearman Rank IC ({ROLLING_IC_WINDOW}-day rolling)",
        "Spearman Rank IC",
    )

    save_path = os.path.join(out_dir, f"IC_heatmap_combined{suffix}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Heatmap saved: {save_path}")


# ======================================================
# Main entry point
# ======================================================

def main() -> None:
    # -- Step 1: Load factor table and compute Y variables --
    print("=" * 60)
    print("Step 1: Load data and compute Y variables")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    factor_df = _read_csv_any_encoding(
        FACTOR_CSV_PATH, index_col=0, parse_dates=True
    )
    factor_df.index = pd.to_datetime(factor_df.index)
    factor_df = factor_df.sort_index()
    factor_df = factor_df[factor_df.index <= CUTOFF_DATE]
    factor_cols = list(factor_df.columns)
    print(f"Factor table: {len(factor_df)} rows, {len(factor_cols)} factors")

    au_df = auto_read_csv(AU9995_PATH, rename_price_to="Au9995")
    au_df = au_df[au_df.index <= CUTOFF_DATE]
    au_df["Au9995"] = pd.to_numeric(au_df["Au9995"], errors="coerce")
    print(f"Au9995 price: {len(au_df)} rows")

    ln_au = np.log(au_df["Au9995"])
    y_df = pd.DataFrame(index=au_df.index)
    for n in Y_RANGE:
        y_df[f"Y_{n}"] = ln_au.shift(-n) - ln_au

    y1_mean = y_df["Y_1"].mean()
    print(f"Y_1 valid rows: {y_df['Y_1'].notna().sum()}")
    print(f"Y_1 mean: {y1_mean:.6f}")
    if abs(y1_mean) > 0.001:
        logger.warning("Y_1 mean outside expected range — check Au9995 data")

    analysis_df = factor_df.join(y_df, how="left")

    # -- Step 2: Compute rolling IC & ICIR --
    print("\n" + "=" * 60)
    print("Step 2: Compute rolling IC & ICIR")
    print("=" * 60)

    if USE_SPLIT:
        label_year = SPLIT_DATE[:4]
        pre_df = analysis_df[analysis_df.index < SPLIT_DATE]
        post_df = analysis_df[analysis_df.index >= SPLIT_DATE]
        runs = [
            (analysis_df, "full"),
            (pre_df, f"pre{label_year}"),
            (post_df, f"post{label_year}"),
        ]
    else:
        runs = [(analysis_df, "")]

    # -- Step 3: Draw heatmaps --
    print("\n" + "=" * 60)
    print("Step 3: Draw heatmaps")
    print("=" * 60)

    for sub_df, lbl in runs:
        print(f"\n>>> Period: {lbl or 'full'} ({len(sub_df)} rows) <<<")
        ic_p, icir_p, ic_s, icir_s = run_ic_analysis(
            sub_df, factor_cols, lbl, OUTPUT_DIR
        )
        draw_and_save_heatmap(
            ic_p, icir_p, ic_s, icir_s, factor_cols, lbl, OUTPUT_DIR
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
