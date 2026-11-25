"""Pairwise correlation analysis with hierarchical clustering.

For each data frequency (daily / weekly / monthly) this script:
    1. Loads every aligned indicator file into a wide DataFrame.
    2. Computes pairwise Pearson correlations.
    3. Reorders rows/columns by hierarchical clustering so that redundant
       indicators visibly cluster together on the heatmap.
    4. Emits the heatmap PNG plus a CSV of every |rho| > HIGH_CORR_THRESHOLD
       pair so that the factor-selection stage can prune duplicates.
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# --- Configuration (override via env vars) ----------------------------------

BASE_DIR = os.environ.get("GOLD_FACTOR_ALIGNED_ROOT", "data/aligned")
OUTPUT_DIR = os.environ.get("GOLD_FACTOR_CORR_OUTPUT", "results/plots/correlation")

FREQUENCIES: dict[str, str] = {
    "daily": "daily",
    "weekly": "weekly",
    "monthly": "monthly",
}

HIGH_CORR_THRESHOLD = 0.7
MIN_CORR_PERIODS = 30
ANNOTATION_MAX_VARS = 20


# --- I/O helpers --------------------------------------------------------------

def _read_file(file_path: Path) -> pd.DataFrame:
    """Read a CSV or Excel file, trying multiple encodings for CSV."""
    if file_path.suffix in (".xlsx", ".xls"):
        return pd.read_excel(file_path)
    for enc in ("utf-8-sig", "gbk", "latin1"):
        try:
            return pd.read_csv(file_path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("utf-8", b"", 0, 1, f"Cannot decode {file_path}")


def _list_data_files(directory: Path) -> list[Path]:
    """Collect CSV / Excel files in *directory*, skipping hidden and log files."""
    files = (
        list(directory.glob("*.csv"))
        + list(directory.glob("*.xlsx"))
        + list(directory.glob("*.xls"))
    )
    return [f for f in files if not f.name.startswith(".") and "log" not in f.name.lower()]


# --- Data loading -------------------------------------------------------------

def load_all_data(data_dir: str | Path) -> pd.DataFrame | None:
    """Load all data files under *data_dir* into a single DataFrame (one column per variable)."""
    all_data: dict[str, pd.Series] = {}
    data_dir = Path(data_dir)

    for file in _list_data_files(data_dir):
        try:
            df = _read_file(file)
            df["date"] = pd.to_datetime(df["date"])
            df = df[~df["date"].duplicated(keep="first")]
            df = df.set_index("date")
            if "price" in df.columns:
                all_data[file.stem] = df["price"]
                print(f"  ✓ {file.name}")
        except (UnicodeDecodeError, pd.errors.ParserError, KeyError) as exc:
            print(f"  ✗ {file.name} - error: {exc}")

    # Subfolders (e.g. monthly ETF flow categories)
    for subfolder in data_dir.iterdir():
        if subfolder.is_dir() and not subfolder.name.startswith("."):
            for file in _list_data_files(subfolder):
                try:
                    df = _read_file(file)
                    df["date"] = pd.to_datetime(df["date"])
                    df = df.set_index("date")
                    if "price" in df.columns:
                        var_name = f"{subfolder.name}_{file.stem}"
                        all_data[var_name] = df["price"]
                        print(f"  ✓ {subfolder.name}/{file.name}")
                except (UnicodeDecodeError, pd.errors.ParserError, KeyError) as exc:
                    print(f"  ✗ {subfolder.name}/{file.name} - error: {exc}")

    return pd.DataFrame(all_data) if all_data else None


# --- Correlation heatmap ------------------------------------------------------

def plot_correlation_matrix(
    df: pd.DataFrame, freq_name: str, output_path: Path
) -> None:
    """Draw a hierarchically-clustered correlation heatmap and save artifacts."""
    corr_matrix = df.corr(method="pearson", min_periods=MIN_CORR_PERIODS)
    corr_matrix = corr_matrix.dropna(how="all", axis=0).dropna(how="all", axis=1)

    if corr_matrix.empty or len(corr_matrix) < 2:
        print("  Insufficient data for correlation matrix — skipping")
        return

    # Vectorized pairwise sample-size matrix
    notna_flags = df[corr_matrix.columns].notna().astype(int)
    sample_sizes = notna_flags.T.dot(notna_flags)

    n_vars = len(corr_matrix.columns)
    fig_size = max(12, n_vars * 0.6)

    g = sns.clustermap(
        corr_matrix,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        figsize=(fig_size, fig_size),
        dendrogram_ratio=0.15,
        cbar_pos=(0.02, 0.83, 0.03, 0.15),
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Correlation"},
        method="average",
        metric="euclidean",
    )

    g.fig.suptitle(
        f"{freq_name} Correlation Matrix (n={n_vars} variables)\n"
        "Ordered by hierarchical clustering",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    g.ax_heatmap.set_xticklabels(
        g.ax_heatmap.get_xticklabels(), rotation=45, ha="right", fontsize=8
    )
    g.ax_heatmap.set_yticklabels(
        g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=8
    )

    if n_vars <= ANNOTATION_MAX_VARS:
        row_order = g.dendrogram_row.reordered_ind
        col_order = g.dendrogram_col.reordered_ind
        for i, row_idx in enumerate(row_order):
            for j, col_idx in enumerate(col_order):
                corr_val = corr_matrix.iloc[row_idx, col_idx]
                if pd.notna(corr_val):
                    g.ax_heatmap.text(
                        j + 0.5, i + 0.5, f"{corr_val:.2f}",
                        ha="center", va="center",
                        color="black" if abs(corr_val) < 0.5 else "white",
                        fontsize=6,
                    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Correlation matrix saved: {output_path}")

    csv_path = output_path.with_suffix(".csv")
    corr_matrix.to_csv(csv_path)
    print(f"  Correlation data saved: {csv_path}")

    sample_path = output_path.parent / f"{freq_name}_sample_sizes.csv"
    sample_sizes.to_csv(sample_path)
    print(f"  Sample-size matrix saved: {sample_path}")

    # Report highly-correlated pairs
    print(f"\n  Highly correlated pairs (|r| > {HIGH_CORR_THRESHOLD}):")
    high_corr: list[dict] = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            n_obs = int(sample_sizes.iloc[i, j])
            if pd.notna(corr_val) and abs(corr_val) > HIGH_CORR_THRESHOLD:
                high_corr.append({
                    "var1": corr_matrix.columns[i],
                    "var2": corr_matrix.columns[j],
                    "correlation": corr_val,
                    "n_obs": n_obs,
                })

    if high_corr:
        high_corr_df = pd.DataFrame(high_corr).sort_values(
            "correlation", key=abs, ascending=False
        )
        print(high_corr_df.to_string(index=False))
        high_corr_path = output_path.parent / f"{freq_name}_high_correlations.csv"
        high_corr_df.to_csv(high_corr_path, index=False)
    else:
        print("  No highly correlated pairs found")


# --- Per-frequency analysis ---------------------------------------------------

def analyze_frequency(freq_name: str, data_dir: str | Path, output_dir: str) -> None:
    """Run full correlation analysis for one frequency tier."""
    print(f"\n{'=' * 70}")
    print(f"Analyzing {freq_name} data")
    print(f"{'=' * 70}")

    print("\nLoading data files:")
    df = load_all_data(data_dir)

    if df is None or df.empty:
        print("  No valid data found")
        return

    print(f"\nData overview:")
    print(f"  Variables: {len(df.columns)}")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")
    print(f"  Observations: {len(df)}")

    print(f"\n  Valid observations per variable:")
    valid_counts = df.notna().sum().sort_values(ascending=False)
    for var, count in valid_counts.items():
        pct = count / len(df) * 100
        date_range = df[var].dropna().index
        if len(date_range) > 0:
            print(f"    {var}: {count} ({pct:.1f}%) "
                  f"[{date_range.min().date()} ~ {date_range.max().date()}]")

    print(f"\n  Using pairwise correlation (each pair uses its joint valid observations)")

    output_path = Path(output_dir) / f"{freq_name}_correlation_matrix.png"
    plot_correlation_matrix(df, freq_name, output_path)


# --- Entry point --------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("Starting correlation analysis")
    print("=" * 70)

    for freq_name, folder_name in FREQUENCIES.items():
        data_dir = Path(BASE_DIR) / folder_name
        if data_dir.exists():
            analyze_frequency(freq_name, data_dir, OUTPUT_DIR)
        else:
            print(f"\nDirectory not found: {data_dir}")

    print("\n" + "=" * 70)
    print("Done!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
