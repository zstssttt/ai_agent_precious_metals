"""Descriptive visualization for indicator data - histogram, Q-Q plot, and time series.

For every aligned indicator, emit a 3-panel figure that lets a reviewer
eyeball data quality before a single factor is computed: are the values
plausibly bounded, is the tail heavier than a normal, do we have any obvious
regime breaks in the time series?
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings("ignore")

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# --- Configuration (override via env vars) ----------------------------------

BASE_DIR = os.environ.get("GOLD_FACTOR_ALIGNED_ROOT", "data/aligned")
OUTPUT_DIR = os.environ.get("GOLD_FACTOR_QC_OUTPUT", "results/plots/distribution_qc")

FREQUENCIES: dict[str, str] = {
    "daily": "daily",
    "weekly": "weekly",
    "monthly": "monthly",
}


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


# --- Core plotting ------------------------------------------------------------

def plot_data_analysis(file_path: Path, output_path: Path) -> None:
    """Generate a 3-panel figure (histogram, Q-Q plot, time series) for one data file."""
    try:
        df = _read_file(file_path)

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        df = df.dropna(subset=["price"])

        if len(df) == 0:
            print(f"  ✗ {file_path.name} - no valid data")
            return

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(file_path.stem, fontsize=14, fontweight="bold")

        # — Histogram —
        ax1 = axes[0]
        ax1.hist(df["price"], bins=50, color="steelblue", alpha=0.7, edgecolor="black")
        ax1.set_xlabel("Price", fontsize=11)
        ax1.set_ylabel("Frequency", fontsize=11)
        ax1.set_title("Histogram", fontsize=12)
        ax1.grid(True, alpha=0.3)

        mean_val = df["price"].mean()
        ax1.axvline(mean_val, color="red", linestyle="--", linewidth=2,
                    label=f"Mean: {mean_val:.2f}")
        ax1.legend()

        # — Q-Q plot —
        ax2 = axes[1]
        stats.probplot(df["price"], dist="norm", plot=ax2)
        ax2.set_title("Q-Q Plot", fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.get_lines()[0].set_color("steelblue")
        ax2.get_lines()[0].set_markersize(4)
        ax2.get_lines()[1].set_color("red")
        ax2.get_lines()[1].set_linewidth(2)

        # — Time series —
        ax3 = axes[2]
        ax3.plot(df["date"], df["price"], color="steelblue", linewidth=1.5)
        ax3.set_xlabel("Date", fontsize=11)
        ax3.set_ylabel("Price", fontsize=11)
        ax3.set_title("Time Series", fontsize=12)
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis="x", rotation=45)

        plt.tight_layout()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"  ✓ {file_path.name}")

    except (UnicodeDecodeError, pd.errors.ParserError, KeyError) as exc:
        print(f"  ✗ {file_path.name} - error: {exc}")


# --- Orchestrator -------------------------------------------------------------

def process_all_data(base_dir: str, output_base_dir: str) -> None:
    """Iterate over all frequency folders and generate charts for every data file."""
    for freq_label, folder_name in FREQUENCIES.items():
        print(f"\n{'=' * 60}")
        print(f"Processing {freq_label} data")
        print(f"{'=' * 60}")

        input_dir = Path(base_dir) / folder_name
        output_dir = Path(output_base_dir) / f"{freq_label}_charts"

        if not input_dir.exists():
            print(f"  Directory not found: {input_dir}")
            continue

        for file in _list_data_files(input_dir):
            plot_data_analysis(file, output_dir / f"{file.stem}.png")

        # Monthly data may contain subfolders (e.g. ETF monthly flows)
        if freq_label == "monthly":
            for subfolder in input_dir.iterdir():
                if subfolder.is_dir() and not subfolder.name.startswith("."):
                    print(f"\n  Processing subfolder: {subfolder.name}")
                    for file in _list_data_files(subfolder):
                        out = output_dir / subfolder.name / f"{file.stem}.png"
                        plot_data_analysis(file, out)


def main() -> None:
    print("=" * 60)
    print("Generating descriptive visualizations")
    print("=" * 60)

    process_all_data(BASE_DIR, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("Done!")
    print(f"Charts saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
