"""
Delayed data processor — aligns weekly or monthly indicators to the
China trading calendar by applying a publication delay and forward-filling.

Usage:
    1. Set DELAY, INPUT_DIR, OUTPUT_DIR, and CALENDAR_PATH below.
    2. Run the script.

Convention: DELAY=0 (daily), DELAY=3 (weekly), DELAY=7 (monthly).
"""

import logging
import os
import traceback
from datetime import timedelta
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# --- Configuration (override via env vars for CI / different environments) ---
DELAY = int(os.environ.get("ALIGN_DELAY", 7))  # weekly=3, monthly=7, daily=0
INPUT_DIR = os.environ.get(
    "ALIGN_INPUT_DIR",
    "data/raw/monthly/US_core_CPI_YoY",
)
OUTPUT_DIR = os.environ.get(
    "ALIGN_OUTPUT_DIR",
    "data/aligned/monthly/US_core_CPI_YoY",
)
CALENDAR_PATH = os.environ.get(
    "TRADING_CALENDAR_PATH",
    "references/china_trading_calendar.xlsx",
)


def _load_trading_dates(calendar_path: str) -> set[str]:
    """Load the China trading calendar and return date strings (YYYY-MM-DD)."""
    calendar_df = pd.read_excel(calendar_path)
    calendar_df["date"] = pd.to_datetime(calendar_df["date"]).dt.strftime(
        "%Y-%m-%d"
    )
    return set(calendar_df["date"])


def _read_indicator_file(file_path: Path) -> pd.DataFrame:
    """Read a CSV or Excel indicator file with automatic encoding detection.

    Returns:
        DataFrame with a parsed 'date' column and a 'price' column.

    Raises:
        ValueError: If no date column is found or no data remains.
    """
    if file_path.suffix in (".xlsx", ".xls"):
        df = pd.read_excel(file_path)
    else:
        try:
            df = pd.read_csv(file_path, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding="gbk")

    date_col = None
    for col in df.columns:
        if col.lower() in ("date", "time", "日期"):
            date_col = col
            break

    if date_col is None:
        raise ValueError(
            f"No date column found; columns are: {df.columns.tolist()}"
        )

    # Fall back to another date-like column if the matched one is all-NaN
    if df[date_col].isna().all():
        for col in df.columns:
            if "date" in col.lower() and not df[col].isna().all():
                date_col = col
                break

    df["date"] = pd.to_datetime(df[date_col])
    df = df.dropna(subset=["date"])

    if len(df) == 0:
        raise ValueError("No rows remain after dropping null dates")

    return df


def _apply_delay_and_upsample(
    df: pd.DataFrame,
    delay: int,
) -> tuple[pd.DataFrame, int]:
    """Shift dates forward by *delay* days and forward-fill to daily frequency.

    Note:
        Adds 'delayed_date' and 'delayed_date_str' columns to *df* in place
        (the caller uses 'delayed_date' for logging).

    Returns:
        (daily_df, nan_count) — daily_df has ['date', 'date_str', 'price'].
    """
    df["delayed_date"] = df["date"] + timedelta(days=delay)

    min_delayed = df["delayed_date"].min()
    max_delayed = df["delayed_date"].max()
    full_range = pd.date_range(min_delayed, max_delayed, freq="D")

    full_df = pd.DataFrame({"date": full_range})
    full_df["date_str"] = full_df["date"].dt.strftime("%Y-%m-%d")

    df["delayed_date_str"] = df["delayed_date"].dt.strftime("%Y-%m-%d")
    price_mapping = df.set_index("delayed_date_str")["price"].to_dict()

    full_df["price"] = full_df["date_str"].map(price_mapping)
    full_df["price"] = full_df["price"].ffill()

    nan_count = int(full_df["price"].isna().sum())
    return full_df, nan_count


def _filter_to_trading_days(
    full_df: pd.DataFrame,
    trading_dates: set[str],
) -> tuple[pd.DataFrame, int]:
    """Keep only rows whose date_str is in the trading calendar.

    Returns:
        (filtered_df, deleted_row_count).
    """
    before_count = len(full_df)
    filtered = full_df[full_df["date_str"].isin(trading_dates)]
    deleted_rows = before_count - len(filtered)
    return filtered, deleted_rows


def _save_processed(output_df: pd.DataFrame, output_path: Path) -> None:
    """Save processed data, choosing format by file extension."""
    if output_path.suffix in (".xlsx", ".xls"):
        output_df.to_excel(output_path, index=False)
    else:
        output_df.to_csv(output_path, index=False, encoding="utf-8-sig")


def process_delayed_data(
    input_dir: str,
    output_dir: str,
    delay: int,
    trading_dates: set[str],
) -> None:
    """Process all indicator files in *input_dir* with delay alignment.

    For each file:
        1. Read the indicator (date, price).
        2. Shift dates forward by *delay* calendar days.
        3. Build a contiguous daily date range and forward-fill prices.
        4. Filter to China trading days only.
        5. Save the result to *output_dir* (same filename).

    A processing_log.csv is written to *output_dir* at the end.
    """
    os.makedirs(output_dir, exist_ok=True)
    logs: list[dict] = []

    for file in Path(input_dir).glob("*"):
        if file.name.startswith(".") or file.suffix not in (
            ".csv", ".xlsx", ".xls"
        ):
            continue

        try:
            df = _read_indicator_file(file)

            original_min = df["date"].min().strftime("%Y-%m-%d")
            original_max = df["date"].max().strftime("%Y-%m-%d")

            full_df, nan_count = _apply_delay_and_upsample(df, delay)
            min_delayed = df["delayed_date"].min()
            max_delayed = df["delayed_date"].max()

            full_df, deleted_rows = _filter_to_trading_days(
                full_df, trading_dates
            )
            added_rows = len(full_df) - len(df)

            output_df = full_df[["date_str", "price"]].copy()
            output_df.columns = ["date", "price"]
            output_df = output_df.sort_values("date").reset_index(drop=True)

            output_path = Path(output_dir) / file.name
            _save_processed(output_df, output_path)

            logs.append({
                "file": file.name,
                "original_date_range": f"{original_min} to {original_max}",
                "delayed_date_range": (
                    f"{min_delayed.strftime('%Y-%m-%d')} to "
                    f"{max_delayed.strftime('%Y-%m-%d')}"
                ),
                "added_rows": added_rows,
                "deleted_rows": deleted_rows,
                "nan_count": nan_count,
                "final_rows": len(output_df),
            })
            print(f"✓ {file.name}")
            print(f"  Original dates: {original_min} ~ {original_max}")
            print(
                f"  After delay:    {min_delayed.strftime('%Y-%m-%d')} ~ "
                f"{max_delayed.strftime('%Y-%m-%d')}"
            )
            print(
                f"  Added: {added_rows}  Removed: {deleted_rows}  "
                f"NaN: {nan_count}  Final rows: {len(output_df)}"
            )
            print()

        except (ValueError, UnicodeDecodeError, pd.errors.ParserError) as exc:
            logger.warning("Skipping %s: %s", file.name, exc)
            print(f"✗ {file.name} - error: {exc}")
            traceback.print_exc()
            logs.append({"file": file.name, "error": str(exc)})

    log_df = pd.DataFrame(logs)
    log_df.to_csv(Path(output_dir) / "processing_log.csv", index=False)
    print(f"\nLog saved to: {output_dir}/processing_log.csv")


def main() -> None:
    print("=" * 70)
    print("Delayed data processor")
    print(f"  DELAY  = {DELAY} days")
    print(f"  Input:   {INPUT_DIR}")
    print(f"  Output:  {OUTPUT_DIR}")
    print("=" * 70)
    print()

    trading_dates = _load_trading_dates(CALENDAR_PATH)
    process_delayed_data(INPUT_DIR, OUTPUT_DIR, DELAY, trading_dates)

    print("=" * 70)
    print("Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
