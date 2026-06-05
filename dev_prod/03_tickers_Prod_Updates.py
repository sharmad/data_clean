"""
update_tickers.py
=================
Production-ready, fully automated version of notebook 03_tickers.ipynb.

What this script does (mirrors the notebook exactly):
  1. Downloads any missing raw daily ticker CSVs from Polygon/Massive.
  2. Cleans each raw CSV (remove rows with no ticker/name, fill NaN types).
  3. Removes ticker symbols that contain a '.' and whose dot-less version
     already appears on the same day (legacy CMCS.A / CMCSA style duplicates).
  4. Builds / extends  data/tickers_v1.csv  — a point-in-time master list
     where every row is one continuous listing window for one ticker, with
     columns: ID, ticker, name, active, start_date, end_date, type, cik,
     composite_figi.
  5. Merges consecutive duplicate rows that are caused by minor Polygon
     metadata changes (name tweaks, CIK corrections, etc.) for the SAME
     economic instrument.
  6. Removes non-common-stock rows: funds, preferreds, warrants, when-issued
     shares, ticker suffixes (.W, .P, …), fifth-character special codes, etc.
  7. Removes "ghost" rows — tickers whose start_date == end_date and that
     are not brand-new listings as of today.
  8. Stamps each surviving row with a unique ID and saves tickers_v1.csv.

Designed to be idempotent: safe to run every day via cron. Only work that
hasn't been done before is performed.

Cron example (run at 06:00 ET every weekday):
  0 6 * * 1-5 /path/to/.venv/bin/python /path/to/update_tickers.py >> /var/log/tickers.log 2>&1

Environment variables expected (same as the notebook):
  ACCESS_KEY_ID    – polygon/massive access key
  MASSIVE_API_KEY  – massive REST API key

Directory layout expected (same as the notebook):
  <repo_root>/
    data/
      polygon/
        raw/
          tickers/      ← one CSV per trading day, e.g. 2024-03-01.csv
          tmp/          ← scratch files for diagnostics
      tickers_v1.csv   ← master output; created on first run
    update_tickers.py  ← this file
"""

import os
import re
import bisect
import logging
import warnings
from datetime import date, timedelta
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# ── suppress noisy pandas FutureWarnings that we can't avoid ─────────────────
warnings.simplefilter(action="ignore", category=FutureWarning)


# =============================================================================
# 0.  LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# =============================================================================
# 1.  CONFIGURATION  (tweak here, not deep inside the code)
# =============================================================================

load_dotenv()

# ── Polygon/Massive credentials ───────────────────────────────────────────────
ACCESS_KEY_ID = os.environ.get("ACCESS_KEY_ID")
API_KEY       = os.environ.get("MASSIVE_API_KEY")

if not ACCESS_KEY_ID:
    raise EnvironmentError("ACCESS_KEY_ID not set in environment / .env file.")
if not API_KEY:
    raise EnvironmentError("MASSIVE_API_KEY not set in environment / .env file.")

# ── Paths ─────────────────────────────────────────────────────────────────────
# Resolve relative to this script's location so it works wherever it's called from.
SCRIPT_DIR        = Path(__file__).resolve().parent
POLYGON_DATA_PATH = SCRIPT_DIR / "data" / "polygon"
RAW_TICKERS_DIR   = POLYGON_DATA_PATH / "raw" / "tickers"
TMP_DIR           = POLYGON_DATA_PATH / "tmp"
MASTER_CSV        = SCRIPT_DIR / "data" / "tickers_v1.csv"

# Create directories if they don't exist yet (first-ever run)
for d in [RAW_TICKERS_DIR, TMP_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Date range ────────────────────────────────────────────────────────────────
# START_DATE is used only when *no* master CSV exists yet (cold start).
# END_DATE is always "today" so the cron job is fully automatic.
START_DATE = date(1999, 1, 1)   # adjust to your Polygon subscription start
END_DATE   = date.today()

# ── Ticker-type exclusion list (applied at download time) ─────────────────────
# These types are excluded when fetching from Polygon.
EXCLUDED_TYPES = {
    "PFD", "WARRANT", "RIGHT", "BOND", "ETF", "ETN", "ETV", "SP",
    "ADRP", "ADRW", "ADRR", "FUND", "BASKET", "UNIT", "LT", "GDR",
    "OTHER", "AGEN", "EQLK", "ETS", "INDEX",
}

# ── Tickers that are explicitly whitelisted from the "fund / bad name" filter ─
TICKER_WHITELIST = {
    "JPM", "EV", "IVZ", "IVR", "BLK", "TCPC", "DJP", "KMI",
    "KMP", "KMR", "MHGC", "MS", "MWD", "MR", "MFUN", "C",
    "CS", "CSR", "UBS", "CRO",
}

# ── Tickers that are explicitly blacklisted (always removed) ──────────────────
TICKER_BLACKLIST = {
    "CIP", "FSMO", "LOR", "CMCA", "EMD", "HORI", "KEMP", "HCD",
    "PMO", "PCV", "FSCO", "AGB", "FIV", "GYB", "IKBCO", "MBINO",
    "INDB.N", "INDB", "JBN", "XFLT", "VKA", "VKC", "VKS",
    "TMT", "TMB", "RMT",
}

# ── Custom na_values used everywhere we read a CSV ────────────────────────────
NA_VALUES = [
    "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan",
    "1.#IND", "1.#QNAN", "<NA>", "N/A", "NULL", "NaN", "None",
    "n/a", "nan", "null",
]


# =============================================================================
# 2.  HELPERS
# =============================================================================

def _to_date(d) -> date:
    """
    Normalise anything date-like to a plain datetime.date.

    get_market_dates() from the 'times' module returns pandas Timestamps,
    while date.fromisoformat() returns datetime.date objects.  Mixing the
    two in set membership tests always returns False even when the calendar
    day is identical, causing every day to look 'missing'.  This helper
    ensures we always compare like with like.
    """
    if type(d) is date:              # already a plain date — fast path
        return d
    if isinstance(d, date):          # datetime.datetime is a subclass of date
        return d.date()
    if hasattr(d, "date"):           # pandas Timestamp, numpy datetime64, …
        return d.date()
    return date.fromisoformat(str(d)[:10])   # last resort via string


def get_market_dates() -> list[date]:
    """
    Return a sorted list of US market trading days.

    We import from the project's own 'times' module (same as the notebook).
    If that module is unavailable we fall back to pandas_market_calendars so
    the script remains runnable in isolation.
    """
    try:
        from times import get_market_dates as _get  # type: ignore
        # Normalise to plain datetime.date — the 'times' module may return
        # pandas Timestamps, which break set-membership tests against
        # date.fromisoformat() results.
        return [_to_date(d) for d in _get()]
    except ImportError:
        log.warning(
            "'times' module not found – falling back to pandas_market_calendars."
        )
        try:
            import pandas_market_calendars as mcal  # type: ignore
        except ImportError:
            raise ImportError(
                "Install 'pandas_market_calendars' or make sure the project "
                "'times' module is on PYTHONPATH."
            )
        nyse = mcal.get_calendar("NYSE")
        schedule = nyse.schedule(
            start_date=str(START_DATE), end_date=str(END_DATE)
        )
        # Always return plain datetime.date objects, not Timestamps
        return [d.date() for d in schedule.index]


def _read_raw_ticker_csv(path: Path) -> pd.DataFrame:
    """Read one raw daily ticker CSV with consistent dtypes."""
    df = pd.read_csv(
        path,
        index_col=0,
        keep_default_na=False,
        na_values=NA_VALUES,
    )
    # Normalise cik: empty string → NaN, then float
    df["cik"] = df["cik"].replace("", np.nan).astype(float)
    return df


def _read_master_csv(path: Path) -> pd.DataFrame:
    """Read the master tickers_v1.csv with proper date parsing."""
    df = pd.read_csv(
        path,
        index_col=0,
        parse_dates=["start_date", "end_date"],
        keep_default_na=False,
        na_values=NA_VALUES,
    )
    df["start_date"] = pd.to_datetime(df["start_date"]).dt.date
    df["end_date"]   = pd.to_datetime(df["end_date"]).dt.date
    df["cik"]        = df["cik"].replace("", np.nan)
    df["name"]       = df["name"].astype(str)
    df["ticker"]     = df["ticker"].astype(str)
    return df


# =============================================================================
# 3.  STEP 1 – DOWNLOAD MISSING RAW TICKER CSVS
# =============================================================================

def download_tickers_for_day(client, day: date) -> pd.DataFrame:
    """
    Fetch the active stock ticker list for *day* from Polygon/Massive,
    filter to common/ADR/ordinary shares (and None-typed), and return a
    clean DataFrame.

    Mirrors the notebook's `download_tickers()` function.
    """
    date_iso = day.isoformat()

    tickers_iter = client.list_tickers(
        date=date_iso, active=True, market="stocks", limit=1000
    )
    df = pd.DataFrame(tickers_iter)

    if df.empty:
        log.warning("Empty ticker list returned for %s", date_iso)
        return df

    # Keep only CS, ADRC, NYRS, OS – and None/NaN typed tickers which may be
    # mis-labelled common stocks (e.g. AAPL was once labelled None).
    df = df[~df["type"].isin(EXCLUDED_TYPES)]
    df.sort_values("ticker", inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df[[
        "ticker", "name", "active", "delisted_utc",
        "last_updated_utc", "cik", "composite_figi", "type",
    ]]


def fetch_missing_raw_tickers(market_days: list[date]) -> None:
    """
    Download and save a CSV for every trading day in [START_DATE, END_DATE]
    that we don't already have on disk.

    This is idempotent – already-downloaded days are skipped.

    Root cause of the original bug
    --------------------------------
    get_market_dates() returned pandas Timestamps while date.fromisoformat()
    returns datetime.date objects.  Comparing them with 'in' / 'not in'
    always evaluated to False (even for the same calendar day), so every day
    appeared missing.  All dates are now normalised via _to_date() at the
    point where get_market_dates() is called, so by the time market_days
    reaches this function every element is already a plain datetime.date.
    """
    from massive.rest import RESTClient  # type: ignore  (project dependency)

    client = RESTClient(api_key=API_KEY)

    # Build the set of already-downloaded days as plain datetime.date objects
    existing: set[date] = {
        date.fromisoformat(f.stem)
        for f in RAW_TICKERS_DIR.glob("*.csv")
    }

    # market_days are already plain date objects (normalised in get_market_dates)
    days_to_fetch = [
        d for d in market_days
        if START_DATE <= d <= END_DATE and d not in existing
    ]

    if not days_to_fetch:
        log.info("All raw ticker CSVs are up to date.")
        return

    log.info("Fetching %d missing raw ticker CSVs …", len(days_to_fetch))
    for day in days_to_fetch:
        try:
            df = download_tickers_for_day(client, day)
            df.to_csv(RAW_TICKERS_DIR / f"{day.isoformat()}.csv")
            log.info("  Downloaded %s  (%d tickers)", day, len(df))
        except Exception as exc:
            log.error("  FAILED to download %s: %s", day, exc)


# =============================================================================
# 4.  STEP 2 – CLEAN RAW CSVS IN-PLACE
# =============================================================================

def clean_raw_csvs() -> None:
    """
    For every raw daily CSV:
      • Remove rows with no ticker or no name.
      • Fill NaN 'type' with 'NONE' (Polygon sometimes omits this field).

    Also removes legacy "dotted-ticker / same-company" duplicates, e.g.
    CMCS.A appearing alongside CMCSA on the same day.

    Files are rewritten in place. Already-clean files are handled gracefully
    because the operations are idempotent.
    """
    files = sorted(RAW_TICKERS_DIR.glob("*.csv"))
    log.info("Cleaning %d raw ticker CSVs …", len(files))

    for fpath in files:
        df = _read_raw_ticker_csv(fpath)

        # ── drop rows missing ticker or name ─────────────────────────────────
        df = df[df["ticker"].notna() & df["name"].notna()]

        # ── fill missing type ─────────────────────────────────────────────────
        df["type"] = df["type"].fillna("NONE")

        # ── remove dotted-ticker duplicates (e.g. CMCS.A when CMCSA exists) ──
        # Only remove the non-dotted version when the dotted version is the
        # 'canonical' one that Polygon uses for older periods.
        with_dots = df[df["ticker"].str.contains(r"\.", regex=True)]
        indices_to_drop = []
        for _, row in with_dots.iterrows():
            ticker_no_dot = row["ticker"].replace(".", "")
            match = df[df["ticker"] == ticker_no_dot]
            if not match.empty and match["name"].values[0] == row["name"]:
                # The plain version is a duplicate of the dotted one – drop it
                indices_to_drop.append(match.index[0])

        if indices_to_drop:
            df.drop(index=indices_to_drop, inplace=True)

        df.reset_index(drop=True).to_csv(fpath)


# =============================================================================
# 5.  STEP 3 – BUILD / EXTEND THE MASTER TICKER LIST
# =============================================================================

# Columns used for identity comparison when detecting delistings/new-listings.
# last_updated_utc and delisted_utc are intentionally excluded (they are
# not point-in-time for a given day).
IDENTITY_COLS = ["ticker", "name", "active", "cik", "composite_figi", "type"]


def _load_day_active(day: date) -> pd.DataFrame:
    """
    Load the raw CSV for *day*, keep only active tickers, de-duplicate,
    and return a clean DataFrame with IDENTITY_COLS.
    """
    fpath = RAW_TICKERS_DIR / f"{day.isoformat()}.csv"
    df = _read_raw_ticker_csv(fpath)

    df.sort_values("last_updated_utc", inplace=True)
    df = df[df["active"] == True][IDENTITY_COLS].copy()
    df.reset_index(drop=True, inplace=True)

    # Remove per-day ticker duplicates: keep the last (most recent) entry
    dup_mask = df["ticker"].duplicated(keep=False)
    if dup_mask.any():
        drop_idx = df[dup_mask]["ticker"].duplicated(keep="last")
        df.drop(index=drop_idx[drop_idx].index, inplace=True)
        df.reset_index(drop=True, inplace=True)

    return df


def _detect_changes(
    our: pd.DataFrame, new: pd.DataFrame
) -> tuple[pd.Series, pd.Series]:
    """
    Compare our current master snapshot with the new day's Polygon list.

    Returns
    -------
    indicator_delisted : boolean Series aligned with *our* index
        True  → this row was in *our* but not in *new*  (delisting/change)
    indicator_new      : boolean Series aligned with *new* index
        True  → this row was in *new*  but not in *our* (new listing)
    """
    merged_del = our[IDENTITY_COLS].merge(
        new[IDENTITY_COLS],
        on=IDENTITY_COLS, how="left", indicator=True,
    )
    # Already-inactive rows in *our* must never be flagged as newly delisted
    merged_del["_merge"] = np.where(
        our["active"], merged_del["_merge"], "both"
    )
    indicator_delisted = merged_del["_merge"] == "left_only"

    merged_new = new[IDENTITY_COLS].merge(
        our[IDENTITY_COLS],
        on=IDENTITY_COLS, how="left", indicator=True,
    )
    indicator_new = merged_new["_merge"] == "left_only"

    return indicator_delisted, indicator_new


def build_master_ticker_list(market_days: list[date]) -> pd.DataFrame:
    """
    Build the master point-in-time ticker list from scratch (cold start).

    Loops over every trading day, initialises on the first available CSV,
    then applies the delisting / new-listing logic described in section 3.2
    of the notebook.

    Returns the final DataFrame (not yet cleaned of funds/preferreds etc.).
    """
    # Find the first day for which we have data and that is >= START_DATE
    available = sorted(
        date.fromisoformat(f.stem)
        for f in RAW_TICKERS_DIR.glob("*.csv")
    )
    if not available:
        raise FileNotFoundError("No raw ticker CSVs found – run fetch first.")

    first_data_day = min(d for d in available if d >= START_DATE)
    last_data_day  = max(d for d in available if d <= END_DATE)

    # Restrict market_days to the range we have data for
    days_in_range = [
        d for d in market_days
        if first_data_day <= d <= last_data_day
    ]

    our: pd.DataFrame | None = None

    for idx, day in enumerate(days_in_range):
        fpath = RAW_TICKERS_DIR / f"{day.isoformat()}.csv"
        if not fpath.exists():
            log.warning("Missing CSV for %s – skipping day.", day)
            continue

        # ── INITIALISATION (very first day) ──────────────────────────────────
        if our is None:
            our = _load_day_active(day).copy()
            our["start_date"] = day
            our["end_date"]   = pd.NaT
            log.info("%s: Initialised with %d tickers.", day, len(our))
            continue

        # ── SUBSEQUENT DAYS ───────────────────────────────────────────────────
        new_day = _load_day_active(day)

        # Sanity check
        if our.duplicated().any():
            raise RuntimeError(f"Duplicates in master list before processing {day}")

        indicator_delisted, indicator_new = _detect_changes(our, new_day)

        # Previous trading day (needed for correct end_date on delistings)
        prev_day = days_in_range[idx - 1]

        # Process delistings: set end_date to *previous* trading day (not today)
        # Rationale in notebook: OHLCV data for FB exists up to 2022-06-08
        # (the day BEFORE it was removed from Polygon's list on 2022-06-09).
        our.loc[indicator_delisted.values, "end_date"] = prev_day
        our.loc[indicator_delisted.values, "active"]   = False

        # Process new listings: append and fill start_date
        if indicator_new.any():
            new_rows = new_day[indicator_new.values].copy()
            our = pd.concat([our, new_rows], ignore_index=True)
            our["start_date"] = our["start_date"].fillna(day)

        # Guard: required columns must never be null
        required = ["ticker", "name", "active", "start_date"]
        if our[required].isnull().values.any():
            bad = our[our[required].isnull().any(axis=1)]
            raise RuntimeError(
                f"Null values in required columns after processing {day}:\n{bad}"
            )

        log.info("%s: %d tickers in master list.", day, len(our))

    # Close-out: fill remaining NaT end_dates with the last data day
    our["end_date"] = our["end_date"].fillna(last_data_day)
    our = our.sort_values(["ticker", "end_date"]).reset_index(drop=True)
    return our


def extend_master_ticker_list(
    existing: pd.DataFrame, market_days: list[date]
) -> pd.DataFrame:
    """
    Incrementally extend an existing master ticker list with new trading days.

    *existing* is the DataFrame read from tickers_v1.csv.
    Only days AFTER the current max end_date are processed.

    Returns the updated DataFrame.
    """
    current_end = existing["end_date"].max()
    log.info("Existing master list ends at %s.", current_end)

    available = sorted(
        date.fromisoformat(f.stem)
        for f in RAW_TICKERS_DIR.glob("*.csv")
    )
    new_days = [
        d for d in market_days
        if d > current_end and d <= END_DATE and d in available
    ]

    if not new_days:
        log.info("Master list is already up to date.")
        return existing

    log.info("Extending master list with %d new trading days …", len(new_days))

    # Re-open end_dates for still-active tickers (they were set to current_end)
    our = existing.copy()
    our.loc[our["active"] == True, "end_date"] = pd.NaT

    # We need the full sorted market_days list for prev_day lookup
    sorted_all = sorted(set(market_days) | set(available))

    for day in new_days:
        fpath = RAW_TICKERS_DIR / f"{day.isoformat()}.csv"
        if not fpath.exists():
            log.warning("Missing CSV for %s – skipping.", day)
            continue

        new_day = _load_day_active(day)

        indicator_delisted, indicator_new = _detect_changes(our, new_day)

        pos = bisect.bisect_left(sorted_all, day)
        prev_day = sorted_all[pos - 1] if pos > 0 else day

        our.loc[indicator_delisted.values, "end_date"] = prev_day
        our.loc[indicator_delisted.values, "active"]   = False

        if indicator_new.any():
            new_rows = new_day[indicator_new.values].copy()
            our = pd.concat([our, new_rows], ignore_index=True)
            our["start_date"] = our["start_date"].fillna(day)

        log.info("%s: %d tickers.", day, len(our))

    last_data_day = max(new_days)
    our["end_date"] = our["end_date"].fillna(last_data_day)
    our = our.sort_values(["ticker", "end_date"]).reset_index(drop=True)
    return our


# =============================================================================
# 6.  STEP 4 – MERGE CONSECUTIVE DUPLICATES  (notebook section 3.5)
# =============================================================================

def merge_consecutive_duplicates(
    df: pd.DataFrame, market_days: list[date]
) -> pd.DataFrame:
    """
    Many tickers appear multiple times in the master list because Polygon
    makes minor metadata changes (name spelling, CIK corrections, FIGI
    updates) without changing the underlying company or ticker.

    If two consecutive rows for the same ticker have back-to-back trading
    days (end_date of row N is immediately followed by start_date of row N+1),
    they represent the *same* listing window and should be merged into one row.

    Merge rules (same as notebook section 3.5):
      name           → last value
      active         → last value
      start_date     → first value
      end_date       → last value
      type           → last value
      cik            → last non-NaN value (forward-fill)
      composite_figi → last non-NaN value (forward-fill)

    Returns a new (smaller) DataFrame.
    """
    market_set   = set(market_days)
    market_sorted = sorted(market_set)

    # Build a fast lookup: date → next trading date
    next_day: dict[date, date] = {}
    for i, d in enumerate(market_sorted[:-1]):
        next_day[d] = market_sorted[i + 1]

    df = df.sort_values(["ticker", "end_date"]).reset_index(drop=True)

    # Find groups of consecutive rows per ticker
    groups: list[list[int]] = []   # each element is a list of row indices to merge
    prev_ticker:    str | None = None
    prev_end:       date | None = None
    current_group:  list[int] = []

    for idx, row in df.iterrows():
        ticker     = row["ticker"]
        start      = row["start_date"]
        end        = row["end_date"]

        if (
            ticker == prev_ticker
            and prev_end is not None
            and next_day.get(prev_end) == start   # back-to-back trading days
        ):
            current_group.append(idx)
        else:
            if len(current_group) > 1:
                groups.append(current_group)
            current_group = [idx]

        prev_ticker = ticker
        prev_end    = end

    if len(current_group) > 1:
        groups.append(current_group)

    # Apply merges
    indices_to_drop: set[int] = set()

    for group in groups:
        sub = df.loc[group]
        # All rows in the group get the merged values written back
        merged_name   = sub["name"].values[-1]
        merged_active = sub["active"].values[-1]
        merged_start  = sub["start_date"].values[0]
        merged_end    = sub["end_date"].values[-1]
        merged_type   = sub["type"].values[-1]
        merged_cik    = sub["cik"].ffill().values[-1]
        merged_figi   = sub["composite_figi"].ffill().values[-1]

        df.loc[group, "name"]           = merged_name
        df.loc[group, "active"]         = merged_active
        df.loc[group, "start_date"]     = merged_start
        df.loc[group, "end_date"]       = merged_end
        df.loc[group, "type"]           = merged_type
        df.loc[group, "cik"]            = merged_cik
        df.loc[group, "composite_figi"] = merged_figi

        # Keep only the first row; the rest are now exact duplicates
        # (drop_duplicates below will remove them)

    df = df.drop_duplicates().reset_index(drop=True)
    log.info("After merging consecutive duplicates: %d rows.", len(df))
    return df


# =============================================================================
# 7.  STEP 5 – REMOVE NON-COMMON-STOCK ROWS  (notebook section 3.6)
# =============================================================================

# ── Word-based fund/ETF name filter ──────────────────────────────────────────
_FUND_WORDS = {
    "fund", "fund,", "fnd", "fd", "aberdeen", "barings,", "blackrock",
    "barclays", "bldrs", "contigent", "citigroup", "direxion", "mfs",
    "eaton", "calamos", "nuveen", "proshares", "suisse", "ishares",
    "jpmorgan", "invesco", "powershares", "gabelli", "morgan", "merrill",
    "merill", "merr", "etf", "etn", "etv", "index", "idx", "indx", "ctf",
    "pwrshrs", "pwrsh", "dbx", "msdw", "structured", "tr", "ubs",
    "bulletshares", "xai", "structrd", "structurd", "putnam", "citigrp",
    "citigp", "citicgroup", "mrgn", "lnk", "pines", "crt", "cert",
    "certificate", "lkd", "lknd", "velocityshs", "structred", "struct",
    "nt", "unit", "units", '"quids"',
}

# ── Ticker-suffix regex (warrants, preferreds, rights etc.) ──────────────────
_SUFFIX_RE = re.compile(
    r"\.(?:WD|W|Z|V|U|P|PRTC|RTS|PRU|PRAC|PRDC|PRPC|PRBC|PREC)$"
)

# ── Nasdaq fifth-character special codes ─────────────────────────────────────
_FIFTH_CHAR_CODES = set("GHIMNOPRTUVWZ")

# ── Preferred / warrant name-words ───────────────────────────────────────────
_PREF_WORDS = {
    "pf", "pfd", "pfr", "pref", "preferred", "exp", "due",
    "expiry", "abs", "warrant", "warrants", "crts",
}

# ── When-issued / ex-distribution name-words ─────────────────────────────────
_WHEN_WORDS = {
    "when", "issued", "when-issued", "ex-distribution", "w.i.", "wts",
}


def _name_contains_any(name: str, words: set[str]) -> bool:
    """True if any token in *name* (lowercased) is in *words*."""
    if not isinstance(name, str):
        return False
    return bool(words.intersection(name.lower().split()))


def _is_fund(row: pd.Series) -> bool:
    """Return True if this row looks like a fund/ETF/index product."""
    ticker = str(row["ticker"])
    name   = str(row["name"])

    if ticker in TICKER_WHITELIST:
        return False

    if ticker in TICKER_BLACKLIST:
        return True

    # Pioneer trust combo
    tokens = name.lower().split()
    if "pioneer" in tokens and "trust" in tokens:
        return True

    return _name_contains_any(name, _FUND_WORDS)


def _is_preferred_or_warrant(row: pd.Series) -> bool:
    """Return True if the row looks like a preferred share, warrant, bond."""
    name = str(row["name"])
    if "%" in name:
        return True
    return _name_contains_any(name, _PREF_WORDS)


def _is_when_issued(row: pd.Series) -> bool:
    """Return True for when-issued or ex-distribution shares."""
    return _name_contains_any(str(row["name"]), _WHEN_WORDS)


def _has_lowercase_ticker(row: pd.Series) -> bool:
    """Tickers with lowercase letters are special-condition shares (e.g. AANw)."""
    ticker = str(row["ticker"])
    return any(c.islower() for c in ticker)


def _has_dotted_suffix(row: pd.Series) -> bool:
    """Tickers like ABEO.W, ACGL.P, ADXS.W etc."""
    return bool(_SUFFIX_RE.search(str(row["ticker"])))


def _is_fifth_char_special(row: pd.Series) -> bool:
    """
    Tickers that are 5 characters long (after removing dots) and whose last
    character is a Nasdaq fifth-character special code.

    E.g. AAPGV (V = when-issued), ADEAV.
    """
    ticker = str(row["ticker"])
    clean  = ticker.replace(".", "")
    return len(clean) == 5 and clean[-1] in _FIFTH_CHAR_CODES


def remove_non_common_stocks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all six exclusion filters from notebook section 3.6 and return
    the filtered DataFrame.
    """
    before = len(df)

    mask_keep = ~(
          df.apply(_is_fund,                   axis=1)
        | df.apply(_is_preferred_or_warrant,   axis=1)
        | df.apply(_is_when_issued,            axis=1)
        | df.apply(_has_lowercase_ticker,      axis=1)
        | df.apply(_has_dotted_suffix,         axis=1)
        | df.apply(_is_fifth_char_special,     axis=1)
    )

    df = df[mask_keep].reset_index(drop=True)
    log.info(
        "Non-common-stock filter: removed %d rows, %d remain.",
        before - len(df), len(df),
    )
    return df


# =============================================================================
# 8.  STEP 6 – REMOVE GHOST TICKERS
# =============================================================================

def remove_ghost_tickers(
    df: pd.DataFrame, last_trading_day: date
) -> pd.DataFrame:
    """
    Remove rows where start_date == end_date (single-day ghost listings),
    UNLESS that start_date is the very last trading day (genuinely new IPOs
    that happened to be fetched on day 1).

    Returns the filtered DataFrame.
    """
    before = len(df)
    mask_keep = (
        (df["end_date"] > df["start_date"])              # more than one day
        | (df["start_date"] == last_trading_day)         # brand-new listing today
    )
    df = df[mask_keep].reset_index(drop=True)
    log.info(
        "Ghost-ticker filter: removed %d rows, %d remain.",
        before - len(df), len(df),
    )
    return df


# =============================================================================
# 9.  STEP 7 – ASSIGN UNIQUE IDs AND SAVE
# =============================================================================

def finalise_and_save(df: pd.DataFrame) -> None:
    """
    Add the unique ID column (ticker-startdate) and write tickers_v1.csv.

    The ID is identical to the notebook's convention so downstream notebooks
    remain compatible.
    """
    df = df.reset_index(drop=True)
    df["ID"] = df["ticker"] + "-" + df["start_date"].astype(str)

    output_cols = [
        "ID", "ticker", "name", "active",
        "start_date", "end_date", "type", "cik", "composite_figi",
    ]
    df = df[output_cols]

    df.to_csv(MASTER_CSV)
    log.info(
        "Saved %d rows (%d unique tickers) to %s",
        len(df), df["ticker"].nunique(), MASTER_CSV,
    )


# =============================================================================
# 10.  ORCHESTRATOR
# =============================================================================

def run() -> None:
    """
    End-to-end pipeline.  Called by __main__ or by a cron / scheduler.

    Steps
    -----
    1. Get the full list of US market trading days.
    2. Download any raw ticker CSVs we're missing.
    3. Clean all raw CSVs in-place.
    4. Build or extend the master ticker list.
    5. Merge consecutive metadata-change duplicates.
    6. Remove non-common-stock entries.
    7. Remove ghost (single-day) tickers.
    8. Save tickers_v1.csv.
    """
    log.info("=" * 60)
    log.info("Ticker pipeline starting  (END_DATE = %s)", END_DATE)
    log.info("=" * 60)

    # ── 1. Trading calendar ───────────────────────────────────────────────────
    market_days = get_market_dates()
    log.info("Trading calendar: %d days from %s to %s.",
             len(market_days), market_days[0], market_days[-1])

    # ── 2. Download missing raw CSVs ──────────────────────────────────────────
    fetch_missing_raw_tickers(market_days)

    # ── 3. Clean raw CSVs ─────────────────────────────────────────────────────
    clean_raw_csvs()

    # ── 4. Build or extend master list ────────────────────────────────────────
    if MASTER_CSV.exists():
        log.info("Master CSV found – extending incrementally.")
        existing = _read_master_csv(MASTER_CSV)
        master   = extend_master_ticker_list(existing, market_days)
    else:
        log.info("No master CSV found – performing cold-start build.")
        master = build_master_ticker_list(market_days)

    # ── 5. Merge consecutive duplicates ───────────────────────────────────────
    master = merge_consecutive_duplicates(master, market_days)

    # ── 6. Remove non-common-stock rows ───────────────────────────────────────
    master = remove_non_common_stocks(master)

    # ── 7. Remove ghost tickers ───────────────────────────────────────────────
    last_trading_day = max(
        date.fromisoformat(f.stem)
        for f in RAW_TICKERS_DIR.glob("*.csv")
        if f.stem <= END_DATE.isoformat()
    )
    master = remove_ghost_tickers(master, last_trading_day)

    # ── 8. Save ───────────────────────────────────────────────────────────────
    finalise_and_save(master)

    log.info("=" * 60)
    log.info("Pipeline complete.")
    log.info("=" * 60)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    run()
