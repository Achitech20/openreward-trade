"""
Prepare per-market CSV files from S3 parquet data.

Downloads archived parquet files from S3, processes them into a unified dataset,
then splits by market_end into individual CSV files suitable for the BtcTraderEnv.

Usage:
    python data/prepare_markets.py --start-date 2026-02-01 --end-date 2026-02-15 --output data/markets/
"""

import argparse
import gc
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# S3 constants
S3_BUCKET = "polymarket-archiver-data"
S3_PREFIX = "polymarket/"
S3_REGION = "eu-west-1"
S3_TOPICS = ["polymarket-events", "exchange-prices", "analytics-events"]


# ---------------------------------------------------------------------------
# Shared helpers (adapted from option-price-model/data_loader.py)
# ---------------------------------------------------------------------------

def parse_timestamp(ts_raw) -> Optional[datetime]:
    if ts_raw is None:
        return None
    try:
        if isinstance(ts_raw, (int, float)):
            if ts_raw > 1e15:
                return datetime.fromtimestamp(ts_raw / 1e9, tz=timezone.utc)
            elif ts_raw > 1e12:
                return datetime.fromtimestamp(ts_raw / 1000, tz=timezone.utc)
            else:
                return datetime.fromtimestamp(ts_raw, tz=timezone.utc)
        elif isinstance(ts_raw, str):
            ts_str = ts_raw.replace("Z", "+00:00")
            if "." in ts_str and "+" in ts_str:
                dot_idx = ts_str.index(".")
                plus_idx = ts_str.index("+")
                decimals = ts_str[dot_idx + 1 : plus_idx]
                if len(decimals) > 6:
                    ts_str = ts_str[: dot_idx + 7] + ts_str[plus_idx:]
            return datetime.fromisoformat(ts_str)
        elif isinstance(ts_raw, datetime):
            if ts_raw.tzinfo is None:
                return ts_raw.replace(tzinfo=timezone.utc)
            return ts_raw
    except Exception:
        pass
    return None


def fetch_s3_data(start_date: str, end_date: str, data_dir: str, topics: List[str] = None) -> int:
    """Download parquet files from S3, skipping files already present locally."""
    try:
        import boto3
    except ImportError:
        logger.error("boto3 not installed. Run: pip install boto3")
        sys.exit(1)

    s3 = boto3.client("s3", region_name=S3_REGION)
    data_path = Path(data_dir)
    topics = topics or S3_TOPICS

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    total_downloaded = 0
    total_skipped = 0

    logger.info(f"Fetching S3 data: {start_date} to {end_date}, topics={topics}")

    current = start
    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        for topic in topics:
            s3_prefix = f"{S3_PREFIX}topic={topic}/date={date_str}/"
            local_dir = data_path / f"topic={topic}" / f"date={date_str}"

            paginator = s3.get_paginator("list_objects_v2")
            try:
                for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=s3_prefix):
                    for obj in page.get("Contents", []):
                        key = obj["Key"]
                        filename = key.split("/")[-1]
                        if not filename.endswith(".parquet"):
                            continue
                        local_file = local_dir / filename
                        if local_file.exists() and local_file.stat().st_size == obj["Size"]:
                            total_skipped += 1
                            continue
                        local_dir.mkdir(parents=True, exist_ok=True)
                        s3.download_file(S3_BUCKET, key, str(local_file))
                        total_downloaded += 1
                        if total_downloaded % 50 == 0:
                            logger.info(f"  Downloaded {total_downloaded} files...")
            except Exception as e:
                logger.warning(f"S3 error for {topic}/{date_str}: {e}")

        current += timedelta(days=1)

    logger.info(f"S3 fetch: {total_downloaded} downloaded, {total_skipped} skipped")
    return total_downloaded


def find_parquet_files(data_path: Path, topic: str,
                       start_date: str = None, end_date: str = None) -> List[Path]:
    topic_dir = data_path / f"topic={topic}"
    if not topic_dir.exists():
        return []
    files = []
    for date_dir in sorted(topic_dir.glob("date=*")):
        date_str = date_dir.name.replace("date=", "")
        if start_date and date_str < start_date:
            continue
        if end_date and date_str > end_date:
            continue
        for f in sorted(date_dir.glob("*.parquet*")):
            files.append(f)
    return files


def _parse_payloads_from_file(filepath: Path) -> pd.DataFrame:
    try:
        df = pd.read_parquet(filepath, columns=["ts", "payload"])
    except Exception as e:
        logger.warning(f"Error reading {filepath.name}: {e}")
        return pd.DataFrame()
    if df.empty:
        return pd.DataFrame()

    payloads = []
    for _, row in df.iterrows():
        payload_str = row.get("payload", "{}")
        try:
            event = json.loads(payload_str) if isinstance(payload_str, str) else payload_str
            if event is None:
                continue
            if "ts" not in event:
                event["_parquet_ts"] = row.get("ts")
            payloads.append(event)
        except (json.JSONDecodeError, TypeError):
            continue
    if not payloads:
        return pd.DataFrame()
    return pd.DataFrame(payloads)


# ---------------------------------------------------------------------------
# Option orderbook loader
# ---------------------------------------------------------------------------

def _process_option_file(filepath: Path) -> tuple:
    raw = _parse_payloads_from_file(filepath)
    if raw.empty or "event_type" not in raw.columns:
        return pd.DataFrame(), pd.DataFrame()
    raw = raw[raw["event_type"].isin(["orderbook", "snapshot"])]
    if raw.empty:
        return pd.DataFrame(), pd.DataFrame()

    ts_col = raw["ts"] if "ts" in raw.columns else raw.get("_parquet_ts")
    if ts_col is None:
        return pd.DataFrame(), pd.DataFrame()
    raw["_ts"] = ts_col.apply(parse_timestamp)
    raw = raw.dropna(subset=["_ts"])
    if raw.empty:
        return pd.DataFrame(), pd.DataFrame()
    raw["_ts"] = pd.to_datetime(raw["_ts"], utc=True)

    results = []
    for mask_val, prefix in [("UP", "up"), ("DOWN", "down")]:
        mask = raw.get("symbol", pd.Series(dtype=str)) == mask_val
        subset = raw[mask]
        if subset.empty:
            results.append(pd.DataFrame())
            continue

        cols = {
            "_ts": "ts",
            "best_bid": f"{prefix}_bid",
            "best_ask": f"{prefix}_ask",
            "bid_size": f"{prefix}_bid_size",
            "ask_size": f"{prefix}_ask_size",
            "market_id": "market_id",
            "market_start": "market_start",
            "market_end": "market_end",
        }
        avail = {k: v for k, v in cols.items() if k in subset.columns}
        if "_ts" not in avail:
            results.append(pd.DataFrame())
            continue

        df = subset[list(avail.keys())].rename(columns=avail).copy()
        for c in [f"{prefix}_bid", f"{prefix}_ask", f"{prefix}_bid_size", f"{prefix}_ask_size"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
        df[f"{prefix}_mid"] = (df[f"{prefix}_bid"] + df[f"{prefix}_ask"]) / 2
        df["ts_500ms"] = df["ts"].dt.floor("500ms")
        group_cols = ["ts_500ms", "market_id"] if "market_id" in df.columns else ["ts_500ms"]
        df = df.groupby(group_cols).last().reset_index()
        df = df.drop(columns=["ts"], errors="ignore")
        results.append(df)

    return results[0], results[1]


def load_option_orderbook(data_path: Path, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    files = find_parquet_files(data_path, "polymarket-events", start_date, end_date)
    if not files:
        logger.warning("No polymarket-events files found")
        return pd.DataFrame()

    logger.info(f"Loading {len(files)} polymarket-events files...")
    up_chunks, down_chunks = [], []

    for i, f in enumerate(files):
        if (i + 1) % 20 == 0:
            logger.info(f"  Processing file {i+1}/{len(files)}...")
        up_df, down_df = _process_option_file(f)
        if not up_df.empty:
            up_chunks.append(up_df)
        if not down_df.empty:
            down_chunks.append(down_df)
        if len(up_chunks) > 50:
            combined = pd.concat(up_chunks, ignore_index=True)
            up_chunks = [combined.groupby(["ts_500ms", "market_id"]).last().reset_index()]
        if len(down_chunks) > 50:
            combined = pd.concat(down_chunks, ignore_index=True)
            down_chunks = [combined.groupby(["ts_500ms", "market_id"]).last().reset_index()]

    if not up_chunks and not down_chunks:
        return pd.DataFrame()

    df_up = pd.concat(up_chunks, ignore_index=True).groupby(["ts_500ms", "market_id"]).last().reset_index() if up_chunks else pd.DataFrame()
    df_down = pd.concat(down_chunks, ignore_index=True).groupby(["ts_500ms", "market_id"]).last().reset_index() if down_chunks else pd.DataFrame()
    del up_chunks, down_chunks
    gc.collect()

    if df_up.empty and df_down.empty:
        return pd.DataFrame()

    if df_up.empty:
        merged = df_down.rename(columns={"ts_500ms": "ts"})
    elif df_down.empty:
        merged = df_up.rename(columns={"ts_500ms": "ts"})
    else:
        up_cols = [c for c in df_up.columns if c.startswith("up_") or c in ["ts_500ms", "market_id", "market_start", "market_end"]]
        down_cols = [c for c in df_down.columns if c.startswith("down_") or c in ["ts_500ms", "market_id"]]
        merged = pd.merge(df_up[up_cols], df_down[down_cols], on=["ts_500ms", "market_id"], how="outer")
        merged = merged.sort_values(["market_id", "ts_500ms"]).reset_index(drop=True)
        merged = merged.rename(columns={"ts_500ms": "ts"})
        del df_up, df_down
        gc.collect()

    for col in ["up_bid", "up_ask", "up_mid", "up_bid_size", "up_ask_size",
                 "down_bid", "down_ask", "down_mid", "down_bid_size", "down_ask_size",
                 "market_start", "market_end"]:
        if col in merged.columns:
            merged[col] = merged.groupby("market_id")[col].ffill()

    merged = merged.dropna(subset=["up_mid", "down_mid"], how="all")
    logger.info(f"Loaded {len(merged)} option orderbook observations")
    return merged


# ---------------------------------------------------------------------------
# BTC price loader
# ---------------------------------------------------------------------------

def _process_btc_file(filepath: Path) -> pd.DataFrame:
    raw = _parse_payloads_from_file(filepath)
    if raw.empty:
        return pd.DataFrame()

    ts_src = raw["ts"] if "ts" in raw.columns else raw.get("_parquet_ts")
    if ts_src is None:
        return pd.DataFrame()

    raw["_ts"] = ts_src.apply(parse_timestamp)
    raw = raw.dropna(subset=["_ts"])
    if raw.empty or "price" not in raw.columns:
        return pd.DataFrame()

    raw["price"] = pd.to_numeric(raw["price"], errors="coerce")
    raw = raw[raw["price"] >= 1000]
    if raw.empty:
        return pd.DataFrame()

    if "symbol" in raw.columns:
        raw["symbol"] = raw["symbol"].fillna("").astype(str)
        raw = raw[raw["symbol"].str.upper().str.contains("BTC") | (raw["symbol"] == "")]
    if raw.empty:
        return pd.DataFrame()

    exchange_col = raw.get("exchange", raw.get("source", pd.Series("", index=raw.index)))
    raw["exchange"] = exchange_col.fillna("").astype(str).str.lower()
    raw["_ts"] = pd.to_datetime(raw["_ts"], utc=True)
    raw["ts_500ms"] = raw["_ts"].dt.floor("500ms")
    return raw[["ts_500ms", "exchange", "price"]].copy()


def load_btc_prices(data_path: Path, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    files = find_parquet_files(data_path, "exchange-prices", start_date, end_date)
    if not files:
        logger.warning("No exchange-prices files found")
        return pd.DataFrame()

    logger.info(f"Loading {len(files)} exchange-prices files...")
    chunks = []
    for i, f in enumerate(files):
        if (i + 1) % 20 == 0:
            logger.info(f"  Processing file {i+1}/{len(files)}...")
        chunk = _process_btc_file(f)
        if not chunk.empty:
            chunks.append(chunk)
        if len(chunks) > 50:
            combined = pd.concat(chunks, ignore_index=True)
            chunks = [combined.groupby(["ts_500ms", "exchange"]).last().reset_index()]

    if not chunks:
        return pd.DataFrame()

    df = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()

    pivot = df.pivot_table(index="ts_500ms", columns="exchange", values="price", aggfunc="last").reset_index()
    pivot.columns = ["ts"] + [f"btc_{col}" for col in pivot.columns[1:]]
    del df
    gc.collect()

    btc_cols = [c for c in pivot.columns if c.startswith("btc_")]
    for col in btc_cols:
        pivot[col] = pivot[col].ffill(limit=10)

    weights = {
        "btc_coinbase": 1.41, "btc_bitstamp": 1.26, "btc_kraken": 1.01,
        "btc_binance": 0.10, "btc_bybit": 0.10, "btc_okx": 0.10,
    }

    def weighted_btc(row):
        total_w, total_p = 0, 0
        for col, w in weights.items():
            if col in row.index and pd.notna(row[col]):
                total_p += row[col] * w
                total_w += w
        return total_p / total_w if total_w > 0 else np.nan

    pivot["btc_price"] = pivot.apply(weighted_btc, axis=1)
    pivot = pivot.dropna(subset=["btc_price"])
    logger.info(f"Loaded {len(pivot)} BTC price observations")
    return pivot


# ---------------------------------------------------------------------------
# Analytics loader
# ---------------------------------------------------------------------------

ANALYTICS_FIELDS = [
    "prob_up", "prob_down", "z_score",
    "vol_15m", "vol_60m",
    "minutes_elapsed", "minutes_remaining",
    "benchmark_price", "distance_usd", "distance_pct",
    "iv_up", "iv_down",
]


def _process_analytics_file(filepath: Path) -> pd.DataFrame:
    try:
        df = pd.read_parquet(filepath, columns=["ts", "payload"])
    except Exception as e:
        logger.warning(f"Error reading {filepath.name}: {e}")
        return pd.DataFrame()
    if df.empty:
        return pd.DataFrame()

    parsed = df["payload"].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
    valid_mask = parsed.apply(lambda x: isinstance(x, dict))
    if not valid_mask.any():
        return pd.DataFrame()

    parsed = parsed[valid_mask]
    parquet_ts = df.loc[valid_mask, "ts"]

    raw = pd.DataFrame(parsed.tolist())
    if "event_type" not in raw.columns:
        return pd.DataFrame()
    analytics_mask = raw["event_type"] == "analytics"
    if not analytics_mask.any():
        return pd.DataFrame()
    raw = raw[analytics_mask].reset_index(drop=True)
    parquet_ts = parquet_ts[analytics_mask.values].reset_index(drop=True)

    if "ts" in raw.columns:
        ts_series = raw["ts"].apply(parse_timestamp)
    else:
        ts_series = parquet_ts.apply(parse_timestamp)
    raw["_ts"] = ts_series
    raw = raw.dropna(subset=["_ts"])
    if raw.empty:
        return pd.DataFrame()

    result = pd.DataFrame()
    result["ts"] = pd.to_datetime(raw["_ts"], utc=True).dt.floor("500ms")
    for field in ANALYTICS_FIELDS:
        if field in raw.columns:
            result[field] = pd.to_numeric(raw[field], errors="coerce").fillna(0).values
        else:
            result[field] = 0
    result = result.groupby("ts").last().reset_index()
    return result


def load_analytics(data_path: Path, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    files = find_parquet_files(data_path, "analytics-events", start_date, end_date)
    if not files:
        logger.warning("No analytics-events files found")
        return pd.DataFrame()

    logger.info(f"Loading {len(files)} analytics-events files...")
    chunks = []
    for i, f in enumerate(files):
        if (i + 1) % 20 == 0:
            logger.info(f"  Processing file {i+1}/{len(files)}...")
        chunk = _process_analytics_file(f)
        if not chunk.empty:
            chunks.append(chunk)

    if not chunks:
        return pd.DataFrame()

    df = pd.concat(chunks, ignore_index=True)
    df = df.groupby("ts").last().reset_index()
    del chunks
    gc.collect()
    logger.info(f"Loaded {len(df)} analytics observations")
    return df


# ---------------------------------------------------------------------------
# Build unified dataset and split into per-market CSVs
# ---------------------------------------------------------------------------

def build_unified_dataset(data_path: Path, start_date: str, end_date: str) -> pd.DataFrame:
    """Merge option orderbook, BTC prices, and analytics on a 500ms grid."""
    logger.info(f"Building unified dataset: {start_date} to {end_date}")

    df_options = load_option_orderbook(data_path, start_date, end_date)
    gc.collect()
    if df_options.empty:
        logger.error("No option data found")
        return pd.DataFrame()

    df_options["ts"] = pd.to_datetime(df_options["ts"], utc=True)

    df_btc = load_btc_prices(data_path, start_date, end_date)
    gc.collect()
    if not df_btc.empty:
        df_btc["ts"] = pd.to_datetime(df_btc["ts"], utc=True)
        df_options = pd.merge_asof(
            df_options.sort_values("ts"),
            df_btc.sort_values("ts"),
            on="ts",
            tolerance=pd.Timedelta("2s"),
            direction="nearest",
        )
        del df_btc
        gc.collect()
        logger.info(f"Merged BTC prices: {df_options['btc_price'].notna().sum()} matched")

    df_analytics = load_analytics(data_path, start_date, end_date)
    gc.collect()
    if not df_analytics.empty:
        df_analytics["ts"] = pd.to_datetime(df_analytics["ts"], utc=True)
        df_options = pd.merge_asof(
            df_options.sort_values("ts"),
            df_analytics.sort_values("ts"),
            on="ts",
            tolerance=pd.Timedelta("2s"),
            direction="nearest",
        )
        del df_analytics
        gc.collect()

    df_options = df_options.sort_values("ts").reset_index(drop=True)
    logger.info(f"Unified dataset: {len(df_options)} rows, {len(df_options.columns)} columns")
    return df_options


def determine_outcome(group: pd.DataFrame) -> str:
    """Determine UP/DOWN outcome from BTC price at market start vs end."""
    btc_prices = group["btc_price"].dropna()
    if len(btc_prices) < 2:
        return "UNKNOWN"
    start_price = btc_prices.iloc[0]
    end_price = btc_prices.iloc[-1]
    return "UP" if end_price >= start_price else "DOWN"


def split_into_markets(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Split unified dataset into per-market CSVs and build an index."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if "market_end" not in df.columns:
        logger.error("No market_end column — cannot split into markets")
        return pd.DataFrame()

    # Parse market_start and market_end to datetime for grouping
    df["market_end_dt"] = df["market_end"].apply(parse_timestamp)
    df["market_start_dt"] = df["market_start"].apply(parse_timestamp) if "market_start" in df.columns else pd.NaT
    df = df.dropna(subset=["market_end_dt"])

    markets = df.groupby("market_end_dt")
    index_rows = []

    for market_end_dt, group in markets:
        group = group.sort_values("ts").reset_index(drop=True)
        if len(group) < 5:
            continue

        market_end_str = market_end_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        market_start_dt = group["market_start_dt"].dropna().iloc[0] if group["market_start_dt"].notna().any() else None
        market_start_str = market_start_dt.strftime("%Y-%m-%dT%H:%M:%SZ") if market_start_dt else ""

        outcome = determine_outcome(group)

        btc_prices = group["btc_price"].dropna()
        btc_start = btc_prices.iloc[0] if len(btc_prices) > 0 else np.nan
        btc_end = btc_prices.iloc[-1] if len(btc_prices) > 0 else np.nan

        # Compute derived columns
        if market_start_dt:
            group["minutes_elapsed"] = (group["ts"] - pd.Timestamp(market_start_dt)).dt.total_seconds() / 60
            group["minutes_remaining"] = (pd.Timestamp(market_end_dt) - group["ts"]).dt.total_seconds() / 60
        elif "minutes_elapsed" not in group.columns:
            first_ts = group["ts"].iloc[0]
            group["minutes_elapsed"] = (group["ts"] - first_ts).dt.total_seconds() / 60
            group["minutes_remaining"] = 15.0 - group["minutes_elapsed"]

        # BTC price changes
        if "btc_price" in group.columns:
            group["btc_price_change_1m"] = group["btc_price"].diff(periods=120).fillna(0)  # 120 * 500ms = 1 min
            group["btc_price_change_5m"] = group["btc_price"].diff(periods=600).fillna(0)  # 600 * 500ms = 5 min

        group["outcome"] = outcome

        # Select output columns
        out_cols = [
            "ts", "minutes_elapsed", "minutes_remaining",
            "up_bid", "up_ask", "up_mid", "down_bid", "down_ask", "down_mid",
            "up_bid_size", "up_ask_size", "down_bid_size", "down_ask_size",
            "btc_price", "btc_price_change_1m", "btc_price_change_5m",
            "vol_15m", "prob_up", "prob_down", "z_score",
            "outcome",
        ]
        out_cols = [c for c in out_cols if c in group.columns]
        out = group[out_cols].copy()

        # Format ts as ISO string
        out["ts"] = out["ts"].dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

        csv_name = f"{market_end_str.replace(':', '-')}.csv"
        csv_path = output_dir / csv_name
        out.to_csv(csv_path, index=False)

        index_rows.append({
            "market_end": market_end_str,
            "market_start": market_start_str,
            "outcome": outcome,
            "num_rows": len(out),
            "btc_start_price": round(btc_start, 2) if pd.notna(btc_start) else "",
            "btc_end_price": round(btc_end, 2) if pd.notna(btc_end) else "",
            "csv_file": csv_name,
        })

    if not index_rows:
        logger.error("No valid markets produced")
        return pd.DataFrame()

    index_df = pd.DataFrame(index_rows).sort_values("market_end").reset_index(drop=True)
    index_df.to_csv(output_dir / "market_index.csv", index=False)

    logger.info(f"Produced {len(index_df)} market CSVs in {output_dir}")
    logger.info(f"Outcomes: UP={sum(1 for r in index_rows if r['outcome']=='UP')}, "
                f"DOWN={sum(1 for r in index_rows if r['outcome']=='DOWN')}, "
                f"UNKNOWN={sum(1 for r in index_rows if r['outcome']=='UNKNOWN')}")
    return index_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Prepare per-market CSV files from S3 parquet data")
    parser.add_argument("--start-date", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--output", default="data/markets/", help="Output directory for market CSVs")
    parser.add_argument("--data-dir", default="data/raw/", help="Local directory for raw parquet files")
    parser.add_argument("--skip-download", action="store_true", help="Skip S3 download, use existing local files")
    args = parser.parse_args()

    data_path = Path(args.data_dir)
    output_path = Path(args.output)

    if not args.skip_download:
        fetch_s3_data(args.start_date, args.end_date, str(data_path))

    df = build_unified_dataset(data_path, args.start_date, args.end_date)
    if df.empty:
        logger.error("No data to process")
        sys.exit(1)

    index_df = split_into_markets(df, output_path)
    if index_df.empty:
        sys.exit(1)

    print(f"\nDone! {len(index_df)} markets written to {output_path}")
    print(f"Index: {output_path / 'market_index.csv'}")


if __name__ == "__main__":
    main()
