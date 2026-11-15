#!/usr/bin/env python3
"""Fetch a bounded NASA POWER reference series and summarize the aggregate cache.

This module only reads the aggregate state-day parquet in inputs/public. It does
not access SIM municipality records or any person-level source.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from repro_paths import source_root, run_root
PROJECT_ROOT = source_root()
PUBLIC_DIR = PROJECT_ROOT / "inputs" / "public"
PANEL_PATH = PUBLIC_DIR / "bra_state_day.parquet"
REFERENCE_OUTPUT = run_root() / "public_reference"
REFERENCE_OUTPUT.mkdir(parents=True, exist_ok=True)
POWER_PATH = REFERENCE_OUTPUT / "nasa_power_br_uf35_2015_daily.json"
QC_PATH = REFERENCE_OUTPUT / "bra_state_day_qc.json"
PROVENANCE_PATH = REFERENCE_OUTPUT / "provenance.json"

POWER_PARAMS = {
    "parameters": "T2M",
    "community": "RE",
    "longitude": -46.6,
    "latitude": -23.5,
    "start": "20150101",
    "end": "20151231",
    "format": "JSON",
}
POWER_BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
LEGACY_SERVER_PATH = (
    "/data/admin/fengshuo/conformal_attribution_20261107/"
    "incoming/bra_daily/bra_state_day.parquet"
)
LEGACY_SHA256 = "fc972f919a5e5d51280a4093b4992d2b5ea7add6c51c2193b741d412825a42d3"
METHOD_REFERENCE_REPO = "gasparrini/2015_gasparrini_Lancet_Rcodedata"
METHOD_REFERENCE_COMMIT = "7c3e03c425529d2f210b1659a01bbc582448b9c0"
METHOD_REFERENCE_DIR = PUBLIC_DIR / "method_reference" / "gasparrini_2015"
METHOD_REFERENCE_FILES = ("README.md", "attrdl.R")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_power_url() -> str:
    return f"{POWER_BASE_URL}?{urllib.parse.urlencode(POWER_PARAMS)}"


def fetch_power_reference(force: bool = False, tries: int = 5) -> dict:
    """Fetch one 2015 daily T2M series at São Paulo capital; keep response bytes."""
    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    if POWER_PATH.exists() and not force:
        payload = POWER_PATH.read_bytes()
        recorded = {}
        if PROVENANCE_PATH.exists():
            try:
                old_provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))
                old_power = old_provenance.get("temperature_reference", {})
                if old_power.get("sha256") == hashlib.sha256(payload).hexdigest() and old_power.get("url") == build_power_url():
                    recorded = old_power
            except (OSError, json.JSONDecodeError):
                recorded = {}
        return {
            **recorded,
            "path": str(POWER_PATH),
            "url": build_power_url(),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
            "reused_existing_response": True,
            "payload": payload,
        }

    request = urllib.request.Request(
        build_power_url(),
        headers={"User-Agent": "aggregate-health-temperature-provenance/1.0"},
    )
    last_error: Exception | None = None
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = response.read()
                if response.status != 200:
                    raise RuntimeError(f"NASA POWER returned HTTP {response.status}")
                # Validate as JSON before storing the original response bytes.
                json.loads(payload.decode("utf-8"))
                temporary = POWER_PATH.with_suffix(POWER_PATH.suffix + ".part")
                temporary.write_bytes(payload)
                temporary.replace(POWER_PATH)
                return {
                    "path": str(POWER_PATH),
                    "url": build_power_url(),
                    "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                    "http_status": int(response.status),
                    "content_type": response.headers.get("Content-Type"),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "bytes": len(payload),
                    "reused_existing_response": False,
                    "payload": payload,
                }
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, ValueError) as exc:
            last_error = exc
            if attempt + 1 < tries:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"NASA POWER download failed after {tries} attempts: {last_error}")


def ensure_method_reference_sources(tries: int = 3) -> list[dict]:
    """Provide the pinned reference link; unlicensed third-party source is not redistributed."""
    return [{"source_url": f"https://github.com/{METHOD_REFERENCE_REPO}/tree/{METHOD_REFERENCE_COMMIT}",
             "status": "reference link only; not needed for the Python analysis"}]


def summarize_panel(path: Path) -> dict:
    """Summarize only the aggregate state-day input, without exposing row values."""
    data = pd.read_parquet(path)
    required = {"state", "date", "deaths", "tmean"}
    if set(data.columns) != required:
        raise ValueError(f"Unexpected aggregate schema: {list(data.columns)}")
    data["state"] = data["state"].astype(str).str.zfill(2)
    data["date"] = pd.to_datetime(data["date"])

    start = data["date"].min()
    end = data["date"].max()
    calendar_days = int((end - start).days + 1)
    states = sorted(data["state"].unique().tolist())
    complete_grid = len(states) * calendar_days
    by_state = data.groupby("state").agg(
        rows=("date", "size"),
        distinct_days=("date", "nunique"),
        date_start=("date", "min"),
        date_end=("date", "max"),
    )
    missing_by_state = {
        str(state): int(calendar_days - row.distinct_days)
        for state, row in by_state.iterrows()
    }
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "schema": {
            "state": str(data["state"].dtype),
            "date": str(data["date"].dtype),
            "deaths": str(data["deaths"].dtype),
            "tmean": str(data["tmean"].dtype),
        },
        "rows": int(len(data)),
        "state_count": int(data["state"].nunique()),
        "state_codes": states,
        "date_start": start.strftime("%Y-%m-%d"),
        "date_end": end.strftime("%Y-%m-%d"),
        "distinct_dates": int(data["date"].nunique()),
        "years": sorted(int(y) for y in data["date"].dt.year.unique()),
        "missing_by_column": {str(k): int(v) for k, v in data.isna().sum().items()},
        "duplicate_rows": int(data.duplicated().sum()),
        "duplicate_state_date_keys": int(data.duplicated(["state", "date"]).sum()),
        "zero_death_rows": int((data["deaths"] == 0).sum()),
        "negative_death_rows": int((data["deaths"] < 0).sum()),
        "tmean_min_c": float(data["tmean"].min()),
        "tmean_max_c": float(data["tmean"].max()),
        "expected_complete_state_days": int(complete_grid),
        "absent_state_day_keys": int(complete_grid - data[["state", "date"]].drop_duplicates().shape[0]),
        "per_state_rows_min": int(by_state["rows"].min()),
        "per_state_rows_max": int(by_state["rows"].max()),
        "per_state_missing_days": missing_by_state,
        "temperature_geography": (
            "NASA POWER daily T2M at one capital-point coordinate per Brazilian UF; "
            "the prior acquisition script used the nearest POWER grid cell, not "
            "an area-weighted state mean."
        ),
    }


def compare_power_to_cached_panel(payload: bytes, panel_path: Path) -> dict:
    """Compare fresh POWER 2015 T2M to the cached São Paulo UF 35 series."""
    response = json.loads(payload.decode("utf-8"))
    values = response["properties"]["parameter"]["T2M"]
    fresh = pd.DataFrame(
        [(pd.to_datetime(day, format="%Y%m%d"), float(value))
         for day, value in values.items()
         if value is not None and float(value) > -100],
        columns=["date", "tmean_power_fresh"],
    )
    panel = pd.read_parquet(panel_path, columns=["state", "date", "tmean"])
    panel["state"] = panel["state"].astype(str).str.zfill(2)
    panel["date"] = pd.to_datetime(panel["date"])
    cached = panel.loc[
        panel["state"].eq("35") & panel["date"].dt.year.eq(2015),
        ["date", "tmean"],
    ]
    merged = fresh.merge(cached, on="date", how="inner")
    differences = merged["tmean_power_fresh"] - merged["tmean"]
    return {
        "uf_code": "35",
        "location": "São Paulo capital reference point",
        "longitude": POWER_PARAMS["longitude"],
        "latitude": POWER_PARAMS["latitude"],
        "period": "2015-01-01 through 2015-12-31",
        "fresh_valid_days": int(len(fresh)),
        "cached_days": int(len(cached)),
        "matched_days": int(len(merged)),
        "exactly_equal_days": int((differences.abs() < 1e-12).sum()),
        "mean_absolute_difference_c": float(differences.abs().mean()) if len(merged) else None,
        "max_absolute_difference_c": float(differences.abs().max()) if len(merged) else None,
        "interpretation": (
            "Descriptive source-provenance comparison for one capital point only; "
            "not a spatial validation or a temperature-error bound."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh-power",
        action="store_true",
        help="Fetch a new NASA POWER response even if the saved response exists.",
    )
    args = parser.parse_args()

    if not PANEL_PATH.exists():
        raise FileNotFoundError(f"Aggregate panel not found: {PANEL_PATH}")
    panel_qc = summarize_panel(PANEL_PATH)
    if panel_qc["sha256"].lower() != LEGACY_SHA256:
        raise ValueError("Cached aggregate panel hash differs from the recorded server source.")

    power_meta = fetch_power_reference(force=args.refresh_power)
    payload = power_meta.pop("payload")
    method_reference_files = ensure_method_reference_sources()
    comparison = compare_power_to_cached_panel(payload, PANEL_PATH)
    qc = {"panel": panel_qc, "fresh_power_comparison": comparison}
    QC_PATH.write_text(json.dumps(qc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    provenance = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "health_panel": {
            "file": "inputs/public/bra_state_day.parquet",
            "status": "legacy aggregate cache, transferred byte-for-byte from existing project storage",
            "server_source_path": LEGACY_SERVER_PATH,
            "sha256": panel_qc["sha256"],
            "bytes": panel_qc["bytes"],
            "derivation_note": (
                "Existing acquisition script documents SIM municipality-day deaths "
                "summed to UF-day and joined to NASA POWER T2M. Source microdata was "
                "not opened by this retrieval task."
            ),
        },
        "temperature_reference": {
            **power_meta,
            "parameter": "T2M",
            "units": "degrees Celsius",
            "temporal_resolution": "daily",
            "point": "UF 35 capital reference point, São Paulo",
            "query_parameters": POWER_PARAMS,
            "official_documentation": "https://power.larc.nasa.gov/docs/services/api/temporal/daily/",
        },
        "method_reference": {
            "repository": f"https://github.com/{METHOD_REFERENCE_REPO}",
            "commit": METHOD_REFERENCE_COMMIT,
            "files": method_reference_files,
            "note": (
                "The original attributable-risk helper is linked for reference only; "
                "it is not redistributed because the source repository has no LICENSE file."
            ),
        },
    }
    PROVENANCE_PATH.write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(qc, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()

