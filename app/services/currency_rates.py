from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib import request as urlrequest



_DEFAULT_FX_API_URL = "https://api.frankfurter.dev/v2/rates?base=USD"
_DEFAULT_FX_CACHE_PATH = "app/resources/fx_rates_usd.json"
_DEFAULT_FX_CACHE_TTL_DAYS = 10


@dataclass(frozen=True)
class FxSnapshot:
    base: str
    rates: dict[str, float]
    provider_date: str | None
    fetched_at: datetime
    is_stale: bool = False


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _cache_path() -> Path:
    raw = os.getenv("FX_CACHE_PATH", _DEFAULT_FX_CACHE_PATH)
    return Path(raw)


def _cache_ttl_days() -> int:
    raw = os.getenv("FX_CACHE_TTL_DAYS")
    if not raw:
        return _DEFAULT_FX_CACHE_TTL_DAYS
    try:
        value = int(raw)
    except ValueError:
        return _DEFAULT_FX_CACHE_TTL_DAYS
    return max(1, value)


def _api_url() -> str:
    return os.getenv("FX_API_URL", _DEFAULT_FX_API_URL)


import requests


def _read_json_from_url(url: str, timeout: int = 20) -> dict[str, Any] | list[dict[str, Any]]:
    resp = requests.get(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "BookingAIAssistant/1.0",
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def _parse_snapshot(data: dict[str, Any], *, is_stale: bool) -> FxSnapshot:
    fetched_at_raw = data.get("fetched_at")
    fetched_at = _utc_now()
    if fetched_at_raw:
        try:
            fetched_at = datetime.fromisoformat(fetched_at_raw)
            if fetched_at.tzinfo is None:
                fetched_at = fetched_at.replace(tzinfo=timezone.utc)
        except ValueError:
            fetched_at = _utc_now()

    rates_raw = data.get("rates") or {}
    rates: dict[str, float] = {}
    for code, value in rates_raw.items():
        try:
            rates[str(code).upper()] = float(value)
        except (TypeError, ValueError):
            continue

    base = str(data.get("base") or "USD").upper()
    rates.setdefault(base, 1.0)

    provider_date = data.get("provider_date") or data.get("date")
    if provider_date is not None:
        provider_date = str(provider_date)

    return FxSnapshot(
        base=base,
        rates=rates,
        provider_date=provider_date,
        fetched_at=fetched_at,
        is_stale=is_stale,
    )


def _snapshot_is_fresh(snapshot: FxSnapshot) -> bool:
    age = _utc_now() - snapshot.fetched_at
    return age < timedelta(days=_cache_ttl_days())


def _load_cached_snapshot() -> FxSnapshot | None:
    path = _cache_path()
    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text())
    except Exception:
        return None

    snapshot = _parse_snapshot(data, is_stale=False)
    return snapshot


def _save_snapshot(snapshot: FxSnapshot) -> None:
    path = _cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "base": snapshot.base,
        "provider_date": snapshot.provider_date,
        "fetched_at": snapshot.fetched_at.isoformat(),
        "rates": snapshot.rates,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def _fetch_latest_snapshot() -> FxSnapshot:
    payload = _read_json_from_url(_api_url())

    rates: dict[str, float] = {}
    provider_date: str | None = None
    base = "USD"

    for row in payload:
        quote = row.get("quote")
        rate = row.get("rate")

        if provider_date is None and row.get("date") is not None:
            provider_date = str(row.get("date"))
        if row.get("base") is not None:
            base = str(row.get("base")).upper()

        if quote is None or rate is None:
            continue

        try:
            rates[str(quote).upper()] = float(rate)
        except (TypeError, ValueError):
            continue

    rates.setdefault(base, 1.0)

    return FxSnapshot(
        base=base,
        rates=rates,
        provider_date=provider_date,
        fetched_at=_utc_now(),
        is_stale=False,
    )


def get_fx_snapshot() -> FxSnapshot | None:
    print("FX DEBUG: get_fx_snapshot called")
    print("FX DEBUG: cache path =", _cache_path().resolve())
    cached = _load_cached_snapshot()
    print("FX DEBUG: cached exists =", cached is not None)

    if cached and _snapshot_is_fresh(cached):
        print("FX DEBUG: cached base =", cached.base)
        print("FX DEBUG: cached rates count =", len(cached.rates))
        print("FX DEBUG: cached fresh =", _snapshot_is_fresh(cached))
        return cached

    try:
        fresh = _fetch_latest_snapshot()
    except Exception as e:
        print("FX DEBUG: fetch failed:", repr(e))
        if cached:
            return FxSnapshot(
                base=cached.base,
                rates=cached.rates,
                provider_date=cached.provider_date,
                fetched_at=cached.fetched_at,
                is_stale=True,
            )
        return None

    _save_snapshot(fresh)
    return fresh


def convert_amount_to_usd(amount: float, currency: str | None) -> tuple[float | None, FxSnapshot | None]:
    if currency is None:
        return None, None

    code = str(currency).strip().upper()
    if not code:
        return None, None

    if code == "USD":
        return float(amount), FxSnapshot(
            base="USD",
            rates={"USD": 1.0},
            provider_date=None,
            fetched_at=_utc_now(),
            is_stale=False,
        )

    snapshot = get_fx_snapshot()
    if snapshot is None:
        return None, None

    quote = snapshot.rates.get(code)
    if quote is None or quote == 0:
        return None, snapshot

    usd_amount = float(amount) / float(quote)
    return usd_amount, snapshot