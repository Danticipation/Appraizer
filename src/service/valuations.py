from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger(__name__)


@dataclass
class ValuationResult:
    vin: str
    wholesale_value: float | None = None
    retail_value: float | None = None
    source: str = "none"
    raw: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class ManheimMMRClient:
    """Async client for Manheim MMR Valuations API (Cox Automotive).

    Registration: developer.manheim.com → API key setup.
    Batch endpoint: POST /apis/marketplace/valuations/batch
    Single endpoint: GET /apis/marketplace/valuations/{vin}
    """

    def __init__(self, api_key: str, base_url: str = "https://api.manheim.com") -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self._enabled = bool(api_key)

    async def get_valuation(self, vin: str, mileage: int | None = None) -> ValuationResult:
        if not self._enabled:
            return ValuationResult(vin=vin, error="mmr_not_configured")

        params: dict[str, Any] = {"format": "json"}
        if mileage is not None:
            params["mileage"] = mileage

        try:
            url = f"{self.base_url}/apis/marketplace/valuations/{vin}"
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(
                    url, params=params,
                    headers={"Authorization": f"Bearer {self.api_key}", "Accept": "application/json"},
                )
                resp.raise_for_status()
            data = resp.json()
            items = data.get("items") or data.get("valuations") or [data]
            best = items[0] if items else {}
            return ValuationResult(
                vin=vin,
                wholesale_value=_safe_float(best.get("wholesale") or best.get("mmr")),
                retail_value=_safe_float(best.get("retail") or best.get("adjustedPricing", {}).get("retail")),
                source="manheim_mmr",
                raw=best,
            )
        except Exception as exc:
            logger.warning("MMR lookup failed for %s: %s", vin, exc)
            return ValuationResult(vin=vin, error=str(exc))

    async def get_valuations_batch(self, vins: list[str]) -> list[ValuationResult]:
        if not self._enabled:
            return [ValuationResult(vin=v, error="mmr_not_configured") for v in vins]

        try:
            url = f"{self.base_url}/apis/marketplace/valuations/batch"
            payload = [{"vin": v} for v in vins]
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    url, json=payload,
                    headers={"Authorization": f"Bearer {self.api_key}", "Accept": "application/json"},
                )
                resp.raise_for_status()
            data = resp.json()
            items = data.get("items") or data.get("results") or []
            by_vin: dict[str, dict[str, Any]] = {}
            for item in items:
                v = item.get("vin", "")
                by_vin[v] = item

            results = []
            for vin in vins:
                item = by_vin.get(vin, {})
                results.append(ValuationResult(
                    vin=vin,
                    wholesale_value=_safe_float(item.get("wholesale") or item.get("mmr")),
                    retail_value=_safe_float(item.get("retail")),
                    source="manheim_mmr_batch" if item else "none",
                    raw=item,
                ))
            return results
        except Exception as exc:
            logger.warning("MMR batch lookup failed: %s", exc)
            return [ValuationResult(vin=v, error=str(exc)) for v in vins]


class ThirdPartyAuctionClient:
    """Fallback client for third-party unified auction APIs.

    Supports auction-api.app, carfast.express, or similar providers.
    Configure via base_url and api_key.
    """

    def __init__(self, api_key: str, base_url: str = "https://api.auction-api.app/v1") -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self._enabled = bool(api_key)

    async def get_sold_price(self, vin: str) -> ValuationResult:
        if not self._enabled:
            return ValuationResult(vin=vin, error="third_party_not_configured")

        try:
            url = f"{self.base_url}/sales/vin/{vin}"
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(
                    url,
                    headers={"X-API-Key": self.api_key, "Accept": "application/json"},
                )
                resp.raise_for_status()
            data = resp.json()
            sales = data.get("sales") or data.get("results") or [data]
            latest = sales[0] if sales else {}
            sold_price = _safe_float(latest.get("sold_price") or latest.get("final_price") or latest.get("close_price"))
            return ValuationResult(
                vin=vin,
                wholesale_value=sold_price,
                source="third_party_auction",
                raw=latest,
            )
        except Exception as exc:
            logger.warning("Third-party auction lookup failed for %s: %s", vin, exc)
            return ValuationResult(vin=vin, error=str(exc))

    async def get_sold_prices_batch(self, vins: list[str]) -> list[ValuationResult]:
        results = []
        for vin in vins:
            results.append(await self.get_sold_price(vin))
        return results


class ValuationsFacade:
    """Unified facade: MMR → third-party → returns best available."""

    def __init__(
        self,
        mmr_client: ManheimMMRClient,
        third_party_client: ThirdPartyAuctionClient,
    ) -> None:
        self.mmr = mmr_client
        self.third_party = third_party_client

    async def get_actual_price(self, vin: str, mileage: int | None = None) -> ValuationResult:
        result = await self.mmr.get_valuation(vin, mileage=mileage)
        if result.wholesale_value is not None:
            return result

        result = await self.third_party.get_sold_price(vin)
        if result.wholesale_value is not None:
            return result

        return ValuationResult(vin=vin, source="none", error="no_source_available")

    async def get_actual_prices_batch(self, vins: list[str]) -> list[ValuationResult]:
        mmr_results = await self.mmr.get_valuations_batch(vins)
        by_vin: dict[str, ValuationResult] = {r.vin: r for r in mmr_results}

        missing = [v for v in vins if by_vin.get(v) is None or by_vin[v].wholesale_value is None]
        if missing:
            fallback = await self.third_party.get_sold_prices_batch(missing)
            for r in fallback:
                if r.wholesale_value is not None:
                    by_vin[r.vin] = r

        return [by_vin.get(v, ValuationResult(vin=v, error="no_source_available")) for v in vins]


def _safe_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        f = float(val)
        return f if f > 0 else None
    except (TypeError, ValueError):
        return None
