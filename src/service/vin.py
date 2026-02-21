from __future__ import annotations

from typing import Any

import httpx

from service.storage import RedisCache


_YEAR_CODE_MAP = {
    "Y": 2000,
    "1": 2001,
    "2": 2002,
    "3": 2003,
    "4": 2004,
    "5": 2005,
    "6": 2006,
    "7": 2007,
    "8": 2008,
    "9": 2009,
    "A": 2010,
    "B": 2011,
    "C": 2012,
    "D": 2013,
    "E": 2014,
    "F": 2015,
    "G": 2016,
    "H": 2017,
    "J": 2018,
    "K": 2019,
    "L": 2020,
    "M": 2021,
    "N": 2022,
    "P": 2023,
    "R": 2024,
    "S": 2025,
    "T": 2026,
}


class VinDecoder:
    def __init__(self, cache: RedisCache, base_url: str, ttl_seconds: int) -> None:
        self.cache = cache
        self.base_url = base_url.rstrip("/")
        self.ttl_seconds = ttl_seconds

    def _fallback_decode(self, vin: str) -> dict[str, Any]:
        year = _YEAR_CODE_MAP.get(vin[9], 0) if len(vin) >= 10 else 0
        return {
            "vin": vin,
            "model_year": year,
            "make": "",
            "model": "",
            "trim": "",
            "engine": "",
            "decode_source": "fallback",
        }

    async def decode(self, vin: str) -> dict[str, Any]:
        cache_key = f"vin_decode:{vin}"
        cached = await self.cache.get_json(cache_key)
        if cached is not None:
            return cached

        try:
            url = f"{self.base_url}/DecodeVinValues/{vin}"
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.get(url, params={"format": "json"})
                resp.raise_for_status()
            payload = resp.json()
            row = (payload.get("Results") or [{}])[0]
            decoded = {
                "vin": vin,
                "model_year": int(row.get("ModelYear") or 0),
                "make": row.get("Make") or "",
                "model": row.get("Model") or "",
                "trim": row.get("Trim") or "",
                "engine": row.get("EngineModel") or "",
                "decode_source": "nhtsa",
            }
        except Exception:
            decoded = self._fallback_decode(vin)

        await self.cache.set_json(cache_key, decoded, ttl_seconds=self.ttl_seconds)
        return decoded

    async def decode_batch(self, vins: list[str], chunk_size: int = 50) -> list[dict[str, Any]]:
        """Batch decode using NHTSA DecodeVINValuesBatch endpoint for efficiency."""
        results: list[dict[str, Any]] = []
        uncached_vins: list[str] = []
        cached_map: dict[str, dict[str, Any]] = {}

        for vin in vins:
            cached = await self.cache.get_json(f"vin_decode:{vin}")
            if cached is not None:
                cached_map[vin] = cached
            else:
                uncached_vins.append(vin)

        for i in range(0, len(uncached_vins), chunk_size):
            chunk = uncached_vins[i : i + chunk_size]
            batch_decoded = await self._batch_request(chunk)
            for vin, decoded in zip(chunk, batch_decoded):
                await self.cache.set_json(f"vin_decode:{vin}", decoded, ttl_seconds=self.ttl_seconds)

            results_for_chunk = batch_decoded
            for vin, decoded in zip(chunk, results_for_chunk):
                cached_map[vin] = decoded

        return [cached_map.get(vin, self._fallback_decode(vin)) for vin in vins]

    async def _batch_request(self, vins: list[str]) -> list[dict[str, Any]]:
        vin_string = ";".join(vins)
        try:
            url = f"{self.base_url}/DecodeVINValuesBatch/"
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.post(url, data={"format": "json", "data": vin_string})
                resp.raise_for_status()
            payload = resp.json()
            rows = payload.get("Results") or []
            decoded_map: dict[str, dict[str, Any]] = {}
            for row in rows:
                v = row.get("VIN", "")
                decoded_map[v] = {
                    "vin": v,
                    "model_year": int(row.get("ModelYear") or 0),
                    "make": row.get("Make") or "",
                    "model": row.get("Model") or "",
                    "trim": row.get("Trim") or "",
                    "engine": row.get("EngineModel") or "",
                    "decode_source": "nhtsa_batch",
                }
            return [decoded_map.get(vin, self._fallback_decode(vin)) for vin in vins]
        except Exception:
            return [self._fallback_decode(vin) for vin in vins]
