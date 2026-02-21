from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import pandas as pd
from sqlalchemy import JSON, Boolean, DateTime, Float, Integer, MetaData, String, Table, Column, insert, select, update
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

try:
    import redis.asyncio as redis
except ModuleNotFoundError:  # pragma: no cover
    redis = None


metadata = MetaData()

appraisals_table = Table(
    "appraisals",
    metadata,
    Column("id", String(36), primary_key=True),
    Column("vin", String(32), nullable=False, index=True),
    Column("input_json", JSON, nullable=False),
    Column("vin_json", JSON, nullable=False, default=dict),
    Column("predicted_value", Float, nullable=False),
    Column("confidence", Float, nullable=False),
    Column("interval_low", Float, nullable=False),
    Column("interval_high", Float, nullable=False),
    Column("population_segment", String(64), nullable=False),
    Column("region", String(64), nullable=False),
    Column("segment", String(64), nullable=False),
    Column("model_version", String(64), nullable=False),
    Column("reviewed_flag", Boolean, nullable=False, default=False),
    Column("route_to_manual", Boolean, nullable=False, default=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)

model_versions_table = Table(
    "model_versions",
    metadata,
    Column("id", String(36), primary_key=True),
    Column("segment", String(64), nullable=False),
    Column("region", String(64), nullable=False),
    Column("population_segment", String(64), nullable=False),
    Column("version", String(64), nullable=False),
    Column("trained_at", DateTime(timezone=True), nullable=False),
    Column("metrics_json", JSON, nullable=False),
    Column("drift_metrics_json", JSON, nullable=True),
    Column("shadow_mae", Float, nullable=True),
    Column("promotion_status", String(32), nullable=True, default="pending"),
    Column("is_active", Boolean, nullable=False, default=False),
)

feedback_actuals_table = Table(
    "feedback_actuals",
    metadata,
    Column("id", String(36), primary_key=True),
    Column("appraisal_id", String(36), nullable=True, index=True),
    Column("vin", String(32), nullable=False, index=True),
    Column("region", String(64), nullable=False),
    Column("segment", String(64), nullable=False),
    Column("actual_close_price", Float, nullable=False),
    Column("source", String(64), nullable=False, default="manual"),
    Column("source_json", JSON, nullable=False, default=dict),
    Column("created_at", DateTime(timezone=True), nullable=False),
)

shadow_metrics_table = Table(
    "shadow_metrics",
    metadata,
    Column("id", String(36), primary_key=True),
    Column("model_version", String(64), nullable=False, index=True),
    Column("segment", String(64), nullable=False),
    Column("region", String(64), nullable=False),
    Column("population_segment", String(64), nullable=False),
    Column("mae", Float, nullable=False),
    Column("rmse", Float, nullable=False),
    Column("bias", Float, nullable=False),
    Column("coverage_pct", Float, nullable=False),
    Column("calibration_error", Float, nullable=False),
    Column("sample_count", Integer, nullable=False),
    Column("window_start", DateTime(timezone=True), nullable=False),
    Column("window_end", DateTime(timezone=True), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("details_json", JSON, nullable=False, default=dict),
)

canary_state_table = Table(
    "canary_state",
    metadata,
    Column("id", String(36), primary_key=True),
    Column("candidate_version", String(64), nullable=False, index=True),
    Column("baseline_version", String(64), nullable=False),
    Column("segment", String(64), nullable=False),
    Column("region", String(64), nullable=False),
    Column("traffic_pct", Float, nullable=False, default=0.10),
    Column("status", String(32), nullable=False, default="shadow"),
    Column("started_at", DateTime(timezone=True), nullable=False),
    Column("promoted_at", DateTime(timezone=True), nullable=True),
    Column("metrics_json", JSON, nullable=False, default=dict),
)


class RedisCache:
    def __init__(self, redis_url: str, namespace: str = "appraisal") -> None:
        self.redis_url = redis_url
        self.namespace = namespace
        self._client: Any = None
        self._mem: dict[str, str] = {}
        self._expiry: dict[str, float] = {}

    def _build_key(self, key: str) -> str:
        return f"{self.namespace}:{key}"

    async def connect(self) -> None:
        if redis is None:
            return
        self._client = redis.from_url(self.redis_url, decode_responses=True)
        try:
            await asyncio.wait_for(self._client.ping(), timeout=0.75)
        except Exception:
            self._client = None

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()

    async def ping(self) -> bool:
        if self._client is None:
            return False
        try:
            return bool(await asyncio.wait_for(self._client.ping(), timeout=0.75))
        except Exception:
            return False

    async def get_json(self, key: str) -> dict[str, Any] | None:
        full_key = self._build_key(key)
        if self._client is not None:
            try:
                raw = await self._client.get(full_key)
                return None if raw is None else json.loads(raw)
            except Exception:
                return None
        now = asyncio.get_event_loop().time()
        if full_key in self._expiry and now > self._expiry[full_key]:
            self._mem.pop(full_key, None)
            self._expiry.pop(full_key, None)
            return None
        raw = self._mem.get(full_key)
        return None if raw is None else json.loads(raw)

    async def set_json(self, key: str, value: dict[str, Any], ttl_seconds: int) -> None:
        full_key = self._build_key(key)
        payload = json.dumps(value)
        if self._client is not None:
            try:
                await self._client.set(full_key, payload, ex=ttl_seconds)
                return
            except Exception:
                pass
        self._mem[full_key] = payload
        self._expiry[full_key] = asyncio.get_event_loop().time() + ttl_seconds


class PostgresStore:
    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self.engine: AsyncEngine | None = None
        self._fallback_mode = False
        self._mem_appraisals: list[dict[str, Any]] = []
        self._mem_model_versions: list[dict[str, Any]] = []
        self._mem_feedback: list[dict[str, Any]] = []
        self._mem_shadow_metrics: list[dict[str, Any]] = []
        self._mem_canary_state: list[dict[str, Any]] = []

    async def connect(self) -> None:
        try:
            self.engine = create_async_engine(self.dsn, future=True)
            await self.init_schema()
        except Exception:
            self._fallback_mode = True
            self.engine = None

    async def close(self) -> None:
        if self.engine is not None:
            await self.engine.dispose()

    async def ping(self) -> bool:
        if self.engine is None:
            return False
        if self._fallback_mode:
            return False
        try:
            async with self.engine.connect() as conn:
                await conn.execute(select(1))
            return True
        except Exception:
            return False

    async def init_schema(self) -> None:
        if self.engine is None:
            return
        async with self.engine.begin() as conn:
            await conn.run_sync(metadata.create_all)

    async def insert_appraisal(self, record: dict[str, Any]) -> str:
        appraisal_id = str(uuid4())
        row = {
            "id": appraisal_id,
            "vin": record["vin"],
            "input_json": record["input_json"],
            "vin_json": record.get("vin_json", {}),
            "predicted_value": float(record["predicted_value"]),
            "confidence": float(record["confidence"]),
            "interval_low": float(record["interval_low"]),
            "interval_high": float(record["interval_high"]),
            "population_segment": record["population_segment"],
            "region": record["region"],
            "segment": record["segment"],
            "model_version": record.get("model_version", "baseline-v1"),
            "reviewed_flag": bool(record.get("reviewed_flag", False)),
            "route_to_manual": bool(record.get("route_to_manual", False)),
            "created_at": datetime.now(timezone.utc),
        }
        if self.engine is None:
            self._mem_appraisals.append(row)
            return appraisal_id
        async with self.engine.begin() as conn:
            await conn.execute(insert(appraisals_table).values(**row))
        return appraisal_id

    async def insert_model_version(
        self,
        *,
        segment: str,
        region: str,
        population_segment: str,
        version: str,
        metrics_json: dict[str, Any],
    ) -> str:
        row_id = str(uuid4())
        row = {
            "id": row_id,
            "segment": segment,
            "region": region,
            "population_segment": population_segment,
            "version": version,
            "trained_at": datetime.now(timezone.utc),
            "metrics_json": metrics_json,
        }
        if self.engine is None:
            self._mem_model_versions.append(row)
            return row_id
        async with self.engine.begin() as conn:
            await conn.execute(insert(model_versions_table).values(**row))
        return row_id

    async def insert_feedback_actual(
        self,
        *,
        vin: str,
        region: str,
        segment: str,
        actual_close_price: float,
        appraisal_id: str | None = None,
        source: str = "manual",
        source_json: dict[str, Any] | None = None,
    ) -> str:
        row_id = str(uuid4())
        row = {
            "id": row_id,
            "appraisal_id": appraisal_id,
            "vin": vin,
            "region": region,
            "segment": segment,
            "actual_close_price": float(actual_close_price),
            "source": source,
            "source_json": source_json or {},
            "created_at": datetime.now(timezone.utc),
        }
        if self.engine is None:
            self._mem_feedback.append(row)
            return row_id
        async with self.engine.begin() as conn:
            await conn.execute(insert(feedback_actuals_table).values(**row))
        return row_id

    async def get_appraisal_by_id(self, appraisal_id: str) -> dict[str, Any] | None:
        if self.engine is None:
            for ap in self._mem_appraisals:
                if ap["id"] == appraisal_id:
                    return ap
            return None
        stmt = select(appraisals_table).where(appraisals_table.c.id == appraisal_id)
        async with self.engine.connect() as conn:
            row = (await conn.execute(stmt)).first()
        return dict(row._mapping) if row else None

    async def fetch_appraisals_with_actuals(
        self,
        window_start: datetime,
        window_end: datetime,
        limit: int = 10000,
    ) -> pd.DataFrame:
        if self.engine is None:
            rows: list[dict[str, Any]] = []
            fb_by_vin: dict[str, dict[str, Any]] = {}
            for fb in self._mem_feedback:
                fb_by_vin[fb["vin"]] = fb
            for ap in self._mem_appraisals:
                if ap["created_at"] < window_start or ap["created_at"] > window_end:
                    continue
                fb = fb_by_vin.get(ap["vin"])
                if fb is None:
                    continue
                rows.append({
                    "vin": ap["vin"],
                    "predicted_value": ap["predicted_value"],
                    "actual_close_price": fb["actual_close_price"],
                    "confidence": ap["confidence"],
                    "interval_low": ap["interval_low"],
                    "interval_high": ap["interval_high"],
                    "model_version": ap["model_version"],
                    "region": ap["region"],
                    "segment": ap["segment"],
                    "population_segment": ap["population_segment"],
                    "route_to_manual": ap["route_to_manual"],
                })
            return pd.DataFrame(rows[:limit])

        j = appraisals_table.join(
            feedback_actuals_table,
            appraisals_table.c.vin == feedback_actuals_table.c.vin,
        )
        stmt = (
            select(
                appraisals_table.c.vin,
                appraisals_table.c.predicted_value,
                feedback_actuals_table.c.actual_close_price,
                appraisals_table.c.confidence,
                appraisals_table.c.interval_low,
                appraisals_table.c.interval_high,
                appraisals_table.c.model_version,
                appraisals_table.c.region,
                appraisals_table.c.segment,
                appraisals_table.c.population_segment,
                appraisals_table.c.route_to_manual,
            )
            .select_from(j)
            .where(appraisals_table.c.created_at >= window_start)
            .where(appraisals_table.c.created_at <= window_end)
            .order_by(appraisals_table.c.created_at.desc())
            .limit(limit)
        )
        async with self.engine.connect() as conn:
            result = (await conn.execute(stmt)).all()
        return pd.DataFrame(
            [dict(r._mapping) for r in result],
            columns=[
                "vin", "predicted_value", "actual_close_price", "confidence",
                "interval_low", "interval_high", "model_version", "region",
                "segment", "population_segment", "route_to_manual",
            ],
        )

    async def insert_shadow_metric(self, record: dict[str, Any]) -> str:
        row_id = str(uuid4())
        row = {
            "id": row_id,
            "model_version": record["model_version"],
            "segment": record["segment"],
            "region": record["region"],
            "population_segment": record["population_segment"],
            "mae": float(record["mae"]),
            "rmse": float(record["rmse"]),
            "bias": float(record["bias"]),
            "coverage_pct": float(record["coverage_pct"]),
            "calibration_error": float(record["calibration_error"]),
            "sample_count": int(record["sample_count"]),
            "window_start": record["window_start"],
            "window_end": record["window_end"],
            "created_at": datetime.now(timezone.utc),
            "details_json": record.get("details_json", {}),
        }
        if self.engine is None:
            self._mem_shadow_metrics.append(row)
            return row_id
        async with self.engine.begin() as conn:
            await conn.execute(insert(shadow_metrics_table).values(**row))
        return row_id

    async def fetch_shadow_metrics(
        self, model_version: str, limit: int = 50,
    ) -> list[dict[str, Any]]:
        if self.engine is None:
            return [r for r in self._mem_shadow_metrics if r["model_version"] == model_version][:limit]
        stmt = (
            select(shadow_metrics_table)
            .where(shadow_metrics_table.c.model_version == model_version)
            .order_by(shadow_metrics_table.c.created_at.desc())
            .limit(limit)
        )
        async with self.engine.connect() as conn:
            rows = (await conn.execute(stmt)).all()
        return [dict(r._mapping) for r in rows]

    async def upsert_canary_state(self, record: dict[str, Any]) -> str:
        row_id = record.get("id") or str(uuid4())
        row = {
            "id": row_id,
            "candidate_version": record["candidate_version"],
            "baseline_version": record["baseline_version"],
            "segment": record["segment"],
            "region": record["region"],
            "traffic_pct": float(record.get("traffic_pct", 0.10)),
            "status": record.get("status", "shadow"),
            "started_at": record.get("started_at", datetime.now(timezone.utc)),
            "promoted_at": record.get("promoted_at"),
            "metrics_json": record.get("metrics_json", {}),
        }
        if self.engine is None:
            for i, existing in enumerate(self._mem_canary_state):
                if existing["candidate_version"] == row["candidate_version"]:
                    self._mem_canary_state[i] = row
                    return row_id
            self._mem_canary_state.append(row)
            return row_id
        async with self.engine.begin() as conn:
            await conn.execute(insert(canary_state_table).values(**row))
        return row_id

    async def get_active_canaries(self) -> list[dict[str, Any]]:
        if self.engine is None:
            return [r for r in self._mem_canary_state if r["status"] in ("shadow", "canary")]
        stmt = select(canary_state_table).where(
            canary_state_table.c.status.in_(["shadow", "canary"])
        )
        async with self.engine.connect() as conn:
            rows = (await conn.execute(stmt)).all()
        return [dict(r._mapping) for r in rows]

    async def update_model_promotion(
        self, version: str, *, promotion_status: str, shadow_mae: float | None = None, drift_metrics_json: dict[str, Any] | None = None,
    ) -> None:
        if self.engine is None:
            for mv in self._mem_model_versions:
                if mv["version"] == version:
                    mv["promotion_status"] = promotion_status
                    if shadow_mae is not None:
                        mv["shadow_mae"] = shadow_mae
                    if drift_metrics_json is not None:
                        mv["drift_metrics_json"] = drift_metrics_json
            return
        values: dict[str, Any] = {"promotion_status": promotion_status}
        if shadow_mae is not None:
            values["shadow_mae"] = shadow_mae
        if drift_metrics_json is not None:
            values["drift_metrics_json"] = drift_metrics_json
        async with self.engine.begin() as conn:
            await conn.execute(
                update(model_versions_table)
                .where(model_versions_table.c.version == version)
                .values(**values)
            )

    async def set_active_model(self, version: str, segment: str, region: str) -> None:
        if self.engine is None:
            for mv in self._mem_model_versions:
                match = mv["segment"] == segment and mv["region"] == region
                mv["is_active"] = match and mv["version"] == version
            return
        async with self.engine.begin() as conn:
            await conn.execute(
                update(model_versions_table)
                .where(model_versions_table.c.segment == segment)
                .where(model_versions_table.c.region == region)
                .values(is_active=False)
            )
            await conn.execute(
                update(model_versions_table)
                .where(model_versions_table.c.version == version)
                .values(is_active=True)
            )

    async def get_active_models(self) -> list[dict[str, Any]]:
        if self.engine is None:
            return [mv for mv in self._mem_model_versions if mv.get("is_active")]
        stmt = select(model_versions_table).where(model_versions_table.c.is_active == True)  # noqa: E712
        async with self.engine.connect() as conn:
            rows = (await conn.execute(stmt)).all()
        return [dict(r._mapping) for r in rows]

    async def get_recent_appraisals(self, limit: int = 50) -> list[dict[str, Any]]:
        if self.engine is None:
            return self._mem_appraisals[-limit:]
        stmt = (
            select(appraisals_table)
            .order_by(appraisals_table.c.created_at.desc())
            .limit(limit)
        )
        async with self.engine.connect() as conn:
            rows = (await conn.execute(stmt)).all()
        return [dict(r._mapping) for r in rows]

    async def fetch_training_frame(self, limit: int = 5000) -> pd.DataFrame:
        if self.engine is None:
            rows = self._mem_appraisals[-limit:]
            out: list[dict[str, Any]] = []
            for row in rows:
                payload = dict(row["input_json"] or {})
                payload.setdefault("region", row["region"])
                payload.setdefault("segment", row["segment"])
                payload["auction_close_price"] = float(row["predicted_value"])
                payload.setdefault("observed_at_epoch", int(datetime.now(timezone.utc).timestamp()))
                out.append(payload)
            return pd.DataFrame(out)
        stmt = (
            select(
                appraisals_table.c.input_json,
                appraisals_table.c.predicted_value,
                appraisals_table.c.region,
                appraisals_table.c.segment,
            )
            .order_by(appraisals_table.c.created_at.desc())
            .limit(limit)
        )
        async with self.engine.connect() as conn:
            rows = (await conn.execute(stmt)).all()
        out: list[dict[str, Any]] = []
        for row in rows:
            payload = dict(row.input_json or {})
            payload.setdefault("region", row.region)
            payload.setdefault("segment", row.segment)
            payload["auction_close_price"] = float(row.predicted_value)
            payload.setdefault("observed_at_epoch", int(datetime.now(timezone.utc).timestamp()))
            out.append(payload)
        return pd.DataFrame(out)

