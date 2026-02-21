from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServiceSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")
    postgres_dsn: str = Field(
        default="postgresql+asyncpg://appraisal:appraisal@localhost:5432/appraisal",
        alias="POSTGRES_DSN",
    )
    kafka_bootstrap_servers: str = Field(default="localhost:9092", alias="KAFKA_BOOTSTRAP_SERVERS")
    kafka_client_id: str = Field(default="appraisal-service", alias="KAFKA_CLIENT_ID")
    nhtsa_base_url: str = Field(default="https://vpic.nhtsa.dot.gov/api/vehicles", alias="NHTSA_BASE_URL")
    model_artifact_dir: str = Field(default="./artifacts", alias="MODEL_ARTIFACT_DIR")
    confidence_threshold: float = Field(default=0.95, alias="CONFIDENCE_THRESHOLD")
    appraisal_cache_ttl_seconds: int = Field(default=86_400, alias="APPRAISAL_CACHE_TTL_SECONDS")
    vin_cache_ttl_seconds: int = Field(default=2_592_000, alias="VIN_CACHE_TTL_SECONDS")
    shadow_window_days: int = Field(default=14, alias="SHADOW_WINDOW_DAYS")
    shadow_mae_threshold: float = Field(default=1500.0, alias="SHADOW_MAE_THRESHOLD")
    canary_traffic_step: float = Field(default=0.10, alias="CANARY_TRAFFIC_STEP")
    canary_shadow_hours: int = Field(default=72, alias="CANARY_SHADOW_HOURS")

    # Auction feed integration
    mmr_api_key: str = Field(default="", alias="MMR_API_KEY")
    mmr_base_url: str = Field(default="https://api.manheim.com", alias="MMR_BASE_URL")
    third_party_auction_api_key: str = Field(default="", alias="THIRD_PARTY_AUCTION_API_KEY")
    third_party_auction_base_url: str = Field(default="https://api.auction-api.app/v1", alias="THIRD_PARTY_AUCTION_BASE_URL")

    # Security
    api_keys: str = Field(default="", alias="API_KEYS")
    rate_limit_rpm: int = Field(default=120, alias="RATE_LIMIT_RPM")

    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_format: str = Field(default="json", alias="LOG_FORMAT")
