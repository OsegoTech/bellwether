"""
Typed configuration for Bellwether: YAML file, overridden by env.

Precedence, lowest to highest:

    YAML file  <  BELLWETHER_* environment variables

Env vars address nested keys with a double underscore:
``BELLWETHER_PROMETHEUS__BASE_URL`` overrides ``prometheus.base_url``. Overrides
deep-merge, so setting one key leaves its YAML siblings in place.

Secrets (API keys, Slack webhook and signing secret, cert passphrases) are env
only. Every field typed ``SecretStr`` is a secret; finding one in the YAML is a
load-time error, as is a secret that an enabled feature needs but the env does
not supply. Failing at load time means a misconfigured deploy dies at start,
not an hour later when the first finding tries to reach a provider.

The executor whitelist is fixed here as a ``Literal``: config can narrow the
allowed actions, never widen them.

Certificates never ride in a mongo URI. The read and write URIs carry host and
auth options only; cert, CA, and passphrase are separate typed fields handed to
``MongoClient`` as keyword arguments. A URI carrying a cert option or a
password is a load-time error.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal, get_args
from urllib.parse import parse_qsl, urlsplit

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from bellwether.models import Severity

logger = logging.getLogger(__name__)

ENV_PREFIX = "BELLWETHER_"
ENV_NESTED_DELIMITER = "__"

Provider = Literal["claude", "openai"]
NotifyChannel = Literal["stdout", "slack"]
ExecutorAction = Literal["kill_op", "create_small_index"]

_PROVIDER_KEY_FIELD: dict[Provider, str] = {
    "claude": "anthropic_api_key",
    "openai": "openai_api_key",
}
_PROVIDER_MODEL_FIELD: dict[Provider, str] = {
    "claude": "claude_model",
    "openai": "openai_model",
}


class ConfigError(Exception):
    """Configuration could not be loaded.

    The message names fields and env vars but never echoes secret values, so
    it is safe to log.
    """


def env_var_for(section: str, field: str) -> str:
    """The env var that sets ``section.field``, e.g. ``BELLWETHER_MONGO__URI``."""
    return f"{ENV_PREFIX}{section.upper()}{ENV_NESTED_DELIMITER}{field.upper()}"


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# URI options that would put a cert, CA, or key passphrase in the URI string —
# current pymongo names plus legacy aliases. Compared lowercased.
_FORBIDDEN_URI_OPTIONS = frozenset(
    {
        "tlscertificatekeyfile",
        "tlscertificatekeyfilepassword",
        "tlscafile",
        "ssl_certfile",
        "ssl_keyfile",
        "ssl_ca_certs",
        "ssl_pem_passphrase",
        "sslpemkeyfile",
        "sslpemkeypassword",
        "sslcafile",
    }
)


def _check_mongo_uri(uri: str, section: str, field: str) -> str:
    """Reject URIs carrying certs or passwords. Messages never echo values."""
    where = f"{section}.{field}"
    parts = urlsplit(uri)
    if parts.scheme != "mongodb":
        raise ValueError(
            f"{where} must use the mongodb:// scheme "
            "(Bellwether connects to named nodes directly; mongodb+srv is not supported)"
        )
    userinfo = parts.netloc.rpartition("@")[0]
    if ":" in userinfo:
        raise ValueError(f"{where} must not embed a password; X.509 auth needs none")
    forbidden = sorted(
        {
            name
            for name, _ in parse_qsl(parts.query, keep_blank_values=True)
            if name.lower() in _FORBIDDEN_URI_OPTIONS
        }
    )
    if forbidden:
        raise ValueError(
            f"{where} must not carry certificate options ({', '.join(forbidden)}); "
            f"set {section}.tls_cert_file and {section}.tls_ca_file instead, "
            f"and the passphrase via {env_var_for(section, 'tls_cert_passphrase')}"
        )
    return uri


class MongoConfig(_Section):
    """The read side. Identity ``meetadev-ai``; no write path exists here.

    ``uri`` supplies auth options (X.509, $external, tls). The host in it is
    replaced per connection attempt by ``target_node`` then ``fallback_nodes``.
    """

    uri: str
    tls_ca_file: Path
    tls_cert_file: Path
    target_node: str
    fallback_nodes: list[str] = Field(default_factory=list)
    tls_cert_passphrase: SecretStr | None = None
    server_selection_timeout_ms: int = Field(default=5000, gt=0)

    @field_validator("uri")
    @classmethod
    def _uri_carries_no_credentials(cls, value: str) -> str:
        return _check_mongo_uri(value, "mongo", "uri")


class PrometheusConfig(_Section):
    enabled: bool = False
    base_url: str | None = None

    @model_validator(mode="after")
    def _base_url_when_enabled(self) -> PrometheusConfig:
        if self.enabled and not self.base_url:
            raise ValueError("prometheus.base_url is required when prometheus.enabled is true")
        return self


class AnalysisConfig(_Section):
    primary_provider: Provider = "claude"
    fallback_provider: Provider | None = "openai"
    claude_model: str | None = None
    openai_model: str | None = None
    max_retries: int = Field(default=1, ge=0)  # retries after the first attempt, per provider
    timeout_seconds: float = Field(default=60.0, gt=0)
    escalate_min_severity: Severity = Severity.WARNING
    anthropic_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None

    @property
    def providers(self) -> tuple[Provider, ...]:
        """Providers in failover order."""
        if self.fallback_provider is None:
            return (self.primary_provider,)
        return (self.primary_provider, self.fallback_provider)

    @model_validator(mode="after")
    def _check_providers(self) -> AnalysisConfig:
        if self.fallback_provider == self.primary_provider:
            raise ValueError(
                "analysis.fallback_provider must differ from analysis.primary_provider "
                "(set it to null to run with a single provider)"
            )
        for provider in self.providers:
            model_field = _PROVIDER_MODEL_FIELD[provider]
            if not getattr(self, model_field):
                raise ValueError(f"analysis.{model_field} is required when {provider} is a provider")
        return self


class NotifyConfig(_Section):
    channels: list[NotifyChannel] = Field(default_factory=lambda: list[NotifyChannel](["stdout"]))
    slack_webhook_url: SecretStr | None = None
    slack_signing_secret: SecretStr | None = None


class StoreConfig(_Section):
    sqlite_path: Path


class ExecutorConfig(_Section):
    """The write side. Identity ``meetadev-ai-exec``, reachable only past approval."""

    enabled: bool = False
    mongo_uri: str | None = None
    tls_cert_file: Path | None = None
    tls_ca_file: Path | None = None
    tls_cert_passphrase: SecretStr | None = None
    allowed_actions: list[ExecutorAction] = Field(default_factory=list)
    document_threshold: int = Field(default=100_000, gt=0)  # create_small_index ceiling

    @field_validator("mongo_uri")
    @classmethod
    def _uri_carries_no_credentials(cls, value: str | None) -> str | None:
        return None if value is None else _check_mongo_uri(value, "executor", "mongo_uri")

    @model_validator(mode="after")
    def _connection_when_enabled(self) -> ExecutorConfig:
        if self.enabled:
            missing = [
                name
                for name in ("mongo_uri", "tls_cert_file", "tls_ca_file")
                if getattr(self, name) is None
            ]
            if missing:
                raise ValueError(
                    "; ".join(
                        f"executor.{name} is required when executor.enabled is true"
                        for name in missing
                    )
                )
        return self


class OplogWindowCollectorConfig(_Section):
    # Gap between the two serverStatus reads that measure the live write rate.
    # 0 disables sampling; the collector then reports the oplog's mean rate.
    sample_interval_seconds: float = Field(default=10.0, ge=0)


class CollectorsConfig(_Section):
    oplog_window: OplogWindowCollectorConfig = Field(default_factory=OplogWindowCollectorConfig)


class OplogWindowDetectorConfig(_Section):
    # Conservative resync estimate: how long a secondary may be down for
    # maintenance and still need to catch up from the oplog.
    maintenance_window_seconds: int = Field(default=3600, gt=0)
    # WARNING while the window is under resync x safety_factor.
    safety_factor: float = Field(default=2.0, ge=1.0)


class DetectorsConfig(_Section):
    oplog_window: OplogWindowDetectorConfig = Field(default_factory=OplogWindowDetectorConfig)


class BellwetherConfig(BaseSettings):
    """Root config. Build it with :func:`load_config`, which applies the YAML layer.

    Constructor kwargs act as the YAML layer: env vars take precedence over them.
    """

    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        env_nested_delimiter=ENV_NESTED_DELIMITER,
        extra="forbid",
        frozen=True,
    )

    mongo: MongoConfig
    prometheus: PrometheusConfig = Field(default_factory=PrometheusConfig)
    analysis: AnalysisConfig
    notify: NotifyConfig = Field(default_factory=NotifyConfig)
    store: StoreConfig
    executor: ExecutorConfig = Field(default_factory=ExecutorConfig)
    collectors: CollectorsConfig = Field(default_factory=CollectorsConfig)
    detectors: DetectorsConfig = Field(default_factory=DetectorsConfig)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Earlier sources win. Env beats init kwargs (the YAML layer); no .env
        # files or secrets dirs — the process environment is the only secret source.
        return (env_settings, init_settings)

    def required_secrets(self) -> list[tuple[str, str]]:
        """(section, field) of every secret the enabled features need."""
        needed = [("analysis", _PROVIDER_KEY_FIELD[p]) for p in self.analysis.providers]
        if "slack" in self.notify.channels:
            needed += [("notify", "slack_webhook_url"), ("notify", "slack_signing_secret")]
        return needed

    @model_validator(mode="after")
    def _separate_write_identity(self) -> BellwetherConfig:
        if self.executor.tls_cert_file is not None and (
            self.executor.tls_cert_file == self.mongo.tls_cert_file
        ):
            raise ValueError(
                "executor.tls_cert_file must be the write identity's own cert "
                "(meetadev-ai-exec), not the read identity's mongo.tls_cert_file"
            )
        return self

    @model_validator(mode="after")
    def _require_secrets(self) -> BellwetherConfig:
        missing = [
            env_var_for(section, field)
            for section, field in self.required_secrets()
            if _is_blank(getattr(getattr(self, section), field))
        ]
        if missing:
            raise ValueError("missing required secret env var(s): " + ", ".join(missing))
        return self


def load_config(path: str | os.PathLike[str]) -> BellwetherConfig:
    """Load config from a YAML file plus BELLWETHER_* env overrides.

    Raises ConfigError on an unreadable file, a secret in the YAML, a missing
    required secret, or any invalid value.
    """
    path = Path(path)
    raw = _read_yaml(path)
    _reject_yaml_secrets(raw, path)
    try:
        config = BellwetherConfig(**raw)
    except ValidationError as exc:
        # `from None`: pydantic's own rendering includes raw input values,
        # which could carry secrets from env.
        raise ConfigError(f"invalid config ({path}):\n{_format_errors(exc)}") from None
    logger.info(
        "config loaded",
        extra={
            "config_path": str(path),
            "target_node": config.mongo.target_node,
            "providers": list(config.analysis.providers),
            "notify_channels": list(config.notify.channels),
            "executor_enabled": config.executor.enabled,
        },
    )
    return config


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read config file {path}: {exc.strerror}") from None
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        # Report position only — the offending line could hold a secret.
        mark = getattr(exc, "problem_mark", None)
        where = f" at line {mark.line + 1}" if mark is not None else ""
        raise ConfigError(f"invalid YAML in {path}{where}") from None
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"config file {path} must be a YAML mapping, got {type(data).__name__}")
    return data


def _reject_yaml_secrets(raw: dict[str, Any], path: Path) -> None:
    offenders = [
        f"{section}.{field} (set {env_var_for(section, field)} instead)"
        for section, model in _sections()
        if isinstance(block := raw.get(section), dict)
        for field in _secret_fields(model)
        if field in block
    ]
    if offenders:
        raise ConfigError(
            f"secrets must come from env, never YAML; remove from {path}: " + "; ".join(offenders)
        )


def _sections() -> Iterator[tuple[str, type[BaseModel]]]:
    for name, info in BellwetherConfig.model_fields.items():
        if isinstance(info.annotation, type) and issubclass(info.annotation, BaseModel):
            yield name, info.annotation


def _secret_fields(model: type[BaseModel]) -> list[str]:
    return [
        name
        for name, info in model.model_fields.items()
        if info.annotation is SecretStr or SecretStr in get_args(info.annotation)
    ]


def _is_blank(value: SecretStr | None) -> bool:
    return value is None or not value.get_secret_value().strip()


def _format_errors(exc: ValidationError) -> str:
    lines = []
    for err in exc.errors():
        msg = err["msg"]
        if err["type"] == "value_error":
            # Our own validators' messages are already fully qualified.
            lines.append(f"  {msg.removeprefix('Value error, ')}")
            continue
        loc = ".".join(str(part) for part in err["loc"])
        if err["type"] in ("literal_error", "enum"):
            # Enumerated fields are never secrets; echoing the input helps.
            msg = f"{msg} (got {err['input']!r})"
        lines.append(f"  {loc}: {msg}")
    return "\n".join(lines)
