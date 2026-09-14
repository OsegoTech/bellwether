"""Acceptance tests for bellwether.config — BUILD_SPEC §3.2.

Spec acceptance:
  - loads the example YAML
  - an env override changes a value
  - a missing required secret raises a clear error naming the missing var

Plus the rules §3.2/§4/§5 state around them: secrets are env-only (never YAML),
secrets are required only when the feature that uses them is enabled, and the
executor whitelist cannot be widened through config.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from bellwether.config import BellwetherConfig, ConfigError, load_config
from bellwether.models import Severity

EXAMPLE_YAML = Path(__file__).resolve().parents[1] / "config" / "bellwether.example.yaml"

ANTHROPIC_KEY = "BELLWETHER_ANALYSIS__ANTHROPIC_API_KEY"
OPENAI_KEY = "BELLWETHER_ANALYSIS__OPENAI_API_KEY"
SLACK_WEBHOOK = "BELLWETHER_NOTIFY__SLACK_WEBHOOK_URL"
SLACK_SIGNING = "BELLWETHER_NOTIFY__SLACK_SIGNING_SECRET"

# Everything the example YAML (claude + openai, stdout + slack) needs.
SECRETS = {
    ANTHROPIC_KEY: "sk-ant-test-0000",
    OPENAI_KEY: "sk-openai-test-0000",
    SLACK_WEBHOOK: "https://hooks.slack.com/services/T000/B000/XXXX",
    SLACK_SIGNING: "slack-signing-test-0000",
}

VOTING_NODES = [
    "node-uae.mongo.internal:27017",
    "node-southafrica.mongo.internal:27017",
    "node-westeurope.mongo.internal:27017",
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate every test from BELLWETHER_* vars in the developer's shell."""
    for name in list(os.environ):
        if name.upper().startswith("BELLWETHER_"):
            monkeypatch.delenv(name)


@pytest.fixture
def secrets_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in SECRETS.items():
        monkeypatch.setenv(name, value)


def minimal() -> dict[str, Any]:
    """Smallest valid config: claude only, no fallback, stdout only."""
    return {
        "mongo": {
            "uri": "mongodb://node-backup.mongo.internal:27017/?authMechanism=MONGODB-X509",
            "tls_ca_file": "/etc/mongodb/tls/ca-chain.cert.pem",
            "tls_cert_file": "/etc/bellwether/tls/meetadev-ai.combined.pem",
            "target_node": "node-backup.mongo.internal:27017",
        },
        "analysis": {
            "primary_provider": "claude",
            "fallback_provider": None,
            "claude_model": "claude-opus-5",
        },
        "notify": {"channels": ["stdout"]},
        "store": {"sqlite_path": "bellwether.db"},
    }


def write_yaml(tmp_path: Path, data: dict[str, Any]) -> Path:
    path = tmp_path / "bellwether.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


# --- Acceptance: loads the example YAML ------------------------------------


def test_loads_example_yaml(secrets_env: None) -> None:
    cfg = load_config(EXAMPLE_YAML)

    assert isinstance(cfg, BellwetherConfig)
    assert cfg.mongo.target_node == "node-backup.mongo.internal:27017"
    assert cfg.mongo.fallback_nodes == VOTING_NODES
    assert cfg.mongo.tls_ca_file == Path("/etc/mongodb/tls/ca-chain.cert.pem")
    assert "MONGODB-X509" in cfg.mongo.uri
    assert cfg.prometheus.enabled is True
    assert cfg.prometheus.base_url == "http://10.3.2.4:9090"
    assert cfg.analysis.primary_provider == "claude"
    assert cfg.analysis.fallback_provider == "openai"
    assert cfg.analysis.escalate_min_severity is Severity.WARNING
    assert cfg.notify.channels == ["stdout", "slack"]
    assert isinstance(cfg.store.sqlite_path, Path)
    assert cfg.executor.enabled is False
    assert set(cfg.executor.allowed_actions) <= {"kill_op", "create_small_index"}


def test_example_secrets_come_from_env(secrets_env: None) -> None:
    cfg = load_config(EXAMPLE_YAML)

    assert cfg.analysis.anthropic_api_key is not None
    assert cfg.analysis.anthropic_api_key.get_secret_value() == SECRETS[ANTHROPIC_KEY]
    assert cfg.notify.slack_signing_secret is not None
    assert cfg.notify.slack_signing_secret.get_secret_value() == SECRETS[SLACK_SIGNING]


# --- Acceptance: env override changes a value -------------------------------


def test_env_override_changes_value(
    secrets_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BELLWETHER_PROMETHEUS__BASE_URL", "http://prom.example:9090")

    cfg = load_config(EXAMPLE_YAML)

    assert cfg.prometheus.base_url == "http://prom.example:9090"
    # Deep merge: the sibling key from YAML survives the override.
    assert cfg.prometheus.enabled is True


def test_env_override_is_type_coerced(
    secrets_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BELLWETHER_ANALYSIS__MAX_RETRIES", "5")
    monkeypatch.setenv("BELLWETHER_PROMETHEUS__ENABLED", "false")
    monkeypatch.setenv("BELLWETHER_ANALYSIS__ESCALATE_MIN_SEVERITY", "critical")

    cfg = load_config(EXAMPLE_YAML)

    assert cfg.analysis.max_retries == 5
    assert cfg.prometheus.enabled is False
    assert cfg.analysis.escalate_min_severity is Severity.CRITICAL


def test_env_override_of_list(secrets_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BELLWETHER_MONGO__FALLBACK_NODES", '["node-a:27017", "node-b:27017"]')

    cfg = load_config(EXAMPLE_YAML)

    assert cfg.mongo.fallback_nodes == ["node-a:27017", "node-b:27017"]


def test_env_override_is_validated(
    secrets_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BELLWETHER_ANALYSIS__ESCALATE_MIN_SEVERITY", "catastrophic")

    with pytest.raises(ConfigError):
        load_config(EXAMPLE_YAML)


# --- Acceptance: missing required secret raises, naming the var -------------


@pytest.mark.parametrize("missing", sorted(SECRETS))
def test_missing_required_secret_names_the_var(
    missing: str, secrets_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(missing)

    with pytest.raises(ConfigError, match=missing):
        load_config(EXAMPLE_YAML)


def test_all_missing_secrets_reported_at_once() -> None:
    with pytest.raises(ConfigError) as excinfo:
        load_config(EXAMPLE_YAML)

    for name in SECRETS:
        assert name in str(excinfo.value)


def test_empty_secret_counts_as_missing(
    secrets_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ANTHROPIC_KEY, "")

    with pytest.raises(ConfigError, match=ANTHROPIC_KEY):
        load_config(EXAMPLE_YAML)


def test_secrets_only_required_for_features_in_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ANTHROPIC_KEY, SECRETS[ANTHROPIC_KEY])

    cfg = load_config(write_yaml(tmp_path, minimal()))

    assert cfg.analysis.openai_api_key is None
    assert cfg.notify.slack_webhook_url is None
    assert cfg.notify.slack_signing_secret is None


def test_fallback_provider_requires_its_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ANTHROPIC_KEY, SECRETS[ANTHROPIC_KEY])
    data = minimal()
    data["analysis"].update(fallback_provider="openai", openai_model="gpt-5")

    with pytest.raises(ConfigError, match=OPENAI_KEY):
        load_config(write_yaml(tmp_path, data))


def test_optional_cert_passphrase_from_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ANTHROPIC_KEY, SECRETS[ANTHROPIC_KEY])
    path = write_yaml(tmp_path, minimal())

    assert load_config(path).mongo.tls_cert_key_password is None

    monkeypatch.setenv("BELLWETHER_MONGO__TLS_CERT_KEY_PASSWORD", "hunter2")
    password = load_config(path).mongo.tls_cert_key_password
    assert password is not None
    assert password.get_secret_value() == "hunter2"


# --- Secrets are env-only, never YAML, never logged -------------------------


@pytest.mark.parametrize(
    ("section", "field", "env_var"),
    [
        ("analysis", "anthropic_api_key", ANTHROPIC_KEY),
        ("analysis", "openai_api_key", OPENAI_KEY),
        ("notify", "slack_webhook_url", SLACK_WEBHOOK),
        ("notify", "slack_signing_secret", SLACK_SIGNING),
        ("mongo", "tls_cert_key_password", "BELLWETHER_MONGO__TLS_CERT_KEY_PASSWORD"),
    ],
)
def test_secret_in_yaml_is_rejected(
    section: str,
    field: str,
    env_var: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ANTHROPIC_KEY, SECRETS[ANTHROPIC_KEY])
    data = minimal()
    data[section][field] = "leaked-into-yaml-9f3a"

    with pytest.raises(ConfigError) as excinfo:
        load_config(write_yaml(tmp_path, data))

    message = str(excinfo.value)
    assert f"{section}.{field}" in message
    assert env_var in message
    assert "leaked-into-yaml-9f3a" not in message


def test_secrets_not_exposed_in_repr(secrets_env: None) -> None:
    cfg = load_config(EXAMPLE_YAML)
    rendered = repr(cfg) + str(cfg)

    for value in SECRETS.values():
        assert value not in rendered


# --- Structural validation --------------------------------------------------


def test_executor_whitelist_cannot_be_widened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ANTHROPIC_KEY, SECRETS[ANTHROPIC_KEY])
    data = minimal()
    data["executor"] = {
        "enabled": True,
        "mongo_uri": "mongodb://node-uae.mongo.internal:27017/",
        "allowed_actions": ["kill_op", "drop_database"],
    }

    with pytest.raises(ConfigError, match="drop_database"):
        load_config(write_yaml(tmp_path, data))


def test_enabled_executor_requires_mongo_uri(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ANTHROPIC_KEY, SECRETS[ANTHROPIC_KEY])
    data = minimal()
    data["executor"] = {"enabled": True, "allowed_actions": ["kill_op"]}

    with pytest.raises(ConfigError, match="executor.mongo_uri"):
        load_config(write_yaml(tmp_path, data))


def test_primary_and_fallback_must_differ(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ANTHROPIC_KEY, SECRETS[ANTHROPIC_KEY])
    data = minimal()
    data["analysis"]["fallback_provider"] = "claude"

    with pytest.raises(ConfigError, match="fallback_provider"):
        load_config(write_yaml(tmp_path, data))


def test_provider_in_use_requires_model_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ANTHROPIC_KEY, SECRETS[ANTHROPIC_KEY])
    data = minimal()
    del data["analysis"]["claude_model"]

    with pytest.raises(ConfigError, match="analysis.claude_model"):
        load_config(write_yaml(tmp_path, data))


def test_unknown_yaml_key_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ANTHROPIC_KEY, SECRETS[ANTHROPIC_KEY])
    data = minimal()
    data["mongo"]["target_nod"] = "typo.mongo.internal:27017"

    with pytest.raises(ConfigError, match="target_nod"):
        load_config(write_yaml(tmp_path, data))


def test_missing_config_file_raises(tmp_path: Path) -> None:
    missing = tmp_path / "nope.yaml"

    with pytest.raises(ConfigError, match="nope.yaml"):
        load_config(missing)


def test_config_is_immutable(secrets_env: None) -> None:
    cfg = load_config(EXAMPLE_YAML)

    with pytest.raises(ValidationError):
        cfg.mongo.target_node = "elsewhere:27017"  # type: ignore[misc]
