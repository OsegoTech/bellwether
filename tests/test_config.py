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

# Nearest voter to the monitoring host first.
FALLBACK_ORDER = [
    "mongo-1.example.internal:27017",
    "mongo-2.example.internal:27017",
    "mongo-3.example.internal:27017",
]

READ_URI = (
    "mongodb://mongo-hidden.example.internal:27017/"
    "?authMechanism=MONGODB-X509&authSource=%24external&tls=true&directConnection=true"
)


# The write side routes through the replica set so index builds reach the primary.
EXEC_RS_URI = (
    "mongodb://mongo-1.example.internal:27017,mongo-2.example.internal:27017,"
    "mongo-3.example.internal:27017/"
    "?replicaSet=rs0&authMechanism=MONGODB-X509&authSource=%24external&tls=true"
)


def executor_block(**overrides: Any) -> dict[str, Any]:
    block: dict[str, Any] = {
        "enabled": True,
        "mongo_uri": EXEC_RS_URI,
        "tls_cert_file": "/etc/bellwether/tls/bellwether-exec.combined.pem",
        "tls_ca_file": "/etc/mongodb/tls/ca-chain.cert.pem",
        "allowed_actions": ["kill_op"],
    }
    block.update(overrides)
    return block


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
            "uri": READ_URI,
            "tls_ca_file": "/etc/mongodb/tls/ca-chain.cert.pem",
            "tls_cert_file": "/etc/bellwether/tls/bellwether-reader.combined.pem",
            "target_node": "mongo-hidden.example.internal:27017",
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
    assert cfg.mongo.target_node == "mongo-hidden.example.internal:27017"
    assert cfg.mongo.fallback_nodes == FALLBACK_ORDER
    assert cfg.mongo.tls_ca_file == Path("/etc/mongodb/tls/ca-chain.cert.pem")
    assert "MONGODB-X509" in cfg.mongo.uri
    assert cfg.prometheus.enabled is True
    assert cfg.prometheus.base_url == "http://prometheus.example.internal:9090"
    assert cfg.analysis.primary_provider == "claude"
    assert cfg.analysis.fallback_provider == "openai"
    assert cfg.analysis.escalate_min_severity is Severity.WARNING
    assert cfg.notify.channels == ["stdout", "slack"]
    assert isinstance(cfg.store.sqlite_path, Path)
    assert cfg.executor.enabled is False
    assert set(cfg.executor.allowed_actions) <= {"kill_op", "create_small_index"}
    assert cfg.executor.document_threshold == 100_000
    # Separate write identity, with its own cert.
    assert cfg.executor.tls_cert_file != cfg.mongo.tls_cert_file


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

    assert load_config(path).mongo.tls_cert_passphrase is None

    monkeypatch.setenv("BELLWETHER_MONGO__TLS_CERT_PASSPHRASE", "hunter2")
    monkeypatch.setenv("BELLWETHER_EXECUTOR__TLS_CERT_PASSPHRASE", "hunter3")
    cfg = load_config(path)
    assert cfg.mongo.tls_cert_passphrase is not None
    assert cfg.mongo.tls_cert_passphrase.get_secret_value() == "hunter2"
    assert cfg.executor.tls_cert_passphrase is not None
    assert cfg.executor.tls_cert_passphrase.get_secret_value() == "hunter3"


# --- Secrets are env-only, never YAML, never logged -------------------------


@pytest.mark.parametrize(
    ("section", "field", "env_var"),
    [
        ("analysis", "anthropic_api_key", ANTHROPIC_KEY),
        ("analysis", "openai_api_key", OPENAI_KEY),
        ("notify", "slack_webhook_url", SLACK_WEBHOOK),
        ("notify", "slack_signing_secret", SLACK_SIGNING),
        ("mongo", "tls_cert_passphrase", "BELLWETHER_MONGO__TLS_CERT_PASSPHRASE"),
        ("executor", "tls_cert_passphrase", "BELLWETHER_EXECUTOR__TLS_CERT_PASSPHRASE"),
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
    data.setdefault(section, {})[field] = "leaked-into-yaml-9f3a"

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
    data["executor"] = executor_block(allowed_actions=["kill_op", "drop_database"])

    with pytest.raises(ConfigError, match="drop_database"):
        load_config(write_yaml(tmp_path, data))


def test_enabled_executor_loads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ANTHROPIC_KEY, SECRETS[ANTHROPIC_KEY])
    data = minimal()
    data["executor"] = executor_block(document_threshold=5000)

    cfg = load_config(write_yaml(tmp_path, data))

    assert cfg.executor.enabled is True
    assert cfg.executor.tls_ca_file == Path("/etc/mongodb/tls/ca-chain.cert.pem")
    assert cfg.executor.document_threshold == 5000


@pytest.mark.parametrize("missing", ["mongo_uri", "tls_cert_file", "tls_ca_file"])
def test_enabled_executor_requires_connection_fields(
    missing: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ANTHROPIC_KEY, SECRETS[ANTHROPIC_KEY])
    data = minimal()
    data["executor"] = executor_block()
    del data["executor"][missing]

    with pytest.raises(ConfigError, match=f"executor.{missing}"):
        load_config(write_yaml(tmp_path, data))


def test_executor_must_not_reuse_read_identity_cert(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ANTHROPIC_KEY, SECRETS[ANTHROPIC_KEY])
    data = minimal()
    data["executor"] = executor_block(tls_cert_file=data["mongo"]["tls_cert_file"])

    with pytest.raises(ConfigError, match="bellwether-exec"):
        load_config(write_yaml(tmp_path, data))


def test_document_threshold_must_be_positive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ANTHROPIC_KEY, SECRETS[ANTHROPIC_KEY])
    data = minimal()
    data["executor"] = executor_block(document_threshold=0)

    with pytest.raises(ConfigError, match="document_threshold"):
        load_config(write_yaml(tmp_path, data))


# --- Decision 6: certificates never ride in a mongo URI ---------------------

FORBIDDEN_URI_PARAMS = [
    "tlsCertificateKeyFile=/etc/bellwether/tls/x.pem",
    "tlsCAFile=/etc/mongodb/tls/ca-chain.cert.pem",
    "tlsCertificateKeyFilePassword=hunter2",
    "tlscertificatekeyfile=/lowercase/still/caught.pem",
    "ssl_certfile=/legacy/alias.pem",
]


@pytest.mark.parametrize("param", FORBIDDEN_URI_PARAMS)
@pytest.mark.parametrize("section", ["mongo", "executor"])
def test_cert_params_in_uri_are_rejected(
    section: str, param: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ANTHROPIC_KEY, SECRETS[ANTHROPIC_KEY])
    data = minimal()
    data["executor"] = executor_block()
    key = "uri" if section == "mongo" else "mongo_uri"
    data[section][key] += "&" + param

    with pytest.raises(ConfigError) as excinfo:
        load_config(write_yaml(tmp_path, data))

    message = str(excinfo.value)
    assert f"{section}.{key}" in message
    assert "tls_cert_file" in message
    assert "hunter2" not in message


def test_password_in_uri_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ANTHROPIC_KEY, SECRETS[ANTHROPIC_KEY])
    data = minimal()
    data["mongo"]["uri"] = "mongodb://someone:s3cret-pw@mongo-hidden.example.internal:27017/"

    with pytest.raises(ConfigError, match="mongo.uri") as excinfo:
        load_config(write_yaml(tmp_path, data))

    assert "s3cret-pw" not in str(excinfo.value)


def test_srv_uri_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Direct per-node connections need a plain mongodb:// URI."""
    monkeypatch.setenv(ANTHROPIC_KEY, SECRETS[ANTHROPIC_KEY])
    data = minimal()
    data["mongo"]["uri"] = "mongodb+srv://cluster.example.internal/"

    with pytest.raises(ConfigError, match="mongo.uri"):
        load_config(write_yaml(tmp_path, data))


def test_example_uris_carry_no_cert_paths(secrets_env: None) -> None:
    cfg = load_config(EXAMPLE_YAML)

    for uri in (cfg.mongo.uri, cfg.executor.mongo_uri or ""):
        assert "tlsCertificateKeyFile" not in uri
        assert "tlsCAFile" not in uri


# --- Correction A: the executor routes through the replica set -----------------------


def test_executor_uri_must_not_use_direct_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ANTHROPIC_KEY, SECRETS[ANTHROPIC_KEY])
    data = minimal()
    data["executor"] = executor_block(mongo_uri=EXEC_RS_URI + "&directConnection=true")

    with pytest.raises(ConfigError, match="directConnection"):
        load_config(write_yaml(tmp_path, data))


def test_executor_uri_must_name_the_replica_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ANTHROPIC_KEY, SECRETS[ANTHROPIC_KEY])
    data = minimal()
    data["executor"] = executor_block(
        mongo_uri="mongodb://mongo-1.example.internal:27017,mongo-2.example.internal:27017/"
        "?authMechanism=MONGODB-X509&authSource=%24external&tls=true"
    )

    with pytest.raises(ConfigError, match="replicaSet"):
        load_config(write_yaml(tmp_path, data))


# --- .env for local development -------------------------------------------------------


def write_dotenv(directory: Path, values: dict[str, str]) -> None:
    (directory / ".env").write_text("".join(f"{name}={value}\n" for name, value in values.items()))


def test_dotenv_values_are_loaded(tmp_path: Path) -> None:
    assert Path.cwd() == tmp_path  # tests/conftest.py: each test runs in its own directory
    write_dotenv(
        tmp_path,
        {
            "BELLWETHER_PROMETHEUS__BASE_URL": "http://from-dotenv:9090",
            ANTHROPIC_KEY: "sk-ant-from-dotenv",
        },
    )
    data = minimal()
    data["prometheus"] = {"enabled": True, "base_url": "http://from-yaml:9090"}

    cfg = load_config(write_yaml(tmp_path, data))

    assert cfg.prometheus.base_url == "http://from-dotenv:9090"  # .env beats YAML
    assert cfg.prometheus.enabled is True  # YAML sibling survives (deep merge)
    key = cfg.analysis.anthropic_api_key
    assert key is not None and key.get_secret_value() == "sk-ant-from-dotenv"


def test_real_environment_beats_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_dotenv(
        tmp_path,
        {
            "BELLWETHER_PROMETHEUS__BASE_URL": "http://from-dotenv:9090",
            ANTHROPIC_KEY: "sk-ant-from-dotenv",
        },
    )
    monkeypatch.setenv("BELLWETHER_PROMETHEUS__BASE_URL", "http://from-env:9090")
    monkeypatch.setenv(ANTHROPIC_KEY, "sk-ant-from-env")

    cfg = load_config(write_yaml(tmp_path, minimal()))

    assert cfg.prometheus.base_url == "http://from-env:9090"
    key = cfg.analysis.anthropic_api_key
    assert key is not None and key.get_secret_value() == "sk-ant-from-env"


# --- Correction C: analysis timeout ---------------------------------------------------


def test_analysis_timeout_defaults_to_120_seconds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ANTHROPIC_KEY, SECRETS[ANTHROPIC_KEY])

    assert load_config(write_yaml(tmp_path, minimal())).analysis.timeout_seconds == 120


def test_example_analysis_timeout_is_120_seconds(secrets_env: None) -> None:
    assert load_config(EXAMPLE_YAML).analysis.timeout_seconds == 120


# --- Correction B: approver allowlist -------------------------------------------------


def test_approver_ids_default_to_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ANTHROPIC_KEY, SECRETS[ANTHROPIC_KEY])

    assert load_config(write_yaml(tmp_path, minimal())).approval.approver_ids == []


def test_approver_ids_from_yaml_and_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ANTHROPIC_KEY, SECRETS[ANTHROPIC_KEY])
    data = minimal()
    data["approval"] = {"approver_ids": ["U0OSEGO"]}
    path = write_yaml(tmp_path, data)

    assert load_config(path).approval.approver_ids == ["U0OSEGO"]

    monkeypatch.setenv("BELLWETHER_APPROVAL__APPROVER_IDS", '["U0OSEGO", "W0TEAMMATE"]')
    assert load_config(path).approval.approver_ids == ["U0OSEGO", "W0TEAMMATE"]


@pytest.mark.parametrize("bad", ["osego", "@osego", "u0osego", "", "U0 OSEGO"])
def test_approver_ids_must_be_slack_user_ids(
    bad: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ANTHROPIC_KEY, SECRETS[ANTHROPIC_KEY])
    data = minimal()
    data["approval"] = {"approver_ids": ["U0OSEGO", bad]}

    with pytest.raises(ConfigError, match="approval.approver_ids"):
        load_config(write_yaml(tmp_path, data))


def test_ui_approval_is_enabled_by_default_and_env_overridable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ANTHROPIC_KEY, SECRETS[ANTHROPIC_KEY])
    path = write_yaml(tmp_path, minimal())

    assert load_config(path).approval.ui_approval_enabled is True

    monkeypatch.setenv("BELLWETHER_APPROVAL__UI_APPROVAL_ENABLED", "false")
    assert load_config(path).approval.ui_approval_enabled is False


def test_example_allowlist_fails_closed(secrets_env: None) -> None:
    # Nobody can approve from Slack until an operator names the approvers.
    assert load_config(EXAMPLE_YAML).approval.approver_ids == []


def test_example_executor_uri_is_a_replica_set_uri(secrets_env: None) -> None:
    uri = load_config(EXAMPLE_YAML).executor.mongo_uri

    assert uri is not None
    assert "replicaSet=rs0" in uri
    assert "directConnection" not in uri
    for node in FALLBACK_ORDER:  # every voting member seeds the connection
        assert node in uri


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
    data["mongo"]["target_nod"] = "typo.example.internal:27017"

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
