# Bellwether — Build Specification

> This document is the executable contract for building Bellwether. It is written
> to be handed to Claude Code, section by section. Each component states its
> purpose, its interface, its real-world grounding, and acceptance tests that must
> pass before moving on. Build in the order given. Do not skip ahead.

---

## 0. What Bellwether is

A predictive reliability service for self-hosted MongoDB replica sets — the
operational intelligence MongoDB gates behind Atlas Performance Advisor and
Enterprise Advanced Ops Manager, rebuilt for Community Edition, reasoned by an
LLM, and gated by a human before any action touches the cluster.

It runs on a schedule. It reads the cluster's diagnostic surface through
read-only collectors, detects trajectories toward known MongoDB failure modes
deterministically, escalates only real findings to an AI analysis stage (Claude
primary, GPT fallback, schema-validated), produces evidence-backed proposals,
notifies humans, and — on explicit human approval — executes only two bounded,
reversible, whitelisted actions. Everything else it proposes for a human to run.

**The guarantee boundary:** nothing the AI produces reaches the cluster without a
recorded human approval. The collectors hold a read-only identity; there is no
write path in the read side of the system at all. The executor is a separate
component with its own identity, reachable only through the approval flow.

---

## 1. Grounding — the reference deployment

The topology Bellwether was built against. Hostnames, addresses and names are
example values (RFC 2606 names, RFC 5737 addresses); substitute your own.

| Thing | Value |
|---|---|
| Replica set | `rs0` |
| Voting nodes | `mongo-2.example.internal:27017` (192.0.2.10), `mongo-3.example.internal:27017` (192.0.2.11), `mongo-1.example.internal:27017` (192.0.2.12) |
| Hidden backup node (collector target) | `mongo-hidden.example.internal:27017` (192.0.2.13) — non-voting, priority 0, hidden |
| Monitoring VM | `monitoring-host` 192.0.2.14, public 198.51.100.20 |
| Prometheus | v3.12.0 on the monitoring VM, 30-day retention, `http://prometheus.example.internal:9090` |
| CA chain | `/etc/mongodb/tls/ca-chain.cert.pem` |
| Auth | MongoDB-X509 over mutual TLS, `authSource=$external` |
| Ledger DB | `appdb`, collection `transactions` (double-entry, `$jsonSchema` validated) |

**Bellwether's identity: `bellwether-reader`**
- DN: `CN=bellwether-reader,OU=cluster-clients,O=Example,DC=example,DC=internal`
- Roles: `clusterMonitor@admin`, `read@local`, `read@appdb`
- Created via the **useradmin** identity (separation of duties — creating users is
  userAdmin's job, not clusterAdmin's), consistent with how Part 8 extended the
  monitor identity for the exporter.
- The `read@appdb` grant exists solely so collectors can read
  `system.profile` in that DB. This is a real widening and the article names it:
  profiler access implies document-read access on that DB. The mitigation is that
  `bellwether-reader` is a dedicated identity, used only by Bellwether, and its use is
  audited.

**Gotchas inherited from the series:**
- x.509 with SAN certs is mandatory. The node certs are SAN-correct (serials
  100a/100b/100c). `pymongo`'s TLS will verify hostnames — do not disable this.
- In systemd unit files, `%24external` must be written `%%24external` (a literal
  `%` is escaped as `%%`). This bit the exporter in Part 8.
- Connection string auth params: `authMechanism=MONGODB-X509`,
  `authSource=%24external`, `tls=true`, `tlsCertificateKeyFile=<combined.pem>`,
  `tlsCAFile=<ca-chain>`.

---

## 2. Project layout

```
bellwether/
├── pyproject.toml
├── README.md
├── config/
│   ├── bellwether.example.yaml
│   └── proposal.schema.json
├── bellwether/
│   ├── __init__.py
│   ├── models.py          # DONE — the data contracts (provided)
│   ├── config.py          # typed config from YAML + env
│   ├── mongo.py           # read-only client
│   ├── prometheus.py      # optional trend enrichment
│   ├── collectors/
│   │   ├── base.py        # Collector ABC
│   │   └── oplog_window.py # first vertical-slice collector
│   ├── detectors/
│   │   ├── base.py        # Detector ABC
│   │   └── oplog_window.py # first vertical-slice detector
│   ├── analysis/
│   │   ├── provider.py    # Provider ABC + failover orchestration
│   │   ├── claude.py
│   │   ├── openai.py
│   │   └── analyst.py     # Finding -> Proposal, schema-validated
│   ├── notify/
│   │   ├── base.py
│   │   ├── stdout.py
│   │   └── slack.py
│   ├── store/
│   │   └── sqlite.py      # append-only proposal + audit store
│   ├── executor/
│   │   ├── whitelist.py   # the two allowed actions
│   │   └── executor.py    # write-capable, own identity, approval-gated
│   ├── pipeline.py        # wires the stages for one run
│   └── cli.py             # entrypoint: run, propose, approve, list
└── tests/
```

---

## 3. Build order and acceptance tests

Build each numbered item fully, make its tests green, then proceed.

### 3.1 `models.py` — DONE
Contracts provided separately. Acceptance: the smoke test constructing a Signal,
Finding, and Proposal passes. (Already verified.)

### 3.2 `config.py`
Typed configuration via `pydantic-settings`, loaded from a YAML file with env
overrides (`BELLWETHER_` prefix). Sections:
- `mongo`: uri, tls_ca_file, tls_cert_file, target_node, fallback_nodes
- `prometheus`: enabled (bool), base_url
- `analysis`: primary_provider, fallback_provider, model names, max_retries,
  timeout_seconds, escalate_min_severity
- `notify`: channels (list), slack webhook + signing secret (from env only)
- `store`: sqlite_path
- `executor`: enabled (bool), mongo_uri (separate write identity), allowed_actions

Secrets (API keys, Slack signing secret, cert passphrases) come from env only,
never YAML. `config.py` raises on missing required secrets at load time, not at
use time.

**Acceptance:** loads the example YAML; env override changes a value; missing
required secret raises a clear error naming the missing var.

### 3.3 `mongo.py` — the read-only client
Wraps a `pymongo.MongoClient` configured for X509/TLS from config. Two hard rules,
enforced in code:
1. Connects to `target_node` with `directConnection=True`; on failure, tries
   `fallback_nodes` in order. Logs which node served the read.
2. Exposes only read helpers: `run_admin_command`, `server_status`, `rs_status`,
   `oplog_stats`, `profile_read`, `index_stats`, `current_op`. There is **no**
   write method. Not "disabled" — absent.

**Acceptance (unit, mocked):** a fake `MongoClient` proves fallback ordering; the
client class exposes no method that issues a write; a monkeypatched failure on the
target node causes a documented fallback to the next node.

### 3.4 The vertical slice — oplog window
The first end-to-end signal. Build all four pieces, then prove a Finding.

**`collectors/base.py`** — `Collector` ABC: `signal_class` property, `collect(mongo)
-> Signal | None`. Deterministic; raises nothing on healthy clusters, returns a
Signal carrying evidence.

**`collectors/oplog_window.py`** — reads oplog stats and current write rate from
the target node. Emits a Signal with evidence:
- `oplog_size_bytes`, `oplog_used_bytes`
- `oplog_window_seconds` (time between first and last oplog entry)
- `write_rate_bytes_per_sec` (from serverStatus oplog metrics or two sampled reads)

**`detectors/base.py`** — `Detector` ABC: `evaluate(signals) -> Finding | None`.

**`detectors/oplog_window.py`** — the MongoDB expertise, encoded. The failure mode
is `oplog_window_below_resync`: if the oplog window (seconds of history it holds)
is shrinking toward the time a secondary would need to catch up after downtime,
a secondary taken down for maintenance cannot resume and needs a full initial
sync. Rule:
- Estimate `resync_seconds` conservatively (config-tunable, default assume a
  secondary could be down for `maintenance_window_seconds`, default 3600).
- If `oplog_window_seconds < resync_seconds * safety_factor` (default 2.0):
  WARNING. If `< resync_seconds`: CRITICAL.
- `horizon_seconds`: if a shrink trend is available (two samples, or Prometheus),
  project when the window crosses `resync_seconds`. Else None.
- Summary is one deterministic sentence with the real numbers.

**Acceptance (unit):** given a Signal with a healthy window (e.g. 6h against a 1h
resync), the detector returns None. Given a window of 40 min against a 1h resync,
it returns a CRITICAL Finding whose evidence includes the observed window and the
resync estimate, and whose summary states both numbers. Given 90 min, WARNING.

### 3.5 `analysis/` — the only AI stage
**`provider.py`** — `Provider` ABC: `analyze(finding, context) -> dict` returning
raw model JSON. Plus `ProviderChain` that tries primary, then fallback, with:
- a per-provider timeout and `max_retries`
- health tracking (consecutive failures)
- the failover is real: same prompt, same expected schema, both providers must be
  interchangeable. If primary returns invalid JSON after retries, fall over to
  fallback. If both fail, raise `AnalysisUnavailable` (the pipeline records the
  finding un-analyzed rather than inventing a proposal).

**`claude.py` / `openai.py`** — thin wrappers over the official SDKs. Each builds
the same prompt from a shared template, requests JSON output, returns parsed dict.
No provider-specific logic leaks upward.

**`analyst.py`** — orchestrates: takes a Finding, assembles bounded context
(the finding's evidence, cluster topology summary, the failure_mode's known
remediation patterns — NOT raw dumps), calls the ProviderChain, validates the
returned dict against `proposal.schema.json` (Pydantic model mirroring
`Proposal`), and returns a `Proposal`. Invalid output is rejected and retried;
never coerced.

The prompt is designed here but its wording is refined in the design chat, not by
Claude Code. Leave a `PROMPT_TEMPLATE` constant with a clear TODO marker and a
minimal working version so tests can run against a mocked provider.

**Acceptance (unit, mocked providers):** a mocked primary returning valid JSON
yields a Proposal referencing the finding_id. A mocked primary returning garbage
twice, with a fallback returning valid JSON, yields a Proposal with
`provider == "openai"`. Both failing raises `AnalysisUnavailable`. A provider
returning JSON that violates the schema is rejected.

### 3.6 `store/sqlite.py` — append-only audit store
SQLite, three tables: `proposals`, `approvals`, `runs`. Append-only semantics —
proposals are never updated; approval state transitions are new rows in
`approvals` keyed by proposal_id, reconstructing current state by latest
timestamp. Provides: `record_proposal`, `record_approval_transition`,
`get_proposal`, `list_pending`, `current_state(proposal_id)`.

**Acceptance:** a proposal round-trips; an approval transition sequence
(pending → approved → executed) reconstructs to EXECUTED; listing pending excludes
decided proposals.

### 3.7 `notify/` — one-way out
`base.py` `Notifier` ABC: `notify(proposal, approval_record) -> None`.
`stdout.py` — renders a proposal as readable text (for CLI and dev).
`slack.py` — posts a Block Kit message via incoming webhook (one-way). The
interactive Approve/Reject buttons reference the proposal_id; the button callback
is handled by the approval endpoint (separate, section 3.9), NOT the webhook.

**Acceptance:** stdout notifier renders all proposal fields; slack notifier builds
valid Block Kit JSON (validated structurally) without network in tests.

### 3.8 `executor/` — bounded, approval-gated, write-capable
`whitelist.py` — exactly two operations, each a function with strict arg
validation:
- `kill_op(opid)` — validates the opid is an integer the detector identified;
  issues `killOp`. Reversible in the operational sense (the op wasn't meant to run).
- `create_small_index(db, collection, keys, estimated_docs)` — refuses if
  `estimated_docs` exceeds a config threshold (default 100_000); builds in the
  background. Reversible (index can be dropped).
Any op not in the whitelist raises `ActionNotWhitelisted`.

`executor.py` — holds a **separate** write-capable MongoDB identity
(`bellwether-exec`, distinct from the read `bellwether-reader`), used only here.
`execute(proposal, approval_record)`:
- refuses unless `approval_record.state == APPROVED`
- refuses unless `proposal.action.kind == EXECUTABLE`
- dispatches to the whitelist op named by `executor_op`
- records the result back through the store (EXECUTED or FAILED)
- never shell-executes the `command` string — that field is display/audit only.

**Acceptance:** executing an un-approved proposal raises. A propose-only proposal
raises. An approved EXECUTABLE proposal with a whitelisted op calls the right
whitelist function with validated args (mocked mongo). A create-index over the doc
threshold refuses.

### 3.9 approval endpoint (`fastapi` app, in `cli.py serve` or `app.py`)
Hosted on the VPS. Receives Slack interactive callbacks. Verifies the Slack
signature (non-negotiable — an unverified approval endpoint is a hole). On a
verified Approve: records the approval transition, and if the proposal is
EXECUTABLE, invokes the executor. On Reject: records rejection. Also serves a
minimal read-only web UI listing proposals and their state.

**Acceptance:** a request with a bad signature is rejected 401. A valid Approve on
an EXECUTABLE proposal transitions state and triggers the (mocked) executor. A
valid Approve on a propose-only proposal transitions to APPROVED and does not call
the executor.

### 3.10 `pipeline.py` + `cli.py`
`pipeline.run_once(config)`: instantiate collectors → collect signals → run
detectors → for each escalating finding, call the analyst → store proposal →
notify. Record a `runs` row (started, finished, findings, proposals, errors).
Non-escalating findings are stored but not analyzed (token gate).

`cli.py`: `bellwether run` (one pipeline pass, for the systemd timer),
`bellwether serve` (the approval endpoint + UI), `bellwether list`,
`bellwether show <id>`, `bellwether approve <id> --by <name>` (manual approval
path independent of Slack, for CLI-only operators).

**Acceptance:** `run` against a mocked cluster with an induced oplog-window
finding produces a stored proposal and a stdout notification, and records a run.

---

## 4. Cross-cutting requirements

- **Typed throughout.** `mypy --strict` clean.
- **No secrets in code or YAML.** Env only; `config.py` validates presence.
- **Structured logging** (stdlib `logging`, JSON formatter). Every stage logs
  start/finish, which node served reads, provider used, tokens if available.
- **Read side has no write path.** `mongo.py` exposes no write method. Only
  `executor.py` holds a write identity, and only it is reachable past approval.
- **Deterministic before AI.** Collectors and detectors never call a model. The
  AI stage only ever sees a Finding that already passed detection.
- **Platform-agnostic.** MongoDB required; Prometheus, Slack optional. No cloud
  SDK anywhere. Config-driven so an adopter points it at their own cluster.
- **`pytest` green, `uv sync` reproducible, README gets a stranger from clone to
  a mocked `bellwether run` in under ten minutes.**

---

## 5. What Claude Code must NOT decide

- The AI prompt wording (marked TODO; refined in design).
- The two executor actions (fixed: kill_op, create_small_index — no additions).
- The identity model (fixed: read `bellwether-reader`, write `bellwether-exec`).
- Adding any agent framework, LangChain, LiteLLM, or motor.
- Widening the executor whitelist for convenience.

If a section seems to need one of these, stop and surface it as a question.
