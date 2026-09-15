# Bellwether

Predictive and advisory reliability for self-hosted MongoDB replica sets.
Bellwether reads your deployment with a read-only identity, finds problems
before they bite, asks an AI model to explain each one and propose a single
remediation, and waits for a human to approve or reject it. It is a service
with an HTTP API: there is no built-in UI.

## 1. What it is

Bellwether does two different things and labels them differently.

**Predictive — trajectory.** Some problems are a line heading toward a
threshold. The oplog window detector measures the oplog's size, used bytes,
time window and the live write rate. It estimates when the window will shrink
below the time a member needs to resync after going down. That is your
maintenance window, with a safety factor. A predictive finding carries a
*horizon*: the estimated time to impact, or "already crossed".

**Advisory — patterns.** Some problems are not time-bound; they are a shape
in the workload. The index advisor sweeps every member's profiler
(`system.profile`), `$indexStats` and `$collStats`. It groups slow operations
into query shapes and reports:
- a missing index behind a collection scan, with the exact candidate index;
- indexes that look redundant or unused;
- a profiler that records nothing.

An advisory finding says *this is wasteful now*. It does not predict *this
will fail at time T*.

Every finding comes from deterministic code with real numbers in it. The AI
stage turns findings into proposals; it never produces findings.

## 2. Guarantee model

What Bellwether guarantees, and how:

- **Reads cannot write.** Collectors reach MongoDB through a client that only
  exposes read operations, and the read identity holds read roles only. The
  read side has no write path.
- **Deterministic detection.** Collectors and detectors are plain code. The
  same measurements produce the same findings.
- **The AI proposes; it cannot act.** Each escalated finding goes to the
  primary model (Claude), falling back to a second provider (OpenAI) on
  failure. The model returns one proposal:
  - a diagnosis;
  - the mechanism;
  - the impact if ignored;
  - one remediation.

  A grounding guard rejects any executable proposal that is not built
  verbatim from the finding's evidence. The model never connects to MongoDB.
- **Two actions, fixed in code.** Only `kill_op` and `create_small_index`
  can ever run. The whitelist is code; config can only narrow it.
  - `kill_op` runs only on the member where the operation is running.
  - `create_small_index` refuses any collection above
    `executor.document_threshold` documents, and re-checks the live count
    right before building.

  Everything else is *propose-only*: Bellwether shows the command and a human
  runs it. The `command` text is display and audit only; Bellwether never
  executes it.
- **Nothing runs without approval.** The executor holds a separate write
  identity. It acts only on a proposal whose approval it re-reads from the
  store, and it refuses anything tampered with.
  - Decisions come from Slack (signature-verified) or from the API (loopback
    only).
  - Both paths check the approver allowlist.
- **Append-only audit.** Proposals, approvals, findings, runs and refused
  decision attempts go to SQLite. Triggers block `UPDATE` and `DELETE`, so
  every state change is a new row.

What it does not guarantee: that the AI's diagnosis is right. That is why a
human decides, and why the proposal carries the evidence it rests on.

## 3. Requirements

- **MongoDB 4.4 or later**, self-hosted, with x.509 client authentication
  over TLS.
- **Python 3.11 or later.**
- **An Anthropic API key**, for the primary analysis provider. An OpenAI key
  is optional; without one, set `analysis.fallback_provider: null` to run
  Claude-only.
- **Slack**, optional, for notifications and one-click approval: an incoming
  webhook plus an app with interactivity pointed at `/slack/actions`.
  `bellwether serve` currently requires the Slack signing secret and at least
  one approver id even if you only use the API.
- **Prometheus**, optional. The config section exists, but no detector reads
  it today.

## 4. Replica set or single instance

**Replica set (recommended).**
- Point `mongo.target_node` at a hidden or secondary member, so diagnostic
  reads never land on the primary.
- List the other members in `fallback_nodes`, in the order to try them.
- The read URI uses `directConnection=true`; Bellwether swaps the host for
  each attempt.
- The index advisor sweeps every listed member, because the profiler and
  `$indexStats` are per member.
- The executor's `mongo_uri` must be a replica-set URI (all voting members,
  `replicaSet=<name>`, no `directConnection`). That way index builds go to
  the primary, and `kill_op` connects directly to the member running the
  operation.

**Single instance (standalone).**
- Set `target_node` to the instance and `fallback_nodes: []`.
- A standalone has no oplog. The oplog collector logs that and yields no
  signal, so there are no predictive oplog findings.
- The index advisor works unchanged.
- The executor requires a replica-set URI, so keep `executor.enabled: false`.
  To use the executor, run the instance as a single-member replica set.

## 5. The identity it needs

Bellwether uses two identities. They are never shared with anything else.

**Read identity (`bellwether-reader`)**, used by every collector:

| Role | Why |
|---|---|
| `clusterMonitor@admin` | `serverStatus`, `replSetGetStatus`, `currentOp`, `$collStats` |
| `read@local` | the oplog (`local.oplog.rs`) |
| `read@<app db>` for each application database | `system.profile`, `$indexStats` |

Granting `read` on an application database is a real widening: profiler
access implies read access to that database's documents. That is the
reason the identity is dedicated to Bellwether and audited.

**Profiler-level blind spot.** Reading a database's profiling *level* requires
the `enableProfiler` action, which only `dbAdmin` holds. Bellwether does not
ask for it. It reads what the profiler recorded, and reports an unreadable
level as `unknown (not readable without dbAdmin; not necessarily off)`.
Turning the profiler on is an operator task. Run it on every member, because
the profiler is per member:

```js
db.getSiblingDB("appdb").setProfilingLevel(1, { slowms: 100 })
```

**x.509** — the user name is the client certificate's subject in RFC 2253
form (`openssl x509 -in reader.pem -noout -subject -nameopt RFC2253`):

```js
db.getSiblingDB("$external").runCommand({
  createUser: "CN=bellwether-reader,OU=clients,O=Example,DC=example,DC=internal",
  roles: [
    { role: "clusterMonitor", db: "admin" },
    { role: "read", db: "local" },
    { role: "read", db: "appdb" }
  ],
  writeConcern: { w: "majority" }
})
```

**SCRAM**, the same roles:

```js
db.getSiblingDB("admin").createUser({
  user: "bellwether-reader",
  pwd: passwordPrompt(),
  roles: [
    { role: "clusterMonitor", db: "admin" },
    { role: "read", db: "local" },
    { role: "read", db: "appdb" }
  ]
})
```

> This release authenticates with **x.509 only**. `mongo.tls_cert_file` is
> required, and credentials in a URI are rejected at load time. The SCRAM form
> is shown for completeness; SCRAM authentication is not implemented yet.

**Write identity (`bellwether-exec`)**, used only by the executor, only after
approval. It needs `killop`, plus `find` (for the live document count) and
`createIndex` on the application database:

```js
db.getSiblingDB("admin").createRole({
  role: "bellwetherExecutor",
  privileges: [
    { resource: { cluster: true }, actions: ["killop"] },
    { resource: { db: "appdb", collection: "" }, actions: ["find", "createIndex"] }
  ],
  roles: []
})
db.getSiblingDB("$external").runCommand({
  createUser: "CN=bellwether-exec,OU=clients,O=Example,DC=example,DC=internal",
  roles: [{ role: "bellwetherExecutor", db: "admin" }]
})
```

Give it its own certificate. Leave `executor.enabled: false` until you want
approvals to act.

## 6. Configure

Copy [`config/bellwether.example.yaml`](config/bellwether.example.yaml) to
`/etc/bellwether/bellwether.yaml` and replace the placeholders. The file
comes from `--config` or `BELLWETHER_CONFIG`. The core of it:

```yaml
mongo:
  uri: "mongodb://mongo-hidden.example.internal:27017/?authMechanism=MONGODB-X509&authSource=%24external&tls=true&directConnection=true"
  tls_ca_file: /etc/bellwether/tls/ca.pem
  tls_cert_file: /etc/bellwether/tls/bellwether-reader.pem   # cert + key, one PEM
  target_node: mongo-hidden.example.internal:27017
  fallback_nodes:
    - mongo-1.example.internal:27017
    - mongo-2.example.internal:27017
    - mongo-3.example.internal:27017

analysis:
  primary_provider: claude
  fallback_provider: openai        # or null
  claude_model: claude-opus-5
  openai_model: gpt-5.6-sol

notify:
  channels: [stdout, slack]

approval:
  approver_ids: ["U0123ABCD"]      # Slack user IDs; also the API's allowlist
  ui_approval_enabled: true        # the API decision endpoint (loopback only)

store:
  sqlite_path: /var/lib/bellwether/bellwether.db   # the append-only audit store

collectors:
  query_profile:
    databases: [appdb]

executor:
  enabled: false
```

**Certificates never go in a URI.** URIs carry the host and auth options only;
the certificate and CA are the `tls_*` fields. A `tlsCertificateKeyFile` in a
URI is a load-time error.

**Secrets never go in the YAML.** A secret found there is a load-time error.
Supply secrets through environment variables, or through a `.env` file in the
working directory for local development. `.env` is gitignored; never commit
it. Precedence is YAML, then `.env`, then environment variables
(`BELLWETHER_<SECTION>__<KEY>`), with later sources winning:

```sh
BELLWETHER_ANALYSIS__ANTHROPIC_API_KEY=...
BELLWETHER_ANALYSIS__OPENAI_API_KEY=...          # if openai is a provider
BELLWETHER_NOTIFY__SLACK_WEBHOOK_URL=...         # if slack is a channel
BELLWETHER_NOTIFY__SLACK_SIGNING_SECRET=...      # if slack is a channel; required by serve
BELLWETHER_MONGO__TLS_CERT_PASSPHRASE=...        # only if the key is encrypted
BELLWETHER_EXECUTOR__TLS_CERT_PASSPHRASE=...
```

Any non-secret key can be overridden the same way; lists take JSON, e.g.
`BELLWETHER_APPROVAL__APPROVER_IDS='["U0123ABCD"]'`.

## 7. Run

Install:

```sh
git clone https://github.com/OsegoTech/bellwether /opt/bellwether
cd /opt/bellwether && python3 -m venv .venv && .venv/bin/pip install .
```

**One pipeline pass**: collect, detect, analyse, notify. It exits non-zero if
any stage recorded an error.

```sh
bellwether run
```

Run it on a timer with the units in [`deploy/systemd/`](deploy/systemd/).
Put the secrets in `/etc/bellwether/bellwether.env` (mode `0600`) as
`KEY=value` lines.

`bellwether-run.service`:

```ini
[Unit]
Description=Bellwether pipeline pass
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
User=bellwether
Group=bellwether
EnvironmentFile=/etc/bellwether/bellwether.env
Environment=BELLWETHER_CONFIG=/etc/bellwether/bellwether.yaml
ExecStart=/opt/bellwether/.venv/bin/bellwether run
TimeoutStartSec=15min
StateDirectory=bellwether
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
PrivateTmp=true
ReadWritePaths=/var/lib/bellwether
```

`bellwether-run.timer`:

```ini
[Unit]
Description=Bellwether pipeline pass every five minutes

[Timer]
OnCalendar=*:0/5
Persistent=true
Unit=bellwether-run.service

[Install]
WantedBy=timers.target
```

```sh
sudo cp deploy/systemd/bellwether-run.* /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now bellwether-run.timer
```

In a unit file, a literal `%` must be written `%%`. So if you put a URI in an
`Environment=` line, `%24external` becomes `%%24external`.

**The API:**

```sh
bellwether serve --host 127.0.0.1 --port 8080
```

`deploy/systemd/bellwether-api.service` runs it the same way. Keep it on
loopback and reach it over an SSH tunnel
(`ssh -L 8080:127.0.0.1:8080 monitoring-host`).

For Slack's callback, put a TLS reverse proxy in front that forwards **only**
`POST /slack/actions`. The decision endpoint's gate is the server's bind
address, not the client's. A proxy that forwards everything would expose it.

Also on the command line: `bellwether list [--pending]`,
`bellwether show <id>`, and `bellwether approve <id> --by NAME`.

## 8. The API

The interactive reference is at **`/docs`** (Swagger UI) and **`/redoc`**. The
schema is at `/openapi.json`. Every response is JSON. Errors have one shape
and one of the codes below:

```json
{ "error": "not_found", "detail": "no proposal 5b0c…" }
```

The codes are `malformed_request` (400), `acknowledgement_required` (400),
`approval_not_permitted` (403), `not_authorized` (403), `not_found` (404) and
`already_decided` (409).

| Method | Path | What it returns |
|---|---|---|
| `GET` | `/healthz` | `{"ok": true}` |
| `GET` | `/api/proposals?state=&limit=50` | Proposals, newest first, each with its state and `created_at` |
| `GET` | `/api/proposals/{id}` | One proposal in full: evidence, current approval record, transitions, audit events |
| `GET` | `/api/proposals/{id}/audit` | The audit trail: transitions, refused attempts, execution result |
| `GET` | `/api/findings?run_id=&limit=50` | Findings, including INFO findings that were only noted |
| `GET` | `/api/runs?limit=50` | Pipeline runs with counts and errors |
| `POST` | `/api/proposals/{id}/decision` | Approve or reject (loopback only, see below) |
| `POST` | `/slack/actions` | Slack interactive callback (signature-verified) |

List pending proposals:

```sh
curl -s 'http://127.0.0.1:8080/api/proposals?state=pending&limit=10'
```

Approve one:

```sh
curl -s -X POST http://127.0.0.1:8080/api/proposals/<proposal_id>/decision \
  -H 'content-type: application/json' \
  -d '{"decision": "approve", "approver_id": "U0123ABCD", "acknowledged": true}'
```

The decision endpoint returns 403 unless the server is bound to a loopback
address and `approval.ui_approval_enabled` is true. The `approver_id` must be
in `approval.approver_ids`. An id outside the list is refused, changes
nothing, and is recorded as an `unauthorized_decision` audit event.

Approving requires `"acknowledged": true`. Only a `pending` proposal moves.
An approved executable proposal goes to the executor through the same code
path as a Slack approval. Read the proposal back to see `executed` or
`failed`.

The API has no authentication of its own: on loopback, host access *is* the
authentication. Do not expose it publicly.

## 9. What it is NOT

- **Not autonomous.** Nothing changes your database without a recorded human
  approval. Only two small, bounded actions can ever execute.
- **Not a monitoring stack.** No dashboards, no metrics storage, no alert
  routing. Keep Prometheus and your pager; Bellwether adds findings with
  reasons.
- **Not a UI.** It is an API with generated docs. Build a frontend on
  `/openapi.json` if you want one.
- **Not for sharded clusters or Atlas.** It targets self-hosted replica sets,
  and standalones with reduced coverage.
- **Not a query rewriter or schema designer.** It suggests indexes from
  measured shapes and explains problems. Anything beyond the two actions is
  a command for a human to run.
- **Not a guarantee of correctness.** The analysis is a model's reading of
  the evidence. The evidence is shown so you can check it.

## 10. License

Apache License 2.0 — see [LICENSE](LICENSE).
