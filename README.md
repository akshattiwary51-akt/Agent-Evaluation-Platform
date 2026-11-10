# Agent Evaluation Platform

Runs repeatable experiments against AI agents, verifies behavior against hidden ground truth,
and explains failures. **Status: Phase 5 of 6** (50-task dataset, evaluators, durable API + worker, comparison, export, dashboard).

## Try it (no API key, no spend)

```bash
make setup          # pip install -e ".[dev]"
make test           # 112 deterministic tests (no database needed)
make test-pg TEST_DATABASE_URL=...   # +38 PostgreSQL tests: queue, leases, recovery, API
make validate-dataset  # reference solver + control agents over all 50 tasks
make smoke          # scorecard of 10 scripted agents against refund_eligible_001
```

`make smoke` shows, for example, an agent that says "Your refund has been issued" without
doing anything: `execution=completed`, `evaluation=fail`, `FALSE_REFUND_CLAIM` + `NO_REFUND`.

## Real model (spends money)

```bash
ANTHROPIC_API_KEY=... python -m scripts.evaluate --model <model-id> [--in-price X --out-price Y]
```
Runs both prompt versions (`support-v1`, `support-v2`) through the tool loop. Cost prints as
`unknown` unless you pass prices (USD per million tokens). The Anthropic request mapping is
tested offline only; check the first live trace before trusting results.

## Run the platform (API + worker)

```bash
docker compose up --build          # postgres -> migrate (one-shot) -> api :8000 + worker
# or locally: make migrate && make seed && make api   (and in another shell: make worker)
curl localhost:8000/datasets       # find the dataset version id
curl localhost:8000/agent-configs  # scripted:* configs are seeded; register more via POST
curl -X POST localhost:8000/experiments -H 'Content-Type: application/json' \
     -H 'Idempotency-Key: my-first-run' \
     -d '{"dataset_version_id":"<id>","agent_config_ids":["<id>"],"split":"validation"}'
curl localhost:8000/experiments/<experiment_id>      # progress, coverage, success metrics
```
Endpoints: `/datasets`, `/agent-configs`, `/experiments` (+ `/{id}`, `/{id}/runs`, `/{id}/cancel`),
`/runs/{id}`, `/attempts/{id}/events`, `/health/live`, `/health/ready`. Interactive docs at `/docs`.
Phase 5 adds `/experiments/{id}/scorecards`, `POST /comparisons`, `/experiments/{id}/export?format=json|csv`,
`/experiments/{id}/projection`, `/prices`, and a dashboard at **`/ui`** (experiments, run detail with the
hidden ground truth in a separate panel, side-by-side comparison; the benchmark screen is a Phase 6 placeholder).
Set `APP_PASSWORD` (and optionally `APP_USER`, default `admin`) to put HTTP Basic auth on the API and dashboard;
`/health/*` stays open. Without it everything is open: don't expose it publicly that way.
Demo 2: `python -m scripts.demo_comparison` compares the careless vs careful stand-ins on the same tasks.
Queue/lease semantics: `docs/architecture.md`.

## With PostgreSQL

```bash
docker compose up --build          # postgres -> migrate (one-shot) -> api on :8000
curl localhost:8000/health/live
# or locally:
export DATABASE_URL=postgresql+psycopg://user:pass@localhost:5432/db
make migrate && make seed && python -m scripts.smoke --db
```

## Dataset (`datasets/support`, v1)
50 hand-designed tasks across the guide's 8 categories, with `manifest.json` (content hash,
splits). Splits are assigned per scenario family, stratified by category (27 dev / 11 val /
12 held-out test), so paraphrases never straddle splits. Keep the test split out of prompt tuning.
`python -m scripts.build_dataset` regenerates tasks and manifest; a test fails if they drift.

## Layout
| Path | Role |
| --- | --- |
| `app/schemas/task.py` | Task schema; `agent_view()` is the only data agents receive |
| `app/environments/support/` | Isolated per-attempt mock retailer (hidden ground truth) |
| `app/tools/gateway.py` | Allowlist, strict Pydantic validation, tracing, limits |
| `app/agents/` | Adapter protocol, scripted (deliberately broken) agents, model tool loop, prompts |
| `app/providers/` | Provider-neutral interface, Anthropic (HTTPX) and fake providers |
| `app/accounting/` | Cost estimation; unknown usage stays unknown |
| `app/evaluators/` | Outcome, policy, trajectory, response evaluators + verdict |
| `app/execution/runner.py` | Runs one attempt end to end |
| `app/execution/queue.py` | PostgreSQL job queue: claim, lease, guarded finalize, recovery, cancel |
| `app/execution/worker.py` | Worker: heartbeats, bounded concurrency, graceful shutdown |
| `app/execution/planning.py` | Experiment + runs + jobs created in one transaction, idempotent |
| `app/api/`, `app/metrics.py` | FastAPI routes; success/coverage metrics with explicit denominators |
| `app/analysis/` | Scorecards, bootstrap CIs, paired comparison + compatibility gate, cost from dated prices, export |
| `app/web/` | Jinja dashboard (`/ui`) |
| `app/db/` | SQLAlchemy models, persistence helper; `migrations/` for Alembic |

See `docs/evaluation-methodology.md` for definitions and known limitations.

## Next (guide section 27)
Phase 6: BFCL integration, versioned benchmark import, repeated trials at scale, optional judge.
