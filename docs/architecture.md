# Architecture notes (Phase 4)

```
client -> FastAPI --(one transaction)--> experiments, runs, jobs  [PostgreSQL]
                                              ^
worker threads: claim (SKIP LOCKED) -> start attempt -> run agent -> guarded finalize
                heartbeat thread: extend lease, watch for cancel      sweeper: recover expired leases
```

## Job lifecycle
`queued -> running -> succeeded | failed | cancelled`, or `running -> queued` (retry).

| Event | What happens |
| --- | --- |
| Claim | One short transaction; `FOR UPDATE SKIP LOCKED`; sets worker id, lease token, expiry |
| Heartbeat | Extends the lease only if the caller still holds the token; also polls the cancel flag |
| Finalize | One transaction: token-guarded job update **and** attempt/trace/score writes. Zero rows matched means a stale worker, so nothing is written |
| Agent failure (wrong result, limits) | Scored and final. Never retried |
| Infrastructure error | New attempt with fresh environment, up to `max_retries`; failed attempt and its model usage are kept; run ends `infrastructure_error`, unscored, if retries run out |
| Worker killed / lease expired | Sweeper marks the attempt `interrupted` (with an `attempt_interrupted` event), requeues the job (bounded), replacement gets a fresh environment |
| Cancel | Queued jobs cancelled immediately; running attempts stop before their next model call or tool execution; cancelled runs are never scored |

A run's counted result is `runs.final_attempt_id`, set only inside the guarded finalize, so a
result cannot be counted twice. Experiments are created with `Idempotency-Key`: same key and
payload returns the same experiment; same key with a different payload is a 409; concurrent
requests with one key create exactly one experiment (unique constraint, loser re-reads).

## Known limitations
- **Delivery is at-least-once.** Agent tools run against an in-memory environment, so a replayed
  attempt cannot double-charge anyone, but a real tool integration would need its own idempotency.
- **A killed worker loses its in-flight trace and token usage.** Events and usage are written at
  finalize, so an `interrupted` attempt shows only the recovery event, and its model spend is not
  recorded (real cost is understated for crashes). Fix if it matters: flush events incrementally.
  Likewise a worker that loses its lease discards its result by design (its spend is lost too).
- **Cancel latency is bounded by the heartbeat interval** (lease/3 by default). A model call
  already in flight finishes; only the next call/tool is prevented.
- **Sweeping runs inside workers.** If every worker is down, expired leases wait until one returns.
- **Experiment `status` is derived on read** from run states; the stored column is only a marker.
- Token/cost aggregation, comparisons, export and the dashboard are Phase 5.
