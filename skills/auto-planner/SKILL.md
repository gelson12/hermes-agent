---
name: auto-planner
description: Decompose multi-step user requests into a plan, execute each subtask through tier-routed auxiliary models, and synthesize the results — invoked automatically by the gateway when a request looks multi-step.
version: 1.0.0
platforms: all
tags: [planning, agentic, auto]
---

# auto-planner

When a user request has multiple steps (e.g. *"find X, then summarize Y, then write Z"*) Hermes's gateway routes the request through the planner pipeline *before* the normal AIAgent loop runs. This skill documents that pipeline so anyone debugging or extending it has the canonical reference.

## When this fires

`agent/planner.py:should_plan()` decides. It triggers when **either**:

- The request contains explicit multi-step markers: *"and then"*, *"first… then…"*, *"step 1/2"*, *"after that"*, *"in parallel"*, etc.
- The request is long (≥120 chars) AND contains ≥3 distinct action verbs (`find`, `fetch`, `summarize`, `compare`, `write`, `deploy`, `test`, etc.).

It's a cheap regex pass — no LLM tokens spent on this decision.

## Procedure

1. **Decompose** — auxiliary model (mid tier, gemini-2.5-flash by default) returns a JSON array of 2-5 subtasks, each with `{id, goal, depends_on, tier, success_criteria}`.
2. **Execute** — subtasks run in dependency-aware waves. Independents run in parallel via `ThreadPoolExecutor` (max 4 workers); dependents wait for their `depends_on` to resolve. Each subtask uses the tier-routed model (`cheap` = Groq llama-3.3-70b, `mid` = Gemini Flash, `big` = Gemini Pro).
3. **Recompose** — auxiliary model (mid tier) synthesizes the subtask outputs into a single coherent answer addressed to the user. No mention of "I decomposed this into subtasks" — the user sees a normal response.
4. **Reflect** (Phase 3) — post-turn, the reflector scores the run. If it scores high (success=true, confidence≥0.7), the distiller (Phase 4) writes a reusable SKILL.md under `~/.hermes/skills/auto/<slug>/`.

## Notes

- **Gating**: `HERMES_PLANNER_ENABLED=true` (default). Flip to false to bypass the planner entirely; the normal AIAgent loop will handle the request.
- **Tier env vars** (shared with `agent/vault_router.py`):
  - `VAULT_ROUTE_CHEAP_MODEL` — default `groq/llama-3.3-70b-versatile`
  - `VAULT_ROUTE_MID_MODEL` — default `gemini-2.5-flash`
  - `VAULT_ROUTE_BIG_MODEL` — default `gemini-2.5-pro`
- **Failure handling**: if any subtask returns `[subtask <id> failed: …]`, the reflector marks `success=false` and the distiller skips. The user still gets the recomposed answer (with the failure acknowledged rather than fabricated around).
- **No real subagent delegation** in v1.0: subtasks run in-process via the auxiliary client. Real `tools/delegate_tool.py` subagent delegation is a future upgrade — switch is on the `_run_one` execution path in `agent/planner.py`.
- **Token budget**: all planner LLM work uses the auxiliary client, separate from the main response budget. The user's chat completion stays in its normal token envelope.

## Logging

Service logs to grep for:
- `hermes.planner.decision subtasks=N tiers=[...]` — planner ran on a request
- `hermes.planner.subtask id=tN model=… elapsed=…ms` — per-subtask completion
- `hermes.reflector.score conf=… success=… …` — post-turn reflection
- `hermes.distiller.created path=… slug=…` — new skill written

## Related files

- `agent/planner.py` — the pipeline itself
- `agent/vault_router.py` — tier resolution (shared)
- `plugins/memory/vault/reflection.py` — Phase 3 post-turn critique
- `plugins/memory/vault/distiller.py` — Phase 4 skill auto-generation
- `gateway/platforms/api_server.py` — invocation site (chat-completions handler)
