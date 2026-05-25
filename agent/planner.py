"""Task planner (Phase 2 of the agentic upgrade).

When a user query has multiple steps, decompose it into 2-5 subtasks, run
each through an auxiliary-tier-routed call, then re-compose the results
into one coherent answer.

The planner runs OFF the main AIAgent path. It is invoked by the API
server's chat-completions handler when:

  HERMES_PLANNER_ENABLED=true (default) AND
  TaskPlanner.should_plan(query) returns True

When invoked, it returns a final answer string AND a structured plan record
so the distiller (Phase 4) can promote successful multi-step procedures into
reusable skills.

All LLM work uses `agent.auxiliary_client.get_text_auxiliary_client`, which
already implements the provider chain fallback. Per-subtask tier routing
(cheap / mid / big) reads the same VAULT_ROUTE_*_MODEL env vars as
`agent.vault_router`, so a single config knob shapes both vault-aware
routing and planner subtask routing.

Output structure exposed to caller:
  {
    "answer": str,                 # final recomposed response
    "plan": [Subtask, ...],        # the decomposition
    "results": [{id, content}],    # per-subtask outputs
    "model_per_subtask": [...],    # what model handled each
    "success": bool,
    "elapsed_ms": int,
  }
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Heuristics — cheap regex pass to decide whether decomposition is worth it.
# ---------------------------------------------------------------------------

_MULTI_VERB_HINT = re.compile(
    r"\b(?:and then|after that|first[,. ]|then[,. ]|next[,. ]|finally|"
    r"step \d|step one|step two|once you|once that|once we|"
    r"in parallel|while you|in the meantime)\b",
    re.IGNORECASE,
)
_VERB_HINT = re.compile(
    r"\b(?:find|fetch|search|read|write|summarize|compare|list|generate|"
    r"create|build|deploy|run|test|analyze|extract|combine|merge|"
    r"refactor|debug|investigate|review|check|verify|email|send|post|"
    r"download|upload|push|pull|commit|publish)\b",
    re.IGNORECASE,
)


def should_plan(query: str, *, min_chars: int = 120, min_verbs: int = 3) -> bool:
    """Cheap decision: is this query worth decomposing?

    Triggers when EITHER:
      - explicit multi-step language ("and then", "first", "step 2", ...)
      - long query (>= min_chars) AND >= min_verbs distinct action verbs
    """
    if not query or not isinstance(query, str):
        return False
    if os.environ.get("HERMES_PLANNER_ENABLED", "true").lower() not in ("1", "true", "yes", "on"):
        return False
    if _MULTI_VERB_HINT.search(query):
        return True
    if len(query) < min_chars:
        return False
    verbs = set(m.group(0).lower() for m in _VERB_HINT.finditer(query))
    return len(verbs) >= min_verbs


# ---------------------------------------------------------------------------
# Subtask record
# ---------------------------------------------------------------------------

@dataclass
class Subtask:
    id: str
    goal: str
    depends_on: List[str] = field(default_factory=list)
    suggested_tier: str = "mid"   # cheap | mid | big
    success_criteria: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Tier resolution — shares env vars with agent.vault_router for consistency
# ---------------------------------------------------------------------------

DEFAULT_CHEAP_MODEL = "groq/llama-3.3-70b-versatile"
DEFAULT_MID_MODEL = "gemini-2.5-flash"
DEFAULT_BIG_MODEL = "gemini-2.5-pro"


def _tier_model(tier: str, fallback: str = "") -> str:
    env_key = {
        "cheap": "VAULT_ROUTE_CHEAP_MODEL",
        "mid": "VAULT_ROUTE_MID_MODEL",
        "big": "VAULT_ROUTE_BIG_MODEL",
    }.get(tier, "")
    override = os.environ.get(env_key, "").strip() if env_key else ""
    if override:
        return override
    defaults = {"cheap": DEFAULT_CHEAP_MODEL, "mid": DEFAULT_MID_MODEL, "big": DEFAULT_BIG_MODEL}
    return defaults.get(tier) or fallback or DEFAULT_MID_MODEL


# ---------------------------------------------------------------------------
# Auxiliary call shared with reflector — kept local to keep the module
# self-contained (no cross-module function dependency).
# ---------------------------------------------------------------------------

def _aux_call(*, system: str, user: str, model_override: Optional[str] = None,
              max_tokens: int = 1500, temperature: float = 0.3) -> Optional[str]:
    try:
        from agent.auxiliary_client import get_text_auxiliary_client
    except Exception as exc:
        logger.debug("planner: auxiliary_client import failed: %s", exc)
        return None
    try:
        client, model = get_text_auxiliary_client(task="planner")
    except Exception as exc:
        logger.debug("planner: get_text_auxiliary_client failed: %s", exc)
        return None
    if client is None or not model:
        logger.debug("planner: no auxiliary backend resolved")
        return None
    use_model = model_override or model
    try:
        resp = client.chat.completions.create(
            model=use_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as exc:
        logger.debug("planner: chat.completions.create failed (%s): %s", use_model, exc)
        return None


# ---------------------------------------------------------------------------
# Decomposition
# ---------------------------------------------------------------------------

_DECOMPOSE_SYSTEM = (
    "You are a task planner. Given a user request, break it into 2-5 sequential or "
    "parallel subtasks that, when executed, will completely answer the request.\n\n"
    "Return ONLY a JSON array (no prose) of the form:\n"
    "[\n"
    "  {\"id\": \"t1\", \"goal\": \"...\", \"depends_on\": [], \"tier\": \"cheap|mid|big\", \"success_criteria\": \"...\"},\n"
    "  ...\n"
    "]\n\n"
    "Rules:\n"
    "- 2-5 subtasks max. Smaller is better.\n"
    "- IDs are short like t1, t2, t3.\n"
    "- depends_on lists subtask IDs that must complete first. Empty list = can run in parallel.\n"
    "- Tier: \"cheap\" for routine lookups/formatting; \"mid\" for normal reasoning; \"big\" for hard reasoning, "
    "long-context synthesis, math, or code generation.\n"
    "- success_criteria is one short sentence describing how to know the subtask worked.\n"
    "- If the request is actually atomic (single step), return a one-element array.\n"
    "- Return raw JSON only — no markdown fences, no prose."
)

_JSON_ARRAY_RE = re.compile(r"\[[\s\S]*\]")


def _parse_subtasks(text: str) -> List[Subtask]:
    if not text:
        return []
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE | re.MULTILINE)
    try:
        parsed = json.loads(cleaned)
    except Exception:
        m = _JSON_ARRAY_RE.search(cleaned)
        if not m:
            return []
        try:
            parsed = json.loads(m.group(0))
        except Exception:
            return []
    if not isinstance(parsed, list):
        return []
    out: List[Subtask] = []
    for i, item in enumerate(parsed[:5]):
        if not isinstance(item, dict):
            continue
        sid = str(item.get("id") or f"t{i+1}").strip()[:8] or f"t{i+1}"
        goal = str(item.get("goal") or "").strip()
        if not goal:
            continue
        deps = item.get("depends_on") or []
        if not isinstance(deps, list):
            deps = []
        deps = [str(d)[:8] for d in deps if d]
        tier = str(item.get("tier") or "mid").lower().strip()
        if tier not in ("cheap", "mid", "big"):
            tier = "mid"
        crit = str(item.get("success_criteria") or "").strip()[:200]
        out.append(Subtask(id=sid, goal=goal[:400], depends_on=deps, suggested_tier=tier, success_criteria=crit))
    return out


def decompose(query: str, *, context: str = "") -> List[Subtask]:
    """Auxiliary-LLM call: query → 2-5 Subtasks. Empty list = couldn't decompose."""
    user_prompt = f"REQUEST:\n{query.strip()[:2000]}"
    if context:
        user_prompt += f"\n\nCONTEXT (recent memory):\n{context.strip()[:1500]}"
    user_prompt += "\n\nReturn JSON array only."
    # Decomposition itself uses the mid tier — it's a reasoning task but small.
    raw = _aux_call(system=_DECOMPOSE_SYSTEM, user=user_prompt,
                    model_override=_tier_model("mid"),
                    max_tokens=900, temperature=0.2)
    if not raw:
        return []
    subtasks = _parse_subtasks(raw)
    if not subtasks:
        logger.debug("planner: decompose returned no valid subtasks (raw=%r)", raw[:200])
    return subtasks


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

_EXECUTE_SYSTEM_TPL = (
    "You are executing one subtask of a larger plan. The user's overall request "
    "and prior subtask results are provided as context.\n\n"
    "Your job: complete THIS subtask and ONLY this subtask. Return the result as "
    "concise prose or structured text (whatever fits). No meta-commentary, no "
    "\"here is the result:\" preambles — just the result itself.\n\n"
    "Success criterion: {criterion}\n"
)


def _execute_subtask(subtask: Subtask, *, overall_query: str,
                     prior_results: Dict[str, str]) -> Tuple[str, str]:
    """Run one subtask. Returns (model_used, result_text). result_text may be
    an error placeholder if the call fails — never raises."""
    deps_block = ""
    if subtask.depends_on:
        relevant = [(d, prior_results.get(d, "")) for d in subtask.depends_on]
        if any(v for _, v in relevant):
            deps_block = "\n\nPRIOR RESULTS:\n" + "\n".join(
                f"[{d}]\n{v.strip()[:1500]}" for d, v in relevant if v
            )
    user = (
        f"OVERALL REQUEST:\n{overall_query.strip()[:1500]}\n\n"
        f"YOUR SUBTASK ({subtask.id}):\n{subtask.goal}"
        f"{deps_block}\n\n"
        "Complete this subtask now."
    )
    system = _EXECUTE_SYSTEM_TPL.format(criterion=subtask.success_criteria or "(none specified)")
    model = _tier_model(subtask.suggested_tier)
    raw = _aux_call(system=system, user=user, model_override=model,
                    max_tokens=1500, temperature=0.4)
    if not raw:
        return model, f"[subtask {subtask.id} failed: no model output]"
    return model, raw


def execute_all(subtasks: List[Subtask], *, overall_query: str) -> Dict[str, Dict[str, Any]]:
    """Run subtasks in dependency order, parallel where possible.

    Returns: {subtask_id: {"result": str, "model": str, "elapsed_ms": int}}
    """
    results: Dict[str, Dict[str, Any]] = {}
    pending = {s.id: s for s in subtasks}
    completed: set = set()

    # Topological wave loop: each wave runs all subtasks whose deps are completed.
    safety_iters = 0
    while pending and safety_iters < 20:
        safety_iters += 1
        wave = [s for s in pending.values() if all(d in completed for d in s.depends_on)]
        if not wave:
            # Circular dep or missing dep — bail.
            logger.debug("planner: stuck wave, pending=%s", list(pending.keys()))
            for s in pending.values():
                results[s.id] = {"result": f"[subtask {s.id} skipped: unmet dependency]",
                                  "model": "(none)", "elapsed_ms": 0}
            break
        # Run wave in parallel.
        prior = {sid: r["result"] for sid, r in results.items()}
        with ThreadPoolExecutor(max_workers=min(len(wave), 4)) as ex:
            futures = {ex.submit(_run_one, s, overall_query, prior): s for s in wave}
            for fut in as_completed(futures):
                s = futures[fut]
                try:
                    model, result, elapsed = fut.result()
                except Exception as exc:
                    model, result, elapsed = "(error)", f"[subtask {s.id} crashed: {exc}]", 0
                results[s.id] = {"result": result, "model": model, "elapsed_ms": elapsed}
                completed.add(s.id)
                pending.pop(s.id, None)
                logger.info("hermes.planner.subtask id=%s model=%s elapsed=%dms",
                            s.id, model, elapsed)
    return results


def _run_one(subtask: Subtask, overall_query: str,
             prior: Dict[str, str]) -> Tuple[str, str, int]:
    t0 = time.time()
    model, result = _execute_subtask(subtask, overall_query=overall_query, prior_results=prior)
    elapsed_ms = int((time.time() - t0) * 1000)
    return model, result, elapsed_ms


# ---------------------------------------------------------------------------
# Recomposition
# ---------------------------------------------------------------------------

_RECOMPOSE_SYSTEM = (
    "You are a synthesizer. The user asked a question. A plan was created and "
    "executed; you have the subtask results below. Synthesize them into ONE "
    "coherent answer addressed to the user. Be concise but complete. Do NOT "
    "mention the plan, subtasks, or that decomposition happened — answer as if "
    "you reasoned it through normally. If a subtask failed, acknowledge what's "
    "missing rather than fabricating."
)


def recompose(query: str, subtasks: List[Subtask], results: Dict[str, Dict[str, Any]]) -> Optional[str]:
    if not subtasks:
        return None
    blocks = []
    for s in subtasks:
        r = results.get(s.id) or {}
        text = r.get("result") or "[no result]"
        blocks.append(f"[{s.id} — {s.goal}]\n{text.strip()[:2500]}")
    payload = "\n\n".join(blocks)
    user = (
        f"ORIGINAL REQUEST:\n{query.strip()[:1500]}\n\n"
        f"SUBTASK RESULTS:\n{payload}\n\n"
        "Synthesize into one coherent answer for the user."
    )
    return _aux_call(system=_RECOMPOSE_SYSTEM, user=user,
                     model_override=_tier_model("mid"),
                     max_tokens=2000, temperature=0.4)


# ---------------------------------------------------------------------------
# Full pipeline entrypoint
# ---------------------------------------------------------------------------

def run(query: str, *, context: str = "") -> Optional[Dict[str, Any]]:
    """Decompose → execute → recompose. Returns None if planning aborted
    (no decomposition possible, no auxiliary backend, etc.).

    Caller is responsible for `should_plan(query)` check before invoking.
    """
    t0 = time.time()
    subtasks = decompose(query, context=context)
    if not subtasks or len(subtasks) < 2:
        # Single-task or empty — not worth the planner overhead, let the
        # normal agent path handle it.
        logger.info("hermes.planner.decision skipped subtasks=%d", len(subtasks))
        return None
    logger.info("hermes.planner.decision subtasks=%d tiers=%s",
                len(subtasks), [s.suggested_tier for s in subtasks])
    results = execute_all(subtasks, overall_query=query)
    answer = recompose(query, subtasks, results)
    if not answer:
        # If recompose failed, at least concatenate subtask outputs.
        answer = "\n\n".join(
            f"{s.goal}: {(results.get(s.id) or {}).get('result', '[no result]')}"
            for s in subtasks
        )
    elapsed_ms = int((time.time() - t0) * 1000)
    success = all(
        not (results.get(s.id) or {}).get("result", "").startswith("[subtask ")
        for s in subtasks
    )
    return {
        "answer": answer,
        "plan": [s.as_dict() for s in subtasks],
        "results": [{"id": sid, "model": r.get("model"), "elapsed_ms": r.get("elapsed_ms"),
                      "content": r.get("result")} for sid, r in results.items()],
        "success": success,
        "elapsed_ms": elapsed_ms,
    }
