"""Skill auto-generator (Phase 4 of the agentic upgrade).

When the planner runs a multi-step task AND reflection scores it as
successful (confidence >= 0.7, success=true), the distiller asks an
auxiliary LLM to summarize the procedure into a reusable SKILL.md file
under `~/.hermes/skills/auto/<slug>/`.

The curator (`agent/curator.py`, already shipped) periodically reviews
agent-created skills and can promote, archive, or refine them — so the
distiller produces drafts and lets the curator handle lifecycle.

Env gate: HERMES_DISTILLER_ENABLED (default true).
"""

from __future__ import annotations

import logging
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gating
# ---------------------------------------------------------------------------

def should_distill(plan: List[Dict[str, Any]], results: List[Dict[str, Any]],
                   reflection: Optional[Dict[str, Any]]) -> bool:
    """Hard gate before we spend tokens on distillation.

    Conditions (all required):
      - planner produced 2+ subtasks
      - all subtasks completed without [error] markers
      - reflection succeeded AND confidence >= 0.7 AND success == true
      - distiller env-enabled
    """
    if os.environ.get("HERMES_DISTILLER_ENABLED", "true").lower() not in ("1", "true", "yes", "on"):
        return False
    if not plan or len(plan) < 2:
        return False
    if not results or len(results) < len(plan):
        return False
    for r in results:
        content = (r.get("content") or "").strip()
        if not content or content.startswith("[subtask "):
            return False
    if not reflection or not isinstance(reflection, dict):
        return False
    if not reflection.get("success"):
        return False
    try:
        if float(reflection.get("confidence", 0)) < 0.7:
            return False
    except Exception:
        return False
    return True


# ---------------------------------------------------------------------------
# Skill path + slug
# ---------------------------------------------------------------------------

def _kebab(text: str, max_len: int = 48) -> str:
    """Lowercase kebab-case slug, alphanumeric only."""
    s = re.sub(r"[^a-z0-9\s-]+", "", (text or "").lower())
    s = re.sub(r"\s+", "-", s.strip())
    s = re.sub(r"-+", "-", s)
    return s[:max_len].strip("-") or "untitled"


def _skills_root() -> Path:
    """Resolve ~/.hermes/skills/auto/ — honors HERMES_HOME env."""
    home = os.environ.get("HERMES_HOME", "").strip()
    if home:
        base = Path(home)
    else:
        base = Path.home() / ".hermes"
    return base / "skills" / "auto"


# ---------------------------------------------------------------------------
# Auxiliary call
# ---------------------------------------------------------------------------

_DISTILL_SYSTEM = (
    "You are a skill author. Given a multi-step procedure that just worked, "
    "write a reusable SKILL.md file that future invocations can use to repeat it.\n\n"
    "Format STRICTLY:\n"
    "---\n"
    "name: <slug, lowercase-with-dashes, <=40 chars>\n"
    "description: <one sentence describing when to use this skill, <=200 chars>\n"
    "version: 1.0.0-auto\n"
    "platforms: all\n"
    "tags: [auto, <2-3 topic tags>]\n"
    "---\n\n"
    "# <Title>\n\n"
    "<Short intro: when to use this skill (1-2 sentences).>\n\n"
    "## Procedure\n\n"
    "1. <step one>\n"
    "2. <step two>\n"
    "...\n\n"
    "## Notes\n\n"
    "<Key gotchas, prerequisites, or things to remember.>\n\n"
    "Be concrete and specific. Do not invent things that weren't in the original procedure.\n"
    "Return the SKILL.md content only — no fences, no preamble, no postscript."
)


def _aux_call(*, system: str, user: str, model_override: Optional[str] = None,
              max_tokens: int = 1500, temperature: float = 0.3) -> Optional[str]:
    try:
        from agent.auxiliary_client import get_text_auxiliary_client
    except Exception:
        return None
    try:
        client, model = get_text_auxiliary_client(task="distiller")
    except Exception:
        return None
    if client is None or not model:
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
        logger.debug("distiller: chat.completions.create failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def distill_async(query: str, plan: List[Dict[str, Any]],
                  results: List[Dict[str, Any]],
                  reflection: Dict[str, Any]) -> None:
    """Fire-and-forget skill generation."""
    thread = threading.Thread(
        target=_do_distill,
        args=(query, plan, results, reflection),
        name="hermes-distiller",
        daemon=True,
    )
    thread.start()


def _do_distill(query: str, plan: List[Dict[str, Any]],
                results: List[Dict[str, Any]],
                reflection: Dict[str, Any]) -> None:
    # Build the user prompt: original query + plan + results.
    plan_block = "\n".join(
        f"- {s.get('id', '?')}: {s.get('goal', '')} [tier={s.get('suggested_tier', 'mid')}]"
        for s in plan
    )
    results_block = "\n\n".join(
        f"[{r.get('id', '?')}]\n{(r.get('content') or '').strip()[:1200]}"
        for r in results
    )
    user_prompt = (
        f"ORIGINAL REQUEST:\n{query.strip()[:1200]}\n\n"
        f"SUCCESSFUL PLAN ({len(plan)} subtasks):\n{plan_block}\n\n"
        f"SUBTASK RESULTS:\n{results_block}\n\n"
        f"REFLECTION: confidence={reflection.get('confidence', '?')}, "
        f"tags={reflection.get('tags', [])}\n\n"
        "Write the SKILL.md."
    )
    raw = _aux_call(system=_DISTILL_SYSTEM, user=user_prompt, max_tokens=1500, temperature=0.3)
    if not raw:
        logger.debug("distiller: no aux output")
        return

    # Parse out the skill name from frontmatter for the directory slug.
    name_match = re.search(r"^name:\s*([a-z0-9-]+)", raw, re.MULTILINE)
    if name_match:
        slug = _kebab(name_match.group(1))
    else:
        # Fall back to slugifying the first 6 words of the query.
        slug = _kebab(" ".join((query or "").split()[:6]))

    skills_dir = _skills_root() / slug
    try:
        skills_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        logger.debug("distiller: mkdir failed for %s: %s", skills_dir, exc)
        return

    skill_path = skills_dir / "SKILL.md"
    # If a same-named auto skill already exists, append a timestamp suffix
    # rather than clobbering.  Curator can dedupe later.
    if skill_path.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        skills_dir = _skills_root() / f"{slug}-{stamp}"
        try:
            skills_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            logger.debug("distiller: mkdir variant failed for %s: %s", skills_dir, exc)
            return
        skill_path = skills_dir / "SKILL.md"

    try:
        skill_path.write_text(raw, encoding="utf-8")
        logger.info("hermes.distiller.created path=%s slug=%s", skill_path, slug)
    except Exception as exc:
        logger.debug("distiller: write failed: %s", exc)
        return

    # Also drop a sibling .metadata.json so the curator + insights engine can
    # tell this is auto-generated and trace its provenance.
    try:
        import json as _json
        (skills_dir / ".metadata.json").write_text(
            _json.dumps({
                "origin": "auto-distiller",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source_query": query[:600],
                "subtask_count": len(plan),
                "reflection_confidence": reflection.get("confidence"),
                "reflection_tags": reflection.get("tags"),
            }, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass
