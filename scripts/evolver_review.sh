#!/usr/bin/env bash
# Review the prompt-evolver's PROPOSED prompt addenda (candidates) before anything is
# promoted, and optionally promote one deliberately. Talks to the live Hermes API
# (the vault is internal to Railway, so review must go through the gateway).
#
#   API_SERVER_KEY=... ./scripts/evolver_review.sh [DOMAIN]
#   API_SERVER_KEY=... ./scripts/evolver_review.sh --promote <DOMAIN> <VARIANT_ID>
#
# HERMES_URL defaults to the production gateway.
set -euo pipefail
HERMES_URL="${HERMES_URL:-https://hermes-agent-production-027d.up.railway.app}"
KEY="${API_SERVER_KEY:?set API_SERVER_KEY}"

if [ "${1:-}" = "--promote" ]; then
  DOM="${2:?domain}"; VID="${3:?variant_id}"
  echo "Promoting candidate $VID for domain '$DOM' (deliberate)…"
  curl -s -X POST "$HERMES_URL/v1/evolver/promote" \
    -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
    -d "{\"domain\":\"$DOM\",\"variant_id\":\"$VID\"}" | python -m json.tool
  exit 0
fi

DOM="${1:-}"
Q=""; [ -n "$DOM" ] && Q="?domain=$DOM"
echo "=== Prompt-evolver candidates${DOM:+ for domain '$DOM'} ==="
curl -s "$HERMES_URL/v1/evolver/candidates$Q" -H "Authorization: Bearer $KEY" | python - <<'PY'
import sys, json
d = json.load(sys.stdin)
print(f"enabled={d.get('enabled')}  domain={d.get('domain')}")
act = (d.get("active_addendum") or "").strip()
print(f"\nACTIVE addendum: {act or '(empty baseline — nothing applied)'}")
cands = d.get("candidates") or []
if not cands:
    print("\nNo candidates proposed yet. (The evolver proposes after ~40 reflected outcomes per domain.)")
else:
    print(f"\n{len(cands)} candidate(s) awaiting review:")
    for c in cands:
        print(f"\n  • [{c.get('variant_id')}] domain={c.get('domain')} "
              f"samples={c.get('samples')} rate={c.get('success_rate')}")
        print(f"    proposed: {c.get('text','')}")
    print("\nTo apply one:  ./scripts/evolver_review.sh --promote <DOMAIN> <VARIANT_ID>")
PY
