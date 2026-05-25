"""Vault memory plugin — Obsidian-on-Railway gateway recall + write-back.

Wires the self-improvement loop into Hermes so every call benefits:
  - Recall: each turn-start, retrieve relevant prior Q+A pairs from the vault.
  - Write-back: each turn-end, persist (user, assistant) for future recall.
  - Routing read-side: agent/vault_router.py reads VaultProvider.last_top_confidence()
    to downgrade to a cheaper model when the vault already covers the topic.
  - Feedback: gateway /v1/feedback endpoint patches the most-recent entry with
    success_flag (Phase 3 — wired from OpenJarvis transcripts).

Activated via `memory.provider: vault` in config.yaml. Requires `VAULT_SECRET` env var.
"""

from .provider import VaultProvider


def register(ctx) -> None:
    """Register Vault as the active memory provider."""
    ctx.register_memory_provider(VaultProvider())
