# Architecture Decision Records

An ADR captures a significant architectural decision, the context that
forced it, and its consequences — written down while the reasoning is
fresh, so it doesn't have to be reconstructed later from code and guesses.
We use the plain Nygard format: Context, Decision, Consequences (or, where
it fits better, a decision with its trade-offs made explicit inline).

These are kept because they're part of what makes the architecture
reviewable — not just what SYNAPSE decided, but why, and what it
deliberately chose not to do yet.

## Index

- [0001 — Shape of the Policy Artifact](0001-policy-artifact.md): why the
  contract between the slow and fast layers is a versioned JSON document
  with a fixed feature order, not a pickled model.
- [0002 — Veto Layer (Hard Invariants)](0002-veto-layer.md): what the veto
  invariants cover for the current dataset, and the per-account data
  limitation they don't cover yet.
- [0003 — Triage Agent for Operational Veto Escalations](0003-triage-agent.md):
  why an LLM was added for exactly one narrow role, kept out of the hot
  path, with a 3-layer discipline (schema, grounding, selective audit)
  that never resolves disagreement by majority vote.
