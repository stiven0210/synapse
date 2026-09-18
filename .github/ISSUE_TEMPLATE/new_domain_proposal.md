---
name: New domain proposal
about: Propose validating SYNAPSE against a new domain
title: "[Domain] "
labels: new-domain
assignees: ''
---

See `CONTRIBUTING.md`'s "Adding a new domain" section for the full bar a
new domain needs to clear before it's added to the README's validated
domains table. This template is the starting discussion, not the PR.

**Domain**
What real-world decision problem is this (e.g., "insurance claim triage",
"log-line anomaly detection")?

**Dataset**
Link to a public, downloadable dataset. If none exists, say so and
describe how you'd generate a realistic synthetic one (see
`docs/DOMINIO2_PERSONALIZACION_POR_CUENTA.md` for how the per-account
fraud domain handled this).

**Why this is a fair test of the two-speed pattern**
What about this domain actually exercises the Calibrator/Executor split —
i.e., why does it need microsecond decisions with an offline-learned
policy, rather than just calling a model directly?

**Expected invariants for the Veto Layer**
What hard, model-independent rules would this domain need (if any)?

**Known limitations up front**
Any data limitation you already know about (e.g., no entity identifier,
no real fraud/anomaly rate available) — see ADR 0002 for how the fraud
domain documented its own limitation instead of hiding it.
