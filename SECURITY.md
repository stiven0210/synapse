# Security Policy

## Scope

SYNAPSE is a research/reference implementation, not a hosted service —
there's no production deployment, API, or user data to compromise. Security
reports here are most relevant to:

- The decision logic itself (Executor, Veto Layer, Policy Artifact
  handling) — anything that could make a wrong decision look valid, or
  make the Veto Layer fail to override when it should.
- The Triage Agent's handling of `ANTHROPIC_API_KEY` and its LLM-facing
  code (`src/agente_triage.py`) — prompt injection, credential handling,
  or ways to make it exceed its rate limit or bypass its circuit breaker.
- Supply-chain issues in `requirements.txt`.

## Reporting a Vulnerability

Please **do not** open a public issue for a security concern. Instead, use
GitHub's [private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing/privately-reporting-a-security-vulnerability)
on this repository (Security tab → "Report a vulnerability"), or contact
the maintainer directly through their GitHub profile.

Include:

- What the issue is and why it matters (e.g., "this lets the Veto Layer be
  bypassed under condition X").
- Steps to reproduce, ideally as a minimal script or failing test.
- The impact you'd expect in a real deployment, if it's not obvious from
  the reproduction.

## Response

This is a single-maintainer project maintained outside of full-time work.
There's no guaranteed SLA, but security reports get priority over feature
work — expect an initial response within a few days.

## Disclosure

Once a fix is available, it will be released and the report credited
(unless you ask not to be) in the fix's commit message or release notes.
