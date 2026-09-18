---
name: Bug report
about: Something in SYNAPSE isn't behaving the way it should
title: "[Bug] "
labels: bug
assignees: ''
---

**What happened**
A clear description of the actual behavior.

**What you expected**
What you expected to happen instead.

**Steps to reproduce**
Minimal steps or, ideally, a failing test / small script.

```
# paste a minimal reproduction here
```

**Domain / component**
Which domain (fraud / per-account fraud / industrial IoT) and which
component (Calibrator, Executor, Veto Layer, Drift Detector, Triage
Agent, etc.) this touches.

**Environment**
- Python version:
- OS:
- Relevant package versions (`pip freeze | grep -E "scikit-learn|pandas|numpy"`):

**Does this affect a decision-path invariant?**
If this could cause the Executor or Veto Layer to produce a wrong or
silent decision, say so explicitly — these get priority.
