"""Offline evaluation harness: WER / DER against hand-corrected references.

Workflow (see eval/README.md):
  1. scripts/eval_bootstrap.py <session-id>   → eval/cases/<name>/ skeleton
  2. hand-correct reference.txt + reference_turns.json, set "verified": true
  3. scripts/eval_run.py                      → metrics table + history row

Metrics are implemented here in pure Python (no jiwer / pyannote.metrics
dependency) so the eval runs anywhere the app runs — including the Pi and
Python 3.14 where heavy metric packages may lack wheels. Implementations
are unit-tested against hand-computed examples in tests/test_evals.py.
"""
