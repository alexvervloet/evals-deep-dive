# Lessons learned

## Use the repository virtual environment for local checks

- **Expected:** `python` would resolve to the local Python interpreter during a
  quick syntax check.
- **Actual:** This machine exposes Python through `.venv/bin/python` (and
  `python3`), so the bare command was unavailable.
- **Next time:** Run this repository's validation commands through
  `.venv/bin/python` from the start. CI should continue using the interpreter
  installed by `actions/setup-python`, where `python` is part of the documented
  runner environment.
