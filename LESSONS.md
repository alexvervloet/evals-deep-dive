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

## Build CI inventories from tracked paths

- **Expected:** The conceptual lesson names in the work plan would map directly
  to example filenames when the CI command was written.
- **Actual:** Three filenames used different, more specific names; a directory
  listing caught the mismatch before commit.
- **Next time:** Resolve every promised runnable path with `git ls-files` before
  encoding an explicit CI inventory, then keep the explicit list so live examples
  are not accidentally executed.

## Scan narrated runtime paths for optimized-away checks

- **Expected:** Adding an optimized test run would be enough to establish that
  behavior does not depend on Python assertions.
- **Actual:** The provider-backed capstone contained a pre-existing runtime
  `assert` on a logically guaranteed report. The new unit suite could not execute
  that paid path and therefore could not expose it dynamically.
- **Next time:** Pair `python -O` tests with a source scan of library, example, and
  capstone paths; reserve `assert` for test assertions, never runtime validation.

## Match verification to the file type

- **Expected:** A small commit loop could apply the same syntax command to each
  changed file.
- **Actual:** It sent `LESSONS.md` to `py_compile`; the commit was unaffected, but
  the check was meaningless and failed on Markdown prose.
- **Next time:** Select checks per artifact (`py_compile` for Python, a Markdown
  link/style check for prose) instead of interpolating heterogeneous paths into one
  command template.
