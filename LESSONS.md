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

## Encode an approximation's validity range as a guard, not a sentence

- **Expected:** Documenting the sequential intervals as a "large-sample
  approximation" would be enough to keep readers out of the range where it fails.
- **Actual:** The guard only required two pairs per look. A simulation showed a
  nominal 95% interval covering 70% at two pairs, 88% at five, and 94% at twenty,
  so a module whose whole subject is spending a declared error budget could spend
  six times it while still printing the declared number.
- **Next time:** When a lesson's credibility rests on a stated error rate, measure
  the approximation's coverage and reject the inputs where it breaks. Prose next to
  a permissive guard is a disclaimer, not a control.

## Build the test table from the docstring, not from the implementation

- **Expected:** A test that asserts one interval per evidence state would prove the
  classifier and its documentation agreed.
- **Actual:** The docstring promised that any interval inside the equivalence band
  supports equivalence, while the code returned the directional state first. The
  test case `[+0.01, +0.02]` was written by reading the code, so it asserted the
  behavior the docstring denied and froze the contradiction in place.
- **Next time:** When a docstring enumerates cases, derive the test table from that
  enumeration before looking at the branches. A test written from the source can
  only ever confirm the source.

## Never narrate a conclusion the script already computed

- **Expected:** A seeded, deterministic lesson could safely print its verdict as
  fixed prose, and the CI check comparing two runs byte for byte would protect it.
- **Actual:** The example printed "decision: HOLD" and a literal "+3.00 pp" beside a
  separately computed evidence state. Both runs agree with each other no matter what
  the statistics say, so that check could never detect the narration drifting from
  the result after a change to the seed, the simulated lift, or the threshold.
- **Next time:** Derive every user-visible verdict from the computed value, and treat
  determinism checks as a reproducibility guarantee only, never as agreement between
  a result and the sentence describing it.

## Re-parse Python after a bulk punctuation rewrite

- **Expected:** Substituting straight quotes for typographic ones across the changed
  files was a cosmetic edit with no way to break code.
- **Actual:** One class docstring ended on a quoted phrase, so the substitution
  produced four consecutive quote characters and an unterminated string literal. The
  `grep` that confirmed the punctuation was gone reported success on a file that no
  longer parsed; only an explicit `ast.parse` surfaced it.
- **Next time:** Follow any repository-wide punctuation or quoting rewrite with a
  parse of every touched source file. Confirming the old characters are absent says
  nothing about whether the result is still valid Python.
