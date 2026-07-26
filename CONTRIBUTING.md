# Contributing to zoomer

Thanks for taking an interest. Bug reports, gesture ideas, and tuning defaults
that work better than ours are all welcome.

## Getting set up

```bash
git clone https://github.com/zoomer-project/zoomer
cd zoomer
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Check everything works before you change anything:

```bash
pytest
ruff check . && ruff format --check . && mypy
```

The default suite needs no camera, no display server, and no network. If it
fails on a clean checkout, that is a bug — please report it.

## The one architectural rule

**Gesture logic must not import a camera, a vision library, or an input
library.** Everything from `zoomer/gestures/`, `zoomer/types.py`, and
`zoomer/calibration.py` uses only the standard library.

That constraint is what makes the product testable. Hand tracking is
non-deterministic and hardware is not available in CI, so any behaviour that
lives on the wrong side of that line can only be verified by a human waving at a
webcam — which means, in practice, that it is not verified at all.

Hardware lives in `zoomer/tracking/` and `zoomer/backends/`, each behind a
protocol, and imports its heavy dependencies lazily inside functions so that
importing `zoomer` on a headless machine still works.

## Tests

Write tests that state a behaviour, not an implementation. Compare:

```python
def test_update_returns_list(self): ...  # tells a reader nothing
def test_closing_the_fingers_zooms_out(self): ...  # states the promise
```

Where a test encodes a decision that is not obvious, put the reasoning in a
comment. Most of the trickier behaviour here exists because of a specific
failure mode, and a future contributor deleting a test they cannot justify is a
real risk.

- `tests/unit/` — one module each, fast, exhaustive on edge cases.
- `tests/e2e/` — whole sessions using the **default production settings**, so
  the constants users actually run with stay under test.
- `tests/support.py` — scripted gestures and stand-ins for hardware. Specify
  motion in hand-widths, never raw normalised coordinates: absolute units
  silently describe a different physical gesture whenever the apparent hand size
  changes.
- Tests marked `@pytest.mark.hardware` are deselected by default. Run them
  yourself with `pytest -m hardware` when you touch camera, model, or input code.

New behaviour needs a test. Changed behaviour needs its test updated — and if
you find yourself loosening an assertion, say why in the commit message.

## Style

`ruff` and `mypy --strict` are enforced in CI; run them locally and there will be
no surprises.

Beyond what the tools check:

- Every public function, class, and module gets a docstring. Say what it does
  and, where it is not obvious, **why it does it that way**.
- Comments explain reasoning, not mechanics. `# increment the counter` is noise;
  `# Truncating toward zero banks the remainder, so a slow gesture covers the
  same distance as a fast one` is worth its line.
- Validate arguments where they enter, and raise errors that tell the user what
  to do next. Most people meeting an error here are trying to get a camera
  working, not debugging our code.

## Commits

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(gestures): arbitrate zoom and scroll with a hysteretic mode lock
fix(tracking): tolerate a truncated landmark result
docs: explain the macOS accessibility requirement
test(e2e): cover switching gestures mid-session
```

Types in use: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `style`,
`build`, `ci`, `chore`. Scopes match the package layout: `gestures`, `tracking`,
`backends`, `config`, `hud`, `app`, `cli`, `core`.

Write the body for someone reading `git log` in a year with no memory of the
discussion. The change itself is visible in the diff; the reasoning is not.

## Pull requests

1. Branch off `main`.
2. Make the change, with tests.
3. Ensure `pytest`, `ruff check .`, `ruff format --check .`, and `mypy` pass.
4. Add an entry to `CHANGELOG.md` under `Unreleased` for anything user-visible.
5. Open the PR and fill in the template.

Small, focused pull requests get reviewed quickly. A large one is best preceded
by an issue, so we can agree on the approach before you spend the time.

## Changing default tuning

Defaults affect everybody, so they need more than a preference:

- Say what hardware and setup you tested on, including camera and distance.
- Explain what was wrong with the old value, concretely.
- Confirm the end-to-end suite still passes — it runs against the defaults, so a
  change that breaks recognition will show up there.

## Code of conduct

Participation is governed by [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## Licence

Contributions are accepted under the [Apache License 2.0](LICENSE).
