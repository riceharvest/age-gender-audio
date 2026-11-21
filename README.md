# age-gender-audio

One line: what it does, for whom, and the existing slot it drops into.

## Install

    pip install age-gender-audio

## Usage

    age-gender-audio world

## Claims rules (honest numbers — keep this section)

- Every number in this README is produced by a command that is quoted next to it (or linked to its result file). No invented, rounded-up or extrapolated numbers.
- Dev-set numbers are labelled **dev-set** everywhere they appear. Held-out numbers say which frozen split, n, and seed produced them.
- "Beats X" only with a paired confidence interval whose lower bound is > 0, under X's own documented setup; otherwise the word is "level with". Losses are shown, not buried.
- Benchmarks link the exact reproduce command and the reader/scorer version.
- No AI credit in commits, code or docs (the commit-msg hook strips AI co-author trailers; the pre-push hook blocks them and scans for secrets).

## Development

    pip install -e . pytest
    python -m pytest -q

CI runs the same tests on every push/PR (`.github/workflows/ci.yml`).
Hooks: `git config core.hooksPath hooks` (done by the scaffold; see `hooks/README.md`).

## License

MIT — see LICENSE.