# Hygiene hooks

- `commit-msg`: strips AI co-author trailers and "generated with" lines from every
  commit message (lab rule: never credit AI in commits).
- `pre-push`: scans only the outgoing commits for (1) AI attribution trailers,
  (2) secrets in added lines (API keys, tokens, private keys), (3) an optional
  forbidden-content regex (`git config hygiene.forbiddenPattern '...'`) for
  private names/paths in public repos.

Install (the scaffold already did this):

    git config core.hooksPath hooks
    chmod +x hooks/commit-msg hooks/pre-push

Note: `core.hooksPath` replaces ALL hooks in `.git/hooks`. Never bypass with
`--no-verify`; a block means fix the commit, not the hook.