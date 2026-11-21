#!/bin/sh
# Install the hygiene hooks (idempotent).
set -eu
cd "$(dirname "$0")/.."
git config core.hooksPath hooks
chmod +x hooks/commit-msg hooks/pre-push
echo "hooks installed: core.hooksPath=hooks"