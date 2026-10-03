#!/usr/bin/env bash
# Sparse-clone the Petrobras 3W dataset into data/raw/3W.
# Only fetches event classes 4, 6, 7, 8, 9 and the dataset.ini config.
set -euo pipefail

REPO_URL="https://github.com/petrobras/3W.git"
TARGET_DIR="data/raw/3W"

if [ -d "$TARGET_DIR/.git" ]; then
  echo "3W repo already cloned at $TARGET_DIR — pulling latest."
  git -C "$TARGET_DIR" pull
  exit 0
fi

mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"

git init
git remote add origin "$REPO_URL"
git config core.sparseCheckout true

cat > .git/info/sparse-checkout <<'EOF'
dataset/4/
dataset/6/
dataset/7/
dataset/8/
dataset/9/
dataset/dataset.ini
EOF

git pull --depth=1 origin main

echo "Done. 3W dataset (classes 4,6,7,8,9) available at $TARGET_DIR"
