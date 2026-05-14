#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash scripts/sync_github_wiki.sh /path/to/your/repo.wiki" >&2
  exit 1
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target_dir="$1"

python3 "$repo_root/scripts/export_github_wiki.py" --output-dir "$repo_root/wiki"
rsync -av --delete "$repo_root/wiki/" "$target_dir/"
