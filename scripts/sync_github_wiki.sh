#!/usr/bin/env bash
set -euo pipefail

delete_mode=1
target_dir=""

usage() {
  cat >&2 <<'EOF'
Usage:
  bash scripts/sync_github_wiki.sh [--no-delete] /path/to/your/repo.wiki

Options:
  --no-delete   Keep existing files in the wiki checkout instead of pruning them.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-delete)
      delete_mode=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      if [[ -n "$target_dir" ]]; then
        usage
        exit 1
      fi
      target_dir="$1"
      shift
      ;;
  esac
done

if [[ -z "$target_dir" ]]; then
  usage
  exit 1
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$target_dir"

python3 "$repo_root/scripts/export_github_wiki.py" --output-dir "$repo_root/wiki"
if [[ "$delete_mode" -eq 1 ]]; then
  rsync -av --delete --exclude=.git/ "$repo_root/wiki/" "$target_dir/"
else
  rsync -av --exclude=.git/ "$repo_root/wiki/" "$target_dir/"
fi
