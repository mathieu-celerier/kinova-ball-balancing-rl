#!/usr/bin/env python3
"""Export the project documentation to GitHub Wiki-compatible Markdown files."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

PAGE_SOURCES: list[tuple[Path, str]] = [
    (REPO_ROOT / "docs" / "index.md", "Home.md"),
    (REPO_ROOT / "docs" / "getting-started.md", "Getting-Started.md"),
    (REPO_ROOT / "docs" / "environment" / "overview.md", "Task-Overview.md"),
    (REPO_ROOT / "docs" / "environment" / "mdp.md", "MDP-Design.md"),
    (REPO_ROOT / "docs" / "environment" / "physics.md", "Physics-and-Control.md"),
    (REPO_ROOT / "docs" / "environment" / "related-work.md", "Related-Work.md"),
    (REPO_ROOT / "docs" / "environment" / "robot-model.md", "Robot-and-MuJoCo-Model.md"),
    (REPO_ROOT / "docs" / "code" / "structure.md", "Project-Structure.md"),
    (REPO_ROOT / "docs" / "code" / "modules.md", "Key-Modules.md"),
]

WIKI_LINKS = {
    "index.md": "Home.md",
    "getting-started.md": "Getting-Started.md",
    "environment/overview.md": "Task-Overview.md",
    "environment/mdp.md": "MDP-Design.md",
    "environment/physics.md": "Physics-and-Control.md",
    "environment/related-work.md": "Related-Work.md",
    "environment/robot-model.md": "Robot-and-MuJoCo-Model.md",
    "code/structure.md": "Project-Structure.md",
    "code/modules.md": "Key-Modules.md",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export docs/ as a GitHub Wiki-compatible Markdown tree."
    )
    parser.add_argument(
        "--output-dir",
        default="wiki",
        help="Destination directory for the wiki Markdown files.",
    )
    parser.add_argument(
        "--repo-url",
        default=None,
        help="GitHub repository URL used to rewrite links to source files.",
    )
    return parser.parse_args()


def _git_repo_url() -> str:
    raw = subprocess.check_output(
        ["git", "remote", "get-url", "origin"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()
    if raw.startswith("git@github.com:"):
        raw = "https://github.com/" + raw.removeprefix("git@github.com:").removesuffix(".git")
    elif raw.startswith("https://github.com/"):
        raw = raw.removesuffix(".git")
    return raw


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _strip_mkdocs_footer(text: str) -> str:
    return text.replace("\n---\n# MjLab Kinova Ball Balancing", "\n\n# MjLab Kinova Ball Balancing", 1)


def _rewrite_links(text: str, repo_url: str) -> str:
    def replace_markdown_link(match: re.Match[str]) -> str:
        label = match.group("label")
        target = match.group("target")

        normalized = target.removeprefix("./").removeprefix("../")
        if normalized in WIKI_LINKS:
            return f"[{label}]({WIKI_LINKS[normalized]})"

        # Rewrite direct links to repository files to absolute GitHub URLs.
        if normalized.startswith(("docs/", "src/", "config/", "scripts/")):
            return f"[{label}]({repo_url}/blob/main/{normalized})"

        return match.group(0)

    return re.sub(r"\[(?P<label>[^\]]+)\]\((?P<target>[^)]+)\)", replace_markdown_link, text)


def _write_page(output_dir: Path, source: Path, output_name: str, repo_url: str) -> None:
    text = _read_text(source)
    text = _strip_mkdocs_footer(text)
    text = _rewrite_links(text, repo_url)
    (output_dir / output_name).write_text(text.rstrip() + "\n", encoding="utf-8")


def _write_sidebar(output_dir: Path) -> None:
    sidebar = """* [Home](Home.md)
* [Getting Started](Getting-Started.md)
* Environment
  * [Task Overview](Task-Overview.md)
  * [MDP Design](MDP-Design.md)
  * [Physics and Control](Physics-and-Control.md)
  * [Related Work](Related-Work.md)
  * [Robot and MuJoCo Model](Robot-and-MuJoCo-Model.md)
* Code Reference
  * [Project Structure](Project-Structure.md)
  * [Key Modules](Key-Modules.md)
"""
    (output_dir / "_Sidebar.md").write_text(sidebar, encoding="utf-8")


def main() -> int:
    args = _parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    repo_url = args.repo_url or _git_repo_url()
    for source, output_name in PAGE_SOURCES:
        _write_page(output_dir, source, output_name, repo_url)
    _write_sidebar(output_dir)
    print(f"Exported wiki pages to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
