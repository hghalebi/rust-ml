#!/usr/bin/env python3
"""Deterministic quality checks for the rust-ml course content."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUTHORED_MODULES = (
    ROOT / "lessons" / "01-foundations",
    ROOT / "lessons" / "02-vectors",
    ROOT / "lessons" / "07-transformer",
)
REQUIRED_LESSON_HEADINGS = (
    "Overview",
    "Learning Goals",
    "Plain-English Explanation",
    "Algebra Form",
    "Rust Form",
    "Why This Matters",
    "Short Practice",
)
MARKDOWN_LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
HTML_SRC_RE = re.compile(r"""<img[^>]+src=["']([^"']+)["']""")


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def resolve_local_target(origin: Path, raw_target: str) -> Path | None:
    if raw_target.startswith(("http://", "https://", "mailto:", "tel:")):
        return None

    target = raw_target.split("#", 1)[0].split("?", 1)[0].strip()
    if not target:
        return None

    if target.startswith("/"):
        return ROOT / target.lstrip("/")

    return (origin.parent / target).resolve()


def check_markdown_links() -> list[str]:
    errors: list[str] = []

    for markdown_file in sorted(ROOT.rglob("*.md")):
        text = markdown_file.read_text(encoding="utf-8")

        for raw_target in MARKDOWN_LINK_RE.findall(text):
            resolved = resolve_local_target(markdown_file, raw_target)
            if resolved is None:
                continue
            if not resolved.exists():
                errors.append(
                    f"{relative(markdown_file)} -> missing local link target: {raw_target}"
                )

        for raw_target in HTML_SRC_RE.findall(text):
            resolved = resolve_local_target(markdown_file, raw_target)
            if resolved is None:
                continue
            if not resolved.exists():
                errors.append(
                    f"{relative(markdown_file)} -> missing local asset target: {raw_target}"
                )

    return errors


def check_authored_module_contract() -> list[str]:
    errors: list[str] = []

    for module_dir in AUTHORED_MODULES:
        required_files = (
            module_dir / "README.md",
            module_dir / "exercises.md",
            module_dir / "solutions.md",
        )
        for required in required_files:
            if not required.exists():
                errors.append(f"{relative(required)} is required for authored modules")

        lesson_files = sorted(module_dir.glob("[0-9][0-9]-*.md"))
        if not lesson_files:
            errors.append(f"{relative(module_dir)} has no ordered lesson files")

    return errors


def check_lesson_sections() -> list[str]:
    errors: list[str] = []

    for module_dir in AUTHORED_MODULES:
        for lesson_file in sorted(module_dir.glob("[0-9][0-9]-*.md")):
            text = lesson_file.read_text(encoding="utf-8")
            if module_dir.name == "07-transformer":
                required_markers = (
                    "## Overview",
                    "## Learning Goals",
                    "### English",
                    "### Algebra",
                    "### Rust",
                )
            else:
                required_markers = tuple(f"## {heading}" for heading in REQUIRED_LESSON_HEADINGS)

            for marker in required_markers:
                if marker not in text:
                    errors.append(f"{relative(lesson_file)} is missing section: {marker}")

            if "```rust" not in text:
                errors.append(f"{relative(lesson_file)} should include at least one Rust block")

            if "TODO" in text:
                errors.append(f"{relative(lesson_file)} contains TODO and should be finished")

    return errors


def check_root_readme_contract() -> list[str]:
    errors: list[str] = []
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")

    required_phrases = (
        "# Rust ML Systems from First Principles",
        "## Start Here",
        "## What Exists Now",
        "## Repo Map",
        "## Running The Code",
    )
    for phrase in required_phrases:
        if phrase not in text:
            errors.append(f"README.md is missing section marker: {phrase}")

    return errors


def main() -> int:
    checks = (
        check_markdown_links,
        check_authored_module_contract,
        check_lesson_sections,
        check_root_readme_contract,
    )

    errors: list[str] = []
    for check in checks:
        errors.extend(check())

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    markdown_count = len(list(ROOT.rglob("*.md")))
    print(f"Course content checks passed across {markdown_count} Markdown files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
