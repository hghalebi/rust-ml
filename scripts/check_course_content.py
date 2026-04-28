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
    ROOT / "lessons" / "03-neuron",
    ROOT / "lessons" / "04-learning",
    ROOT / "lessons" / "07-transformer",
)
PLANNED_MODULES = (
    ROOT / "lessons" / "05-mlp",
    ROOT / "lessons" / "06-attention",
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


def expected_course_module(module_dir: Path) -> int:
    return int(module_dir.name.split("-", 1)[0]) - 1


def check_module_readmes() -> list[str]:
    errors: list[str] = []

    authored_markers = (
        "Status: active.",
        "## Outcomes",
        "## Lessons",
        "## Practice",
        "## Code Artifact",
        "## Prerequisite",
        "## Before You Move On",
    )
    planned_markers = (
        "Status: planned.",
        "## Goal",
        "## Planned Lesson Ladder",
        "## Planned Practice",
        "## Code Artifact",
        "## Prerequisite",
        "## Planned Outcome",
    )

    for module_dir in AUTHORED_MODULES + PLANNED_MODULES:
        readme = module_dir / "README.md"
        if not readme.exists():
            errors.append(f"{relative(readme)} is required")
            continue

        text = readme.read_text(encoding="utf-8")
        expected_mapping = f"This folder maps to course Module {expected_course_module(module_dir)}."
        if expected_mapping not in text:
            errors.append(f"{relative(readme)} should contain mapping line: {expected_mapping}")

        markers = authored_markers if module_dir in AUTHORED_MODULES else planned_markers
        for marker in markers:
            if marker not in text:
                errors.append(f"{relative(readme)} is missing section: {marker}")

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

            if text.startswith("# Lesson "):
                errors.append(
                    f"{relative(lesson_file)} should use a concept-first title, not a global lesson number"
                )

    return errors


def check_structure_guide() -> list[str]:
    guide = ROOT / "lessons" / "COURSE-STRUCTURE.md"
    if not guide.exists():
        return [f"{relative(guide)} is required"]

    text = guide.read_text(encoding="utf-8")
    required_phrases = (
        "# Course Structure",
        "## Translation Contract",
        "## Current Learning Paths",
        "## Naming Rules",
        "## Module Contract",
        "## Lesson Contract",
        "## Review Checklist",
    )
    return [
        f"{relative(guide)} is missing section marker: {phrase}"
        for phrase in required_phrases
        if phrase not in text
    ]


def check_lessons_index_contract() -> list[str]:
    errors: list[str] = []
    readme = ROOT / "lessons" / "README.md"
    text = readme.read_text(encoding="utf-8")

    required_phrases = (
        "# Lessons",
        "## Course Map",
        "## Current Recommended Paths",
        "Module 6 | Authored preview",
    )
    for phrase in required_phrases:
        if phrase not in text:
            errors.append(f"lessons/README.md is missing section marker: {phrase}")

    if "Module 7" in text:
        errors.append("lessons/README.md should not reference a duplicate Module 7 entry")

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
        "lessons/COURSE-STRUCTURE.md",
    )
    for phrase in required_phrases:
        if phrase not in text:
            errors.append(f"README.md is missing section marker: {phrase}")

    return errors


def main() -> int:
    checks = (
        check_markdown_links,
        check_authored_module_contract,
        check_module_readmes,
        check_lesson_sections,
        check_structure_guide,
        check_lessons_index_contract,
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
