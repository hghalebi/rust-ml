#!/usr/bin/env python3
"""Check public learner-facing content for private or secret-shaped leaks."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PUBLIC_MARKDOWN = (
    ROOT / "README.md",
    ROOT / "LEARNING-PATH.md",
    ROOT / "PUBLIC_CONTENT.md",
    ROOT / "CS336-RUST-EQUIVALENT.md",
    ROOT / "book" / "README.md",
)

PUBLIC_DIRS = (
    ROOT / "assignments",
    ROOT / "lessons",
    ROOT / "code",
    ROOT / "references",
)

PUBLIC_SUFFIXES = {".md", ".rs", ".toml"}
SKIPPED_DIR_NAMES = {
    ".git",
    ".idea",
    "target",
}

SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("OpenAI-style API key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b")),
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b")),
    ("Slack token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b")),
    ("AWS access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("private key block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("database URL with credentials", re.compile(r"\bpostgres(?:ql)?://[^\s:@]+:[^\s:@]+@[^\s]+")),
    ("database URL assignment", re.compile(r"\bDATABASE_URL\s*=")),
)

PRIVATE_CONTEXT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("local user path", re.compile(r"/Users/[A-Za-z0-9._-]+")),
    ("Codex memory path", re.compile(r"\.codex/memories|\.khowlege")),
    ("session transcript path", re.compile(r"\.codex/sessions")),
)

PUBLIC_STYLE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("loaded metaphor", re.compile(r"\bhostage\b", re.IGNORECASE)),
    ("casual panic metaphor", re.compile(r"\bpanic soup\b", re.IGNORECASE)),
    ("casual tensor metaphor", re.compile(r"\banonymous tensor soup\b", re.IGNORECASE)),
    ("casual variable metaphor", re.compile(r"\bvariable soup\b", re.IGNORECASE)),
    ("casual technical metaphor", re.compile(r"\bwizardry\b", re.IGNORECASE)),
)


def iter_public_files() -> list[Path]:
    files: set[Path] = set()

    for path in PUBLIC_MARKDOWN:
        if path.exists():
            files.add(path)

    for public_dir in PUBLIC_DIRS:
        if not public_dir.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(public_dir):
            dirnames[:] = [
                dirname for dirname in dirnames if dirname not in SKIPPED_DIR_NAMES
            ]
            directory = Path(dirpath)
            for filename in filenames:
                path = directory / filename
                if path.is_file() and path.suffix in PUBLIC_SUFFIXES:
                    files.add(path)

    return sorted(files)


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def scan_file(path: Path) -> list[str]:
    errors: list[str] = []
    text = path.read_text(encoding="utf-8")

    for label, pattern in (
        SECRET_PATTERNS + PRIVATE_CONTEXT_PATTERNS + PUBLIC_STYLE_PATTERNS
    ):
        match = pattern.search(text)
        if match:
            line = text[: match.start()].count("\n") + 1
            errors.append(f"{relative(path)}:{line}: found {label}")

    for line_number, line in enumerate(text.splitlines(), start=1):
        if line.rstrip() != line:
            errors.append(f"{relative(path)}:{line_number}: trailing whitespace")

    return errors


def main() -> int:
    public_files = iter_public_files()
    errors: list[str] = []

    for path in public_files:
        errors.extend(scan_file(path))

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    print(f"Public content checks passed across {len(public_files)} files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
