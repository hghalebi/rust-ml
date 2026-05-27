#!/usr/bin/env python3
"""Run every active teaching-crate example once."""

from __future__ import annotations

import subprocess
import sys
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = ROOT / "code"
WORKSPACE_MANIFEST = CODE_ROOT / "Cargo.toml"
RAW_HELPER_SIGNATURE_RE = re.compile(
    r"("
    r"\bVec\s*<"
    r"|&\s*\["
    r"|&\s*str\b"
    r"|&\s*'static\s+str\b"
    r"|\bf32\b"
    r"|\bf64\b"
    r"|\busize\b"
    r"|\bu64\b"
    r"|\bu128\b"
    r"|\bi64\b"
    r"|\bbool\b"
    r")"
)
FUNCTION_START_RE = re.compile(r"^\s*fn\s+(\w+)\s*[<(]")


def collect_function_signatures(lines: list[str]) -> list[tuple[int, str, str]]:
    signatures: list[tuple[int, str, str]] = []
    collecting = False
    start_line = 0
    function_name = ""
    current: list[str] = []
    paren_depth = 0

    for line_number, line in enumerate(lines, start=1):
        if not collecting:
            function = FUNCTION_START_RE.search(line)
            if function is None:
                continue

            collecting = True
            start_line = line_number
            function_name = function.group(1)
            current = []
            paren_depth = 0

        current.append(line.strip())
        paren_depth += line.count("(") - line.count(")")

        if paren_depth <= 0 and ("{" in line or ";" in line):
            signatures.append((start_line, function_name, " ".join(current)))
            collecting = False

    return signatures


def load_workspace_members() -> list[str]:
    members: list[str] = []
    in_members = False

    for line in WORKSPACE_MANIFEST.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()

        if stripped.startswith("members") and "=" in stripped and "[" in stripped:
            in_members = True
            continue

        if in_members and stripped.startswith("]"):
            break

        if in_members:
            value = stripped.rstrip(",")
            if len(value) >= 2 and value[0] == value[-1] == '"':
                members.append(value[1:-1])

    return sorted(members)


def read_package_name(cargo_manifest: Path) -> str | None:
    in_package = False

    for line in cargo_manifest.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped == "[package]":
            in_package = True
            continue

        if in_package and stripped.startswith("["):
            return None

        if in_package and stripped.startswith("name") and "=" in stripped:
            _, raw_value = stripped.split("=", 1)
            value = raw_value.strip()
            if len(value) >= 2 and value[0] == value[-1] == '"':
                return value[1:-1]

    return None


def check_example_helper_signatures(example: Path) -> list[str]:
    errors: list[str] = []
    lines = example.read_text(encoding="utf-8").splitlines()

    for line_number, function_name, signature in collect_function_signatures(lines):
        if function_name == "main":
            continue

        if RAW_HELPER_SIGNATURE_RE.search(signature):
            relative = example.relative_to(ROOT)
            errors.append(
                f"{relative}:{line_number} has a raw primitive/container helper signature; "
                "public examples should validate raw literals inline with TryFrom and keep helpers typed"
            )

    return errors


def main() -> int:
    executed = 0
    signature_errors: list[str] = []

    for member in load_workspace_members():
        crate_dir = CODE_ROOT / member
        package_name = read_package_name(crate_dir / "Cargo.toml")
        if package_name is None:
            print(f"ERROR: code/{member}/Cargo.toml must define package.name")
            return 1

        for example in sorted((crate_dir / "examples").glob("*.rs")):
            signature_errors.extend(check_example_helper_signatures(example))
            result = subprocess.run(
                [
                    "cargo",
                    "run",
                    "--quiet",
                    "--manifest-path",
                    "code/Cargo.toml",
                    "-p",
                    package_name,
                    "--example",
                    example.stem,
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            if result.returncode != 0:
                print(f"ERROR: {package_name} example {example.stem} failed")
                if result.stdout:
                    print(result.stdout)
                if result.stderr:
                    print(result.stderr)
                return result.returncode

            executed += 1

    if signature_errors:
        for error in signature_errors:
            print(f"ERROR: {error}")
        return 1

    if executed == 0:
        print("ERROR: no teaching examples were executed")
        return 1

    print(f"Executed {executed} runnable teaching examples.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
