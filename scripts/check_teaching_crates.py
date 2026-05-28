#!/usr/bin/env python3
"""Check active teaching-crate structure against the repo pedagogy contract."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = ROOT / "code"
WORKSPACE_MANIFEST = CODE_ROOT / "Cargo.toml"

README_MARKERS = (
    "Status: active.",
    "## Owns",
    "## Current State",
    "## Layout",
    "## Learning Ladder",
    "## Category Lens",
    "## Run",
    "## Scope",
)


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def toml_string_value(line: str) -> str | None:
    if "=" not in line:
        return None

    _, raw_value = line.split("=", 1)
    value = raw_value.strip()
    if len(value) >= 2 and value[0] == value[-1] == '"':
        return value[1:-1]

    return None


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

        if in_package and stripped.startswith("name"):
            return toml_string_value(stripped)

    return None


def check_crate(member: str) -> list[str]:
    errors: list[str] = []
    crate_dir = CODE_ROOT / member
    cargo_manifest = crate_dir / "Cargo.toml"
    readme = crate_dir / "README.md"
    lib = crate_dir / "src" / "lib.rs"
    error_module = crate_dir / "src" / "error.rs"
    examples_dir = crate_dir / "examples"

    for required in (cargo_manifest, readme, lib, error_module, examples_dir):
        if not required.exists():
            errors.append(f"{relative(required)} is required for active teaching crates")

    if not cargo_manifest.exists():
        return errors

    package_name = read_package_name(cargo_manifest)
    if package_name is None:
        errors.append(f"{relative(cargo_manifest)} must define package.name")
    elif not package_name.startswith("rust_ml_"):
        errors.append(f"{relative(cargo_manifest)} package.name should start with rust_ml_")

    manifest_text = cargo_manifest.read_text(encoding="utf-8")
    if 'thiserror = "2"' not in manifest_text:
        errors.append(f"{relative(cargo_manifest)} must depend on thiserror = \"2\"")

    if readme.exists():
        readme_text = readme.read_text(encoding="utf-8")
        for marker in README_MARKERS:
            if marker not in readme_text:
                errors.append(f"{relative(readme)} is missing section marker: {marker}")

        if package_name and package_name not in readme_text:
            errors.append(f"{relative(readme)} should mention package {package_name}")

        if examples_dir.exists():
            for example in sorted(examples_dir.glob("*.rs")):
                if example.stem not in readme_text:
                    errors.append(
                        f"{relative(readme)} should mention example {example.stem}"
                    )

    if lib.exists():
        lib_text = lib.read_text(encoding="utf-8")
        if not lib_text.startswith("//!"):
            errors.append(f"{relative(lib)} should start with a module-level doc comment")
        if "pub mod error;" not in lib_text:
            errors.append(f"{relative(lib)} must expose the typed error module")

    if error_module.exists():
        error_text = error_module.read_text(encoding="utf-8")
        if "use thiserror::Error;" not in error_text:
            errors.append(f"{relative(error_module)} must use thiserror::Error")
        if "#[derive(Debug, Error" not in error_text:
            errors.append(f"{relative(error_module)} must derive thiserror::Error")
        if "pub enum" not in error_text:
            errors.append(f"{relative(error_module)} must expose a public error enum")

    if examples_dir.exists():
        examples = sorted(examples_dir.glob("*.rs"))
        if not examples:
            errors.append(f"{relative(examples_dir)} must contain runnable examples")

    src_files = sorted((crate_dir / "src").glob("*.rs")) if (crate_dir / "src").exists() else []
    if src_files and not any("#[cfg(test)]" in path.read_text(encoding="utf-8") for path in src_files):
        errors.append(f"{relative(crate_dir / 'src')} must include unit tests")

    return errors


def check_cross_references(members: list[str]) -> list[str]:
    errors: list[str] = []
    code_readme = ROOT / "code" / "README.md"
    teaching_contract = ROOT / "scripts" / "check_rust_teaching_contract.py"

    code_readme_text = code_readme.read_text(encoding="utf-8")
    teaching_contract_text = teaching_contract.read_text(encoding="utf-8")

    for member in members:
        cargo_manifest = CODE_ROOT / member / "Cargo.toml"
        package_name = read_package_name(cargo_manifest) if cargo_manifest.exists() else None

        if f"[{member}](" not in code_readme_text:
            errors.append(f"{relative(code_readme)} should list teaching crate {member}")
        if package_name and package_name not in code_readme_text:
            errors.append(f"{relative(code_readme)} should mention package {package_name}")
        if f'CODE_ROOT / "{member}" / "src"' not in teaching_contract_text:
            errors.append(
                f"{relative(teaching_contract)} strict public API paths must include code/{member}/src"
            )

    return errors


def main() -> int:
    members = load_workspace_members()
    errors: list[str] = []

    if not members:
        errors.append(f"{relative(WORKSPACE_MANIFEST)} must define workspace members")

    for member in members:
        errors.extend(check_crate(member))

    errors.extend(check_cross_references(members))

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    print(f"Teaching crate checks passed across {len(members)} active crates.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
