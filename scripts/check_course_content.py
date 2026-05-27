#!/usr/bin/env python3
"""Deterministic quality checks for the rust-ml course content."""

from __future__ import annotations

import re
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUTHORED_MODULES = (
    ROOT / "lessons" / "01-foundations",
    ROOT / "lessons" / "02-vectors",
    ROOT / "lessons" / "03-neuron",
    ROOT / "lessons" / "04-learning",
    ROOT / "lessons" / "05-mlp",
    ROOT / "lessons" / "06-attention",
    ROOT / "lessons" / "07-transformer",
    ROOT / "lessons" / "08-language-modeling",
    ROOT / "lessons" / "09-systems",
    ROOT / "lessons" / "10-kernels",
    ROOT / "lessons" / "11-inference",
)
PLANNED_MODULES = ()
CS336_ASSIGNMENTS_DIR = ROOT / "assignments" / "cs336-rust"
CS336_ASSIGNMENT_DOCS = (
    CS336_ASSIGNMENTS_DIR / "01-basics.md",
    CS336_ASSIGNMENTS_DIR / "02-systems.md",
    CS336_ASSIGNMENTS_DIR / "03-scaling.md",
    CS336_ASSIGNMENTS_DIR / "04-data.md",
    CS336_ASSIGNMENTS_DIR / "05-alignment.md",
)
REQUIRED_LESSON_HEADINGS = (
    "Overview",
    "Learning Goals",
    "Plain-English Explanation",
    "Algebra Form",
    "Rust Form",
    "Why This Matters",
    "Concept Trace",
    "Short Practice",
)
REQUIRED_CONCEPT_TRACE_MARKERS = (
    "Object/newtype",
    "Invariant",
    "Map",
    "Runnable proof",
    "Failure signal",
)
REQUIRED_ASSIGNMENT_HEADINGS = (
    "## Goal",
    "## What You Build",
    "## Active Starter Crate",
    "## Object/Map Preflight",
    "## Expected Deliverables",
    "## Newtype And Category-Theory Lens",
    "## Required Checks",
    "## Assessment Rubric",
    "## Failure Signals",
    "## Suggested Repo Integration",
)
RAW_DEFAULT_LANGUAGE = (
    "raw math layer",
    "raw storage",
    "raw vector layer",
    "raw tensor layer",
)
RUST_SNIPPET_FORBIDDEN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("unwrap", re.compile(r"\.unwrap\s*\(")),
    ("expect", re.compile(r"\.expect\s*\(")),
    ("expect_err", re.compile(r"\.expect_err\s*\(")),
    ("panic macro", re.compile(r"\bpanic!\s*\(")),
    ("todo macro", re.compile(r"\btodo!\s*\(")),
    ("unimplemented macro", re.compile(r"\bunimplemented!\s*\(")),
    ("unreachable macro", re.compile(r"\bunreachable!\s*\(")),
    ("Result<_, String>", re.compile(r"Result\s*<[^>\n]+,\s*String\s*>")),
)
MARKDOWN_LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
HTML_SRC_RE = re.compile(r"""<img[^>]+src=["']([^"']+)["']""")
PACKAGE_NAME_RE = re.compile(r'^\s*name\s*=\s*"([^"]+)"\s*$', re.MULTILINE)
RUNNABLE_PROOF_RE = re.compile(r"- \*\*Runnable proof:\*\* (.+)")
DOCUMENTED_CARGO_RUN_RE = re.compile(
    r"cargo run --manifest-path code/Cargo\.toml -p ([a-zA-Z0-9_-]+) --example ([a-zA-Z0-9_-]+)"
)
DOCUMENTED_CARGO_TEST_RE = re.compile(
    r"cargo test --manifest-path code/Cargo\.toml"
    r"(?: -p ([a-zA-Z0-9_-]+))?"
    r"(?: --workspace)?"
    r"(?: --all-targets)?"
)
DOCUMENTED_PYTHON_CHECK_RE = re.compile(r"python3 (scripts/[a-zA-Z0-9_-]+\.py)")
SKIPPED_SCAN_DIRS = {
    ".git",
    ".idea",
    "target",
}


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def read_package_name(cargo_manifest: Path) -> str | None:
    text = cargo_manifest.read_text(encoding="utf-8")
    match = PACKAGE_NAME_RE.search(text)
    if match is None:
        return None
    return match.group(1)


def package_dirs_by_name() -> dict[str, Path]:
    packages: dict[str, Path] = {}
    for cargo_manifest in sorted((ROOT / "code").glob("*/Cargo.toml")):
        package_name = read_package_name(cargo_manifest)
        if package_name is not None:
            packages[package_name] = cargo_manifest.parent
    return packages


def line_number_for_match(text: str, start: int) -> int:
    return text.count("\n", 0, start) + 1


def iter_public_markdown_files() -> list[Path]:
    markdown_files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [
            dirname for dirname in dirnames if dirname not in SKIPPED_SCAN_DIRS
        ]
        directory = Path(dirpath)
        for filename in filenames:
            path = directory / filename
            if path.suffix == ".md":
                markdown_files.append(path)
    return markdown_files


def validate_cargo_run_command(
    origin: Path,
    line_number: int,
    package_name: str,
    example_name: str,
    packages: dict[str, Path],
) -> list[str]:
    package_dir = packages.get(package_name)
    if package_dir is None:
        return [
            f"{relative(origin)}:{line_number} documented command references "
            f"unknown package: {package_name}"
        ]

    example_file = package_dir / "examples" / f"{example_name}.rs"
    if not example_file.exists():
        return [
            f"{relative(origin)}:{line_number} documented command references "
            f"missing example: {relative(example_file)}"
        ]

    return []


def validate_cargo_test_command(
    origin: Path,
    line_number: int,
    package_name: str | None,
    packages: dict[str, Path],
) -> list[str]:
    if package_name is None:
        if not (ROOT / "code" / "Cargo.toml").exists():
            return [
                f"{relative(origin)}:{line_number} documented workspace test "
                "command requires code/Cargo.toml"
            ]
        return []

    if package_name not in packages:
        return [
            f"{relative(origin)}:{line_number} documented test command references "
            f"unknown package: {package_name}"
        ]

    return []


def validate_python_check_command(
    origin: Path,
    line_number: int,
    script: str,
) -> list[str]:
    script_path = ROOT / script
    if script_path.exists():
        return []

    return [
        f"{relative(origin)}:{line_number} documented Python command references "
        f"missing script: {relative(script_path)}"
    ]


def validate_documented_commands(
    origin: Path,
    text: str,
    packages: dict[str, Path],
) -> list[str]:
    errors: list[str] = []

    for match in DOCUMENTED_CARGO_RUN_RE.finditer(text):
        line_number = line_number_for_match(text, match.start())
        package_name, example_name = match.groups()
        errors.extend(
            validate_cargo_run_command(
                origin,
                line_number,
                package_name,
                example_name,
                packages,
            )
        )

    for match in DOCUMENTED_CARGO_TEST_RE.finditer(text):
        line_number = line_number_for_match(text, match.start())
        errors.extend(
            validate_cargo_test_command(origin, line_number, match.group(1), packages)
        )

    for match in DOCUMENTED_PYTHON_CHECK_RE.finditer(text):
        line_number = line_number_for_match(text, match.start())
        errors.extend(validate_python_check_command(origin, line_number, match.group(1)))

    return errors


def validate_runnable_proof(
    lesson_file: Path,
    line_number: int,
    proof: str,
    packages: dict[str, Path],
) -> list[str]:
    cargo_run = DOCUMENTED_CARGO_RUN_RE.search(proof)
    if cargo_run is not None:
        package_name, example_name = cargo_run.groups()
        return validate_cargo_run_command(
            lesson_file,
            line_number,
            package_name,
            example_name,
            packages,
        )

    cargo_test = DOCUMENTED_CARGO_TEST_RE.search(proof)
    if cargo_test is not None:
        return validate_cargo_test_command(
            lesson_file,
            line_number,
            cargo_test.group(1),
            packages,
        )

    python_check = DOCUMENTED_PYTHON_CHECK_RE.search(proof)
    if python_check is not None:
        return validate_python_check_command(
            lesson_file,
            line_number,
            python_check.group(1),
        )

    return [f"{relative(lesson_file)} has unsupported runnable proof command: {proof}"]


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

    for markdown_file in iter_public_markdown_files():
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


def check_documented_commands() -> list[str]:
    errors: list[str] = []
    packages = package_dirs_by_name()

    for markdown_file in iter_public_markdown_files():
        text = markdown_file.read_text(encoding="utf-8")
        errors.extend(validate_documented_commands(markdown_file, text, packages))

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


def check_practice_contract() -> list[str]:
    errors: list[str] = []

    for module_dir in AUTHORED_MODULES:
        exercises = module_dir / "exercises.md"
        if exercises.exists():
            text = exercises.read_text(encoding="utf-8")
            for marker in ("## Failure Signals", "## Debugging Hints"):
                if marker not in text:
                    errors.append(f"{relative(exercises)} is missing practice marker: {marker}")

        solutions = module_dir / "solutions.md"
        if solutions.exists():
            text = solutions.read_text(encoding="utf-8")
            if "## Self-Check" not in text:
                errors.append(f"{relative(solutions)} is missing practice marker: ## Self-Check")
            if "## Exercise " in text:
                errors.append(
                    f"{relative(solutions)} should use Solution headings, not Exercise headings"
                )

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
    packages = package_dirs_by_name()

    for module_dir in AUTHORED_MODULES:
        for lesson_file in sorted(module_dir.glob("[0-9][0-9]-*.md")):
            text = lesson_file.read_text(encoding="utf-8")
            if module_dir.name == "07-transformer":
                required_markers = (
                    "## Overview",
                    "## Learning Goals",
                    "## Concept Trace",
                    "### English",
                    "### Algebra",
                    "### Rust",
                )
            else:
                required_markers = tuple(f"## {heading}" for heading in REQUIRED_LESSON_HEADINGS)

            for marker in required_markers:
                if marker not in text:
                    errors.append(f"{relative(lesson_file)} is missing section: {marker}")

            if "## Concept Trace" in text:
                for marker in REQUIRED_CONCEPT_TRACE_MARKERS:
                    if marker not in text:
                        errors.append(
                            f"{relative(lesson_file)} is missing concept-trace marker: {marker}"
                        )

                runnable_proofs = list(RUNNABLE_PROOF_RE.finditer(text))
                if len(runnable_proofs) != 1:
                    errors.append(
                        f"{relative(lesson_file)} should contain exactly one Runnable proof line"
                    )
                for runnable_proof in runnable_proofs:
                    errors.extend(
                        validate_runnable_proof(
                            lesson_file,
                            line_number_for_match(text, runnable_proof.start()),
                            runnable_proof.group(1),
                            packages,
                        )
                    )

            if "```rust" not in text:
                errors.append(f"{relative(lesson_file)} should include at least one Rust block")

            if "TODO" in text:
                errors.append(f"{relative(lesson_file)} contains TODO and should be finished")

            if text.startswith("# Lesson "):
                errors.append(
                    f"{relative(lesson_file)} should use a concept-first title, not a global lesson number"
                )

    return errors


def check_no_raw_default_language() -> list[str]:
    errors: list[str] = []

    for markdown_file in iter_public_markdown_files():
        text = markdown_file.read_text(encoding="utf-8")
        lowered = text.lower()
        for phrase in RAW_DEFAULT_LANGUAGE:
            start = lowered.find(phrase)
            if start != -1:
                errors.append(
                    f"{relative(markdown_file)}:{line_number_for_match(text, start)} "
                    f"uses raw-default wording `{phrase}`; describe checked dense values or boundary literals instead"
                )

    return errors


def check_rust_snippet_contract() -> list[str]:
    errors: list[str] = []

    for markdown_file in iter_public_markdown_files():
        lines = markdown_file.read_text(encoding="utf-8").splitlines()
        in_rust_block = False

        for line_number, line in enumerate(lines, start=1):
            stripped = line.strip()
            if stripped.startswith("```"):
                if in_rust_block:
                    in_rust_block = False
                elif stripped.startswith("```rust"):
                    in_rust_block = True
                continue

            if not in_rust_block:
                continue

            for label, pattern in RUST_SNIPPET_FORBIDDEN_PATTERNS:
                if pattern.search(line):
                    errors.append(
                        f"{relative(markdown_file)}:{line_number} Rust snippet uses {label}; "
                        "public examples should return typed errors instead"
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
        "## Newtype And Category-Theory Spine",
        "## Concept Atlas Contract",
        "## Strict Rust Teaching Contract",
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
        "The Learning Lens",
        "Concept Atlas",
        "## Course Map",
        "## Current Recommended Paths",
        "Module 6 | Authored",
        "Module 7 | Authored",
        "Module 8 | Authored",
        "Module 9 | Authored",
        "Module 10 | Authored",
    )
    for phrase in required_phrases:
        if phrase not in text:
            errors.append(f"lessons/README.md is missing section marker: {phrase}")

    return errors


def check_root_readme_contract() -> list[str]:
    errors: list[str] = []
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")

    required_phrases = (
        "# Rust ML Systems from First Principles",
        "LEARNING-PATH.md",
        "lessons/CONCEPT-ATLAS.md",
        "PUBLIC_CONTENT.md",
        "CS336-RUST-EQUIVALENT.md",
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


def check_learning_path_contract() -> list[str]:
    errors: list[str] = []
    path = ROOT / "LEARNING-PATH.md"
    if not path.exists():
        return [f"{relative(path)} is required"]

    text = path.read_text(encoding="utf-8")
    required_phrases = (
        "# Learning Path",
        "plain English <-> algebra <-> Rust newtypes <-> composable maps",
        "lessons/CONCEPT-ATLAS.md",
        "## Recommended Route",
        "## How To Study",
        "## Mastery Checks",
        "## Public Resource Standard",
        "CS336 Rust Equivalent",
    )
    for phrase in required_phrases:
        if phrase not in text:
            errors.append(f"{relative(path)} is missing learning-path marker: {phrase}")

    return errors


def check_concept_atlas_contract() -> list[str]:
    errors: list[str] = []
    atlas = ROOT / "lessons" / "CONCEPT-ATLAS.md"
    if not atlas.exists():
        return [f"{relative(atlas)} is required"]

    text = atlas.read_text(encoding="utf-8")
    required_phrases = (
        "# Concept Atlas",
        "## How To Read The Atlas",
        "## Core Object Ladder",
        "## Map Ladder",
        "## Runnable Proofs",
        "## CS336 Extension",
        "## Mastery Trace",
        "ML idea -> meaningful Rust type -> checked map -> runnable proof",
        "RawText",
        "TokenId",
        "ComputeBudgetFlops",
        "AuditRecord",
    )
    for phrase in required_phrases:
        if phrase not in text:
            errors.append(f"{relative(atlas)} is missing concept-atlas marker: {phrase}")

    return errors


def check_cs336_assignment_contract() -> list[str]:
    errors: list[str] = []

    readme = CS336_ASSIGNMENTS_DIR / "README.md"
    if not readme.exists():
        errors.append(f"{relative(readme)} is required")
    else:
        text = readme.read_text(encoding="utf-8")
        for phrase in (
            "# CS336 Rust Assignments",
            "not copied from Stanford's handouts or repositories",
            "code/category_lens",
            "rust_ml_category_lens",
            "## Sequence",
            "## Shared Rules",
            "## Completion Standard",
        ):
            if phrase not in text:
                errors.append(f"{relative(readme)} is missing assignment marker: {phrase}")

    for assignment in CS336_ASSIGNMENT_DOCS:
        if not assignment.exists():
            errors.append(f"{relative(assignment)} is required")
            continue

        text = assignment.read_text(encoding="utf-8")
        for heading in REQUIRED_ASSIGNMENT_HEADINGS:
            if heading not in text:
                errors.append(f"{relative(assignment)} is missing section: {heading}")

        for phrase in ("newtype", "composition", "cargo run"):
            if phrase not in text:
                errors.append(f"{relative(assignment)} is missing pedagogy marker: {phrase}")

        for phrase in (
            "**Objects:**",
            "**Maps:**",
            "**Composition path:**",
            "**Invariant to protect with newtypes:**",
        ):
            if phrase not in text:
                errors.append(f"{relative(assignment)} is missing object/map preflight marker: {phrase}")

        if "TODO" in text:
            errors.append(f"{relative(assignment)} contains TODO and should be finished")

    return errors


def check_cs336_rust_track_contract() -> list[str]:
    errors: list[str] = []
    required_files = (
        ROOT / "CS336-RUST-EQUIVALENT.md",
        ROOT / "references" / "courses" / "cs336-language-modeling-from-scratch.md",
        ROOT / "assignments" / "cs336-rust" / "README.md",
        ROOT / "assignments" / "cs336-rust" / "01-basics.md",
        ROOT / "assignments" / "cs336-rust" / "02-systems.md",
        ROOT / "assignments" / "cs336-rust" / "03-scaling.md",
        ROOT / "assignments" / "cs336-rust" / "04-data.md",
        ROOT / "assignments" / "cs336-rust" / "05-alignment.md",
        ROOT / "code" / "category_lens" / "README.md",
        ROOT / "code" / "category_lens" / "src" / "lib.rs",
        ROOT / "code" / "alignment" / "README.md",
        ROOT / "code" / "alignment" / "src" / "lib.rs",
        ROOT / "code" / "data" / "README.md",
        ROOT / "code" / "data" / "src" / "lib.rs",
        ROOT / "code" / "evaluation" / "README.md",
        ROOT / "code" / "evaluation" / "src" / "lib.rs",
        ROOT / "code" / "inference" / "README.md",
        ROOT / "code" / "inference" / "src" / "lib.rs",
        ROOT / "code" / "kernels" / "README.md",
        ROOT / "code" / "kernels" / "src" / "lib.rs",
        ROOT / "code" / "lm_basics" / "README.md",
        ROOT / "code" / "lm_basics" / "src" / "lib.rs",
        ROOT / "code" / "parallelism" / "README.md",
        ROOT / "code" / "parallelism" / "src" / "lib.rs",
        ROOT / "code" / "scaling" / "README.md",
        ROOT / "code" / "scaling" / "src" / "lib.rs",
        ROOT / "code" / "systems" / "README.md",
        ROOT / "code" / "systems" / "src" / "lib.rs",
    )

    for required in required_files:
        if not required.exists():
            errors.append(f"{relative(required)} is required for the CS336 Rust track")

    overview = ROOT / "CS336-RUST-EQUIVALENT.md"
    if overview.exists():
        text = overview.read_text(encoding="utf-8")
        for phrase in (
            "not Stanford coursework",
            "Lecture-To-Rust Map",
            "Assignment Sequence",
            "Concept Atlas",
            "newtype-protected meaning",
            "code/category_lens",
            "rust_ml_category_lens",
            "code/alignment",
            "code/data",
            "code/evaluation",
            "code/inference",
            "code/kernels",
            "code/lm_basics",
            "code/parallelism",
            "code/scaling",
            "code/systems",
        ):
            if phrase not in text:
                errors.append(f"{relative(overview)} is missing CS336 track marker: {phrase}")

    r1 = ROOT / "assignments" / "cs336-rust" / "01-basics.md"
    if r1.exists():
        text = r1.read_text(encoding="utf-8")
        for phrase in (
            "Active Starter Crate",
            "code/lm_basics",
            "rust_ml_lm_basics",
        ):
            if phrase not in text:
                errors.append(f"{relative(r1)} is missing R1 executable marker: {phrase}")

    r2 = ROOT / "assignments" / "cs336-rust" / "02-systems.md"
    if r2.exists():
        text = r2.read_text(encoding="utf-8")
        for phrase in (
            "Active Starter Crate",
            "code/systems",
            "rust_ml_systems",
        ):
            if phrase not in text:
                errors.append(f"{relative(r2)} is missing R2 executable marker: {phrase}")

    r3 = ROOT / "assignments" / "cs336-rust" / "03-scaling.md"
    if r3.exists():
        text = r3.read_text(encoding="utf-8")
        for phrase in (
            "Active Starter Crate",
            "code/scaling",
            "rust_ml_scaling",
        ):
            if phrase not in text:
                errors.append(f"{relative(r3)} is missing R3 executable marker: {phrase}")

    r4 = ROOT / "assignments" / "cs336-rust" / "04-data.md"
    if r4.exists():
        text = r4.read_text(encoding="utf-8")
        for phrase in (
            "Active Starter Crate",
            "code/data",
            "rust_ml_data",
        ):
            if phrase not in text:
                errors.append(f"{relative(r4)} is missing R4 executable marker: {phrase}")

    r5 = ROOT / "assignments" / "cs336-rust" / "05-alignment.md"
    if r5.exists():
        text = r5.read_text(encoding="utf-8")
        for phrase in (
            "Active Starter Crate",
            "code/alignment",
            "rust_ml_alignment",
        ):
            if phrase not in text:
                errors.append(f"{relative(r5)} is missing R5 executable marker: {phrase}")

    source_map = ROOT / "references" / "courses" / "cs336-language-modeling-from-scratch.md"
    if source_map.exists():
        text = source_map.read_text(encoding="utf-8")
        for phrase in (
            "Spring 2026",
            "Public Assignment Tracks",
            "Public Lecture Map",
            "Public Lecture Material Links",
            "lecture_17.py",
            "It is not a copy",
        ):
            if phrase not in text:
                errors.append(f"{relative(source_map)} is missing source-map marker: {phrase}")

    return errors


def main() -> int:
    checks = (
        check_markdown_links,
        check_documented_commands,
        check_authored_module_contract,
        check_practice_contract,
        check_module_readmes,
        check_lesson_sections,
        check_no_raw_default_language,
        check_rust_snippet_contract,
        check_structure_guide,
        check_lessons_index_contract,
        check_root_readme_contract,
        check_learning_path_contract,
        check_concept_atlas_contract,
        check_cs336_assignment_contract,
        check_cs336_rust_track_contract,
    )

    errors: list[str] = []
    for check in checks:
        errors.extend(check())

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    markdown_count = len(iter_public_markdown_files())
    print(f"Course content checks passed across {markdown_count} Markdown files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
