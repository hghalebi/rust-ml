#!/usr/bin/env python3
"""Check Rust teaching-code contracts that are easy to regress."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = ROOT / "code"

STRICT_PUBLIC_API_PATHS = (
    CODE_ROOT / "alignment" / "src",
    CODE_ROOT / "attention" / "src",
    CODE_ROOT / "category_lens" / "src",
    CODE_ROOT / "data" / "src",
    CODE_ROOT / "evaluation" / "src",
    CODE_ROOT / "inference" / "src",
    CODE_ROOT / "kernels" / "src",
    CODE_ROOT / "lm_basics" / "src",
    CODE_ROOT / "mlp" / "src",
    CODE_ROOT / "neuron" / "src",
    CODE_ROOT / "parallelism" / "src",
    CODE_ROOT / "scaling" / "src",
    CODE_ROOT / "systems" / "src",
    CODE_ROOT / "transformer" / "src",
)

STRICT_TYPED_ERROR_PATHS = (
    CODE_ROOT / "alignment" / "src" / "error.rs",
    CODE_ROOT / "attention" / "src" / "error.rs",
    CODE_ROOT / "data" / "src" / "error.rs",
    CODE_ROOT / "evaluation" / "src" / "error.rs",
    CODE_ROOT / "inference" / "src" / "error.rs",
    CODE_ROOT / "kernels" / "src" / "error.rs",
    CODE_ROOT / "lm_basics" / "src" / "error.rs",
    CODE_ROOT / "mlp" / "src" / "error.rs",
    CODE_ROOT / "neuron" / "src" / "error.rs",
    CODE_ROOT / "parallelism" / "src" / "error.rs",
    CODE_ROOT / "scaling" / "src" / "error.rs",
    CODE_ROOT / "systems" / "src" / "error.rs",
    CODE_ROOT / "transformer" / "src" / "error.rs",
)

STRICT_TYPED_TEST_HELPER_PATHS = (
    CODE_ROOT / "alignment" / "src" / "lib.rs",
    CODE_ROOT / "attention" / "src" / "lib.rs",
    CODE_ROOT / "data" / "src" / "lib.rs",
    CODE_ROOT / "evaluation" / "src" / "lib.rs",
    CODE_ROOT / "inference" / "src" / "lib.rs",
    CODE_ROOT / "kernels" / "src" / "lib.rs",
    CODE_ROOT / "lm_basics" / "src" / "lib.rs",
    CODE_ROOT / "mlp" / "src" / "lib.rs",
    CODE_ROOT / "neuron" / "src" / "lib.rs",
    CODE_ROOT / "parallelism" / "src" / "lib.rs",
    CODE_ROOT / "scaling" / "src" / "lib.rs",
    CODE_ROOT / "systems" / "src" / "lib.rs",
    CODE_ROOT / "transformer" / "src" / "attention.rs",
    CODE_ROOT / "transformer" / "src" / "math.rs",
    CODE_ROOT / "transformer" / "src" / "transformer.rs",
    CODE_ROOT / "transformer" / "src" / "types.rs",
)

STRICT_TYPED_TEST_BODY_PATHS = STRICT_PUBLIC_API_PATHS

PANIC_PATTERNS = (
    ("unwrap", re.compile(r"\.unwrap\s*\(")),
    ("expect", re.compile(r"\.expect\s*\(")),
    ("expect_err", re.compile(r"\.expect_err\s*\(")),
    ("panic macro", re.compile(r"\bpanic!\s*\(")),
    ("todo macro", re.compile(r"\btodo!\s*\(")),
    ("unimplemented macro", re.compile(r"\bunimplemented!\s*\(")),
    ("unreachable macro", re.compile(r"\bunreachable!\s*\(")),
)

STRING_ERROR_RE = re.compile(r"Result\s*<[^>\n]+,\s*String\s*>")
RAW_STRING_RESULT_RE = re.compile(r"Result\s*<\s*String\s*,")
PUBLIC_FUNCTION_RE = re.compile(r"^\s*pub\s+fn\s+")
PUBLIC_TRAIT_RE = re.compile(r"^\s*pub\s+trait\s+")
TRAIT_METHOD_RE = re.compile(r"^\s*fn\s+\w+\s*")
PUBLIC_TYPE_ALIAS_RE = re.compile(r"^\s*pub\s+type\s+\w+\s*=\s*(.+);")
ASSOCIATED_TYPE_ASSIGNMENT_RE = re.compile(r"^\s*type\s+\w+\s*=\s*(.+);")
PUBLIC_CONST_OR_STATIC_RE = re.compile(r"^\s*pub\s+(?:const|static)\s+\w+\s*:\s*(.+)=")
PUBLIC_TUPLE_FIELD_RE = re.compile(r"^\s*pub\s+struct\s+\w+\s*\(\s*pub\b")
PUBLIC_FIELD_RE = re.compile(r"^\s*pub\s+\w+\s*:\s*")
ENUM_FIELD_RE = re.compile(r"^\s*\w+\s*:\s*(.+),\s*$")
RAW_COMPARISON_IMPL_RE = re.compile(
    r"^\s*impl\s+Partial(?:Eq|Ord)\s*<\s*"
    r"(?:usize|u64|i64|f64|f32|bool|String|&str)"
    r"\s*>\s+for\s+\w+"
)
RAW_ADAPTER_IMPL_RE = re.compile(
    r"^\s*impl\s+[\w:<>]+\s+for\s+"
    r"(?:usize|u64|u128|i64|f64|f32|bool|String|&str)\b"
)
RAW_CONTAINER_TRY_FROM_IMPL_RE = re.compile(
    r"^\s*impl\s+TryFrom\s*<\s*(?:Vec\s*<|&\s*\[|\[)"
)
RAW_ACCESSOR_RE = re.compile(r"\.as_(?:usize|u64|u128|f64|f32|raw_slice)\s*\(")
RAW_PUBLIC_TYPE_RE = re.compile(
    r"("
    r"\bVec\s*<"
    r"|\bString\b"
    r"|&\s*str\b"
    r"|&\s*'static\s+str\b"
    r"|\busize\b"
    r"|\bu64\b"
    r"|\bi64\b"
    r"|\bf64\b"
    r"|\bf32\b"
    r"|\bbool\b"
    r"|\bHashMap\s*<"
    r"|\bBTreeMap\s*<"
    r"|\bserde_json::Value\b"
    r"|&\s*\["
    r")"
)
SKIPPED_RUST_DIR_NAMES = {
    ".git",
    ".idea",
    "target",
}


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def is_scanned_rust_file(path: Path) -> bool:
    if path.suffix != ".rs":
        return False
    return not any(part in SKIPPED_RUST_DIR_NAMES for part in path.relative_to(ROOT).parts)


def scanned_rust_files_under(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if is_scanned_rust_file(path) else []

    files: list[Path] = []
    stack = [path]
    while stack:
        current = stack.pop()
        if current.name in SKIPPED_RUST_DIR_NAMES:
            continue

        try:
            entries = list(current.iterdir())
        except FileNotFoundError:
            continue

        for entry in entries:
            if entry.name in SKIPPED_RUST_DIR_NAMES:
                continue
            if entry.is_dir():
                stack.append(entry)
            elif is_scanned_rust_file(entry):
                files.append(entry)

    return files


def rust_files_under(paths: tuple[Path, ...]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.exists():
            files.extend(scanned_rust_files_under(path))
    return sorted(files)


def all_rust_files() -> list[Path]:
    if not CODE_ROOT.exists():
        return []
    return sorted(scanned_rust_files_under(CODE_ROOT))


def collect_public_function_signatures(lines: list[str]) -> list[tuple[int, str]]:
    signatures: list[tuple[int, str]] = []
    collecting = False
    start_line = 0
    current: list[str] = []
    paren_depth = 0

    for line_number, line in enumerate(lines, start=1):
        if not collecting and not PUBLIC_FUNCTION_RE.search(line):
            continue

        if not collecting:
            collecting = True
            start_line = line_number
            current = []
            paren_depth = 0

        current.append(line.strip())
        paren_depth += line.count("(") - line.count(")")

        if paren_depth <= 0 and ("{" in line or ";" in line):
            signatures.append((start_line, " ".join(current)))
            collecting = False

    return signatures


def collect_public_trait_method_signatures(lines: list[str]) -> list[tuple[int, str]]:
    signatures: list[tuple[int, str]] = []
    in_public_trait = False
    brace_depth = 0
    collecting = False
    start_line = 0
    current: list[str] = []
    paren_depth = 0

    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()

        if not in_public_trait and PUBLIC_TRAIT_RE.search(line) and "{" in line:
            in_public_trait = True
            brace_depth = line.count("{") - line.count("}")
            continue

        if not in_public_trait:
            continue

        if not collecting and TRAIT_METHOD_RE.search(line):
            collecting = True
            start_line = line_number
            current = []
            paren_depth = 0

        if collecting:
            current.append(stripped)
            paren_depth += line.count("(") - line.count(")")
            if paren_depth <= 0 and (";" in line or "{" in line):
                signatures.append((start_line, " ".join(current)))
                collecting = False

        brace_depth += line.count("{") - line.count("}")
        if brace_depth <= 0:
            in_public_trait = False
            collecting = False

    return signatures


def collect_test_function_signatures(lines: list[str]) -> list[tuple[int, str]]:
    signatures: list[tuple[int, str]] = []
    in_test_module = False
    brace_depth = 0
    collecting = False
    start_line = 0
    current: list[str] = []
    paren_depth = 0

    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()

        if not in_test_module and stripped == "mod tests {":
            in_test_module = True
            brace_depth = line.count("{") - line.count("}")
            continue

        if not in_test_module:
            continue

        if not collecting and stripped.startswith("fn "):
            collecting = True
            start_line = line_number
            current = []
            paren_depth = 0

        if collecting:
            current.append(stripped)
            paren_depth += line.count("(") - line.count(")")
            if paren_depth <= 0 and ("{" in line or ";" in line):
                signatures.append((start_line, " ".join(current)))
                collecting = False

        brace_depth += line.count("{") - line.count("}")
        if brace_depth <= 0:
            in_test_module = False
            collecting = False

    return signatures


def check_no_panic_shortcuts() -> list[str]:
    errors: list[str] = []
    for path in all_rust_files():
        text = path.read_text(encoding="utf-8")
        for label, pattern in PANIC_PATTERNS:
            for match in pattern.finditer(text):
                line_number = text.count("\n", 0, match.start()) + 1
                errors.append(
                    f"{relative(path)}:{line_number} uses {label}; return typed errors instead"
                )
    return errors


def check_no_string_error_types() -> list[str]:
    errors: list[str] = []
    for path in all_rust_files():
        text = path.read_text(encoding="utf-8")
        for match in STRING_ERROR_RE.finditer(text):
            line_number = text.count("\n", 0, match.start()) + 1
            errors.append(
                f"{relative(path)}:{line_number} uses Result<_, String>; define a thiserror error"
            )
    return errors


def check_no_raw_string_results() -> list[str]:
    errors: list[str] = []
    for path in rust_files_under(STRICT_PUBLIC_API_PATHS):
        text = path.read_text(encoding="utf-8")
        for match in RAW_STRING_RESULT_RE.finditer(text):
            line_number = text.count("\n", 0, match.start()) + 1
            errors.append(
                f"{relative(path)}:{line_number} returns Result<String, _>; validate then wrap in a newtype"
            )
    return errors


def check_typed_error_fields() -> list[str]:
    errors: list[str] = []

    for path in STRICT_TYPED_ERROR_PATHS:
        lines = path.read_text(encoding="utf-8").splitlines()
        in_public_enum = False
        brace_depth = 0

        for line_number, line in enumerate(lines, start=1):
            stripped = line.strip()

            if not in_public_enum and stripped.startswith("pub enum ") and "{" in stripped:
                in_public_enum = True
                brace_depth = line.count("{") - line.count("}")
                continue

            if not in_public_enum:
                continue

            field = ENUM_FIELD_RE.search(line)
            if field and RAW_PUBLIC_TYPE_RE.search(field.group(1)):
                errors.append(
                    f"{relative(path)}:{line_number} exposes a raw public error field: {stripped}"
                )

            brace_depth += line.count("{") - line.count("}")
            if brace_depth <= 0:
                in_public_enum = False

    return errors


def check_public_enum_variants() -> list[str]:
    errors: list[str] = []

    for path in rust_files_under(STRICT_PUBLIC_API_PATHS):
        lines = path.read_text(encoding="utf-8").splitlines()
        in_public_enum = False
        brace_depth = 0

        for line_number, line in enumerate(lines, start=1):
            stripped = line.strip()

            if not in_public_enum and stripped.startswith("pub enum ") and "{" in stripped:
                in_public_enum = True
                brace_depth = line.count("{") - line.count("}")
                continue

            if not in_public_enum:
                continue

            if (
                stripped
                and not stripped.startswith(("///", "//", "#"))
                and RAW_PUBLIC_TYPE_RE.search(stripped)
            ):
                errors.append(
                    f"{relative(path)}:{line_number} exposes a raw primitive/container in a public enum variant: {stripped}"
                )

            brace_depth += line.count("{") - line.count("}")
            if brace_depth <= 0:
                in_public_enum = False

    return errors


def check_strict_public_api() -> list[str]:
    errors: list[str] = []
    for path in rust_files_under(STRICT_PUBLIC_API_PATHS):
        lines = path.read_text(encoding="utf-8").splitlines()

        for line_number, signature in collect_public_function_signatures(lines):
            if RAW_PUBLIC_TYPE_RE.search(signature):
                errors.append(
                    f"{relative(path)}:{line_number} exposes a raw primitive/container in a public function: {signature}"
                )

        for line_number, signature in collect_public_trait_method_signatures(lines):
            if RAW_PUBLIC_TYPE_RE.search(signature):
                errors.append(
                    f"{relative(path)}:{line_number} exposes a raw primitive/container in a public trait method: {signature}"
                )

        for line_number, line in enumerate(lines, start=1):
            associated_type = ASSOCIATED_TYPE_ASSIGNMENT_RE.search(line)
            if associated_type and RAW_PUBLIC_TYPE_RE.search(associated_type.group(1)):
                errors.append(
                    f"{relative(path)}:{line_number} exposes a raw associated type assignment: {line.strip()}"
                )
            type_alias = PUBLIC_TYPE_ALIAS_RE.search(line)
            if type_alias and RAW_PUBLIC_TYPE_RE.search(type_alias.group(1)):
                errors.append(
                    f"{relative(path)}:{line_number} exposes a raw public type alias: {line.strip()}"
                )
            const_or_static = PUBLIC_CONST_OR_STATIC_RE.search(line)
            if const_or_static and RAW_PUBLIC_TYPE_RE.search(const_or_static.group(1)):
                errors.append(
                    f"{relative(path)}:{line_number} exposes a raw public constant/static type: {line.strip()}"
                )
            if RAW_COMPARISON_IMPL_RE.search(line):
                errors.append(
                    f"{relative(path)}:{line_number} compares a semantic type directly with a raw primitive; compare two semantic values instead"
                )
            if RAW_ADAPTER_IMPL_RE.search(line):
                errors.append(
                    f"{relative(path)}:{line_number} implements a public adapter trait for a raw primitive; use TryFrom<raw> for the semantic type at the boundary"
                )
            if RAW_CONTAINER_TRY_FROM_IMPL_RE.search(line):
                errors.append(
                    f"{relative(path)}:{line_number} implements public TryFrom for a raw container; use typed constructors or crate-private boundary helpers"
                )
            if PUBLIC_TUPLE_FIELD_RE.search(line):
                errors.append(
                    f"{relative(path)}:{line_number} exposes a public tuple field; use a private newtype field"
                )
            if PUBLIC_FIELD_RE.search(line) and RAW_PUBLIC_TYPE_RE.search(line):
                errors.append(
                    f"{relative(path)}:{line_number} exposes a raw public field: {line.strip()}"
                )

    return errors


def check_typed_test_helper_signatures() -> list[str]:
    errors: list[str] = []

    for path in rust_files_under(STRICT_TYPED_TEST_HELPER_PATHS):
        lines = path.read_text(encoding="utf-8").splitlines()
        for line_number, signature in collect_test_function_signatures(lines):
            if RAW_PUBLIC_TYPE_RE.search(signature):
                errors.append(
                    f"{relative(path)}:{line_number} test helper exposes a raw primitive/container signature: {signature}"
                )

    return errors


def check_no_raw_test_accessors() -> list[str]:
    errors: list[str] = []

    for path in rust_files_under(STRICT_TYPED_TEST_BODY_PATHS):
        lines = path.read_text(encoding="utf-8").splitlines()
        in_test_module = False
        brace_depth = 0

        for line_number, line in enumerate(lines, start=1):
            stripped = line.strip()

            if not in_test_module and stripped == "mod tests {":
                in_test_module = True
                brace_depth = line.count("{") - line.count("}")
                continue

            if not in_test_module:
                continue

            if RAW_ACCESSOR_RE.search(line):
                errors.append(
                    f"{relative(path)}:{line_number} test body uses a raw accessor; assert through semantic values or Display instead"
                )

            brace_depth += line.count("{") - line.count("}")
            if brace_depth <= 0:
                in_test_module = False

    return errors


def main() -> int:
    errors = (
        check_no_panic_shortcuts()
        + check_no_string_error_types()
        + check_no_raw_string_results()
        + check_typed_error_fields()
        + check_public_enum_variants()
        + check_strict_public_api()
        + check_typed_test_helper_signatures()
        + check_no_raw_test_accessors()
    )

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    strict_paths = ", ".join(relative(path) for path in STRICT_PUBLIC_API_PATHS)
    print(
        "Rust teaching contract checks passed: no unwrap/expect/panic/todo/unimplemented/unreachable, "
        "no String errors, no Result<String, _> validation leaks, "
        "typed public error fields for active teaching error modules, "
        "no raw primitive public enum payloads, "
        "no raw associated type assignments, "
        "no raw primitive public adapter impls, "
        "no raw public container TryFrom adapters, "
        "typed helper signatures in migrated test modules, "
        "no raw scalar accessors in strict migrated test bodies, "
        f"strict public API paths: {strict_paths}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
