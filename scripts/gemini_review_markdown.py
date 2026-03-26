#!/usr/bin/env python3
"""Review Markdown teaching content with Gemini against a technical-writing rubric."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import textwrap
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROMPT_PATH = ROOT / ".github" / "prompts" / "gemini-technical-writing-review.md"
DEFAULT_OUTPUT = ROOT / "gemini-writing-review.md"
REVIEWABLE_ROOTS = (
    ROOT / "lessons",
    ROOT / "code",
    ROOT / "references",
    ROOT / "book",
    ROOT / "README.md",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-sha")
    parser.add_argument("--head-sha")
    parser.add_argument("--all-markdown", action="store_true")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--max-files", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def repo_relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def run_git(args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def discover_markdown_files(args: argparse.Namespace) -> list[Path]:
    if args.all_markdown:
        return [
            path
            for path in sorted(ROOT.rglob("*.md"))
            if is_reviewable_markdown(path)
        ]

    if args.base_sha and args.head_sha:
        output = run_git(["diff", "--name-only", args.base_sha, args.head_sha, "--", "*.md"])
        files = [ROOT / line.strip() for line in output.splitlines() if line.strip()]
        return sorted(path for path in files if path.exists() and is_reviewable_markdown(path))

    output = run_git(["status", "--short", "--", "*.md"])
    files = []
    for line in output.splitlines():
        candidate = line[3:].strip()
        if candidate:
            path = ROOT / candidate
            if path.exists() and is_reviewable_markdown(path):
                files.append(path)
    return sorted(files)


def is_reviewable_markdown(path: Path) -> bool:
    resolved = path.resolve()
    if resolved == (ROOT / "README.md").resolve():
        return True

    return any(
        root.is_dir() and resolved.is_relative_to(root.resolve())
        for root in REVIEWABLE_ROOTS
        if root.is_dir()
    )


def load_prompt_template() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8").strip()


def build_prompt(template: str, markdown_file: Path) -> str:
    file_path = repo_relative(markdown_file)
    body = markdown_file.read_text(encoding="utf-8")
    return textwrap.dedent(
        f"""
        {template}

        File path: {file_path}

        Content to review:

        ```markdown
        {body}
        ```
        """
    ).strip()


def call_gemini(prompt: str, api_key: str, model: str) -> str:
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt,
                    }
                ]
            }
        ]
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            parsed = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemini API request failed: {error.code} {body}") from error

    candidates = parsed.get("candidates", [])
    if not candidates:
        raise RuntimeError(f"Gemini API returned no candidates: {parsed}")

    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(part.get("text", "") for part in parts)
    if not text.strip():
        raise RuntimeError(f"Gemini API returned an empty review: {parsed}")
    return text.strip()


def write_report(output_path: Path, report: str) -> None:
    output_path.write_text(report, encoding="utf-8")

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as summary_file:
            summary_file.write(report)
            summary_file.write("\n")


def main() -> int:
    args = parse_args()
    files = discover_markdown_files(args)

    if not files:
        report = "# Gemini writing review\n\nNo Markdown files matched the current review scope.\n"
        write_report(Path(args.output), report)
        print("No Markdown files matched the review scope.")
        return 0

    files = files[: args.max_files]
    template = load_prompt_template()
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    model = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash").strip()

    if args.dry_run or not api_key:
        file_list = "\n".join(f"- {repo_relative(path)}" for path in files)
        report = (
            "# Gemini writing review\n\n"
            f"Model: `{model}`\n\n"
            "Gemini review was not executed.\n\n"
            f"Dry run: `{args.dry_run}`\n"
            f"API key configured: `{bool(api_key)}`\n\n"
            "Files that would be reviewed:\n"
            f"{file_list}\n"
        )
        write_report(Path(args.output), report)
        print("Gemini review skipped.")
        return 0

    sections = ["# Gemini writing review", "", f"Model: `{model}`", ""]
    for markdown_file in files:
        prompt = build_prompt(template, markdown_file)
        review = call_gemini(prompt, api_key, model)
        sections.append(review)
        sections.append("")

    report = "\n".join(sections).rstrip() + "\n"
    write_report(Path(args.output), report)
    print(f"Wrote Gemini review to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
