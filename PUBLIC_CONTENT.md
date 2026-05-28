# Public Content Boundary

This repository is intended to be a public learner-facing resource.

The public surface is:

- [`README.md`](README.md)
- [`LEARNING-PATH.md`](LEARNING-PATH.md)
- [`CS336-RUST-EQUIVALENT.md`](CS336-RUST-EQUIVALENT.md)
- [`lessons/`](lessons/README.md)
- [`assignments/`](assignments/cs336-rust/README.md)
- [`code/`](code/README.md)
- public reference notes under [`references/`](references/README.md)

## Allowed Public Content

Public files may contain:

- explanations for learners
- exercises and solutions
- runnable commands
- source code intended for study
- public paper or repository references
- public project roadmap statements

## Not Allowed In Public Learner Content

Public learner content must not contain:

- credentials
- secret values
- local machine paths
- private operational notes
- deployment-only instructions
- personal data
- unpublished maintainer context

## Validation

Run:

```bash
python3 scripts/check_public_content.py
python3 scripts/check_rust_teaching_contract.py
```

These checks scan learner-facing Markdown and teaching code for common secret-shaped values, private-local path leaks, panic shortcuts, raw public domain types in the strict teaching crates, and untyped `String` error surfaces.
