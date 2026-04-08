# Book

This directory is intentionally thin in the bootstrap pass.

It is reserved for a future presentation layer such as mdBook once the authored lesson structure in [`lessons/`](../lessons/README.md) is stable enough to mirror safely.

## Current Rule

- `lessons/` is the source of truth.
- `book/` is non-authoritative.
- [`lessons/COURSE-STRUCTURE.md`](../lessons/COURSE-STRUCTURE.md) defines the canonical curriculum contract.

## Why

Duplicating the course into both `lessons/` and `book/` this early would create drift.

## Later Use

When the course is stable enough for mdBook or a site wrapper, this directory can host that presentation layer without changing where the content is authored.

Until then, keeping `book/` intentionally small is a coherence rule, not a missing-feature accident.
