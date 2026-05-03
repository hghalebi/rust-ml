# Book Wrapper

This directory is intentionally thin, but it now includes the book-level front matter.

Current book artifact:

- [Cover and front matter](COVER.md)

This directory is reserved for a future presentation layer such as mdBook once the authored lesson structure in [`lessons/`](../lessons/README.md) is stable enough to mirror safely.

## Current Rule

- `lessons/` is the source of truth.
- `book/COVER.md` is front matter and positioning, not a duplicate lesson surface.
- `book/` is non-authoritative for lesson content.
- [`lessons/COURSE-STRUCTURE.md`](../lessons/COURSE-STRUCTURE.md) defines the canonical curriculum contract.

## Why

Duplicating the course into both `lessons/` and `book/` this early would create drift.

## Later Use

When the course is stable enough for mdBook or a site wrapper, this directory can host that presentation layer without changing where the content is authored. The current cover can become the opening page for that wrapper.

Until then, keeping `book/` intentionally small is a coherence rule, not a missing-feature accident.
