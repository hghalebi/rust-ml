# Inference Solutions

## Solution 1: Read the typed state loop

For the cache trace example, expect the generated length to match the cache entry
count for generated steps, and one final cache line for each generated token.

`ContextTokens + TokenId -> ContextTokens` is the typed map that enforces a
consistent context update at each step.

The first cache role records the initial prompt continuation before generation
steps; later entries represent generated-token state that should match generation
counts.

## Solution 2: Validate the public release rule

`TraceVisibility::Public` is allowed.

`TraceVisibility::ResearchRestricted` and `TraceVisibility::Private` are blocked.

`PublicDecodeTrace::from_reviewed_trace` returns `Err` with a typed message
describing that non-public visibility cannot be published.

## Solution 3: Compare two latency assemblies

Total latency is composed from two typed components:

`prefill + (per-token * generated_token_count)`

Per-token latency must multiply by `GeneratedTokenCount` because runtime grows as
one token is added each decode step, not as a raw integer.

`LatencyStatus::WithinBudget` means `total <= limit` by constructor-checked
typed comparison.

## Self-Check

- Can you explain the three lanes that change in each decode step?
- Can you name the constructor that creates a public-trace boundary?
- Can you describe the latency composition without using raw arithmetic?
- Can you predict when a reviewed trace should be blocked from publication?
