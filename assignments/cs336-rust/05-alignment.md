# R5 Alignment: Post-Training Signals

## Goal

Model post-training as explicit data and feedback, not as mysterious magic.

The learner should separate:

```text
instruction examples, preferences, rewards, verifier signals, policy updates
```

## What You Build

Create a small post-training pipeline with:

- instruction-response examples for supervised finetuning
- preference pairs with chosen and rejected responses
- a tiny reward or preference-scoring function
- verifier feedback for a toy reasoning task
- an audit log for each training signal

## Active Starter Crate

The first executable artifact is [`code/alignment`](../../code/alignment/README.md).

It starts with typed post-training signal flow:

```text
PromptedResponse -> PreferenceSignal -> UpdateSignal -> AuditRecord
AuditRecord -> AlignmentWorkflow -> AlignmentTransition
ReviewedAlignmentWorkflow -> PublicAlignmentRelease
```

Run it with:

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_alignment --example 01_instruction_example
cargo run --manifest-path code/Cargo.toml -p rust_ml_alignment --example 02_preference_signal
cargo run --manifest-path code/Cargo.toml -p rust_ml_alignment --example 03_verifier_feedback
cargo run --manifest-path code/Cargo.toml -p rust_ml_alignment --example 04_audit_record
cargo run --manifest-path code/Cargo.toml -p rust_ml_alignment --example 05_alignment_workflow
cargo run --manifest-path code/Cargo.toml -p rust_ml_alignment --example 06_public_release
```

## Object/Map Preflight

Before implementation, write this preflight in your assignment notes:

- **Objects:** `Instruction`, `Response`, `PreferencePair`, `RewardScore`, `VerifierResult`, `UpdateSignal`, `AuditRecord`, `AlignmentWorkflow`, `AlignmentStage`, `AlignmentVisibility`, and `PublicAlignmentRelease`.
- **Maps:** form prompted response, compare preference pair, score reward, preserve verifier feedback, create update signal, write audit record, move the workflow through checked lifecycle stages, then review it for public release.
- **Composition path:** `PromptedResponse -> PreferenceSignal -> VerifierFeedback -> UpdateSignal -> AuditRecord -> AlignmentWorkflow -> PublicAlignmentRelease`.
- **Invariant to protect with newtypes:** feedback must keep its source, role, run identity, lifecycle stage, failure state, and public-release class visible.

## Expected Deliverables

- one instruction-response example for supervised learning
- one preference pair with chosen and rejected responses
- one reward-score ordering fixture
- one verifier-feedback example that keeps failures visible
- one audit record that preserves the signal source and run identity
- one workflow trace that rejects applying an update before audit approval
- one public-release trace that rejects restricted or private workflows

## Newtype And Category-Theory Lens

Use newtypes for:

- `Instruction`
- `Response`
- `ChosenResponse`
- `RejectedResponse`
- `PreferencePair`
- `RewardScore`
- `VerifierResult`
- `AlignmentRunId`
- `AlignmentWorkflow`
- `AlignmentStage`
- `AlignmentTransition`
- `AlignmentVisibility`
- `PublicAlignmentRelease`

The feedback composition is:

```text
PromptedResponse -> PreferenceSignal -> UpdateSignal -> AuditRecord
AuditRecord -> AlignmentWorkflow -> AlignmentTransition
ReviewedAlignmentWorkflow -> PublicAlignmentRelease
```

The important invariant is that feedback must keep its source and meaning, and
workflow stages must prevent out-of-order updates. A learner-facing public
release also requires explicit visibility review.

## Required Checks

- reject preference pairs where chosen and rejected text are identical
- keep verifier failures visible
- test reward-score ordering on a tiny fixture
- record every post-training signal with an auditable source
- reject workflow transitions that skip the audit gate
- reject restricted or private workflows at the public-release boundary

## Assessment Rubric

- **Signal separation:** instructions, responses, preferences, rewards, verifier results, and audit records remain distinct typed concepts.
- **Feedback integrity:** every update signal keeps its source, meaning, and run identity.
- **Failure visibility:** verifier failures are represented as data, not hidden behind a success path.
- **Lifecycle safety:** an update cannot be applied before a signal is audited.
- **Public safety:** public examples are constructed only from reviewed, public, audit-complete workflows.
- **Safety restraint:** the assignment explains toy post-training flow without claiming production alignment guarantees.

## Failure Signals

- chosen and rejected responses can be identical and still form a preference pair
- verifier failures are dropped, converted into success, or logged only as unstructured text
- reward score comparisons use raw floating-point values across the public boundary
- audit records omit the source of a post-training signal
- workflow code permits `UpdateApplied` without an audit-approved transition
- restricted or private feedback can appear in learner-facing public release material

## Suggested Repo Integration

Start from the active `code/alignment` crate. Keep this assignment toy-sized and transparent.

The purpose is to understand alignment data flow and failure modes, not to claim production alignment safety.
