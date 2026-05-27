# alignment

Status: active.

This crate is the first executable companion for [R5 Alignment](../../assignments/cs336-rust/05-alignment.md) in the CS336 Rust equivalent track.

It teaches post-training as typed, auditable signal flow:

```text
PromptedResponse -> PreferenceSignal -> UpdateSignal -> AuditRecord
AuditRecord -> AlignmentWorkflow -> AlignmentTransition
ReviewedAlignmentWorkflow -> PublicAlignmentRelease
```

## Owns

- assignment: [R5 Alignment](../../assignments/cs336-rust/05-alignment.md)
- track: [CS336 Rust Equivalent](../../CS336-RUST-EQUIVALENT.md)

## Current State

- active teaching crate
- typed instructions, responses, chosen responses, rejected responses, reward scores, reward margins, verifier results, run IDs, signal sources, and audit notes
- supervised instruction-response examples
- preference pairs that reject identical chosen and rejected responses
- reward bundles that preserve chosen/rejected roles before producing a signal
- reward margins with checked finite score arithmetic
- verifier feedback that keeps failures visible
- update signals and audit records that preserve source and meaning
- workflow stages and transitions that reject out-of-order updates
- reviewed alignment workflows and public releases that reject restricted or private signals

## Layout

```text
src/
  error.rs
  lib.rs
examples/
  01_instruction_example.rs
  02_preference_signal.rs
  03_verifier_feedback.rs
  04_audit_record.rs
  05_alignment_workflow.rs
  06_public_release.rs
```

## Learning Ladder

1. `01_instruction_example` records a supervised instruction-response example.
2. `02_preference_signal` compares chosen and rejected responses with reward scores.
3. `03_verifier_feedback` keeps a failed reasoning verifier result visible.
4. `04_audit_record` wraps a post-training signal in an auditable record.
5. `05_alignment_workflow` moves an audit record through collect, audit, ready, and applied stages.
6. `06_public_release` checks that only public, audit-complete workflows reach learner-facing material.

## Category Lens

Read the crate as maps between post-training signal objects:

```text
Instruction + Response -> PromptedResponse
ChosenResponse + RejectedResponse -> PreferencePair
PreferencePair + PreferenceRewards -> PreferenceSignal
PreferenceSignal | VerifierFeedback -> UpdateSignal -> AuditRecord
AuditRecord -> AlignmentWorkflow -> AlignmentTransition
AlignmentWorkflow + AlignmentVisibility -> PublicAlignmentRelease
```

The composition rule is provenance. A learning signal is not complete until it
keeps its source, role, and audit note attached.

Reward arithmetic is intentionally written as typed operations:

```rust
RewardScore - RewardScore -> Result<RewardMargin, AlignmentError>
PreferencePair + PreferenceRewards -> Result<PreferenceSignal, AlignmentError>
```

Workflow movement is also typed:

```rust
AlignmentWorkflow -> Result<AlignmentWorkflow, AlignmentError>
```

The workflow rejects illegal orderings, such as applying an update before audit
approval.

## Three Lenses

**Rust syntax:** `ReviewedAlignmentWorkflow` wraps an `AlignmentWorkflow` with an
`AlignmentVisibility` enum before `PublicAlignmentRelease::from_reviewed_workflow`
can run.

**ML concept:** post-training traces may contain sensitive examples or restricted
review evidence. A public teaching artifact needs an explicit release gate, not
only a finished training update.

**Category-theory concept:** this is another typed map in the chain. The output
object `PublicAlignmentRelease` exists only when the input object has both the
right publication class and the completed audit lifecycle.

## Run

```bash
cargo test --manifest-path code/Cargo.toml -p rust_ml_alignment --all-targets
```

```bash
cargo run --manifest-path code/Cargo.toml -p rust_ml_alignment --example 01_instruction_example
cargo run --manifest-path code/Cargo.toml -p rust_ml_alignment --example 02_preference_signal
cargo run --manifest-path code/Cargo.toml -p rust_ml_alignment --example 03_verifier_feedback
cargo run --manifest-path code/Cargo.toml -p rust_ml_alignment --example 04_audit_record
cargo run --manifest-path code/Cargo.toml -p rust_ml_alignment --example 05_alignment_workflow
cargo run --manifest-path code/Cargo.toml -p rust_ml_alignment --example 06_public_release
```

## Scope

This crate is not a production alignment system.

The goal is to make signal provenance visible: every instruction, preference,
reward, verifier result, update signal, and workflow transition keeps its role
and source. Public examples add one more rule: restricted or private workflows
cannot become learner-facing release artifacts.
