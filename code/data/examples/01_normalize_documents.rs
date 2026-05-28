use rust_ml_data::{DocumentId, NormalizationRules, RawDocument, RawText, SourceName};

fn main() -> Result<(), rust_ml_data::Error> {
    let document = RawDocument::new(
        DocumentId::try_from("doc-001")?,
        SourceName::try_from("public-notes")?,
        RawText::try_from("  Rust   ML\nMaps   Are   Composable  ")?,
    );
    let normalized = NormalizationRules.normalize(&document)?;

    println!("document = {}", normalized.id());
    println!("source   = {}", normalized.source());
    println!("text     = {}", normalized.text());
    println!("tokens   = {}", normalized.token_count());
    println!("dedup    = {}", normalized.dedup_key());

    Ok(())
}
