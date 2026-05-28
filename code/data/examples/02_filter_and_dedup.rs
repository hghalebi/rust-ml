use rust_ml_data::{
    DedupIndex, DocumentFilter, DocumentId, FilterDecision, MinTokenCount, NormalizationRules,
    RawDocument, RawText, SourceName,
};

fn main() -> Result<(), rust_ml_data::Error> {
    let rules = NormalizationRules;
    let filter = DocumentFilter::new(MinTokenCount::try_from(2)?);
    let mut dedup_index = DedupIndex::new();
    let documents = [
        RawDocument::new(
            DocumentId::try_from("doc-a")?,
            SourceName::try_from("public-notes")?,
            RawText::try_from("Category theory names composable maps.")?,
        ),
        RawDocument::new(
            DocumentId::try_from("doc-b")?,
            SourceName::try_from("public-notes")?,
            RawText::try_from("category theory names composable maps.")?,
        ),
        RawDocument::new(
            DocumentId::try_from("doc-c")?,
            SourceName::try_from("public-notes")?,
            RawText::try_from("tiny")?,
        ),
    ];

    for document in documents {
        let normalized = rules.normalize(&document)?;
        match filter.decide(normalized, &mut dedup_index) {
            FilterDecision::Accepted(accepted) => {
                println!(
                    "accepted {} from {}",
                    accepted.document().id(),
                    accepted.document().source()
                );
            }
            FilterDecision::Rejected(rejected) => {
                println!(
                    "rejected {}: {}",
                    rejected.document().id(),
                    rejected.reason()
                );
            }
        }
    }

    Ok(())
}
