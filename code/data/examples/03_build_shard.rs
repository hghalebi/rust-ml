use rust_ml_data::{
    CorpusShard, DedupIndex, DocumentFilter, DocumentId, MinTokenCount, NormalizationRules,
    RawDocument, RawText, SourceName,
};

fn main() -> Result<(), rust_ml_data::Error> {
    let rules = NormalizationRules;
    let filter = DocumentFilter::new(MinTokenCount::try_from(2)?);
    let mut dedup_index = DedupIndex::new();
    let documents = [
        RawDocument::new(
            DocumentId::try_from("doc-a")?,
            SourceName::try_from("notes")?,
            RawText::try_from("typed data keeps provenance visible")?,
        ),
        RawDocument::new(
            DocumentId::try_from("doc-b")?,
            SourceName::try_from("transcripts")?,
            RawText::try_from("dedup keys protect the corpus")?,
        ),
    ];
    let decisions = documents
        .into_iter()
        .map(|document| {
            rules
                .normalize(&document)
                .map(|normalized| filter.decide(normalized, &mut dedup_index))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let shard = CorpusShard::from_decisions(decisions)?;

    println!("accepted documents = {}", shard.accepted_count()?);
    for source in shard.sources() {
        println!("source             = {}", source);
    }

    Ok(())
}
