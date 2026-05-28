use rust_ml_data::{
    DatasetCard, DatasetVisibility, DocumentCount, LicenseName, ManifestName, PublicCorpusManifest,
    SourceName, TokenBudget,
};

fn public_card() -> Result<DatasetCard, rust_ml_data::Error> {
    Ok(DatasetCard::new(
        SourceName::try_from("synthetic-public-notes")?,
        LicenseName::try_from("CC0-synthetic")?,
        DatasetVisibility::Public,
        DocumentCount::try_from(2)?,
        TokenBudget::try_from(128)?,
    ))
}

fn restricted_card() -> Result<DatasetCard, rust_ml_data::Error> {
    Ok(DatasetCard::new(
        SourceName::try_from("classroom-only-notes")?,
        LicenseName::try_from("not-for-public-release")?,
        DatasetVisibility::ResearchRestricted,
        DocumentCount::try_from(1)?,
        TokenBudget::try_from(64)?,
    ))
}

fn main() -> Result<(), rust_ml_data::Error> {
    let mut manifest = PublicCorpusManifest::from_card(
        ManifestName::try_from("learner-face-v1")?,
        public_card()?,
    )?;

    println!("manifest: {}", manifest.name());
    println!(
        "public totals: {} documents, {} tokens",
        manifest.totals().documents(),
        manifest.totals().tokens()
    );

    let restricted = restricted_card()?;
    println!(
        "candidate {} is {}",
        restricted.source(),
        restricted.visibility()
    );

    match manifest.add_card(restricted) {
        Ok(()) => println!("restricted source entered the public manifest"),
        Err(error) => println!("blocked from public manifest: {}", error),
    }

    println!(
        "final public totals: {} documents, {} tokens",
        manifest.totals().documents(),
        manifest.totals().tokens()
    );

    Ok(())
}
