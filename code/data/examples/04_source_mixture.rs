use rust_ml_data::{MixtureWeight, SamplingMixture, SourceName, SourceWeight};

fn main() -> Result<(), rust_ml_data::Error> {
    let mixture = SamplingMixture::from_weights([
        SourceWeight::new(
            SourceName::try_from("notes")?,
            MixtureWeight::try_from(0.70)?,
        ),
        SourceWeight::new(
            SourceName::try_from("transcripts")?,
            MixtureWeight::try_from(0.30)?,
        ),
    ])?;

    println!("total weight = {}", mixture.total_weight());
    for source_weight in mixture.source_weights() {
        println!("{} -> {}", source_weight.source(), source_weight.weight());
    }

    Ok(())
}
