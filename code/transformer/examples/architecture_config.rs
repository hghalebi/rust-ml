use rust_ml_transformer::{
    FeedForwardWidth, HeadCount, LayerCount, ModelError, TransformerConfig, VectorLength,
};

fn main() -> Result<(), ModelError> {
    let config = TransformerConfig::new(
        VectorLength::try_from(8)?,
        LayerCount::try_from(2)?,
        HeadCount::try_from(2)?,
        FeedForwardWidth::try_from(32)?,
    )?;

    println!("model width: {}", config.model_width());
    println!("layers: {}", config.layer_count());
    println!("heads: {}", config.head_count());
    println!("head width: {}", config.attention_head_width()?);
    println!(
        "encoder parameters without embeddings: {}",
        config.encoder_parameter_estimate()?
    );

    Ok(())
}
