use rust_ml_systems::{AttentionEstimate, ElementSize, ModelWidth, SequenceLength};

fn main() -> Result<(), rust_ml_systems::Error> {
    let estimate = AttentionEstimate::new(SequenceLength::try_from(8)?, ModelWidth::try_from(16)?);

    println!("score FLOPs      = {}", estimate.score_flops()?);
    println!("value-mix FLOPs  = {}", estimate.value_mix_flops()?);
    println!("total FLOPs      = {}", estimate.total_flops()?);
    println!(
        "score matrix size = {}",
        estimate.score_matrix_bytes(ElementSize::float32())?
    );

    Ok(())
}
