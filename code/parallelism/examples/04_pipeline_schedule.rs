use rust_ml_parallelism::{LayerCount, MicroBatchCount, PipelineLayout, WorldSize};

fn main() -> Result<(), rust_ml_parallelism::Error> {
    let layout = PipelineLayout::new(LayerCount::try_from(12)?, WorldSize::try_from(3)?)?;
    let schedule = (layout.stage_count() + MicroBatchCount::try_from(4)?)?;

    println!("layers = {}", layout.layer_count());
    println!("pipeline stages = {}", layout.stage_count());
    println!("layers per stage = {}", layout.layers_per_rank());
    println!("forward schedule = {schedule}");
    Ok(())
}
