use rust_ml_parallelism::{ModelWidth, TensorParallelLayout, WorldSize};

fn main() -> Result<(), rust_ml_parallelism::Error> {
    let layout = TensorParallelLayout::new(ModelWidth::try_from(1024)?, WorldSize::try_from(4)?)?;

    println!("model width = {}", layout.model_width());
    println!("world size = {}", layout.world_size());
    println!("width per rank = {}", layout.shard_width());
    Ok(())
}
