use rust_ml_parallelism::{
    DataParallelLayout, GlobalBatchSize, ParallelStrategy, TensorElement, TensorLine, WorldSize,
    shard_tensor_line,
};

fn batch_losses() -> Result<TensorLine, rust_ml_parallelism::Error> {
    TensorLine::from_values([
        TensorElement::try_from(0.9)?,
        TensorElement::try_from(0.7)?,
        TensorElement::try_from(0.5)?,
        TensorElement::try_from(0.3)?,
    ])
}

fn main() -> Result<(), rust_ml_parallelism::Error> {
    let world = WorldSize::try_from(2)?;
    let layout = DataParallelLayout::new(GlobalBatchSize::try_from(4)?, world)?;
    let plan = shard_tensor_line(batch_losses()?, world, ParallelStrategy::DataParallel)?;

    println!("global batch = {}", layout.global_batch());
    println!("local batch = {}", layout.local_batch());
    for shard in plan.shards() {
        println!(
            "{} owns {} values from offset {}",
            shard.rank(),
            shard.values().length(),
            shard.start()
        );
    }
    Ok(())
}
