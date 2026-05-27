use rust_ml_parallelism::{
    CollectiveTrace, CommunicationBytes, ParallelStrategy, TensorElement, TensorLine, WorldSize,
    shard_tensor_line,
};

fn rank_values() -> Result<TensorLine, rust_ml_parallelism::Error> {
    TensorLine::from_values([
        TensorElement::try_from(1.0)?,
        TensorElement::try_from(2.0)?,
        TensorElement::try_from(3.0)?,
        TensorElement::try_from(4.0)?,
    ])
}

fn main() -> Result<(), rust_ml_parallelism::Error> {
    let plan = shard_tensor_line(
        rank_values()?,
        WorldSize::try_from(2)?,
        ParallelStrategy::DataParallel,
    )?;
    let trace = CollectiveTrace::all_reduce(plan, CommunicationBytes::try_from(64)?)?;

    println!("collective = {}", trace.kind());
    println!("reduced value = {}", trace.reduced_value());
    println!("communication = {}", trace.communication());
    Ok(())
}
