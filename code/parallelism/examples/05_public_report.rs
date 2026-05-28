use rust_ml_parallelism::{
    CollectiveTrace, CommunicationBytes, ParallelStrategy, ParallelTraceVisibility,
    PublicParallelismReport, ReviewedCollectiveTrace, TensorElement, TensorLine, WorldSize,
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

fn collective_trace() -> Result<CollectiveTrace, rust_ml_parallelism::Error> {
    let plan = shard_tensor_line(
        rank_values()?,
        WorldSize::try_from(2)?,
        ParallelStrategy::DataParallel,
    )?;

    CollectiveTrace::all_reduce(plan, CommunicationBytes::try_from(64)?)
}

fn main() -> Result<(), rust_ml_parallelism::Error> {
    let report = PublicParallelismReport::from_reviewed_trace(ReviewedCollectiveTrace::new(
        collective_trace()?,
        ParallelTraceVisibility::Public,
    ))?;

    println!("public collective = {}", report.kind());
    println!("public reduced value = {}", report.reduced_value());
    println!("public communication = {}", report.communication());

    let private = PublicParallelismReport::from_reviewed_trace(ReviewedCollectiveTrace::new(
        collective_trace()?,
        ParallelTraceVisibility::Private,
    ));

    if let Err(error) = private {
        println!("blocked from public parallelism report: {error}");
    }

    Ok(())
}
