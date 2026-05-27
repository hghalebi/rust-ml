use rust_ml_kernels::{
    ElementSize, KernelMatrix, KernelScalar, KernelTraceVisibility, KernelVector,
    PublicKernelReport, ReviewedTiledMatVecTrace, TileColumns, TileRows, TileShape,
    TiledMatVecTrace,
};

fn vector(
    values: impl IntoIterator<Item = KernelScalar>,
) -> Result<KernelVector, rust_ml_kernels::Error> {
    KernelVector::from_values(values)
}

fn matrix() -> Result<KernelMatrix, rust_ml_kernels::Error> {
    KernelMatrix::from_rows([
        vector([
            KernelScalar::try_from(1.0)?,
            KernelScalar::try_from(2.0)?,
            KernelScalar::try_from(3.0)?,
        ])?,
        vector([
            KernelScalar::try_from(4.0)?,
            KernelScalar::try_from(5.0)?,
            KernelScalar::try_from(6.0)?,
        ])?,
    ])
}

fn tiled_trace() -> Result<TiledMatVecTrace, rust_ml_kernels::Error> {
    TiledMatVecTrace::run(
        matrix()?,
        vector([
            KernelScalar::try_from(1.0)?,
            KernelScalar::try_from(0.5)?,
            KernelScalar::try_from(2.0)?,
        ])?,
        TileShape::new(TileRows::try_from(1)?, TileColumns::try_from(2)?),
        ElementSize::float32(),
    )
}

fn main() -> Result<(), rust_ml_kernels::Error> {
    let report = PublicKernelReport::from_reviewed_trace(ReviewedTiledMatVecTrace::new(
        tiled_trace()?,
        KernelTraceVisibility::Public,
    ))?;

    println!(
        "public tile windows = {}",
        report.tile_plan().windows().count()
    );
    println!("public FLOPs = {}", report.flops());
    println!("public HBM bytes = {}", report.hbm_bytes());

    let private = PublicKernelReport::from_reviewed_trace(ReviewedTiledMatVecTrace::new(
        tiled_trace()?,
        KernelTraceVisibility::Private,
    ));

    if let Err(error) = private {
        println!("blocked from public kernel report: {error}");
    }

    Ok(())
}
