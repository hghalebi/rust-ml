use rust_ml_kernels::{
    ElementSize, KernelMatrix, KernelScalar, KernelVector, TileColumns, TileRows, TileShape,
    TiledMatVecTrace,
};

fn main() -> Result<(), rust_ml_kernels::Error> {
    let matrix = KernelMatrix::from_rows([
        KernelVector::from_values([
            KernelScalar::try_from(1.0)?,
            KernelScalar::try_from(2.0)?,
            KernelScalar::try_from(3.0)?,
        ])?,
        KernelVector::from_values([
            KernelScalar::try_from(4.0)?,
            KernelScalar::try_from(5.0)?,
            KernelScalar::try_from(6.0)?,
        ])?,
    ])?;
    let vector = KernelVector::from_values([
        KernelScalar::try_from(1.0)?,
        KernelScalar::try_from(0.5)?,
        KernelScalar::try_from(2.0)?,
    ])?;
    let trace = TiledMatVecTrace::run(
        matrix,
        vector,
        TileShape::new(TileRows::try_from(1)?, TileColumns::try_from(2)?),
        ElementSize::float32(),
    )?;

    println!("tiles = {}", trace.tile_plan().windows().count());
    for value in trace.output().values() {
        println!("output value = {value}");
    }
    Ok(())
}
