use rust_ml_kernels::{KernelScalar, KernelVector, RowReductionTrace};

fn main() -> Result<(), rust_ml_kernels::Error> {
    let row = KernelVector::from_values([
        KernelScalar::try_from(1.0)?,
        KernelScalar::try_from(2.0)?,
        KernelScalar::try_from(3.5)?,
    ])?;
    let trace = RowReductionTrace::sum(row)?;

    println!("row sum = {}", trace.output());
    println!("elements = {}", trace.element_count());
    println!("flops = {}", trace.flops());
    Ok(())
}
