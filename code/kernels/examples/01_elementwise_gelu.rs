use rust_ml_kernels::{ElementSize, ElementwiseTrace, KernelScalar, KernelVector};

fn main() -> Result<(), rust_ml_kernels::Error> {
    let input = KernelVector::from_values([
        KernelScalar::try_from(-1.0)?,
        KernelScalar::try_from(0.0)?,
        KernelScalar::try_from(1.0)?,
    ])?;
    let trace = ElementwiseTrace::gelu(input, ElementSize::float32())?;

    println!("elements = {}", trace.element_count());
    println!("flops = {}", trace.flops());
    println!("hbm = {}", trace.hbm_bytes());
    for value in trace.output().values() {
        println!("gelu value = {value}");
    }
    Ok(())
}
