use rust_ml_systems::{Bytes, BytesPerSecond, MemoryLevel, MemoryTransfer};

fn main() -> Result<(), rust_ml_systems::Error> {
    let activation_tile = Bytes::try_from(16_384_u64)?;

    let shared_memory = MemoryTransfer::new(
        MemoryLevel::SharedMemory,
        activation_tile,
        BytesPerSecond::try_from(8_000_000_000_000_u128)?,
    );
    let host_memory = MemoryTransfer::new(
        MemoryLevel::HostMemory,
        activation_tile,
        BytesPerSecond::try_from(32_000_000_000_u128)?,
    );

    println!(
        "{} transfer: {} at {} -> {}",
        shared_memory.level(),
        shared_memory.bytes_moved(),
        shared_memory.bandwidth(),
        shared_memory.estimated_elapsed()?
    );
    println!(
        "{} transfer: {} at {} -> {}",
        host_memory.level(),
        host_memory.bytes_moved(),
        host_memory.bandwidth(),
        host_memory.estimated_elapsed()?
    );

    Ok(())
}
