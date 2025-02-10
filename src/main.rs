#![allow(dead_code)]

mod gpu_burn;

use clap::Parser;
use cudarc::nvrtc::Ptx;
use gpu_burn::GpuBurn;

use std::sync::Arc;

use cudarc::cublas::sys::CUDA_VERSION;
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
use cudarc::driver::result::device::total_mem;
use cudarc::driver::sys::CUdevice_attribute;
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr, DevicePtrMut};

const MATRIX_SIZE: usize = 8192;
const COMPARE_KERNEL: &str = include_str!(concat!(env!("OUT_DIR"), "/compare.ptx"));

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Memory to use in MB or percentage (when % symbol is present)
    #[arg(short = 'm')]
    memory: Option<String>,

    /// Use doubles instead of singles
    #[arg(short = 'd', default_value_t = false)]
    use_doubles: bool,

    /// Try to use tensor cores
    #[arg(short = 't', long = "tc", default_value_t = false)]
    use_tensor_cores: bool,

    /// List available GPUs
    #[arg(short = 'l', default_value_t = false)]
    list_gpus: bool,

    /// GPU index to use
    #[arg(short = 'i', default_value_t = 0)]
    gpu_index: usize,

    /// Number of seconds to run the test
    #[arg(default_value_t = 10)]
    seconds: usize,
}

fn list_gpus() -> anyhow::Result<()> {
    for gpu_i in 0..CudaDevice::count()? {
        println!(
            "{gpu_i}: {}",
            CudaDevice::new(gpu_i as usize).unwrap().name()?
        );
    }
    Ok(())
}

fn print_gpu_info(gpu: &CudaDevice) -> anyhow::Result<()> {
    let name = gpu.name()?;
    let compute_major =
        gpu.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?;
    let compute_minor =
        gpu.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?;

    let total_global_mem_bytes = unsafe { total_mem(*gpu.cu_device())? };
    let total_global_mem_mb = (total_global_mem_bytes as f32 / 1048576.0f32) as usize;

    let max_clock_rate = gpu.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_CLOCK_RATE)?;
    let max_clock_rate_mhz = (max_clock_rate as f32 * 1e-3) as usize;
    let max_clock_rate_ghz = max_clock_rate as f32 * 1e-6;

    let mem_clock_rate_mhz =
        (gpu.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE)? as f32 * 1e-3)
            as usize;

    println!(
        "{name}: 
    CUDA Driver Version:                           {CUDA_VERSION}
    CUDA Capability Major/Minor version number:    {compute_major}.{compute_minor}
    Total amount of global memory:                 {total_global_mem_mb} MBytes ({total_global_mem_bytes} bytes)
    GPU Max Clock rate:                            {max_clock_rate_mhz} MHz ({max_clock_rate_ghz:.2} GHz)
    Memory Clock rate:                             {mem_clock_rate_mhz} MHz
"
    );
    Ok(())
}

fn launch<T>(
    run_length: usize,
    use_tensors: bool,
    use_bytes: usize,
    gpu: Arc<CudaDevice>,
    kernel_file: &str,
) -> anyhow::Result<()>
where
    T: num_traits::Float + Unpin + cudarc::driver::DeviceRepr + rand::distr::uniform::SampleUniform,
    CudaBlas: Gemm<T>,
{
    use rand::Rng;
    let mut rng = rand::rng();
    let h_a: Vec<T> = (0..(MATRIX_SIZE * MATRIX_SIZE))
        .map(|_| rng.random_range(T::zero()..T::one()))
        .collect();
    let h_b: Vec<T> = (0..(MATRIX_SIZE * MATRIX_SIZE))
        .map(|_| rng.random_range(T::zero()..T::one()))
        .collect();

    // TODO: Create multiple threads?

    let mut gpu_burn = GpuBurn::<T>::new(
        gpu.clone(),
        use_bytes,
        use_tensors,
        run_length,
        h_a,
        h_b,
        MATRIX_SIZE,
        kernel_file,
    )?;
    gpu_burn.burn()?;

    Ok(())
}

fn calculate_memory(mem: &Option<String>, gpu: &CudaDevice) -> anyhow::Result<usize> {
    if let Some(mem_str) = mem {
        if mem_str.ends_with('%') {
            // Handle percentage
            let percentage = mem_str
                .trim_end_matches('%')
                .parse::<f32>()
                .map_err(|_| anyhow::anyhow!("Invalid percentage value"))?;
            if !(0.0..=100.0).contains(&percentage) {
                return Err(anyhow::anyhow!("Percentage must be between 0 and 100"));
            }
            // Calculate actual memory based on percentage
            let total_mem = unsafe { total_mem(*gpu.cu_device())? };
            Ok(((total_mem as f32 * percentage) / 100.0) as usize)
        } else {
            // Handle MB value
            Ok(mem_str
                .parse::<usize>()
                .map_err(|_| anyhow::anyhow!("Invalid memory value"))?
                * 1024
                * 1024) // Convert MB to bytes
        }
    } else {
        Ok(unsafe { total_mem(*gpu.cu_device())? as usize })
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    if args.list_gpus {
        println!("Listing GPUs:");
        return list_gpus();
    }
    let gpu_count = CudaDevice::count()? as usize;
    if args.gpu_index > gpu_count {
        return Err(anyhow::anyhow!(
            "Invalid GPU index range: {}",
            args.gpu_index
        ));
    }

    let gpu = CudaDevice::new_with_stream(args.gpu_index)?;

    let gpu_bytes = calculate_memory(&args.memory, &gpu)?;

    println!("Selected {}: {}", args.gpu_index, gpu.name()?);
    println!("Burning for {} seconds", args.seconds);

    if args.use_doubles {
        println!("Using f64");
        launch::<f64>(
            args.seconds,
            args.use_tensor_cores,
            gpu_bytes,
            gpu,
            COMPARE_KERNEL,
        )?;
    } else {
        println!("Using f32");
        launch::<f32>(
            args.seconds,
            args.use_tensor_cores,
            gpu_bytes,
            gpu,
            COMPARE_KERNEL,
        )?;
    }

    Ok(())
}
