use std::sync::Arc;

use cudarc::{
    cublas::{CudaBlas, Gemm, GemmConfig},
    driver::{result, sys, CudaDevice, CudaFunction, CudaSlice, LaunchAsync, LaunchConfig},
    nvrtc::Ptx,
};

pub struct GpuBurn<T>
where
    T: num_traits::Float + Unpin + cudarc::driver::DeviceRepr + rand::distr::uniform::SampleUniform,
{
    pub device: Arc<CudaDevice>,
    pub memory_bytes: usize,
    pub use_tensor_cores: bool,
    pub run_seconds: usize,

    d_a: CudaSlice<T>,
    d_b: CudaSlice<T>,
    d_c: CudaSlice<T>,
    d_faultyelems: CudaSlice<i32>,
    iters: usize,
    matrix_size: usize,
    compare_fn: CudaFunction,
}

const GRID_SIZE: u32 = 8192;
const BLOCK_SIZE: u32 = 16;

impl<T> GpuBurn<T>
where
    T: num_traits::Float + Unpin + cudarc::driver::DeviceRepr + rand::distr::uniform::SampleUniform,
    CudaBlas: Gemm<T>,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: Arc<CudaDevice>,
        memory_bytes: usize,
        use_tensor_cores: bool,
        run_seconds: usize,
        a: Vec<T>,
        b: Vec<T>,
        matrix_size: usize,
        kernel_file: &str,
    ) -> anyhow::Result<Self> {
        // Put A and B on device.
        let elem_size = size_of::<T>();
        let d_a: CudaSlice<T> = device.htod_copy(a)?;
        let d_b: CudaSlice<T> = device.htod_copy(b)?;
        let result_size = elem_size * matrix_size * matrix_size;
        let d_c: CudaSlice<T> = unsafe { device.alloc(result_size) }?;

        let d_faultyelems: CudaSlice<i32> = device.alloc_zeros(1)?;
        let iters: usize = (memory_bytes - 2 * result_size) / result_size;

        device.load_ptx(
            Ptx::from_src(kernel_file),
            "Compare",
            &["compare", "compareD"],
        )?;

        let compare_fn = {
            if size_of::<T>() == 4 {
                device.get_func("Compare", "compare").unwrap()
            } else {
                device.get_func("Compare", "compareD").unwrap()
            }
        };

        Ok(Self {
            device,
            memory_bytes,
            use_tensor_cores,
            run_seconds,
            d_a,
            d_b,
            d_c,
            d_faultyelems,
            iters,
            matrix_size,
            compare_fn,
        })
    }

    fn avail_memory(&self) -> anyhow::Result<usize> {
        self.device.bind_to_thread()?;
        let (free, _total) = result::mem_get_info()?;
        Ok(free)
    }

    fn total_memory(&self) -> anyhow::Result<usize> {
        self.device.bind_to_thread()?;
        let (_free, total) = result::mem_get_info()?;
        Ok(total)
    }

    pub fn burn(&mut self) -> anyhow::Result<()> {
        self.device.bind_to_thread()?;
        self.compute()?;
        self.compare()?;
        self.device.synchronize()?;
        Ok(())
    }

    fn compute(&mut self) -> anyhow::Result<()> {
        let blas = CudaBlas::new(self.device.clone())?;
        unsafe {
            Gemm::<T>::gemm(
                &blas,
                GemmConfig {
                    transa: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
                    transb: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
                    m: self.matrix_size as i32,
                    n: self.matrix_size as i32,
                    k: self.matrix_size as i32,
                    alpha: T::one(),
                    beta: T::zero(),
                    lda: self.matrix_size as i32,
                    ldb: self.matrix_size as i32,
                    ldc: self.matrix_size as i32,
                },
                &self.d_a,
                &self.d_b,
                &mut self.d_c,
            )?
        }
        self.device.synchronize()?;
        Ok(())
    }

    fn compare(&mut self) -> anyhow::Result<i32> {
        self.device.bind_to_thread()?;
        let compare_fn = self.compare_fn.clone();
        compare_fn.set_function_cache_config(sys::CUfunc_cache::CU_FUNC_CACHE_PREFER_L1)?;
        self.device.memset_zeros(&mut self.d_faultyelems)?;

        let d_faultyelems = self.device.alloc_zeros::<i32>(1)?;
        unsafe {
            compare_fn.launch(
                LaunchConfig {
                    grid_dim: (GRID_SIZE / BLOCK_SIZE, GRID_SIZE / BLOCK_SIZE, 1),
                    block_dim: (BLOCK_SIZE, BLOCK_SIZE, 1),
                    shared_mem_bytes: 0,
                },
                (&self.d_c, &d_faultyelems, self.iters),
            )?
        };
        if let Err(e) = self.device.synchronize() {
            println!("ERROR: {}", e);
            return Err(anyhow::anyhow!("Error: {}", e));
        }
        let mut h_faultyelems = [0i32];
        if let Err(e) = self
            .device
            .dtoh_sync_copy_into(&d_faultyelems, &mut h_faultyelems)
        {
            println!("ERROR: {}", e);
            return Err(anyhow::anyhow!("Error: {}", e));
        }
        Ok(h_faultyelems[0])
    }
}

impl<T> Drop for GpuBurn<T>
where
    T: num_traits::Float + Unpin + cudarc::driver::DeviceRepr + rand::distr::uniform::SampleUniform,
{
    fn drop(&mut self) {
        // Don't unwrap or expect here - just log any errors
        if let Err(e) = self
            .device
            .bind_to_thread()
            .and_then(|_| self.device.synchronize())
        {
            eprintln!("Warning: Error synchronizing device during cleanup: {}", e);
        }
    }
}
