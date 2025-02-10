use std::env;
use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;

fn compile_cuda_dir(kernel_dir: &Path, include_dir: &Path, out_dir: &Path) -> anyhow::Result<()> {
    let arch = "compute_86"; // For example, using SM 8.6 (Ampere architecture).
    let code = "sm_86"; // For the same SM 8.6 (Ampere architecture).

    for entry in fs::read_dir(kernel_dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.is_file() {
            let extension = path.extension().unwrap().to_str().unwrap();
            match extension {
                "cu" => {
                    println!("cargo::rerun-if-changed={}", &path.display());
                    let kernel_name = path.file_stem().unwrap().to_str().unwrap();
                    let ptx_file = out_dir.join(format!("{}.ptx", kernel_name));
                    let nvcc_status = Command::new("nvcc")
                        .arg("-ptx")
                        .arg("-o")
                        .arg(&ptx_file)
                        .arg(&path)
                        .arg(format!("-arch={}", arch))
                        .arg(format!("-code={}", code))
                        .arg("-I")
                        .arg(include_dir)
                        .output()
                        .unwrap();
                    if !nvcc_status.status.success() {
                        panic!(
                            "nvcc failes: {}\n{}",
                            path.display(),
                            String::from_utf8_lossy(&nvcc_status.stderr)
                        );
                    }
                }
                _ => continue,
            }
        }
    }
    Ok(())
}

fn main() -> anyhow::Result<()> {
    println!("cargo::rerun-if-changed=cuda");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    compile_cuda_dir(
        Path::new("src/cuda/kernels"),
        Path::new("src/cuda/includes"),
        out_dir.as_path(),
    )?;

    Ok(())
}
