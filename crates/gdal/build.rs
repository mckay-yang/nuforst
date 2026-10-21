use std::path::{Path, PathBuf};

fn has_gdal_library(lib_dir: &Path) -> bool {
    lib_dir.join("libgdal.dylib").is_file() || lib_dir.join("libgdal.so").is_file()
}

fn main() {
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let lib_dir = manifest_dir.join("../../.conda/gdal/lib");
    println!(
        "cargo:rerun-if-changed={}",
        lib_dir.join("libgdal.dylib").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        lib_dir.join("libgdal.so").display()
    );
    let Ok(lib_dir) = lib_dir.canonicalize() else {
        return;
    };
    if has_gdal_library(&lib_dir) {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
    }
}
