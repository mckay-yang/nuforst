use anyhow::{Context, Result};
use std::path::PathBuf;

pub fn data_root() -> Result<PathBuf> {
    std::env::var_os("NUFROST_DATA_ROOT")
        .map(PathBuf::from)
        .context("set NUFROST_DATA_ROOT to the directory containing cache/, products/, and tests/")
}

pub const CONFIG: &[u8] = include_bytes!("../../config/nufrost.json");
