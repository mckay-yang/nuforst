use serde::{Deserialize, Serialize};

/// Reconstruction algorithm identifier.
///
/// Mirrors the Python convention: `"nufrost"`, `"hants"`, `"zhu2015"`.
/// Serde renames keep the wire-format (JSON) in snake_case.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Algorithm {
    #[serde(rename = "nufrost")]
    Nufrost,
    #[serde(rename = "hants")]
    Hants,
    #[serde(rename = "zhu2015")]
    Zhu2015,
}

impl Algorithm {
    /// Parse from a lowercase string (e.g. from config or CLI).
    pub fn from_str(s: &str) -> Option<Self> {
        match s.trim().to_lowercase().as_str() {
            "nufrost" => Some(Self::Nufrost),
            "hants" => Some(Self::Hants),
            "zhu2015" => Some(Self::Zhu2015),
            _ => None,
        }
    }
}

/// Input time-series for a single pixel.
///
/// All arrays must have the same length.  The `valid_mask` flags which
/// observations should be used by the reconstruction algorithm (e.g. finite,
/// within the valid reflectance range).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeries {
    /// Days since the first observation (t0 = 0.0).
    pub timestamps_days: Vec<f64>,
    /// Raw observation values (may contain NaN / invalid entries).
    pub observations: Vec<f64>,
    /// `true` for each valid observation.
    pub valid_mask: Vec<bool>,
    /// The target time for prediction, in the same day-relative unit.
    pub target_time_day: f64,
}

// ── ndarray type aliases ─────────────────────────────────────────────────
// Convenience aliases for common array shapes used across algorithms.

/// One-dimensional array of f64 (time-series observations, timestamps).
pub type Array1D = ndarray::Array1<f64>;

/// Two-dimensional array of f64 (spatial prediction grids, raster bands).
pub type Array2D = ndarray::Array2<f64>;

/// Three-dimensional array of f64 (full time × rows × cols cubes).
pub type Array3D = ndarray::Array3<f64>;

/// Mask array (1-D boolean).
pub type Mask1D = ndarray::Array1<bool>;

// ── Metadata ──────────────────────────────────────────────────────────────

/// Minimal raster band metadata needed for I/O and projection.
///
/// The Python equivalent lives in `data_loader.py` (band descriptions,
/// `rasterio` dataset properties).  This struct is a snapshot of the fields
/// the Rust layer needs to reason about raster shape and geo-referencing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BandMetadata {
    /// Band description string (often contains the Sentinel-2 timestamp).
    pub name: String,
    /// Raster width in pixels.
    pub width: u32,
    /// Raster height in pixels.
    pub height: u32,
    /// GDAL data-type name, e.g. `"UInt16"`, `"Float32"`.
    pub dtype: String,
    /// No-data value, if defined by the source raster.
    pub no_data: Option<f64>,
    /// Authority:code string, e.g. `"EPSG:32610"`.
    pub crs: String,
    /// Affine geotransform: `[x_origin, pixel_width, 0, y_origin, 0, -pixel_height]`.
    pub transform: [f64; 6],
}
