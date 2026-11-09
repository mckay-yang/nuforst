use chrono::NaiveDateTime;

use crate::error::NufrostError;

/// Supported ISO‑8601 / Sentinel‑2 timestamp format strings.
///
/// Formats are tried in order until one succeeds.  All formats are treated as
/// **UTC** (matching the Python `pd.to_datetime(ts, utc=True)` / pandas path).
const PARSE_FORMATS: &[&str] = &[
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y%m%dT%H%M%S",
    "%Y%m%d",
];

/// Scan `desc` for the first run of 8 digits, 'T', and 6 digits,
/// returning that substring. Matches Python `_parse_band_timestamp()` in
/// `data_loader.py` which uses `re.search(r'(\d{8}T\d{6})', name)`.
pub fn find_timestamp_substring(desc: &str) -> Option<&str> {
    let bytes = desc.as_bytes();
    let len = bytes.len();
    let mut i = 0;
    while i + 14 < len {
        // Check for 8 digits
        if bytes[i..i + 8].iter().all(u8::is_ascii_digit)
            && bytes[i + 8] == b'T'
            && bytes[i + 9..i + 15].iter().all(u8::is_ascii_digit)
        {
            return Some(&desc[i..i + 15]);
        }
        i += 1;
    }
    None
}

/// Parse a timestamp string into **seconds since the Unix epoch**.
///
/// All formats are interpreted in **UTC**, matching the Python precedence of
/// `pandas.to_datetime(ts, utc=True)` → `.timestamp()`.
///
/// Returns `None` if none of the supported formats match.
pub fn parse_iso8601_to_epoch_seconds(ts: &str) -> Option<f64> {
    let ts = ts.trim();
    if ts.is_empty() {
        return None;
    }
    for fmt in PARSE_FORMATS {
        if let Ok(dt) = NaiveDateTime::parse_from_str(ts, fmt) {
            return Some(dt.and_utc().timestamp() as f64);
        }
    }
    None
}

/// Convert a slice of epoch seconds into *relative seconds* since the earliest
/// observation.
///
/// This matches the Python `_to_seconds_since_start()` helper:
///
/// ```python
/// t0 = np.min(ts_utc)
/// return ts_utc - t0
/// ```
pub fn to_seconds_since_start(epoch_seconds: &[f64]) -> Vec<f64> {
    if epoch_seconds.is_empty() {
        return Vec::new();
    }
    let t0 = epoch_seconds.iter().copied().fold(f64::INFINITY, f64::min);
    epoch_seconds.iter().map(|t| t - t0).collect()
}

/// Parse an array of timestamp strings, returning the **epoch seconds** for
/// each entry that could be parsed, or an error if any entry fails.
///
/// This is the Rust equivalent of Python's `timestamps_to_seconds()`.
pub fn parse_timestamps_to_epoch_seconds(timestamps: &[&str]) -> Result<Vec<f64>, NufrostError> {
    let mut out = Vec::with_capacity(timestamps.len());
    for ts in timestamps {
        let epoch = parse_iso8601_to_epoch_seconds(ts)
            .ok_or_else(|| NufrostError::InvalidTimestamp(ts.to_string()))?;
        out.push(epoch);
    }
    Ok(out)
}

/// Combine parsing and relative conversion in one step.
///
/// Returns (epoch_seconds, relative_days) where relative_days is
/// `(epoch_seconds - min(epoch_seconds)) / 86 400.0`.
pub fn parse_to_relative_days(timestamps: &[&str]) -> Result<(Vec<f64>, Vec<f64>), NufrostError> {
    let epoch = parse_timestamps_to_epoch_seconds(timestamps)?;
    if epoch.is_empty() {
        return Ok((vec![], vec![]));
    }
    let t0 = epoch.iter().copied().fold(f64::INFINITY, f64::min);
    let relative_days: Vec<f64> = epoch.iter().map(|t| (t - t0) / 86_400.0).collect();
    Ok((epoch, relative_days))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_sentinel2_format() {
        let ts = "20171221T035139";
        let epoch = parse_iso8601_to_epoch_seconds(ts).expect("should parse");
        // 2017-12-21T03:51:39 UTC in epoch seconds.
        // Python: datetime(2017,12,21,3,51,39).timestamp() = 1513828299.0
        // (exact depends on timezone handling, but for UTC it's well-known)
        assert!((epoch - 1_513_828_299.0).abs() < 1.0);
    }

    #[test]
    fn parse_iso8601_dashed() {
        let ts = "2026-01-01T00:00:00";
        let epoch = parse_iso8601_to_epoch_seconds(ts).expect("should parse");
        assert!((epoch - 1_767_225_600.0).abs() < 1.0);
    }

    #[test]
    fn parse_empty_returns_none() {
        assert!(parse_iso8601_to_epoch_seconds("").is_none());
    }

    #[test]
    fn parse_invalid_returns_none() {
        assert!(parse_iso8601_to_epoch_seconds("not-a-date").is_none());
    }

    #[test]
    fn relative_seconds_empty() {
        assert!(to_seconds_since_start(&[]).is_empty());
    }

    #[test]
    fn relative_seconds_identity() {
        let epoch = &[100.0, 200.0, 150.0];
        let rel = to_seconds_since_start(epoch);
        assert!((rel[0] - 0.0).abs() < 1e-10);
        assert!((rel[1] - 100.0).abs() < 1e-10);
        assert!((rel[2] - 50.0).abs() < 1e-10);
    }

    #[test]
    fn parse_to_relative_days_basic() {
        let timestamps = &["20171221T035139", "20180105T035139"];
        let (epoch, rel_days) = parse_to_relative_days(timestamps).unwrap();
        assert!(rel_days[0].abs() < 1e-10);
        // 15 days difference (Dec 21 → Jan 5, same-second)
        assert!((rel_days[1] - 15.0).abs() < 1e-10);
        // Verify epoch values are increasing → parsed correctly
        assert!(epoch[0] < epoch[1]);
    }
}
