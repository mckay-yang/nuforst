//! Fixed-input predictions for the vector NUFROST model under `config/nufrost.json`.
//!
//! Refactors of the fitter must keep these frequencies and reflectance values.

use nufrost_core::{nufrost_fit_pixel_vector, nufrost_predict_vector, NufrostConfig};

const PAPER_CONFIG: &str = include_str!("../../../config/nufrost.json");

fn seasonal_observations() -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = 48;
    let times: Vec<f64> = (0..n).map(|i| i as f64 * 8.0).collect();
    let amplitudes = [180.0, 260.0, 310.0, 900.0, 420.0, 280.0];
    let baselines = [900.0, 1100.0, 1400.0, 2800.0, 1600.0, 1200.0];
    let mut observations = Vec::with_capacity(amplitudes.len());
    for (band, (&amplitude, baseline)) in amplitudes.iter().zip(baselines).enumerate() {
        let series = times
            .iter()
            .enumerate()
            .map(|(i, &day)| {    
                if i == 10 {
                    return f64::NAN;
                }
                let annual = (2.0 * std::f64::consts::PI * day / 365.25).sin();
                let trend = 0.15 * day * (band as f64 + 1.0);
                baseline + amplitude * annual + trend
            })
            .collect();
        observations.push(series);
    }
    (times, observations)
}

#[test]
fn paper_config_vector_prediction_matches_recorded_values() {
    let config = NufrostConfig::from_json(PAPER_CONFIG.as_bytes()).unwrap();
    let (times, observations) = seasonal_observations();
    let fit = nufrost_fit_pixel_vector(&times, &observations, &config);
    let target_day = 180.0;
    let predicted = nufrost_predict_vector(&fit, target_day);

    assert!(fit.valid);
    assert_eq!(fit.freqs.len(), EXPECTED_FREQS.len());
    for (got, expected) in fit.freqs.iter().zip(EXPECTED_FREQS) {
        let scale = expected.abs().max(1.0e-12);
        assert!(
            (got - expected).abs() / scale < 1.0e-8,
            "freq {got:.12e} != {expected:.12e}"
        );
    }
    assert_eq!(predicted.len(), EXPECTED_PREDICTIONS.len());
    for (got, expected) in predicted.iter().zip(EXPECTED_PREDICTIONS) {
        assert!(
            (got - expected).abs() < 1.0e-6,
            "prediction {got:.10} != {expected:.10}"
        );
    }
}

const EXPECTED_FREQS: [f64; 23] = [
    3.078211189913e-8,
    6.156422379827e-8,
    9.234633569740e-8,
    1.231284475965e-7,
    1.539105594957e-7,
    1.846926713948e-7,
    2.154747832939e-7,
    2.462568951931e-7,
    2.770390070922e-7,
    3.078211189913e-7,
    3.386032308905e-7,
    3.693853427896e-7,
    4.001674546887e-7,
    4.309495665879e-7,
    4.617316784870e-7,
    4.925137903861e-7,
    5.232959022853e-7,
    5.540780141844e-7,
    5.848601260835e-7,
    6.156422379827e-7,
    6.464243498818e-7,
    6.772064617809e-7,
    7.079885736801e-7,
];

const EXPECTED_PREDICTIONS: [f64; 6] = [
    933.4947603023,
    1164.0684867306,
    1493.0929603918,
    2946.7385469279,
    1752.0711966394,
    1372.7850864228,
];
