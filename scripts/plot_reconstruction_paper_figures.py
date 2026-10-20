#!/usr/bin/env python3
"""Build publication figures from the Rust sample-cache evaluation outputs.

The script expects one prediction CSV per method, all produced from the same
cache, task count, and evaluation seed. It writes clean source-data tables and
SVG/PDF/TIFF/PNG exports for the band comparison, gap-index diagnostic, and
representative reconstruction cases.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"

BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]
METHODS = ["nufrost", "zhu2015", "hants"]
METHOD_LABELS = {"nufrost": "NUFROST", "zhu2015": "Zhu2015", "hants": "HANTS"}
METHOD_COLORS = {"nufrost": "#0F4D92", "zhu2015": "#D98E2B", "hants": "#767676"}
METHOD_MARKERS = {"nufrost": "o", "zhu2015": "s", "hants": "^"}
BAND_BAR_COLORS = {
    "nufrost": "#6E9CCB",
    "zhu2015": "#DDBB69",
    "hants": "#CF7B72",
}
BAND_ERROR_COLORS = {
    "nufrost": "#3F6F9F",
    "zhu2015": "#9A792F",
    "hants": "#945049",
}
REFLECTANCE_SCALE = 10_000.0


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "axes.linewidth": 0.8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "legend.fontsize": 8,
            "legend.frameon": False,
        }
    )


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.11,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=9.5,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def load_predictions(eval_dir: Path, cache_dir: Path) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    truth_reference: pd.DataFrame | None = None
    for method in METHODS:
        path = eval_dir / f"{method}_predictions.csv"
        frame = pd.read_csv(path)
        required = {
            "sample_idx",
            "holdout_seed",
            "gap_regular",
            "gap_periodic",
            *[f"truth_{band}" for band in BANDS],
            *[f"pred_{band}" for band in BANDS],
        }
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")
        key = ["sample_idx", "holdout_seed"]
        frame = frame.sort_values(key).reset_index(drop=True)
        truths = frame[key + [f"truth_{band}" for band in BANDS]]
        if truth_reference is None:
            truth_reference = truths.copy()
        else:
            aligned = truth_reference.merge(truths, on=key, suffixes=("_a", "_b"))
            if len(aligned) != len(truth_reference):
                raise ValueError(f"{method} does not contain the same evaluation tasks")
            for band in BANDS:
                if not np.allclose(
                    aligned[f"truth_{band}_a"], aligned[f"truth_{band}_b"], equal_nan=True
                ):
                    raise ValueError(f"truth_{band} differs for {method}")
        keep = key + (["gap_regular", "gap_periodic"] if merged is None else [])
        if merged is None:
            keep += [f"truth_{band}" for band in BANDS]
        keep += [f"pred_{band}" for band in BANDS]
        frame = frame[keep].rename(
            columns={f"pred_{band}": f"{method}_{band}" for band in BANDS}
        )
        merged = frame if merged is None else merged.merge(frame, on=key, how="inner")

    assert merged is not None
    with (cache_dir / "meta.json").open("r", encoding="utf-8") as stream:
        meta = json.load(stream)
    index = np.memmap(
        cache_dir / meta["index_file"], dtype="<u8", mode="r", shape=(meta["n_samples"], 4)
    )
    sample_idx = merged["sample_idx"].to_numpy(dtype=np.int64)
    if sample_idx.min() < 0 or sample_idx.max() >= len(index):
        raise ValueError("prediction sample_idx falls outside the cache index")
    merged["scene_id"] = np.asarray(index[sample_idx, 0], dtype=np.int64)
    for band in BANDS:
        merged[f"truth_{band}"] /= REFLECTANCE_SCALE
        for method in METHODS:
            merged[f"{method}_{band}"] /= REFLECTANCE_SCALE
    return merged


def cluster_interval(
    values: np.ndarray,
    scenes: np.ndarray,
    statistic: str,
    rng: np.random.Generator,
    n_boot: int,
) -> tuple[float, float, float]:
    finite = np.isfinite(values)
    values = values[finite]
    scenes = scenes[finite]
    if values.size == 0:
        return np.nan, np.nan, np.nan
    unique = np.unique(scenes)
    sums = []
    counts = []
    for scene in unique:
        scene_values = values[scenes == scene]
        transformed = np.square(scene_values) if statistic == "rmse" else np.abs(scene_values)
        sums.append(float(np.sum(transformed)))
        counts.append(int(scene_values.size))
    sums_arr = np.asarray(sums)
    counts_arr = np.asarray(counts)
    point = sums_arr.sum() / counts_arr.sum()
    if statistic == "rmse":
        point = np.sqrt(point)
    draws = rng.integers(0, len(unique), size=(n_boot, len(unique)))
    boot = sums_arr[draws].sum(axis=1) / counts_arr[draws].sum(axis=1)
    if statistic == "rmse":
        boot = np.sqrt(boot)
    low, high = np.quantile(boot, [0.025, 0.975])
    return float(point), float(low), float(high)


def build_band_source(
    data: pd.DataFrame, n_boot: int, rng: np.random.Generator
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    scenes = data["scene_id"].to_numpy()
    for method in METHODS:
        for band in BANDS + ["Overall"]:
            if band == "Overall":
                errors = np.concatenate(
                    [
                        data[f"{method}_{item}"].to_numpy()
                        - data[f"truth_{item}"].to_numpy()
                        for item in BANDS
                    ]
                )
                error_scenes = np.tile(scenes, len(BANDS))
            else:
                errors = data[f"{method}_{band}"].to_numpy() - data[f"truth_{band}"].to_numpy()
                error_scenes = scenes
            for metric in ("rmse", "mae"):
                estimate, ci_low, ci_high = cluster_interval(
                    errors, error_scenes, metric, rng, n_boot
                )
                rows.append(
                    {
                        "method": METHOD_LABELS[method],
                        "method_key": method,
                        "band": band,
                        "metric": metric.upper(),
                        "estimate": estimate,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "samples": int(np.isfinite(errors).sum()),
                        "aois": int(np.unique(error_scenes[np.isfinite(errors)]).size),
                        "interval": "95% AOI-cluster bootstrap CI",
                    }
                )
    return pd.DataFrame(rows)


def build_gap_source(
    data: pd.DataFrame, n_boot: int, rng: np.random.Generator, n_bins: int = 8
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for gap_key, gap_label in (
        ("gap_regular", "I_gap"),
        ("gap_periodic", "I_periodic"),
    ):
        gap = data[gap_key].to_numpy()
        edges = np.unique(np.quantile(gap[np.isfinite(gap)], np.linspace(0, 1, n_bins + 1)))
        if len(edges) < 3:
            raise ValueError(f"{gap_key} has too few unique values for binning")
        edges[0] = np.nextafter(edges[0], -np.inf)
        bin_ids = np.digitize(gap, edges[1:-1], right=True)
        for bin_id in range(len(edges) - 1):
            subset = data.loc[bin_ids == bin_id]
            if subset.empty:
                continue
            x = subset[gap_key].to_numpy()
            scenes = subset["scene_id"].to_numpy()
            for method in METHODS:
                errors = np.concatenate(
                    [
                        subset[f"{method}_{band}"].to_numpy()
                        - subset[f"truth_{band}"].to_numpy()
                        for band in BANDS
                    ]
                )
                error_scenes = np.tile(scenes, len(BANDS))
                estimate, ci_low, ci_high = cluster_interval(
                    errors, error_scenes, "rmse", rng, n_boot
                )
                rows.append(
                    {
                        "gap_index": gap_label,
                        "gap_key": gap_key,
                        "bin": bin_id + 1,
                        "range_start": float(edges[bin_id]),
                        "range_end": float(edges[bin_id + 1]),
                        "x_median": float(np.median(x)),
                        "method": METHOD_LABELS[method],
                        "method_key": method,
                        "rmse": estimate,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "samples": int(len(subset)),
                        "aois": int(subset["scene_id"].nunique()),
                        "interval": "95% AOI-cluster bootstrap CI",
                    }
                )
    return pd.DataFrame(rows)


def choose_cases(data: pd.DataFrame) -> pd.DataFrame:
    work = data.copy()
    for method in METHODS:
        errors = np.column_stack(
            [work[f"{method}_{band}"] - work[f"truth_{band}"] for band in BANDS]
        )
        work[f"{method}_rmse"] = np.sqrt(np.nanmean(np.square(errors), axis=1))
    work["advantage"] = (
        work[["zhu2015_rmse", "hants_rmse"]].min(axis=1) - work["nufrost_rmse"]
    )
    gap = work["gap_regular"]
    quantiles = gap.quantile([0.2, 0.4, 0.6, 0.8]).to_dict()
    strata = [
        ("Case 1", "Low native-gap stratum", gap <= quantiles[0.2]),
        (
            "Case 2",
            "Moderate native-gap stratum",
            (gap >= quantiles[0.4]) & (gap <= quantiles[0.6]),
        ),
        ("Case 3", "High native-gap stratum", gap >= quantiles[0.8]),
    ]
    selected: list[dict[str, object]] = []
    used_scenes: set[int] = set()
    for label, support_stratum, mask in strata:
        candidates = work.loc[mask].copy()
        med_error = candidates["nufrost_rmse"].median()
        med_adv = candidates["advantage"].median()
        error_scale = max(candidates["nufrost_rmse"].mad(), 1.0e-9) if hasattr(candidates["nufrost_rmse"], "mad") else max((candidates["nufrost_rmse"] - med_error).abs().median(), 1.0e-9)
        adv_scale = max((candidates["advantage"] - med_adv).abs().median(), 1.0e-9)
        candidates["selection_score"] = (
            (candidates["nufrost_rmse"] - med_error).abs() / error_scale
            + (candidates["advantage"] - med_adv).abs() / adv_scale
        )
        candidates = candidates.sort_values(["selection_score", "sample_idx"])
        available = candidates.loc[~candidates["scene_id"].isin(used_scenes)]
        chosen = available if not available.empty else candidates
        chosen_index = chosen.index[0]
        record = {
            "case_label": label,
            "selection_stratum": support_stratum,
            "sample_idx": int(candidates.at[chosen_index, "sample_idx"]),
            "holdout_seed": int(candidates.at[chosen_index, "holdout_seed"]),
            "scene_id": int(candidates.at[chosen_index, "scene_id"]),
            "gap_regular": float(candidates.at[chosen_index, "gap_regular"]),
            "gap_periodic": float(candidates.at[chosen_index, "gap_periodic"]),
            "nufrost_rmse": float(candidates.at[chosen_index, "nufrost_rmse"]),
            "zhu2015_rmse": float(candidates.at[chosen_index, "zhu2015_rmse"]),
            "hants_rmse": float(candidates.at[chosen_index, "hants_rmse"]),
            "advantage": float(candidates.at[chosen_index, "advantage"]),
            "selection_score": float(candidates.at[chosen_index, "selection_score"]),
        }
        selected.append(record)
        used_scenes.add(record["scene_id"])
    columns = [
        "case_label",
        "selection_stratum",
        "sample_idx",
        "holdout_seed",
        "scene_id",
        "gap_regular",
        "gap_periodic",
        "nufrost_rmse",
        "zhu2015_rmse",
        "hants_rmse",
        "advantage",
        "selection_score",
    ]
    return pd.DataFrame(selected, columns=columns)


def plot_band_comparison(source: pd.DataFrame, output_stem: Path) -> None:
    categories = BANDS + ["Overall"]
    x = np.arange(len(categories))
    bar_width = 0.24
    offsets = np.linspace(-bar_width, bar_width, len(METHODS))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.75), constrained_layout=True)
    for ax, metric, label in zip(axes, ("RMSE", "MAE"), ("a", "b")):
        panel = source[source["metric"] == metric]
        panel_high = 0.0
        for method, offset in zip(METHODS, offsets):
            rows = panel[panel["method_key"] == method].set_index("band").loc[categories]
            y = rows["estimate"].to_numpy() * 1_000
            low = rows["ci_low"].to_numpy() * 1_000
            high = rows["ci_high"].to_numpy() * 1_000
            panel_high = max(panel_high, float(np.nanmax(high)))
            ax.bar(
                x + offset,
                y,
                width=bar_width,
                yerr=np.vstack([y - low, high - y]),
                color=BAND_BAR_COLORS[method],
                edgecolor="white",
                linewidth=0.45,
                capsize=2.2,
                error_kw={
                    "ecolor": BAND_ERROR_COLORS[method],
                    "elinewidth": 0.8,
                    "capthick": 0.8,
                },
                label=METHOD_LABELS[method],
                zorder=3,
            )
        ax.axvline(5.5, color="#D0D0D0", linewidth=0.8)
        ax.set_xticks(x, categories)
        ax.set_ylim(0, panel_high * 1.08)
        ax.set_ylabel(rf"Reflectance {metric} ($\times 10^{{-3}}$)")
        ax.grid(axis="y", color="#E8E8E8", linewidth=0.6)
        ax.set_axisbelow(True)
        add_panel_label(ax, label)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.52, 1.10),
        ncol=3,
    )
    save_figure(fig, output_stem)


def plot_gap_indices(source: pd.DataFrame, output_stem: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), constrained_layout=True)
    for ax, (gap_key, xlabel), label in zip(
        axes,
        (("gap_regular", r"$I_{\mathrm{gap}}$ quantile bin (low $\rightarrow$ high)"),
         ("gap_periodic", r"$I_{\mathrm{periodic}}$ quantile bin (low $\rightarrow$ high)")),
        ("a", "b"),
    ):
        panel = source[source["gap_key"] == gap_key]
        for method in METHODS:
            rows = panel[panel["method_key"] == method].sort_values("bin")
            x = rows["bin"].to_numpy()
            y = rows["rmse"].to_numpy() * 1_000
            low = rows["ci_low"].to_numpy() * 1_000
            high = rows["ci_high"].to_numpy() * 1_000
            ax.fill_between(x, low, high, color=METHOD_COLORS[method], alpha=0.12, linewidth=0)
            ax.plot(
                x,
                y,
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                markersize=4,
                linewidth=1.5,
                label=METHOD_LABELS[method],
            )
        tick_rows = panel[panel["method_key"] == "nufrost"].sort_values("bin")
        tick_bins = tick_rows["bin"].to_numpy()
        ax.set_xticks(tick_bins, [str(value) for value in tick_bins])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"Reflectance RMSE ($\times 10^{-3}$)")
        ax.grid(axis="y", color="#E8E8E8", linewidth=0.6)
        add_panel_label(ax, label)
    axes[0].legend(loc="upper left")
    save_figure(fig, output_stem)


def plot_representative_cases(
    selection: pd.DataFrame, curve_dir: Path, source_path: Path, output_stem: Path
) -> bool:
    frames: list[pd.DataFrame] = []
    for index, row in selection.reset_index(drop=True).iterrows():
        path = curve_dir / f"case_{index + 1}.csv"
        if not path.exists():
            return False
        frame = pd.read_csv(path)
        frame["case_label"] = row["case_label"]
        frame["gap_regular"] = row["gap_regular"]
        frame["gap_periodic"] = row["gap_periodic"]
        for column in ["observed", "nufrost", "hants", "zhu2015"]:
            frame[column] = frame[column] / REFLECTANCE_SCALE
        frames.append(frame)
    source = pd.concat(frames, ignore_index=True)
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source.to_csv(source_path, index=False)

    display_bands = ["B4", "B8", "B11"]
    fig, axes = plt.subplots(3, 3, figsize=(7.0, 6.1), constrained_layout=True)
    for row_index, selected in selection.reset_index(drop=True).iterrows():
        frame = frames[row_index]
        for col_index, band in enumerate(display_bands):
            ax = axes[row_index, col_index]
            band_data = frame[frame["band"] == band]
            observed = band_data[band_data["kind"] == "observed"]
            target = band_data[band_data["kind"] == "target"]
            curve = band_data[band_data["kind"] == "curve"]
            if not target.empty:
                ax.axvline(
                    float(target["time_year"].iloc[0]),
                    color="#BDBDBD",
                    linestyle="--",
                    linewidth=0.7,
                    zorder=1,
                )
            ax.scatter(
                observed["time_year"],
                observed["observed"],
                s=7,
                facecolor="#B8B8B8",
                edgecolor="none",
                alpha=0.55,
                zorder=2,
            )
            for method in METHODS:
                ax.plot(
                    curve["time_year"],
                    curve[method],
                    color=METHOD_COLORS[method],
                    linewidth=1.2,
                    label=METHOD_LABELS[method],
                    zorder=3,
                )
            ax.scatter(
                target["time_year"],
                target["observed"],
                s=34,
                marker="*",
                facecolor="black",
                edgecolor="white",
                linewidth=0.4,
                label="Withheld truth",
                zorder=6,
            )
            ax.grid(axis="y", color="#ECECEC", linewidth=0.5)
            if row_index == 0:
                ax.set_title(band)
            if col_index == 0:
                ax.set_ylabel(
                    f"{selected['case_label']}\nReflectance",
                    multialignment="center",
                )
            if row_index == 2:
                ax.set_xlabel("Time since first acquisition (years)")
            else:
                ax.set_xticklabels([])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    ordered = [0, 1, 2, 3]
    fig.legend(
        [handles[i] for i in ordered],
        [labels[i] for i in ordered],
        loc="upper center",
        bbox_to_anchor=(0.52, 1.035),
        ncol=4,
    )
    for label, ax in zip("abcdefghi", axes.ravel()):
        add_panel_label(ax, label)
    save_figure(fig, output_stem)
    return True


def write_notes(path: Path, n_tasks: int, n_boot: int, cases_ready: bool) -> None:
    text = f"""# Reconstruction figure contract and QA notes

Core conclusion: NUFROST is evaluated against HANTS and Zhu2015 on identical
holdout tasks, with comparisons resolved by spectral band and illustrated by
three traceable cases.

- Backend: Python/matplotlib only.
- Target: IEEE TGRS double-column figures, 7.0 inches wide (within the 7.16-inch IEEE limit).
- Evaluation tasks available to every method: {n_tasks}.
- Bands: {', '.join(BANDS)}.
- Intervals: 95% AOI-cluster bootstrap confidence intervals ({n_boot} replicates).
- Gap-index curves are retained as diagnostics and are not intended for the main text.
- Representative cases: deterministic median-profile selection across three
  native temporal-support strata; main-figure labels are neutral and do not imply
  a quantitative gap-robustness result; curves ready: {cases_ready}.
- Raster/vector exports: SVG, PDF, 600-dpi TIFF, and 300-dpi PNG preview.
- Text remains editable in SVG and PDF.
- No local contrast manipulation, clipping, or smoothing is applied to the data.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-dir", type=Path, default=Path("data/tests/reconstruction_eval_20260809")
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache/sentinel-2/100k"))
    parser.add_argument(
        "--curve-dir",
        type=Path,
        default=Path("data/tests/reconstruction_eval_20260809/representative_cases"),
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path("data/assets/figures/reconstruction_evaluation"),
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/assets/records/reconstruction_evaluation"),
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260809)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_style()
    args.source_dir.mkdir(parents=True, exist_ok=True)
    data = load_predictions(args.eval_dir, args.cache_dir)
    rng = np.random.default_rng(args.bootstrap_seed)
    band_source = build_band_source(data, args.bootstrap_replicates, rng)
    gap_source = build_gap_source(data, args.bootstrap_replicates, rng)
    selection = choose_cases(data)

    band_source.to_csv(args.source_dir / "band_comparison_source.csv", index=False)
    gap_source.to_csv(args.source_dir / "gap_indices_source.csv", index=False)
    selection.to_csv(args.source_dir / "representative_case_selection.csv", index=False)

    plot_band_comparison(
        band_source, args.figure_dir / "method_band_comparison_v01"
    )
    plot_gap_indices(gap_source, args.figure_dir / "gap_index_comparison_v01")
    cases_ready = plot_representative_cases(
        selection,
        args.curve_dir,
        args.source_dir / "representative_cases_source.csv",
        args.figure_dir / "representative_cases_v02",
    )
    write_notes(
        args.source_dir / "figure_contract_and_qa.md",
        len(data),
        args.bootstrap_replicates,
        cases_ready,
    )
    print(selection.to_csv(index=False))
    if not cases_ready:
        print(f"Representative curve CSVs are not complete under {args.curve_dir}")


if __name__ == "__main__":
    main()
