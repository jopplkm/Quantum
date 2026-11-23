# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "stim", "pymatching>=2.3", "matplotlib"]
# ///
"""Compare equal-TV changes to three decoder prior coordinates."""

import argparse
import json
from math import exp, log, sqrt
from pathlib import Path

import numpy as np
import pymatching
import stim
from scipy.stats import beta

from q3_exact_ml import exact_joint_distribution
from q3_experiment import (
    TRUE_PRIOR, decoder_model, edge_weights, fault_signatures, memory_circuit, paired_stats,
)

PARAMETERS = ("strength", "bias", "correlation")
EQUAL_PAULI_PRIOR = (0.004, 0.004, 0.004)


def pauli_prior(strength, bias, log_odds_ratio):
    qx = strength * (1 + bias) / 2
    qz = strength * (1 - bias) / 2
    odds_ratio = exp(log_odds_ratio)
    b = 1 + (odds_ratio - 1) * (qx + qz)
    c = odds_ratio * qx * qz
    py = 2 * c / (b + sqrt(b * b - 4 * (odds_ratio - 1) * c))
    return np.array([1 - qx - qz + py, qx - py, py, qz - py])


def matched_tv_priors(levels, true_prior):
    px, py, pz = true_prior
    qx, qz = px + py, pz + py
    log_odds_ratio = log((1 - px - py - pz) * py / (px * pz))
    center = np.array([qx + qz, (qx - qz) / (qx + qz), log_odds_ratio])
    baseline = pauli_prior(*center)
    np.testing.assert_allclose(baseline[1:], true_prior, atol=1e-15)
    # Search each coordinate separately while keeping the other two fixed.
    for level in levels:
        for index, name in enumerate(PARAMETERS):
            broad = level > 0.002
            signs = (-1,) if broad and name != "bias" else (-1, 1)
            for sign in signs:
                def prior_at(step):
                    coordinates = center.copy()
                    coordinates[index] += sign * step
                    prior = pauli_prior(*coordinates)
                    return coordinates, prior, float(abs(baseline - prior).sum() / 2)

                low, high = 0.0, (0.999 * center[0], 0.999, 30.0)[index]
                if prior_at(high)[2] < level:
                    raise ValueError(f"Cannot bracket TV={level} for {name}")
                for _ in range(55):
                    mid = (low + high) / 2
                    if prior_at(mid)[2] < level:
                        low = mid
                    else:
                        high = mid
                coordinates, prior, tv = prior_at(high)
                if np.any(prior < 0) or not np.isclose(tv, level, atol=1e-14):
                    raise ValueError(f"Infeasible TV={level} for {name}")
                yield name, sign, coordinates.tolist(), prior.tolist(), tv


def make_decoder(ideal, signatures, prior, correlated):
    if not correlated:
        px, py, pz = prior
        qx, qz = px + py, py + pz
        prior = (qx * (1 - qz), qx * qz, (1 - qx) * qz)
    dem = decoder_model(ideal, signatures, prior)
    return pymatching.Matching.from_detector_error_model(dem, enable_correlations=correlated)


def paired_interval(reference_failed, failed):
    stats = paired_stats(reference_failed, failed)
    worse = stats["only_comparison_prior_failed"]
    better = stats["only_true_prior_failed"]
    if worse + better:
        # Bonferroni-combine two exact binomial intervals for the discordant rates.
        n = len(failed)
        tail = 0.05 / 4

        def bounds(count):
            return (
                beta.ppf(tail, count, n - count + 1) if count else 0.0,
                beta.ppf(1 - tail, count + 1, n - count) if count < n else 1.0,
            )

        worse_low, worse_high = bounds(worse)
        better_low, better_high = bounds(better)
        stats["95_percent_ci"] = [worse_low - better_high, worse_high - better_low]
    return stats


def analyze(distance, basis, shots, seed, priors, true_prior):
    ideal = memory_circuit(basis, distance, lambda _: stim.Circuit()).detector_error_model()
    signatures = fault_signatures(basis, distance)
    if distance == 3:
        joint = exact_joint_distribution(signatures, ideal.num_detectors, true_prior)
        masks = np.flatnonzero(joint.sum(axis=1))
        joint = joint[masks]
        dets = ((masks[:, None] >> np.arange(ideal.num_detectors)) & 1).astype(np.uint8)
    else:
        def noise(data):
            circuit = stim.Circuit()
            circuit.append("PAULI_CHANNEL_1", data, true_prior)
            return circuit

        circuit = memory_circuit(basis, distance, noise)
        dets, actual = circuit.compile_detector_sampler(seed=seed).sample(
            shots, separate_observables=True
        )

    rows, baselines = [], {}
    for mode, correlated in (("standard", False), ("correlated", True)):
        decoder = make_decoder(ideal, signatures, true_prior, correlated)
        reference = decoder.decode_batch(dets, enable_correlations=correlated)[:, 0]
        reference_weights = edge_weights(decoder)
        if distance == 3:
            positions = np.arange(len(reference))
            baseline = float(joint[positions, 1 - reference].sum())
        else:
            reference_failed = reference != actual[:, 0]
            baseline = float(reference_failed.mean())
        baselines[mode] = baseline

        for name, sign, coords, prior, tv in priors:
            decoder = make_decoder(ideal, signatures, prior[1:], correlated)
            predicted = decoder.decode_batch(dets, enable_correlations=correlated)[:, 0]
            if name in ("correlation", "independence"):
                weights = edge_weights(decoder)
                if weights.keys() != reference_weights.keys() or any(
                    abs(weights[edge] - weight) > 1e-10
                    for edge, weight in reference_weights.items()
                ):
                    raise RuntimeError("Correlation perturbation changed initial weights")
            if distance == 3:
                ler = float(joint[positions, 1 - predicted].sum())
                stats = {"ler_difference": ler - baseline}
            else:
                failed = predicted != actual[:, 0]
                ler = float(failed.mean())
                stats = paired_interval(reference_failed, failed)
            rows.append({
                "decoder": mode, "parameter": name, "direction": sign, "tv": tv,
                "strength_bias_log_odds_ratio": coords, "prior_i_x_y_z": prior, "ler": ler,
                "prediction_disagreements": int(np.count_nonzero(predicted != reference)),
                **stats,
            })
    return {
        "distance": distance, "basis": basis,
        "method": "exact enumeration" if distance == 3 else "paired simulation",
        "shots": None if distance == 3 else shots, "matched_baselines": baselines,
        "exact_ml_ler": float(joint.min(axis=1).sum()) if distance == 3 else None,
        "rows": rows,
    }


def plot(results, output, tv):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.3), layout="constrained", sharex=True)
    for ax, result in zip(axes.flat, results):
        ax.axhline(0, color="0.55", linewidth=0.8)
        for center, mode, color, marker in (
            (-0.1, "standard", "#1b6580", "o"),
            (0.1, "correlated", "#a5582c", "s"),
        ):
            rows = [r for r in result["rows"] if r["decoder"] == mode]
            for index, parameter in enumerate(PARAMETERS):
                selected = [
                    r for r in rows if r["parameter"] == parameter and np.isclose(r["tv"], tv)
                ]
                for row in selected:
                    x = index + center + row["direction"] * 0.035
                    y = row["ler_difference"] * 100
                    interval = row.get("95_percent_ci")
                    if interval is not None:
                        ax.vlines(x, interval[0]*100, interval[1]*100, color=color, alpha=0.55)
                    ax.scatter(x, y, marker=marker, color=color, s=25,
                               label=mode.capitalize() if index == 0 and row["direction"] == -1 else None)
        ax.set_title(f"d={result['distance']} · {result['basis'].upper()}-memory")
        ax.set_xticks(range(len(PARAMETERS)), ["Strength", "X/Z bias", "X–Z correlation"])
        ax.grid(axis="y", alpha=0.18)
    for ax in axes[:, 0]:
        ax.set_ylabel("LER change (percentage points)")
    axes[0, 0].legend(frameon=False)
    fig.suptitle(f"Q3 | Prior sensitivity at equal total variation (TV = {tv:.1%})")
    fig.supxlabel("Left/right offsets show negative/positive perturbations within each decoder")
    fig.savefig(output, dpi=200)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--equal-pauli", action="store_true")
    args = parser.parse_args()
    if args.shots < 1:
        parser.error("shots must be positive")
    true_prior = EQUAL_PAULI_PRIOR if args.equal_pauli else TRUE_PRIOR
    levels = (0.0005, 0.0009, 0.005) if args.equal_pauli else (0.0005, 0.0009, 0.005, 0.01)
    prefix = "q3_equal_pauli" if args.equal_pauli else "q3_parameter_sensitivity"
    priors = list(matched_tv_priors(levels, true_prior))
    if args.equal_pauli:
        px, py, pz = true_prior
        qx, qz = px + py, pz + py
        strength, bias = qx + qz, (qx - qz) / (qx + qz)
        prior = pauli_prior(strength, bias, 0)
        # At fixed marginals, all four probabilities shift by |py - qx*qz|.
        tv = 2 * abs(py - qx * qz)
        priors.append(("independence", -1, [strength, bias, 0], prior.tolist(), float(tv)))
    results = [
        analyze(distance, basis, args.shots, args.seed, priors, true_prior)
        for distance in (3, 5) for basis in ("z", "x")
    ]
    report = {"true_prior_x_y_z": true_prior, "tv_levels": levels, "seed": args.seed, "results": results}
    Path(f"{prefix}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    plot(results, f"{prefix}.png", levels[-1])


if __name__ == "__main__":
    main()
