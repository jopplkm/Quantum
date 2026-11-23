# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "stim", "pymatching>=2.3", "matplotlib"]
# ///
"""Pauli-projected Figure 7 comparison with standard and correlated MWPM."""

import argparse
import itertools
import json
from math import expm1
from pathlib import Path

import numpy as np
import stim
from scipy.stats import beta

from q3_experiment import fault_signatures, independent_rates, memory_circuit
from q3_parameter_sensitivity import make_decoder


LABELS = tuple(a + b for a, b in itertools.product("IXYZ", repeat=2))
FIGURE_CASES = ((7, 10.88), (9, 16.32))
PAULI_DIAGONALS = {
    "II": (2, 2, 2, 2),
    "IZ": (2, 0, 2, 0),
    "ZI": (2, 2, 0, 0),
    "ZZ": (2, 0, 0, 2),
    "off_diagonal": (1, 1, 1, 1),
}


def projected_priors(t1, delta):
    # Figure 7 uses T1 = T2 and t = 1, giving equal X, Y and Z probabilities.
    p = -expm1(-1 / t1) / 4
    matched = np.array([1 - 3 * p, p, p, p])
    priors = {"matched": matched}
    for name, diagonal in PAULI_DIAGONALS.items():
        prior = matched + delta * np.array(diagonal)
        priors[name] = prior / prior.sum()
    return priors


def validate_joint_prior(prior):
    """Check that the DEM mechanisms reconstruct the projected Pauli prior."""
    rates = independent_rates(prior[1:])
    if any(not 0 <= rate <= 1 for rate in rates):
        raise ValueError(f"Prior cannot use this independent-mechanism DEM: {prior}")
    reconstructed = np.array([1.0, 0.0, 0.0, 0.0])
    for mask, rate in zip((1, 2, 3), rates):
        reconstructed = (1 - rate) * reconstructed + rate * reconstructed[np.arange(4) ^ mask]
    np.testing.assert_allclose(reconstructed, prior, atol=1e-13, rtol=0)


def difference_ci(worse, better, shots):
    if worse + better == 0:
        bound = 1 - 0.05 ** (1 / shots)
        return [-bound, bound]
    # Two simultaneous exact binomial intervals bound the difference of paired rates.
    def bounds(count):
        tail = 0.05 / 4
        low = beta.ppf(tail, count, shots - count + 1) if count else 0.0
        high = beta.ppf(1 - tail, count + 1, shots - count) if count < shots else 1.0
        return low, high

    worse_low, worse_high = bounds(worse)
    better_low, better_high = bounds(better)
    return [float(worse_low - better_high), float(worse_high - better_low)]


def run_case(distance, t1, basis, shots, seed, delta):
    priors = projected_priors(t1, delta)
    for prior in priors.values():
        validate_joint_prior(prior)

    def noise(data):
        layer = stim.Circuit()
        layer.append("PAULI_CHANNEL_1", data, priors["matched"][1:])
        return layer

    sampler = memory_circuit(basis, distance, noise).compile_detector_sampler(seed=seed)
    ideal = memory_circuit(basis, distance, lambda _: stim.Circuit()).detector_error_model()
    signatures = fault_signatures(basis, distance)
    decoders = {}
    counts = {}
    for correlated in (False, True):
        decoders[correlated] = {
            name: make_decoder(ideal, signatures, prior[1:], correlated)
            for name, prior in priors.items()
        }
        counts[correlated] = {name: np.zeros(3, dtype=np.int64) for name in priors}

    for start in range(0, shots, 50_000):
        dets, actual = sampler.sample(
            min(50_000, shots - start), separate_observables=True, bit_packed=True
        )
        for correlated, models in decoders.items():
            def failed(decoder):
                prediction = decoder.decode_batch(
                    dets, bit_packed_shots=True, enable_correlations=correlated
                )
                return prediction[:, 0] != actual[:, 0]

            reference = failed(models["matched"])
            counts[correlated]["matched"][0] += reference.sum()
            for name in PAULI_DIAGONALS:
                error = failed(models[name])
                counts[correlated][name] += (
                    error.sum(), (error & ~reference).sum(), (reference & ~error).sum()
                )

    results = {}
    for correlated, group in counts.items():
        mode = "correlated" if correlated else "standard"
        results[mode] = {}
        for name, (failures, worse, better) in group.items():
            assert failures - group["matched"][0] == worse - better
            results[mode][name] = {
                "failures": int(failures), "ler": float(failures / shots),
                "only_perturbed_failed": int(worse), "only_matched_failed": int(better),
                "delta_ler": float((worse - better) / shots),
                "95_percent_ci": difference_ci(worse, better, shots) if name != "matched" else None,
            }
    return {
        "distance": distance, "T1": t1, "T2": t1, "basis": basis, "shots": shots,
        "priors_i_x_y_z": {name: prior.tolist() for name, prior in priors.items()},
        "decoders": results,
    }


def plot(results, delta, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 6.8), layout="constrained", sharex=True)
    for ax, result in zip(axes.flat, results):
        positions = np.arange(len(LABELS))
        ax.axhline(0, color="0.45", linewidth=0.8)
        for mode, offset, color in (("standard", -0.18, "#246582"), ("correlated", 0.18, "#ae582d")):
            data = result["decoders"][mode]
            rows = [data[label if label in PAULI_DIAGONALS else "off_diagonal"] for label in LABELS]
            values = np.array([row["delta_ler"] for row in rows]) * 100
            intervals = np.array([row["95_percent_ci"] for row in rows]) * 100
            x = positions + offset
            ax.bar(x, values, width=0.34, color=color, label=mode.capitalize(), alpha=0.84)
            ax.errorbar(x, values, yerr=[values - intervals[:, 0], intervals[:, 1] - values],
                        fmt=".", color=color, markersize=3, capsize=1.8, linewidth=0.8)
        ax.set_title(f"d={result['distance']}, T₁=T₂={result['T1']:.2f} · {result['basis'].upper()}-memory")
        ax.set_xticks(positions, LABELS)
        ax.grid(axis="y", alpha=0.18)
    for ax in axes[:, 0]:
        ax.set_ylabel("ΔLER (percentage points)")
    axes[0, 0].legend(frameon=False)
    fig.supxlabel("Original perturbation label · 16 labels map to 5 distinct Pauli priors")
    fig.suptitle(f"Figure 7 Pauli projection | ΔLER vs. matched MWPM (δ={delta:g}, paired 95% CI)")
    fig.savefig(output, dpi=200)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--delta", type=float, default=0.25)
    parser.add_argument("--output", type=Path, default=Path("q3_figure7"))
    args = parser.parse_args()
    if args.shots < 1 or not 0 <= args.delta <= 0.25:
        parser.error("shots must be positive and delta must be between 0 and 0.25")
    results = []
    for distance, t1 in FIGURE_CASES:
        for basis in ("z", "x"):
            result = run_case(distance, t1, basis, args.shots, args.seed, args.delta)
            results.append(result)
            print(f"d={distance} {basis.upper()}-memory completed", flush=True)
    report = {
        "delta": args.delta, "seed": args.seed,
        "label_to_prior": {label: label if label in PAULI_DIAGONALS else "off_diagonal" for label in LABELS},
        "cases": results,
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    plot(results, args.delta, args.output.with_suffix(".png"))


if __name__ == "__main__":
    main()
