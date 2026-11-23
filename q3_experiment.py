# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "stim", "pymatching>=2.3"]
# ///
"""Compare MWPM priors at fixed X/Z component marginals."""

import argparse
import json
from math import sqrt
from pathlib import Path

import numpy as np
import pymatching
import stim


TRUE_PRIOR = (0.001, 0.010, 0.001)
QX = QZ = 0.011
INDEPENDENT_PRIOR = (QX * (1 - QZ), QX * QZ, (1 - QX) * QZ)


def memory_circuit(basis, distance, noise):
    """Insert one noise layer between two ideal stabilizer rounds."""
    ideal = stim.Circuit.generated(
        f"surface_code:rotated_memory_{basis}", distance=distance, rounds=2
    ).flattened()
    data = [
        target.value
        for instruction in ideal
        if instruction.name in ("M", "MX")
        for target in instruction.targets_copy()
    ]
    first_reset = next(i for i, gate in enumerate(ideal) if gate.name == "MR")
    if len(data) != distance**2 or len(set(data)) != len(data):
        raise ValueError("Unexpected data qubits in generated circuit")
    return ideal[: first_reset + 1] + noise(data) + ideal[first_reset + 1 :]


def pauli_noise(data):
    circuit = stim.Circuit()
    circuit.append("PAULI_CHANNEL_1", data, TRUE_PRIOR)
    return circuit


def independent_rates(prior):
    """Represent this Pauli channel exactly with independent Pauli flips."""
    px, py, pz = prior
    x_parity = 1 - 2 * (px + py)
    y_parity = 1 - 2 * (px + pz)
    z_parity = 1 - 2 * (py + pz)
    return (
        (1 - sqrt(x_parity * y_parity / z_parity)) / 2,
        (1 - sqrt(x_parity * z_parity / y_parity)) / 2,
        (1 - sqrt(y_parity * z_parity / x_parity)) / 2,
    )


def fault_signatures(basis, distance):
    """Extract each data-qubit Pauli component's detector and observable flips."""
    signatures = []
    for index in range(distance**2):
        components = []
        for gate in ("X_ERROR", "Z_ERROR"):
            def single_fault(data):
                circuit = stim.Circuit()
                circuit.append(gate, [data[index]], 0.125)
                return circuit

            dem = memory_circuit(basis, distance, single_fault).detector_error_model()
            error, = (term for term in dem if term.type == "error")
            components.append(error.targets_copy())
        signatures.append(components)
    return signatures


def decoder_model(ideal_detectors, signatures, prior):
    dem = ideal_detectors.copy()
    rx, ry, rz = independent_rates(prior)
    for x_targets, z_targets in signatures:
        if rx > 1e-15:
            dem.append("error", rx, x_targets)
        if rz > 1e-15:
            dem.append("error", rz, z_targets)
        if ry > 1e-15:
            joint = x_targets + [stim.target_separator()] + z_targets
            dem.append("error", ry, joint)
    return dem


def edge_weights(decoder):
    return {
        tuple(sorted((u, v))) if v is not None else (u, None): attrs["weight"]
        for u, v, attrs in decoder.edges()
    }


def paired_stats(true_failed, other_failed):
    shots = len(true_failed)
    only_true = int((true_failed & ~other_failed).sum())
    only_other = int((other_failed & ~true_failed).sum())
    difference = (only_other - only_true) / shots
    interval = None
    if shots > 1 and only_true + only_other >= 30:
        variance = (only_true + only_other) / shots - difference**2
        margin = 1.96 * sqrt(variance / (shots - 1))
        interval = [difference - margin, difference + margin]
    elif only_true + only_other == 0:
        # With zero discordances, bound their total probability exactly.
        bound = 1 - 0.05 ** (1 / shots)
        interval = [-bound, bound]
    return {
        "only_true_prior_failed": only_true,
        "only_comparison_prior_failed": only_other,
        "ler_difference": difference,
        "95_percent_ci": interval,
    }


def run(shots, seed, basis, distance, sweep=False):
    physical = memory_circuit(basis, distance, pauli_noise)
    detectors, actual = physical.compile_detector_sampler(seed=seed).sample(
        shots, separate_observables=True
    )
    ideal_dem = memory_circuit(
        basis, distance, lambda data: stim.Circuit()
    ).detector_error_model()
    signatures = fault_signatures(basis, distance)
    results, failures, standard_weights, standard_predictions = {}, {}, {}, {}
    correlated_weights = {}
    for name, prior in (("true", TRUE_PRIOR), ("independent", INDEPENDENT_PRIOR)):
        joint_dem = decoder_model(ideal_dem, signatures, prior)
        px, py, pz = prior
        qx, qz = px + py, py + pz
        marginal_prior = (qx * (1 - qz), qx * qz, (1 - qx) * qz)
        # Build standard matching directly from the marginal mechanisms.
        marginal_dem = decoder_model(ideal_dem, signatures, marginal_prior)
        for mode, correlated in (("standard", False), ("correlated", True)):
            decoder = pymatching.Matching.from_detector_error_model(
                joint_dem if correlated else marginal_dem, enable_correlations=correlated
            )
            predicted = decoder.decode_batch(detectors, enable_correlations=correlated)
            key = f"{mode}_{name}"
            failures[key] = np.any(predicted != actual, axis=1)
            results[key] = {
                "failures": int(failures[key].sum()),
                "ler": float(failures[key].mean()),
            }
            if not correlated:
                standard_weights[name] = edge_weights(decoder)
                standard_predictions[name] = predicted
            else:
                correlated_weights[name] = edge_weights(decoder)

    true_weights = standard_weights["true"]
    independent_weights = standard_weights["independent"]
    same_edges = true_weights.keys() == independent_weights.keys()
    max_gap = max(
        (abs(true_weights[edge] - independent_weights[edge]) for edge in true_weights),
        default=0.0,
    ) if same_edges else None
    disagreement = int(np.any(
        standard_predictions["true"] != standard_predictions["independent"], axis=1
    ).sum())
    # Equal-cost paths can be resolved differently when floating-point weights differ.
    if not same_edges or max_gap > 1e-10:
        raise RuntimeError("Standard MWPM does not have matched edge weights")
    correlated_true = correlated_weights["true"]
    correlated_independent = correlated_weights["independent"]
    if correlated_true.keys() != correlated_independent.keys():
        raise RuntimeError("Correlated MWPM does not have matched edge sets")
    correlated_gap = max(
        abs(weight - correlated_independent[edge])
        for edge, weight in correlated_true.items()
    )
    if correlated_gap > 1e-10:
        raise RuntimeError("Correlated MWPM does not have matched initial edge weights")

    true_failed = failures["correlated_true"]
    independent_failed = failures["correlated_independent"]
    only_true = int((true_failed & ~independent_failed).sum())
    only_independent = int((independent_failed & ~true_failed).sum())
    paired_difference = (only_independent - only_true) / shots
    interval = None
    if shots > 1 and only_true + only_independent >= 30:
        variance = ((only_true + only_independent) / shots - paired_difference**2)
        margin = 1.96 * sqrt(variance / (shots - 1))
        interval = [paired_difference - margin, paired_difference + margin]
    result = {
        "distance": distance,
        "basis": basis,
        "shots": shots,
        "seed": seed,
        "stim": stim.__version__,
        "pymatching": pymatching.__version__,
        "results": results,
        "standard_check": {
            "same_edges": same_edges,
            "max_absolute_weight_gap": max_gap,
            "prediction_disagreements": disagreement,
        },
        "correlated_paired": {
            "max_initial_edge_weight_gap": correlated_gap,
            "only_true_prior_failed": only_true,
            "only_independent_prior_failed": only_independent,
            "ler_independent_minus_true": paired_difference,
            "approximate_95_percent_ci": interval,
        },
    }
    if sweep:
        rows = []
        for py in (QX * QZ, 0.0025, 0.005, 0.0075, TRUE_PRIOR[1], 0.0105):
            prior = (QX - py, py, QZ - py)
            dem = decoder_model(ideal_dem, signatures, prior)
            decoder = pymatching.Matching.from_detector_error_model(dem, enable_correlations=True)
            weights = edge_weights(decoder)
            if weights.keys() != correlated_true.keys():
                raise RuntimeError(f"Initial edge set mismatch at p_Y={py}")
            gap = max(abs(weights[edge] - weight) for edge, weight in correlated_true.items())
            if gap > 1e-10:
                raise RuntimeError(f"Initial edge weight mismatch at p_Y={py}")
            predicted = decoder.decode_batch(detectors, enable_correlations=True)
            failed = np.any(predicted != actual, axis=1)
            rows.append({
                "prior_py": py,
                "failures": int(failed.sum()),
                "ler": float(failed.mean()),
                "max_initial_edge_weight_gap": gap,
                **paired_stats(true_failed, failed),
            })
        result["correlation_sweep"] = rows
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--basis", choices=("x", "z"), default="z")
    parser.add_argument("--distance", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sweep", action="store_true")
    args = parser.parse_args()
    if args.shots < 1 or args.distance < 3 or args.distance % 2 == 0:
        parser.error("shots must be positive and distance must be odd and at least 3")
    result = json.dumps(run(args.shots, args.seed, args.basis, args.distance, args.sweep), indent=2)
    print(result)
    if args.output:
        args.output.write_text(result + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
