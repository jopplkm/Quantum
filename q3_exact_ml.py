# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "stim", "pymatching>=2.3"]
# ///
"""Compare exact d=3 logical decisions under the Q3 physical Pauli channel."""

import json
from pathlib import Path

import numpy as np
import pymatching
import stim

from q3_experiment import QX, QZ, TRUE_PRIOR, decoder_model, fault_signatures, memory_circuit


def signature_mask(targets, num_detectors):
    mask = 0
    for target in targets:
        if target.is_relative_detector_id():
            mask ^= 1 << target.val
        elif target.is_logical_observable_id() and target.val == 0:
            mask ^= 1 << num_detectors
        else:
            raise ValueError(f"Unsupported fault target: {target}")
    return mask


def exact_joint_distribution(signatures, num_detectors, prior=TRUE_PRIOR):
    """Enumerate all Pauli patterns, then group by detector mask and logical flip."""
    pattern_masks = np.array([0], dtype=np.int64)
    probabilities = np.array([1.0])
    pauli_probabilities = np.array([1 - sum(prior), *prior])
    for x_targets, z_targets in signatures:
        x = signature_mask(x_targets, num_detectors)
        z = signature_mask(z_targets, num_detectors)
        pattern_masks = (pattern_masks[:, None] ^ [0, x, x ^ z, z]).ravel()
        probabilities = (probabilities[:, None] * pauli_probabilities).ravel()
    joint = np.bincount(pattern_masks, weights=probabilities, minlength=2 << num_detectors)
    if not np.isclose(joint.sum(), 1, rtol=0, atol=1e-12):
        raise RuntimeError("Pauli pattern probabilities do not sum to one")
    return joint.reshape(2, -1).T


def analyze(basis):
    distance = 3
    ideal = memory_circuit(basis, distance, lambda _: stim.Circuit()).detector_error_model()
    signatures = fault_signatures(basis, distance)
    joint = exact_joint_distribution(signatures, ideal.num_detectors)
    masks = np.flatnonzero(joint.sum(axis=1))
    joint = joint[masks]
    dets = ((masks[:, None] >> np.arange(ideal.num_detectors)) & 1).astype(np.uint8)
    ml = joint.argmax(axis=1)
    predictions = {}
    errors = {}
    for name, py in (("low", 0.0025), ("true", TRUE_PRIOR[1])):
        prior = (QX - py, py, QZ - py)
        dem = decoder_model(ideal, signatures, prior)
        decoder = pymatching.Matching.from_detector_error_model(dem, enable_correlations=True)
        predictions[name] = decoder.decode_batch(dets, enable_correlations=True)[:, 0]
        errors[name] = joint[np.arange(len(masks)), 1 - predictions[name]]

    delta = errors["true"] - errors["low"]
    ml_error = joint.min(axis=1)
    if any(np.any(error < ml_error) for error in errors.values()):
        raise RuntimeError("A matching prediction beat exact ML on a syndrome")
    disagreement = np.flatnonzero(predictions["low"] != predictions["true"])
    tolerance = 1e-14
    low_wins = delta > tolerance
    true_wins = delta < -tolerance
    rows = [
        {
            "detectors": np.flatnonzero(dets[i]).tolist(),
            "syndrome_probability": float(joint[i].sum()),
            "logical_one_given_syndrome": float(joint[i, 1] / joint[i].sum()),
            "ml_flip": None if abs(joint[i, 0] - joint[i, 1]) <= tolerance else int(ml[i]),
            "low_prior_flip": int(predictions["low"][i]),
            "true_prior_flip": int(predictions["true"][i]),
            "true_minus_low_ler_contribution": float(delta[i]),
        }
        for i in sorted(disagreement, key=lambda i: abs(delta[i]), reverse=True)
    ]
    return {
        "basis": basis,
        "patterns": 4**(distance**2),
        "reachable_syndromes": len(masks),
        "exact_ler": {
            "ml": float(ml_error.sum()),
            "low_prior": float(errors["low"].sum()),
            "true_prior": float(errors["true"].sum()),
        },
        "differing_syndromes": len(rows),
        "low_prior_ml_matches_among_disagreements": int(low_wins.sum()),
        "true_prior_ml_matches_among_disagreements": int(true_wins.sum()),
        "ml_ties_among_disagreements": int(len(rows) - low_wins.sum() - true_wins.sum()),
        "low_prior_ler_gain_on_its_wins": float(delta[low_wins].sum()),
        "true_prior_ler_gain_on_its_wins": float(-delta[true_wins].sum()),
        "syndrome_details": rows,
    }


if __name__ == "__main__":
    results = [analyze(basis) for basis in ("z", "x")]
    Path("q3_exact_ml_results.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    for result in results:
        print(json.dumps({key: value for key, value in result.items() if key != "syndrome_details"}))
