"""Deterministic two-bin checks for the manuscript's GLS identifiability gate.

This fixture tests the algebra in paper/main.tex.  It is not experimental data,
an uncertainty model, or evidence that the stipulated potential is physical.
"""

from pathlib import Path

import numpy as np


OUTPUT_PATH = Path(__file__).with_name("identifiability_audit_data.dat")


def projected_fit(
    y: np.ndarray,
    template: np.ndarray,
    nuisance: np.ndarray,
    covariance: np.ndarray,
) -> tuple[float, float]:
    """Return projected template norm and coefficient, or NaN if degenerate."""

    covariance_inv = np.linalg.inv(covariance)
    nuisance_gram = nuisance.T @ covariance_inv @ nuisance
    projector = (
        np.eye(template.size)
        - nuisance @ np.linalg.inv(nuisance_gram) @ nuisance.T @ covariance_inv
    )
    projected_norm = float(template.T @ covariance_inv @ projector @ template)
    if np.isclose(projected_norm, 0.0, atol=1.0e-12):
        return projected_norm, float("nan")
    estimate = float(
        template.T @ covariance_inv @ projector @ y / projected_norm
    )
    return projected_norm, estimate


def main() -> None:
    covariance = np.eye(2)
    template = np.array([1.0, 0.0])
    y = 0.25 * template

    identifiable_nuisance = np.array([[0.0], [1.0]])
    absorbed_nuisance = np.array([[1.0], [0.0]])

    positive_norm, estimate = projected_fit(
        y, template, identifiable_nuisance, covariance
    )
    zero_norm, absorbed_estimate = projected_fit(
        y, template, absorbed_nuisance, covariance
    )

    assert np.isclose(positive_norm, 1.0, atol=1.0e-12)
    assert np.isclose(estimate, 0.25, atol=1.0e-12)
    assert np.isclose(zero_norm, 0.0, atol=1.0e-12)
    assert np.isnan(absorbed_estimate)

    lines = [
        "# Deterministic GLS identifiability fixture",
        "# scenario projected_template_norm kappa_hat identifiable",
        f"orthogonal_nuisance {positive_norm:.12f} {estimate:.12f} true",
        f"template_absorbed {zero_norm:.12f} nan false",
    ]
    OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"orthogonal nuisance: norm={positive_norm:.12f}, kappa_hat={estimate:.12f}")
    print(f"template absorbed: norm={zero_norm:.12f}, identifiable=false")
    print(f"wrote {OUTPUT_PATH.name}")


if __name__ == "__main__":
    main()
