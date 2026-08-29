#!/usr/bin/env python3
"""Finite-mode source and stationary-response study for the manuscript.

The filename is retained for archive compatibility. The calculation does not
derive a gravitational interaction and does not simulate a sonic horizon. It
implements three deliberately limited steps:

1. evaluate the exact relative-entropy/free-energy identity for two ideal,
   energy-matched finite-mode preparations;
2. propagate a freely postulated source through the homogeneous static
   Gross--Pitaevskii/Bogoliubov response kernel; and
3. express a prospective bound on the free coupling as a function of an
   externally supplied residual-amplitude budget.

All plotted response amplitudes are coupling-normalized. The script therefore
contains no predicted coupling, detection threshold, or gravitational claim.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parent

# Dimensionless finite-band illustration used in the manuscript.
BASELINE_PHONONS = 1000.0
BASELINE_MODES = 100
X_BAR = 2.0  # beta * hbar * omega for the equal-frequency illustration

# Dimensionless stationary-response grid, shared with the independent GPE
# validation script. The response is evaluated at fixed total particle number,
# so the spatially uniform Fourier component is removed.
XI = 1.0 / np.sqrt(2.0)
SOURCE_WIDTH = 10.0
DOMAIN_LENGTH = 400.0
N_GRID = 2048


def thermal_occupation(x_bar):
    """Bose occupation at dimensionless frequency ``x_bar``."""
    if x_bar <= 0:
        raise ValueError("x_bar must be positive")
    return 1.0 / np.expm1(x_bar)


def mode_entropy(n_bar):
    """Entropy in nats of a one-mode thermal state."""
    if n_bar < 0:
        raise ValueError("occupation must be non-negative")
    if n_bar == 0:
        return 0.0
    return (1.0 + n_bar) * np.log1p(n_bar) - n_bar * np.log(n_bar)


def finite_mode_contrast(n_phonons, n_modes, x_bar):
    """Return the two relative entropies and their exact entropy difference.

    ``displaced`` is an ideal displaced Gibbs state. ``heated`` is an ideal
    product Gibbs preparation whose occupation is raised uniformly across the
    specified finite band. They are equal-energy constructions, not models of
    the phase-randomized classical fields used in the separate GPE script.
    """
    if n_phonons <= 0 or n_modes <= 0:
        raise ValueError("n_phonons and n_modes must be positive")
    n_ref = thermal_occupation(x_bar)
    injected_per_mode = n_phonons / n_modes
    delta_entropy = n_modes * (
        mode_entropy(n_ref + injected_per_mode) - mode_entropy(n_ref)
    )
    d_displaced = n_phonons * x_bar
    d_heated = d_displaced - delta_entropy
    return d_displaced, d_heated, delta_entropy


def source_envelope(x_values, width):
    """Unit-peak exponential source envelope."""
    if width <= 0:
        raise ValueError("width must be positive")
    return np.exp(-np.abs(x_values) / width)


def static_response(source, dx, xi):
    """Homogeneous static GP/BdG response to a dimensionless source.

    In Fourier space the implemented kernel is

        delta n_k / n_0 = source_k / (1 + (k xi)^2 / 2).

    This is a stationary linear-response check. It is not a transonic or
    time-dependent condensate calculation.
    """
    k_values = 2.0 * np.pi * np.fft.fftfreq(source.size, d=dx)
    kernel = 1.0 / (1.0 + 0.5 * (k_values * xi) ** 2)
    source_k = np.fft.fft(source)
    source_k[0] = 0.0  # fixed-N constraint; uniform shift enters chemical potential
    return np.fft.ifft(source_k * kernel).real


def response_templates(n_phonons=BASELINE_PHONONS, n_modes=BASELINE_MODES,
                       x_bar=X_BAR, source_width=SOURCE_WIDTH, xi=XI):
    """Return coupling-normalized response templates."""
    dx = DOMAIN_LENGTH / N_GRID
    x_values = (np.arange(N_GRID) - N_GRID // 2) * dx
    envelope = source_envelope(x_values, source_width)
    d_displaced, d_heated, delta_entropy = finite_mode_contrast(
        n_phonons, n_modes, x_bar
    )
    unit_response = static_response(envelope, dx, xi)
    return {
        "x": x_values,
        "unit_response": unit_response,
        "displaced_per_kappa": d_displaced * unit_response,
        "heated_per_kappa": d_heated * unit_response,
        "contrast_per_kappa": delta_entropy * unit_response,
        "d_displaced": d_displaced,
        "d_heated": d_heated,
        "delta_entropy": delta_entropy,
    }


def prospective_kappa_bound(residual_amplitude, delta_entropy, response_peak):
    """Transfer an externally established residual budget to a coupling bound."""
    if residual_amplitude <= 0 or delta_entropy <= 0 or response_peak <= 0:
        raise ValueError("bound inputs must be positive")
    return residual_amplitude / (delta_entropy * response_peak)


def save_outputs(results):
    """Write the tracked figure and data tables."""
    x_values = results["x"]
    unit_response = results["unit_response"]
    response_peak = float(np.max(np.abs(unit_response)))

    residual_grid = np.logspace(-6, -2, 81)
    bounds = np.array([
        prospective_kappa_bound(value, results["delta_entropy"], response_peak)
        for value in residual_grid
    ])

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 5))
    ax_left.plot(
        x_values / SOURCE_WIDTH,
        results["displaced_per_kappa"],
        color="tab:blue",
        linewidth=2,
        label=r"displaced Gibbs pair: $\delta n/(n_0\kappa)$",
    )
    ax_left.plot(
        x_values / SOURCE_WIDTH,
        results["heated_per_kappa"],
        color="0.45",
        linestyle="--",
        linewidth=2,
        label=r"heated Gibbs pair: $\delta n/(n_0\kappa)$",
    )
    ax_left.plot(
        x_values / SOURCE_WIDTH,
        results["contrast_per_kappa"],
        color="tab:green",
        linewidth=2,
        label=r"contrast: $\Delta S\,r(x)$",
    )
    ax_left.set_xlabel(r"$x/w$")
    ax_left.set_ylabel("coupling-normalized response")
    ax_left.set_title("Conditional stationary response templates")
    ax_left.grid(True, alpha=0.25)
    ax_left.legend(fontsize=8)

    ax_right.loglog(residual_grid, bounds, color="tab:purple", linewidth=2)
    ax_right.set_xlabel(r"calibrated residual budget $\delta A_{\rm res}$")
    ax_right.set_ylabel(r"prospective bound on $|\kappa|$")
    ax_right.set_title("Bound transfer function, not a forecast")
    ax_right.grid(True, which="both", alpha=0.25)

    fig.tight_layout()
    figure_path = BASE_DIR / "bec_sonic_horizon_results.png"
    fig.savefig(
        figure_path,
        dpi=150,
        bbox_inches="tight",
        metadata={"Software": "matplotlib"},
    )
    plt.close(fig)

    np.savetxt(
        BASE_DIR / "bec_sonic_horizon_data.dat",
        np.column_stack([
            x_values,
            unit_response,
            results["displaced_per_kappa"],
            results["heated_per_kappa"],
            results["contrast_per_kappa"],
        ]),
        header=(
            "x unit_response displaced_response_per_kappa "
            "heated_response_per_kappa contrast_response_per_kappa"
        ),
        fmt="%.9e",
    )

    mode_grid = np.unique(np.round(np.logspace(0, 3, 61)).astype(int))
    rows = []
    for n_modes in mode_grid:
        d_displaced, d_heated, delta_entropy = finite_mode_contrast(
            BASELINE_PHONONS, int(n_modes), X_BAR
        )
        rows.append([
            n_modes,
            d_displaced,
            d_heated,
            delta_entropy,
            delta_entropy / d_displaced,
            delta_entropy * response_peak,
        ])
    np.savetxt(
        BASE_DIR / "bec_sonic_horizon_robustness.dat",
        np.asarray(rows),
        header=(
            "n_modes D_displaced D_heated delta_entropy contrast_fraction "
            "peak_contrast_response_per_kappa"
        ),
        fmt="%.9e",
    )
    return figure_path, residual_grid, bounds


def main():
    results = response_templates()
    figure_path, residual_grid, bounds = save_outputs(results)
    response_peak = float(np.max(np.abs(results["unit_response"])))
    print("Finite-mode relative-entropy source study")
    print("=" * 48)
    print(f"D(displaced || Gibbs) = {results['d_displaced']:.6f} nats")
    print(f"D(heated || Gibbs)    = {results['d_heated']:.6f} nats")
    print(f"Exact difference ΔS   = {results['delta_entropy']:.6f} nats")
    print(
        "Contrast fraction       = "
        f"{results['delta_entropy'] / results['d_displaced']:.6f}"
    )
    print(f"Fixed-N unit-response peak = {response_peak:.9f}")
    print(
        "Prospective |kappa| range for residual budgets "
        f"[{residual_grid[0]:.1e}, {residual_grid[-1]:.1e}]: "
        f"[{bounds[0]:.3e}, {bounds[-1]:.3e}]"
    )
    print("No value of kappa is derived, fitted, or forecast.")
    print(f"Figure saved to {figure_path}")


if __name__ == "__main__":
    main()
