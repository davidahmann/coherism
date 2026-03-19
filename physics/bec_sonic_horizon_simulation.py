#!/usr/bin/env python3
"""
Illustrative acoustic-model scaling study for the Coherism manuscript.

This script mirrors the scaling ansatz used in the analog-gravity appendix of
`coherism.tex`. It is not a full Gross-Pitaevskii/Bogoliubov simulation.

Representative prediction: coherent phonon injection produces
delta-rho/rho_0 ~ 1e-6 near the sonic horizon, while a matched thermal control
has no leading-order informational contribution in the adopted ansatz.

The baseline parameters are chosen to be compatible with the Gross-Pitaevskii
healing-length relation xi = hbar / (sqrt(2) m c_s).
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Physical constants for 87Rb BEC
HBAR = 1.055e-34  # J·s
KB = 1.381e-23    # J/K
M_RB = 1.44e-25   # kg (87Rb mass)
BASE_DIR = Path(__file__).resolve().parent

# BEC parameters (typical experimental values)
N_0 = 1e20        # Number density: 10^14 cm^-3 = 10^20 m^-3
XI = 0.3e-6       # Healing length: 0.3 μm
C_S = HBAR / (np.sqrt(2) * M_RB * XI)  # GP-compatible sound speed: ~1.73 mm/s
L_COH = 10e-6     # Coherence length: 10 μm
ALPHA = 1.0       # O(1) coupling constant
RHO_MASS_0 = N_0 * M_RB
R_H = 50e-6       # Horizon radius: 50 μm
BASELINE_PHONONS = 1000

# Derived quantities
KAPPA_EFF = ALPHA * HBAR * C_S / (RHO_MASS_0 * XI**2 * L_COH**2)


def gp_compatible_xi(c_s):
    """Healing length implied by the GP relation for a given sound speed."""
    return HBAR / (np.sqrt(2) * M_RB * c_s)


def compute_kappa_eff(c_s, rho_mass_0, xi, l_coh):
    """Phenomenological acoustic coupling used in the manuscript."""
    return ALPHA * HBAR * c_s / (rho_mass_0 * xi**2 * l_coh**2)


def hawking_temperature(A, r_H):
    """
    Acoustic Hawking temperature for draining vortex.
    T_H = ℏA / (2π k_B r_H²)

    Parameters:
        A: Vortex strength (m²/s)
        r_H: Horizon radius (m)
    """
    return HBAR * A / (2 * np.pi * KB * r_H**2)


def thermal_occupation(omega, T_H):
    """Bose-Einstein occupation number."""
    if T_H <= 0 or omega <= 0:
        return 0
    x = HBAR * omega / (KB * T_H)
    if x > 100:
        return 0
    return 1.0 / (np.exp(x) - 1)


def relative_entropy_coherent(alpha_sq, n_bar):
    """
    Relative entropy S(ρ||σ) for coherent state vs thermal.
    S = |α|² + n̄ log(1 + |α|²/(n̄+1))
    """
    if n_bar < 1e-10:
        return alpha_sq
    return alpha_sq + n_bar * np.log(1 + alpha_sq / (n_bar + 1))


def theta_tt_coherent(alpha_sq_total, n_bar):
    """
    Informational stress tensor Θ_tt for coherent phonons.
    Θ_tt = κ_eff * ℏ * c_s / ξ⁴ * Σ|α_k|² * (1 + 1/(2n̄+1))
    """
    factor = 1 + 1 / (2 * n_bar + 1) if n_bar > 0 else 2
    return KAPPA_EFF * HBAR * C_S / XI**4 * alpha_sq_total * factor


def theta_tt_coherent_model(alpha_sq_total, n_bar, kappa_eff, c_s, xi):
    """Parameterized coherent informational stress for scan calculations."""
    factor = 1 + 1 / (2 * n_bar + 1) if n_bar > 0 else 2
    return kappa_eff * HBAR * c_s / xi**4 * alpha_sq_total * factor


def theta_tt_thermal(n_bar_total):
    """
    Informational stress tensor Θ_tt for thermal phonons.
    For thermal state ρ = σ, relative entropy S(σ||σ) = 0.
    Therefore Θ_tt = 0 for thermal phonons.
    """
    return 0.0


def density_modulation(theta_tt):
    """
    Density modulation from informational stress.
    δρ/ρ₀ = Θ_tt / (ρ₀ c_s²)
    """
    return theta_tt / (RHO_MASS_0 * C_S**2)


def density_modulation_model(theta_tt, rho_mass_0, c_s):
    """Parameterized density response used in robustness scans."""
    return theta_tt / (rho_mass_0 * c_s**2)


def simulate_coherent_profile_with_params(r_values, r_H, c_s, n_0, xi, l_coh, n_phonons):
    """Coherent profile for an arbitrary parameter set within the same ansatz."""
    rho_mass_0 = n_0 * M_RB
    kappa_eff = compute_kappa_eff(c_s, rho_mass_0, xi, l_coh)
    T_H = hawking_temperature(c_s * r_H, r_H)
    omega_typical = c_s / xi
    n_bar = thermal_occupation(omega_typical, T_H)

    distance = np.maximum(np.abs(r_values - r_H), xi)
    profile = np.exp(-distance / l_coh)
    alpha_sq = n_phonons * profile
    theta = theta_tt_coherent_model(alpha_sq, n_bar, kappa_eff, c_s, xi)
    return density_modulation_model(theta, rho_mass_0, c_s)


def max_modulation_within_lcoh(r_vals_m, delta_rho_over_rho, r_h_m, l_coh_m):
    """Peak signal inside one coherence length of the horizon."""
    mask = np.abs(r_vals_m - r_h_m) <= l_coh_m
    return float(np.max(np.abs(delta_rho_over_rho[mask])))


def coherent_amplitude(c_s, n_0, xi, l_coh, n_phonons, r_h=R_H, num_points=500):
    """Primary observable A for a chosen parameter set."""
    r_vals = np.linspace(0.2 * r_h, 3.0 * r_h, num_points)
    delta = simulate_coherent_profile_with_params(
        r_vals, r_h, c_s, n_0, xi, l_coh, n_phonons
    )
    return max_modulation_within_lcoh(r_vals, delta, r_h, l_coh)


def robustness_scan_data():
    """One-at-a-time parameter scans around the baseline model."""
    factors = np.linspace(0.75, 1.25, 11)
    curves = {
        r'$n_0$': [
            coherent_amplitude(C_S, N_0 * f, XI, L_COH, BASELINE_PHONONS)
            for f in factors
        ],
        r'$\xi$': [
            coherent_amplitude(C_S, N_0, XI * f, L_COH, BASELINE_PHONONS)
            for f in factors
        ],
        r'$L_{\mathrm{coh}}$': [
            coherent_amplitude(C_S, N_0, XI, L_COH * f, BASELINE_PHONONS)
            for f in factors
        ],
        r'$N_{\mathrm{phonon}}$': [
            coherent_amplitude(C_S, N_0, XI, L_COH, BASELINE_PHONONS * f)
            for f in factors
        ],
        r'$c_s$ (GP)': [
            coherent_amplitude(C_S * f, N_0, gp_compatible_xi(C_S * f), L_COH, BASELINE_PHONONS)
            for f in factors
        ],
    }
    return factors, curves


def simulate_horizon_profile(r_values, r_H, A, N_coherent, is_coherent=True):
    """
    Simulate density modulation profile near sonic horizon.

    Parameters:
        r_values: Radial positions (m)
        r_H: Horizon radius (m)
        A: Vortex strength (m²/s)
        N_coherent: Total phonon number
        is_coherent: True for coherent state, False for thermal

    Returns:
        delta_rho_over_rho: Density modulation at each position
    """
    if is_coherent:
        return simulate_coherent_profile_with_params(
            r_values, r_H, C_S, N_0, XI, L_COH, N_coherent
        )
    return np.zeros_like(r_values)


def run_simulation():
    """Run the full BEC sonic horizon simulation."""

    # Experimental parameters
    r_H = R_H
    A = C_S * r_H    # Vortex strength set so |v| = c_s at r_H
    N_phonons = BASELINE_PHONONS

    # Spatial grid: from 0.5*r_H to 2*r_H
    r_min = 0.2 * r_H
    r_max = 3.0 * r_H
    r_values = np.linspace(r_min, r_max, 500)

    # Compute Hawking temperature
    T_H = hawking_temperature(A, r_H)
    omega_typical = C_S / XI
    n_bar = thermal_occupation(omega_typical, T_H)

    print("BEC Sonic Horizon Simulation")
    print("=" * 50)
    print(f"Parameters:")
    print(f"  Sound speed c_s = {C_S*1e3:.2f} mm/s")
    print(f"  Number density n₀ = {N_0:.1e} m⁻³")
    print(f"  Mass density ρ₀ = {RHO_MASS_0:.2e} kg/m³")
    print(f"  Healing length ξ = {XI*1e6:.2f} μm")
    print(f"  Coherence length L_coh = {L_COH*1e6:.1f} μm")
    print(f"  Horizon radius r_H = {r_H*1e6:.1f} μm")
    print(f"  Hawking temperature T_H = {T_H*1e9:.2f} nK")
    print(f"  Thermal occupation n̄ = {n_bar:.2f}")
    print(f"  Effective coupling κ_eff = {KAPPA_EFF:.2e}")
    print(f"  Injected phonons N = {N_phonons}")
    print()
    print("Model status: illustrative scaling implementation, not a GP/BdG solver.")
    print("Baseline is GP-compatible: xi = hbar / (sqrt(2) m c_s).")
    print()

    # Simulate coherent phonon injection
    delta_rho_coherent = simulate_horizon_profile(
        r_values, r_H, A, N_phonons, is_coherent=True
    )

    # Simulate thermal phonon injection
    delta_rho_thermal = simulate_horizon_profile(
        r_values, r_H, A, N_phonons, is_coherent=False
    )

    # Print key results
    max_coherent = np.max(np.abs(delta_rho_coherent))
    max_thermal = np.max(np.abs(delta_rho_thermal))

    print("Results:")
    print(f"  Coherent injection: max |δρ/ρ₀| = {max_coherent:.2e}")
    print(f"  Thermal injection:  max |δρ/ρ₀| = {max_thermal:.2e}")
    print(f"  Ratio (coherent/thermal): {'∞ (thermal = 0)' if max_thermal == 0 else f'{max_coherent/max_thermal:.1f}'}")
    print()
    print(f"  Paper prediction: δρ/ρ₀ ~ 10⁻⁶ ✓" if 1e-7 < max_coherent < 1e-5 else
          f"  Note: Result {max_coherent:.2e} differs from 10⁻⁶ estimate")

    return {
        'r': r_values,
        'r_H': r_H,
        'delta_rho_coherent': delta_rho_coherent,
        'delta_rho_thermal': delta_rho_thermal,
        'T_H': T_H,
        'n_bar': n_bar,
        'kappa_eff': KAPPA_EFF,
        'N_phonons': N_phonons
    }


def plot_results(results, save_path=None):
    """Generate publication-quality figure."""
    if save_path is None:
        save_path = BASE_DIR / 'bec_sonic_horizon_results.png'

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    r = results['r'] * 1e6  # Convert to μm
    r_H = results['r_H'] * 1e6

    # Left panel: Density modulation profile
    ax1.plot(r, np.abs(results['delta_rho_coherent']), 'b-', linewidth=2,
             label='Coherent phonons (Coherism)')
    ax1.plot(r, np.abs(results['delta_rho_thermal']), color='gray', linestyle='--', linewidth=2,
             label='Thermal / baseline (null)')
    ax1.axhline(y=1e-6, color='b', linestyle='--', alpha=0.5,
                label=r'Target: $10^{-6}$')
    ax1.axhline(y=1e-7, color='r', linestyle=':', alpha=0.7,
                label=r'Falsify: $10^{-7}$')
    ax1.axvline(x=r_H, color='k', linestyle='-', alpha=0.3, linewidth=2)
    ax1.text(r_H + 2, 1.6e-6, 'Horizon', fontsize=10, alpha=0.7)

    ax1.set_xlabel(r'Radial position $r$ ($\mu$m)', fontsize=12)
    ax1.set_ylabel(r'$|\delta\rho/\rho_0|$', fontsize=12)
    ax1.set_title('Density Modulation Near Sonic Horizon', fontsize=12)
    ax1.set_ylim(0, 2.2e-6)
    ax1.set_xlim(r[0], r[-1])
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Right panel: one-at-a-time robustness scan
    factors, curves = robustness_scan_data()
    styles = [
        ('tab:blue', r'$n_0$'),
        ('tab:orange', r'$\xi$'),
        ('tab:green', r'$L_{\mathrm{coh}}$'),
        ('tab:red', r'$N_{\mathrm{phonon}}$'),
        ('tab:purple', r'$c_s$ (GP)'),
    ]
    for color, label in styles:
        ax2.plot(factors, curves[label], linewidth=2, color=color, label=label)

    ax2.set_yscale('log')
    ax2.set_xlabel('Scale Factor Relative to Baseline', fontsize=12)
    ax2.set_ylabel(r'$A = \max_{|r-r_H|\leq L_{\mathrm{coh}}} |\delta\rho/\rho_0|$', fontsize=11)
    ax2.set_title('Robustness Around Baseline', fontsize=12)
    ax2.set_xlim(0.75, 1.25)
    ax2.set_ylim(1e-7, 2e-5)

    ax2.axhline(y=1e-6, color='blue', linestyle='--', alpha=0.5)
    ax2.axhline(y=1e-7, color='red', linestyle=':', alpha=0.7)
    ax2.axvline(x=1.0, color='k', linestyle='-', alpha=0.2)
    ax2.text(1.24, 1.05e-6, r'$10^{-6}$ target', fontsize=9, color='blue', ha='right')
    ax2.text(1.24, 1.15e-7, r'$10^{-7}$ falsify', fontsize=9, color='red', ha='right')

    ax2.legend(fontsize=8, loc='upper left')
    ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Figure saved to {save_path}")


def save_data(results, data_path=None, robustness_path=None):
    """Save profile and robustness data to files."""
    if data_path is None:
        data_path = BASE_DIR / 'bec_sonic_horizon_data.dat'
    if robustness_path is None:
        robustness_path = BASE_DIR / 'bec_sonic_horizon_robustness.dat'
    np.savetxt(
        data_path,
        np.column_stack([
            results['r'],
            results['delta_rho_coherent'],
            results['delta_rho_thermal']
        ]),
        header='r(m) delta_rho_coherent delta_rho_thermal',
        fmt='%.6e'
    )
    print(f"Data saved to {data_path}")

    factors, curves = robustness_scan_data()
    curve_order = [r'$n_0$', r'$\xi$', r'$L_{\mathrm{coh}}$', r'$N_{\mathrm{phonon}}$', r'$c_s$ (GP)']
    np.savetxt(
        robustness_path,
        np.column_stack([factors] + [np.asarray(curves[key]) for key in curve_order]),
        header='scale_factor n0_scan xi_scan Lcoh_scan Nphonon_scan cs_gp_scan',
        fmt='%.6e'
    )
    print(f"Robustness data saved to {robustness_path}")


def main():
    """Run simulation and generate outputs."""
    results = run_simulation()
    plot_results(results)
    save_data(results)
    print("\nSimulation complete.")


if __name__ == "__main__":
    main()
