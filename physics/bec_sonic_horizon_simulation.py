#!/usr/bin/env python3
"""
BEC Sonic Horizon Simulation
=============================
Simulates the coherism prediction for density modulation near a sonic horizon
in a Bose-Einstein condensate.

Key prediction: Coherent phonon injection produces δρ/ρ₀ ~ 10⁻⁶,
while thermal phonons produce no modulation.

This implements equations from the coherism.tex manuscript:
- Eq. (N.7): κ_eff effective coupling
- Eq. (N.8): Θ_tt informational stress tensor  
- Eq. (N.9): δρ/ρ₀ density modulation

See Appendix N (Analog Gravity Predictions) for derivations.
"""

import numpy as np
import matplotlib.pyplot as plt

# Physical constants for 87Rb BEC
HBAR = 1.055e-34  # J·s
KB = 1.381e-23    # J/K
M_RB = 1.44e-25   # kg (87Rb mass)

# BEC parameters (typical experimental values)
C_S = 1e-3        # Sound speed: 1 mm/s
RHO_0 = 1e20      # Background density: 10^14 cm^-3 = 10^20 m^-3
XI = 0.3e-6       # Healing length: 0.3 μm
L_COH = 10e-6     # Coherence length: 10 μm
ALPHA = 1.0       # O(1) coupling constant

# Derived quantities
KAPPA_EFF = ALPHA * HBAR * C_S / (RHO_0 * M_RB * XI**2 * L_COH**2)
# Paper predicts κ_eff ~ 10^-8


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
    return theta_tt / (RHO_0 * M_RB * C_S**2)


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
    T_H = hawking_temperature(A, r_H)
    omega_typical = C_S / XI  # Typical phonon frequency
    n_bar = thermal_occupation(omega_typical, T_H)

    delta_rho = np.zeros_like(r_values)

    for i, r in enumerate(r_values):
        # Modulation is strongest near horizon, falls off as 1/|r - r_H|
        distance = np.abs(r - r_H)
        if distance < XI:
            distance = XI  # Regularize at healing length

        # Spatial profile: peaks at horizon, decays over coherence length
        profile = np.exp(-distance / L_COH)

        if is_coherent:
            # Coherent state: |α|² = N
            alpha_sq = N_coherent * profile
            theta = theta_tt_coherent(alpha_sq, n_bar)
        else:
            # Thermal state: S(σ||σ) = 0, no informational stress
            theta = theta_tt_thermal(n_bar * profile)

        delta_rho[i] = density_modulation(theta)

    return delta_rho


def run_simulation():
    """Run the full BEC sonic horizon simulation."""

    # Experimental parameters
    r_H = 50e-6      # Horizon radius: 50 μm
    A = C_S * r_H    # Vortex strength set so |v| = c_s at r_H
    N_phonons = 1000 # Number of injected phonons

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
    print(f"  Sound speed c_s = {C_S*1e3:.1f} mm/s")
    print(f"  Density ρ₀ = {RHO_0:.1e} m⁻³")
    print(f"  Healing length ξ = {XI*1e6:.2f} μm")
    print(f"  Coherence length L_coh = {L_COH*1e6:.1f} μm")
    print(f"  Horizon radius r_H = {r_H*1e6:.1f} μm")
    print(f"  Hawking temperature T_H = {T_H*1e9:.2f} nK")
    print(f"  Thermal occupation n̄ = {n_bar:.2f}")
    print(f"  Effective coupling κ_eff = {KAPPA_EFF:.2e}")
    print(f"  Injected phonons N = {N_phonons}")
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


def plot_results(results, save_path='bec_sonic_horizon_results.png'):
    """Generate publication-quality figure."""

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

    # Right panel: Scaling with injected phonon number + simple grid check
    def max_modulation_within_Lcoh(r_vals_m, delta_rho_over_rho, r_h_m, l_coh_m):
        mask = np.abs(r_vals_m - r_h_m) <= l_coh_m
        return float(np.max(np.abs(delta_rho_over_rho[mask])))

    r_h_m = results['r_H']
    a = C_S * r_h_m
    n_values = np.unique(np.round(np.logspace(2, 4, 9)).astype(int))
    grid_sizes = [200, 500, 2000]

    for n_grid in grid_sizes:
        r_vals = np.linspace(0.2 * r_h_m, 3.0 * r_h_m, n_grid)
        a_vals = []
        for n_ph in n_values:
            d_coh = simulate_horizon_profile(r_vals, r_h_m, a, n_ph, is_coherent=True)
            a_vals.append(max_modulation_within_Lcoh(r_vals, d_coh, r_h_m, L_COH))
        ax2.plot(n_values, a_vals, marker='o', linewidth=2, label=f'{n_grid} grid pts')

    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel(r'Injected phonons $N_{\mathrm{phonon}}$', fontsize=12)
    ax2.set_ylabel(r'$A = \max_{|r-r_H|\leq L_{\mathrm{coh}}} |\delta\rho/\rho_0|$', fontsize=11)
    ax2.set_title('Scaling + Grid Convergence Check', fontsize=12)
    ax2.set_ylim(1e-10, 1e-4)

    ax2.axhline(y=1e-6, color='blue', linestyle='--', alpha=0.5)
    ax2.axhline(y=1e-7, color='red', linestyle=':', alpha=0.7)
    ax2.text(1.4e4, 1.05e-6, r'$10^{-6}$ target', fontsize=9, color='blue', ha='right')
    ax2.text(1.4e4, 1.15e-7, r'$10^{-7}$ falsify', fontsize=9, color='red', ha='right')

    ax2.legend(fontsize=9, loc='upper left')
    ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Figure saved to {save_path}")


def save_data(results, data_path='bec_sonic_horizon_data.dat'):
    """Save simulation data to file."""
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


def main():
    """Run simulation and generate outputs."""
    results = run_simulation()
    plot_results(results)
    save_data(results)
    print("\nSimulation complete.")


if __name__ == "__main__":
    main()
