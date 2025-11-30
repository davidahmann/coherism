#!/usr/bin/env python3
"""
Coherism FRW Simulation
=======================
Coupled evolution of FRW cosmology with coherence feedback.

This implements the equations from Appendix: Numerical Simulation.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Physical constants (natural units: c = hbar = 1)
G_N = 1.0  # Newton's constant (normalized)
M_P = 1.0 / np.sqrt(8 * np.pi * G_N)  # Reduced Planck mass

# Model parameters
# Physical coupling is Planck-suppressed: kappa ~ G_N/L_coh^2 ~ 10^-70
# In normalized units, we use alpha ~ 10^-10 to represent this suppression
# (capturing the key physics that coherence has negligible gravitational effect)
ALPHA = 1e-11  # Coherence functional coefficient (Planck-suppressed)
V_D = 1.0      # Diamond volume (normalized)
GAMMA_DEC = 0.01  # Decoherence rate (in units of H)

# Initial conditions
RHO_M_0 = 0.3  # Matter density today (normalized)
RHO_R_0 = 1e-4  # Radiation density today
RHO_LAMBDA = 0.7  # Cosmological constant

def friedmann_H2(a, rho_m_0, rho_r_0, rho_lambda, rho_coh):
    """Compute H^2 from Friedmann equation."""
    rho_m = rho_m_0 * a**(-3)
    rho_r = rho_r_0 * a**(-4)
    rho_total = rho_m + rho_r + rho_lambda + rho_coh
    return (8 * np.pi * G_N / 3) * rho_total

def coherence_density(H, N_sq, alpha=ALPHA, V_D=V_D):
    """Compute coherence energy density from squeezed occupation."""
    return alpha * H**4 / (4 * np.pi**2 * V_D) * N_sq

def equation_of_state(N_sq, dN_sq_dlna):
    """Compute w_coh = p_coh / rho_coh."""
    if N_sq < 1e-10:
        return -1.0
    return -1.0 + dN_sq_dlna / (3 * N_sq)

def derivatives(log_a, y, rho_m_0, rho_r_0, rho_lambda, gamma_dec, alpha):
    """
    Compute dy/d(log a) for the coupled system.
    
    y = [N_sq] (squeezed occupation number)
    """
    N_sq = max(y[0], 0)  # Ensure non-negative
    a = np.exp(log_a)
    
    # Compute H (iteratively with coherence feedback)
    # First approximation without coherence
    H2_approx = friedmann_H2(a, rho_m_0, rho_r_0, rho_lambda, 0)
    H_approx = np.sqrt(max(H2_approx, 1e-20))
    
    # Coherence density
    rho_coh = coherence_density(H_approx, N_sq, alpha)
    
    # Refined H with coherence
    H2 = friedmann_H2(a, rho_m_0, rho_r_0, rho_lambda, rho_coh)
    H = np.sqrt(max(H2, 1e-20))
    
    # Coherence evolution: dN_sq/d(log a) = source - decay
    # Source: 1 mode exits horizon per e-fold (during inflation)
    # Decay: decoherence at rate gamma_dec * H
    
    # Inflation check: H roughly constant means inflation
    # For simplicity, we model source as always present
    source = 1.0 if H > 0.1 else 0.1  # Reduced source post-inflation
    decay = gamma_dec * N_sq
    
    dN_sq_dlna = source - decay
    
    return [dN_sq_dlna]

def run_simulation(log_a_span=(-10, 2), n_points=1000):
    """
    Run the coupled FRW-coherence simulation.
    
    Parameters:
        log_a_span: tuple of (log(a_initial), log(a_final))
        n_points: number of output points
    
    Returns:
        dict with simulation results
    """
    # Initial conditions
    y0 = [0.0]  # Start with no squeezed modes
    
    # Solve ODE
    log_a_eval = np.linspace(log_a_span[0], log_a_span[1], n_points)
    
    sol = solve_ivp(
        derivatives,
        log_a_span,
        y0,
        args=(RHO_M_0, RHO_R_0, RHO_LAMBDA, GAMMA_DEC, ALPHA),
        t_eval=log_a_eval,
        method='RK45',
        max_step=0.1
    )
    
    # Extract results
    log_a = sol.t
    a = np.exp(log_a)
    N_sq = np.maximum(sol.y[0], 0)
    
    # Compute derived quantities
    H = np.zeros_like(a)
    rho_coh = np.zeros_like(a)
    w_coh = np.zeros_like(a)
    rho_total = np.zeros_like(a)
    
    for i in range(len(a)):
        H2 = friedmann_H2(a[i], RHO_M_0, RHO_R_0, RHO_LAMBDA, 0)
        H[i] = np.sqrt(max(H2, 1e-20))
        rho_coh[i] = coherence_density(H[i], N_sq[i])
        
        # Finite difference for dN/dlna
        if i > 0 and i < len(a) - 1:
            dN_dlna = (N_sq[i+1] - N_sq[i-1]) / (log_a[i+1] - log_a[i-1])
        else:
            dN_dlna = 0
        w_coh[i] = equation_of_state(N_sq[i], dN_dlna)
        
        rho_m = RHO_M_0 * a[i]**(-3)
        rho_r = RHO_R_0 * a[i]**(-4)
        rho_total[i] = rho_m + rho_r + RHO_LAMBDA + rho_coh[i]
    
    return {
        'log_a': log_a,
        'a': a,
        'N_sq': N_sq,
        'H': H,
        'rho_coh': rho_coh,
        'w_coh': w_coh,
        'rho_total': rho_total,
        'rho_coh_frac': rho_coh / rho_total
    }

def plot_results(results, save_path='coherism_frw_results.png'):
    """Generate plots of simulation results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    log_a = results['log_a']
    
    # Panel 1: Squeezed occupation
    ax1 = axes[0, 0]
    ax1.plot(log_a, results['N_sq'], 'b-', linewidth=2)
    ax1.set_xlabel(r'$\log(a)$')
    ax1.set_ylabel(r'$\mathcal{N}_{\mathrm{sq}}$')
    ax1.set_title('Squeezed Mode Occupation')
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Coherence density fraction
    ax2 = axes[0, 1]
    ax2.semilogy(log_a, results['rho_coh_frac'], 'r-', linewidth=2)
    ax2.set_xlabel(r'$\log(a)$')
    ax2.set_ylabel(r'$\rho_{\mathrm{coh}} / \rho_{\mathrm{tot}}$')
    ax2.set_title('Coherence Energy Fraction')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(1e-15, 1)
    
    # Panel 3: Equation of state
    ax3 = axes[1, 0]
    ax3.plot(log_a, results['w_coh'], 'g-', linewidth=2)
    ax3.axhline(y=-1, color='k', linestyle='--', alpha=0.5, label=r'$w = -1$')
    ax3.axhline(y=0, color='k', linestyle=':', alpha=0.5, label=r'$w = 0$')
    ax3.set_xlabel(r'$\log(a)$')
    ax3.set_ylabel(r'$w_{\mathrm{coh}}$')
    ax3.set_title('Coherence Equation of State')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-1.5, 0.5)
    ax3.legend()
    
    # Panel 4: Hubble parameter
    ax4 = axes[1, 1]
    ax4.semilogy(log_a, results['H'], 'm-', linewidth=2)
    ax4.set_xlabel(r'$\log(a)$')
    ax4.set_ylabel(r'$H$')
    ax4.set_title('Hubble Parameter')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to {save_path}")

def main():
    """Run simulation and generate output."""
    print("Coherism FRW Simulation")
    print("=" * 50)
    print(f"Parameters: alpha={ALPHA}, gamma_dec={GAMMA_DEC}")
    print(f"Initial: rho_m={RHO_M_0}, rho_r={RHO_R_0}, rho_Lambda={RHO_LAMBDA}")
    print()
    
    # Run simulation
    # Paper claims 60 e-folds: log(a) from -60 to ~0
    print("Running simulation...")
    results = run_simulation(log_a_span=(-60, 2), n_points=1000)
    
    # Print summary
    print("\nResults at a = 1 (today):")
    idx_today = np.argmin(np.abs(results['a'] - 1.0))
    print(f"  N_sq = {results['N_sq'][idx_today]:.2f}")
    print(f"  rho_coh/rho_tot = {results['rho_coh_frac'][idx_today]:.2e}")
    print(f"  w_coh = {results['w_coh'][idx_today]:.4f}")
    
    # Generate plots
    print("\nGenerating plots...")
    plot_results(results)
    
    # Save data
    data_path = 'coherism_frw_data.dat'
    np.savetxt(
        data_path,
        np.column_stack([
            results['log_a'],
            results['a'],
            results['N_sq'],
            results['rho_coh_frac'],
            results['w_coh']
        ]),
        header='log_a a N_sq rho_coh_frac w_coh',
        fmt='%.6e'
    )
    print(f"Data saved to {data_path}")
    
    print("\nSimulation complete.")

if __name__ == "__main__":
    main()
