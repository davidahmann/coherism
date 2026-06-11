#!/usr/bin/env python3
"""
Illustrative acoustic-model scaling study for the Coherism manuscript.

This script mirrors the acoustic implementation in the analog-gravity appendix
of `coherism.tex`. It is not a full Gross-Pitaevskii/Bogoliubov simulation, but
it includes a direct static GP/BdG-style linear-response benchmark for the
density-response step.

The informational source is the exact Gaussian-state identity

    S(rho || sigma_beta) = beta (E_rho - E_sigma) - (S_vN(rho) - S_vN(sigma)),

i.e. relative entropy = free-energy excess over the Hawking-temperature
reference, in nats. Both the coherent and the energy-matched thermal injection
therefore source the informational stress; at matched injected energy they
differ exactly by the von Neumann entropy of the injected occupation,

    Delta A = kappa_eff * Delta S_vN * (profile peak).

The primary observable is this coherent-minus-thermal differential.

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
L_COH = 10e-6     # Injected packet envelope width: 10 μm
CHI_AN = 3.0      # O(1) acoustic calibration constant for the ansatz
RHO_MASS_0 = N_0 * M_RB
R_H = 50e-6       # Horizon radius: 50 μm

# Horizon steepness: surface gravity kappa_H = |d(c_s - v)/dr| at the horizon.
# kappa_H = 8.2e2 1/s corresponds to a gradient length c_s/kappa_H ~ 2.1 μm
# (steep, Steinhauer-class horizon) and T_H ~ 1.0 nK.
KAPPA_H = 8.2e2   # 1/s

# Injection design (soft, broadband band centred on the Hawking scale)
BASELINE_PHONONS = 1000   # total injected phonon number N
BASELINE_MODES = 100      # number of populated modes M in the band
X_BAR = 2.0               # band-averaged hbar*omega / (k_B T_H)

# Derived quantities
KAPPA_EFF = CHI_AN * (1.0 / (N_0 * L_COH**3)) * (XI / L_COH) ** 3


def gp_compatible_xi(c_s):
    """Healing length implied by the GP relation for a given sound speed."""
    return HBAR / (np.sqrt(2) * M_RB * c_s)


def compute_kappa_eff(n_0, xi, l_coh):
    """Dimensionless phenomenological acoustic coupling used in the manuscript."""
    return CHI_AN * (1.0 / (n_0 * l_coh**3)) * (xi / l_coh) ** 3


def hawking_temperature(kappa_h):
    """Acoustic Hawking temperature T_H = hbar kappa_H / (2 pi k_B)."""
    return HBAR * kappa_h / (2 * np.pi * KB)


def thermal_occupation(x):
    """Bose-Einstein occupation for x = hbar*omega / (k_B T_H)."""
    if x <= 0:
        return 0.0
    if x > 100:
        return 0.0
    return 1.0 / (np.exp(x) - 1.0)


def mode_entropy(n):
    """Von Neumann entropy (nats) of a single-mode thermal state with mean n."""
    if n <= 0:
        return 0.0
    return (1.0 + n) * np.log(1.0 + n) - n * np.log(n)


def s_rel_coherent(n_phonons, x_bar):
    """
    Exact relative entropy (nats) of a displaced thermal state w.r.t. the
    Hawking-temperature reference: S = beta * Delta E = N * x_bar.
    Displacement adds energy but leaves the entropy unchanged.
    """
    return n_phonons * x_bar


def delta_s_vn(n_phonons, n_modes, n_bar):
    """
    Von Neumann entropy increase (nats) of a thermal injection that raises the
    occupation of M modes from n_bar to n_bar + N/M.
    """
    n_per_mode = n_phonons / n_modes
    return n_modes * (mode_entropy(n_bar + n_per_mode) - mode_entropy(n_bar))


def s_rel_thermal(n_phonons, n_modes, x_bar, n_bar):
    """
    Exact relative entropy (nats) of the energy-matched thermal injection:
    S = beta * Delta E - Delta S_vN.
    """
    return n_phonons * x_bar - delta_s_vn(n_phonons, n_modes, n_bar)


def bdg_static_density_response(source_profile, dx, xi, pad_factor=8):
    """
    Static GP/BdG linear-response benchmark for a specified source profile.

    In Fourier space the benchmark response is
        delta_rho_k / rho_0 = source_k / (1 + 0.5 * (k * xi)^2),
    which is the standard linearized GP response including the leading
    quantum-pressure correction.
    """
    n_points = len(source_profile)
    n_pad = pad_factor * n_points
    padded = np.zeros(n_pad)
    start = (n_pad - n_points) // 2
    padded[start:start + n_points] = source_profile
    k_values = 2.0 * np.pi * np.fft.fftfreq(n_pad, d=dx)
    filter_kernel = 1.0 / (1.0 + 0.5 * (k_values * xi) ** 2)
    response = np.fft.ifft(np.fft.fft(padded) * filter_kernel).real
    return response[start:start + n_points]


def envelope(r_values, r_h, xi, l_coh):
    """Normalized injected-packet envelope centred on the horizon."""
    distance = np.maximum(np.abs(r_values - r_h), xi)
    return np.exp(-distance / l_coh)


def source_profile(r_values, s_rel, r_h, c_s, n_0, xi, l_coh):
    """
    Dimensionless informational source S(r) = Theta_tt / (rho_0 c_s^2)
    for an injection with total relative entropy s_rel (nats).
    """
    kappa_eff = compute_kappa_eff(n_0, xi, l_coh)
    return kappa_eff * s_rel * envelope(r_values, r_h, xi, l_coh)


def max_modulation_within_lcoh(r_vals_m, delta_rho_over_rho, r_h_m, l_coh_m):
    """Peak signal inside one coherence length of the horizon."""
    mask = np.abs(r_vals_m - r_h_m) <= l_coh_m
    return float(np.max(np.abs(delta_rho_over_rho[mask])))


def observables(c_s, n_0, xi, l_coh, n_phonons, n_modes, kappa_h,
                x_bar=X_BAR, r_h=R_H, num_points=500, use_bdg=False):
    """
    Compute (A_coh, A_th, Delta A) for a parameter set.

    A_coh, A_th: peak |delta rho / rho_0| within L_coh of the horizon for the
    coherent and energy-matched thermal injections. Delta A = A_coh - A_th
    is the primary observable.
    """
    n_bar = thermal_occupation(x_bar)
    s_coh = s_rel_coherent(n_phonons, x_bar)
    s_th = s_rel_thermal(n_phonons, n_modes, x_bar, n_bar)
    r_vals = np.linspace(0.2 * r_h, 3.0 * r_h, num_points)
    prof_coh = source_profile(r_vals, s_coh, r_h, c_s, n_0, xi, l_coh)
    prof_th = source_profile(r_vals, s_th, r_h, c_s, n_0, xi, l_coh)
    if use_bdg:
        dx = r_vals[1] - r_vals[0]
        prof_coh = bdg_static_density_response(prof_coh, dx, xi)
        prof_th = bdg_static_density_response(prof_th, dx, xi)
    a_coh = max_modulation_within_lcoh(r_vals, prof_coh, r_h, l_coh)
    a_th = max_modulation_within_lcoh(r_vals, prof_th, r_h, l_coh)
    return a_coh, a_th, a_coh - a_th


def baseline_observables(use_bdg=False):
    return observables(C_S, N_0, XI, L_COH, BASELINE_PHONONS, BASELINE_MODES,
                       KAPPA_H, use_bdg=use_bdg)


def robustness_scan_data():
    """One-at-a-time parameter scans of the differential around the baseline."""
    factors = np.linspace(0.75, 1.25, 11)

    def scan(param):
        rows = []
        for f in factors:
            kwargs = dict(c_s=C_S, n_0=N_0, xi=XI, l_coh=L_COH,
                          n_phonons=BASELINE_PHONONS, n_modes=BASELINE_MODES,
                          kappa_h=KAPPA_H, x_bar=X_BAR)
            if param == 'n_0':
                kwargs['n_0'] = N_0 * f
            elif param == 'xi':
                kwargs['xi'] = XI * f
            elif param == 'l_coh':
                kwargs['l_coh'] = L_COH * f
            elif param == 'n_phonons':
                kwargs['n_phonons'] = BASELINE_PHONONS * f
            elif param == 'n_modes':
                kwargs['n_modes'] = max(1, round(BASELINE_MODES * f))
            elif param == 't_h':
                # fixed injection band omega; T_H scales -> x_bar rescales
                kwargs['x_bar'] = X_BAR / f
            elif param == 'c_s_gp':
                kwargs['c_s'] = C_S * f
                kwargs['xi'] = gp_compatible_xi(C_S * f)
            rows.append(observables(**kwargs))
        return np.asarray(rows)

    labels = {
        'n_0': r'$n_0$',
        'xi': r'$\xi$',
        'l_coh': r'$L_{\mathrm{coh}}$',
        'n_phonons': r'$N_{\mathrm{phonon}}$',
        'n_modes': r'$M$',
        't_h': r'$T_H$',
        'c_s_gp': r'$c_s$ (GP)',
    }
    scans = {labels[p]: scan(p) for p in labels}
    return factors, scans


def bdg_shift_summary():
    """Relative GP/BdG-vs-hydro shift of the differential at baseline and in scans."""
    _, _, d_hydro = baseline_observables(use_bdg=False)
    _, _, d_bdg = baseline_observables(use_bdg=True)
    baseline_shift = abs(d_bdg - d_hydro) / d_hydro

    factors = np.linspace(0.75, 1.25, 5)
    max_shift = baseline_shift
    for f in factors:
        for kwargs in (
            dict(xi=XI * f),
            dict(l_coh=L_COH * f),
        ):
            base = dict(c_s=C_S, n_0=N_0, xi=XI, l_coh=L_COH,
                        n_phonons=BASELINE_PHONONS, n_modes=BASELINE_MODES,
                        kappa_h=KAPPA_H)
            base.update(kwargs)
            _, _, dh = observables(**base, use_bdg=False)
            _, _, db = observables(**base, use_bdg=True)
            max_shift = max(max_shift, abs(db - dh) / dh)
    return baseline_shift, max_shift


def run_simulation():
    """Run the full BEC sonic horizon study."""
    T_H = hawking_temperature(KAPPA_H)
    n_bar = thermal_occupation(X_BAR)
    n_per_mode = BASELINE_PHONONS / BASELINE_MODES
    omega_bar = X_BAR * KB * T_H / HBAR
    k_bar = omega_bar / C_S
    s_coh = s_rel_coherent(BASELINE_PHONONS, X_BAR)
    ds_vn = delta_s_vn(BASELINE_PHONONS, BASELINE_MODES, n_bar)
    s_th = s_coh - ds_vn

    print("BEC Sonic Horizon Study (free-energy informational source)")
    print("=" * 60)
    print("Parameters:")
    print(f"  Sound speed c_s = {C_S*1e3:.2f} mm/s")
    print(f"  Number density n₀ = {N_0:.1e} m⁻³")
    print(f"  Mass density ρ₀ = {RHO_MASS_0:.2e} kg/m³")
    print(f"  Healing length ξ = {XI*1e6:.2f} μm")
    print(f"  Envelope width L_coh = {L_COH*1e6:.1f} μm")
    print(f"  Horizon radius r_H = {R_H*1e6:.1f} μm")
    print(f"  Surface gravity κ_H = {KAPPA_H:.1e} s⁻¹ (gradient length "
          f"{C_S/KAPPA_H*1e6:.1f} μm)")
    print(f"  Hawking temperature T_H = {T_H*1e9:.2f} nK")
    print(f"  Injection band: x̄ = ħω̄/k_B T_H = {X_BAR:.1f} "
          f"(ω̄ = {omega_bar:.0f} rad/s, k̄ξ = {k_bar*XI:.3f})")
    print(f"  Reference occupation n̄(x̄) = {n_bar:.3f}")
    print(f"  Injected phonons N = {BASELINE_PHONONS}, modes M = {BASELINE_MODES}, "
          f"n' = N/M = {n_per_mode:.1f}")
    print(f"  Dimensionless coupling κ_eff = {KAPPA_EFF:.2e}")
    print()
    print("Informational source (exact Gaussian-state identity, nats):")
    print(f"  ŝ_coh = β ΔE             = {s_coh:.0f}")
    print(f"  ΔS_vN (thermal injection) = {ds_vn:.1f}")
    print(f"  ŝ_th  = β ΔE − ΔS_vN      = {s_th:.0f}")
    print(f"  Differential fraction ΔS_vN/(β ΔE) = {ds_vn/s_coh:.1%}")
    print()
    print("Model status: illustrative scaling implementation, not a GP/BdG solver.")
    print("Baseline is GP-compatible: xi = hbar / (sqrt(2) m c_s).")
    print()

    a_coh, a_th, d_a = baseline_observables(use_bdg=False)
    a_coh_bdg, a_th_bdg, d_a_bdg = baseline_observables(use_bdg=True)
    baseline_shift, max_scan_shift = bdg_shift_summary()

    r_vals = np.linspace(0.2 * R_H, 3.0 * R_H, 500)
    prof_coh = source_profile(r_vals, s_coh, R_H, C_S, N_0, XI, L_COH)
    prof_th = source_profile(r_vals, s_th, R_H, C_S, N_0, XI, L_COH)
    dx = r_vals[1] - r_vals[0]
    prof_coh_bdg = bdg_static_density_response(prof_coh, dx, XI)

    print("Results:")
    print(f"  Coherent injection:  A_coh = {a_coh:.2e}  (GP/BdG: {a_coh_bdg:.2e})")
    print(f"  Thermal injection:   A_th  = {a_th:.2e}  (GP/BdG: {a_th_bdg:.2e})")
    print(f"  PRIMARY OBSERVABLE:  ΔA = A_coh − A_th = {d_a:.2e}")
    print(f"  ΔA (GP/BdG benchmark)                  = {d_a_bdg:.2e}")
    print(f"  GP/BdG shift of ΔA at baseline = {baseline_shift:.2%}")
    print(f"  Max GP/BdG shift of ΔA in scans = {max_scan_shift:.2%}")
    print(f"  Falsification threshold for ΔA: 5×10⁻⁸ "
          f"({'✓ prediction above threshold' if d_a > 5e-8 else '✗ below threshold'})")

    return {
        'r': r_vals,
        'r_H': R_H,
        'profile_coherent': prof_coh,
        'profile_coherent_bdg': prof_coh_bdg,
        'profile_thermal': prof_th,
        'profile_differential': prof_coh - prof_th,
        'T_H': T_H,
        'n_bar': n_bar,
        'kappa_eff': KAPPA_EFF,
        'N_phonons': BASELINE_PHONONS,
        'M_modes': BASELINE_MODES,
        's_coh': s_coh,
        's_th': s_th,
        'ds_vn': ds_vn,
        'A_coh': a_coh,
        'A_th': a_th,
        'Delta_A': d_a,
        'Delta_A_bdg': d_a_bdg,
        'benchmark_peak_shift': baseline_shift,
        'max_scan_shift': max_scan_shift,
    }


def plot_results(results, save_path=None):
    """Generate publication-quality figure."""
    if save_path is None:
        save_path = BASE_DIR / 'bec_sonic_horizon_results.png'

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    r = results['r'] * 1e6
    r_H = results['r_H'] * 1e6

    ax1.plot(r, np.abs(results['profile_coherent']), 'b-', linewidth=2,
             label=r'Coherent: $\hat{s}=\beta\,\Delta E$')
    ax1.plot(r, np.abs(results['profile_thermal']), color='gray', linestyle='--',
             linewidth=2, label=r'Thermal: $\hat{s}=\beta\,\Delta E-\Delta S_{\mathrm{vN}}$')
    ax1.plot(r, np.abs(results['profile_differential']), 'g-', linewidth=2,
             label=r'Differential $\Delta(\delta\rho/\rho_0)$')
    ax1.plot(r, np.abs(results['profile_coherent_bdg']), color='tab:orange',
             linestyle='-.', linewidth=1.5, label='Coherent (GP/BdG benchmark)')
    ax1.axhline(y=5e-8, color='r', linestyle=':', alpha=0.7,
                label=r'Falsify $\Delta A$: $5\times10^{-8}$')
    ax1.axvline(x=r_H, color='k', linestyle='-', alpha=0.3, linewidth=2)
    ax1.text(r_H + 2, 1.7e-6, 'Horizon', fontsize=10, alpha=0.7)

    ax1.set_xlabel(r'Radial position $r$ ($\mu$m)', fontsize=12)
    ax1.set_ylabel(r'$|\delta\rho/\rho_0|$', fontsize=12)
    ax1.set_title('Density Modulation Near Sonic Horizon', fontsize=12)
    ax1.set_yscale('log')
    ax1.set_ylim(1e-8, 3e-6)
    ax1.set_xlim(r[0], r[-1])
    ax1.legend(loc='lower left', fontsize=8)
    ax1.grid(True, alpha=0.3, which='both')

    factors, scans = robustness_scan_data()
    styles = [
        ('tab:blue', r'$n_0$'),
        ('tab:orange', r'$\xi$'),
        ('tab:green', r'$L_{\mathrm{coh}}$'),
        ('tab:red', r'$N_{\mathrm{phonon}}$'),
        ('tab:brown', r'$M$'),
        ('tab:pink', r'$T_H$'),
        ('tab:purple', r'$c_s$ (GP)'),
    ]
    for color, label in styles:
        ax2.plot(factors, scans[label][:, 2], linewidth=2, color=color, label=label)

    ax2.set_yscale('log')
    ax2.set_xlabel('Scale Factor Relative to Baseline', fontsize=12)
    ax2.set_ylabel(r'$\Delta A = A_{\mathrm{coh}} - A_{\mathrm{th}}$', fontsize=11)
    ax2.set_title('Robustness of the Differential', fontsize=12)
    ax2.set_xlim(0.75, 1.25)
    ax2.set_ylim(2e-8, 3e-6)

    ax2.axhline(y=5e-8, color='red', linestyle=':', alpha=0.7)
    ax2.axvline(x=1.0, color='k', linestyle='-', alpha=0.2)
    ax2.text(1.24, 5.6e-8, r'$5\times10^{-8}$ falsify', fontsize=9,
             color='red', ha='right')

    ax2.legend(fontsize=8, loc='upper left', ncol=2)
    ax2.grid(True, alpha=0.3, which='both')
    ax2.text(
        0.76,
        2.6e-8,
        (
            f"GP/BdG shift of ΔA:\n"
            f"baseline = {results['benchmark_peak_shift']*100:.2f}%\n"
            f"all scans < {results['max_scan_shift']*100:.2f}%"
        ),
        fontsize=8,
        bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.85,
              'edgecolor': '0.7'}
    )

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
            results['profile_coherent'],
            results['profile_coherent_bdg'],
            results['profile_thermal'],
            results['profile_differential'],
        ]),
        header=('r(m) delta_rho_coherent_hydro delta_rho_coherent_bdg '
                'delta_rho_thermal delta_rho_differential'),
        fmt='%.6e'
    )
    print(f"Data saved to {data_path}")

    factors, scans = robustness_scan_data()
    order = [r'$n_0$', r'$\xi$', r'$L_{\mathrm{coh}}$', r'$N_{\mathrm{phonon}}$',
             r'$M$', r'$T_H$', r'$c_s$ (GP)']
    columns = [factors]
    for key in order:
        columns.append(scans[key][:, 0])  # A_coh
        columns.append(scans[key][:, 2])  # Delta A
    np.savetxt(
        robustness_path,
        np.column_stack(columns),
        header=('scale_factor ' + ' '.join(
            f'{name}_Acoh {name}_DeltaA' for name in
            ['n0', 'xi', 'Lcoh', 'Nphonon', 'M', 'TH', 'cs_gp'])),
        fmt='%.6e'
    )
    print(f"Robustness data saved to {robustness_path}")


def main():
    results = run_simulation()
    plot_results(results)
    save_data(results)

    factors, scans = robustness_scan_data()
    d_min = min(float(np.min(s[:, 2])) for s in scans.values())
    d_max = max(float(np.max(s[:, 2])) for s in scans.values())
    print(f"\nΔA range across ±25% one-at-a-time scans: "
          f"[{d_min:.2e}, {d_max:.2e}]")
    print("Simulation complete.")


if __name__ == "__main__":
    main()
