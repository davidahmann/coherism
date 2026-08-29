#!/usr/bin/env python3
"""Stationary-response check and classical-field confound study.

This script goes one step beyond the static linear-response kernel used in
`bec_sonic_horizon_simulation.py`. It solves the 1D GPE

    i dpsi/dt = [ -(1/2) d^2/dx^2 + g |psi|^2 ] psi      (hbar = m = 1)

with g = n0 = 1 (so c_s = 1, xi = 1/sqrt(2)) and performs two studies:

1. KERNEL VALIDATION. The static density response to a weak external
   potential is computed by imaginary-time relaxation and compared with the
   analytic linearized-GP kernel

       delta n_k / n0 = -(delta U_k / g n0) / (1 + (k xi)^2 / 2),

   which is the response relation used for the observable in the manuscript.

2. LOCALIZED FIXED-N VALIDATION. The exact localized profile, periodic grid,
   and zero-mode convention used in the manuscript are checked against an
   independently relaxed stationary GPE solution.

3. ORDINARY-GP PREPARATION DIFFERENCE. Deterministic-phase and independently
   phase-randomized classical fields with the same amplitude spectrum are
   evolved in real time with no extra source. Their time-averaged density
   profiles are differenced. This is a seeded classical-field surrogate for a
   preparation-dependent nuisance, not a thermal density operator and not an
   exact full-nonlinear-GP energy match. It therefore diagnoses a confound but
   does not calibrate an experimental noise floor.

This is an illustrative 1D testbed with a uniform background and no horizon
flow. It neither validates a transonic geometry nor constrains the free
phenomenological coupling used in the manuscript.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RNG = np.random.default_rng(20260610)

# --- units: hbar = m = 1, g = 1, n0 = 1  ->  c_s = 1, xi = 1/sqrt(2)
G = 1.0
N0 = 1.0
C_S = np.sqrt(G * N0)
XI = 1.0 / np.sqrt(2.0 * G * N0)

# --- grid
NX = 2048
L = 400.0
DX = L / NX
X = np.arange(NX) * DX
K = 2.0 * np.pi * np.fft.fftfreq(NX, d=DX)
KIN_HALF = np.exp(-0.5j * (K ** 2) * 0.01)  # placeholder; rebuilt per dt


def split_step(psi, dt, n_steps, potential=None, record=None):
    """Standard Strang-split GPE propagator. Optionally records |psi|^2."""
    kin = np.exp(-0.25j * (K ** 2) * dt)
    profiles = []
    for step in range(n_steps):
        psi = np.fft.ifft(kin * np.fft.fft(psi))
        phase = G * np.abs(psi) ** 2
        if potential is not None:
            phase = phase + potential
        psi = psi * np.exp(-1j * phase * dt)
        psi = np.fft.ifft(kin * np.fft.fft(psi))
        if record is not None and step >= record[0] and (step - record[0]) % record[1] == 0:
            profiles.append(np.abs(psi) ** 2)
    return psi, profiles


def imaginary_time_ground_state(potential, dt=0.01, n_steps=4000):
    """Imaginary-time relaxation to the stationary state in `potential`."""
    psi = np.sqrt(N0) * np.ones(NX, dtype=complex)
    kin = np.exp(-0.25 * (K ** 2) * dt)
    for _ in range(n_steps):
        psi = np.fft.ifft(kin * np.fft.fft(psi)).real.astype(complex)
        psi = psi * np.exp(-(G * np.abs(psi) ** 2 + potential) * dt)
        psi = np.fft.ifft(kin * np.fft.fft(psi)).real.astype(complex)
        norm = np.sqrt(np.mean(np.abs(psi) ** 2) / N0)
        psi = psi / norm
    return psi


def kernel_validation():
    """Static GP response vs the analytic linearized kernel."""
    k_test = np.array([0.05, 0.1, 0.2, 0.5, 1.0, 1.5]) / XI  # k*xi values below
    v0 = 1e-3
    measured = []
    predicted = []
    for k in k_test:
        # commensurate wavenumber
        m_idx = max(1, round(k * L / (2 * np.pi)))
        k_c = 2 * np.pi * m_idx / L
        pot = v0 * np.cos(k_c * X)
        psi = imaginary_time_ground_state(pot)
        dn = np.abs(psi) ** 2 - N0
        # projection on cos(k_c x)
        amp = 2.0 * np.mean(dn * np.cos(k_c * X))
        measured.append(abs(amp) / N0)
        predicted.append((v0 / (G * N0)) / (1.0 + 0.5 * (k_c * XI) ** 2))
    return np.array([max(1, round(k * L / (2 * np.pi))) * 2 * np.pi / L * XI
                     for k in k_test]), np.array(measured), np.array(predicted)


def localized_fixed_number_validation(source_width=10.0, v0=1e-3):
    """Validate the manuscript's localized response at fixed particle number.

    Imaginary-time relaxation preserves the mean density. The analytic
    comparison therefore removes the source's zero Fourier mode, which is
    absorbed into the stationary chemical potential at fixed total number.
    """
    centered_x = (X + L / 2.0) % L - L / 2.0
    envelope = np.exp(-np.abs(centered_x) / source_width)
    potential = v0 * envelope
    psi = imaginary_time_ground_state(potential)
    measured = -(np.abs(psi) ** 2 - N0) / (v0 / G)

    source_k = np.fft.fft(envelope)
    source_k[0] = 0.0
    kernel = 1.0 / (1.0 + 0.5 * (K * XI) ** 2)
    predicted = np.fft.ifft(source_k * kernel).real

    peak = float(np.max(np.abs(predicted)))
    max_relative = float(np.max(np.abs(measured - predicted)) / peak)
    l2_relative = float(
        np.linalg.norm(measured - predicted) / np.linalg.norm(predicted)
    )
    return {
        'x': centered_x,
        'measured': measured,
        'predicted': predicted,
        'response_peak': peak,
        'max_relative': max_relative,
        'l2_relative': l2_relative,
        'source_mean': float(np.mean(envelope)),
        'v0': v0,
    }


def injection_field(amplitudes, k_modes, phases):
    """Small-amplitude multi-mode perturbation on the uniform background."""
    delta = np.zeros(NX, dtype=complex)
    for a, k, ph in zip(amplitudes, k_modes, phases):
        delta = delta + a * np.exp(1j * (k * X + ph))
    return (np.sqrt(N0) + delta).astype(complex)


def band_modes(n_modes, k_center=0.15 / XI, rel_width=0.4):
    """Commensurate wavenumbers in a band around k_center."""
    k_lo = k_center * (1 - rel_width)
    k_hi = k_center * (1 + rel_width)
    m_lo = int(np.ceil(k_lo * L / (2 * np.pi)))
    m_hi = int(np.floor(k_hi * L / (2 * np.pi)))
    m_all = np.linspace(m_lo, m_hi, n_modes)
    m_all = np.unique(np.round(m_all).astype(int))
    return 2 * np.pi * m_all / L


def time_averaged_profile(psi0, t_settle=40.0, t_window=160.0, dt=0.02):
    """Real-time evolution; returns density profile averaged over the window."""
    n_settle = int(t_settle / dt)
    n_total = int((t_settle + t_window) / dt)
    _, profiles = split_step(psi0, dt, n_total, record=(n_settle, 25))
    return np.mean(profiles, axis=0)


def confound_study(n_modes, total_eps2=2.5e-3, n_realizations=16):
    """
    Deterministic-phase vs phase-randomized fields at matched amplitude spectrum.

    total_eps2 sets the summed |amplitude|^2; first-order density modulation
    is delta n / n ~ 2 sqrt(total_eps2) ~ 0.1 at the default.
    """
    k_modes = band_modes(n_modes)
    amps = np.sqrt(total_eps2 / len(k_modes)) * np.ones(len(k_modes))

    psi_coh = injection_field(amps, k_modes, np.zeros(len(k_modes)))
    prof_coh = time_averaged_profile(psi_coh)

    prof_randomized = np.zeros(NX)
    for _ in range(n_realizations):
        phases = RNG.uniform(0, 2 * np.pi, len(k_modes))
        psi_randomized = injection_field(amps, k_modes, phases)
        prof_randomized += time_averaged_profile(psi_randomized)
    prof_randomized /= n_realizations

    first_order = 2.0 * np.sqrt(total_eps2)
    diff = (prof_coh - prof_randomized) / N0
    return {
        'k_modes': k_modes,
        'prof_coh': prof_coh,
        'prof_randomized': prof_randomized,
        'diff': diff,
        'max_diff': float(np.max(np.abs(diff))),
        'rms_diff': float(np.sqrt(np.mean(diff ** 2))),
        'first_order': first_order,
        'n_modes': len(k_modes),
    }


def main():
    print("1D GPE stationary-response and confound study")
    print("=" * 60)
    print(f"Grid: NX = {NX}, L = {L} (dx = {DX:.3f}), xi = {XI:.3f}, c_s = {C_S:.1f}")
    print()

    # --- Study 1: kernel validation
    print("Study 1: static GP response vs analytic linearized kernel")
    kxi, measured, predicted = kernel_validation()
    max_kernel_err = float(np.max(np.abs(measured - predicted) / predicted))
    for kx, m, p in zip(kxi, measured, predicted):
        print(f"  k·ξ = {kx:5.3f}:  measured {m:.4e}   kernel {p:.4e}   "
              f"dev {abs(m-p)/p:6.2%}")
    print(f"  Max deviation from kernel: {max_kernel_err:.2%}")
    print()

    # --- Study 2: ordinary-GP preparation difference at two bandwidths
    print("Study 2: localized fixed-N response on the manuscript grid")
    localized = localized_fixed_number_validation()
    print(f"  Fixed-N response peak: {localized['response_peak']:.9f}")
    print(f"  Maximum profile deviation / peak: {localized['max_relative']:.4%}")
    print(f"  Relative L2 deviation: {localized['l2_relative']:.4%}")
    print()

    # --- Study 3: generic preparation difference at two bandwidths
    print("Study 3: deterministic vs randomized phases (no extra source)")
    results = {}
    for n_modes in (4, 8):
        res = confound_study(n_modes)
        results[n_modes] = res
        print(f"  M = {res['n_modes']} modes: first-order δn/n ≈ {res['first_order']:.2e}; "
              f"time-averaged differential: max = {res['max_diff']:.2e}, "
              f"rms = {res['rms_diff']:.2e}")
    r4, r8 = results[4], results[8]
    ratio = r8['rms_diff'] / r4['rms_diff'] if r4['rms_diff'] > 0 else float('inf')
    print(f"  Confound rms ratio (M=8 / M=4) = {ratio:.2f}")
    print()
    print("Interpretation: this seeded ordinary-GP example is a generic control")
    print("motivation only. It is neither a thermal density operator, an exactly")
    print("energy-matched comparison, a calibrated nuisance model, nor an")
    print("experimental uncertainty budget.")

    # --- figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.6))

    ax1.plot(kxi, predicted, 'k-', linewidth=2, label='Linearized-GP kernel')
    ax1.plot(kxi, measured, 'o', color='tab:blue', markersize=7,
             label='Imaginary-time GPE')
    ax1.set_xlabel(r'$k\,\xi$', fontsize=12)
    ax1.set_ylabel(r'$|\delta n_k/n_0|$ for $V_0/(gn_0)=10^{-3}$', fontsize=10)
    ax1.set_title('Static response: GPE vs analytic kernel', fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    order = np.argsort(localized['x'])
    ax2.plot(localized['x'][order], localized['predicted'][order],
             color='black', linewidth=2, label='Fixed-N linear response')
    ax2.plot(localized['x'][order], localized['measured'][order],
             color='tab:orange', linestyle='--', linewidth=1.5,
             label='Stationary GPE')
    ax2.set_xlim(-50, 50)
    ax2.set_xlabel(r'$x$', fontsize=12)
    ax2.set_ylabel(r'$-(\delta n/n_0)/(V_0/gn_0)$', fontsize=10)
    ax2.set_title('Localized fixed-N validation', fontsize=12)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    ax3.plot(X, r8['diff'], color='tab:green', linewidth=1.2,
             label=f"M = {r8['n_modes']} modes")
    ax3.plot(X, r4['diff'], color='tab:purple', linewidth=1.2, alpha=0.7,
             label=f"M = {r4['n_modes']} modes")
    ax3.set_xlabel(r'$x/\xi_{\mathrm{unit}}$', fontsize=12)
    ax3.set_ylabel(r'$[\bar{n}_{0\phi}(x) - \bar{n}_{\mathrm{rand}}(x)]/n_0$',
                   fontsize=11)
    ax3.set_title('Generic preparation differential',
                  fontsize=12)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = BASE_DIR / 'gpe_protocol_results.png'
    plt.savefig(
        fig_path,
        dpi=150,
        bbox_inches='tight',
        metadata={'Software': 'matplotlib'},
    )
    plt.close()
    print(f"\nFigure saved to {fig_path}")

    data_path = BASE_DIR / 'gpe_protocol_data.dat'
    np.savetxt(
        data_path,
        np.column_stack([
            X,
            r4['diff'],
            r8['diff'],
            localized['x'],
            localized['predicted'],
            localized['measured'],
        ]),
        header=('x diff_zero_phase_minus_randomized_M4 '
                'diff_zero_phase_minus_randomized_M8 ; '
                'localized_x fixedN_predicted fixedN_measured ; '
                f'kernel_max_dev={max_kernel_err:.4e} '
                f"localized_peak={localized['response_peak']:.9e} "
                f"localized_max_rel={localized['max_relative']:.4e} "
                f"localized_l2_rel={localized['l2_relative']:.4e} "
                f"first_order={r4['first_order']:.4e} "
                f"maxdiff_M4={r4['max_diff']:.4e} "
                f"maxdiff_M8={r8['max_diff']:.4e} seed=20260610"),
        fmt='%.6e'
    )
    print(f"Data saved to {data_path}")
    print("\nGPE study complete; no thermal-state or detection claim is made.")


if __name__ == "__main__":
    main()
