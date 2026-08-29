# Model Status and Identifiability Audit

This note records what the current methods paper does and does not establish. It
is supporting documentation, not an independent claim source.

## Exact finite-mode result

For a finite set of positive-frequency bosonic modes, `beta > 0`, and a finite
Gibbs partition function,

`D(rho || sigma_beta) = beta * (E_rho - E_sigma) - (S_rho - S_sigma)`

is exact whenever the displayed quantities are finite. An ideal displaced
Gibbs state and an ideal equal-energy heated product Gibbs state therefore obey

`D_displaced - D_heated = Delta S_heated`.

This is a statement about the defined density operators. It does not show that
a laboratory phase-randomization procedure prepares the heated quantum state.

## Stipulated preparation-indexed potential

The paper assigns

`delta U_i(x) = -g * n_0 * kappa * D_i * q(x)`.

`kappa`, the sign, the profile `q(x)`, the scalar-to-local map, and the source
intervention are free choices. They are not derived from microscopic condensate
physics or gravity. No independent mechanism is supplied that would activate
the potential according to an unknown preparation property. The construction is
therefore a statistical identifiability thought experiment, not a physical-source
proposal.

## Fixed-particle-number response

The stationary GP/BdG response is evaluated at fixed total particle number. The
uniform Fourier component is absorbed into the chemical potential, so
`delta n_(k=0) = 0`. For nonzero modes,

`delta n_k / n_0 = -(delta U_k / g n_0) / (1 + (k * xi)^2 / 2)`.

For the declared exponential profile, periodic length 400, 2048-point grid, and
`xi = 1/sqrt(2)`, the fixed-number response peak is `0.902969201`. The resulting
contrast normalization is `Delta S * R_peak = 262.525274`.

An independent imaginary-time GP relaxation uses the same profile, grid, and
fixed-number convention. Its maximum profile difference is `0.0226%` of peak;
its relative L2 difference is `0.0076%`. This checks implementation of the
conditional response. It does not validate the stipulated map.

## Generic preparation example

The seeded 1D GPE calculation also compares aligned-phase and randomized-phase
pure classical fields with the same modal amplitudes. The populated band is
approximately `0.100 <= k*xi <= 0.200`. The order-`10^-3` differences show only
that ordinary preparation-dependent structure can exist.

The calculation is not:

- the ideal heated density operator;
- exactly matched in full nonlinear GP energy;
- projected through the paper's signed estimator;
- a covariance or convergence study; or
- a representative nuisance or uncertainty budget.

## Conditional inference

With a signed template `t`, declared nuisance matrix `N`, calibrated covariance
`C`, and the corresponding nuisance projector `P_N`, the standard GLS estimator
is identifiable only when

`t^T C^-1 P_N t > 0`.

If a future experiment independently establishes a residual budget for the
declared signed template after validated nuisance removal, the scalar planning
relation is

`|kappa| <= delta A_res / 262.525274`.

Residual budgets from `1e-6` through `1e-2` therefore map algebraically to
`3.809e-9` through `3.809e-5`. The repository does not measure, choose, or
forecast such a budget.

## Evidence required before an empirical coefficient claim

1. Define a source-on/source-off intervention independent of programming the
   assumed potential.
2. Prepare and independently characterize the intended quantum states.
3. Match particle number, energy, density, momentum, and drive histories to
   declared tolerances.
4. Predeclare the signed profile, zero-mode convention, preprocessing, nuisance
   basis, covariance, estimator, and decision rule.
5. Validate nuisance adequacy and injected-signal recovery on held-out controls.
6. Demonstrate numerical and experimental convergence.
7. State any result only as a bound within the declared potential and nuisance
   model.

## Nonclaims

The current paper does not claim:

- a new source law or microscopic interaction;
- an operational physical-source experiment;
- a covariant theory of gravity or informational stress tensor;
- an acoustic metric, flow, horizon, or analogue-gravity result;
- a numerical detection or falsification threshold; or
- a result suitable for *Foundations of Physics* without genuinely new science.
