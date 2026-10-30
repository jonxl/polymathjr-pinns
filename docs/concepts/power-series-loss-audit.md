# Power-series loss audit

The shared power-series trainer uses the Taylor/derivative coefficient notation
`ψₙ`:

$$
u(x)=\sum_{n=0}^{N}\psi_n\frac{x^n}{n!},
\qquad \psi_n=u^{(n)}(0).
$$

The factorial is deliberately absent from coefficient supervision. The network
output and analytic target are both raw `ψₙ`, so

$$
L_{\mathrm{coeff}}=\frac1K\sum_{n=0}^{K-1}(\hat\psi_n-\psi_n)^2.
$$

The factorial is used when the predicted series is evaluated. For derivative
order `r`,

$$
u^{(r)}(x)=\sum_{n=r}^{N}\hat\psi_n
\frac{x^{n-r}}{(n-r)!}.
$$

The ODE residual is averaged over the `P` collocation points:

$$
L_{\mathrm{ODE}}=\frac1P\sum_{j=1}^{P}
\left|\mathcal D[\hat u](x_j)\right|^2.
$$

The separate initial-condition diagnostic is still computed and logged, but it
is not part of the shared training objective. The active objective remains

```text
pde_weight * num_supervised * pde + supervised_weight * supervised
```

The `num_supervised` multiplier is separate from the factorial convention.

## What was fixed

The earlier implementation introduced `monomial_from_derivative`, which divided
the analytic `ψₙ` targets by `n!` before coefficient MSE. The batched GPU path did
the same and differentiated plain monomials with falling factorials. That trained
a different convention:

```text
u(x) = Σ cₙ xⁿ,  cₙ = ψₙ/n!
```

The shared loss path now keeps `ψₙ` unchanged and uses inverse factorials in the
scalar PDE matrix, batched GPU derivative matrices, boundary diagnostic vectors,
and scalar reconstruction helpers. `true_solutions(::BatchBuffers)` uses the
same Taylor basis. The separate monomial conversion function is removed from
the training path.

The change applies to newly trained models. Existing checkpoints trained under
the monomial-output convention must not be evaluated as Taylor-coefficient
models; they need old-basis metadata or retraining.

## Evidence

The focused test [test_power_series_math.jl](../../test/test_power_series_math.jl)
checks scalar, cached, and batched losses against independent Taylor
differentiation. It checks mean-loss normalization, autodiff gradients against
finite differences, and reconstruction of `e^(2x)` from raw derivative
coefficients.

```bash
julia --project=. test/test_power_series_math.jl
```

CUDA value and gradient checks run when a functional CUDA device is available.
The audit machine has no functional CUDA device, so GPU execution still needs
to be checked on the training machine.
