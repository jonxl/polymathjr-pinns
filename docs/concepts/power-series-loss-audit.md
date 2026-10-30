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

## Comparing the two representations

Both trainers record the same three components and their total objective.
The categories have the same roles, but the solution representations and
supervision targets differ. Equivalent exact solutions do not imply equivalent
training objectives or interchangeable loss magnitudes.

| Component | Power series | Eigenvalue |
|---|---|---|
| Physics residual | ODE residual evaluated through the truncated polynomial | ODE residual evaluated through the exponential representation |
| Supervision | Error in predicted series coefficients | Error in reconstructed solution values |
| Initial conditions | Error in the polynomial’s initial value and derivative | Error in the predicted amplitude and initial derivative |
| Total objective | Weighted physics + **coefficient** supervision | Weighted physics + **solution** supervision |

The equations below describe one ODE; the batched trainer also averages over
the equations in the batch. Let the initial conditions be at zero:

$$
u''-\tau u'+\Delta u=0,
\qquad u(0)=a,\qquad u'(0)=b.
$$

The power-series model predicts derivative coefficients, while the eigenvalue
model predicts four parameters:

$$
\hat u_{\mathrm{PS}}(x)
=\sum_{n=0}^{N}\hat\psi_n\frac{x^n}{n!},
\qquad
\hat u_{\mathrm{EV}}(x)
=e^{\hat\mu x}\left[\hat A C(\hat k,x)+\hat B S(\hat k,x)\right].
$$

Here the exact basis functions are

$$
C(k,x)=\sum_{m=0}^{\infty}\frac{k^m x^{2m}}{(2m)!},
\qquad
S(k,x)=\sum_{m=0}^{\infty}\frac{k^m x^{2m+1}}{(2m+1)!}.
$$

The eigenvalue training implementation evaluates finite approximations of these
basis functions. It uses their analytic identities to construct the residual;
this should not be confused with an independent differentiation check of the
finite approximation.

### Physics residual

Both paths target the same differential equation. With the same normalized ODE
and the same collocation points, their intended residual metric is

$$
L_{\mathrm{PDE}}^{(r)}
=\frac1P\sum_{j=1}^{P}
\left|\hat u_r''(x_j)-\tau\hat u_r'(x_j)+\Delta\hat u_r(x_j)\right|^2,
\qquad r\in\{\mathrm{PS},\mathrm{EV}\}.
$$

### Supervision

For the power-series path, the first K coefficients are supervised. For the
eigenvalue path, the reconstructed function is compared with the analytic
solution at P collocation points:

$$
L_{\mathrm{sup}}^{\mathrm{PS}}
=\frac1K\sum_{n=0}^{K-1}|\hat\psi_n-\psi_n|^2,
\qquad
L_{\mathrm{sup}}^{\mathrm{EV}}
=\frac1P\sum_{j=1}^{P}|\hat u_{\mathrm{EV}}(x_j)-u(x_j)|^2.
$$

These measure different errors. A smaller coefficient MSE cannot be ranked
against a solution MSE as evidence of a better model.

### Initial-condition diagnostic

For degree at least one, the initial values reduce to

$$
L_{\mathrm{IC}}^{\mathrm{PS}}
=|\hat\psi_0-a|^2+|\hat\psi_1-b|^2,
\qquad
L_{\mathrm{IC}}^{\mathrm{EV}}
=|\hat A-a|^2+|\hat\mu\hat A+\hat B-b|^2.
$$

Both measure the same two constraints through different representations. This
component is saved as `bc` in the loss CSV, but is not optimized in the current
shared objective.

### Total objective and training histories

For each representation, the shared trainer uses

$$
L_{\mathrm{total}}^{(r)}
=w_{\mathrm{PDE}}K L_{\mathrm{PDE}}^{(r)}
+w_{\mathrm{sup}}L_{\mathrm{sup}}^{(r)},
\qquad K=\texttt{num\_supervised}.
$$

The factor K also applies to the eigenvalue path in the current implementation;
it does not mean that path supervises K coefficients. The CSV stores the
weighted total and the **raw** `pde`, `supervised`, and `bc` components. Adding
the three raw components therefore does not reproduce the total.

Plot all four recorded quantities against the actual training iterations for
each model. These curves reveal optimization behavior within that model;
their magnitudes do not rank the two representations. Older power-series
histories may use monomial-coefficient supervision, as described below.

For comparative accuracy and generalization, reconstruct both solutions and
evaluate them against the same analytic reference on identical points:

$$
E_r=\frac{\left(\sum_j|\hat u_r(x_j)-u(x_j)|^2\right)^{1/2}}
{\left(\sum_j|u(x_j)|^2\right)^{1/2}}.
$$

The numerical evaluator guards a zero denominator. Generalization maps,
transfer, and threshold areas should use this common solution metric. An exact
degree-N Taylor baseline additionally separates power-series truncation error
from learning error.

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
