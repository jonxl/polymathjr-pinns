# Interactive shape-comparison explorer, backed by GLMakie.
#
# explore(runs, mix) opens one window comparing several initial
# smoke shapes side by side under a single time slider:
#
#   - Top row: one "smoke on air" render per shape (circle,
#     rectangle, trapezoid, bump, …). A "field" dropdown selects
#     WHICH spatial diagnostic every panel shows at once:
#     density, vertical momentum, ux, uy, temperature,
#     non-equilibrium |f − M[f]|, heat flux qy, stress anisotropy.
#   - A "graphs" dropdown (grouped into sections Conserved /
#     Relaxation / Monitors) selects one scalar-vs-time
#     diagnostic, drawn as one curve per shape in a single strip
#     below the renders — so the renders keep the space.
#
# `runs` is a vector of (name, times, history). Spatial fields
# are computed ON DEMAND per frame (only global color limits and
# the small scalar series are precomputed), so holding several
# shapes' histories at once stays within memory.
#
# Every Makie name is qualified with GLMakie. — Plots (loaded by
# viz/Viz.jl) exports clashing names like heatmap!/lines!.
#
# Depends on boltzmann/ (incl. Diagnostics.jl) and viz/Viz.jl.
# Run from explore.jl — needs a display.
import GLMakie

# The spatial field arrays available to the "field" dropdown.
const FIELD_OPTIONS = ["density", "vertical momentum", "bulk velocity ux",
                       "bulk velocity uy", "temperature",
                       "non-equilibrium |f−M|", "heat flux qy", "stress anisotropy"]
const FIELD_SIGNED = Dict("density" => false, "vertical momentum" => true,
                          "bulk velocity ux" => true, "bulk velocity uy" => true,
                          "temperature" => false, "non-equilibrium |f−M|" => false,
                          "heat flux qy" => true, "stress anisotropy" => false)

# One spatial field array of the selected quantity for species i.
function field_species(F, q, i, grid)
  q == "density"              && return moments(F, 0, 0, grid)[i]
  q == "vertical momentum"    && return moments(F, 0, 1, grid)[i]
  q == "bulk velocity ux"     && return local_fields(F[i], grid)[2]
  q == "bulk velocity uy"     && return local_fields(F[i], grid)[3]
  q == "temperature"          && return local_fields(F[i], grid)[4]
  q == "non-equilibrium |f−M|" && return noneq_field(F[i], grid)
  q == "heat flux qy"         && return heat_flux(F[i], grid)[2]
  q == "stress anisotropy"    && return stress_anisotropy(F[i], grid)
  error("unknown field $q")
end

function explore(runs, mix::Mixture{N}) where {N}
  grid  = mix.grid
  names = [sp.name for sp in mix.species]   # ["smoke", "air"]
  ts    = runs[1].times
  nt    = length(ts)
  ns    = length(runs)
  dA, _ = phase_weights(grid)

  # Scalar diagnostics offered by the "graphs" dropdown, grouped
  # into sections via a "Section · name" prefix (one curve per
  # shape; smoke species).
  DIAGS = ["Conserved · smoke mass", "Conserved · smoke y-momentum",
           "Conserved · smoke energy",
           "Relaxation · entropy H", "Relaxation · mean temperature",
           "Relaxation · non-equilibrium ‖f−M‖",
           "Monitors · positivity min f", "Monitors · plume height ȳ"]

  # Initial air density per shape (background reference for the
  # -- one discarding pass: numeric ranges + scalar series -------
  # Each species' field is colored across its OWN numeric
  # [min, max] over all shapes and frames — purely data-driven.
  println("Precomputing comparison diagnostics ($ns shapes × $nt frames)...")
  ov_lo = Dict(q =>  Inf for q in FIELD_OPTIONS)   # smoke (red) low
  ov_hi = Dict(q => -Inf for q in FIELD_OPTIONS)   # smoke (red) high
  bg_lo = Dict(q =>  Inf for q in FIELD_OPTIONS)   # air (blue) low
  bg_hi = Dict(q => -Inf for q in FIELD_OPTIONS)   # air (blue) high
  scal   = Dict(d => [zeros(nt) for _ in 1:ns] for d in DIAGS)

  for (si, r) in enumerate(runs), k in 1:nt
    F = r.history[k]
    scal["Conserved · smoke mass"][si][k]        = total_moment(F, 0, 0, grid)[1]
    scal["Conserved · smoke y-momentum"][si][k]  = total_moment(F, 0, 1, grid)[1]
    scal["Conserved · smoke energy"][si][k]      = total_energy(F, grid)[1]
    scal["Relaxation · entropy H"][si][k]        = entropy_functional(F[1], grid)
    scal["Relaxation · mean temperature"][si][k] = mean_temperature(F[1], grid)
    neq = noneq_field(F[1], grid)
    scal["Relaxation · non-equilibrium ‖f−M‖"][si][k] = sqrt(sum(abs2, neq) * dA)
    scal["Monitors · positivity min f"][si][k]   = min_density(F[1])
    scal["Monitors · plume height ȳ"][si][k]     = center_of_mass(F[1], grid)[2]

    for q in FIELD_OPTIONS
      sm = field_species(F, q, 1, grid)
      ai = field_species(F, q, 2, grid)
      ov_lo[q] = min(ov_lo[q], minimum(sm)); ov_hi[q] = max(ov_hi[q], maximum(sm))
      bg_lo[q] = min(bg_lo[q], minimum(ai)); bg_hi[q] = max(bg_hi[q], maximum(ai))
    end
  end

  # -- window ----------------------------------------------------
  GLMakie.activate!()
  fig = GLMakie.Figure(size = (max(360 * ns + 120, 900), 860))

  sl = GLMakie.Slider(fig[5, 1:ns], range = 1:nt, startvalue = 1)
  k  = sl.value
  t_now = GLMakie.lift(j -> ts[j], k)

  GLMakie.Label(fig[1, 1:ns],
    GLMakie.lift(j -> "t = $(round(ts[j], digits = 3))   (0 … $(round(ts[end], digits = 3)))", k),
    fontsize = 20, font = :bold)

  # Two dropdowns: spatial field (all panels) and scalar graph.
  controls = fig[2, 1:ns] = GLMakie.GridLayout()
  GLMakie.Label(controls[1, 1], "field:", halign = :right)
  field_menu = GLMakie.Menu(controls[1, 2], options = FIELD_OPTIONS,
                            default = "density", width = 210)
  GLMakie.Label(controls[1, 3], "graphs:", halign = :right)
  graph_menu = GLMakie.Menu(controls[1, 4], options = DIAGS,
                            default = "Relaxation · mean temperature", width = 260)
  GLMakie.colgap!(controls, 8)

  # Cold air: opaque white→blue by value. Hot smoke: light→deep
  # red with rising opacity so it overlays the air. Both mapped
  # across each species' own numeric [min, max] (padded when a
  # field is constant).
  air_cmap   = [GLMakie.RGBf(1 - 0.90t, 1 - 0.75t, 1 - 0.25t) for t in range(0, 1, length = 256)]
  smoke_cmap = [GLMakie.RGBAf(0.95 - 0.35t, 0.15 - 0.15t, 0.15 - 0.15t, t) for t in range(0, 1, length = 256)]
  pad(lo, hi) = hi > lo ? (lo, hi) : (lo - 1e-9, hi + 1e-9)

  # -- top: one smoke-on-air render per shape --------------------
  renders = fig[3, 1:ns] = GLMakie.GridLayout()
  local hm_bg, hm_ov
  for (si, r) in enumerate(runs)
    ax = GLMakie.Axis(renders[1, si],
                      title = GLMakie.lift(q -> "$(r.name) — $q", field_menu.selection),
                      xlabel = "x", ylabel = "y", aspect = GLMakie.DataAspect())
    bg = GLMakie.lift((j, q) -> field_species(r.history[j], q, 2, grid), k, field_menu.selection)
    bgrange = GLMakie.lift(q -> pad(bg_lo[q], bg_hi[q]), field_menu.selection)
    hm_bg = GLMakie.heatmap!(ax, grid.x, grid.y, bg; colorrange = bgrange, colormap = air_cmap)
    ov = GLMakie.lift((j, q) -> field_species(r.history[j], q, 1, grid), k, field_menu.selection)
    ovrange = GLMakie.lift(q -> pad(ov_lo[q], ov_hi[q]), field_menu.selection)
    hm_ov = GLMakie.heatmap!(ax, grid.x, grid.y, ov; colorrange = ovrange, colormap = smoke_cmap)
  end
  # Shared colorbars for the whole row (limits are global).
  GLMakie.Colorbar(renders[1, ns + 1], hm_bg,
    label = GLMakie.lift(q -> "$(names[2]) (cold) $q", field_menu.selection),
    width = 12, height = GLMakie.Relative(0.8))
  GLMakie.Colorbar(renders[1, ns + 2], hm_ov,
    label = GLMakie.lift(q -> "$(names[1]) (hot) $q", field_menu.selection),
    width = 12, height = GLMakie.Relative(0.8))

  # -- bottom: selected scalar diagnostic, one curve per shape ---
  gax = GLMakie.Axis(fig[4, 1:ns],
                     title = GLMakie.lift(d -> d, graph_menu.selection),
                     xlabel = "time t", titlesize = 13)
  for (si, r) in enumerate(runs)
    GLMakie.lines!(gax, ts, GLMakie.lift(d -> scal[d][si], graph_menu.selection),
                   label = r.name, linewidth = 2)
  end
  GLMakie.vlines!(gax, t_now, color = (:black, 0.5), linewidth = 1)
  GLMakie.axislegend(gax, labelsize = 9, position = :lt)

  GLMakie.rowsize!(fig.layout, 3, GLMakie.Relative(0.62))  # renders dominate
  GLMakie.rowsize!(fig.layout, 4, GLMakie.Relative(0.22))

  GLMakie.display(fig)
  return fig
end
