#!/usr/bin/env julia
# Shared versions of Jeet's bb2a102a experiments. Legacy scripts remain reproducible.
function usage()
    println("""
    Geometry experiments comparing power-series and eigenvalue representations.

    julia --project=. scripts/shared/geometry.jl train OUTPUT_ROOT [EXPERIMENTS] [N] [ITERS] [NG]
    julia --project=. scripts/shared/geometry.jl frozen STUDY_ROOT OUTPUT_ROOT [EXPERIMENTS] [N] [NG]
    julia --project=. scripts/shared/geometry.jl design [EXPERIMENTS] [N] [NG]

    EXPERIMENTS: comma-separated names or all (default):
      family_disks,spiral_transfer,pair_transfer,parabola_segment,parabola_slide,
      parabola_curves,parabola_bias,axis_transfer,axis_sweep
    N and NG default to Jeet’s per-experiment point counts and grid resolutions.
    ITERS defaults to 3000 shared GPU Adam iterations. Both representations use identical geometries,
    trace/determinant inputs, seed 1234, and the shared GPU-capable trainer.
    Results include analytic solution relative-L2, per-point errors, convergence
    histories, checkpoints and viewer panelset.json. Once both heads finish,
    comparison.json pairs their charts with common heatmap scales. controls.json
    records the shared equations and controls; mismatched runs are rejected.
    Training uses the dual branch's PDE + supervision objective; IC is diagnostic.
    This differs from the legacy scripts' Adam/LBFGS training-to-floor protocol.

    Frozen mode loads the 14 validation-selected checkpoints without retraining.
    Its disk/axis/parabola measurements are probes of the pretrained models;
    they do not reproduce geometry-specific training/transfer claims.
    OUTPUT_ROOT must be outside STUDY_ROOT to preserve the portable bundle.

    Design mode prints dataset sizes without loading the ML/GPU stack.
    For per-GPU dispatch of matched training, use scripts/generalization_variants.jl
    --experiments family_disks,axis_transfer (the new experiments are opt-in there).
    """)
end
if isempty(ARGS) || ARGS[1] in ("-h","--help")
    usage()
    exit(0)
end
include("../../utils/generalization_geometry.jl")
using .GeneralizationGeometry
optional_int(index)=length(ARGS)>=index ? parse(Int,ARGS[index]) : nothing
function parse_exps(value)
    value=="all" && return EXPERIMENTS
    names=Tuple(Symbol.(split(value,',')))
    !isempty(names) && all(n->n in EXPERIMENTS,names) || error("Unknown experiment list: $value")
    names
end
mode=ARGS[1]
if mode=="design"
    length(ARGS)<=4 || error("Too many arguments")
    names=parse_exps(get(ARGS,2,"all")); n=optional_int(3); ng=optional_int(4)
    for name in names
        spec=design(name;n,ng)
        println(name,": ",length(spec.cases)," training cases per representation, ",
            sum(length(p.second) for p in spec.probes)," probe points per model")
    end
elseif mode in ("train","frozen")
    (mode=="train" ? 2<=length(ARGS)<=6 : 3<=length(ARGS)<=6) || error("Invalid argument count; use --help")
    include("../../utils/geometry_experiments.jl")
    if mode=="train"
        names=parse_exps(get(ARGS,3,"all")); n=optional_int(4)
        iters=parse(Int,get(ARGS,5,"3000")); ng=optional_int(6)
        for name in names, rep in (:power_series,:eigenvalue)
            cfg=GeometryExperiments.Experiments.ExperimentConfig(;representation=rep,n_per_region=something(n,jeet_controls(name).n),maxiters=iters)
            GeometryExperiments.run_geometry(cfg,name;output_root=ARGS[2],ng)
        end
    else
        names=parse_exps(get(ARGS,4,"all")); n=optional_int(5); ng=optional_int(6)
        for name in names
            GeometryExperiments.run_frozen_geometry(ARGS[2],name;output_root=ARGS[3],n,ng)
        end
    end
else
    error("Unknown mode: $mode; use --help")
end
