module GeometryExperiments
using JSON, CUDA, SHA, Statistics, Random
include("experiments.jl")
using .Experiments
using .Experiments.PanelSpec
include("geometry_comparison.jl")
using .GeometryComparison
const CONTRACT_LOCK = ReentrantLock()
include("generalization_geometry.jl")
using .GeneralizationGeometry
include("geometry_panels.jl")
using .GeometryPanels
const PINN = Experiments.PINN
export run_geometry, run_frozen_geometry

function write_json(path, payload)
    mkpath(dirname(path))
    open(path,"w") do io
        JSON.print(io,payload,2)
    end
end
# A representation may join an experiment only with the same equations and controls.
function ensure_contract(root, spec, cfg, nx)
    contract=Dict("design"=>spec,"power_series_degree"=>cfg.N,
        "network_hidden_width"=>cfg.neuron_count,"model_seed"=>cfg.seed,
        "adam_iterations"=>cfg.maxiters,"collocation_points"=>cfg.num_points,
        "x_left"=>cfg.x_left,"x_right"=>cfg.x_right,"solution_points"=>nx,
        "pde_weight"=>cfg.pde_weight,"supervised_weight"=>cfg.supervised_weight,
        "input_encoding"=>"trace_determinant","power_series_output_basis"=>"taylor_derivative",
        "plugboard_storage_basis"=>"derivative","initial_conditions"=>[1.,0.],
        "optimizer"=>"shared_gpu_adam")
    normalized=JSON.parse(JSON.json(contract))
    path=joinpath(root,"controls.json")
    lock(CONTRACT_LOCK) do
        if isfile(path)
            JSON.parsefile(path)==normalized || error("Representation controls differ from $path; use identical settings or a new output root")
        else
            write_json(path,contract)
        end
    end
    open(io->bytes2hex(sha256(io)),path)
end
function finish_panels(dir, spec, results, rep, digest)
    ps=panels_for(spec,results,rep,"matched_training")
    ps.meta["design_sha256"]=digest
    controls=JSON.parsefile(joinpath(dirname(dir),"controls.json"))
    ps.meta["training_controls"]=Dict(k=>controls[k] for k in
        ("power_series_degree","adam_iterations","network_hidden_width","model_seed",
         "collocation_points","solution_points","x_left","x_right","optimizer"))
    lock(CONTRACT_LOCK) do
        save_experiment(joinpath(dir,"panelset.json"),ps)
        root=dirname(dir)
        paths=[joinpath(root,String(r),"panelset.json") for r in (:power_series,:eigenvalue)]
        if all(isfile,paths)
            variation_path=joinpath(root,"power_series_variation","panelset.json")
            isfile(variation_path) && push!(paths,variation_path)
            paired=paired_panels(load_experiment.(paths)...)
            save_experiment(joinpath(root,"comparison.json"),paired)
        end
    end
end
jsonpoints(points) = [Dict("tau"=>t,"delta"=>d) for (t,d) in points]
# Preserve point order directly; Dict iteration must never determine plot positions.
function predict_errors(model, points; batch_size=1024, nx=200)
    batch_size > 0 && nx >= 2 || error("Invalid evaluation batch size or x resolution")
    errors = Float64[]
    xs = range(model.settings.x_left,model.settings.x_right,length=nx)
    for firstidx in 1:batch_size:length(points)
        chunk = points[firstidx:min(firstidx+batch_size-1,length(points))]
        X = reduce(hcat,[Float32[t,d] for (t,d) in chunk])
        output = Array(first(model.net(model.to_device(X),model.p,model.st)))
        append!(errors,point_errors(output,model.settings.representation,chunk,xs))
    end
    errors
end
function device_model(r; use_gpu=CUDA.functional())
    transfer = use_gpu ? CUDA.cu : identity
    (; net=r.net,p=transfer(r.p),st=transfer(r.st),settings=r.settings,to_device=transfer)
end
function score(model, points; kwargs...)
    errors = predict_errors(model,points;kwargs...)
    finite = filter(isfinite,errors)
    Dict("points"=>jsonpoints(points), "errors"=>[isfinite(e) ? e : nothing for e in errors],
         "failed_points"=>count(!isfinite,errors),
         "mean_rel_l2"=>all(isfinite,errors) ? mean(errors) : nothing,
         "median_rel_l2"=>isfinite(median(errors)) ? median(errors) : nothing,
         "finite_mean_rel_l2"=>isempty(finite) ? nothing : mean(finite),
         "fraction_below_0.1"=>count(e->isfinite(e)&&e<=.1,errors)/length(errors))
end
function evaluate_design(model, spec; batch_size=1024, nx=200)
    Dict(label=>score(model,points;batch_size,nx) for (label,points) in spec.probes)
end
function diagnostics(model, c; batch_size=1024, nx=200, tolerance=.1,
                     reach_step=.05, reach_max=8., nangles=120, family_shells=false)
    extra = Dict{String,Any}()
    if c.center !== nothing
        radii = collect(0.:reach_step:reach_max)
        angles = range(0.,2π,length=nangles+1)[1:end-1]
        points = [(c.center[1]+r*cos(a),c.center[2]+r*sin(a)) for a in angles for r in radii]
        errors = predict_errors(model,points;batch_size,nx)
        reach = reach_summary(errors,radii,nangles;tolerance,radius=c.radius)
        extra["reach"] = Dict("median"=>reach.median_reach,"margin"=>reach.margin,
            "radii"=>reach.reaches,"censored"=>reach.censored,
            "step"=>reach_step,"cap"=>reach_max,"tolerance"=>tolerance,
            "patch_kind"=>startswith(c.label,"center") ? "segment" : "disk")
        if family_shells
            # Reference family_disks: 20 shells × 600 random angles, seed reset per model.
            radii=collect(.2:.4:8.)
            rng=MersenneTwister(2024)
            shell_points=Tuple{Float64,Float64}[]
            for r in radii, _ in 1:600
                phi=2π*rand(rng,Float32)
                push!(shell_points,(Float64(Float32(c.center[1]+Float32(r)*cos(phi))),
                                    Float64(Float32(c.center[2]+Float32(r)*sin(phi)))))
            end
            # Shell-major order is transposed to reuse the radius-major binning below.
            points=[shell_points[(i-1)*600+j] for j in 1:600 for i in eachindex(radii)]
            nangles=600
            errors=predict_errors(model,points;batch_size,nx)
            extra["shell_sampling"]=Dict("seed"=>2024,"points_per_shell"=>600,"width"=>.4,
                "points"=>jsonpoints(points))
        end
        # Same radial probes also show in-family/out-of-family extrapolation.
        shells = Dict{String,Any}()
        for reg in (:saddle,:stable_node,:unstable_node,:stable_spiral,:unstable_spiral,:center)
            xs=Float64[]; means=Any[]
            for (i,r) in enumerate(radii)
                inds = [i+(j-1)*length(radii) for j in 1:nangles
                    if region(points[i+(j-1)*length(radii)]...) === reg]
                isempty(inds) && continue
                es=errors[inds]; push!(xs,r); push!(means,all(isfinite,es) ? mean(es) : nothing)
            end
            shells[String(reg)]=Dict("radius"=>xs,"mean_rel_l2"=>means)
        end
        extra["shells_by_region"]=shells
    end
    if c.window !== nothing
        lo,hi=c.window; distances=collect(0.:.03:2.5)
        for (label,edge,dir) in (("left",lo,-1),("right",hi,1))
            points=[(edge+dir*d,(edge+dir*d)^2/4) for d in distances]
            e=predict_errors(model,points;batch_size,nx)
            reach=reach_summary(e,distances,1;tolerance)
            extra["$(label)_reach"]=Dict("distance"=>reach.median_reach,"censored"=>only(reach.censored))
        end
    end
    extra
end

"""Train each geometry with the shared GPU trainer, identical inputs/seeds for both heads.
Uses the dual branch's Adam objective (PDE + supervision; IC diagnostic), not the
legacy scripts' Adam/LBFGS floor objective. Saves convergence evidence per model.
"""
function run_geometry(cfg::ExperimentConfig,name::Symbol; output_root="results/geometry",
                      ng=nothing,ntest=nothing,placements=0,batch_size=1024,nx=200)
    cfg.maxiters > 0 || error("Training iterations must be positive")
    spec=design(name;n=cfg.n_per_region,ng,ntest,placements,seed=cfg.seed)
    dir=joinpath(output_root,String(name),String(cfg.representation))
    ispath(joinpath(dir,"results.json")) && error("Results already exist: $dir")
    digest=ensure_contract(dirname(dir),spec,cfg,nx)
    mkpath(dir)
    write_json(joinpath(dir,"design.json"),spec)
    results=Dict{String,Any}()
    for c in spec.cases
        @info "Geometry training" experiment=name case=c.label representation=cfg.representation
        dataset=Experiments.Plugboard.generate_grid_dataset(
            first.(c.points),last.(c.points),cfg.N)
        length(dataset)==length(c.points) || error("Plugboard collapsed duplicate equations in $(c.label)")
        r=Experiments.train_one(cfg,dataset,c.label;output_root=dir,input_encoding=:trace_determinant)
        model=device_model(r)
        probes=evaluate_design(model,spec;batch_size,nx)
        train_score=score(model,c.points;batch_size,nx)
        extra=diagnostics(model,c;batch_size,nx,reach_step=spec.controls.reach_step,
            reach_max=spec.controls.reach_max,nangles=spec.controls.nangles,
            family_shells=name===:family_disks)
        if name === :family_disks && c.center !== nothing
            tau_axis = collect(range(c.center[1]-1., c.center[1]+1., length=201))
            delta_axis = collect(range(c.center[2]-1., c.center[2]+1., length=201))
            local_points = [(t,d) for d in delta_axis for t in tau_axis]
            extra["local_plane"] = Dict("tau_axis"=>tau_axis,"delta_axis"=>delta_axis,
                "score"=>score(model,local_points;batch_size,nx))
        end
        if haskey(probes,"left") && haskey(probes,"right")
            l=probes["left"]["mean_rel_l2"]; rr=probes["right"]["mean_rel_l2"]
            extra["right_minus_left_bias"]=l===nothing || rr===nothing ? nothing : rr-l
        end
        results[c.label]=Dict("training"=>train_score,"probes"=>probes,"diagnostics"=>extra,
            "loss_history"=>Experiments.read_loss_history(r.dir),
            "checkpoint"=>relpath(joinpath(r.dir,"model.checkpoint"),dir))
        write_json(joinpath(dir,"results.json"),Dict("mode"=>"matched_training",
            "config"=>cfg,"metric"=>"mean per-ODE analytic solution relative L2",
            "objective"=>"shared Adam PDE + supervised; IC diagnostic",
            "source_commit"=>"bb2a102a","design_sha256"=>digest,"models"=>results))
    end
    finish_panels(dir,spec,results,cfg.representation,digest)
    results
end

"""Probe frozen validation-selected models; no claim of training on these geometries."""
function run_frozen_geometry(root,name::Symbol;output_root,n=nothing,ng=nothing,ntest=nothing,
                             batch_size=1024,nx=200,use_gpu=CUDA.functional())
    root=realpath(root)
    # Resolve existing ancestors before creating anything, including symlinked parents.
    out=abspath(output_root)
    ancestor=out; suffix=String[]
    while !ispath(ancestor)
        pushfirst!(suffix,basename(ancestor)); ancestor=dirname(ancestor)
    end
    out=joinpath(realpath(ancestor),suffix...)
    (out==root || startswith(out,root*"/")) && error("Frozen derived outputs must be outside the study bundle")
    config=JSON.parsefile(joinpath(root,"config.json"))["training"]
    selection=JSON.parsefile(joinpath(root,"selection","selected-checkpoints.json"))
    get(selection,"selection_complete",false) || error("Checkpoint selection is incomplete")
    models=selection["models"]
    reps=Symbol.(get(JSON.parsefile(joinpath(root,"config.json")),"representations",["power_series","eigenvalue"]))
    expected=Set(vcat(["global/$rep" for rep in reps],
        ["family/$reg/$rep" for reg in (:saddle,:stable_node,:unstable_node,:stable_spiral,:unstable_spiral,:center)
         for rep in reps]))
    Set(keys(models))==expected || error("Selected models do not match the configured experiments")
    for selected in values(models)
        path=joinpath(root,selected["checkpoint"])
        open(io->bytes2hex(sha256(io)),path)==selected["checkpoint_sha256"] ||
            error("Frozen checkpoint changed: $path")
    end
    spec=design(name;n,ng,ntest)
    for rep in reps
        dir=joinpath(out,String(name),String(rep))
        ispath(joinpath(dir,"results.json")) && error("Results already exist: $dir")
        results=Dict{String,Any}()
        for key in sort!([k for k in keys(models) if endswith(k,"/$rep")])
            selected=models[key]; path=joinpath(root,selected["checkpoint"])
            digest=open(io->bytes2hex(sha256(io)),path)
            digest==selected["checkpoint_sha256"] || error("Frozen checkpoint changed: $path")
            net,p,st,meta=PINN.SafeTensorSnapshots.load_any_model(path)
            Symbol(meta["representation"])===rep || error("Representation mismatch: $path")
            cfg=ExperimentConfig(;representation=rep,N=Int(config["power_series_degree"]),
                neuron_count=Int(config["network_hidden_width"]),seed=Int(config["model_seed"]))
            dataset=Experiments.Plugboard.generate_grid_dataset([0.],[0.],cfg.N)
            settings=Experiments.settings_for(cfg,dataset;input_encoding=:trace_determinant)
            model=device_model((;net,p,st,settings);use_gpu)
            probes=evaluate_design(model,spec;batch_size,nx)
            # Patches are probes of pretrained models, not new training data.
            patches=Dict(c.label=>score(model,c.points;batch_size,nx) for c in spec.cases)
            results[key]=Dict("probes"=>probes,"patch_probes"=>patches,
                "checkpoint"=>selected["checkpoint"],"checkpoint_sha256"=>digest)
        end
        write_json(joinpath(dir,"design.json"),spec)
        write_json(joinpath(dir,"results.json"),Dict("mode"=>"frozen_probe",
            "study_root"=>root,"source_commit"=>"bb2a102a","models"=>results))
        save_experiment(joinpath(dir,"panelset.json"),panels_for(spec,results,rep,"frozen_probe"))
    end
end
end
