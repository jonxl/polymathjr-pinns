module GeometryExperiments
using JSON, CUDA, SHA, Statistics
include("experiments.jl")
using .Experiments
using .Experiments.PanelSpec
include("generalization_geometry.jl")
using .GeneralizationGeometry
const PINN = Experiments.PINN
export run_geometry, run_frozen_geometry

function write_json(path, payload)
    mkpath(dirname(path))
    open(path,"w") do io
        JSON.print(io,payload,2)
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
                     reach_step=.05, reach_max=8., nangles=120)
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
function panels_for(spec, results, rep, mode)
    panels=Panel[]
    for label in sort!(collect(keys(results)))
        scores=results[label]["probes"]
        plane=scores["plane"]
        # Raw JSON carries null plus failed_points; plots use a finite failure cap.
        es=[e === nothing ? 1e6 : e for e in plane["errors"]]
        push!(panels,heatmap_panel("$(label)-plane","$label ($rep)",
            permutedims(reshape(es,length(spec.axis),length(spec.axis))),
            string.(spec.axis),string.(spec.axis); xlabel="τ",ylabel="Δ",
            annotate=false,x_values=spec.axis,y_values=spec.axis,contour_levels=[.1],floor_val=1e-6))
        if haskey(scores,"parabola")
            s=scores["parabola"]
            push!(panels,lines_panel("$(label)-parabola","$label: along parabola",
                [("solution relative L2",[p["tau"] for p in s["points"]],
                  [e === nothing ? 1e6 : max(e,1e-12) for e in s["errors"]])];
                xlabel="τ",ylabel="relative L2",log_y=true))
        end
        labels=sort!([k for k in keys(scores) if k ∉ ("plane","parabola")])
        if !isempty(labels)
            values=[something(scores[k]["mean_rel_l2"],1e6) for k in labels]
            push!(panels,grouped_bar_panel("$(label)-transfer","$label: probe errors",labels,
                [("mean solution relative L2",max.(values,1e-12))];log_y=true))
        end
    end
    reach_labels=sort!([label for label in keys(results)
        if haskey(get(results[label],"diagnostics",Dict()),"reach")])
    if !isempty(reach_labels)
        push!(panels,grouped_bar_panel("reach-margins","Median reach beyond patch edge ($rep)",
            reach_labels,[("reach minus patch radius",
                [results[label]["diagnostics"]["reach"]["margin"] for label in reach_labels])];
            ylabel="distance",log_y=false))
    end
    bias_cases=[c for c in spec.cases if haskey(results,c.label) &&
        get(get(results[c.label],"diagnostics",Dict()),"right_minus_left_bias",nothing)!==nothing]
    if !isempty(bias_cases)
        push!(panels,lines_panel("parabola-bias","Fixed-endpoint right minus left error ($rep)",
            [("bias",[sum(c.window)/2 for c in bias_cases],
                [results[c.label]["diagnostics"]["right_minus_left_bias"] for c in bias_cases])];
            xlabel="training window center τ",ylabel="right minus left relative L2"))
    end
    PanelSet("$(spec.name) $rep ($mode)",Dict{String,Any}("representation"=>String(rep),
        "experiment"=>String(spec.name),"mode"=>mode,"source_commit"=>"bb2a102a",
        "truth"=>"Float64 analytic solution; y(0)=1, y'(0)=0",
        "plot_nonfinite_cap"=>1e6),panels)
end

"""Train each geometry with the shared GPU trainer, identical inputs/seeds for both heads.
Uses the dual branch's Adam objective (PDE + supervision; IC diagnostic), not the
legacy scripts' Adam/LBFGS floor objective. Saves convergence evidence per model.
"""
function run_geometry(cfg::ExperimentConfig,name::Symbol; output_root="results/geometry",
                      ng=31,ntest=400,placements=0,batch_size=1024,nx=200)
    cfg.maxiters > 0 || error("Training iterations must be positive")
    spec=design(name;n=cfg.n_per_region,ng,ntest,placements,seed=cfg.seed)
    dir=joinpath(output_root,String(name),String(cfg.representation))
    ispath(joinpath(dir,"results.json")) && error("Results already exist: $dir")
    mkpath(dir)
    write_json(joinpath(dir,"design.json"),spec)
    results=Dict{String,Any}()
    for c in spec.cases
        @info "Geometry training" experiment=name case=c.label representation=cfg.representation
        dataset=Experiments.Plugboard.generate_grid_dataset(
            first.(c.points),last.(c.points),cfg.N)
        r=Experiments.train_one(cfg,dataset,c.label;output_root=dir,input_encoding=:trace_determinant)
        model=device_model(r)
        probes=evaluate_design(model,spec;batch_size,nx)
        train_score=score(model,c.points;batch_size,nx)
        extra=diagnostics(model,c;batch_size,nx)
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
            "source_commit"=>"bb2a102a","models"=>results))
    end
    save_experiment(joinpath(dir,"panelset.json"),panels_for(spec,results,cfg.representation,"matched_training"))
    results
end

"""Probe frozen validation-selected models; no claim of training on these geometries."""
function run_frozen_geometry(root,name::Symbol;output_root,n=200,ng=31,ntest=400,
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
    expected=Set(vcat(["global/$rep" for rep in (:power_series,:eigenvalue)],
        ["family/$reg/$rep" for reg in (:saddle,:stable_node,:unstable_node,:stable_spiral,:unstable_spiral,:center)
         for rep in (:power_series,:eigenvalue)]))
    Set(keys(models))==expected || error("Expected the 14 selected dual-study models")
    for selected in values(models)
        path=joinpath(root,selected["checkpoint"])
        open(io->bytes2hex(sha256(io)),path)==selected["checkpoint_sha256"] ||
            error("Frozen checkpoint changed: $path")
    end
    spec=design(name;n,ng,ntest)
    for rep in (:power_series,:eigenvalue)
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
