module GeometryPanels
using ..PanelSpec
using Statistics
export panels_for

function training_patch(c)
    Dict{String,Any}("points"=>[collect(p) for p in c.points],
        "center"=>c.center === nothing ? nothing : collect(c.center),
        "radius"=>c.radius, "window"=>c.window === nothing ? nothing : collect(c.window),
        "kind"=>startswith(c.label,"center") ? "segment" : c.center !== nothing ? "disk" : "points")
end

# Integrate a nodal success indicator with trapezoid weights: the full grid
# covers exactly its bounding square, including half-weight boundary nodes.
function threshold_area(errors, axis; tolerance=.1)
    n=length(axis)
    n>=2 && length(errors)==n*n || error("Area requires an ordered square grid")
    all(diff(axis).>0) || error("Area coordinates must be increasing")
    weights=[i==1 ? (axis[2]-axis[1])/2 : i==n ? (axis[n]-axis[n-1])/2 :
             (axis[i+1]-axis[i-1])/2 for i in 1:n]
    sum(weights[i]*weights[j] for j in 1:n, i in 1:n
        if errors[i+(j-1)*n] !== nothing && isfinite(errors[i+(j-1)*n]) && errors[i+(j-1)*n]<=tolerance; init=0.)
end

function annotate_curve!(panel, c, tolerance)
    panel.opts["horizontal_lines"]=[Dict("value"=>tolerance,"label"=>"relative L2 = $tolerance")]
    if c !== nothing && c.radius !== nothing
        label=startswith(c.label,"center") ? "segment half-length" : "training disk edge"
        panel.opts["vertical_lines"]=[Dict("value"=>c.radius,"label"=>label)]
    end
end

function outside_patch(c, t, d)
    c.center !== nothing && c.radius !== nothing || error("Patch has no radial extent")
    cx,cy=c.center
    startswith(c.label,"center") ? !(isapprox(t,cx;atol=1e-7) && abs(d-cy)<=c.radius) :
        hypot(t-cx,d-cy)>c.radius
end

function panels_for(spec, results, rep, mode)
    panels=Panel[]
    cases=Dict(c.label=>c for c in spec.cases)
    tolerance=spec.controls.tolerance
    matched=mode=="matched_training"
    for label in sort!(collect(keys(results)))
        scores=results[label]["probes"]
        plane=scores["plane"]
        # Raw JSON carries null plus failed_points; plots use a finite failure cap.
        es=[e === nothing ? 1e6 : e for e in plane["errors"]]
        push!(panels,heatmap_panel("$(label)-plane","$label ($rep)",
            permutedims(reshape(es,length(spec.axis),length(spec.axis))),
            string.(spec.axis),string.(spec.axis); xlabel="τ",ylabel="Δ",
            annotate=false,x_values=spec.axis,y_values=spec.axis,contour_levels=[tolerance],floor_val=1e-6,
            colorbar_label="log10 relative L2"))
        if matched && haskey(cases,label)
            panels[end].data["training_patch"]=training_patch(cases[label])
            panels[end].opts["caption"]="Dots are the actual training equations; the outline marks the training patch. The white contour marks relative solution error $tolerance when it crosses the grid."
        end
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
    for label in sort!(collect(keys(results)))
        diag=get(results[label],"diagnostics",Dict())
        if haskey(diag,"local_plane")
            local_plane=diag["local_plane"]
            tx=local_plane["tau_axis"]; dy=local_plane["delta_axis"]
            errors=[e === nothing ? 1e6 : e for e in local_plane["score"]["errors"]]
            p=heatmap_panel("$(label)-local-plane","$label: near training cloud ($rep)",
                permutedims(reshape(errors,length(tx),length(dy))),string.(tx),string.(dy);
                xlabel="τ",ylabel="Δ",annotate=false,x_values=tx,y_values=dy,
                contour_levels=[tolerance],floor_val=1e-6,colorbar_label="log10 relative L2")
            if matched && haskey(cases,label)
                p.data["training_patch"]=training_patch(cases[label])
            end
            p.opts["caption"]="Local evaluation around the training cloud: $(length(tx)) × $(length(dy)) points. Dots mark training equations; the white contour marks relative solution error $tolerance."
            push!(panels,p)
        end
        if haskey(diag,"shells_by_region")
            shells=diag["shells_by_region"]
            series=[(reg,shells[reg]["radius"],
                [e===nothing ? 1e6 : max(e,1e-12) for e in shells[reg]["mean_rel_l2"]])
                for reg in sort!(collect(keys(shells))) if !isempty(shells[reg]["radius"])]
            push!(panels,lines_panel("$(label)-shells","$label: radial extrapolation ($rep)",
                series;xlabel="distance from patch center",ylabel="mean relative L2",log_y=true))
            annotate_curve!(panels[end],get(cases,label,nothing),tolerance)
            panels[end].opts["caption"]="Mean relative solution error on radial shells, separated by ODE family. The vertical line marks the training extent; the horizontal line marks the common error tolerance."
        end
        if haskey(diag,"reach")
            reach=diag["reach"]
            angles=collect(0:length(reach["radii"])-1).* (360/length(reach["radii"]))
            p=lines_panel("$(label)-directional-reach","$label: directional reach ($rep)",
                [("first sampled threshold failure",angles,reach["radii"])];
                xlabel="direction (degrees)",ylabel="distance from patch center")
            p.data["censored"]=reach["censored"]
            c=get(cases,label,nothing)
            if c !== nothing && c.radius !== nothing
                p.opts["horizontal_lines"]=[Dict("value"=>c.radius,"label"=>startswith(c.label,"center") ? "segment half-length" : "training disk edge")]
            end
            p.opts["caption"]="Reach is the first sampled radius whose error exceeds $(reach["tolerance"]), with radial step $(reach["step"]). Triangles mark rays still accurate at the search cap $(reach["cap"]): their true reach is not resolved."
            push!(panels,p)
        end
    end
    if matched
        labels=sort!([label for label in keys(results) if haskey(cases,label)])
        if !isempty(labels)
            areas=[threshold_area(results[label]["probes"]["plane"]["errors"],spec.axis;tolerance) for label in labels]
            p=grouped_bar_panel("geometry-areas","Area below error tolerance ($rep)",labels,
                [("relative L2 ≤ $tolerance",areas)];ylabel="area in the evaluated plane",log_y=false)
            p.opts["show_values"]=true
            p.opts["caption"]="Area on the saved evaluation domain where relative solution error ≤ $tolerance, estimated with trapezoid-weighted grid samples. Failed predictions count as outside tolerance."
            push!(panels,p)
            radial=[label for label in labels if cases[label].center !== nothing]
            if !isempty(radial)
                areas_out=Float64[]
                for label in radial
                    errors=results[label]["probes"]["plane"]["errors"]
                    mask=[outside_patch(cases[label],t,d) for d in spec.axis for t in spec.axis]
                    push!(areas_out,threshold_area([mask[i] ? errors[i] : nothing for i in eachindex(errors)],spec.axis;tolerance))
                end
                p=grouped_bar_panel("outside-patch-area","Accurate area outside training patch ($rep)",radial,
                    [("outside patch, relative L2 ≤ $tolerance",areas_out)];ylabel="area in the evaluated plane",log_y=false)
                p.opts["show_values"]=true
                p.opts["caption"]="Trapezoid-weighted grid area meeting relative error ≤ $tolerance outside the training disk or segment. This measures area, while directional reach measures contiguous accuracy from the patch center."
                push!(panels,p)
            end
            trained=[label for label in labels if haskey(results[label],"training")]
            if !isempty(trained)
                means=[something(results[label]["training"]["mean_rel_l2"],1e6) for label in trained]
                p=grouped_bar_panel("training-solution-error","Solution error on training equations ($rep)",trained,
                    [("mean relative L2",means)];ylabel="mean relative solution error",log_y=true)
                p.opts["horizontal_lines"]=[Dict("value"=>tolerance,"label"=>"relative L2 = $tolerance")]
                p.opts["caption"]="Analytic solution error on the actual training equations after the fixed iteration budget; this is not the optimization loss. It reveals poor fitting before interpreting outside-patch error."
                push!(panels,p)
            end
        end
    end
    window_cases=[c for c in spec.cases if haskey(results,c.label) &&
        haskey(get(results[c.label],"diagnostics",Dict()),"left_reach")]
    if !isempty(window_cases)
        centers=[sum(c.window)/2 for c in window_cases]
        reaches(side)=[results[c.label]["diagnostics"][string(side,"_reach")]["distance"] for c in window_cases]
        left= reaches("left"); right=reaches("right")
        push!(panels,lines_panel("window-reach","Along-parabola reach ($rep)",
            [("outward",centers,[c<0 ? l : r for (c,l,r) in zip(centers,left,right)]),
             ("inward",centers,[c<0 ? r : l for (c,l,r) in zip(centers,left,right)])];
            xlabel="training window center τ",ylabel="reach beyond window"))
    end
    reach_labels=sort!([label for label in keys(results)
        if haskey(get(results[label],"diagnostics",Dict()),"reach")])
    if !isempty(reach_labels)
        push!(panels,grouped_bar_panel("reach-margins","Median reach beyond patch edge ($rep)",
            reach_labels,[("reach minus patch radius",
                [results[label]["diagnostics"]["reach"]["margin"] for label in reach_labels])];
            ylabel="distance",log_y=false))
        panels[end].opts["show_values"]=true
        panels[end].opts["horizontal_lines"]=[Dict("value"=>0.,"label"=>"training extent")]
        panels[end].opts["caption"]="Median first-failure radius minus the patch radius (segment half-length for center). Negative margins indicate failure within that extent. Capped rays are shown in the directional reach figures."
    end
    bias_cases=[c for c in spec.cases if haskey(results,c.label) &&
        haskey(get(results[c.label],"diagnostics",Dict()),"right_minus_left_bias")]
    if !isempty(bias_cases)
        push!(panels,lines_panel("parabola-bias","Fixed-endpoint right minus left error ($rep)",
            [("bias",[sum(c.window)/2 for c in bias_cases],
                [something(results[c.label]["diagnostics"]["right_minus_left_bias"],1e6) for c in bias_cases])];
            xlabel="training window center τ",ylabel="right minus left relative L2"))
    end
    PanelSet("$(spec.name) $rep ($mode)",Dict{String,Any}("representation"=>String(rep),
        "experiment"=>String(spec.name),"mode"=>mode,"source_commit"=>"bb2a102a",
        "truth"=>"Float64 analytic solution; y(0)=1, y'(0)=0",
        "plot_nonfinite_cap"=>1e6),panels)
end

end
