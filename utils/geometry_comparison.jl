module GeometryComparison
using ..PanelSpec
export paired_panels

"""Require matching chart definitions and place both heads on common heatmap scales."""
function paired_panels(power::PanelSet, eigen::PanelSet)
    power.meta["experiment"] == eigen.meta["experiment"] || error("Experiment mismatch")
    power.meta["mode"] == eigen.meta["mode"] || error("Training/evaluation mode mismatch")
    power.meta["design_sha256"] == eigen.meta["design_sha256"] || error("Experimental controls differ")
    length(power.panels)==length(eigen.panels) || error("Representation chart counts differ")
    panels=Panel[]
    for (p,e) in zip(power.panels,eigen.panels)
        (p.id,p.kind,p.xlabel,p.ylabel)==(e.id,e.kind,e.xlabel,e.ylabel) || error("Chart definitions differ: $(p.id)")
        for key in ("categories","row_labels","col_labels","x_values","y_values")
            get(p.data,key,nothing)==get(e.data,key,nothing) || error("Chart coordinates differ: $(p.id)/$key")
        end
        if p.kind===:lines
            [s["x"] for s in p.data["series"]]==[s["x"] for s in e.data["series"]] ||
                error("Curve sample coordinates differ: $(p.id)")
        end
        opts=deepcopy(p.opts)
        if p.kind===:heatmap
            values=vcat(reduce(vcat,p.data["matrix"]),reduce(vcat,e.data["matrix"]))
            values=filter(isfinite,Float64.(values))
            get(opts,"log_color",false) && (values=log10.(max.(values,get(opts,"floor_val",1e-6))))
            lo,hi=extrema(values); hi==lo && (hi=lo+1.)
            opts["clims"]=[lo,hi]
        end
        if p.kind in (:lines,:grouped_bar)
            key=p.kind===:lines ? "y" : "values"
            values=Float64[v for panel in (p,e) for series in panel.data["series"] for v in series[key] if isfinite(v)]
            log_y=get(opts,"log_y",false)
            log_y && (values=filter(>(0),values))
            if !isempty(values)
                lo,hi=extrema(values)
                if log_y
                    opts["ylims"]=[lo/1.1,hi*1.1]
                else
                    p.kind===:grouped_bar && (lo=min(lo,0.); hi=max(hi,0.))
                    pad=max((hi-lo)*.05,1e-6)
                    opts["ylims"]=[lo-pad,hi+pad]
                end
            end
        end
        for (rep,panel) in (("power_series",p),("eigenvalue",e))
            push!(panels,Panel(panel.kind,"$rep/$(panel.id)",panel.title,panel.xlabel,panel.ylabel,
                deepcopy(panel.data),deepcopy(opts)))
        end
    end
    PanelSet("$(power.meta["experiment"]): power series vs eigenvalue",
        merge(deepcopy(power.meta),Dict("representation"=>"comparison","layout_columns"=>2)),panels)
end
end
