# Pure figure-contract tests. No network, optimizer, checkpoint, or CUDA module.
using Test, JSON
include("../viz/PanelSpec.jl")
using .PanelSpec
include("../utils/generalization_geometry.jl")
using .GeneralizationGeometry
include("../utils/geometry_panels.jl")
using .GeometryPanels
include("../utils/geometry_comparison.jl")
using .GeometryComparison

@testset "Grid area and training footprint" begin
    @test GeometryPanels.threshold_area(fill(0.,9),[-1.,0.,1.])≈4.
    @test GeometryPanels.threshold_area(fill(.2,9),[-1.,0.,1.])==0.
    @test GeometryPanels.threshold_area([nothing,NaN,Inf,0.,0.,0.,0.,0.,0.],[-1.,0.,1.])≈3.
    @test_throws ErrorException GeometryPanels.threshold_area([0.],[-1.,1.])
    spec=design(:family_disks;n=8,ng=5,ntest=8)
    center=only(filter(c->c.label=="center",spec.cases))
    @test GeometryPanels.training_patch(center)["kind"]=="segment"
    @test !GeometryPanels.outside_patch(center,center.center...)
    @test GeometryPanels.outside_patch(center,.1,center.center[2])
    saddle=first(spec.cases)
    @test !GeometryPanels.outside_patch(saddle,saddle.center...)
    @test GeometryPanels.outside_patch(saddle,saddle.center[1]+1.,saddle.center[2])
end

# Synthetic values exercise export/plotting only; these are NOT trained results.
spec=design(:family_disks;n=16,ng=21,ntest=8)
function fixture_results(spec, multiplier)
    Dict(c.label=>Dict(
        "training"=>Dict("mean_rel_l2"=>.02multiplier),
        "probes"=>Dict("plane"=>Dict("errors"=>[multiplier*(.01+.05hypot(t-c.center[1],d-c.center[2])) for d in spec.axis for t in spec.axis])),
        "diagnostics"=>Dict(
            "local_plane"=>Dict("tau_axis"=>[c.center[1]-1.,c.center[1],c.center[1]+1.],
                "delta_axis"=>[c.center[2]-1.,c.center[2],c.center[2]+1.],
                "score"=>Dict("errors"=>collect(1:9).*multiplier./100)),
            "reach"=>Dict("radii"=>[.5,.7,8.,.6],"censored"=>[false,false,true,false],"step"=>.04,"cap"=>8.,"tolerance"=>.1,"margin"=>.35),
            "shells_by_region"=>Dict("saddle"=>Dict("radius"=>[.2,.6,1.],"mean_rel_l2"=>[.02,.09,.2].*multiplier))
        )) for c in spec.cases)
end
sets=PanelSet[]
@testset "Disk figure coverage and pairing without training" begin
    for (rep,multiplier) in ((:power_series,1.),(:eigenvalue,1.5))
        ps=panels_for(spec,fixture_results(spec,multiplier),rep,"matched_training")
        ps.meta["design_sha256"]="synthetic-test-only"
        ps.meta["fixture"]="Synthetic renderer test, not trained results"
        push!(sets,ps)
        ids=[p.id for p in ps.panels]
        @test all(id->id in ids,["saddle-local-plane","saddle-plane","saddle-shells","saddle-directional-reach","geometry-areas","outside-patch-area","training-solution-error","reach-margins"])
        plane=only(filter(p->p.id=="saddle-plane",ps.panels))
        @test length(plane.data["training_patch"]["points"])==16
        @test plane.data["training_patch"]["radius"]==.3
        center=only(filter(p->p.id=="center-plane",ps.panels))
        @test center.data["training_patch"]["kind"]=="segment"
        radial=only(filter(p->p.id=="saddle-shells",ps.panels))
        @test radial.opts["vertical_lines"][1]["value"]==.3
        @test radial.opts["horizontal_lines"][1]["value"]==.1
        reach=only(filter(p->p.id=="saddle-directional-reach",ps.panels))
        @test reach.data["censored"]==[false,false,true,false]
    end
    paired=paired_panels(sets...)
    @test length(paired.panels)==2length(sets[1].panels)
    @test paired.panels[1].data["training_patch"]==sets[1].panels[1].data["training_patch"]
    # Same geometry, but captions may contain representation-specific evidence.
    sets[2].panels[1].opts["caption"]="Eigenvalue-specific note"
    @test paired_panels(sets...).panels[2].opts["caption"]=="Eigenvalue-specific note"
end

if !isempty(ARGS)
    for (rep,ps) in zip(("power_series","eigenvalue"),sets)
        save_panels(joinpath(ARGS[1],"family_disks",rep,"panelset.json"),ps)
    end
    println("Wrote synthetic figure fixtures only; no training executed")
end
