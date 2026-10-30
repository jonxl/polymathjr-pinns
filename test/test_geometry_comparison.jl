using Test
include("../viz/PanelSpec.jl")
using .PanelSpec
include("../utils/geometry_comparison.jl")
using .GeometryComparison
@testset "Paired charts share geometry and colour scales" begin
    meta=Dict{String,Any}("experiment"=>"family_disks","mode"=>"matched_training","design_sha256"=>"same")
    p=heatmap_panel("plane","Power series",[.001 .01; .1 1.],["a","b"],["a","b"])
    e=heatmap_panel("plane","Eigenvalue",[.01 .1; 1. 10.],["a","b"],["a","b"])
    paired=paired_panels(PanelSet("ps",meta,[p]),PanelSet("eig",meta,[e]))
    @test length(paired.panels)==2
    @test paired.panels[1].opts["clims"]==paired.panels[2].opts["clims"]==[-3.,1.]
    @test paired.panels[1].data==p.data
    @test paired.panels[2].data==e.data
    @test_throws ErrorException paired_panels(PanelSet("ps",meta,[p]),PanelSet("eig",merge(meta,Dict("design_sha256"=>"different")),[e]))
    @test_throws ErrorException paired_panels(PanelSet("ps",meta,[p]),PanelSet("eig",meta,Panel[]))
    shifted=heatmap_panel("plane","Eigenvalue",ones(2,2),["a","b"],["x","y"])
    @test_throws ErrorException paired_panels(PanelSet("ps",meta,[p]),PanelSet("eig",meta,[shifted]))
end
@testset "Curve and bar comparisons share value scales" begin
    meta=Dict{String,Any}("experiment"=>"parabola_bias","mode"=>"matched_training","design_sha256"=>"same")
    p=lines_panel("bias","Power series",[("bias",[-1.,1.],[-.2,.1])])
    e=lines_panel("bias","Eigenvalue",[("bias",[-1.,1.],[-.5,.3])])
    paired=paired_panels(PanelSet("ps",meta,[p]),PanelSet("eig",meta,[e]))
    @test paired.meta["layout_columns"]==2
    @test paired.panels[1].opts["ylims"]==paired.panels[2].opts["ylims"]
    @test paired.panels[1].opts["ylims"][1]<-.5
    @test paired.panels[1].opts["ylims"][2]>.3
    shifted=lines_panel("bias","Eigenvalue",[("bias",[-2.,2.],[-.5,.3])])
    @test_throws ErrorException paired_panels(PanelSet("ps",meta,[p]),PanelSet("eig",meta,[shifted]))
end
