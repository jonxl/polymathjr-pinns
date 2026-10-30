using Test
include("../utils/generalization_geometry.jl")
using .GeneralizationGeometry

@testset "Matched geometry designs" begin
    counts=(7,1,2,1,10,9,15,2,18)
    for (name,count) in zip(EXPERIMENTS,counts)
        a=design(name;n=20,ng=5,ntest=20)
        @test a==design(name;n=20,ng=5,ntest=20)
        @test length(a.cases)==count
        @test all(c->length(c.points)>=20,a.cases)
        plane=Dict(a.probes)["plane"]
        @test plane[2]==(a.axis[2],a.axis[1])
        @test plane[6]==(a.axis[1],a.axis[2])
    end
    family=design(:family_disks;n=100,placements=1)
    @test length(family.cases)==12
    for c in family.cases
        c.label=="origin" && continue
        reg=Symbol(first(split(c.label,"-placement")))
        @test all(p->region(p...)==reg,c.points)
    end
    for name in (:parabola_segment,:parabola_slide,:parabola_curves,:parabola_bias)
        @test all(p->isapprox(p[2],p[1]^2/4;atol=1e-6),reduce(vcat,[c.points for c in design(name).cases]))
    end
    @test_throws ErrorException design(:bogus)
    @test_throws ErrorException design(:axis_transfer;n=0)
end
@testset "Common analytic solution metric" begin
    xs=collect(range(0.,1.,length=101))
    for (t,d) in ((0.,1.),(0.,-1.),(-2.,1.),(2.,1.),(-20.,100.),(20.,100.))
        @test exact_solution(t,d,0.)≈1.
        mu=t/2; k=mu^2-d
        @test only(point_errors(reshape([mu,k,1.,-mu],4,1),:eigenvalue,[(t,d)],xs))<1e-12
    end
    @test exact_solution(0.,1.,1.)≈cos(1.)
    @test exact_solution(0.,-1.,1.)≈cosh(1.)
    @test exact_solution(2.,1.,.5)≈.5exp(.5)
    @test only(point_errors(reshape([1.,0.,-.5],3,1),:power_series,[(0.,1.)],xs))>0
    @test only(point_errors(reshape([1.,0.],2,1),:power_series,[(2.,0.)],xs))<1e-12
    @test isinf(only(point_errors(fill(NaN,4,1),:eigenvalue,[(0.,1.)],xs)))
end
@testset "Reach stops at first failure" begin
    radii=[0.,.1,.2,.3]
    r=reach_summary([0.,.05,.2,0., 0.,0.,0.,0., NaN,0.,0.,0.],radii,3;radius=.1)
    @test r.reaches==[.2,.3,0.]
    @test r.censored==[false,true,false]
    @test r.margin≈.1
end

@testset "Jeet dataset controls" begin
    budgets=[400,400,400,300,200,80,120,300,250]
    grids=[161,81,91,81,81,81,81,121,91]
    case_counts=[7,1,2,1,10,9,15,2,18]
    multipliers=[1,2,2,1,1,1,1,5,1]
    for (name,n,ng,ncases,multiple) in zip(EXPERIMENTS,budgets,grids,case_counts,multipliers)
        spec=design(name)
        @test spec.n==n
        @test spec.ng==ng
        @test length(spec.cases)==ncases
        @test all(c->length(c.points)==n*multiple,spec.cases)
        @test all(c->length(unique(c.points))==length(c.points),spec.cases)
        @test all(c->all(p->all(x->Float64(Float32(x))==x,p),c.points),spec.cases)
    end
    @test design(:parabola_slide).ntest==300
    @test design(:parabola_curves).ntest==800
    @test extrema(design(:spiral_transfer).axis)==(-4.,4.)
    # Independent reference sunflower calculation, including Float32 rounding.
    c=(-1.55f0,2.9f0); R=.3f0; n=400; ga=Float32(π*(3-sqrt(5)))
    reference=[(Float64(c[1]+R*sqrt((Float32(i)-.5f0)/n)*cos(ga*i)),
                Float64(c[2]+R*sqrt((Float32(i)-.5f0)/n)*sin(ga*i))) for i in 1:n]
    @test disk(c,R,n)==reference
end
