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
        @test all(p->p[2]==p[1]^2/4,reduce(vcat,[c.points for c in design(name).cases]))
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
