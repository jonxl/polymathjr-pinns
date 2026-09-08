# Geometry and analytic scoring for the experiments introduced in bb2a102a.
# No ML dependencies: designs can be inspected/tested without loading CUDA.
module GeneralizationGeometry
using Random, Statistics
export EXPERIMENTS, design, disk, exact_solution, point_errors, reach_summary, region
const EXPERIMENTS = (:family_disks, :spiral_transfer, :pair_transfer,
    :parabola_segment, :parabola_slide, :parabola_curves, :parabola_bias,
    :axis_transfer, :axis_sweep)
const CENTERS = [:saddle => (0., -2.25), :stable_node => (-3.45, 1.05),
    :unstable_node => (3.45, 1.05), :stable_spiral => (-1.55, 2.9),
    :unstable_spiral => (1.55, 2.9), :center => (0., 2.25)]
region(t, d) = d < 0 ? :saddle : d == 0 ? :degenerate : t == 0 ? :center :
    t*t == 4d ? :star : t*t > 4d ? (t < 0 ? :stable_node : :unstable_node) :
    (t < 0 ? :stable_spiral : :unstable_spiral)
function disk(c, radius, n)
    n > 0 && radius > 0 || error("Positive point count and radius required")
    [(c[1] + radius*sqrt((i-.5)/n)*cos(i*π*(3-sqrt(5))),
      c[2] + radius*sqrt((i-.5)/n)*sin(i*π*(3-sqrt(5)))) for i in 1:n]
end
patch(reg, c, radius, n) = reg === :center ?
    [(0., d) for d in range(c[2]-radius, c[2]+radius, length=n)] : disk(c, radius, n)
parabola(a, b, n) = [(t, t*t/4) for t in range(a,b,length=n)]
case(label, points; center=nothing, radius=nothing, window=nothing) =
    (; label=String(label), points, center, radius, window)

"""Deterministic matched training and probe coordinates, independent of representation.
`n` is the point budget per patch/segment; unions contain multiple patches.
Placement studies exclude the center ray and origin (neither has a 2-D interior).
"""
function design(name::Symbol; n=200, ng=31, ntest=400, placements=0, seed=1234)
    name in EXPERIMENTS || error("Unknown geometry experiment: $name")
    n >= 2 && ng >= 2 && ntest >= 2 && placements >= 0 || error("Invalid geometry sizes")
    cases = NamedTuple[]; probes = Pair{String,Vector{Tuple{Float64,Float64}}}[]
    limits = name === :family_disks ? 8. : name === :axis_sweep ? 4. :
             name === :axis_transfer ? 6. : name === :parabola_segment ? 5.5 : 4.5
    axis = collect(range(-limits, limits, length=ng))
    push!(probes, "plane" => [(t,d) for d in axis for t in axis])
    if name in (:family_disks, :spiral_transfer, :pair_transfer)
        centers = name === :spiral_transfer ?
            [:saddle=>(0.,-1.5), :stable_node=>(-3.,1.), :unstable_node=>(3.,1.),
             :stable_spiral=>(-1.,2.), :unstable_spiral=>(1.,2.), :center=>(0.,1.5)] : CENTERS
        radius = name === :family_disks ? .3 : .5
        patches = Dict(reg => patch(reg,c,radius,n) for (reg,c) in centers)
        for (reg, _) in centers
            push!(probes, String(reg) => patches[reg])
        end
        if name === :family_disks
            for (reg,c) in vcat(centers, [:origin=>(0.,0.)])
                push!(cases, case(reg, patch(reg,c,radius,n); center=c, radius))
            end
            rng = MersenneTwister(seed)
            for (reg,_) in centers
                reg === :center && continue
                for j in 1:placements
                    found = false
                    for _ in 1:100_000
                        c = Tuple(7 .* rand(rng,2) .- 3.5)
                        pts = disk(c,radius,n)
                        if all(p -> region(p...) === reg, pts)
                            push!(cases,case("$(reg)-placement-$j",pts; center=c,radius))
                            found = true; break
                        end
                    end
                    found || error("Unable to place a disk in $reg")
                end
            end
        else
            push!(cases,case("spirals",vcat(patches[:stable_spiral],patches[:unstable_spiral])))
            name === :pair_transfer && push!(cases,case("nodes",vcat(patches[:stable_node],patches[:unstable_node])))
        end
    elseif startswith(String(name), "parabola")
        centers, width, extent = name === :parabola_segment ? ([0.],1.5,4.5) :
            name === :parabola_slide ? (vcat(-reverse(collect(.75:.75:3.75)),collect(.75:.75:3.75)),.75,4.5) :
            name === :parabola_curves ? (collect(-3.:.75:3.),.3,20.) : (collect(-3.5:.5:3.5),.5,20.)
        for c in centers
            push!(cases,case("tau-$c",parabola(c-width,c+width,n); window=(c-width,c+width)))
        end
        push!(probes,"parabola" => parabola(-extent,extent,ntest))
        push!(probes,"left" => [(-extent,extent^2/4)])
        push!(probes,"right" => [(extent,extent^2/4)])
    else
        positions = name === :axis_transfer ? collect(-3.:1.5:3.) : collect(-3.:.75:3.)
        for ax in (:x,:y)
            centers = [ax === :x ? (p,0.) : (0.,p) for p in positions]
            if name === :axis_transfer
                push!(cases,case(ax,reduce(vcat,[disk(c,.5,n) for c in centers])))
            else
                for (p,c) in zip(positions,centers)
                    push!(cases,case("$ax-$p",disk(c,.5,n); center=c,radius=.5))
                end
            end
            for (p,c) in zip(positions,centers)
                push!(probes,"$ax-$p" => disk(c,.5,n))
            end
        end
    end
    (; name, cases, probes, axis, seed, n, placements)
end

# Float64 analytic reference, including the repeated-root limit k=0.
function exact_solution(t, d, x; A=1., derivative=0.)
    mu = Float64(t)/2; k = mu^2-Float64(d); B = derivative-mu*A
    if k > 0
        q = sqrt(k)
        return .5*((A+B/q)*exp((mu+q)*x)+(A-B/q)*exp((mu-q)*x))
    elseif k < 0
        q = sqrt(-k)
        return exp(mu*x)*(A*cos(q*x)+B*sin(q*x)/q)
    end
    exp(mu*x)*(A+B*x)
end
function point_errors(outputs, representation, points, xs)
    size(outputs,2) == length(points) || error("Output/point count mismatch")
    errors = Float64[]
    for (i,(t,d)) in enumerate(points)
        o = Float64.(outputs[:,i])
        predicted = if representation === :power_series
            [evalpoly(x,o) for x in xs]
        elseif representation === :eigenvalue
            mu,k,A,B = o
            [exact_solution(2mu,mu^2-k,x; A,derivative=mu*A+B) for x in xs]
        else
            error("Unknown representation $representation")
        end
        truth = [exact_solution(t,d,x) for x in xs]
        # Diverged predictions count as failures, never successful reach.
        e = sqrt(sum(abs2,predicted.-truth)/(sum(abs2,truth)+1e-12))
        push!(errors,isfinite(e) ? e : Inf)
    end
    errors
end

"""First threshold failure on each ordered ray; capped rays are marked censored."""
function reach_summary(errors, radii, nangles; tolerance=.1, radius=0.)
    length(errors) == length(radii)*nangles || error("Reach grid size mismatch")
    reaches = Float64[]; censored = Bool[]
    for ray in eachcol(reshape(errors,length(radii),nangles))
        bad = findfirst(e -> !isfinite(e) || e > tolerance,ray)
        push!(reaches,bad === nothing ? last(radii) : radii[bad])
        push!(censored,bad === nothing)
    end
    (; reaches, censored, median_reach=median(reaches), margin=median(reaches)-radius)
end
end
