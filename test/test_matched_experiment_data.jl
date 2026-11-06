using Test
include("../utils/dual_dataset.jl")
include("../utils/loss_functions.jl")
using .DualDataset
const LF = loss_functions

@testset "Three experiments share ODEs, initial conditions and reference solutions" begin
    mktempdir() do root
        for (split_name, seed) in ((:train, 10001), (:validation, 20001), (:test, 30001))
            recipe = CanonicalODESplit(split_name, 37, seed, 20; shuffle_seed=seed+30000)
            stored = materialize_split(recipe, joinpath(root, String(split_name)))
            # Independent readers model jobs running on separate devices/processes.
            readers = [load_materialized_split(stored.root, recipe) for _ in 1:3]
            @test all(dataset_id(r) == dataset_id(stored) for r in readers)
            for region in (nothing, DualDataset.REGIONS...), epoch in (1, 2)
                get_items(r) = split_name === :train ?
                    (region === nothing ? batch_items(r, epoch, 1, 8) :
                     family_batch_items(r, region, epoch, 1, 8)) :
                    evaluation_batch_items(r, 1, 8; region=region)
                batches = get_items.(readers)
                @test batches[1] == batches[2] == batches[3]
                buffers = map(zip((:power_series, :eigenvalue, :power_series_variation), batches)) do (rep, items)
                    settings = (representation=rep, n_terms_for_power_series=20,
                        num_supervised=21, num_points=5, x_left=0f0,
                        xs=Float32[0, .25, .5, .75, 1],
                        ode_matrices=Dict(items), input_encoding=:trace_determinant)
                    rep === :eigenvalue ? LF.precompute_batch_eig_buffers(settings, items, false, identity) :
                        LF.precompute_batch_buffers(settings, items, false, identity)
                end
                default, eigen, variation = buffers
                @test default.X == eigen.X == variation.X
                @test default.BC == variation.BC == vcat(eigen.A0, eigen.A1)
                @test all(==(1f0), eigen.A0)
                @test all(==(0f0), eigen.A1)
                @test default.DATA == variation.DATA
                @test LF.true_solutions(default) ≈ LF.true_solutions(eigen)
                @test LF.true_solutions(default) == LF.true_solutions(variation)
            end
        end
    end
end
