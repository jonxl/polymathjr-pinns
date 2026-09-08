using Test, JSON
include("../utils/geometry_experiments.jl")
using .GeometryExperiments
const GE=GeometryExperiments
const E=GE.Experiments
@testset "Ordered batched analytic scoring for both heads" begin
    points=[(2.,0.),(-2.,1.),(0.,1.)]
    dataset=E.Plugboard.generate_grid_dataset(first.(points),last.(points),20)
    cfg=E.ExperimentConfig(;representation=:eigenvalue)
    settings=E.settings_for(cfg,dataset;input_encoding=:trace_determinant)
    net=(X,p,st)->(vcat(X[1:1,:]./2, X[1:1,:].^2 ./4 .- X[2:2,:],
                       ones(Float32,1,size(X,2)), -X[1:1,:]./2),st)
    model=(;net,p=nothing,st=nothing,settings,to_device=identity)
    @test maximum(GE.predict_errors(model,points;batch_size=2))<1e-10
    scored=GE.score(model,points;batch_size=1)
    @test scored["points"]==GE.jsonpoints(points)
    @test scored["failed_points"]==0
    cfgps=E.ExperimentConfig(;representation=:power_series)
    settingsps=E.settings_for(cfgps,dataset;input_encoding=:trace_determinant)
    @test E.PINN.io_dims(settingsps)==(2,21)
    spec=GE.GeneralizationGeometry.design(:parabola_segment;n=2,ng=2,ntest=3)
    mktempdir() do dir
        GE.write_json(joinpath(dir,"design.json"),spec)
        @test JSON.parsefile(joinpath(dir,"design.json"))["name"]=="parabola_segment"
        GE.write_json(joinpath(dir,"cfg.json"),cfg)
        @test JSON.parsefile(joinpath(dir,"cfg.json"))["representation"]=="eigenvalue"
    end
end
@testset "Shared training smoke: both representations" begin
    mktempdir() do dir
        for rep in (:power_series,:eigenvalue)
            cfg=E.ExperimentConfig(;representation=rep,N=4,neuron_count=4,
                                  maxiters=1,n_per_region=2,num_points=3)
            result=GE.run_geometry(cfg,:parabola_segment;output_root=dir,ng=2,ntest=3,nx=5)
            @test length(result)==1
            path=joinpath(dir,"parabola_segment",String(rep))
            @test isfile(joinpath(path,"panelset.json"))
            data=JSON.parsefile(joinpath(path,"results.json"))
            @test data["mode"]=="matched_training"
            checkpoint=joinpath(path,data["models"]["tau-0.0"]["checkpoint"])
            @test isfile(checkpoint)
            net,p,st,metadata=E.PINN.SafeTensorSnapshots.load_any_model(checkpoint)
            @test size(first(net(zeros(Float32,2,1),p,st)),1)==(rep===:power_series ? 5 : 4)
        end
    end
end
