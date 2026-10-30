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
        paired=JSON.parsefile(joinpath(dir,"parabola_segment","comparison.json"))
        ps=JSON.parsefile(joinpath(dir,"parabola_segment","power_series","panelset.json"))
        eig=JSON.parsefile(joinpath(dir,"parabola_segment","eigenvalue","panelset.json"))
        @test length(paired["panels"])==2length(ps["panels"])
        @test [p["id"] for p in ps["panels"]]==[p["id"] for p in eig["panels"]]
        @test ps["meta"]["design_sha256"]==eig["meta"]["design_sha256"]
    end
end

@testset "Plugboard targets and matched-control enforcement" begin
    dataset=E.Plugboard.generate_grid_dataset([2.],[1.],4)
    derivatives=only(values(dataset))
    @test derivatives≈[1.,0.,-1.,-2.,-3.]
    cfg=E.ExperimentConfig(;N=4)
    settings=E.settings_for(cfg,dataset;input_encoding=:trace_determinant)
    buf=E.buffers_for(settings,dataset)
    @test vec(buf.X)==[2.,1.]
    @test vec(buf.DATA)≈[1.,0.,-.5,-1/3,-1/8]
    spec=GE.GeneralizationGeometry.design(:parabola_segment;n=2,ng=2)
    mktempdir() do root
        first_hash=GE.ensure_contract(root,spec,cfg,200)
        eigen=E.ExperimentConfig(;representation=:eigenvalue,N=4)
        @test GE.ensure_contract(root,spec,eigen,200)==first_hash
        changed=E.ExperimentConfig(;representation=:eigenvalue,N=4,seed=999)
        @test_throws ErrorException GE.ensure_contract(root,spec,changed,200)
        @test_throws ErrorException GE.ensure_contract(root,spec,eigen,100)
        @test_throws ErrorException GE.ensure_contract(root,GE.GeneralizationGeometry.design(:parabola_segment;n=3,ng=2),eigen,200)
    end
end
