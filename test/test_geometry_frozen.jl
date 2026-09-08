using Test, JSON, SHA
include("../utils/geometry_experiments.jl")
const GE=GeometryExperiments
const E=GE.Experiments
@testset "Frozen geometry selection and bundle preservation" begin
    mktempdir() do base
        root=joinpath(base,"study"); out=joinpath(base,"derived")
        mkpath(root)
        GE.write_json(joinpath(root,"config.json"),Dict("training"=>Dict(
            "power_series_degree"=>4,"network_hidden_width"=>4,"model_seed"=>1234)))
        models=Dict{String,Any}()
        for rep in (:power_series,:eigenvalue)
            dataset=E.Plugboard.generate_grid_dataset([0.],[0.],4)
            cfg=E.ExperimentConfig(;representation=rep,N=4,neuron_count=4)
            settings=E.settings_for(cfg,dataset;input_encoding=:trace_determinant)
            net,p,st=E.PINN.initialize_network(settings;use_gpu=false)
            path=joinpath(root,"$rep.checkpoint")
            E.PINN.SafeTensorSnapshots.save_checkpoint(path,p,net,1234;representation=rep,
                extra_metadata=Dict("input_encoding"=>"trace_determinant"))
            digest=open(io->bytes2hex(sha256(io)),path)
            for scope in ("global","family/saddle","family/stable_node","family/unstable_node",
                          "family/stable_spiral","family/unstable_spiral","family/center")
                models["$scope/$rep"]=Dict("checkpoint"=>basename(path),"checkpoint_sha256"=>digest)
            end
        end
        GE.write_json(joinpath(root,"selection","selected-checkpoints.json"),
            Dict("selection_complete"=>true,"models"=>models))
        @test_throws ErrorException GE.run_frozen_geometry(root,:parabola_bias;
            output_root=joinpath(root,"forbidden"),n=2,ng=2,ntest=3,use_gpu=false)
        @test !ispath(joinpath(root,"forbidden"))
        before=sort!(readdir(root))
        GE.run_frozen_geometry(root,:parabola_bias;output_root=out,n=2,ng=2,ntest=3,nx=5,use_gpu=false)
        @test before==sort!(readdir(root))
        for rep in (:power_series,:eigenvalue)
            result=JSON.parsefile(joinpath(out,"parabola_bias",String(rep),"results.json"))
            @test result["mode"]=="frozen_probe"
            @test length(result["models"])==7
            @test all(m->!haskey(m,"training"),values(result["models"]))
        end
        open(joinpath(root,"power_series.checkpoint"),"a") do io
            write(io,"tampered")
        end
        @test_throws ErrorException GE.run_frozen_geometry(root,:axis_transfer;
            output_root=joinpath(base,"tamper-check"),n=2,ng=2,ntest=3,use_gpu=false)
        @test !ispath(joinpath(base,"tamper-check"))
    end
end
