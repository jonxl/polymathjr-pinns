using Test, TOML, JSON, Lux, Zygote
include("../src/PINNStudies.jl")
const PS=PINNStudies
const P=PS.PINN
const E=PS.Experiments
const LF=P.loss_functions

@testset "Shared module identities" begin
    @test E.PINN === P
    @test PS.GeometryExperiments.Experiments === E
    @test PS.GeneralizationVariants.Experiments === E
    @test PS.Variants.PINN === P
    @test P.loss_functions.helper_funcs === PS.helper_funcs
end

@testset "Resolved configuration" begin
    c=PS.ExperimentConfiguration.load_config("configs/power_series_descending.toml")
    @test c["model"]["representation"] == "power_series"
    @test c["objective"]["coefficient_weighting"] == "descending"
    c["model"]["hidden_widths"] = [5,3]
    c["model"]["seed"] = 22
    c["data"]["seed"] = 11
    spec = PS.GeneralizationVariants.GeneralizationSpec(:showcase,:power_series_variation; config=c,maxiters=3)
    @test spec.representation === :power_series
    @test spec.coefficient_weighting === :descending
    cfg = E.ExperimentConfig(; PS.GeneralizationVariants.config_keywords(spec)...)
    @test cfg.model.hidden_widths == [5,3]
    @test cfg.seed == 22 && cfg.data_seed == 11
    @test cfg.loss.coefficient_weighting === :descending
    @test_throws ErrorException P.ModelConfig(hidden_widths=[0])
    @test_throws ErrorException P.LossConfig(coefficient_weighting=:unknown)
    mktempdir() do dir
        path=joinpath(dir,"bad.toml")
        write(path,"[training]\nmaxitres=1\n")
        @test_throws ErrorException PS.ExperimentConfiguration.load_config(path)
    end
end

# e^x solves u'' - u = 0. Analytic truth is independent of the truncation degree.
dataset=Dict{Any,Any}(reshape(Float32[-1,0,1],3,1)=>ones(Float32,9))
@testset "Model and loss choices preserve objectives" begin
    for rep in (:power_series,:power_series_variation,:eigenvalue)
        cfg=E.ExperimentConfig(representation=rep,N=4,neuron_count=4,maxiters=2,num_points=5)
        settings=E.settings_for(cfg,dataset)
        @test settings.representation == (rep == :eigenvalue ? :eigenvalue : :power_series)
        @test settings.loss.coefficient_weighting == (rep == :power_series_variation ? :descending : :uniform)
        net,p,st=P.initialize_network(settings)
        @test length(net.layers) == 4
        buf=E.buffers_for(settings,dataset)
        out=first(net(buf.X,p,st))
        components=rep == :eigenvalue ? LF.batched_eigenvalue_losses(out,buf) : LF.batched_power_series_losses(out,buf;variation=rep==:power_series_variation)
        total,_=P.global_loss_batched(p,settings,net,st,buf)
        @test total ≈ 5*components[1]+components[3]
        @test all(isfinite,only(Zygote.gradient(q->first(P.global_loss_batched(q,settings,net,st,buf)),p)))
    end
    for activation in (:sigmoid,:relu,:tanh)
        cfg=E.ExperimentConfig(N=4,model=P.ModelConfig(hidden_widths=[7,3],activation=activation))
        settings=E.settings_for(cfg,dataset)
        net,p,st=P.initialize_network(settings)
        @test length(net.layers)==3
        mktempdir() do dir
            path=joinpath(dir,"model.checkpoint")
            P.SafeTensorSnapshots.save_checkpoint(path,p,net,settings.seed)
            loaded,q,st2,_=P.SafeTensorSnapshots.load_checkpoint(path)
            x=zeros(Float32,P.io_dims(settings)[1])
            @test first(net(x,p,st)) ≈ first(loaded(x,q,st2))
        end
    end
end

@testset "Analytic reference and truncation" begin
    errors=Float32[]
    for N in (2,4,8)
        settings=E.settings_for(E.ExperimentConfig(N=N,num_points=10),dataset)
        buf=E.buffers_for(settings,dataset)
        truth=LF.analytic_solutions(buf,settings.xs)
        @test vec(truth) ≈ exp.(settings.xs)
        push!(errors,LF.relative_l2(LF.truncated_solutions(buf),truth))
    end
    @test errors[1]>errors[2]>errors[3]
    short = Dict{Any,Any}(reshape(Float32[-1,0,1],3,1)=>ones(Float32,2))
    settings=E.settings_for(E.ExperimentConfig(N=4,num_points=10),short)
    buf=E.buffers_for(settings,short)
    expected=[sum(x^n/factorial(n) for n in 0:4) for x in settings.xs]
    @test vec(LF.truncated_solutions(buf)) ≈ expected
    @test !(LF.truncated_solutions(buf) ≈ LF.batched_reconstruct(buf.DATA,buf))
end

@testset "Training artifacts and interruption" begin
    mktempdir() do dir
        cfg=E.ExperimentConfig(N=4,neuron_count=4,maxiters=2,num_points=5,metrics_interval=100)
        settings=E.settings_for(cfg,dataset)
        events=Symbol[]
        result=P.train_pinn(settings,dir;run_id="tiny",on_event=e->push!(events,e.kind))
        @test result.status == :completed
        @test result.iterations == 2
        @test length(result.history)==2
        @test :finished in events
        saved=TOML.parsefile(joinpath(dir,"config.toml"))
        @test saved["objective"]["pde_multiplier"]==5
        @test occursin("weighted_pde",readline(joinpath(dir,"loss.csv")))
        @test JSON.parsefile(joinpath(dir,"status.json"))["status"] == "completed"
        interrupted=P.train_pinn(settings,joinpath(dir,"interrupt");run_id="interrupt",
            on_event=e->e.kind === :metrics ? throw(InterruptException()) : nothing)
        @test interrupted.status == :interrupted
        @test_throws InterruptException collect(interrupted)
        @test !isfile(joinpath(dir,"interrupt","model.checkpoint"))
    end
end

@testset "Run isolation and aggregate failures" begin
    R=PS.RunManagement
    mktempdir() do dir
        a=R.create_run(Dict("test"=>true);root=dir)
        b=R.create_run(Dict("test"=>true);root=dir)
        @test a.root != b.root
        @test dirname(a.root)==dirname(b.root)
        @test_throws ErrorException R.create_run(Dict("test"=>true);output=a.root)
        jobs=[(output_dir=joinpath(a.root,"one"),device_id=1),(output_dir=joinpath(a.root,"two"),device_id=1)]
        done=String[]
        @test_throws ErrorException R.run_jobs(jobs;run=a) do job
            basename(job.output_dir)=="one" && error("deliberate test failure")
            push!(done,"two")
        end
        @test done==["two"]
        @test a.manifest["status"]=="failed"
        @test a.manifest["jobs"]["two"]["status"]=="completed"
    end
end
