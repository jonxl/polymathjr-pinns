module RunManagement
using Dates, UUIDs, JSON, TOML, Logging, SHA
export RunContext, create_run, job_status!, finish_run!, run_jobs, with_run_logging, new_run_directory, attach_run
mutable struct RunContext
    root::String
    manifest::Dict{String,Any}
    lock::ReentrantLock
end
function persist!(run)
    path=joinpath(run.root,"manifest.json")
    tmp=path*".tmp"
    open(tmp,"w") do io; JSON.print(io,run.manifest,2); end
    mv(tmp,path;force=true)
end
function new_run_directory(;root="results",date=Dates.today())
    parent=joinpath(root,string(date)); mkpath(parent)
    dir=joinpath(parent,"run-"*Dates.format(now(),"HHMMSS")*"-"*string(uuid4())[1:8])
    mkdir(dir)
    return dir
end
"Attach status tracking only after a legacy runner has validated its immutable configuration."
function attach_run(root)
    path=joinpath(root,"manifest.json")
    manifest=isfile(path) ? JSON.parsefile(path) : Dict{String,Any}("schema_version"=>1,"created_at"=>string(now()),"jobs"=>Dict{String,Any}())
    manifest["status"]="running"
    run=RunContext(root,manifest,ReentrantLock()); persist!(run)
    return run
end
function create_run(config; root="results", date=Dates.today(), output=nothing)
    dir=output === nothing ? new_run_directory(;root,date) : String(output)
    ispath(dir) && (!isdir(dir) || !isempty(readdir(dir))) && error("Output root is not empty: $dir")
    mkpath(dir)
    mkdir(joinpath(dir,".run-owner")) # Atomic reservation, including explicit empty output paths.
    revision=try readchomp(`git rev-parse HEAD`) catch; "unknown" end
    dirty=try !isempty(readchomp(`git status --porcelain`)) catch; true end
    run=RunContext(dir,Dict{String,Any}("schema_version"=>1,"status"=>"running", "created_at"=>string(now()),
      "source_revision"=>revision,"source_dirty"=>dirty,"jobs"=>Dict{String,Any}(),"report"=>"report"),ReentrantLock())
    open(joinpath(dir,"config.toml"),"w") do io; TOML.print(io,config); end
    source_root=normpath(joinpath(@__DIR__,".."))
    files=String["Project.toml","Manifest.toml"]
    for folder in ("src","utils","architectures","scripts","configs","viz")
        for (parent,_,names) in walkdir(joinpath(source_root,folder)), name in names
            any(ext->endswith(name,ext),(".jl",".py",".sh",".toml",".css",".html",".ts",".js")) && push!(files,relpath(joinpath(parent,name),source_root))
        end
    end
    hashes=Dict{String,String}()
    for file in files
        source=joinpath(source_root,file)
        isfile(source) || continue
        target=joinpath(dir,"source",file); mkpath(dirname(target)); cp(source,target)
        hashes[file]=bytes2hex(sha256(read(source)))
    end
    run.manifest["source_files"]=hashes
    persist!(run)
    return run
end
function with_run_logging(f,run)
    open(joinpath(run.root,"run.log"),"a") do io
        redirect_stdout(io) do
            redirect_stderr(io) do
                with_logger(ConsoleLogger(io)) do
                    f()
                end
            end
        end
    end
end
function job_status!(run, id, status; error=nothing)
    lock(run.lock) do
        run.manifest["jobs"][id]=Dict("status"=>String(status),"updated_at"=>string(now()),"error"=>error)
        persist!(run)
    end
end
function finish_run!(run)
    lock(run.lock) do
        states=[x["status"] for x in values(run.manifest["jobs"])]
        run.manifest["status"]=any(==("failed"),states) ? "failed" : any(==("interrupted"),states) ? "interrupted" : all(==("completed"),states) ? "completed" : "incomplete"
        run.manifest["finished_at"]=string(now()); persist!(run)
    end
    return run.manifest["status"]
end
"Run independent jobs to completion; report aggregate failure only after all workers finish."
function run_jobs(worker, jobs; run=nothing, devices=1)
    locks=[ReentrantLock() for _ in 1:devices]
    outcomes=fill(:pending,length(jobs))
    for job in jobs
        run === nothing || job_status!(run,basename(job.output_dir),:pending)
    end
    tasks=map(collect(enumerate(jobs))) do (i,job)
        Threads.@spawn lock(locks[job.device_id]) do
            id=basename(job.output_dir)
            run === nothing || job_status!(run,id,:running)
            try
                worker(job)
                outcomes[i]=:completed
                run === nothing || job_status!(run,id,:completed)
            catch e
                outcomes[i]=e isa InterruptException ? :interrupted : :failed
                @error "Job failed" job=id exception=(e,catch_backtrace())
                run === nothing || job_status!(run,id,outcomes[i];error=sprint(showerror,e))
            end
        end
    end
    foreach(fetch,tasks)
    run === nothing || finish_run!(run)
    all(==(:completed),outcomes) || error("$(count(!=(:completed),outcomes)) job(s) failed or interrupted; other independent jobs finished")
    return jobs
end
end
