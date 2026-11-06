module TrainingEvents
using CSV, DataFrames, JSON, TOML
export TrainingResult, write_training_artifacts, write_resolved_settings

struct TrainingResult{P,N,S,H}
    parameters::P
    model::N
    state::S
    run_id::String
    history::H
    status::Symbol
    iterations::Int
end
# Preserve legacy tuple destructuring, but never silently treat interrupted work as complete.
function Base.iterate(r::TrainingResult, i=1)
    r.status === :completed || throw(InterruptException())
    values=(r.parameters,r.model,r.state,r.run_id,r.history)
    i > 5 ? nothing : (values[i],i+1)
end
Base.length(::TrainingResult)=5

function write_resolved_settings(dir, s; batch_size, metrics_interval, checkpoint_epochs)
    mkpath(dir)
    cfg=Dict(
      "model"=>Dict("seed"=>s.seed,"representation"=>String(s.representation), "hidden_widths"=>s.model.hidden_widths,
        "activation"=>String(s.model.activation), "degree"=>s.n_terms_for_power_series, "input_encoding"=>String(s.input_encoding)),
      "objective"=>Dict("coefficient_weighting"=>String(s.loss.coefficient_weighting),
        "pde_weight"=>s.pde_weight, "supervised_weight"=>s.supervised_weight,
        "pde_multiplier"=>s.pde_weight*s.num_supervised, "num_supervised"=>s.num_supervised,
        "formula"=>"pde_multiplier * pde + supervised_weight * supervised", "diagnostic_terms"=>["bc"]),
      "data"=>Dict("dataset_size"=>length(s.ode_matrices),"num_points"=>s.num_points,"x_left"=>s.x_left,"x_right"=>s.x_right),
      "training"=>Dict("optimizer"=>"adam","learning_rate"=>0.001,"maxiters"=>s.maxiters_lbfgs,
        "batch_size"=>batch_size,"metrics_interval"=>metrics_interval,"checkpoint_epochs"=>checkpoint_epochs))
    open(joinpath(dir,"config.toml"),"w") do io; TOML.print(io,cfg); end
end
function write_training_artifacts(dir, event; write_loss_csv=true)
    event.kind === :finished || return
    r=event.result
    open(joinpath(dir,"status.json"),"w") do io
        JSON.print(io,Dict("status"=>String(r.status),"iterations"=>r.iterations,"run_id"=>r.run_id),2)
    end
    if write_loss_csv && !isempty(r.history)
        CSV.write(joinpath(dir,"loss.csv"),DataFrame(
          iteration=event.iterations,total=[h.total for h in r.history],
          bc=[h.bc for h in r.history],pde=[h.pde for h in r.history],supervised=[h.supervised for h in r.history],
          weighted_pde=[h.pde*event.pde_multiplier for h in r.history],
          weighted_supervised=[h.supervised*event.supervised_weight for h in r.history]))
    end
end
end
