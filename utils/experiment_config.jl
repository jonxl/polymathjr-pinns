module ExperimentConfiguration
using TOML
export ModelConfig, LossConfig, load_config, canonical_representation, coefficient_weighting

canonical_representation(rep::Symbol) = rep === :power_series_variation ? :power_series : rep
coefficient_weighting(rep::Symbol) = rep === :power_series_variation ? :descending : :uniform

struct ModelConfig
    hidden_widths::Vector{Int}
    activation::Symbol
    function ModelConfig(; hidden_widths=[64,64,64], activation=:sigmoid)
        !isempty(hidden_widths) && all(>(0), hidden_widths) || error("hidden_widths must be positive")
        activation in (:sigmoid, :tanh, :relu) || error("Unsupported activation: $activation")
        new(Int.(hidden_widths), Symbol(activation))
    end
end
struct LossConfig
    coefficient_weighting::Symbol
    ic_weight::Float32  # weight of the initial-condition term in the optimised objective
    function LossConfig(; coefficient_weighting=:uniform, ic_weight=1.0f0)
        coefficient_weighting in (:uniform, :descending) || error("Unknown coefficient weighting")
        isfinite(ic_weight) && ic_weight >= 0 || error("ic_weight must be finite and nonnegative")
        new(Symbol(coefficient_weighting), Float32(ic_weight))
    end
end

const DEFAULTS = Dict{String,Any}(
    "model" => Dict("representation"=>"power_series", "seed"=>1234, "hidden_widths"=>[64,64,64], "activation"=>"sigmoid", "degree"=>20, "input_encoding"=>"coefficients"),
    "objective" => Dict("coefficient_weighting"=>"uniform", "pde_weight"=>1.0, "ic_weight"=>1.0, "supervised_weight"=>1.0),
    "data" => Dict("seed"=>1234, "n_per_region"=>0, "num_points"=>50, "x_left"=>0.0, "x_right"=>1.0),
    "training" => Dict("maxiters"=>3000, "batch_size"=>0, "checkpoint_epochs"=>0, "metrics_interval"=>100),
    "run" => Dict("root"=>"results", "experiments"=>"all"))

function load_config(path=nothing)
    result=deepcopy(DEFAULTS)
    if path !== nothing
        for (section, values) in TOML.parsefile(path)
            haskey(result, section) || error("Unknown config section: $section")
            for (key, value) in values
                haskey(result[section], key) || error("Unknown configuration key: $section.$key")
                result[section][key]=value
            end
        end
    end
    m=result["model"]; o=result["objective"]; t=result["training"]; d=result["data"]
    rep=Symbol(m["representation"])
    if rep === :power_series_variation
        m["representation"]="power_series"; o["coefficient_weighting"]="descending"
    end
    m["representation"] in ("power_series", "eigenvalue") || error("Unknown representation")
    m["input_encoding"] in ("coefficients","trace_determinant") || error("Unknown input encoding")
    d["n_per_region"] >= 0 || error("n_per_region must be nonnegative (0 selects experiment defaults)")
    ModelConfig(hidden_widths=m["hidden_widths"], activation=Symbol(m["activation"]))
    LossConfig(coefficient_weighting=Symbol(o["coefficient_weighting"]), ic_weight=o["ic_weight"])
    m["representation"] == "eigenvalue" && o["coefficient_weighting"] != "uniform" && error("Coefficient weighting only applies to power series")
    m["degree"] >= 1 && m["degree"] <= 40 || error("degree must be in 1:40")
    t["maxiters"] > 0 && t["metrics_interval"] > 0 && t["batch_size"] >= 0 && t["checkpoint_epochs"] >= 0 || error("Invalid training limits")
    d["num_points"] > 0 && isfinite(d["x_left"]) && isfinite(d["x_right"]) && d["x_left"] < d["x_right"] || error("Invalid evaluation interval")
    all(v -> isfinite(v) && v >= 0, (o["pde_weight"],o["ic_weight"],o["supervised_weight"])) || error("Invalid objective weights")
    return result
end
end
