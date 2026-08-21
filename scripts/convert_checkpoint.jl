#=
Raw `.checkpoint` -> shared `.safetensors` conversion step.

Converts the native Julia-Serialization checkpoint artifacts produced by training
into Hugging Face safetensors files for sharing/interop. Training itself never
writes safetensors; run this script explicitly when you need the shared format.

Usage:
    julia --project=. scripts/convert_checkpoint.jl <dir>
        Convert every *.checkpoint in <dir> (e.g. results/run-*/snapshots).

    julia --project=. scripts/convert_checkpoint.jl <in.checkpoint>
        Convert a single file to a sibling .safetensors.

    julia --project=. scripts/convert_checkpoint.jl <in.checkpoint> <out.safetensors>
        Convert a single file to an explicit output path.
=#

include("../utils/safetensors_utils.jl")
using .SafeTensorSnapshots

function checkpoint_to_safetensors_path(src::String)
  if endswith(src, ".checkpoint")
    return src[1:end-length(".checkpoint")] * ".safetensors"
  end
  return src * ".safetensors"
end

function convert_one(src::String, dst::String)
  @info "Converting" src=src dst=dst
  SafeTensorSnapshots.convert_to_safetensors(src, dst)
end

function convert_dir(dir::String)
  isdir(dir) || error("Not a directory: $dir")
  files = sort(filter(f -> endswith(f, ".checkpoint"), readdir(dir)))
  isempty(files) && error("No .checkpoint files found in $dir")
  for f in files
    convert_one(joinpath(dir, f), joinpath(dir, checkpoint_to_safetensors_path(f)))
  end
end

args = ARGS
isempty(args) && error(
  "Usage: julia --project=. scripts/convert_checkpoint.jl <dir|in.checkpoint> [out.safetensors]")

if length(args) == 1
  a = args[1]
  if isdir(a)
    convert_dir(a)
  else
    isfile(a) || error("Not a file or directory: $a")
    convert_one(a, checkpoint_to_safetensors_path(a))
  end
elseif length(args) == 2
  isfile(args[1]) || error("Input file not found: $(args[1])")
  convert_one(args[1], args[2])
else
  error("Too many arguments.")
end

@info "Done."
