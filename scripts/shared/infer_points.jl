#!/usr/bin/env julia

using JSON

using PINNStudies: DualStudyEvaluation
using .DualStudyEvaluation

function usage()
  println("""
  Evaluate new (tau, delta) points with all 14 frozen selected models.

  Usage:
    julia --project=. scripts/shared/infer_points.jl STUDY_ROOT POINTS.json OUTPUT.json

  POINTS.json may be either a JSON array or an object with a `points` array:
    [{"name":"new-1", "tau":1.25, "delta":0.5}]

  For a finalized study, write OUTPUT.json outside STUDY_ROOT so that adding a
  derived result does not invalidate the original bundle manifest.
  """)
end

function main(args=ARGS)
  length(args) == 3 || (usage(); return 1)
  root, input_path, output_path = args
  input = JSON.parsefile(input_path)
  points = input isa AbstractVector ? input : get(input, "points", nothing)
  points isa AbstractVector || error("POINTS.json must be an array or contain a `points` array")
  run_custom_inference(root, points, output_path)
  println("Custom frozen-model inference written to $output_path")
  return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
  exit(main())
end
