#!/usr/bin/env julia

using PINNStudies: DualStudyEvaluation
using .DualStudyEvaluation

function usage()
  println("""
  Validation-select checkpoints or run the protected final test.

  Usage:
    julia --project=. scripts/shared/evaluate_study.jl select STUDY_ROOT [BATCH_SIZE]
    julia --project=. scripts/shared/evaluate_study.jl infer  STUDY_ROOT [BATCH_SIZE]
    julia --project=. scripts/shared/evaluate_study.jl suite  STUDY_ROOT

  `select` reads only the materialized validation split. `infer` refuses to run
  until selection is complete, verifies selected checkpoint checksums, and then
  reads the materialized final test split exactly once per requested metric.
  """)
end

length(ARGS) in (2, 3) || (usage(); exit(1))
mode = ARGS[1]
root = ARGS[2]
batch_size = length(ARGS) == 3 ? parse(Int, ARGS[3]) : 16_384

if mode == "select"
  select_checkpoints(root; evaluation_batch_size=batch_size)
elseif mode == "infer"
  run_final_test(root; evaluation_batch_size=batch_size)
elseif mode == "suite"
  run_generalization_suite(root)
else
  usage()
  error("Unknown mode: $mode")
end
