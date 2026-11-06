# Lightweight allocator shared with the main run manager (no training imports).
include("../../utils/run_management.jl")
print(RunManagement.new_run_directory())
