# Merge Plan

## Constraints

- Do not run project code, experiments, training, tests, Julia entrypoints, or visualization scripts during this merge.
- Preserve both representation concepts. Do not silently rewrite either branch's math.
- Keep the power-series and eigenvalue paths explicit and separate at representation boundaries.
- Drop boundary-condition loss from the optimized objective for this merge direction.
- Keep final model export as `model.safetensors`; intermediate checkpoints are optional checkpoint artifacts.

## Current State

- Branch: `jon/dual-representation`.
- Merge in progress from `origin/jeet-generalization-experiments`.
- No unresolved index conflicts remain.
- Staged changes contain the experiment branch: standalone power-series/eigenvalue scripts, README updates, dependency updates, and `.gitignore` changes.
- Unstaged changes contain integration work in core training, losses, data generation, checkpoints, and representation routing.
- Untracked files include shared experiment orchestration, panel viewer code, and shared JSON datasets.

## Decisions

1. Keep both representations:
   - `:power_series` stays the power-series representation.
   - `:eigenvalue` stays the eigenvalue/unified exponential representation.
   - Shared abstractions are acceptable only where they do not blur the math.

2. Drop BC from training objective:
   - Keep BC computation only as a diagnostic component if useful.
   - Remove `bc_weight` from settings, CLI/docs/output metadata if it is no longer used.
   - Make the objective formula explicit and concise.

3. Do not shard JSON:
   - Prefer one coherent JSON artifact per experiment/run when practical.
   - Memory pressure is not expected to come from JSON loading compared with training cost.
   - Only split JSON if a specific downstream consumer or file-size limit forces it.

4. Reduce comment noise:
   - Remove long explanatory blocks that restate code mechanics.
   - Keep short comments only where they protect representation semantics or merge intent.
   - Move broader conceptual explanation into docs if needed, not hot source files.

5. Review `.gitignore`:
   - Avoid broad rules like `/*.md` if they might hide plans/docs.
   - Keep generated runtime outputs ignored.
   - Decide explicitly whether shared JSON files are source fixtures or generated artifacts.

6. Do not commit yet:
   - First normalize the integration layer.
   - Then review staged vs unstaged changes.
   - Commit only after the merged tree has coherent representation boundaries and artifact policy.

## Work Plan

### Phase 1: Inventory Without Execution

- Inspect staged, unstaged, and untracked files with Git/read-only commands only.
- Identify all places where `bc_weight`, checkpointing, representation routing, and JSON outputs are defined.
- Identify generated artifacts versus source fixtures.

### Phase 2: Representation Boundary Cleanup

- Keep `PINNSettings.representation` or equivalent explicit routing.
- Ensure power-series and eigenvalue loss/reconstruction paths do not change each other's math.
- Keep shared batching/helpers only where they operate on representation-neutral tensors or dispatch cleanly by representation.

### Phase 3: Objective Cleanup

- Remove BC from the optimized total loss.
- Keep BC metric logging only if it remains cheap and clearly diagnostic.
- Remove stale `bc_weight` plumbing from settings, metadata, docs, and examples.
- Make output JSON identify the objective components that were actually optimized.

### Phase 4: Safetensors Artifact Policy

- Save final trained MLP weights at run root as `model.safetensors`.
- Save intermediate checkpoint weights under `snapshots/iter-NNNNNNN.safetensors` only when checkpointing is enabled.
- Preserve legacy `.bin` loading for old runs if it does not complicate current code.

### Phase 5: JSON/Data Policy

- Keep run/experiment results consolidated unless there is a proven file-size or consumer limit.
- Decide whether `data/shared_*.json` files are checked-in fixtures or generated outputs.
- If generated, ignore them; if fixtures, document their role.

### Phase 6: Comment and Docs Pass

- Remove redundant source comments.
- Keep concise representation notes near dispatch/loss boundaries.
- Update docs only where they describe current CLI/artifacts/objective.

### Phase 7: Final Merge Review

- Check `git diff --check` and staged status.
- Review final file list.
- Do not run project code.
- Prepare a merge summary for the eventual commit.
