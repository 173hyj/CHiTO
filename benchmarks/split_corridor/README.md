# Shared-node split evaluation

This is the frozen three-scene evaluation used when formalizing CHiTO. Each scene has five saved corridor/IK inputs, with identical input pairs for the shared-endpoint baseline and the split method. **This evaluates optimization replay, not end-to-end random generation success.** Geometry uses six arm links, no torch/gripper, and no obstacle inflation.

| Scene | Shared strict success | Split strict success | Shared geometry pass | Split geometry pass |
|---|---:|---:|---:|---:|
| scene1 | 3/5 | 3/5 | 3/5 | 4/5 |
| scene2 | 5/5 | 5/5 | 5/5 | 5/5 |
| scene3 | 5/5 | 5/5 | 5/5 | 5/5 |
| Total | 13/15 | 13/15 | 13/15 | 14/15 |

Strict split success requires the point/edge checks and final membership of every original copy. Shared baseline only constrains the current local QP endpoints and does not impose that same all-copy final gate. The corridor acceptance scopes differ; geometry-only pass counts are shown separately.

For scene1, three strict successes increased the independently sampled minimum clearance from 5.718→7.270, 6.706→10.102, and 6.814→10.582 mm. Their joint paths became longer. Trial 2 remained colliding; trial 4 became geometrically safe but two copies were approximately 141.94 mm outside their cell, so it remained a strict failure. Trial indices start at zero.

For scene2 and scene3, removing consecutive duplicate configurations gives the exact same ten joint paths within 1e-12 rad. Splitting only increased the node count from 5 to 8. Average recorded runtime rose from 0.274→0.911 s and 0.213→0.701 s respectively; these diagnostic timings describe this experiment, not an isolated performance benchmark.

## Files and provenance

- `inputs/sceneN/run_i`: saved seed, guide and portable effective configuration.
- `reference/sceneN/{shared,split}/run_i`: historical full-run trajectory, result and split provenance where present.
- `runs.csv`: all 30 full-run metrics, including independent ≤1 degree sampled minimum distances and strict/geometry outcomes.
- `summary.json`, `verification.json`, `behavior-audit.json`: historical aggregate and independent audits.
- `provenance.json`: experimental source/binary/archive hashes and hashes of the published frozen files. Source commit `4687c362d224fdf20c887e2df22d5ac3e5de05a8` refers to the archived experimental Git history, not a commit in this GitHub repository.
- `integration-verification.json`: fresh public-entry replay comparison after integration, when included.

Input scene paths were rewritten relative to their config files and `formal_method: true` was added. Reference CSVs and result JSON are unchanged. All paired global-only paths were identical in the original audit (15 pairs); source/binary/input hashes and endpoint checks passed. The original evidence archive SHA256 is `f255716d29ce6917469d0c6f6217a81cbb3ef1e784fa3980e4372e1f26230439`.

Distances sampled along joints and subdivided endpoint hulls are complementary approximate checks. Neither establishes an exact continuous collision proof. The preset still splits all internal shared nodes, including already safe ones.

## Reproduce on an installed dependency environment

From the repository root:

```bash
source /opt/ros/humble/setup.bash
cmake -S chito_public/src/CHiTO_Planner -B build \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DDRAKE_PREFIX=/opt/drake
cmake --build build -j2
ctest --test-dir build --output-on-failure
python3 benchmarks/split_corridor/replay.py --executable build/chito_replay --output results/frozen-replay
```

The replay uses the public refinement functions with an explicitly supplied `GenerationResult`; it does not run corridor/IK generation. Formal runs use the public default method. Shared baselines use the earlier actual-merit / feasibility-first / moving-endpoint flags, with splitting off and the same no-tool geometry. No private absolute paths or experiment environment switches are required. Use `--scene 1` to restrict replay. Output directories must be new.

The replay exits unsuccessfully if trajectories differ from reference by ≥1e-8 rad or strict success changes. This checks reproducibility of the implementation integration, not general performance on new tasks. For end-to-end generation use the main `chito_optimizer` CLI.
