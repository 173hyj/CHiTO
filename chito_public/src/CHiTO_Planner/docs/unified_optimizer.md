# Unified optimizer

## Translation-only experimental SCP

Set `translation_convexification: true` to jointly optimize adjacent endpoint variables with linearized common separating-plane constraints. Each segment retains BOTH endpoint boxes with their own current orientations and dimensions. Their convex envelope defines the segment geometry; only the two center translations are linearized. Endpoint orientations and separate support radii are frozen within each QP and rebuilt after updates. No orientation derivatives or rotational sweep compensation are included. The global QP includes both endpoint rows, fixed start/goal, trust/joint bounds and nonnegative slack penalties. Backtracking accepts decreasing actual translation-model violation. Final validation uses the same translation support-plane model and writes `validation_model: endpoint_boxes_translation_linearization`; it does not use the old endpoint-hull validator for acceptance. This mode retains the original global/local-repair flow as warm start and disables optional final smoothing. The library flag defaults OFF; bundled scene presets enable the complete pipeline. It and is not a rotational-motion safety guarantee. Diagnostic messages record initial/final support gaps and solved QP counts. Negative support gaps may be conservative separation failures rather than physical penetration depths. See the experiment report for fixed settings and limitations.

The default build creates one C++ library (`chito_optimizer_library`) and one CLI (`chito_optimizer`). Corridor search, IK and refinement communicate in memory, without subprocesses or required intermediate files. Supported geometry is the supplied `.scene` box/sphere format. YAML files configure the optimizer; YAML obstacle geometry is not accepted by this integrated generator.

Build in an initialized ROS Humble environment with native Drake and OSQP installed:

```bash
colcon build --packages-select CHiTO_Planner --cmake-args -DDRAKE_PREFIX=/opt/drake -DBUILD_TESTING=ON
source install/setup.bash
ros2 run CHiTO_Planner chito_optimizer --config "$(ros2 pkg prefix CHiTO_Planner)/share/CHiTO_Planner/config/chito_scene1.yaml" --output results/scene1
```

`DRAKE_PREFIX` is configurable; `/opt/drake` is the tested installation location. Existing standalone corridor tools can be built with `-DCHITO_BUILD_CORRIDOR_TOOLS=ON`.

Library usage (joint vectors are radians):

```cpp
#include <chito_planner/config.hpp>
auto request = chito_planner::read_optimizer_config("scene1.yaml");
request.output_directory = "results/scene1"; // empty means no filesystem exports
auto result = chito_planner::CHiTOOptimizer{}.solve(request);
if (result.success) { /* consume result.refinement.trajectory */ }
```

Downstream CMake uses the exported namespaced target:

```cmake
find_package(CHiTO_Planner REQUIRED)
target_link_libraries(your_target PRIVATE CHiTO_Planner::chito_optimizer_library)
```

The library creates its own ROS context and does not initialize or shut down the caller's global context. All endpoint positions must lie within the configured workspace. Final success requires valid waypoints, endpoints and every swept-link edge check. This stricter criterion differs from the old batch success column, which checked waypoint minimum distance only.

Each solve exports `result.json`, `trajectory.csv` when a candidate exists, `effective_config.yaml`, and self-contained `playback.html`. Set `export_intermediates: true` to additionally save diagnostic `seed.csv` and `guide.yaml`; these files are never used to connect stages. Diagnostic guide joints are radians (`joint_unit: rad`), unlike the legacy generator's degree exports.

Final smoothing defaults OFF. If requested, it runs only after a valid pre-smooth solution exists; an invalid smoothed candidate is rolled back and recorded in `smoothing_rolled_back`. The validation model remains an approximation, not a physical execution guarantee. Export failures are returned separately via `export_success`/`export_message`; planning success does not imply successful artifact creation. Existing result directories are rejected rather than silently reusing stale trajectories.

After initial refinement, point-safe paths with unsafe connecting edges receive up to `continuous_repair_passes` additional repair passes (default20). `repair_waypoint_limit` (default256) bounds expansion during this additional repair stage; it is not a limit on the original input or initial refinement. Set the pass count to0 for comparison against the staged refinement. Final node/edge validation is applied in either case. These stricter checks and optional bounded repairs change acceptance relative to the legacy batch metric; report them explicitly in experiments.
Run `ros2 run CHiTO_Planner chito_optimizer --config CONFIG --output DIR`.
Optional `--trials N` runs N independent solves in one process with seeds incremented from the configured seed. `--seed N` overrides that seed; `--smooth` enables final smoothing. Results are written into `DIR/run_0`, `DIR/run_1`, etc. Use a fresh output directory for each invocation.

Each YAML requires `scene` and exactly one explicit unit key for each endpoint: `q_start_rad` or `q_start_deg`, and `q_goal_rad` or `q_goal_deg`. Endpoints require six finite values. Relative scene paths resolve against the config's directory. Installed configs expect bundled scenes in `config/scenes`.

Optional keys match OptimizerRequest: domain_lb, domain_ub, base_xyz, base_yaw_deg, random_seed, corridor_max_iterations, sampling_scale, ik_max_iterations, ik_position_tolerance, optimization_max_iterations, safety_distance, minimum_clearance, continuous_clearance, final_smoothing, export_intermediates. Iteration counts and tolerances must be positive; distances nonnegative; domain bounds ordered.

Exit codes: 0 all solves/export succeeded; 1 planning failure; 2 invalid arguments/config; 3 export failure. A final stdout JSON object reports trials, successes, planning_failures, and export_failures. Solver diagnostics may precede this line.
