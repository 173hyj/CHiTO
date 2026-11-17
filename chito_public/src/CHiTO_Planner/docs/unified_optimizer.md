# Unified optimizer

The supported formal entry points are `chito_optimizer` and `CHiTOOptimizer::solve`. The default build creates `chito_optimizer_library` and `chito_optimizer`; generation, IK, refinement and validation pass data in memory.

## Build and call

See the root [README](../../../../README.md) for dependencies and build commands. All joint vectors in the C++ API are radians.

```cpp
#include <chito_planner/config.hpp>
auto request = chito_planner::read_optimizer_config("scene1.yaml");
request.output_directory = "results/scene1"; // empty disables exports
// request.formal_method is true by default.
auto result = chito_planner::CHiTOOptimizer{}.solve(request);
if (result.success) { /* consume result.refinement.trajectory */ }
```

```cmake
find_package(CHiTO_Planner REQUIRED)
target_link_libraries(your_target PRIVATE CHiTO_Planner::chito_optimizer_library)
```

The library owns an isolated ROS context; it does not initialize or shut down the caller's global context. Historical raw nodes keep their separate legacy defaults. Legacy translation-envelope research requires `formal_method: false` and is not enabled by installed presets.

```bash
ros2 run CHiTO_Planner chito_optimizer --scene scene2 --output results/demo
ros2 run CHiTO_Planner chito_optimizer --config custom.yaml --output results/batch --trials 5 --seed 20262115
```

`--smooth` enables optional final smoothing, which rolls back when it violates acceptance. Default smoothing is off. Use a new output directory for each invocation.

## Configuration

Each YAML requires `scene`, exactly one of `q_start_rad` / `q_start_deg`, and exactly one of `q_goal_rad` / `q_goal_deg`. Each endpoint is six finite numbers. Relative scene paths resolve against the YAML directory. The `.scene` generator supports boxes/spheres. Bounds, limits, units and tolerances are validated before planning.

[Formal method defaults](formal_method.md) describe the supported settings. The three source presets reference `../`-relative bundled scenes; installed copies reference `scenes/` automatically. `formal_method` is exported in effective configurations for explicit provenance.

The formal API fixes the collision model to six arm links. It does not accept torch/gripper options. The original corridor generation can succeed with colliding links because corridor membership is a TCP condition, not whole-arm feasibility. Optimization and final validation determine whole-arm success.

## Outputs and acceptance

Each trial writes `result.json`, `effective_config.yaml`, `playback.html`, and `trajectory.csv` when a candidate exists. `export_intermediates: true` also saves the original `seed.csv` and `guide.yaml` (radians, original cell assignments). Early failures may have no seed/trajectory. Playback is a link schematic, not the collision meshes.

`result.json` reports `success`, `status`, stage status, timing, waypoint and continuous safety, distances, path length and original-scene validation. `minimum_distance` is a waypoint metric; edge safety is separately recorded in `continuous_safe`. The JSON validation model `endpoint_swept_hulls` describes approximate hull checks. Final strict success additionally requires every original split copy to satisfy its own corridor. A failure status `corridor_constraint_failed` survives original-scene revalidation even when `waypoint_safe` and `continuous_safe` are true.

Other failures include `invalid_input`, generation-stage failures, `optimization_failed`, and `validation_failed`. Failed candidates are retained for diagnosis, not presented as usable plans. Export failures use separate `export_success` / `export_message`; planning success does not imply export success. Existing trial directories with a `result.json` are rejected for export.

The CLI writes `summary.json` and a final stdout JSON summary. Exit codes: 0 all trials and exports succeeded; 1 planning failure; 2 invalid arguments/config; 3 export failure. Solver log lines may precede the summary.

## Testing

```bash
colcon test --packages-select CHiTO_Planner --event-handlers console_direct+
colcon test-result --verbose
```

Regression tests cover analytical QPs, long-edge collisions, two-sided midpoint checking, objective consistency, revisiting failed nodes, collision-free repair windows, corridor halfspaces/FK membership, copy provenance/reset, strict final validation, public default splitting, configuration and API exports. `BUILD_TESTING=OFF` excludes diagnostic runners. See [examples/](../../../../examples/README.md) for one complete runnable example per scene.
