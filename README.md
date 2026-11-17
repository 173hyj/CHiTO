# CHiTO

CHiTO is a ROS 2 / MoveIt implementation of convex-set-guided hierarchical trajectory optimization for a six-link UR5 arm.

The default method is **corridor → IK → actual-merit global QP → independent left/right node copies → corridor-constrained local QP → connection repair → final validation**. Start and goal stay fixed. Every internal shared node has an incoming and an outgoing copy; each original copy stays in its assigned convex corridor. Newly inserted repair points use collision constraints without a mandatory TCP corridor constraint.

## Build and run

Requires Ubuntu 22.04, ROS 2 Humble with MoveIt, Eigen, FCL, yaml-cpp, Drake and OSQP 1.x. The tested environment uses Drake 1.51.1 and OSQP 1.0.0. Set `DRAKE_PREFIX` to your native Drake installation.

```bash
source /opt/ros/humble/setup.bash
cd chito_public
colcon build --packages-select CHiTO_Planner --cmake-args \
  -DCMAKE_BUILD_TYPE=Release -DDRAKE_PREFIX=/opt/drake -DBUILD_TESTING=ON
colcon test --packages-select CHiTO_Planner --event-handlers console_direct+
colcon test-result --verbose
source install/setup.bash
ros2 run CHiTO_Planner chito_optimizer --scene scene1 --output results/scene1
```

Use `--scene scene2` or `--scene scene3` for the other presets, `--trials 5` for incrementing seeds, or `--config FILE` for a custom request. Choose a fresh output directory. The source YAML files also work directly with `--config`; scene paths resolve relative to the YAML.

The formal preset uses the arm's six link collision boxes, **without a torch or gripper**. Corridor membership constrains TCP positions of original node copies; whole-arm obstacle checking remains separate. The default edge clearance is 5 mm, the point penalty activates at 50 mm, and final smoothing is off. Those are different quantities.

## Method and examples

- [Formal method, constraints and code mapping](chito_public/src/CHiTO_Planner/docs/formal_method.md)
- [C++ API, configuration and output contract](chito_public/src/CHiTO_Planner/docs/unified_optimizer.md)
- [中文流程与运行说明](chito_public/src/CHiTO_Planner/docs/complete_pipeline.md)
- [One complete example for each of the three scenes](examples/README.md)

Final success requires fixed endpoints, point and swept-edge clearance checks, and corridor membership for every original internal copy. A geometrically safe path outside an assigned corridor remains a failure. Finite angular subdivision and endpoint hulls are approximate checks, not exact continuous collision detection.

## Public entry points

`chito_optimizer` and `CHiTOOptimizer::solve` are the supported formal entry points. They pass corridor, IK and optimization data in memory and export JSON, CSV, effective YAML and standalone HTML playback. `formal_method: true` is the default; no experiment environment variables are needed.

Historical `hierarchical_opt`, standalone corridor/torch nodes, launch files and translation-envelope research modes remain for compatibility. They are not the formal preset. Set `formal_method: false` explicitly to use legacy research settings; this opt-out alone does not reconstruct every old experiment. The raw ROS node retains its separate legacy defaults.
