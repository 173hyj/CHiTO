# CHiTO

## Unified optimizer (v0.1.0)

The unified C++ entry runs corridor search, IK, OSQP refinement and final safety validation in one process. Stages pass data in memory; no intermediate files or separate node invocations are required.

```bash
cd chito_public
colcon build --packages-select CHiTO_Planner --cmake-args -DDRAKE_PREFIX=/opt/drake -DBUILD_TESTING=ON
source install/setup.bash
ros2 run CHiTO_Planner chito_optimizer --config "$(ros2 pkg prefix CHiTO_Planner)/share/CHiTO_Planner/config/chito_scene2.yaml" --output results/scene2
```

Native Drake, ROS 2 Humble/MoveIt, FCL, Eigen, yaml-cpp and OSQP 1.x are required. `DRAKE_PREFIX` is configurable. The tested environment uses Drake1.51.1 and OSQP1.0.0 on Ubuntu22.04.

Use `--trials 50` for a seeded in-process batch. Exported results include a trajectory CSV, JSON diagnostics, effective configuration and standalone HTML playback. Final success requires waypoint and swept-edge checks. Optional smoothing rolls back if it breaks feasibility. The stricter validation and bounded extra edge repairs change acceptance relative to legacy batch metrics.

See [unified optimizer documentation](chito_public/src/CHiTO_Planner/docs/unified_optimizer.md) for the C++ `CHiTOOptimizer::solve` API, configuration, failure handling and outputs.

**CHiTO** is a ROS 2 / MoveIt workspace for **Convex-Set-Guided Hierarchical Trajectory Optimization** of manipulators in cluttered and narrow workspaces.

The workspace provides the complete experimental software for the CHiTO framework, including workspace convex-set guidance, IK-feasible initialization, locally convexified quadratic optimization, continuous-time safety checking, and final trajectory refinement for UR5-based experiments.

## Method Components

- **Convex-set scaffold**: expanded free-space convex sets are represented as half-space polytopes and organized as a feasible corridor sequence.
- **IK-feasible initialization**: Cartesian anchors inside the corridor are converted into continuous joint-space waypoints through MoveIt IK and dynamic-programming branch selection.
- **Hierarchical QP refinement**: collision and corridor terms are locally linearized, then solved as bounded quadratic subproblems with trust-region control.
- **Continuous-time safety**: adjacent manipulator states are checked with swept link volumes to reduce interpolation-time collision risk.
- **Global smoothing**: optional post-refinement improves trajectory continuity while retaining corridor and clearance guidance.

## Workspace Layout

```text
chito_public/
  src/
    CHiTO_Planner/                 Main CHiTO algorithm package
      include/chito_planner/       Reusable core and optimization interfaces
      src/core/                    Geometry and swept-volume utilities
      src/optimization/            Dense QP backend
      src/nodes/                   Hierarchical optimizer implementation slices
      src/*.cpp                    ROS nodes for initialization, optimization, and visualization
      config/                      Optimization and kinematics parameters
      launch/                      Experiment launch files
      docs/                        Method-to-code and reproducibility notes
    collision/                     Example obstacle and convex-scene descriptions
    hyj_ur5_robotiq_description/   UR5 + end-effector description
    planner/                       Benchmark and demo launch/configuration tools
    ur5_robotiq_moveit_config/     MoveIt configuration package
```

## Primary Entry Points

- `corridor_viz_moveit_initik`: generates IK-feasible seed paths from convex-set corridors.
- `hierarchical_opt`: runs the CHiTO hierarchical trajectory optimization and continuous safety refinement.
- `scene_spacerrt_dh_ik_stats`: evaluates corridor-guided IK statistics for benchmark scenes.

See [CHiTO_Planner/docs/method_mapping.md](chito_public/src/CHiTO_Planner/docs/method_mapping.md) and [CHiTO_Planner/docs/reproducibility.md](chito_public/src/CHiTO_Planner/docs/reproducibility.md) for the method mapping and suggested experiment workflow.
