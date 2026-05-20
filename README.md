# CHiTO

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
