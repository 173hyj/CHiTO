# Reproducibility Notes

## Environment Assumptions

The current workspace targets a ROS 2 + MoveIt environment with Eigen, FCL, Drake, yaml-cpp, and Gurobi available. `CMakeLists.txt` keeps Drake and Gurobi paths explicit because the original experiments were run in a fixed laboratory workstation environment.

Before building, verify:

- `DRAKE_PREFIX` points to the installed Drake prefix.
- `GUROBI_HOME` points to a valid Gurobi installation.
- The UR5 + end-effector description is available through the provided xacro and MoveIt configuration packages.

## Suggested Build

From `chito_public`:

```bash
colcon build --symlink-install --packages-select CHiTO_Planner
source install/setup.bash
```

## Suggested Pipeline

1. Launch the UR5 MoveIt configuration and RViz demo.
2. Generate or load a convex corridor scene from `collision/convex_scene.yaml`.
3. Run an initialization node to produce IK-feasible dense seeds.
4. Run `hierarchical_opt` with `use_convexset_guidance:=true`.
5. Enable `enable_final_global_smooth:=true` for the final refinement used in trajectory-quality comparisons.

## Key Metrics

The optimizer supports batch-mode CSV output for:

- planning success
- runtime
- number of optimized waypoints
- normalized path length
- minimum clearance

These metrics support success-rate, efficiency, and trajectory-quality comparisons.

## Open-source solver and optional corridor generation

The OSQP backend requires native OSQP 1.x (`osqp::osqp`). It preserves
`0.5*x.transpose()*Q*x + c.transpose()*x` and box bounds; only solved
status with checked feasible bounds is accepted. No commercial solver license is needed.

Use `-DCHITO_BUILD_CORRIDOR_TOOLS=OFF` to build the optimizer against supplied
seed/corridor data without Drake. The default `ON` retains all corridor tools.
Set `-DDRAKE_PREFIX=/opt/drake` (or `drake_DIR`) for a native Drake installation.
Ubuntu 22.04 was validated with the official Drake 1.51.1 Jammy binary.
CMake's imported Drake target supplies the required C++ standard.

Validated native generator smoke: `scene_spacerrt_dh_ik_stats` with scene1,
one trial, and `exit_after_batch=true`, producing seed CSV, corridor text,
and exact generation-time guide YAML without a commercial license.
