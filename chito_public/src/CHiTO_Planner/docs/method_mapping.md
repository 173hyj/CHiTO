# Method-to-Code Mapping

This note summarizes how the main modules of the CHiTO framework are organized in the software, making the corridor construction, IK initialization, optimization, and safety-checking stages easy to inspect.

## Convex-Set Corridor Guidance

The convex-set sequence is loaded as half-space polytopes `A x <= b` and dense seed information in `hierarchical_opt`.

Relevant code:

- `load_convexset_guide_from_yaml_` in `src/hierarchical_opt.cpp`
- `ConvexSetGuideData` in `include/chito_planner/core/robot_geometry.hpp`
- `build_convexset_guidance_c_` in `src/hierarchical_opt.cpp`
- `build_poly_membership_linear_term_` in `src/hierarchical_opt.cpp`

Paper correspondence:

- overlap-based convex-set sequence
- corridor membership penalties
- Cartesian directional priors

## IK-Feasible Initialization

The corridor initialization nodes sample anchor points inside polytopes, solve IK, and select a continuous joint branch using dynamic programming.

Relevant code:

- `Corridor_Viz_Moveit_Initik.cpp`
- `Corridor_Viz_Moveit_Initik_withtorch.cpp`
- `corridor_viz_moveit_ik.cpp`

Paper correspondence:

- workspace feasibility converted into joint-space executability
- branch continuity across adjacent corridor regions

## Hierarchical Trajectory Optimization

The optimizer assembles a smoothness quadratic backbone and adds locally linearized clearance and corridor terms. The dense bounded QP backend is isolated in a reusable optimization module.

Relevant code:

- `build_Q_c_oldstyle_from` in `src/hierarchical_opt.cpp`
- `one_iter_step` in `src/hierarchical_opt.cpp`
- `solve_box_qp` in `src/optimization/qp_solver.cpp`
- `include/chito_planner/optimization/qp_solver.hpp`

Paper correspondence:

- local quadratic subproblems
- trust-region acceptance
- collision-aware and corridor-guided refinement

## Continuous-Time Safety

Adjacent robot states are checked by constructing swept link volumes and testing them against obstacles with FCL.

Relevant code:

- `edge_continuous_safe_convexbox_` in `src/hierarchical_opt.cpp`
- `make_link_swept_hull_data_` in `src/hierarchical_opt.cpp`
- `build_watertight_convex_hull` in `src/core/convex_hull.cpp`
- `make_link_box` and `make_box_from_pose` in `src/core/robot_geometry.cpp`

Paper correspondence:

- interpolation safety between discrete waypoints
- conservative swept-volume collision checks

## Final Global Smoothing

The optional final smoothing stage re-solves the full trajectory with stronger smoothness and weaker clearance/corridor guidance.

Relevant code:

- `run_final_global_smooth_` in `src/hierarchical_opt.cpp`

Paper correspondence:

- trajectory continuity improvement after local hierarchical refinement
