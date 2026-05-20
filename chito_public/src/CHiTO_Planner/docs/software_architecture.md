# Software Architecture

`CHiTO_Planner` is organized around a small reusable core and ROS-facing experiment nodes.

## Reusable Core

- `include/chito_planner/core/robot_geometry.hpp`
  Defines obstacle, link-volume, and convex-set guidance data used by the optimizer.
- `src/core/robot_geometry.cpp`
  Constructs FCL collision geometry for manipulator links and end-effectors.
- `include/chito_planner/core/convex_hull.hpp`
  Defines the watertight hull interface used by swept-volume safety checking.
- `src/core/convex_hull.cpp`
  Implements plane clustering and polygonal hull reconstruction for small swept-volume point sets.
- `include/chito_planner/optimization/qp_solver.hpp`
  Defines the dense bounded QP interface.
- `src/optimization/qp_solver.cpp`
  Implements the Gurobi backend for local convexified subproblems.
- `src/nodes/hierarchical_opt_*.ipp`
  Splits the main hierarchical optimizer into obstacle loading, geometry, global optimization, continuity repair, batch evaluation, visualization, and state-management sections.

## ROS Nodes

- Initialization and visualization nodes remain as executable ROS nodes because they depend on parameters, MoveIt state, RViz markers, and experiment-specific logging.
- `hierarchical_opt.cpp` acts as the ROS node shell. Its algorithmic sections are included from `src/nodes/` so the implementation can be reviewed according to the CHiTO pipeline stages.

## Design Rationale

The structure separates algorithmic primitives from experiment orchestration:

- core geometry can be inspected independently from ROS visualization
- QP solving is isolated from trajectory-cost assembly
- convex-set guidance data has a named type shared by loading, optimization, and documentation
- the main optimizer remains a readable pipeline rather than a monolithic collection of unrelated helper functions
