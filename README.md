# CHiTO
Corridor-guided hierarchical trajectory optimization framework for manipulators
# CHiTO

**CHiTO** (Corridor-guided Hierarchical Trajectory Optimization) is a ROS 2 workspace for manipulator motion planning in cluttered environments.  
It combines **workspace corridor generation**, **IK-feasible waypoint construction**, and **trajectory optimization / refinement** for collision-aware planning.

This repository contains the current public workspace used for corridor-guided planning experiments with a **UR5-based manipulator system**.

---

## Features

- Workspace corridor construction from obstacle scenes
- Corridor-guided waypoint / seed generation
- IK-feasible path generation for manipulator motion
- Hierarchical trajectory refinement
- Trajectory optimization with collision-aware cost terms
- Support for narrow-passage and constrained manipulation scenes
- Visualization tools for corridor structure, IK paths, and optimized trajectories

---

## Workspace Structure

```text
chito_public/
└── src
    ├── CHiTO_Planner                # Main planning / optimization package
    ├── collision                    # Example collision scenes
    ├── hyj_ur5_robotiq_description  # UR5 + Robotiq robot description
    ├── planner                      # Additional planning / benchmark tools
    └── ur5_robotiq_moveit_config    # MoveIt configuration package
