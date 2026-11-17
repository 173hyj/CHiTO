# Formal CHiTO method

The supported method is selected by `OptimizerRequest::formal_method = true` (the default). The three installed scene presets select it explicitly. `chito_optimizer` runs this method without environment switches.

## Stages

1. **Corridor and IK.** Search workspace free-space convex sets, choose corridor anchors and generate a joint-space seed with its original cell IDs. A TCP in a free corridor does not imply all robot links are collision-free.
2. **Global QP.** Refine the full seed with fixed true endpoints, joint limits and trust regions. Actual geometric merit must justify acceptance; the former forced acceptance fallback is disabled. Corridor guidance is soft at this global stage. No new seed pull is introduced.
3. **Independent copies.** Freeze the global path for this stage. Replace every internal shared node by incoming/outgoing copies. Each original edge optimizes its own pair independently; true start/goal stay single and fixed. For example `[q0,q1,q2,q3,q4]` becomes `[q0,q1_in,q1_out,q2_in,q2_out,q3_in,q3_out,q4]`.
4. **Local QP.** Use collision constraints and feasibility-first acceptance. Linearize normalized halfspaces of each endpoint's original cell around its TCP FK/Jacobian. Recheck exact FK membership after a candidate update, with a 1e-6 m tolerance. Every original copy in a repair window keeps its own cell constraint. Pair smoothness penalizes adjacent joint differences; the wider repair window uses the inherited bending term.
5. **Assemble and stitch.** Keep both side copies without averaging or deduplicating. Check each new connection. A collision-free inserted midpoint is insufficient: its connections to both ends must also pass. Check changed nodes, internal edges and the two external halo edges before accepting a window update. Expand windows when a colliding halo node would otherwise be fixed; a colliding true start/goal fails closed. New repair points have no mandatory TCP corridor constraint.
6. **Final validation.** Recheck fixed endpoints, finite joints, all arm waypoints, subdivided swept-link edges, and all original internal copy TCP memberships. Original-scene revalidation preserves `corridor_constraint_failed`; it cannot turn a corridor failure into success. State/provenance are reset for each solve.

Splitting is unconditional for internal shared nodes. Selective splitting only on difficult edges has not been implemented. The implementation does not constrain the entire arm inside a corridor, or force new repair points' TCPs into corridor cells.

## Default settings

| Setting | Default | Meaning |
|---|---:|---|
| `formal_method` | true | Enables actual-merit global acceptance, feasibility-first local repair, moving corridor endpoints and split nodes together |
| `safety_distance` | 0.05 m | Point collision penalty activation distance |
| `minimum_clearance` | 0 m | Final point clearance threshold |
| `continuous_clearance` | 0.005 m | Swept-edge clearance threshold |
| `continuous_joint_step_deg` | 1 degree | Maximum per-joint subdivision step before hull checks |
| `repair_interior_points` | 1 | Newly inserted points per attempted repair edge |
| `repair_revisit_existing` | true | Existing failed repair nodes can participate again |
| `continuous_repair_passes` | 20 | Additional bounded repair passes after initial refinement |
| `repair_waypoint_limit` | 256 | Split/repair growth limit; it does not truncate an oversized supplied seed |
| `optimization_max_iterations` | 20 | Global iteration limit |
| `validate_original_scene` | true | Revalidate against original obstacle geometry |
| `final_smoothing` | false | Optional smoothing must preserve validity or be rolled back |

The formal method disables torch/gripper geometry in the refinement node, matching the six-link integrated generator and playback. Tool-specific planning remains a separate legacy capability and is not evaluated by the included benchmark.

Formal mode requires `continuous_convexification: true`. Incompatible `translation_convexification`, `accept_approximate_hull`, and `continuous_collision_refinement` research modes are rejected instead of silently bypassing the split method. They remain available with explicit `formal_method: false`. Optional obstacle inflation changes the planning scene; final validation uses the original scene, and copy membership still refers to the planning corridor. Inflation was not used in the paired evaluation.

## Code mapping

All paths below are relative to `chito_public/src/CHiTO_Planner`.

| Component | Code |
|---|---|
| Public defaults/config/export | `include/chito_planner/optimizer.hpp`, `src/optimizer_config.cpp`, `src/unified_optimizer.cpp` |
| Global merit acceptance | `src/hierarchical_opt.cpp`, `src/nodes/hierarchical_opt_global_optimization.ipp`, `src/nodes/hierarchical_opt_batch_io.ipp` |
| Split layout and pair QPs | `include/chito_planner/core/split_corridor.hpp`, `src/nodes/hierarchical_opt_split_corridor.ipp` |
| Cell constraints and window repair | `core/corridor_endpoint.hpp` under include/chito_planner, `src/nodes/hierarchical_opt_corridor_endpoints.ipp` |
| Collision-free halo windows | `include/chito_planner/core/repair_window.hpp` |
| Feasibility acceptance | `include/chito_planner/core/local_acceptance.hpp` |
| Local edge refinement | `src/nodes/hierarchical_opt_continuity.ipp` |
| Final corridor gate | `QPTrajOptNode::refine` and `validate_raw` in `src/hierarchical_opt.cpp` |
| Public regression | `tests/formal_method_test.cpp` |

## Evidence and limitations

The [paired benchmark](../../../../benchmarks/split_corridor/README.md) includes 15 frozen inputs and 30 full outputs. Both methods obtain 13 strict successes; split geometry passes 14 versus 13. Shared-endpoint strict success constrains endpoints during each local call; split strict success additionally demands all original copies end inside their assigned cells. These are different corridor acceptance scopes.

The method can fail on collision, corridor infeasibility, solver failure or work budgets. A safe point alone never establishes safe interpolation. Subdivided endpoint hull checking reduces missed interpolation collisions but is not an exact representation of all intermediate rotating link geometry. The tests do not establish hardware execution safety.
