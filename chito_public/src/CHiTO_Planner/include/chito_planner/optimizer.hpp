#pragma once
#include "chito_planner/core/robot_geometry.hpp"
#include <rclcpp/rclcpp.hpp>
#include <string>
#include <cstdint>

namespace chito_planner {
using JointVector = core::JointVector6;
using JointPath = std::vector<JointVector>;
struct OptimizerRequest {
  // Default CHiTO method: actual-merit global QP, feasibility-first local
  // repair, corridor-constrained incoming/outgoing copies and validated stitching.
  // Set false only to reproduce legacy research variants.
  bool formal_method{true};
  std::string scene_path;
  JointVector q_start{JointVector::Zero()}, q_goal{JointVector::Zero()}; // radians
  Eigen::Vector3d domain_lb{-1.,-1.,0.}, domain_ub{1.,1.,1.};
  Eigen::Vector3d base_xyz{Eigen::Vector3d::Zero()};
  double base_yaw_deg{180.};
  std::uint32_t random_seed{20261115};
  int corridor_max_iterations{200};
  double sampling_scale{0.7};
  int ik_max_iterations{200};
  double ik_position_tolerance{1e-4};
  int optimization_max_iterations{20};
  int continuous_repair_passes{20};
  int repair_waypoint_limit{256};
  int repair_interior_points{1};
  bool repair_revisit_existing{true};
  double safety_distance{0.05};
  double minimum_clearance{0.0};
  double continuous_clearance{0.005};
  double continuous_joint_step_deg{1.0};
  bool continuous_convexification{true};
  bool translation_convexification{false};
  bool accept_approximate_hull{false};
  bool continuous_collision_refinement{false};
  double obstacle_inflation{0.0};
  bool validate_original_scene{true};
  bool final_smoothing{false};
  bool export_intermediates{false};
  std::string output_directory;
};
// Shared by YAML loading, full solves and supplied-seed refinement.
inline bool formal_options_valid(const OptimizerRequest& r) {
  return !r.formal_method || (r.continuous_convexification &&
    !r.translation_convexification && !r.continuous_collision_refinement &&
    !r.accept_approximate_hull);
}
struct GenerationResult {
  bool success{false};
  std::string status, message;
  JointPath seed;
  core::ConvexSetGuideData guide;
  std::vector<int> ik_fallback;
  double generation_ms{0.};
};
struct RefinementResult {
  bool success{false}, waypoint_safe{false}, continuous_safe{false};
  bool smoothing_rolled_back{false};
  int continuous_repairs{0};
  std::string status, message;
  JointPath trajectory;
  double optimization_ms{0.}, validation_ms{0.};
  double minimum_distance{0.}, normalized_length{0.};
};
struct OptimizerResult {
  bool success{false}, export_success{true};
  std::string status, message, export_message;
  GenerationResult generation;
  RefinementResult refinement;
  double total_ms{0.};
};
class CHiTOOptimizer {
 public:
  OptimizerResult solve(const OptimizerRequest& request) const;
};
namespace detail {
GenerationResult generate_corridor(const OptimizerRequest&, const rclcpp::NodeOptions&);
RefinementResult refine_trajectory(const OptimizerRequest&, const GenerationResult&, const rclcpp::NodeOptions&);
RefinementResult validate_original_trajectory(const OptimizerRequest&, const GenerationResult&, const RefinementResult&, const rclcpp::NodeOptions&);
}
}  // namespace chito_planner
