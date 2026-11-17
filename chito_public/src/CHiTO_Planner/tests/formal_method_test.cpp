#include "chito_planner/config.hpp"
#include <cassert>
#include <iostream>
int main() {
  auto r=chito_planner::read_optimizer_config(CHITO_FORMAL_CONFIG);
  r.q_goal=r.q_start; r.translation_convexification=false; r.continuous_convexification=true;
  r.optimization_max_iterations=1;
  r.continuous_repair_passes=1;
  rclcpp::init(0,nullptr);
  chito_planner::GenerationResult g;g.success=true;
  g.seed.assign(5,r.q_start);g.guide.loaded=true;g.guide.q_seed_dense=g.seed;
  g.guide.sigma_dense.assign(5,0);
  Eigen::MatrixXd A(6,3);A<<1,0,0,-1,0,0,0,1,0,0,-1,0,0,0,1,0,0,-1;
  g.guide.poly_A={A};g.guide.poly_b={Eigen::VectorXd::Constant(6,10.)};
  g.guide.anchor_xyz={Eigen::Vector3d::Zero()};
  auto incompatible=r;incompatible.continuous_convexification=false;
  auto bad=chito_planner::detail::refine_trajectory(incompatible,g,rclcpp::NodeOptions{});
  if(bad.status!="invalid_input"){std::cerr<<"Supplied-seed formal entry must reject incompatible options, got "<<bad.status<<'\n';return 21;}
  incompatible=r;incompatible.translation_convexification=true;
  bad=chito_planner::detail::refine_trajectory(incompatible,g,rclcpp::NodeOptions{});
  assert(bad.status=="invalid_input");
  auto out=chito_planner::detail::refine_trajectory(r,g,rclcpp::NodeOptions{});
  if(out.trajectory.size()!=8){std::cerr<<"Default public refinement must split 5 nodes into 8; got "<<out.trajectory.size()<<'\n';return 10;}
  assert((out.trajectory.front()-r.q_start).norm()<1e-12);
  assert((out.trajectory.back()-r.q_goal).norm()<1e-12);
  auto checked=chito_planner::detail::validate_original_trajectory(r,g,out,rclcpp::NodeOptions{});
  assert(checked.waypoint_safe && checked.continuous_safe && checked.success);
  out.status="corridor_constraint_failed";out.success=false;
  checked=chito_planner::detail::validate_original_trajectory(r,g,out,rclcpp::NodeOptions{});
  assert(!checked.success && checked.status=="corridor_constraint_failed");
  // Infeasible source cells must fail even for a collision-free static path.
  g.guide.poly_b={Eigen::VectorXd::Constant(6,-10.)};
  auto impossible=chito_planner::detail::refine_trajectory(r,g,rclcpp::NodeOptions{});
  assert(!impossible.success && impossible.status=="corridor_constraint_failed");
  rclcpp::shutdown();
  std::cout<<"Public formal method: split layout, endpoints and strict corridor gate passed\n";
}
