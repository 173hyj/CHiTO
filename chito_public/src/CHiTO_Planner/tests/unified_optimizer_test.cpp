#include "chito_planner/optimizer.hpp"
#include "chito_planner/config.hpp"
#include <yaml-cpp/yaml.h>
#include <cassert>
#include <limits>
#include <filesystem>
#include <fstream>
#include <regex>
int main() {
  chito_planner::CHiTOOptimizer optimizer;
  chito_planner::OptimizerRequest request;
  auto result = optimizer.solve(request);
  assert(!result.success && result.status == "invalid_input");
  request.scene_path = "/nonexistent/chito.scene";
  result = optimizer.solve(request);
  assert(!result.success && result.status == "invalid_input");
  request.q_start[0] = std::numeric_limits<double>::quiet_NaN();
  result = optimizer.solve(request);
  assert(!result.success && result.status == "invalid_input");
  request = chito_planner::OptimizerRequest{};
  request.scene_path=CHITO_TEST_SCENE;
  const double pi=std::acos(-1.);
  request.q_start << 39,-67,63,-179,-97,-215;
  request.q_goal << 67,-98,117,-134,-77,-212;
  request.q_start*=pi/180.;request.q_goal*=pi/180.;
  const bool global_context_before=rclcpp::ok();
  auto first=optimizer.solve(request), second=optimizer.solve(request);
  assert(first.generation.success && second.generation.success);
  assert(!first.refinement.trajectory.empty());
  assert(first.status==second.status && first.refinement.trajectory.size()==second.refinement.trajectory.size());
  for(size_t i=0;i<first.refinement.trajectory.size();++i)
    assert((first.refinement.trajectory[i]-second.refinement.trajectory[i]).norm()<1e-9);
  assert(rclcpp::ok()==global_context_before);
  request.translation_convexification=true;
  auto protected_path=optimizer.solve(request);
  std::regex point_margins("baseline_point_min=([-0-9.]+), final_point_min=([-0-9.]+)");
  std::smatch point_match;
  assert(std::regex_search(protected_path.refinement.message,point_match,point_margins));
  const double before=std::stod(point_match[1]),after=std::stod(point_match[2]);
  assert(before<0. || after>=0.);
  request.translation_convexification=false;
  const auto dir=std::filesystem::temp_directory_path()/std::filesystem::path("chito-contract-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  request.output_directory=dir.string();request.export_intermediates=true;
  auto exported=optimizer.solve(request);assert(exported.export_success);
  auto guide=YAML::LoadFile((dir/"guide.yaml").string());
  assert(guide["q_seed_dense"].size()==exported.generation.seed.size());
  for(const auto& row:guide["q_seed_dense"])assert(row.IsSequence() && row.size()==6);
  auto restored=chito_planner::read_optimizer_config((dir/"effective_config.yaml").string());
  assert((restored.q_start-request.q_start).norm()<1e-14);
  assert(std::filesystem::file_size(dir/"playback.html")>1000);
  std::filesystem::remove_all(dir);
  request.output_directory.clear();
  request.domain_lb=Eigen::Vector3d(5,5,5);request.domain_ub=Eigen::Vector3d(6,6,6);
  result=optimizer.solve(request);assert(!result.success && result.status=="invalid_input");
}
