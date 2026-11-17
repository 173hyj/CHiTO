#include "chito_planner/optimizer.hpp"
#include "chito_planner/playback.hpp"
#include <yaml-cpp/yaml.h>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <sstream>
#include <stdexcept>
namespace chito_planner {
namespace {
struct InflatedScene {
  std::filesystem::path temporary;
  ~InflatedScene(){if(!temporary.empty()){std::error_code e;std::filesystem::remove(temporary,e);}}
  std::string create(const std::string& source,double radius){
    std::ifstream in(source);if(!in)throw std::runtime_error("Cannot read original scene");
    std::vector<std::string> lines;std::string line;while(std::getline(in,line))lines.push_back(line);
    int changed=0;
    for(size_t i=0;i<lines.size();++i)if(!lines[i].empty()&&lines[i][0]=='*'){
      if(i+5>=lines.size())throw std::runtime_error("Malformed inflation input");
      std::istringstream kind(lines[i+4]),dims(lines[i+5]);std::string type;kind>>type;std::ostringstream out;out<<std::setprecision(17);
      if(type=="box"){double x,y,z;if(!(dims>>x>>y>>z)||x<=0||y<=0||z<=0)throw std::runtime_error("Invalid box");out<<x+2*radius<<' '<<y+2*radius<<' '<<z+2*radius;}
      else if(type=="sphere"){double r;if(!(dims>>r)||r<=0)throw std::runtime_error("Invalid sphere");out<<r+radius;}
      else throw std::runtime_error("Inflation supports scene box/sphere only");
      lines[i+5]=out.str();++changed;
    }
    if(!changed)throw std::runtime_error("No inflation primitives found");
    temporary=std::filesystem::temp_directory_path()/("chito-inflated-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".scene");
    std::ofstream out(temporary);out.exceptions(std::ios::failbit|std::ios::badbit);for(const auto& s:lines)out<<s<<'\n';out.close();return temporary.string();
  }
};
using Clock = std::chrono::steady_clock;
double milliseconds(Clock::time_point start) {
  return std::chrono::duration<double,std::milli>(Clock::now()-start).count();
}
std::string quote(const std::string& text) {
  std::ostringstream out; out << '"';
  for (unsigned char c : text) {
    if (c=='"' || c=='\\') out << '\\' << c;
    else if (c<32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c) << std::dec;
    else out << c;
  }
  return out.str()+'"';
}
void write_path(const std::filesystem::path& file, const JointPath& path) {
  std::ofstream out(file); out.exceptions(std::ios::failbit | std::ios::badbit);
  out << "idx,q1_rad,q2_rad,q3_rad,q4_rad,q5_rad,q6_rad\n" << std::setprecision(17);
  for (size_t i=0;i<path.size();++i) { out << i; for (int k=0;k<6;++k) out << ',' << path[i][k]; out << '\n'; }
  out.close();
}
template<class V> void yaml_vector(YAML::Node node,const V& v) {
  for (int i=0;i<v.size();++i) node.push_back(v[i]);
}
void export_result(const OptimizerRequest& request, const OptimizerResult& result) {
  const auto folder=std::filesystem::path(request.output_directory);
  std::filesystem::create_directories(folder);
  if (std::filesystem::exists(folder/"result.json"))
    throw std::runtime_error("Output already contains a CHiTO result; choose a fresh output directory");
  if (!result.refinement.trajectory.empty()) write_path(folder/"trajectory.csv",result.refinement.trajectory);
  YAML::Node cfg;
  cfg["formal_method"]=request.formal_method;
  cfg["scene"]=std::filesystem::absolute(request.scene_path).string();
  yaml_vector(cfg["q_start_rad"],request.q_start); yaml_vector(cfg["q_goal_rad"],request.q_goal);
  yaml_vector(cfg["domain_lb"],request.domain_lb); yaml_vector(cfg["domain_ub"],request.domain_ub);
  yaml_vector(cfg["base_xyz"],request.base_xyz); cfg["base_yaw_deg"]=request.base_yaw_deg;
  cfg["random_seed"]=request.random_seed; cfg["final_smoothing"]=request.final_smoothing;
  cfg["translation_convexification"]=request.translation_convexification;
  cfg["accept_approximate_hull"]=request.accept_approximate_hull;
  cfg["continuous_collision_refinement"]=request.continuous_collision_refinement;
  cfg["obstacle_inflation"]=request.obstacle_inflation;cfg["validate_original_scene"]=request.validate_original_scene;
  cfg["corridor_max_iterations"]=request.corridor_max_iterations; cfg["sampling_scale"]=request.sampling_scale;
  cfg["ik_max_iterations"]=request.ik_max_iterations; cfg["ik_position_tolerance"]=request.ik_position_tolerance;
  cfg["optimization_max_iterations"]=request.optimization_max_iterations;
  cfg["continuous_repair_passes"]=request.continuous_repair_passes;cfg["repair_waypoint_limit"]=request.repair_waypoint_limit;
  cfg["repair_interior_points"]=request.repair_interior_points;
  cfg["repair_revisit_existing"]=request.repair_revisit_existing;
  cfg["safety_distance"]=request.safety_distance; cfg["minimum_clearance"]=request.minimum_clearance;
  cfg["continuous_clearance"]=request.continuous_clearance;
  cfg["continuous_joint_step_deg"]=request.continuous_joint_step_deg;
  cfg["continuous_convexification"]=request.continuous_convexification;
  cfg["export_intermediates"]=request.export_intermediates;
  std::ofstream config(folder/"effective_config.yaml");config.exceptions(std::ios::failbit|std::ios::badbit);config << cfg;config.close();
  if (request.export_intermediates && !result.generation.seed.empty()) {
    write_path(folder/"seed.csv",result.generation.seed);
    YAML::Node guide;guide["joint_unit"]="rad";
    for (const auto& q:result.generation.guide.q_seed_dense) { YAML::Node n(YAML::NodeType::Sequence);yaml_vector(n,q);guide["q_seed_dense"].push_back(n); }
    guide["sigma_dense"]=result.generation.guide.sigma_dense;
    guide["ik_fallback_dense"]=result.generation.ik_fallback;
    for (size_t j=0;j<result.generation.guide.poly_A.size();++j) {
      YAML::Node p;yaml_vector(p["anchor_xyz"],result.generation.guide.anchor_xyz.at(j));
      const auto& A=result.generation.guide.poly_A[j];
      for (int row=0;row<A.rows();++row) { YAML::Node n; for(int k=0;k<A.cols();++k)n.push_back(A(row,k));p["A"].push_back(n); }
      yaml_vector(p["b"],result.generation.guide.poly_b.at(j));guide["poly_info"].push_back(p);
    }
    std::ofstream out(folder/"guide.yaml");out.exceptions(std::ios::failbit|std::ios::badbit);out << guide;out.close();
  }
  export_playback(request,result,(folder/"playback.html").string());
  std::ofstream out(folder/"result.json");out.exceptions(std::ios::failbit|std::ios::badbit);out << std::setprecision(17) << std::boolalpha;
  auto number=[](double x){return std::isfinite(x)?std::to_string(x):std::string("null");};
  out << "{\n\"success\":" << result.success << ",\n\"status\":" << quote(result.status)
      << ",\n\"validation_model\":" << quote(request.translation_convexification?(request.accept_approximate_hull?"approximate_endpoint_hull_acceptance":"endpoint_boxes_translation_linearization"):"endpoint_swept_hulls")
      << ",\n\"message\":" << quote(result.message) << ",\n\"seed\":" << request.random_seed
      << ",\n\"obstacle_inflation\":" << request.obstacle_inflation
      << ",\n\"final_scene\":" << quote((request.validate_original_scene||request.obstacle_inflation>0)?"original_uninflated":"planning_scene")
      << ",\n\"generation_ms\":" << number(result.generation.generation_ms)
      << ",\n\"optimization_ms\":" << number(result.refinement.optimization_ms)
      << ",\n\"validation_ms\":" << number(result.refinement.validation_ms)
      << ",\n\"total_ms\":" << number(result.total_ms)
      << ",\n\"waypoints\":" << result.refinement.trajectory.size()
      << ",\n\"minimum_distance\":" << number(result.refinement.minimum_distance)
      << ",\n\"normalized_length\":" << number(result.refinement.normalized_length)
      << ",\n\"waypoint_safe\":" << result.refinement.waypoint_safe
      << ",\n\"continuous_safe\":" << result.refinement.continuous_safe
      << ",\n\"smoothing_rolled_back\":" << result.refinement.smoothing_rolled_back
      << ",\n\"continuous_repairs\":" << result.refinement.continuous_repairs
      << ",\n\"generation_status\":" << quote(result.generation.status)
      << ",\n\"refinement_status\":" << quote(result.refinement.status)
      << ",\n\"ik_fallback_count\":";
  int fallback=0;for(int x:result.generation.ik_fallback)fallback+=x!=0;out << fallback << "\n}\n";out.close();
}
}
OptimizerResult CHiTOOptimizer::solve(const OptimizerRequest& request) const {
  OptimizerResult result;const auto started=Clock::now();
  const auto finish=[&]() {
    result.total_ms=milliseconds(started);
    if (!request.output_directory.empty()) try { export_result(request,result); }
    catch(const std::exception& e) {result.export_success=false;result.export_message=e.what();}
    return result;
  };
  const bool finite= request.q_start.allFinite() && request.q_goal.allFinite() && request.domain_lb.allFinite() && request.domain_ub.allFinite() && request.base_xyz.allFinite();
  std::error_code scene_error;
  const bool valid_scene=std::filesystem::is_regular_file(request.scene_path,scene_error);
  if ((!formal_options_valid(request)) || !finite || request.scene_path.empty() || !valid_scene || std::filesystem::path(request.scene_path).extension()!=".scene"
      || !(request.domain_lb.array()<request.domain_ub.array()).all() || !std::isfinite(request.base_yaw_deg)
      || request.corridor_max_iterations<1 || request.ik_max_iterations<1 || request.optimization_max_iterations<1
      || request.continuous_repair_passes<0 || request.repair_waypoint_limit<2
      || request.repair_interior_points<1 || request.repair_interior_points>31
      || !std::isfinite(request.sampling_scale) || request.sampling_scale<=0 || request.sampling_scale>1
      || !std::isfinite(request.ik_position_tolerance) || request.ik_position_tolerance<=0
      || !std::isfinite(request.safety_distance) || request.safety_distance<0
      || !std::isfinite(request.minimum_clearance) || request.minimum_clearance<0
      || !std::isfinite(request.continuous_joint_step_deg) || request.continuous_joint_step_deg<0
      || !std::isfinite(request.continuous_clearance) || request.continuous_clearance<0) {
    result.status="invalid_input";result.message="Invalid scene, joint values, bounds, seed or solver settings";result.total_ms=milliseconds(started);return result;
  }
  std::string phase="corridor_failed";
  if(!std::isfinite(request.obstacle_inflation)||request.obstacle_inflation<0){result.status="invalid_input";result.message="Invalid obstacle inflation";return result;}
  try {
    InflatedScene inflated;auto planning=request;
    if(request.obstacle_inflation>0)planning.scene_path=inflated.create(request.scene_path,request.obstacle_inflation);
    auto context=std::make_shared<rclcpp::Context>();context->init(0,nullptr);
    struct Guard {std::shared_ptr<rclcpp::Context> ctx;~Guard(){if(ctx->is_valid())ctx->shutdown("CHiTO solve finished");}} guard{context};
    rclcpp::NodeOptions options;options.context(context);options.use_global_arguments(false);
    result.generation=detail::generate_corridor(planning,options);
    if (!result.generation.success) { result.status=result.generation.status;result.message=result.generation.message;return finish(); }
    phase="optimization_failed";
    result.refinement=detail::refine_trajectory(planning,result.generation,options);
    if(request.validate_original_scene||request.obstacle_inflation>0)
      result.refinement=detail::validate_original_trajectory(request,result.generation,result.refinement,options);
    result.success=result.refinement.success && result.refinement.waypoint_safe && result.refinement.continuous_safe;
    result.status=result.success?"success":result.refinement.status;
    result.message=result.refinement.message;
  } catch(const std::exception& e) {result.status=phase;result.message=e.what();}
  return finish();
}
}
