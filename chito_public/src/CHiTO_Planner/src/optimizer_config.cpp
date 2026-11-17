#include "chito_planner/config.hpp"
#include <yaml-cpp/yaml.h>
#include <filesystem>
#include <cmath>
#include <stdexcept>
namespace chito_planner {
namespace {
template<class V> void vector_value(const YAML::Node& n, V& v, int count, bool degrees=false) {
 if (!n.IsSequence() || n.size()!=static_cast<size_t>(count)) throw std::invalid_argument("Expected finite vector of length " + std::to_string(count));
 for(int i=0;i<count;++i) {v[i]=n[i].as<double>();if(degrees)v[i]=v[i]*std::acos(-1.)/180.; if(!std::isfinite(v[i])) throw std::invalid_argument("Nonfinite vector value");}
}
}
void validate_optimizer_request(const OptimizerRequest& r) {
 if(!formal_options_valid(r))
   throw std::invalid_argument("Formal CHiTO requires continuous_convexification and excludes legacy envelope/refinement modes");
 if(!std::isfinite(r.obstacle_inflation)||r.obstacle_inflation<0)throw std::invalid_argument("Obstacle inflation must be finite and nonnegative");
 if(std::filesystem::path(r.scene_path).extension()!=".scene") throw std::invalid_argument("Unified corridor generation requires a .scene file with boxes/spheres");
 if(r.scene_path.empty() || !std::filesystem::is_regular_file(r.scene_path)) throw std::invalid_argument("Scene file does not exist: "+r.scene_path);
 if(!r.q_start.allFinite() || !r.q_goal.allFinite() || !r.domain_lb.allFinite() || !r.domain_ub.allFinite() || !r.base_xyz.allFinite() || !std::isfinite(r.base_yaw_deg)) throw std::invalid_argument("Nonfinite geometry");
 if((r.domain_lb.array()>=r.domain_ub.array()).any()) throw std::invalid_argument("Domain lower bounds must precede upper bounds");
 if(r.corridor_max_iterations<=0 || r.ik_max_iterations<=0 || r.optimization_max_iterations<=0) throw std::invalid_argument("Iteration limits must be positive");
 if(r.continuous_repair_passes<0 || r.repair_waypoint_limit<2) throw std::invalid_argument("Invalid continuous repair budget");
 if(r.repair_interior_points<1 || r.repair_interior_points>31) throw std::invalid_argument("repair_interior_points must be in [1,31]");
 for(double x:{r.sampling_scale,r.ik_position_tolerance}) if(!std::isfinite(x)||x<=0) throw std::invalid_argument("Scale and tolerance must be positive");
 for(double x:{r.safety_distance,r.minimum_clearance,r.continuous_clearance,r.continuous_joint_step_deg}) if(!std::isfinite(x)||x<0) throw std::invalid_argument("Distances must be nonnegative");
}
OptimizerRequest read_optimizer_config(const std::string& filename) {
 auto n=YAML::LoadFile(filename); if(!n.IsMap()) throw std::invalid_argument("Config must be a mapping");
 OptimizerRequest r;
 auto path=std::filesystem::path(n["scene"].as<std::string>());
 if(path.is_relative()) path=std::filesystem::absolute(filename).parent_path()/path;
 r.scene_path=path.lexically_normal().string();
 for(const auto& key:{"q_start","q_goal"}) {
  std::string k(key); bool rad=bool(n[k+"_rad"]), deg=bool(n[k+"_deg"]);
  if(rad==deg) throw std::invalid_argument("Specify exactly one of "+k+"_rad and "+k+"_deg");
  vector_value(n[k+(rad?"_rad":"_deg")],k=="q_start"?r.q_start:r.q_goal,6,!rad);
 }
 if(n["domain_lb"]) vector_value(n["domain_lb"],r.domain_lb,3);
 if(n["domain_ub"]) vector_value(n["domain_ub"],r.domain_ub,3);
 if(n["base_xyz"]) vector_value(n["base_xyz"],r.base_xyz,3);
#define READ(field,type) if(n[#field]) r.field=n[#field].as<type>();
 READ(formal_method,bool)
 READ(base_yaw_deg,double) READ(random_seed,std::uint32_t)
 READ(corridor_max_iterations,int) READ(ik_max_iterations,int) READ(optimization_max_iterations,int)
 READ(continuous_repair_passes,int) READ(repair_waypoint_limit,int)
 READ(repair_interior_points,int) READ(repair_revisit_existing,bool) READ(continuous_joint_step_deg,double)
 READ(sampling_scale,double) READ(ik_position_tolerance,double) READ(safety_distance,double)
 READ(minimum_clearance,double) READ(continuous_clearance,double) READ(continuous_convexification,bool) READ(translation_convexification,bool) READ(accept_approximate_hull,bool)
 READ(final_smoothing,bool) READ(export_intermediates,bool) READ(continuous_collision_refinement,bool)
 READ(obstacle_inflation,double) READ(validate_original_scene,bool)
#undef READ
 validate_optimizer_request(r); return r;
}
}
