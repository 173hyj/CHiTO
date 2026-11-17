#include "chito_planner/config.hpp"
#include <yaml-cpp/yaml.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
namespace fs=std::filesystem;
int main(int argc,char**argv){
 try{
  if(argc!=4)throw std::invalid_argument("usage: chito_replay INPUT_DIR OUTPUT_DIR formal|shared");
  const fs::path input=argv[1],output=argv[2];const std::string mode=argv[3];
  if(mode!="formal" && mode!="shared")throw std::invalid_argument("mode must be formal or shared");
  if(fs::exists(output))throw std::invalid_argument("output directory must be new");
  auto request=chito_planner::read_optimizer_config((input/"effective_config.yaml").string());
  auto data=YAML::LoadFile((input/"guide.yaml").string());
  chito_planner::GenerationResult g;g.success=true;g.guide.loaded=true;
  auto vec=[](const YAML::Node& n){chito_planner::JointVector q;if(n.size()!=6)throw std::invalid_argument("expected six joints");for(int k=0;k<6;++k)q[k]=n[k].as<double>();return q;};
  std::ifstream csv(input/"seed.csv");if(!csv)throw std::invalid_argument("missing seed.csv");std::string line,cell;std::getline(csv,line);
  while(std::getline(csv,line)){std::istringstream row(line);std::getline(row,cell,',');chito_planner::JointVector q;for(int k=0;k<6;++k){if(!std::getline(row,cell,','))throw std::invalid_argument("invalid seed row");q[k]=std::stod(cell);}g.seed.push_back(q);}
  for(const auto& row:data["q_seed_dense"])g.guide.q_seed_dense.push_back(vec(row));
  g.guide.sigma_dense=data["sigma_dense"].as<std::vector<int>>();
  for(const auto& poly:data["poly_info"]){Eigen::MatrixXd A(poly["A"].size(),3);Eigen::VectorXd b(poly["b"].size());if(A.rows()!=b.size())throw std::invalid_argument("halfspace dimensions");for(int j=0;j<A.rows();++j){b[j]=poly["b"][j].as<double>();for(int k=0;k<3;++k)A(j,k)=poly["A"][j][k].as<double>();}g.guide.poly_A.push_back(A);g.guide.poly_b.push_back(b);Eigen::Vector3d anchor;for(int k=0;k<3;++k)anchor[k]=poly["anchor_xyz"][k].as<double>();g.guide.anchor_xyz.push_back(anchor);}
  if(g.seed.size()!=g.guide.q_seed_dense.size()||g.seed.size()!=g.guide.sigma_dense.size())throw std::invalid_argument("guide/seed dimensions");
  for(size_t j=0;j<g.seed.size();++j)if(!g.seed[j].allFinite()||(g.seed[j]-g.guide.q_seed_dense[j]).norm()>1e-12)throw std::invalid_argument("seed/guide mismatch");
  rclcpp::init(0,nullptr);rclcpp::NodeOptions options;options.use_global_arguments(false);
  if(mode=="shared"){
   request.formal_method=false;
   options.parameter_overrides({rclcpp::Parameter("global_actual_acceptance",true),rclcpp::Parameter("local_feasibility_first",true),rclcpp::Parameter("corridor_moving_endpoints",true),rclcpp::Parameter("split_corridor_nodes",false),rclcpp::Parameter("torch_enable",false),rclcpp::Parameter("torch_in_collision",false),rclcpp::Parameter("gripper_enable",false)});
  }
  auto result=chito_planner::detail::refine_trajectory(request,g,options);
  if(request.validate_original_scene)result=chito_planner::detail::validate_original_trajectory(request,g,result,options);
  fs::create_directories(output);std::ofstream out(output/"trajectory.csv");out<<std::setprecision(17)<<"idx,q1_rad,q2_rad,q3_rad,q4_rad,q5_rad,q6_rad\n";
  for(size_t j=0;j<result.trajectory.size();++j){out<<j;for(int k=0;k<6;++k)out<<','<<result.trajectory[j][k];out<<'\n';}out.close();
  std::ofstream json(output/"result.json");json<<std::boolalpha<<std::setprecision(17)<<"{\"success\":"<<result.success<<",\"status\":\""<<result.status<<"\",\"waypoint_safe\":"<<result.waypoint_safe<<",\"continuous_safe\":"<<result.continuous_safe<<",\"nodes\":"<<result.trajectory.size()<<",\"minimum_distance\":"<<result.minimum_distance<<",\"optimization_ms\":"<<result.optimization_ms<<"}\n";
  rclcpp::shutdown();return 0; // Replay completion; inspect result.success for planning success.
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
