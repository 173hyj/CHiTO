#include "chito_planner/config.hpp"
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <limits>
#include <stdexcept>
int main(int argc,char** argv) {
 try {
  std::string config,output,scene; int trials=1; bool smooth=false,seed_set=false; std::uint32_t seed=0;
  auto number=[](const std::string& s)->unsigned long long {size_t p=0; if(s.empty()||s[0]=='-') throw std::invalid_argument("Invalid integer"); auto v=std::stoull(s,&p); if(p!=s.size()) throw std::invalid_argument("Invalid integer"); return v;};
  for(int i=1;i<argc;++i) {
   std::string a=argv[i]; if(a=="--smooth") {smooth=true;continue;}
   if(a=="--help") {std::cout<<"chito_optimizer (--scene scene1|scene2|scene3 | --config FILE) --output DIR [--trials N] [--seed N] [--smooth]\n";return 0;}
   if(i+1>=argc) throw std::invalid_argument("Missing option value"); std::string v=argv[++i];
   if(a=="--config") config=v; else if(a=="--output") output=v;
   else if(a=="--scene") scene=v;
   else if(a=="--trials") {auto x=number(v);if(x==0||x>1000000) throw std::invalid_argument("Trials out of range");trials=static_cast<int>(x);}
   else if(a=="--seed") {auto x=number(v);if(x>UINT32_MAX) throw std::invalid_argument("Seed out of range");seed=static_cast<std::uint32_t>(x);seed_set=true;}
   else throw std::invalid_argument("Unknown option: "+a);
  }
  if(!scene.empty()) {
   if(!config.empty())throw std::invalid_argument("Choose either --scene or --config");
   if(scene!="scene1"&&scene!="scene2"&&scene!="scene3")throw std::invalid_argument("Unknown scene");
   config=(std::filesystem::path(ament_index_cpp::get_package_share_directory("CHiTO_Planner"))/"config"/("chito_"+scene+".yaml")).string();
  }
  if(config.empty()||output.empty()) throw std::invalid_argument("--scene or --config, and --output are required");
  auto request=chito_planner::read_optimizer_config(config); if(seed_set) request.random_seed=seed;
  if(static_cast<unsigned long long>(request.random_seed)+trials-1>UINT32_MAX) throw std::invalid_argument("Trial seeds overflow");
  if(smooth) request.final_smoothing=true;
  if(request.translation_convexification&&request.final_smoothing)throw std::invalid_argument("Final smoothing is not enabled in the endpoint-envelope pipeline");
  rclcpp::init(0,nullptr); chito_planner::CHiTOOptimizer optimizer; int successes=0,failed=0,exports=0; auto initial=request.random_seed;
  for(int i=0;i<trials;++i) {
   request.random_seed=initial+i;
   request.output_directory=(std::filesystem::path(output)/("run_"+std::to_string(i))).string();
   auto result=optimizer.solve(request); if(result.success) ++successes;else ++failed; if(!result.export_success) ++exports;
   if(!result.success) std::cerr << "trial " << i << ": " << result.status << ": " << result.message << '\n';
   if(!result.export_success) std::cerr << "trial " << i << " export: " << result.export_message << '\n';
  }
  rclcpp::shutdown();
  std::ofstream summary(std::filesystem::path(output)/"summary.json");
  if(summary){summary<<"{\"trials\":"<<trials<<",\"successes\":"<<successes<<",\"planning_failures\":"<<failed<<",\"export_failures\":"<<exports<<"}\n";summary.close();if(!summary)++exports;}
  else {std::cerr<<"Cannot write batch summary\n";++exports;}
  std::cout<<"{\"trials\":"<<trials<<",\"successes\":"<<successes<<",\"planning_failures\":"<<failed<<",\"export_failures\":"<<exports<<"}\n";
  return exports?3:(failed?1:0);
 } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 2;}
}
