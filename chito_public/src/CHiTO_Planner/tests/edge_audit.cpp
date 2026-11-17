#include "chito_planner/config.hpp"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
int main(int argc,char**argv){
 if(argc!=2)return 2;std::filesystem::path root(argv[1]);rclcpp::init(0,nullptr);std::ofstream metrics(root.parent_path()/"edge-metrics.tsv");
 for(const auto& scene:{"scene1","scene2","scene3"})for(int trial=0;trial<5;++trial){
  auto dir=root/(std::string(scene)+"_1pt")/("run_"+std::to_string(trial));
  auto r=chito_planner::read_optimizer_config((dir/"effective_config.yaml").string());
  std::ifstream in(dir/"trajectory.csv");std::string line;std::getline(in,line);chito_planner::JointPath path;
  while(std::getline(in,line)){std::istringstream row(line);std::string c;std::getline(row,c,',');chito_planner::JointVector q;
   for(int k=0;k<6;++k){std::getline(row,c,',');q[k]=std::stod(c);}path.push_back(q);}
  for(size_t edge=0;edge+1<path.size();++edge){
   chito_planner::RefinementResult pair;pair.trajectory={path[edge],path[edge+1]};
   auto coarse=chito_planner::detail::validate_original_trajectory(r,{},pair,rclcpp::NodeOptions{});
   auto delta=(path[edge+1]-path[edge]).cwiseAbs().maxCoeff();int n=std::max(1,int(std::ceil(delta/(std::acos(-1.)/180.))));
   chito_planner::RefinementResult dense;
   for(int j=0;j<=n;++j)dense.trajectory.push_back((1-double(j)/n)*path[edge]+(double(j)/n)*path[edge+1]);
   auto fine=chito_planner::detail::validate_original_trajectory(r,{},dense,rclcpp::NodeOptions{});
   metrics<<std::setprecision(17)<<"EDGE\t"<<scene<<'\t'<<trial<<'\t'<<edge<<'\t'<<delta*180/std::acos(-1.)<<'\t'<<coarse.minimum_distance<<'\t'<<coarse.continuous_safe<<'\t'<<fine.minimum_distance<<'\t'<<fine.continuous_safe<<'\n';
  }
 }
 rclcpp::shutdown();
}
