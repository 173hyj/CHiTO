#include "chito_planner/config.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
int main(int argc,char**argv){
 if(argc!=3)return 2;rclcpp::init(0,nullptr);
 auto r=chito_planner::read_optimizer_config(argv[1]);
 std::ifstream f(argv[2]);std::string line;std::getline(f,line);chito_planner::RefinementResult input;
 while(std::getline(f,line)){std::istringstream row(line);std::string cell;std::getline(row,cell,',');chito_planner::JointVector q;for(int k=0;k<6;++k){std::getline(row,cell,',');q[k]=std::stod(cell);}input.trajectory.push_back(q);}
 auto out=chito_planner::detail::validate_original_trajectory(r,{},input,rclcpp::NodeOptions{});
 std::cout<<"point_safe="<<out.waypoint_safe<<" hull_safe="<<out.continuous_safe<<'\n';
 rclcpp::shutdown();
 if(!out.waypoint_safe)return 3;
 if(out.continuous_safe){std::cerr<<"Missed known interpolation collision on 137-degree edge\n";return 1;}
 return 0;
}
