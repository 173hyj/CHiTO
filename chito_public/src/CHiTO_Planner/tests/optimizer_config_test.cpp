#include "chito_planner/config.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <cmath>
int main() {
 auto dir=std::filesystem::temp_directory_path()/"chito_config_test"; std::filesystem::create_directories(dir);
 std::ofstream(dir/"scene.scene")<<"test";
 auto write=[&](const std::string& text){std::ofstream(dir/"test.yaml")<<text;};
 auto read=[&](){return chito_planner::read_optimizer_config((dir/"test.yaml").string());};
 auto reject=[&](const std::string& text){write(text);try {read();return false;}catch(const std::exception&){return true;}};
 const std::string head="scene: scene.scene\n";
 write(head+"q_start_deg: [180, 0, 0, 0, 0, 0]\nq_goal_rad: [1, 2, 3, 4, 5, 6]\n");
 auto r=read();
 if(!r.formal_method || !r.continuous_convexification || !r.repair_revisit_existing || r.continuous_clearance!=.005 || r.continuous_joint_step_deg!=1. || !r.validate_original_scene)return 10;
 const std::string joints="q_start_rad: [0,0,0,0,0,0]\nq_goal_rad: [0,0,0,0,0,0]\n";
 if(!reject(head+joints+"translation_convexification: true\n"))return 11;
 if(!reject(head+joints+"continuous_convexification: false\n"))return 12;
 write(head+joints+"formal_method: false\ntranslation_convexification: true\n");
 auto legacy=read();if(legacy.formal_method || !legacy.translation_convexification)return 13;
 if(r.repair_interior_points!=1)return 6;
 write(head+"q_start_deg: [180,0,0,0,0,0]\nq_goal_rad: [1,2,3,4,5,6]\nrepair_interior_points: 3\n");
 if(read().repair_interior_points!=3)return 7;
 if(!reject(head+"q_start_rad: [0,0,0,0,0,0]\nq_goal_rad: [0,0,0,0,0,0]\nrepair_interior_points: 0\n"))return 8;
 if(!reject(head+"q_start_rad: [0,0,0,0,0,0]\nq_goal_rad: [0,0,0,0,0,0]\nrepair_interior_points: 32\n"))return 9;
 if(std::abs(r.q_start[0]-std::acos(-1.))>1e-12||r.q_goal[5]!=6||std::filesystem::path(r.scene_path)!=dir/"scene.scene") return 1;
 if(!reject(head+"q_start_deg: [0]\nq_goal_rad: [0,0,0,0,0,0]\n")) return 2;
 if(!reject(head+"q_start_rad: [0,0,0,0,0,0]\nq_start_deg: [0,0,0,0,0,0]\nq_goal_rad: [0,0,0,0,0,0]\n")) return 3;
 if(!reject(head+"q_start_rad: [.nan,0,0,0,0,0]\nq_goal_rad: [0,0,0,0,0,0]\n")) return 4;
 if(!reject(head+"q_start_rad: [0,0,0,0,0,0]\nq_goal_rad: [0,0,0,0,0,0]\ndomain_lb: [2,2,2]\n")) return 5;
 std::filesystem::remove(dir/"test.yaml");std::filesystem::remove(dir/"scene.scene");std::filesystem::remove(dir);
 std::cout<<"Config parsing tests passed\n";
}
