cmake_minimum_required(VERSION 3.8)
project(planner)

# 建议使用 C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(geometric_shapes REQUIRED)
find_package(moveit_ros_planning REQUIRED)   # ✅ robot_model_loader / planning_scene_monitor 在这里
find_package(moveit_core REQUIRED)    
# ===== 可执行程序 =====
add_executable(plan_cpp_demo src/plan_cpp_demo.cpp)
ament_target_dependencies(
  plan_cpp_demo
  rclcpp
  geometry_msgs
  moveit_ros_planning_interface
  moveit_msgs
  geometric_shapes
)
add_executable(trajectory_player src/trajectory_player.cpp)
ament_target_dependencies(trajectory_player
  rclcpp
  moveit_msgs
  trajectory_msgs
  moveit_ros_planning
  moveit_core
)
add_executable(summarize_perrun src/summarize_perrun.cpp)
ament_target_dependencies(summarize_perrun rclcpp)

install(TARGETS
  summarize_perrun
  DESTINATION lib/${PROJECT_NAME}
)
# ========= traj_player_qp =========
add_executable(traj_player_qp src/traj_player_qp.cpp)

ament_target_dependencies(traj_player_qp
  rclcpp
  std_msgs
  trajectory_msgs
  moveit_core
  moveit_ros_planning
  moveit_ros_planning_interface
  moveit_msgs
)

target_compile_features(traj_player_qp PUBLIC cxx_std_17)

install(TARGETS traj_player_qp
  DESTINATION lib/${PROJECT_NAME}
)
# ---------- rename_final_path_header ----------
add_executable(rename_final_path_header src/rename_final_path_header.cpp)
ament_target_dependencies(rename_final_path_header
  rclcpp
  std_msgs
  trajectory_msgs
  moveit_core
  moveit_ros_planning
  moveit_msgs
)

install(TARGETS rename_final_path_header
  DESTINATION lib/${PROJECT_NAME}
)
# 只加载障碍物的节点
add_executable(scene_scaler src/scene_scaler.cpp)
target_compile_features(scene_scaler PUBLIC cxx_std_17)
target_compile_options(scene_scaler PRIVATE -Wall -Wextra -Wpedantic)

install(TARGETS scene_scaler
  DESTINATION lib/${PROJECT_NAME}
)
add_executable(plan_benchmark_scene3_gripper_waypoints
  src/plan_benchmark_scene3_gripper_waypoints.cpp
)

ament_target_dependencies(plan_benchmark_scene3_gripper_waypoints
  rclcpp
  trajectory_msgs
  geometry_msgs
  shape_msgs
  moveit_msgs
  moveit_ros_planning_interface
  moveit_core
  moveit_ros_planning
  geometric_shapes
)

install(TARGETS plan_benchmark_scene3_gripper_waypoints
  DESTINATION lib/${PROJECT_NAME}
)
add_executable(trajectory_review_scene3 src/trajectory_review_scene3.cpp)

ament_target_dependencies(trajectory_review_scene3
  rclcpp
  std_msgs
  trajectory_msgs
  visualization_msgs
  moveit_msgs
  moveit_core
  moveit_ros_planning
  moveit_ros_planning_interface
)

install(TARGETS trajectory_review_scene3
  DESTINATION lib/${PROJECT_NAME}
)

add_executable(load_obstacles_node src/load_obstacles_node.cpp)
ament_target_dependencies(
  load_obstacles_node
  rclcpp
  moveit_ros_planning_interface
  moveit_msgs
  geometric_shapes
  geometry_msgs
)

install(TARGETS load_obstacles_node
  RUNTIME DESTINATION lib/${PROJECT_NAME})
  
add_executable(plan_benchmark_demo src/plan_benchmark_demo.cpp)
ament_target_dependencies(plan_benchmark_demo
  rclcpp
  geometry_msgs
  moveit_ros_planning_interface
  moveit_ros_planning
  moveit_core
  moveit_msgs
  geometric_shapes
)

install(TARGETS plan_benchmark_demo
  DESTINATION lib/${PROJECT_NAME})

# ===== 安装目标 =====
install(TARGETS plan_cpp_demo
  RUNTIME DESTINATION lib/${PROJECT_NAME})
install(TARGETS
  trajectory_player
  DESTINATION lib/${PROJECT_NAME}
)
# 如有 Python 脚本就装；没有则忽略（不报错）
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/scripts/plan_with_obstacles.py)
  install(PROGRAMS scripts/plan_with_obstacles.py
    DESTINATION lib/${PROJECT_NAME})
endif()

# 安装 launch 文件
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/launch)
  install(DIRECTORY launch/
    DESTINATION share/${PROJECT_NAME}/launch)
endif()

# 安装 meshes（背景/障碍物 STL）
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/meshes)
  install(DIRECTORY meshes/
    DESTINATION share/${PROJECT_NAME}/meshes)
endif()
# 安装 config（YAML 参数文件）
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/config)
  install(DIRECTORY config/
    DESTINATION share/${PROJECT_NAME}/config)
endif()

ament_package()

