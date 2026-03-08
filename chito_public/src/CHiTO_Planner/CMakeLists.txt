cmake_minimum_required(VERSION 3.10)
project(CHiTO_Planner)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# =========================
# Basic dependencies
# =========================
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(fcl REQUIRED)

# =========================
# tf2 / MoveIt
# =========================
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_ros_planning REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(shape_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)

# =========================
# Drake
# =========================
set(DRAKE_PREFIX "/home/hyj/drake/build/install")
set(drake_DIR "${DRAKE_PREFIX}/lib/cmake/drake")
find_package(drake CONFIG REQUIRED)

# =========================
# Gurobi
# =========================
if(NOT DEFINED ENV{GUROBI_HOME})
  set(GUROBI_HOME "/opt/gurobi1203/linux64")
else()
  set(GUROBI_HOME $ENV{GUROBI_HOME})
endif()

find_path(GUROBI_INCLUDE_DIR
  NAMES gurobi_c++.h
  HINTS ${GUROBI_HOME}/include /opt/gurobi1203/linux64/include
)

find_library(GUROBI_CXX_LIB
  NAMES gurobi_c++ gurobi_c++.so
  HINTS ${GUROBI_HOME}/lib /opt/gurobi1203/linux64/lib
)

find_library(GUROBI_CORE_LIB
  NAMES gurobi120
  HINTS ${GUROBI_HOME}/lib /opt/gurobi1203/linux64/lib
)

if(NOT GUROBI_INCLUDE_DIR OR NOT GUROBI_CXX_LIB OR NOT GUROBI_CORE_LIB)
  message(FATAL_ERROR "Failed to find Gurobi. Set GUROBI_HOME correctly.")
endif()

# =========================================================
# 1. corridor_spacerrt_ik_viz
# =========================================================
add_executable(corridor_spacerrt_ik_viz_node
  src/corridor_spacerrt_ik_viz.cpp
)
ament_target_dependencies(corridor_spacerrt_ik_viz_node
  rclcpp
  geometry_msgs
  visualization_msgs
)
target_include_directories(corridor_spacerrt_ik_viz_node PRIVATE ${EIGEN3_INCLUDE_DIRS})
target_link_libraries(corridor_spacerrt_ik_viz_node
  drake::drake
)

# =========================================================
# 2. corridor_viz_moveit_ik
# =========================================================
add_executable(corridor_viz_moveit_ik
  src/corridor_viz_moveit_ik.cpp
)
ament_target_dependencies(corridor_viz_moveit_ik
  rclcpp
  visualization_msgs
  geometry_msgs
  tf2_ros
  moveit_core
  moveit_ros_planning
)
target_link_libraries(corridor_viz_moveit_ik
  drake::drake
  Eigen3::Eigen
  yaml-cpp
)

# =========================================================
# 3. Corridor_Viz_Moveit_Initik
# =========================================================
add_executable(corridor_viz_moveit_initik
  src/Corridor_Viz_Moveit_Initik.cpp
)
ament_target_dependencies(corridor_viz_moveit_initik
  rclcpp
  visualization_msgs
  geometry_msgs
  tf2_ros
  moveit_core
  moveit_ros_planning
)
target_link_libraries(corridor_viz_moveit_initik
  drake::drake
  Eigen3::Eigen
  yaml-cpp
)

# =========================================================
# 4. Corridor_Viz_Moveit_Initik_withtorch
# =========================================================
add_executable(corridor_viz_moveit_initik_withtorch
  src/Corridor_Viz_Moveit_Initik_withtorch.cpp
)
ament_target_dependencies(corridor_viz_moveit_initik_withtorch
  rclcpp
  visualization_msgs
  geometry_msgs
  tf2_ros
  moveit_core
  moveit_ros_planning
)
target_link_libraries(corridor_viz_moveit_initik_withtorch
  drake::drake
  Eigen3::Eigen
  yaml-cpp
)

# =========================================================
# 5. generate_perturbed_joint_paths
# =========================================================
add_executable(generate_perturbed_joint_paths
  src/generate_perturbed_joint_paths.cpp
)
ament_target_dependencies(generate_perturbed_joint_paths
  rclcpp
)

# =========================================================
# 6. hierarchical_opt
# =========================================================
add_executable(hierarchical_opt
  src/hierarchical_opt.cpp
)
target_include_directories(hierarchical_opt PRIVATE
  ${GUROBI_INCLUDE_DIR}
  ${EIGEN3_INCLUDE_DIRS}
)
ament_target_dependencies(hierarchical_opt
  rclcpp
  visualization_msgs
  geometry_msgs
  moveit_core
  moveit_ros_planning
  moveit_ros_planning_interface
)
target_link_libraries(hierarchical_opt
  ${GUROBI_CXX_LIB}
  ${GUROBI_CORE_LIB}
  fcl
  pthread
  yaml-cpp
  Eigen3::Eigen
)
set_target_properties(hierarchical_opt PROPERTIES
  INSTALL_RPATH "${GUROBI_HOME}/lib"
  BUILD_RPATH   "${GUROBI_HOME}/lib"
)

# =========================================================
# 7. scene_spacerrt_dh_ik_stats
# =========================================================
add_executable(scene_spacerrt_dh_ik_stats
  src/scene_spacerrt_dh_ik_stats.cpp
)
ament_target_dependencies(scene_spacerrt_dh_ik_stats
  rclcpp
  visualization_msgs
  geometry_msgs
)
target_link_libraries(scene_spacerrt_dh_ik_stats
  drake::drake
  Eigen3::Eigen
  yaml-cpp
)

# =========================================================
# 8. scene_spacerrt_dh_ik_stats_withtorch
# =========================================================
add_executable(scene_spacerrt_dh_ik_stats_withtorch
  src/scene_spacerrt_dh_ik_stats_withtorch.cpp
)
ament_target_dependencies(scene_spacerrt_dh_ik_stats_withtorch
  rclcpp
  visualization_msgs
  geometry_msgs
)
target_link_libraries(scene_spacerrt_dh_ik_stats_withtorch
  drake::drake
  Eigen3::Eigen
)

# =========================
# Install targets
# =========================
install(TARGETS
  corridor_spacerrt_ik_viz_node
  corridor_viz_moveit_ik
  corridor_viz_moveit_initik
  corridor_viz_moveit_initik_withtorch
  generate_perturbed_joint_paths
  hierarchical_opt
  scene_spacerrt_dh_ik_stats
  scene_spacerrt_dh_ik_stats_withtorch
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

# =========================
# Install launch / config
# =========================
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

ament_package()
