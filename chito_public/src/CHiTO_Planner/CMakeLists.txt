cmake_minimum_required(VERSION 3.10)
project(CHiTO_Planner)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# =========================
# Basic dependencies
# =========================
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(ament_index_cpp REQUIRED)
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
option(CHITO_BUILD_CORRIDOR_TOOLS "Build legacy standalone Drake-based corridor tools" OFF)
option(CHITO_BUILD_UNIFIED_OPTIMIZER "Build in-memory CHiTO optimizer and CLI" ON)
if(CHITO_BUILD_CORRIDOR_TOOLS OR CHITO_BUILD_UNIFIED_OPTIMIZER)
set(DRAKE_PREFIX "" CACHE PATH "Drake installation prefix")
if(DRAKE_PREFIX)
list(PREPEND CMAKE_PREFIX_PATH "${DRAKE_PREFIX}")
endif()
find_package(drake CONFIG REQUIRED)
endif()
find_package(osqp CONFIG REQUIRED)

# =========================================================
# CHiTO reusable algorithm modules
# =========================================================
add_library(chito_planner_core
  src/core/convex_hull.cpp
  src/core/robot_geometry.cpp
  src/optimization/qp_solver.cpp
)
target_include_directories(chito_planner_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${EIGEN3_INCLUDE_DIRS}
)
ament_target_dependencies(chito_planner_core
  rclcpp
)
target_link_libraries(chito_planner_core
  osqp::osqp
  fcl
  Eigen3::Eigen
  pthread
)


# =========================================================
# 1. corridor_spacerrt_ik_viz
# =========================================================
if(CHITO_BUILD_CORRIDOR_TOOLS)
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

endif()

# =========================================================
# 2. corridor_viz_moveit_ik
# =========================================================
if(CHITO_BUILD_CORRIDOR_TOOLS)
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

endif()

# =========================================================
# 3. Corridor_Viz_Moveit_Initik
# =========================================================
if(CHITO_BUILD_CORRIDOR_TOOLS)
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

endif()

# =========================================================
# 4. Corridor_Viz_Moveit_Initik_withtorch
# =========================================================
if(CHITO_BUILD_CORRIDOR_TOOLS)
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

endif()

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
  chito_planner_core
  osqp::osqp
  fcl
  pthread
  yaml-cpp
  Eigen3::Eigen
)


# =========================================================
# 7. scene_spacerrt_dh_ik_stats
# =========================================================
if(CHITO_BUILD_CORRIDOR_TOOLS)
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

endif()

# =========================================================
# 8. scene_spacerrt_dh_ik_stats_withtorch
# =========================================================
if(CHITO_BUILD_CORRIDOR_TOOLS)
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

endif()

# =========================
# Install targets
# =========================
install(TARGETS
  chito_planner_core
  generate_perturbed_joint_paths
  hierarchical_opt
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

# =========================
# Install launch / config
# =========================
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
  PATTERN "chito_scene*.yaml" EXCLUDE
)
foreach(scene RANGE 1 3)
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/config/chito_scene${scene}.yaml" preset)
  string(REPLACE "../../collision/" "scenes/" preset "${preset}")
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/chito_scene${scene}.yaml" "${preset}")
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/chito_scene${scene}.yaml"
    DESTINATION share/${PROJECT_NAME}/config)
endforeach()

ament_export_include_directories(include)
ament_export_libraries(chito_planner_core)

if(CHITO_BUILD_UNIFIED_OPTIMIZER)
  set_target_properties(chito_planner_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
  add_library(chito_optimizer_library SHARED
    src/unified_optimizer.cpp src/optimizer_config.cpp src/optimizer_playback.cpp
    src/hierarchical_opt.cpp src/scene_spacerrt_dh_ik_stats.cpp)
  target_compile_definitions(chito_optimizer_library PRIVATE CHITO_STAGE_LIBRARY)
  set_target_properties(chito_optimizer_library PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
  target_include_directories(chito_optimizer_library PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
  target_link_libraries(chito_optimizer_library PRIVATE chito_planner_core drake::drake yaml-cpp PUBLIC Eigen3::Eigen fcl)
  ament_target_dependencies(chito_optimizer_library PUBLIC rclcpp visualization_msgs geometry_msgs moveit_core moveit_ros_planning moveit_ros_planning_interface)
  add_executable(chito_optimizer src/chito_optimizer_cli.cpp)
  target_link_libraries(chito_optimizer chito_optimizer_library)
  ament_target_dependencies(chito_optimizer ament_index_cpp)
  install(TARGETS chito_optimizer_library EXPORT chito_optimizer_targets
    ARCHIVE DESTINATION lib LIBRARY DESTINATION lib)
  install(TARGETS chito_optimizer DESTINATION lib/${PROJECT_NAME})
  install(FILES ../collision/scene1.scene ../collision/scene2.scene ../collision/scene3.scene
    DESTINATION share/${PROJECT_NAME}/config/scenes)
  ament_export_targets(chito_optimizer_targets HAS_LIBRARY_TARGET)
  ament_export_dependencies(rclcpp visualization_msgs geometry_msgs moveit_core moveit_ros_planning moveit_ros_planning_interface Eigen3 fcl)
endif()

if(BUILD_TESTING)
  enable_testing()
  add_executable(repair_interpolation_test tests/repair_interpolation_test.cpp)
  target_include_directories(repair_interpolation_test PRIVATE include ${EIGEN3_INCLUDE_DIRS})
  target_compile_options(repair_interpolation_test PRIVATE -UNDEBUG)
  add_test(NAME repair_interpolation COMMAND repair_interpolation_test)
  add_executable(qp_solver_test tests/qp_solver_test.cpp src/optimization/qp_solver.cpp)
  target_include_directories(qp_solver_test PRIVATE include ${EIGEN3_INCLUDE_DIRS})
  target_link_libraries(qp_solver_test osqp::osqp Eigen3::Eigen)
  target_compile_options(qp_solver_test PRIVATE -UNDEBUG)
  add_test(NAME qp_solver_analytical COMMAND qp_solver_test)
  add_executable(swept_sphere_test tests/swept_sphere_test.cpp)
  target_link_libraries(swept_sphere_test fcl Eigen3::Eigen)
  target_compile_options(swept_sphere_test PRIVATE -UNDEBUG)
  add_test(NAME swept_sphere_clearance COMMAND swept_sphere_test)
  if(CHITO_BUILD_UNIFIED_OPTIMIZER)
  add_executable(unified_optimizer_test tests/unified_optimizer_test.cpp)
  target_include_directories(unified_optimizer_test PRIVATE include)
  target_link_libraries(unified_optimizer_test chito_optimizer_library yaml-cpp)
  ament_target_dependencies(unified_optimizer_test rclcpp)
  target_compile_options(unified_optimizer_test PRIVATE -UNDEBUG)
  target_compile_definitions(unified_optimizer_test PRIVATE CHITO_TEST_SCENE="${CMAKE_CURRENT_SOURCE_DIR}/../collision/scene2.scene")
  add_test(NAME unified_optimizer_contract COMMAND unified_optimizer_test)
  add_executable(optimizer_config_test tests/optimizer_config_test.cpp)
  target_link_libraries(optimizer_config_test chito_optimizer_library)
  target_compile_options(optimizer_config_test PRIVATE -UNDEBUG)
  add_test(NAME unified_optimizer_configuration COMMAND optimizer_config_test)
  endif()
endif()
ament_package()

if(CHITO_BUILD_CORRIDOR_TOOLS)
install(TARGETS corridor_spacerrt_ik_viz_node corridor_viz_moveit_ik corridor_viz_moveit_initik corridor_viz_moveit_initik_withtorch scene_spacerrt_dh_ik_stats scene_spacerrt_dh_ik_stats_withtorch DESTINATION lib/${PROJECT_NAME})
endif()

if(BUILD_TESTING AND CHITO_BUILD_UNIFIED_OPTIMIZER)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  add_test(NAME failed_node_revisited COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/revisit_regression.py
    $<TARGET_FILE:chito_optimizer>
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/revisit_scene1.yaml
    ${CMAKE_CURRENT_SOURCE_DIR}/../collision/scene1.scene
    ${CMAKE_CURRENT_BINARY_DIR}/revisit-regression)

  add_test(NAME swept_merit_consistency COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/swept_merit_regression.py
    $<TARGET_FILE:chito_optimizer>
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/merit_scene3.yaml
    ${CMAKE_CURRENT_SOURCE_DIR}/../collision/scene3.scene
    ${CMAKE_CURRENT_BINARY_DIR}/merit-regression)
endif()

if(BUILD_TESTING AND CHITO_BUILD_UNIFIED_OPTIMIZER)
add_executable(edge_audit tests/edge_audit.cpp)
target_link_libraries(edge_audit chito_optimizer_library)
ament_target_dependencies(edge_audit rclcpp)
endif()

if(BUILD_TESTING AND CHITO_BUILD_UNIFIED_OPTIMIZER)
add_executable(subedge_regression tests/subedge_regression.cpp)
target_link_libraries(subedge_regression chito_optimizer_library)
ament_target_dependencies(subedge_regression rclcpp)
add_test(NAME long_joint_edge_collision COMMAND subedge_regression
 ${CMAKE_CURRENT_SOURCE_DIR}/tests/subedge_scene2.yaml
 ${CMAKE_CURRENT_SOURCE_DIR}/tests/subedge_scene2.csv)
endif()

if(BUILD_TESTING)
add_executable(global_acceptance_trial tests/global_acceptance_trial.cpp src/optimizer_config.cpp)
target_link_libraries(global_acceptance_trial chito_planner_core osqp::osqp fcl pthread yaml-cpp Eigen3::Eigen)
ament_target_dependencies(global_acceptance_trial rclcpp visualization_msgs geometry_msgs moveit_core moveit_ros_planning moveit_ros_planning_interface)
endif()

if(BUILD_TESTING)
add_executable(local_acceptance_test tests/local_acceptance_test.cpp)
target_compile_options(local_acceptance_test PRIVATE -UNDEBUG)
target_include_directories(local_acceptance_test PRIVATE include)
add_test(NAME local_feasibility_acceptance COMMAND local_acceptance_test)
endif()

if(BUILD_TESTING)
add_executable(repair_window_test tests/repair_window_test.cpp)
target_compile_options(repair_window_test PRIVATE -UNDEBUG)
target_include_directories(repair_window_test PRIVATE include)
add_test(NAME collision_free_window COMMAND repair_window_test)
endif()

if(BUILD_TESTING)
add_executable(corridor_endpoint_test tests/corridor_endpoint_test.cpp)
target_compile_options(corridor_endpoint_test PRIVATE -UNDEBUG)
target_link_libraries(corridor_endpoint_test chito_planner_core osqp::osqp Eigen3::Eigen)
add_test(NAME corridor_moving_endpoint_constraints COMMAND corridor_endpoint_test)
endif()

if(BUILD_TESTING)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/corridor_fixture)
configure_file(tests/corridor_fixture/effective_config.yaml.in corridor_fixture/effective_config.yaml @ONLY)
configure_file(tests/corridor_fixture/guide.yaml corridor_fixture/guide.yaml COPYONLY)
configure_file(tests/corridor_fixture/seed.csv corridor_fixture/seed.csv COPYONLY)
add_test(NAME corridor_endpoint_actual_arm COMMAND global_acceptance_trial
 ${CMAKE_CURRENT_BINARY_DIR}/corridor_fixture ${CMAKE_CURRENT_BINARY_DIR}/corridor-check fixed corridor-check)
add_test(NAME corridor_endpoint_batch_reset COMMAND global_acceptance_trial
 ${CMAKE_CURRENT_BINARY_DIR}/corridor_fixture ${CMAKE_CURRENT_BINARY_DIR}/corridor-reset-check fixed corridor-reset-check)
endif()

if(BUILD_TESTING)
add_executable(split_corridor_test tests/split_corridor_test.cpp)
target_compile_options(split_corridor_test PRIVATE -UNDEBUG)
target_include_directories(split_corridor_test PRIVATE include)
add_test(NAME split_corridor_layout_acceptance COMMAND split_corridor_test)
add_test(NAME split_corridor_actual_arm COMMAND global_acceptance_trial
 ${CMAKE_CURRENT_BINARY_DIR}/corridor_fixture ${CMAKE_CURRENT_BINARY_DIR}/split-check fixed split-check)
endif()

if(BUILD_TESTING)
add_test(NAME split_corridor_public_validation_gate COMMAND global_acceptance_trial
 ${CMAKE_CURRENT_BINARY_DIR}/corridor_fixture ${CMAKE_CURRENT_BINARY_DIR}/split-public-check fixed split-public-gate-check)
endif()

if(BUILD_TESTING AND CHITO_BUILD_UNIFIED_OPTIMIZER)
  add_executable(formal_method_test tests/formal_method_test.cpp)
  target_link_libraries(formal_method_test chito_optimizer_library yaml-cpp)
  target_compile_options(formal_method_test PRIVATE -UNDEBUG)
  target_compile_definitions(formal_method_test PRIVATE CHITO_FORMAL_CONFIG="${CMAKE_CURRENT_BINARY_DIR}/corridor_fixture/effective_config.yaml")
  add_test(NAME formal_method_public_refinement COMMAND formal_method_test)
endif()

if(BUILD_TESTING AND CHITO_BUILD_UNIFIED_OPTIMIZER)
  add_executable(chito_replay tests/chito_replay.cpp)
  target_link_libraries(chito_replay chito_optimizer_library yaml-cpp)
endif()
