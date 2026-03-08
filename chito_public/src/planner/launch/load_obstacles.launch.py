# iris_planner_demo/launch/load_obstacles.launch.py
# -*- coding: utf-8 -*-

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node

from moveit_configs_utils import MoveItConfigsBuilder
from ament_index_python.packages import get_package_share_directory
from pathlib import Path

def generate_launch_description():
    # 参数：是否启动 RViz
    use_rviz_arg = DeclareLaunchArgument(
        "use_rviz", default_value="true",
        description="Start RViz with MoveIt config"
    )
    use_rviz = LaunchConfiguration("use_rviz")

    # MoveIt 配置（UR5）
    moveit_config = (
        MoveItConfigsBuilder("ur5", package_name="ur5_moveit_config")
        .to_moveit_configs()
    )

    # move_group 节点
    move_group = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[moveit_config.to_dict()],
    )

    # 解析 ur5_moveit_config 包内的 RViz 配置（若不存在则不传 -d）
    rviz_args = []
    try:
        ur5_cfg_share = Path(get_package_share_directory("ur5_moveit_config"))
        rviz_file = ur5_cfg_share / "config" / "moveit.rviz"
        if rviz_file.is_file():
            rviz_args = ["-d", str(rviz_file)]
    except Exception:
        # 找不到包或 rviz 文件时，直接用空参数启动 rviz
        rviz_args = []

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        output="screen",
        arguments=rviz_args,
        parameters=[moveit_config.to_dict()],
        condition=IfCondition(use_rviz),
    )

    # 仅加载障碍物的节点
    load_obs = Node(
        package="iris_planner_demo",
        executable="load_obstacles_node",
        output="screen",
        parameters=[
            {"frame_id": "world"},
            {"clear_existing": True},
            {"obstacle_count": 2},

            {"use_background": True},
            {"background_mesh": "package://iris_planner_demo/meshes/a_mesh3.stl"},
            {"background_pose_xyz_rpy": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
            {"background_scale_xyz": [0.001, 0.001, 0.001]},

            {"obstacle1_id": "obs_1"},
            {"obstacle1_mesh": "package://iris_planner_demo/meshes/a_mesh3.stl"},
            {"obstacle1_pose_xyz_rpy": [0.0, 0.00, 0.0, 0.0, 0.0, 0.0]},
            {"obstacle1_scale_xyz": [0.001, 0.001, 0.001]},

            {"obstacle2_id": "obs_2"},
            {"obstacle2_mesh": "package://iris_planner_demo/meshes/a_mesh3.stl"},
            {"obstacle2_pose_xyz_rpy": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
            {"obstacle2_scale_xyz": [0.001, 0.001, 0.001]},
        ],
    )

    return LaunchDescription([
        use_rviz_arg,
        move_group,
        rviz,
        load_obs,
    ])

