from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.substitutions import Command
from launch_ros.parameter_descriptions import ParameterValue
import os

from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import generate_move_group_launch

def generate_launch_description():
    # 你的 xacro 路径（按你给的实际位置）
    xacro_path = os.path.join(
        get_package_share_directory("hyj_ur5_robotiq_description"),
        "urdf",
        "ur5_robotiq_2f85.urdf.xacro",
    )

    robot_description = {
        "robot_description": ParameterValue(
            Command(["xacro ", xacro_path]),
            value_type=str
        )
    }

    # MoveIt config（仍用 ur5_robotiq_moveit_config 的 SRDF/kinematics/pipelines 等）
    moveit_config = (
        MoveItConfigsBuilder("ur5_robotiq_2f85", package_name="ur5_robotiq_moveit_config")
        .to_moveit_configs()
    )

    # 只启动 move_group（不需要 demo 全家桶）
    move_group_ld = generate_move_group_launch(moveit_config)

    # ✅ 发布 /joint_states（否则 move_group/current_state_monitor 永远是空）
    jsp = Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
        name="joint_state_publisher",
        output="screen",
        parameters=[robot_description],
    )

    # TF（可选但建议有）
    rsp = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[robot_description],
    )

    # 你的 benchmark 参数
    bench_yaml = os.path.join(
        get_package_share_directory("iris_planner_demo"),
        "config",
        "scene3_gripper.yaml",
    )

    bench = Node(
        package="iris_planner_demo",
        executable="plan_benchmark_scene3_gripper_waypoints",
        name="plan_benchmark_scene3_gripper_waypoints",
        output="screen",
        parameters=[
            robot_description,
            bench_yaml,
            # 如果你想保险点，也可以把 moveit_config 的那堆一起塞进来：
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
            moveit_config.joint_limits,
        ],
    )

    return LaunchDescription([
        rsp,
        jsp,
        move_group_ld,
        bench,
    ])
