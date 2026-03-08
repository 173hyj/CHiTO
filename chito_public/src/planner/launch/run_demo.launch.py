# iris_planner_demo/launch/run_demo.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder

def generate_launch_description():
    # 1) 载入 ur5_moveit_config 的 MoveIt 全量参数
    moveit_config = (
        MoveItConfigsBuilder("ur5", package_name="ur5_moveit_config")
        .to_moveit_configs()
    )

    # 2) 启动 move_group（提供 SRDF/规划流水线/robot_description 等）
    move_group = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[moveit_config.to_dict()],
    )

    # 3) 启动你的 C++ demo，并把 moveit 的参数一并传入
    plan_demo = Node(
        package="iris_planner_demo",
        executable="plan_cpp_demo",
        output="screen",
        parameters=[
            moveit_config.to_dict(),                # 让节点也拿到 robot_description 等
            {"planning_group": "arm"},              # 你的 SRDF 里 group 叫 arm
            {"planner_id": "RRTConnectkConfigDefault"},
            {"planning_time": 5.0},

            # 背景/障碍物（注意这些 STL 必须真实存在于包内的 meshes/ 目录）
            {"background_mesh": "package://iris_planner_demo/meshes/a_mesh.stl"},



            {"background_pose_xyz_rpy": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
            {"background_scale_xyz": [0.001, 0.001, 0.001]},

            {"obstacle1_mesh":  "package://iris_planner_demo/meshes/b_mesh.stl"},
            {"obstacle1_pose_xyz_rpy": [0.0, 0.00, 0.0, 0.0, 0.0, 0.0]},
            {"obstacle1_scale_xyz": [0.001, 0.001, 0.001]},

            {"obstacle2_mesh":  "package://iris_planner_demo/meshes/a_mesh.stl"},
            {"obstacle2_pose_xyz_rpy": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
            {"obstacle2_scale_xyz": [0.001, 0.001, 0.001]},

            # run_demo.launch.py 里的 plan_cpp_demo 节点参数中加入：
            {"q_start_deg": [0.0, -70.0, 60.0, 0.0, 75.0, 0.0]},
            {"q_goal_deg":  [10.0, -90.0, 90.0, 0.0, 90.0, 0.0]},
#{"q_start": [0.0, -1.2, 1.2, 0.0, 1.3, 0.0]},
#{"q_goal":  [0.2, -1.57, 1.57, 0.0, 1.57, 0.0]},

        ],
    )

    # 4) 如需 RViz 就取消注释
    # rviz = Node(
    #     package="rviz2",
    #     executable="rviz2",
    #     output="screen",
    #     arguments=["-d", str(moveit_config.package_share_directory / "config/moveit.rviz")],
    #     parameters=[moveit_config.to_dict()],
    # )

    return LaunchDescription([
        move_group,
        plan_demo,
        # rviz,
    ])

