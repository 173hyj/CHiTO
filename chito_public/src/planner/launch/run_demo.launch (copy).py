# iris_planner_demo/launch/run_demo.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder

def generate_launch_description():
    # 1) 载入 ur5_moveit_config 的全部 MoveIt 配置
    moveit_config = (
        MoveItConfigsBuilder("ur5", package_name="ur5_moveit_config")
        .to_moveit_configs()
    )

    # 2) 启动 move_group（必要：解析 SRDF 分组、规划流水线等）
    move_group = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[moveit_config.to_dict()],
    )

 nodes.append(
        Node(
            package="iris_planner_demo",
            executable="plan_cpp_demo",
            output="screen",
            parameters=[
                {"planning_group": "arm"},
                {"planner_id": "RRTConnectkConfigDefault"},
                {"planning_time": 5.0},

                # 背景/障碍物 mesh 路径（也可以不传，用默认）
                {"background_mesh": "package://iris_planner_demo/meshes/floor_plane.stl"},
                {"background_pose_xyz_rpy": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
                {"background_scale_xyz": [0.001, 0.001, 0.001]},

                {"obstacle1_mesh": "package://iris_planner_demo/meshes/obstacle_box1.stl"},
                {"obstacle1_pose_xyz_rpy": [0.0, 0.00, 0.0, 0.0, 0.0, 0.0]},
                {"obstacle1_scale_xyz": [0.001, 0.001, 0.001]},

                {"obstacle2_mesh": "package://iris_planner_demo/meshes/obstacle_box2.stl"},
                {"obstacle2_pose_xyz_rpy": [0.00, 0.00, 0.0, 0.0, 0.0, 0.0]},
                {"obstacle2_scale_xyz": [0.001, 0.001, 0.001]},

                # 默认关节终点（也可从这里覆盖）
                {"q_goal": [0.0, -1.57, 1.57, 0.0, 1.57, 0.0]},
            ],
        )
    )
    # 3) 启动你的 demo，可共享同一份参数字典
    plan_demo = Node(
    package="iris_planner_demo",
    executable="plan_cpp_demo",
    output="screen",
    parameters=[moveit_config.to_dict(), {"planning_group": "arm"}],  # ← 这里设为 arm
)


    # 4) （可选）启动 RViz，加载 moveit.rviz
    # rviz = Node(
    #     package="rviz2",
    #     executable="rviz2",
    #     name="rviz2",
    #     output="screen",
    #     arguments=["-d", str(moveit_config.package_share_directory / "config/moveit.rviz")],
    #     parameters=[moveit_config.to_dict()],
    # )

    return LaunchDescription([move_group, plan_demo])  # 如需 RViz，把 rviz 加进来

