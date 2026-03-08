from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder

def generate_launch_description():
    mc = MoveItConfigsBuilder("ur5", package_name="ur5_moveit_config").to_moveit_configs()

    move_group = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            mc.to_dict(),
            mc.ompl_planning_yaml,
            {"planning_pipelines": ["ompl"], "default_planning_pipeline": "ompl"},
        ],
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        output="screen",
        arguments=["-d", mc.rviz_config],
        parameters=[mc.to_dict()],
    )

    demo = Node(
        package="iris_planner_demo",
        executable="plan_with_obstacles.py",
        output="screen",
        parameters=[
            {"planning_group": "manipulator"},
            {"ompl_planner_id": "RRTConnectkConfigDefault"},
            {"planning_time": 5.0},
            {"num_planning_attempts": 1},

            {"start_joints": [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]},
            {"goal_joints":  [1.2,  -1.2,  1.0,  -1.0,   1.2,  0.0]},

            # 示例：毫米 STL 用 0.001 缩放；改成你的绝对路径后去掉注释
             {"background.stl": "/home/hyj/iris_rviz_ws/src/welding_torch_description/meshes/collision/a_mesh.stl"},
             {"background.xyz": [0.0, 0.0, 0.0]},
             {"background.rpy": [0.0, 0.0, 0.0]},
             {"background.scale": [0.001, 0.001, 0.001]},
             {"obstacles.stls": ["/home/hyj/iris_rviz_ws/src/welding_torch_description/meshes/collision/b_mesh.stl"]},
             {"obstacles.poses.xyz": [[0.0, 0.00, 0.0]]},
             {"obstacles.poses.rpy": [[0.0, 0.0, 0.0]]},
             {"obstacles.scales": [[0.001, 0.001, 0.001]]},
        ],
    )

    return LaunchDescription([move_group, rviz, demo])
