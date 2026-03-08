from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import Command, PathJoinSubstitution, TextSubstitution
import os

def _pick_first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

def launch_setup(context, *args, **kwargs):
    # 1) 先起官方 ur_moveit_config（会起 move_group / robot_state_publisher 等）
    ur_moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("ur_moveit_config"),
                "launch",
                "ur_moveit.launch.py",
            ])
        ),
        launch_arguments={"ur_type": "ur5"}.items(),
    )

    # 2) URDF Xacro 一般固定在这（你机器上已存在）
    urdf_xacro = os.path.join(
        FindPackageShare("ur_description").perform(context),
        "urdf", "ur.urdf.xacro"
    )

    # 3) 在多个可能的 SRDF 路径里找一个存在的
    #   某些版本是 srdf/ur.srdf；有的在 config/；也有 ur5 或 .srdf.xacro 变体
    urmc = FindPackageShare("ur_moveit_config").perform(context)
    srdf_candidates = [
        os.path.join(urmc, "srdf",  "ur.srdf"),
        os.path.join(urmc, "config","ur.srdf"),
        os.path.join(urmc, "srdf",  "ur5.srdf"),
        os.path.join(urmc, "config","ur5.srdf"),
        os.path.join(urmc, "srdf",  "ur.srdf.xacro"),
        os.path.join(urmc, "config","ur.srdf.xacro"),
        os.path.join(urmc, "srdf",  "ur5.srdf.xacro"),
        os.path.join(urmc, "config","ur5.srdf.xacro"),
    ]
    srdf_path = _pick_first_existing(srdf_candidates)

    # 4) 组装参数字典
    # 注意：xacro 的 urdf 只需要 `ur_type:=ur5`；不要传 name:=ur（会报 Undefined substitution argument name）
    robot_description_param = {
        "robot_description": Command(["xacro ", urdf_xacro, " ur_type:=ur5"])
    }
    # SRDF 如果找到了，传；找不到就不传（让 move_group 自己带 SRDF）
    params = [robot_description_param]
    if srdf_path is not None:
        # .srdf.xacro 用 xacro 处理；.srdf 直接 cat
        if srdf_path.endswith(".xacro"):
            robot_semantic_param = {
                "robot_description_semantic": Command(["xacro ", srdf_path])
            }
        else:
            robot_semantic_param = {
                "robot_description_semantic": Command(["cat ", srdf_path])
            }
        params.append(robot_semantic_param)
    else:
        # 打个提示（只在屏幕上打印，不影响运行）
        print(f"[bench_ur5] SRDF not found in known locations. "
              f"Will rely on ur_moveit_config's move_group to provide semantics.")

    # 5) 我们的基准节点（延时启动，给 move_group 几秒初始化）
    planners = [
        "RRTConnectkConfigDefault",
        "PRMkConfigDefault",
        "RRTstarkConfigDefault",
    ]
    bench_node = Node(
        package="iris_rviz_cpp",
        executable="ompl_multi_planners_bench",
        output="screen",
        parameters=params + [{
            "group_name": "ur_manipulator",
            "q_start_deg": [27.0, -90.0, 90.0, 0.0, 75.6, 0.0],
            "q_goal_deg":  [97.0, -46.8, 50.4, -3.6, 99.0, 0.0],
            "planners": planners,
            "planning_time": 3.0
        }],
    )
    delayed_bench = TimerAction(period=6.0, actions=[bench_node])  # 慢机建议 6s

    return [ur_moveit_launch, delayed_bench]

def generate_launch_description():
    return LaunchDescription([OpaqueFunction(function=launch_setup)])

