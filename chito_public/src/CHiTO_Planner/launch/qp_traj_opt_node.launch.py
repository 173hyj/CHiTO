import os

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    qp_node = Node(
        package='iris_rviz_cpp',
        executable='qp_traj_opt_node',
        name='qp_traj_opt',
        output='screen',
        parameters=[{
            # 初始路径文件（CSV）
            'init_q_file': '/home/hyj/iris_rviz_ws/src/iris_rviz_cpp/src/corridors_out1/ik_results/corridor_0_20251112_170109_ik.csv',
            'q_file_is_deg_default': True,

            # 障碍物 .scene（旧格式）
            'convex_scene_yaml': '/home/hyj/iris_rviz_ws/src/collision/box2.scene',

            # QP / 碰撞相关参数
            'd_safe': 0.15,
            'cont_min_d_safe': 0.0,
            'mu': 0.1,
            'alpha': 0.5,
            'trust_s': 0.1,
            'max_iters': 3,
            'max_trust_attempts': 3,

            # 提前停止条件
            'stop_when_min_d_ge': True,
            'stop_min_d': 0.09,
            'min_d_ignore_warmup': False,
            'warmup_safe_iters': 2,

            # 迭代周期
            'iter_period_ms': 0,

            # 关节 6 不再强制锁死
            'fix_last_joint_to_zero': False,

            # 调试输出
            'debug_print_q': True,
            'qos_transient_local': True,
        }]
    )

    return LaunchDescription([qp_node])
