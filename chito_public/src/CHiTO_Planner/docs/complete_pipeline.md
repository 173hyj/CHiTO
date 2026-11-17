# CHiTO 正式流程

当前正式方法为：**走廊搜索 → IK → 实际代价验收的全局 QP → 共享节点左右拆分 → 各自原走廊内的局部 QP → 连接修补 → 原场景复验**。

起终点固定。每个内部共享节点拆成左右两份，分别参与两侧边优化，然后保留两份并检查连接。原节点副本的 TCP 必须留在各自原凸走廊内；新插入的修补点不强制 TCP 在走廊内。机械臂所有连杆仍需独立检查障碍物碰撞。

中点无碰撞不意味着修补完成：还要分别检查中点到两端的连接。局部窗口两侧的固定邻接点若碰撞，就扩展窗口；真正的 start/goal 碰撞则失败。候选需要同时通过受影响节点、内部边、外侧两条连接边和副本走廊检查。

## 构建和运行

在装好 ROS 2 Humble/MoveIt、Drake、OSQP 1.x、FCL、Eigen、yaml-cpp 的环境中：

```bash
source /opt/ros/humble/setup.bash
cd chito_public
colcon build --packages-select CHiTO_Planner --cmake-args \
  -DCMAKE_BUILD_TYPE=Release -DDRAKE_PREFIX=/opt/drake -DBUILD_TESTING=ON
colcon test --packages-select CHiTO_Planner --event-handlers console_direct+
colcon test-result --verbose
source install/setup.bash
ros2 run CHiTO_Planner chito_optimizer --scene scene1 --output results/scene1
```

可换成 `scene2`、`scene3`；`--trials 5` 连续运行五个随机种子，`--seed` 覆盖起始种子。每次使用新的输出目录。自定义配置用 `--config FILE`。正式 API 与三个预设默认启用 `formal_method: true`，无需实验环境变量。

默认是**无焊枪、无夹爪的六连杆模型**。局部边目标裕度 5 mm；单点惩罚激活距离 50 mm；最终单点阈值 0 mm。这三个量用途不同。全局迭代上限 20，附加修补轮数 20，路径扩展上限 256，每次插入 1 点，允许重复修补已有失败节点，最终平滑默认关闭。

`hierarchical_opt` 原始 ROS 节点、旧 launch 文件和末端平移包络研究模式保留用于兼容；正式入口是 `chito_optimizer` 和 `CHiTOOptimizer::solve`。旧变体需要明确设置 `formal_method: false`。正式模式不允许混入另一套包络验收而绕过拆点流程。

## 输出与判定

每次运行导出 JSON 状态、有效配置、关节轨迹 CSV 和 HTML 连杆回放。启用中间导出后还有原始走廊/IK 节点和 guide。原始副本越出指定走廊时，即使几何无碰撞也保留 `corridor_constraint_failed`，原场景复验不会将它改判成功。

[三个场景示例](../../../../examples/README.md)各保留一组完整文件：可直接运行的配置、走廊与 IK 初始节点、优化后的关节轨迹、状态 JSON 和 HTML 连杆回放。

当前是无条件拆共享节点，尚未实现仅在困难边触发。有限角度细分与端点凸包检查不等价于精确连续碰撞检测。
