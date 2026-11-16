# CHiTO 完整流程

当前预设固定为：走廊搜索 → IK → 原全局QP → 原局部插点/中点QP修补 → 两端盒子包络联合凸化 → 近似包络验收 → 导出。

各阶段在同一个C++调用内传递数据。用户不用手动传递走廊或IK文件；保存的中间文件用于复查。

## V100环境运行

在现有 `codex-chito-humble` 容器内：

```bash
source /opt/ros/humble/setup.bash
source /work/repo/chito_public/install/setup.bash
ros2 run CHiTO_Planner chito_optimizer --scene scene2 --output /work/chito-results/demo
```

切换 `scene1`、`scene2`、`scene3` 即可换场景。批量添加 `--trials 50`，随机种子依次递增；`--seed 20262115` 可指定起始种子。每次选用新输出目录。

自定义场景/参数使用 `--config your.yaml` 代替 `--scene`。障碍物为支持盒子/球体的`.scene`格式；YAML用于优化器配置，不是障碍物几何格式。

## 从源码构建

解压后进入 `chito_public`，在已安装ROS Humble/MoveIt、FCL、Eigen、yaml-cpp、Drake和OSQP1.x的环境中运行：

```bash
colcon build --packages-select CHiTO_Planner --cmake-args -DDRAKE_PREFIX=/opt/drake -DBUILD_TESTING=ON
source install/setup.bash
ros2 run CHiTO_Planner chito_optimizer --scene scene2 --output results/demo
```

## 每次输出

```text
output/
  summary.json
  run_0/
    result.json
    effective_config.yaml
    seed.csv
    guide.yaml
    trajectory.csv
    playback.html
  run_1/ ...
```

早期生成失败时，尚未产生的初值/轨迹文件可以缺失；以result.json的阶段状态为准。playback.html是连杆示意回放，不是碰撞体网格。

result.json记录成功/失败、失败原因、时长、模型标签以及离散优化前后点间距诊断。summary.json统计批量成功、规划失败及导出失败。进程退出码：0全部成功；1存在规划失败；2配置/参数错误；3导出失败。

## 固定的模型与保护

- 原离散优化结果作为新增凸化起点。
- 新更新不能破坏已有离散可行性；未离散可行时最小点间距不能恶化。
- 两端盒子各保留实际方向和尺寸，共同决定凸包络。
- QP中只线性化两端中心平移；每轮重建包络和分离平面。
- 全局QP含两端约束、固定起终点、关节/信赖域边界及非负松弛变量。
- 提供的预设启用 `translation_convexification: true`、`accept_approximate_hull: true`，最终按FCL近似端点凸包验收。
- 最终全局平滑关闭，之前的中点扫掠软惩罚实验关闭。
- 这是近似包络判据，不是对中间旋转或硬件运动的安全证明。

流程完整不代表所有任务都能规划成功。当前连续包络求解能力仍有限，失败会保留诊断，不会通过降低判据伪装成成功。
