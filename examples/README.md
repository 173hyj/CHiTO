# Three scene examples / 三个场景示例

Each scene has one complete example using the formal CHiTO method and the six-link arm without a torch or gripper.

每个场景仅保留一个示例，使用正式 CHiTO 方法与无焊枪、无夹爪的六连杆模型。

| Scene | Files | Trajectory playback |
|---|---|---|
| scene1 | [scene1/](scene1/README.md) | [playback.html](scene1/playback.html) |
| scene2 | [scene2/](scene2/README.md) | [playback.html](scene2/playback.html) |
| scene3 | [scene3/](scene3/README.md) | [playback.html](scene3/playback.html) |

Download the repository and open each `playback.html` locally. GitHub displays the HTML source rather than executing it. The playback includes obstacles, the corridor/IK seed path and the optimized arm trajectory. It is a schematic view of link centerlines.

下载仓库后在本地浏览器打开 `playback.html`，可播放机械臂轨迹，并查看障碍物、初始路径与优化路径。

After building and sourcing the workspace, run these commands from the repository root:

```bash
ros2 run CHiTO_Planner chito_optimizer --config examples/scene1/effective_config.yaml --output results/scene1
ros2 run CHiTO_Planner chito_optimizer --config examples/scene2/effective_config.yaml --output results/scene2
ros2 run CHiTO_Planner chito_optimizer --config examples/scene3/effective_config.yaml --output results/scene3
```

Choose fresh output directories. Each example contains its effective configuration, `seed.csv`, `guide.yaml`, `trajectory.csv`, `result.json` and standalone playback. CSV joint angles use radians. The guide records original seed cell assignments; the trajectory retains the separately optimized incoming/outgoing copies.
