# Scene 2

Open [playback.html](playback.html) locally to view the arm trajectory and seed path.

From the repository root, after building and sourcing `chito_public/install/setup.bash`:

```bash
ros2 run CHiTO_Planner chito_optimizer --config examples/scene2/effective_config.yaml --output results/scene2
```

Use a fresh output directory. The example configuration resolves its scene file relative to this folder.

- [effective_config.yaml](effective_config.yaml): full configuration and fixed random seed.
- [seed.csv](seed.csv): corridor/IK seed, with joint angles in radians.
- [guide.yaml](guide.yaml): convex corridor and original cell assignments.
- [trajectory.csv](trajectory.csv): optimized joint trajectory in radians.
- [result.json](result.json): output status and validation fields.
- [playback.html](playback.html): self-contained schematic playback; no server is required.
