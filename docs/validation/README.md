# Formal method integration verification

Release build and all 17 CTests passed on 2026-11-17. `ctest.log` is the fresh full log; `release.json` records source/log hashes and environment. The installed `ros2 run CHiTO_Planner chito_optimizer --scene sceneN` completed successfully for all three default presets, each with eight waypoints and original-scene validation (`installed-cli.json`). These three smoke cases are not a success-rate estimate.

All 30 frozen paired full optimizations reproduced the historical trajectories exactly (maximum joint error 0.0), including all strict success outcomes. See `benchmarks/split_corridor/integration-verification.json`. Integration source review found and fixed a supplied-seed mode compatibility bypass; a red regression confirmed the gap before the guard was added. Fresh full tests cover the fix.

A separate BUILD_TESTING=OFF configure confirms diagnostic runners are excluded. Legacy objective audit depending on private experiment paths was removed. Historical experiment data.zip is retained unchanged and is not the new benchmark; use benchmarks/split_corridor for the formal-method comparison.
