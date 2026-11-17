# Formal CHiTO integration plan

> For agentic workers: execute sequentially with executing-plans; request code review before publication.

**Goal:** Publish the verified shared-node split method as the default public CHiTO method.

**Architecture:** Retain corridor generation and IK, actual-merit global QP, feasibility-first local QP, independent incoming/outgoing copies constrained to original cells, and validated stitching. The unified API owns the default method and six-link geometry; explicit legacy mode preserves research comparisons.

**Tech Stack:** C++17, ROS 2 Humble, MoveIt, Drake, OSQP 1.x, FCL, yaml-cpp.

## 1. Integrate tested source
- [x] Freeze upstream 89334e0ddf0a373fce34678238e24bbea6035800 in a clean checkout, separate from dirty experiments.
- [x] Copy only changed algorithm headers, implementation slices, CMake and tests from experimental commit 4687c36. Exclude nested duplicate packages, generated corridor files, build trees and unrelated configs.
- [x] Build Release and run the imported 16-test baseline.

## 2. Public defaults and regression
Files: optimizer.hpp, optimizer_config.cpp, unified_optimizer.cpp, hierarchical_opt.cpp, tests/formal_method_test.cpp, tests/optimizer_config_test.cpp, CMakeLists.txt.
- [x] Add a public fixture regression requiring a five-node guide to produce eight copies, fixed true endpoints, and preserved corridor failures. Run it before changing behavior; expect failure because public refinement does not enable splitting.
- [x] Add formal_method=true, fixed six-link geometry, continuous refinement defaults and config round trip. Formal mode sets all four verified switches; legacy mode is explicit. Reject incompatible translation-envelope mode.
- [x] Preserve diagnostic baselines by explicitly selecting legacy mode in their fixtures/runner.
- [x] Run the new regression, config tests and full suite.

## 3. Documentation and reproducibility
Files: README.md, docs/formal_method.md, docs/complete_pipeline.md, docs/unified_optimizer.md, config/chito_scene{1,2,3}.yaml, benchmarks/split_corridor/.
- [x] Replace obsolete default-envelope instructions. Document pipeline, constraint scope, tool model, failure statuses, build, CLI/API, and legacy opt-out.
- [x] Include paired three-scene CSV, relative-path frozen inputs and replay script, provenance hashes and honest results: strict 13/15 unchanged, geometry 13/15 to 14/15. Explain duplicate nodes/runtime and longer paths.
- [x] Rebuild, run installed CLI on all three scenes, and replay paired inputs via public refinement.

## 4. Review and GitHub
- [x] Review source/default wiring and final validation with an independent reviewer; address material findings.
- [x] Run fresh full tests, inspect staged scope and artifact sizes, and commit.
- [ ] Upload to 173hyj/CHiTO using a reviewed integration branch/PR and merge if repository permits. Attach any created PR to this chat. Verify remote SHA/tree and published defaults.
