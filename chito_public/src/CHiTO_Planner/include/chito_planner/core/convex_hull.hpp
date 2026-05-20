#pragma once

#include <Eigen/Dense>

#include <vector>

namespace chito_planner::core {

struct HullPolygon {
  Eigen::Vector3d normal{Eigen::Vector3d::UnitZ()};
  double offset{0.0};
  std::vector<int> vertex_indices;
};

struct WatertightHull {
  std::vector<Eigen::Vector3d> vertices;
  std::vector<HullPolygon> faces;
};

// Builds a watertight polygonal hull from a small 3-D point set. CHiTO uses this
// for conservative continuous-time safety checks between adjacent robot states.
WatertightHull build_watertight_convex_hull(
    const std::vector<Eigen::Vector3d>& points);

}  // namespace chito_planner::core
