#include "chito_planner/core/convex_hull.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <utility>

namespace chito_planner::core {
namespace {

constexpr double kPointEps = 1e-6;
constexpr double kNormalEps = 1e-6;
constexpr double kOffsetEps = 1e-6;

struct PlaneKey {
  int nx;
  int ny;
  int nz;
  int nd;

  bool operator<(const PlaneKey& other) const {
    if (nx != other.nx) return nx < other.nx;
    if (ny != other.ny) return ny < other.ny;
    if (nz != other.nz) return nz < other.nz;
    return nd < other.nd;
  }
};

PlaneKey make_plane_key(const Eigen::Vector3d& normal_raw, double offset_raw) {
  Eigen::Vector3d normal = normal_raw;
  double norm = normal.norm();
  if (norm < 1e-12) {
    normal = Eigen::Vector3d::UnitZ();
    norm = 1.0;
  }
  normal /= norm;

  if (std::fabs(normal.x()) > kNormalEps) {
    if (normal.x() < 0.0) {
      normal = -normal;
      offset_raw = -offset_raw;
    }
  } else if (std::fabs(normal.y()) > kNormalEps) {
    if (normal.y() < 0.0) {
      normal = -normal;
      offset_raw = -offset_raw;
    }
  } else if (normal.z() < 0.0) {
    normal = -normal;
    offset_raw = -offset_raw;
  }

  auto quantize = [](double value, double eps) {
    return static_cast<int>(std::llround(value / eps));
  };
  return {quantize(normal.x(), kNormalEps),
          quantize(normal.y(), kNormalEps),
          quantize(normal.z(), kNormalEps),
          quantize(offset_raw, kOffsetEps)};
}

std::pair<std::vector<Eigen::Vector3d>, std::vector<int>> deduplicate_points(
    const std::vector<Eigen::Vector3d>& points) {
  std::vector<Eigen::Vector3d> unique;
  std::vector<int> remap(points.size(), -1);

  for (size_t i = 0; i < points.size(); ++i) {
    int id = -1;
    for (size_t j = 0; j < unique.size(); ++j) {
      if ((points[i] - unique[j]).squaredNorm() < kPointEps * kPointEps) {
        id = static_cast<int>(j);
        break;
      }
    }
    if (id < 0) {
      id = static_cast<int>(unique.size());
      unique.push_back(points[i]);
    }
    remap[i] = id;
  }

  return {unique, remap};
}

void plane_basis(const Eigen::Vector3d& normal,
                 Eigen::Vector3d& u,
                 Eigen::Vector3d& v) {
  const Eigen::Vector3d tangent =
      (std::fabs(normal.x()) > 0.8) ? Eigen::Vector3d::UnitY()
                                    : Eigen::Vector3d::UnitX();
  u = (tangent - normal * tangent.dot(normal)).normalized();
  v = normal.cross(u).normalized();
}

std::vector<int> convex_hull_2d(const std::vector<Eigen::Vector2d>& points) {
  const int n = static_cast<int>(points.size());
  if (n <= 2) {
    std::vector<int> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    return indices;
  }

  std::vector<std::pair<Eigen::Vector2d, int>> sorted;
  sorted.reserve(n);
  for (int i = 0; i < n; ++i) sorted.push_back({points[i], i});
  std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
    if (a.first.x() == b.first.x()) return a.first.y() < b.first.y();
    return a.first.x() < b.first.x();
  });

  auto cross = [](const Eigen::Vector2d& a,
                  const Eigen::Vector2d& b,
                  const Eigen::Vector2d& c) {
    return (b.x() - a.x()) * (c.y() - a.y()) -
           (b.y() - a.y()) * (c.x() - a.x());
  };

  std::vector<int> hull(2 * n);
  int k = 0;
  for (int i = 0; i < n; ++i) {
    while (k >= 2 &&
           cross(sorted[hull[k - 2]].first, sorted[hull[k - 1]].first,
                 sorted[i].first) <= 0.0) {
      --k;
    }
    hull[k++] = i;
  }
  for (int i = n - 2, t = k + 1; i >= 0; --i) {
    while (k >= t &&
           cross(sorted[hull[k - 2]].first, sorted[hull[k - 1]].first,
                 sorted[i].first) <= 0.0) {
      --k;
    }
    hull[k++] = i;
  }
  hull.resize(k - 1);

  std::vector<int> result;
  result.reserve(hull.size());
  for (int h : hull) result.push_back(sorted[h].second);
  return result;
}

}  // namespace

WatertightHull build_watertight_convex_hull(
    const std::vector<Eigen::Vector3d>& points_in) {
  WatertightHull hull;
  auto [points, remap] = deduplicate_points(points_in);
  (void)remap;

  const int n = static_cast<int>(points.size());
  hull.vertices = points;
  if (n < 4) return hull;

  Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
  for (const auto& p : points) centroid += p;
  centroid /= static_cast<double>(n);

  std::map<PlaneKey, std::vector<int>> plane_vertices;
  std::map<PlaneKey, Eigen::Vector4d> plane_equations;
  constexpr double kAreaEps = 1e-12;
  constexpr double kSideEps = 1e-9;

  for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < n; ++j) {
      for (int k = j + 1; k < n; ++k) {
        const Eigen::Vector3d p0 = points[i];
        const Eigen::Vector3d p1 = points[j];
        const Eigen::Vector3d p2 = points[k];
        Eigen::Vector3d normal = (p1 - p0).cross(p2 - p0);
        const double area = normal.norm();
        if (area < kAreaEps) continue;
        normal /= area;
        double offset = normal.dot(p0);

        double min_side = 1e18;
        double max_side = -1e18;
        for (int m = 0; m < n; ++m) {
          const double signed_distance = normal.dot(points[m]) - offset;
          min_side = std::min(min_side, signed_distance);
          max_side = std::max(max_side, signed_distance);
          if (min_side < -kSideEps && max_side > kSideEps) {
            min_side = -1.0;
            max_side = 1.0;
            break;
          }
        }
        if (!((max_side <= kSideEps) || (min_side >= -kSideEps))) continue;

        if (normal.dot(centroid) - offset > 0.0) {
          normal = -normal;
          offset = -offset;
        }

        const PlaneKey key = make_plane_key(normal, offset);
        auto& ids = plane_vertices[key];
        if (plane_equations.find(key) == plane_equations.end()) {
          plane_equations[key] =
              Eigen::Vector4d(normal.x(), normal.y(), normal.z(), offset);
        }

        auto push_unique = [&ids](int id) {
          if (std::find(ids.begin(), ids.end(), id) == ids.end()) {
            ids.push_back(id);
          }
        };
        push_unique(i);
        push_unique(j);
        push_unique(k);
        for (int t = 0; t < n; ++t) {
          const double signed_distance = normal.dot(points[t]) - offset;
          if (std::fabs(signed_distance) <= 5e-6) push_unique(t);
        }
      }
    }
  }

  for (const auto& [key, ids] : plane_vertices) {
    const auto& equation = plane_equations[key];
    const Eigen::Vector3d normal(equation[0], equation[1], equation[2]);
    const double offset = equation[3];
    if (ids.size() < 3) continue;

    Eigen::Vector3d u;
    Eigen::Vector3d v;
    plane_basis(normal, u, v);

    std::vector<Eigen::Vector2d> projected;
    projected.reserve(ids.size());
    for (int id : ids) {
      const auto& point = points[id];
      projected.emplace_back(point.dot(u), point.dot(v));
    }

    const auto local_hull = convex_hull_2d(projected);
    if (local_hull.size() < 3) continue;

    HullPolygon polygon;
    polygon.normal = normal;
    polygon.offset = offset;
    polygon.vertex_indices.reserve(local_hull.size());
    for (int h : local_hull) polygon.vertex_indices.push_back(ids[h]);
    hull.faces.push_back(std::move(polygon));
  }

  return hull;
}

}  // namespace chito_planner::core
