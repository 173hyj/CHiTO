#include "chito_planner/core/robot_geometry.hpp"

#include <cmath>

namespace chito_planner::core {

LinkBox make_link_box(const Eigen::Vector3d& p1,
                      const Eigen::Vector3d& p2,
                      double half_width,
                      double half_length) {
  Eigen::Vector3d z = p2 - p1;
  double length = z.norm();
  if (length < 1e-9) {
    z = Eigen::Vector3d::UnitZ();
    length = 1e-6;
  } else {
    z /= length;
  }

  Eigen::Vector3d x = (std::fabs(z.x()) > 0.9) ? Eigen::Vector3d::UnitY()
                                                : Eigen::Vector3d::UnitX();
  Eigen::Vector3d y = z.cross(x).normalized();
  x = y.cross(z).normalized();

  Eigen::Isometry3d world_from_box = Eigen::Isometry3d::Identity();
  world_from_box.linear().col(0) = x;
  world_from_box.linear().col(1) = y;
  world_from_box.linear().col(2) = z;
  world_from_box.translation() = 0.5 * (p1 + p2);

  const Eigen::Vector3d size(2.0 * half_width, 2.0 * half_length, length);
  auto geom = std::make_shared<fcl::Boxd>(size.x(), size.y(), size.z());
  auto obj = std::make_shared<fcl::CollisionObjectd>(geom);
  obj->setTransform(fcl::Transform3d(world_from_box.matrix()));
  obj->computeAABB();

  return {obj, world_from_box, size};
}

LinkBox make_box_from_pose(const Eigen::Isometry3d& world_from_box,
                           double sx,
                           double sy,
                           double sz) {
  auto geom = std::make_shared<fcl::Boxd>(sx, sy, sz);
  auto obj = std::make_shared<fcl::CollisionObjectd>(geom);
  obj->setTransform(fcl::Transform3d(world_from_box.matrix()));
  obj->computeAABB();

  return {obj, world_from_box, Eigen::Vector3d(sx, sy, sz)};
}

}  // namespace chito_planner::core
