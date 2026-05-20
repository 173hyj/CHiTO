#pragma once

#include <fcl/fcl.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <memory>
#include <vector>

namespace chito_planner::core {

using JointVector6 = Eigen::Matrix<double, 6, 1>;

enum class ObType { BOX, SPHERE, CYLINDER };

struct Obstacle {
  ObType type{ObType::BOX};
  Eigen::Vector3d center{Eigen::Vector3d::Zero()};
  Eigen::Vector3d size{Eigen::Vector3d::Zero()};
  Eigen::Vector3d rpy_deg{Eigen::Vector3d::Zero()};
  std::shared_ptr<fcl::CollisionObjectd> obj;
};

struct LinkBox {
  std::shared_ptr<fcl::CollisionObjectd> obj;
  Eigen::Isometry3d T{Eigen::Isometry3d::Identity()};
  Eigen::Vector3d size{Eigen::Vector3d::Zero()};
};

struct ConvexSetGuideData {
  bool loaded{false};

  std::vector<JointVector6> q_seed_dense;
  std::vector<JointVector6> q_rep;
  std::vector<JointVector6> q_paths5;
  std::vector<int> sigma_dense;
  std::vector<Eigen::Vector3d> anchor_xyz;

  std::vector<Eigen::MatrixXd> poly_A;
  std::vector<Eigen::VectorXd> poly_b;
};

LinkBox make_link_box(const Eigen::Vector3d& p1,
                      const Eigen::Vector3d& p2,
                      double half_width,
                      double half_length);

LinkBox make_box_from_pose(const Eigen::Isometry3d& world_from_box,
                           double sx,
                           double sy,
                           double sz);

}  // namespace chito_planner::core
