#pragma once
#include <Eigen/Dense>
namespace chito_planner::core {
inline void add_linearized_squared_hinge(double gap, const Eigen::VectorXd& gradient,
 const Eigen::VectorXd& x, double weight, Eigen::MatrixXd& Q, Eigen::VectorXd& c) {
 if (!(gap>0.) || !gradient.allFinite()) return;
 Q.noalias() += weight*gradient*gradient.transpose();
 c.noalias() += weight*(gap-gradient.dot(x))*gradient;
}
}
