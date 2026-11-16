#pragma once

#include <Eigen/Dense>

namespace chito_planner::optimization {

struct QPResult {
  bool solved{false};
  Eigen::VectorXd xnew;
  double objective{0.0};
};

// General convex QP: min 0.5 x'Qx+c'x, subject to lower <= A*x <= upper.
QPResult solve_linear_qp(const Eigen::MatrixXd& Q,
                         const Eigen::VectorXd& c,
                         const Eigen::MatrixXd& A,
                         const Eigen::VectorXd& lower,
                         const Eigen::VectorXd& upper,
                         const Eigen::VectorXd& x0);

// Dense trust-region QP backend used by CHiTO's local convexified subproblems:
// min 0.5 x'Qx + c'x, subject to lb <= x <= ub.
QPResult solve_box_qp(const Eigen::MatrixXd& Q,
                      const Eigen::VectorXd& c,
                      const Eigen::VectorXd& lb,
                      const Eigen::VectorXd& ub,
                      const Eigen::VectorXd& x0);

}  // namespace chito_planner::optimization
