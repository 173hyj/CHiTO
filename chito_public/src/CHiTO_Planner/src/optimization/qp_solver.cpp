#include "chito_planner/optimization/qp_solver.hpp"

#include <gurobi_c++.h>
#include <rclcpp/rclcpp.hpp>

#include <cmath>
#include <vector>

namespace chito_planner::optimization {

QPResult solve_box_qp(const Eigen::MatrixXd& Q,
                      const Eigen::VectorXd& c,
                      const Eigen::VectorXd& lb,
                      const Eigen::VectorXd& ub,
                      const Eigen::VectorXd& x0) {
  const int n = x0.size();
  QPResult result;

  try {
    GRBEnv env = GRBEnv(true);
    env.set("LogToConsole", "0");
    env.start();
    GRBModel model(env);

    std::vector<GRBVar> xvars;
    xvars.reserve(n);
    for (int i = 0; i < n; ++i) {
      auto var = model.addVar(lb(i), ub(i), 0.0, GRB_CONTINUOUS);
      var.set(GRB_DoubleAttr_Start, x0(i));
      xvars.push_back(var);
    }
    model.update();

    GRBQuadExpr quad = 0.0;
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        const double qij = Q(i, j);
        if (std::fabs(qij) > 1e-16) {
          quad += 0.5 * qij * xvars[i] * xvars[j];
        }
      }
    }

    GRBLinExpr linear = 0.0;
    for (int i = 0; i < n; ++i) {
      if (std::fabs(c(i)) > 1e-16) linear += c(i) * xvars[i];
    }

    model.setObjective(quad + linear, GRB_MINIMIZE);
    model.optimize();

    if (model.get(GRB_IntAttr_Status) == GRB_OPTIMAL) {
      result.solved = true;
      result.xnew.resize(n);
      for (int i = 0; i < n; ++i) {
        result.xnew(i) = xvars[i].get(GRB_DoubleAttr_X);
      }
      result.objective = model.get(GRB_DoubleAttr_ObjVal);
    } else {
      RCLCPP_WARN(rclcpp::get_logger("chito_qp_solver"),
                  "Gurobi did not return an optimal solution. status=%d",
                  model.get(GRB_IntAttr_Status));
    }
  } catch (GRBException& e) {
    RCLCPP_ERROR(rclcpp::get_logger("chito_qp_solver"),
                 "Gurobi error %d: %s",
                 e.getErrorCode(),
                 e.getMessage().c_str());
  } catch (const std::exception& e) {
    RCLCPP_ERROR(rclcpp::get_logger("chito_qp_solver"),
                 "Exception: %s",
                 e.what());
  }

  return result;
}

}  // namespace chito_planner::optimization
