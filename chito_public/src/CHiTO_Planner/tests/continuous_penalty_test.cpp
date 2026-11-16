#include "chito_planner/core/continuous_penalty.hpp"
#include <cassert>
#include <cmath>
int main(){Eigen::VectorXd x(2),g(2); x<<1,2;g<<2,-1;
Eigen::MatrixXd Q=Eigen::MatrixXd::Zero(2,2);Eigen::VectorXd c=Eigen::VectorXd::Zero(2);
chito_planner::core::add_linearized_squared_hinge(0.3,g,x,4.,Q,c);
assert((Q-4*g*g.transpose()).norm()<1e-12);
assert((Q*x+c-4*.3*g).norm()<1e-12);
assert(Q.determinant()>-1e-12);
}
