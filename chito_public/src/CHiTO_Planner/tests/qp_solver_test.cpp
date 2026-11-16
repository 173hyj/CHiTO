#include "chito_planner/optimization/qp_solver.hpp"
#include <cassert>
#include <cmath>
int main(){using namespace chito_planner::optimization;
 Eigen::MatrixXd Q=2*Eigen::MatrixXd::Identity(2,2); Eigen::VectorXd c(2),l(2),u(2),x(2);c<<-4,2;l<<0,0;u<<1,3;x.setZero();
 auto r=solve_box_qp(Q,c,l,u,x);assert(r.solved);assert((r.xnew-Eigen::Vector2d(1,0)).norm()<1e-5);assert(std::abs(r.objective+3)<1e-5);
 l(0)=2;assert(!solve_box_qp(Q,c,l,u,x).solved);l(0)=0;Q(0,0)=-2;assert(!solve_box_qp(Q,c,l,u,x).solved);
 Q=2*Eigen::MatrixXd::Identity(2,2);l.setConstant(-10);u.setConstant(10);r=solve_box_qp(Q,c,l,u,x);assert(r.solved);assert((r.xnew-Eigen::Vector2d(2,-1)).norm()<1e-5);
 // Two adjacent endpoint translations and two nonnegative collision slacks.
 // Each endpoint must reach 1, but trust bound fixes endpoint 0 at <=0.25.
 Eigen::MatrixXd H=Eigen::MatrixXd::Identity(4,4);H(2,2)=H(3,3)=100;
 Eigen::VectorXd cost=Eigen::VectorXd::Zero(4),lo(6),hi(6),initial=Eigen::VectorXd::Zero(4);
 Eigen::MatrixXd A=Eigen::MatrixXd::Zero(6,4);A.topRows(4).setIdentity();
 A(4,0)=1;A(4,2)=1;A(5,1)=1;A(5,3)=1;
 lo<<-0.25,-2,0,0,1,1;hi<<0.25,2,10,10,INFINITY,INFINITY;
 auto joint=solve_linear_qp(H,cost,A,lo,hi,initial);assert(joint.solved);
 assert(std::abs(joint.xnew[0]-0.25)<1e-5);
 assert(std::abs(joint.xnew[2]-0.75)<1e-5);
 assert(std::abs(joint.xnew[1]-100.0/101)<1e-5);
 assert(std::abs(joint.xnew[3]-1.0/101)<1e-5);
 assert((A*joint.xnew-lo).minCoeff()>-1e-6);
 A(0,0)=NAN;assert(!solve_linear_qp(H,cost,A,lo,hi,initial).solved);
}
