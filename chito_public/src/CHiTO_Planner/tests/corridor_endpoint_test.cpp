#include "chito_planner/core/corridor_endpoint.hpp"
#include "chito_planner/optimization/qp_solver.hpp"
#include <iostream>
using namespace chito_planner::core;
int main(){int errors=0;auto check=[&](bool ok,const char* s){if(!ok){++errors;std::cerr<<"FAIL "<<s<<'\n';}};
 Eigen::MatrixXd A(2,3);A<<2,0,0,-1,0,0;Eigen::VectorXd b(2);b<<2,0;
 Eigen::Vector3d p(.9,0,0);Eigen::Matrix<double,3,6> J=Eigen::Matrix<double,3,6>::Zero();J(0,0)=1;
 Eigen::Matrix<double,6,1> q=Eigen::Matrix<double,6,1>::Zero();q[0]=.4;
 Eigen::MatrixXd C;Eigen::VectorXd upper;
 check(linearized_corridor(A,b,p,J,q,C,upper),"valid normalized halfspaces");
 check(C.rows()==2&&C.cols()==6&&upper.size()==2,"constraint dimensions");
 if(upper.size()==2){check(std::abs(C(0,0)-1)<1e-12&&std::abs(upper[0]-.5)<1e-12,"affine upper includes current q");check(std::abs(C(1,0)+1)<1e-12&&std::abs(upper[1]-.5)<1e-12,"affine lower includes current q");}
 check(corridor_residual(A,b,p)<0,"exact inside accepted");
 check(corridor_residual(A,b,Eigen::Vector3d(1.01,0,0))>.009,"exact outside rejected after linear solve");
 check(corridor_residual(A,b,Eigen::Vector3d(1,0,0))==0,"boundary allowed");
 auto bad=A;bad.row(0).setZero();check(!linearized_corridor(bad,b,p,J,q,C,upper),"degenerate plane rejected");
 check(!linearized_corridor(A,b,Eigen::Vector3d(NAN,0,0),J,q,C,upper),"nonfinite state rejected");
 check(corridor_cell({0,2,4},.49)==0&&corridor_cell({0,2,4},.5)==2&&corridor_cell({0,2,4},1.8)==4,"stable nearest original association tie right");
 check(corridor_cell({},.5)==-1&&corridor_cell({0},NAN)==-1,"invalid provenance rejected");
 std::vector<double> u{0,1,2};check(insert_provenance(u,0,2)&&u.size()==5&&std::abs(u[2]-2./3)<1e-12,"insert provenance interpolation");
 check(insert_provenance(u,1,1)&&std::abs(u[2]-.5)<1e-12&&corridor_cell({0,2,4},u[2])==2,"repeated insertion keeps original coordinate");
 auto w=moving_endpoint_window({.1,.1,-.1,.1,.1},2);check(w.valid&&w.left==1&&w.right==3,"three node window has movable endpoints");
 w=moving_endpoint_window({.1,-.1,.1,.1,-.1,.1},2);check(w.valid&&w.left==1&&w.right==4,"colliding halo released rather than fixed");
 check(!moving_endpoint_window({-.1,.1,.1},1).valid,"colliding global endpoint rejected");
 if(linearized_corridor(A,b,p,J,q,C,upper)){
  Eigen::MatrixXd H=Eigen::MatrixXd::Identity(6,6),K(8,6);K.topRows(6).setIdentity();K.bottomRows(2)=C;
  Eigen::VectorXd c=Eigen::VectorXd::Zero(6),lo=Eigen::VectorXd::Constant(8,-INFINITY),hi=Eigen::VectorXd::Constant(8,10.);c[0]=-2;hi.tail(2)=upper;
  auto sol=chito_planner::optimization::solve_linear_qp(H,c,K,lo,hi,q);check(sol.solved&&std::abs(sol.xnew[0]-.5)<1e-5,"endpoint can move to corridor boundary");
  lo[0]=.6;check(!chito_planner::optimization::solve_linear_qp(H,c,K,lo,hi,q).solved,"infeasible trust and cell constraints rejected");
 }
 std::cout<<"Corridor endpoint test failures="<<errors<<'\n';return errors?1:0;
}
