#include "chito_planner/optimization/qp_solver.hpp"
#include <osqp.h>
#include <Eigen/Eigenvalues>
#include <vector>
#include <cmath>
namespace chito_planner::optimization {
QPResult solve_linear_qp(const Eigen::MatrixXd& Q,const Eigen::VectorXd& c,const Eigen::MatrixXd& constraints,const Eigen::VectorXd& lb,const Eigen::VectorXd& ub,const Eigen::VectorXd& x0){
 QPResult r; const int n=x0.size(),m=constraints.rows();
 if(n==0||Q.rows()!=n||Q.cols()!=n||c.size()!=n||constraints.cols()!=n||lb.size()!=m||ub.size()!=m||!constraints.allFinite()||!Q.allFinite()||!c.allFinite()||!x0.allFinite()||(lb.array()>ub.array()).any()||lb.hasNaN()||ub.hasNaN())return r;
 Eigen::MatrixXd H=(Q+Q.transpose())*.5;
 Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(H,Eigen::EigenvaluesOnly);
 if(eig.info()!=Eigen::Success||eig.eigenvalues().minCoeff() < -1e-9)return r;
 std::vector<OSQPInt> pp(n+1),pi,ap(n+1),ai;std::vector<OSQPFloat> pv,av,cv(n),lv(m),uv(m),warm(n);
 for(int j=0;j<n;++j){pp[j]=pv.size();for(int i=0;i<=j;++i)if(H(i,j)!=0){pi.push_back(i);pv.push_back(H(i,j));}ap[j]=av.size();for(int i=0;i<m;++i)if(constraints(i,j)!=0){ai.push_back(i);av.push_back(constraints(i,j));}cv[j]=c[j];warm[j]=x0[j];}pp[n]=pv.size();ap[n]=av.size();for(int i=0;i<m;++i){lv[i]=std::max(-double(OSQP_INFTY),std::min(double(OSQP_INFTY),lb[i]));uv[i]=std::max(-double(OSQP_INFTY),std::min(double(OSQP_INFTY),ub[i]));}
 OSQPCscMatrix P,A;OSQPCscMatrix_set_data(&P,n,n,pv.size(),pv.data(),pi.data(),pp.data());OSQPCscMatrix_set_data(&A,m,n,av.size(),av.data(),ai.data(),ap.data());
 OSQPSettings settings;osqp_set_default_settings(&settings);settings.verbose=0;settings.eps_abs=1e-7;settings.eps_rel=1e-7;settings.max_iter=20000;settings.polishing=1;
 OSQPSolver* solver=nullptr;auto code=osqp_setup(&solver,&P,cv.data(),&A,lv.data(),uv.data(),m,n,&settings);
 if(code!=0){if(solver)osqp_cleanup(solver);return r;}osqp_warm_start(solver,warm.data(),nullptr);code=osqp_solve(solver);
 if(code==0&&solver->info->status_val==OSQP_SOLVED){r.xnew.resize(n);for(int i=0;i<n;++i)r.xnew[i]=solver->solution->x[i];r.solved=r.xnew.allFinite()&&((constraints*r.xnew).array()>=lb.array()-1e-6).all()&&((constraints*r.xnew).array()<=ub.array()+1e-6).all();r.objective=.5*r.xnew.dot(H*r.xnew)+c.dot(r.xnew);}osqp_cleanup(solver);return r;
}
QPResult solve_box_qp(const Eigen::MatrixXd& Q,const Eigen::VectorXd& c,const Eigen::VectorXd& lb,const Eigen::VectorXd& ub,const Eigen::VectorXd& x0){
 return solve_linear_qp(Q,c,Eigen::MatrixXd::Identity(x0.size(),x0.size()),lb,ub,x0);
}
}
