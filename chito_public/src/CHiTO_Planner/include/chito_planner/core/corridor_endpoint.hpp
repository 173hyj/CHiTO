#pragma once
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
namespace chito_planner::core {
inline bool normalized_corridor(const Eigen::MatrixXd& A,const Eigen::VectorXd& b,Eigen::MatrixXd& C,Eigen::VectorXd& h){
 if(A.rows()<1||A.cols()!=3||b.size()!=A.rows()||!A.allFinite()||!b.allFinite())return false;
 C=A;h=b;for(int i=0;i<A.rows();++i){double n=A.row(i).norm();if(n<1e-12)return false;C.row(i)/=n;h[i]/=n;}return true;
}
inline bool linearized_corridor(const Eigen::MatrixXd& A,const Eigen::VectorXd& b,const Eigen::Vector3d& p,const Eigen::Matrix<double,3,6>& J,const Eigen::Matrix<double,6,1>& q,Eigen::MatrixXd& C,Eigen::VectorXd& upper){
 Eigen::MatrixXd planes;Eigen::VectorXd h;
 if(!p.allFinite()||!J.allFinite()||!q.allFinite()||!normalized_corridor(A,b,planes,h))return false;
 C=planes*J;upper=h-planes*p+C*q;return C.allFinite()&&upper.allFinite();
}
inline double corridor_residual(const Eigen::MatrixXd& A,const Eigen::VectorXd& b,const Eigen::Vector3d& p){
 Eigen::MatrixXd C;Eigen::VectorXd h;if(!p.allFinite()||!normalized_corridor(A,b,C,h))return std::numeric_limits<double>::infinity();return (C*p-h).maxCoeff();
}
inline int corridor_cell(const std::vector<int>& original,double u){
 if(original.empty()||!std::isfinite(u)||u<0.||u>double(original.size()-1))return -1;
 return original[std::min(int(original.size()-1),int(std::floor(u+.5)))];
}
inline bool insert_provenance(std::vector<double>& u,int edge,int count){
 if(edge<0||edge+1>=int(u.size())||count<1||!std::isfinite(u[edge])||!std::isfinite(u[edge+1])||u[edge]>u[edge+1])return false;
 std::vector<double> points;for(int j=1;j<=count;++j)points.push_back(u[edge]+double(j)/(count+1)*(u[edge+1]-u[edge]));u.insert(u.begin()+edge+1,points.begin(),points.end());return true;
}
struct MovingEndpointWindow{int left{-1},right{-1};bool valid{false};};
inline MovingEndpointWindow moving_endpoint_window(const std::vector<double>& d,int center){
 if(center<=0||center+1>=int(d.size()))return {};
 for(double v:d)if(!std::isfinite(v))return {};
 if(d.front()<0.||d.back()<0.)return {};
 int l=center-1,r=center+1;while(l>0&&d[l-1]<0.)--l;while(r+1<int(d.size())&&d[r+1]<0.)++r;
 return {l,r,true};
}
}
