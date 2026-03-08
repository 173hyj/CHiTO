// corridor_spacerrt_ik_viz.cpp
// 一体化流程：IRIS 双树(spaceRRT) -> 安全走廊 -> 采样 -> DLS 位置IK -> RViz 可视化

#include <rclcpp/rclcpp.hpp>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <random>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <array>
#include <queue>
#include <optional>
#include <memory>
#include <filesystem>
#include <cctype>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "geometry_msgs/msg/point.hpp"

using visualization_msgs::msg::Marker;
using visualization_msgs::msg::MarkerArray;
using geometry_msgs::msg::Point;

// ---------- Drake ----------
#include "drake/geometry/optimization/convex_set.h"
#include "drake/geometry/optimization/hpolyhedron.h"
#include "drake/geometry/optimization/vpolytope.h"
#include "drake/geometry/optimization/iris.h"
#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/solve.h"

using drake::geometry::optimization::HPolyhedron;
using drake::geometry::optimization::VPolytope;
using drake::geometry::optimization::ConvexSet;
using drake::geometry::optimization::ConvexSets;
using drake::geometry::optimization::Iris;
namespace mp = drake::solvers;

// ============================ 通用小工具 ============================
static std::string trim(const std::string& s) {
  size_t a = s.find_first_not_of(" \t\r\n");
  size_t b = s.find_last_not_of(" \t\r\n");
  if (a == std::string::npos) return std::string();
  return s.substr(a, b - a + 1);
}
static std::string to_upper_copy(std::string s) {
  for (auto& c : s) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
  return s;
}

// ============================ RViz Marker ============================
static Marker MakeLineMarker(const std::string& frame,
                             const std::string& ns, int id,
                             double width, double r, double g, double b, double a=1.0) {
  Marker m;
  m.header.frame_id = frame;
  m.ns = ns; m.id = id;
  m.type = Marker::LINE_LIST;
  m.action = Marker::ADD;
  m.scale.x = width;
  m.color.r = r; m.color.g = g; m.color.b = b; m.color.a = a;
  m.pose.orientation.w = 1.0;
  return m;
}
static void AddEdge(Marker* m, const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
  Point pa; pa.x = a.x(); pa.y = a.y(); pa.z = a.z();
  Point pb; pb.x = b.x(); pb.y = b.y(); pb.z = b.z();
  m->points.push_back(pa); m->points.push_back(pb);
}
static Marker MakeSphereMarker(const std::string& frame, const std::string& ns, int id,
                               const Eigen::Vector3d& c, double radius,
                               double r, double g, double b, double a) {
  Marker m;
  m.header.frame_id = frame;
  m.ns = ns; m.id = id;
  m.type = Marker::SPHERE;
  m.action = Marker::ADD;
  m.pose.position.x = c.x();
  m.pose.position.y = c.y();
  m.pose.position.z = c.z();
  m.pose.orientation.w = 1.0;
  m.scale.x = m.scale.y = m.scale.z = 2.0 * radius;
  m.color.r = r; m.color.g = g; m.color.b = b; m.color.a = a;
  return m;
}

// ============================ 几何与多面体工具 ============================
static std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>
VerticesFromH3D(const HPolyhedron& H, double feas_tol=1e-9, double dup_tol=1e-8) {
  const auto& A = H.A(); const auto& b = H.b();
  const int m = A.rows();
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> V;
  for (int i=0;i<m;++i) for (int j=i+1;j<m;++j) for (int k=j+1;k<m;++k) {
    Eigen::Matrix3d M; M.row(0)=A.row(i); M.row(1)=A.row(j); M.row(2)=A.row(k);
    double det = M.determinant(); if (std::abs(det) < 1e-12) continue;
    Eigen::Vector3d rhs(b(i), b(j), b(k));
    Eigen::Vector3d x = M.fullPivLu().solve(rhs);
    if ((A * x - b).maxCoeff() <= feas_tol) {
      bool dup=false; for (const auto& q : V) if ((x-q).norm() <= dup_tol) {dup=true; break;}
      if (!dup) V.push_back(x);
    }
  }
  return V;
}
static std::vector<std::pair<int,int>>
EdgesFromActiveSets(const HPolyhedron& H,
                    const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>& V,
                    double tol=1e-7) {
  const auto& A = H.A(); const auto& b = H.b();
  const int m = A.rows(), n = (int)V.size();
  std::vector<std::vector<int>> act(n);
  for (int i=0;i<n;++i)
    for (int r=0;r<m;++r)
      if (std::abs(A.row(r).dot(V[i]) - b(r)) <= tol) act[i].push_back(r);
  auto share_two = [&](int i, int j)->bool{
    int cnt=0; for (int r1:act[i]) for (int r2:act[j]) if (r1==r2) if(++cnt>=2) return true;
    return false;
  };
  std::vector<std::pair<int,int>> E;
  for (int i=0;i<n;++i) for (int j=i+1;j<n;++j)
    if ((V[i]-V[j]).norm()>1e-9 && share_two(i,j)) E.emplace_back(i,j);
  return E;
}
static inline bool PointInH(const HPolyhedron& H, const Eigen::Vector3d& x, double tol=1e-9){
  return (H.A() * x - H.b()).maxCoeff() <= tol;
}

// 轴对齐包围盒线框（调试/展示障碍）
static void DrawHullAsAabb(const Eigen::Matrix3Xd& X, Marker* m) {
  Eigen::Vector3d lo = X.rowwise().minCoeff();
  Eigen::Vector3d hi = X.rowwise().maxCoeff();
  Eigen::Vector3d p000(lo.x(), lo.y(), lo.z());
  Eigen::Vector3d p100(hi.x(), lo.y(), lo.z());
  Eigen::Vector3d p110(hi.x(), hi.y(), lo.z());
  Eigen::Vector3d p010(lo.x(), hi.y(), lo.z());
  Eigen::Vector3d p001(lo.x(), lo.y(), hi.z());
  Eigen::Vector3d p101(hi.x(), lo.y(), hi.z());
  Eigen::Vector3d p111(hi.x(), hi.y(), hi.z());
  Eigen::Vector3d p011(lo.x(), hi.y(), hi.z());
  AddEdge(m,p000,p100); AddEdge(m,p100,p110); AddEdge(m,p110,p010); AddEdge(m,p010,p000);
  AddEdge(m,p001,p101); AddEdge(m,p101,p111); AddEdge(m,p111,p011); AddEdge(m,p011,p001);
  AddEdge(m,p000,p001); AddEdge(m,p100,p101); AddEdge(m,p110,p111); AddEdge(m,p010,p011);
}

// 球体外包近似（Fibonacci 方向）-> shared_ptr<HPolyhedron>
// 之前：std::shared_ptr<HPolyhedron>
static std::unique_ptr<HPolyhedron>
MakeSpherePolyApprox(const Eigen::Vector3d& c, double r, int num_dirs = 64) {
  Eigen::MatrixXd A(num_dirs, 3);
  Eigen::VectorXd b(num_dirs);
  const double phi = (1.0 + std::sqrt(5.0)) / 2.0;
  for (int i = 0; i < num_dirs; ++i) {
    double t = (i + 0.5) / num_dirs;
    double z = 1.0 - 2.0 * t;
    double ang = 2.0 * M_PI * (i / phi - std::floor(i / phi));
    double rxy = std::sqrt(std::max(0.0, 1.0 - z * z));
    Eigen::Vector3d n(rxy * std::cos(ang), rxy * std::sin(ang), z);
    A.row(i) = n.transpose();
    b(i) = n.dot(c) + r;
  }
  return std::make_unique<HPolyhedron>(A, b);
}


// 在 Ax<=b 可行域内，按多方向做支持函数 LP 取极点
static std::vector<Eigen::Vector3d> ExtremePointsByLP(const Eigen::MatrixXd& A,
                                                      const Eigen::VectorXd& b,
                                                      int K = 32,
                                                      double feas_tol = 1e-9) {
  std::vector<Eigen::Vector3d> pts; pts.reserve(2*K);
  const double phi = (1.0 + std::sqrt(5.0)) / 2.0;
  auto dir_at = [&](int i)->Eigen::Vector3d{
    double t = (i + 0.5) / double(K);
    double z = 1.0 - 2.0 * t;
    double ang = 2.0 * M_PI * (i / phi - std::floor(i / phi));
    double rxy = std::sqrt(std::max(0.0, 1.0 - z*z));
    return Eigen::Vector3d(rxy*std::cos(ang), rxy*std::sin(ang), z);
  };
  auto solve_dir = [&](const Eigen::Vector3d& d)->std::optional<Eigen::Vector3d>{
    mp::MathematicalProgram prog;
    auto x = prog.NewContinuousVariables<3>("x");
    prog.AddLinearConstraint(A * x <= b + Eigen::VectorXd::Constant(b.size(), feas_tol));
    prog.AddLinearCost(-d, 0.0, x); // maximize d^T x
    auto res = mp::Solve(prog);
    if (!res.is_success()) return std::nullopt;
    Eigen::Vector3d sol; for (int i=0;i<3;++i) sol(i)=res.GetSolution(x(i));
    return sol;
  };
  auto push_unique = [&](const Eigen::Vector3d& p){
    for (auto& q : pts) if ((p-q).norm() <= 1e-6) return;
    pts.push_back(p);
  };
  for (int i=0;i<K;++i) {
    auto d = dir_at(i);
    if (auto p1 = solve_dir( d)) push_unique(*p1);
    if (auto p2 = solve_dir(-d)) push_unique(*p2);
  }
  return pts;
}

// 扁平交集 -> 加一对共面不等式，稳定顶点枚举
static bool AddCoplanarityIfFlat(Eigen::MatrixXd* A, Eigen::VectorXd* B,
                                 double ratio_thresh = 1e-2,
                                 double feas_tol = 1e-9) {
  auto P = ExtremePointsByLP(*A, *B, 32, feas_tol);
  if (P.size() < 3) return false;
  Eigen::Matrix3Xd M(3, P.size());
  for (size_t i=0;i<P.size();++i) M.col(i) = P[i];
  Eigen::Vector3d mu = M.rowwise().mean(); M.colwise() -= mu;
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(M, Eigen::ComputeThinU | Eigen::ComputeThinV);
  auto s = svd.singularValues(); if (s.size()<3 || s(0)<=1e-12) return false;
  if (s(2)/s(0) < ratio_thresh) {
    Eigen::Vector3d n = svd.matrixU().col(2);
    double c = n.dot(mu);
    Eigen::MatrixXd Anew(A->rows()+2, 3); Eigen::VectorXd Bnew(B->rows()+2);
    Anew << *A,  n.transpose(), (-n).transpose();
    Bnew << *B,  c,             -c;
    *A = std::move(Anew); *B = std::move(Bnew);
    return true;
  }
  return false;
}

// 交集可行性快速判断
static std::optional<HPolyhedron>
IntersectIfNonempty(const HPolyhedron& P, const HPolyhedron& Q, double feas_tol=1e-9) {
  int m1 = P.A().rows(), m2 = Q.A().rows();
  Eigen::MatrixXd A(m1+m2, 3); Eigen::VectorXd b(m1+m2);
  A << P.A(), Q.A(); b << P.b(), Q.b();
  mp::MathematicalProgram prog; auto x = prog.NewContinuousVariables<3>("x");
  prog.AddLinearConstraint(A * x <= b + Eigen::VectorXd::Constant(b.size(), feas_tol));
  prog.AddQuadraticCost(Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero(), x);
  auto res = mp::Solve(prog);
  if (!res.is_success()) return std::nullopt;
  return HPolyhedron(A,b);
}

// 门户：如果交集近似共面，补共面不等式，增强稳定性
static HPolyhedron PortalBetween(const HPolyhedron& P, const HPolyhedron& Q,
                                 double feas_tol = 1e-9, rclcpp::Logger logger = rclcpp::get_logger("portal")) {
  const int m1 = P.A().rows(), m2 = Q.A().rows();
  Eigen::MatrixXd A(m1 + m2, 3); Eigen::VectorXd b(m1 + m2);
  A << P.A(), Q.A(); b << P.b(), Q.b();

  { mp::MathematicalProgram prog; auto x = prog.NewContinuousVariables<3>("x");
    prog.AddLinearConstraint(A * x <= b + Eigen::VectorXd::Constant(b.size(), feas_tol));
    prog.AddQuadraticCost(Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero(), x);
    auto res = mp::Solve(prog);
    if (!res.is_success()) return HPolyhedron::MakeBox(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
  }

  Eigen::MatrixXd A_out = A; Eigen::VectorXd b_out = b;
  const bool added = AddCoplanarityIfFlat(&A_out, &b_out, 1e-2, feas_tol);
  if (added) RCLCPP_INFO(logger, "[PortalBetween] planar intersection -> add coplanarity rows (%d->%d)",
                         (int)A.rows(), (int)A_out.rows());
  return HPolyhedron(A_out, b_out);
}

// ============================ spaceRRT 双树结构 ============================
struct PolyNode {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  HPolyhedron H;
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> V;
  Eigen::Vector3d center{Eigen::Vector3d::Zero()};
};
struct Tree {
  std::vector<PolyNode, Eigen::aligned_allocator<PolyNode>> nodes;
  std::vector<std::vector<int>> adj;
};

static Eigen::Vector3d CentroidFromVerts(
    const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>& V) {
  if (V.empty()) return Eigen::Vector3d::Zero();
  Eigen::Vector3d c = Eigen::Vector3d::Zero();
  for (const auto& p : V) c += p;
  return c / double(V.size());
}

static Eigen::Vector3d GuidedSampleTowardCenter(
    const Tree& T, const Eigen::Vector3d& lb, const Eigen::Vector3d& ub, double scale_factor) {
  std::mt19937 rng(std::random_device{}());
  std::uniform_real_distribution<double> ux(lb.x(), ub.x());
  std::uniform_real_distribution<double> uy(lb.y(), ub.y());
  std::uniform_real_distribution<double> uz(lb.z(), ub.z());
  Eigen::Vector3d p_sample(ux(rng), uy(rng), uz(rng));
  if (T.nodes.empty()) return p_sample;
  int best = -1; double bd = 1e300; Eigen::Vector3d c_best;
  for (size_t i=0;i<T.nodes.size();++i) {
    double d = (T.nodes[i].center - p_sample).norm();
    if (d < bd) { bd = d; best = (int)i; c_best = T.nodes[i].center; }
  }
  Eigen::Vector3d v = c_best - p_sample;
  double n = v.norm(); if (n < 1e-9) return c_best;
  v/=n;
  // 简洁估计一条方向上的可行半径（也可以从顶点集估计）
  double r_max = 0.1;
  if (!T.nodes[best].V.empty()) {
    double r = 1e300;
    for (const auto& q : T.nodes[best].V) {
      double t = (q - c_best).dot(-v);
      if (t > 1e-6) r = std::min(r, t);
    }
    if (std::isfinite(r)) r_max = r;
  }
  double alpha = scale_factor + (1.0 - scale_factor) * 0.5;
  return c_best - alpha * r_max * v;
}

static std::vector<int> BfsPath(const std::vector<std::vector<int>>& adj, int s, int t) {
  std::vector<int> prev(adj.size(), -1);
  std::queue<int> q; q.push(s); prev[s] = -2;
  while(!q.empty()){
    int u=q.front(); q.pop(); if (u==t) break;
    for(size_t v=0; v<adj.size(); ++v)
      if (adj[u][v] && prev[v]==-1){ prev[v]=u; q.push((int)v); }
  }
  if (prev[t]==-1) return {};
  std::vector<int> path; for(int v=t; v!=-2; v=prev[v]) path.push_back(v);
  std::reverse(path.begin(), path.end()); return path;
}

// ============================ UR5 DH & IK ============================
struct DH { double alpha, a, d, theta_offset; };
static std::vector<DH> MakeUr5DH() {
  std::vector<DH> dh(6);
  dh[0] = {  M_PI/2,   0.0,      0.089159,  0.0 };
  dh[1] = {  0.0,     -0.425,    0.0,       0.0 };
  dh[2] = {  0.0,     -0.392,    0.0,       0.0 };
  dh[3] = {  M_PI/2,   0.0,      0.10915,   0.0 };
  dh[4] = { -M_PI/2,   0.0,      0.09465,   0.0 };
  dh[5] = {  0.0,      0.0,      0.0823,    0.0 };
  return dh;
}
static Eigen::Matrix4d A_i(double th,double d,double a,double al){
  const double ct=std::cos(th), st=std::sin(th), ca=std::cos(al), sa=std::sin(al);
  Eigen::Matrix4d T;
  T<< ct,-st*ca, st*sa, a*ct,
      st, ct*ca,-ct*sa, a*st,
       0,    sa,    ca,    d,
       0,     0,     0,    1;
  return T;
}
// Tj[0]=I, Tj[i+1] = T0_i+1
static void fk_all(const std::vector<DH>& dh,
                   const Eigen::Matrix<double,6,1>& q,
                   std::array<Eigen::Isometry3d,7>& Tj)
{
  Tj[0].setIdentity();
  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  for(int i=0;i<6;++i){
    T = T * A_i(q(i)+dh[i].theta_offset, dh[i].d, dh[i].a, dh[i].alpha);
    Tj[i+1].matrix() = T;
  }
}
static Eigen::Matrix<double,3,6> positionJacobianAtPoint(
    const std::array<Eigen::Isometry3d,7>& Tj,
    const Eigen::Vector3d& p_world) {
  Eigen::Matrix<double,3,6> Jp; Jp.setZero();
  for(int i=0;i<6;++i){
    const Eigen::Vector3d oi = Tj[i].translation();
    const Eigen::Vector3d zi = Tj[i].linear().col(2);
    Jp.col(i) = zi.cross(p_world - oi);
  }
  return Jp;
}

struct IkOptions { int max_iters{200}; double pos_tol{1e-4}; double step_limit{0.2}; double lambda{3e-2}; };
static bool ik_pos_solve(const std::vector<DH>& dh, const Eigen::Isometry3d& T_align,
                         const Eigen::Vector3d& target_world,
                         Eigen::Matrix<double,6,1>& q_io,
                         const IkOptions& opt, rclcpp::Logger logger)
{
  for (int it=0; it<opt.max_iters; ++it) {
    std::array<Eigen::Isometry3d,7> Tj;
    fk_all(dh, q_io, Tj);
    for(int j=0;j<=6;++j) Tj[j] = T_align * Tj[j];
    Eigen::Vector3d p = Tj[6].translation();
    Eigen::Vector3d e = target_world - p;
    if (e.norm() < opt.pos_tol) return true;

    auto Jp = positionJacobianAtPoint(Tj, p);
    Eigen::Matrix3d JJt = Jp * Jp.transpose();
    JJt += (opt.lambda * opt.lambda) * Eigen::Matrix3d::Identity();
    Eigen::Vector3d y = JJt.ldlt().solve(e);
    Eigen::Matrix<double,6,1> dq = Jp.transpose() * y;
    for (int k=0;k<6;++k) dq(k) = std::clamp(dq(k), -opt.step_limit, opt.step_limit);
    q_io += dq;
  }
  // final check
  {
    std::array<Eigen::Isometry3d,7> Tj;
    fk_all(dh, q_io, Tj);
    for (int j = 0; j <= 6; ++j) Tj[j] = T_align * Tj[j];
    const double err = (target_world - Tj[6].translation()).norm();
    if (err < opt.pos_tol) return true;
    RCLCPP_WARN(logger, "[IK] not converged, |e|=%.6g", err);
    return false;
  }
}

// —— 可视化用“连杆包围盒”（不依赖 FCL）
struct LinkBoxViz {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::Isometry3d T;
  Eigen::Vector3d   size;
};
static LinkBoxViz make_link_box(const Eigen::Vector3d& p1, const Eigen::Vector3d& p2,
                                double half_W, double half_L){
  Eigen::Vector3d z = p2 - p1;
  double len = z.norm();
  if(len < 1e-9){ z={0,0,1}; len=1e-6; } else { z/=len; }
  Eigen::Vector3d x = (std::fabs(z.x())>0.9)?Eigen::Vector3d(0,1,0):Eigen::Vector3d(1,0,0);
  Eigen::Vector3d y = z.cross(x).normalized(); x = y.cross(z).normalized();
  double sx=2*half_W, sy=2*half_L, sz=len;
  Eigen::Vector3d c = 0.5*(p1+p2);
  Eigen::Isometry3d Tw=Eigen::Isometry3d::Identity();
  Tw.linear().col(0)=x; Tw.linear().col(1)=y; Tw.linear().col(2)=z; Tw.translation()=c;
  return LinkBoxViz{Tw,{sx,sy,sz}};
}
static std::vector<LinkBoxViz, Eigen::aligned_allocator<LinkBoxViz>>
build_link_boxes_for_q(const std::vector<DH>& dh, const Eigen::Isometry3d& T_align,
                       const Eigen::Matrix<double,6,1>& q, double half_W, double half_L)
{
  std::array<Eigen::Isometry3d,7> Tj; fk_all(dh, q, Tj);
  for (int j=0; j<=6; ++j) Tj[j] = T_align * Tj[j];
  std::array<Eigen::Vector3d,7> Jpos;
  for (int i=0;i<=6;++i) Jpos[i] = Tj[i].translation();
  std::vector<LinkBoxViz, Eigen::aligned_allocator<LinkBoxViz>> boxes; boxes.reserve(6);
  for (int i=0;i<6;++i) boxes.push_back(make_link_box(Jpos[i], Jpos[i+1], half_W, half_L));
  return boxes;
}

// 在 POLY 内随机采样一点
static Eigen::Vector3d SamplePointInPoly(const HPolyhedron& H, std::mt19937& rng, int max_trials=200) {
  auto V = VerticesFromH3D(H);
  if (V.size() < 4) {
    Eigen::Vector3d c = Eigen::Vector3d::Zero();
    if (!V.empty()) { for (auto& v: V) c += v; c /= std::max<size_t>(1,V.size()); }
    return c;
  }
  Eigen::Vector3d lo = V[0], hi = V[0];
  for (auto& v: V) { lo = lo.cwiseMin(v); hi = hi.cwiseMax(v); }
  std::uniform_real_distribution<double> dx(lo.x(), hi.x());
  std::uniform_real_distribution<double> dy(lo.y(), hi.y());
  std::uniform_real_distribution<double> dz(lo.z(), hi.z());
  for (int t=0;t<max_trials;++t) {
    Eigen::Vector3d x(dx(rng),dy(rng),dz(rng));
    if (PointInH(H, x)) return x;
  }
  Eigen::Vector3d c = Eigen::Vector3d::Zero();
  for (auto& v: V) c += v; c /= V.size();
  return c;
}

// ============================ 主节点 ============================
class SpaceRrtIrisIkVizNode : public rclcpp::Node {
public:
  SpaceRrtIrisIkVizNode() : Node("corridor_spacerrt_ik_viz") {
    // 参数
    frame_ = this->declare_parameter<std::string>("frame_id", "map");
    pub_period_ms_ = this->declare_parameter<int>("pub_period_ms", 500);
// ===== 批量运行相关参数 =====
int batch_trials = this->declare_parameter<int>("batch_trials", 50);
std::string out_dir = this->declare_parameter<std::string>(
    "out_dir", "/home/hyj/iris_rviz_ws/src/iris_rviz_cpp/src/tmp/spacerrt_runs");  // 修改为你喜欢的路径
bool q_save_deg = this->declare_parameter<bool>("q_save_deg", true);

// 确保输出目录存在
{ std::error_code ec; std::filesystem::create_directories(out_dir, ec); }

    // 域
    std::vector<double> lb = this->declare_parameter<std::vector<double>>("domain_lb", {-1.0,-1.0,0.0});
    std::vector<double> ub = this->declare_parameter<std::vector<double>>("domain_ub", { 1.0, 1.0,1.0});
    lb_ = Eigen::Vector3d(lb[0],lb[1],lb[2]); ub_ = Eigen::Vector3d(ub[0],ub[1],ub[2]);
    domain_ = HPolyhedron::MakeBox(lb_, ub_);

    // 起止点
    auto start = this->declare_parameter<std::vector<double>>("start_xyz", { 0.351, 0.341, 0.419});
    auto goal  = this->declare_parameter<std::vector<double>>("goal_xyz",  {-0.200, 0.724, 0.280});
    p_start_ = Eigen::Vector3d(start[0],start[1],start[2]);
    p_goal_  = Eigen::Vector3d(goal[0],goal[1],goal[2]);

    // 障碍（与之前例子一致，便于对比）
    init_obstacles_like_viewer_();

    // IRIS & RRT 参数
    max_iter_     = this->declare_parameter<int>("max_iter", 200);
    scale_factor_ = this->declare_parameter<double>("scale_factor", 0.7);

    // IK 参数
    auto q_start_deg = this->declare_parameter<std::vector<double>>(
      "q_start_deg", std::vector<double>{27, -90.0,  90.0, 0.0, 75.6, 0.0});
    auto q_goal_deg  = this->declare_parameter<std::vector<double>>(
      "q_goal_deg",  std::vector<double>{97, -46.8, 50.4, -3.6, 99.0, 0.0});
    if (q_start_deg.size()!=6 || q_goal_deg.size()!=6) {
      RCLCPP_FATAL(get_logger(), "q_start_deg / q_goal_deg must be length-6");
      rclcpp::shutdown(); return;
    }
    for(int i=0;i<6;++i){ q_start_(i)=q_start_deg[i]*M_PI/180.0; q_goal_(i)=q_goal_deg[i]*M_PI/180.0; }

    double align_yaw_deg = this->declare_parameter<double>("align_yaw_deg", 180.0);
    auto align_xyz = this->declare_parameter<std::vector<double>>("align_xyz", {0.0,0.0,0.0});
    T_align_.setIdentity();
    T_align_.linear() = Eigen::AngleAxisd(align_yaw_deg * M_PI / 180.0, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    if (align_xyz.size()==3) T_align_.translation() = Eigen::Vector3d(align_xyz[0],align_xyz[1],align_xyz[2]);

    ik_opt_.max_iters  = this->declare_parameter<int>("ik_max_iters", 200);
    ik_opt_.pos_tol    = this->declare_parameter<double>("ik_pos_tol", 1e-4);
    ik_opt_.step_limit = this->declare_parameter<double>("ik_step_limit", 0.2);
    ik_opt_.lambda     = this->declare_parameter<double>("ik_lambda", 3e-2);

    half_W_ = this->declare_parameter<double>("link_half_width",  0.025);
    half_L_ = this->declare_parameter<double>("link_half_length", 0.025);

    poly_line_width_  = this->declare_parameter<double>("poly_line_width",  0.0020);
    inter_line_width_ = this->declare_parameter<double>("inter_line_width", 0.0100);

    poly_rgba_  = get_rgba_("poly_color_rgba",  {0.10,0.60,1.00,1.0});
    inter_rgba_ = get_rgba_("inter_color_rgba", {0.95,0.75,0.10,1.0});

    // Publisher（latched）
    auto qos = rclcpp::QoS(rclcpp::KeepLast(10)).reliable().transient_local();
    pub_ = this->create_publisher<MarkerArray>("/spacerrt_iris_ik_viz", qos);
    timer_ = this->create_wall_timer(std::chrono::milliseconds(pub_period_ms_),
                                     std::bind(&SpaceRrtIrisIkVizNode::publish_all_, this));

    // ===== 批量运行 N 次 =====
for (int run = 0; run < batch_trials; ++run) {
  // 方便可复现：给内部若干用到的随机行为一个“不同种子”的扰动
  // （本例中多数随机在函数里 new rng；这里我们只在日志/文件名上区分）
  RCLCPP_INFO(get_logger(), "===== Batch run %d / %d =====", run+1, batch_trials);

  // --- 清空上一轮状态 ---
  Ts_ = Tree{}; Tg_ = Tree{};
  final_polys_.clear(); final_inters_.clear();
  interleaved_.clear(); inter_kind_.clear();
  path_q_.clear(); sampled_.clear();
  connected_ = false; conn_s_ = conn_g_ = -1;

  // 1) 起止 IRIS
  P_start_ = Iris(obstacles_, p_start_, domain_);
  P_goal_  = Iris(obstacles_, p_goal_ , domain_);

  // 2) RRT 扩张
  add_node(Ts_, P_start_);
  add_node(Tg_, P_goal_);
  if (auto inter0 = IntersectIfNonempty(Ts_.nodes[0].H, Tg_.nodes[0].H)) {
    connected_ = true; conn_s_=0; conn_g_=0; interH_ = *inter0;
  } else {
    expand_rrt_till_connect_();
  }

  if (!connected_) {
    RCLCPP_WARN(get_logger(), "[Run %d] Trees not connected; saving partial data skipped.", run);
    continue; // 本轮跳过存盘
  }

  // 3) 生成交错走廊
  build_final_path_();

  // 4) 采样 + IK
  dh_ = MakeUr5DH();
  build_path_by_sampling_and_ik_();

  // 5) 存盘
  char buf_corr[512], buf_q[512];
  std::snprintf(buf_corr, sizeof(buf_corr), "%s/run_%03d_corridor.txt", out_dir.c_str(), run);
  std::snprintf(buf_q,    sizeof(buf_q),    "%s/run_%03d_q.txt",        out_dir.c_str(), run);

  bool ok1 = SaveCorridorTxt(buf_corr, frame_, p_start_, p_goal_, interleaved_, inter_kind_);
  bool ok2 = SaveJointPathTxt(buf_q, path_q_, q_save_deg, /*precision=*/6);
  if (ok1 && ok2) {
    RCLCPP_INFO(get_logger(), "Saved: %s  and  %s", buf_corr, buf_q);
  } else {
    RCLCPP_WARN(get_logger(), "Saving failed for run %d (corridor=%d, q=%d)", run, ok1, ok2);
  }

  // 6) 可选：发布本轮可视化（默认开启，便于实时观察）
  publish_all_();
}

RCLCPP_INFO(get_logger(), "===== Batch finished: %d runs to %s =====", batch_trials, out_dir.c_str());

  }

private:
static bool SaveCorridorTxt(const std::string& filepath,
                            const std::string& frame,
                            const Eigen::Vector3d& p_start,
                            const Eigen::Vector3d& p_goal,
                            const std::vector<HPolyhedron>& interleaved,
                            const std::vector<char>& kinds /*'P' or 'I'*/) {
  if (interleaved.size() != kinds.size()) return false;
  std::ofstream fout(filepath);
  if (!fout.is_open()) return false;

  fout << "# SAFETY_CORRIDOR v1\n";
  fout << "frame: " << frame << "\n";
  fout << std::setprecision(16);
  fout << "start: " << p_start.x() << " " << p_start.y() << " " << p_start.z() << "\n";
  fout << "goal:  " << p_goal.x()  << " " << p_goal.y()  << " " << p_goal.z()  << "\n";
  fout << "segments: " << interleaved.size() << "\n\n";

  for (size_t i=0;i<interleaved.size();++i) {
    const auto& H = interleaved[i];
    const auto& A = H.A();
    const auto& b = H.b();
    std::string type = (kinds[i]=='I' ? "INTER" : "POLY");
    fout << "SEG " << i << " TYPE " << type << " ROWS " << A.rows() << "\n";
    for (int r=0; r<A.rows(); ++r) {
      fout << A(r,0) << " " << A(r,1) << " " << A(r,2) << " " << b(r) << "\n";
    }
    fout << "ENDSEG\n\n";
  }
  return true;
}
static bool SaveJointPathTxt(const std::string& filepath,
                             const std::vector<Eigen::Matrix<double,6,1>,
                                               Eigen::aligned_allocator<Eigen::Matrix<double,6,1>>>& path_q,
                             bool in_degree = true, int precision = 6) {
  std::ofstream fout(filepath);
  if (!fout.is_open()) return false;

  auto to_deg = [](double r){ return r * 180.0 / M_PI; };
  fout.setf(std::ios::fixed); fout << std::setprecision(precision);
  fout << "# JOINT_PATH " << (in_degree ? "deg" : "rad") << "\n";
  fout << "waypoints: " << path_q.size() << "\n";
  for (size_t t=0; t<path_q.size(); ++t) {
    fout << "t " << t;
    for (int k=0;k<6;++k) {
      double v = in_degree ? to_deg(path_q[t](k)) : path_q[t](k);
      fout << " " << v;
    }
    fout << "\n";
  }
  return true;
}

  // ---------- 参数辅助 ----------
  std::array<double,4> get_rgba_(const std::string& name, const std::array<double,4>& def) {
    auto v = this->declare_parameter<std::vector<double>>(name,
      std::vector<double>{def[0],def[1],def[2],def[3]});
    if (v.size()!=4) v = {def[0],def[1],def[2],def[3]};
    return {v[0],v[1],v[2],v[3]};
  }

  // ---------- RRT 扩张 ----------
  size_t add_node(Tree& T, const HPolyhedron& H) {
    PolyNode n;
    n.H = H;
    n.V = VerticesFromH3D(H);
    n.center = n.V.empty() ? Eigen::Vector3d::Zero() : CentroidFromVerts(n.V);
    T.nodes.push_back(std::move(n));
    const size_t N = T.nodes.size();
    T.adj.resize(N);
    for (auto& row : T.adj) row.resize(N, 0);
    if (N >= 2) {
      size_t id = N - 1;
      for (size_t j = 0; j + 1 < N; ++j) {
        auto inter = IntersectIfNonempty(T.nodes[id].H, T.nodes[j].H);
        if (inter) { T.adj[id][j] = T.adj[j][id] = 1; }
      }
    }
    return N - 1;
  }

  std::optional<size_t> expand_once_(const std::string& tag, Tree& T) {
    Eigen::Vector3d q = GuidedSampleTowardCenter(T, lb_, ub_, scale_factor_);
    HPolyhedron H;
    try { H = Iris(obstacles_, q, domain_); }
    catch (const std::exception& e) {
      RCLCPP_WARN(get_logger(), "[%s] IRIS exception: %s", tag.c_str(), e.what());
      return std::nullopt;
    }
    auto V = VerticesFromH3D(H);
    // 过滤太扁/太小的候选
    if (V.size() < 4) { RCLCPP_INFO(get_logger(), "[%s] too few vertices, skip.", tag.c_str()); return std::nullopt; }
    size_t id = add_node(T, H);
    RCLCPP_INFO(get_logger(), "[%s] added node %zu (verts=%zu)", tag.c_str(), id, T.nodes[id].V.size());
    return id;
  }

  void expand_rrt_till_connect_() {
    connected_ = false; conn_s_=-1; conn_g_=-1;
    for (int it=1; it<=max_iter_; ++it) {
      RCLCPP_INFO(get_logger(), "[Iter %d] S_nodes=%zu, G_nodes=%zu", it, Ts_.nodes.size(), Tg_.nodes.size());
      auto id_s = expand_once_("S", Ts_);
      auto id_g = expand_once_("G", Tg_);
      if (id_s) {
        for (size_t j=0; j<Tg_.nodes.size(); ++j) {
          auto inter = IntersectIfNonempty(Ts_.nodes[*id_s].H, Tg_.nodes[j].H);
          if (inter) { connected_=true; conn_s_=(int)*id_s; conn_g_=(int)j; interH_=*inter; break; }
        }
      }
      if (!connected_ && id_g) {
        for (size_t i=0; i<Ts_.nodes.size(); ++i) {
          auto inter = IntersectIfNonempty(Ts_.nodes[i].H, Tg_.nodes[*id_g].H);
          if (inter) { connected_=true; conn_s_=(int)i; conn_g_=(int)*id_g; interH_=*inter; break; }
        }
      }
      if (connected_) { RCLCPP_INFO(get_logger(), "✅ Trees connected at S[%d] <-> G[%d]", conn_s_, conn_g_); break; }
    }
  }

  // ---------- 构造交错走廊 ----------
  void build_final_path_() {
    auto path_s = BfsPath(Ts_.adj, 0, conn_s_);
    auto path_g = BfsPath(Tg_.adj, 0, conn_g_);
    std::reverse(path_g.begin(), path_g.end());
    final_polys_.clear(); final_inters_.clear();
    interleaved_.clear(); inter_kind_.clear();

    for (int idx : path_s) final_polys_.push_back(Ts_.nodes[idx].H);
    for (int idx : path_g) final_polys_.push_back(Tg_.nodes[idx].H);

    for (size_t i=0; i+1<path_s.size(); ++i) {
      final_inters_.push_back(PortalBetween(Ts_.nodes[path_s[i]].H, Ts_.nodes[path_s[i+1]].H, 1e-9, get_logger()));
    }
    final_inters_.push_back(PortalBetween(Ts_.nodes[conn_s_].H, Tg_.nodes[conn_g_].H, 1e-9, get_logger()));
    for (size_t i=0; i+1<path_g.size(); ++i) {
      final_inters_.push_back(PortalBetween(Tg_.nodes[path_g[i]].H, Tg_.nodes[path_g[i+1]].H, 1e-9, get_logger()));
    }

    for (size_t i=0; i<final_polys_.size(); ++i) {
      interleaved_.push_back(final_polys_[i]); inter_kind_.push_back('P');
      if (i < final_inters_.size()) { interleaved_.push_back(final_inters_[i]); inter_kind_.push_back('I'); }
    }
  }

  // ---------- 采样 + IK ----------
  struct SamplePoint { EIGEN_MAKE_ALIGNED_OPERATOR_NEW Eigen::Vector3d p; char kind; int seg_idx; };
  void build_path_by_sampling_and_ik_() {
    // 中间所有 INTER + 非首尾 POLY
    std::vector<int> poly_idx;
    for (int i=0;i<(int)interleaved_.size();++i) if (inter_kind_[i]=='P') poly_idx.push_back(i);
    int first_poly = poly_idx.empty()? -1 : poly_idx.front();
    int last_poly  = poly_idx.empty()? -1 : poly_idx.back();

    std::mt19937 rng(std::random_device{}());
    path_q_.clear();
    path_q_.push_back(q_start_);
    sampled_.clear();

    Eigen::Matrix<double,6,1> q_curr = q_start_;

    for (int i=0; i<(int)interleaved_.size(); ++i) {
      const bool is_poly  = (inter_kind_[i]=='P');
      const bool is_inter = (inter_kind_[i]=='I');
      const bool need_sample = is_inter || (is_poly && i!=first_poly && i!=last_poly);
      if (!need_sample) continue;

      Eigen::Vector3d target = SamplePointInPoly(interleaved_[i], rng);
      Eigen::Matrix<double,6,1> q_next = q_curr;
      bool ok = ik_pos_solve(dh_, T_align_, target, q_next, ik_opt_, get_logger());
      if (!ok) {
        bool solved=false;
        for (int rep=0; rep<20 && !solved; ++rep) {
          target = SamplePointInPoly(interleaved_[i], rng);
          q_next = q_curr;
          solved = ik_pos_solve(dh_, T_align_, target, q_next, ik_opt_, get_logger());
        }
        if (!solved) {
          RCLCPP_WARN(get_logger(), "[SEG %d %c] IK failed; keep previous q.", i, inter_kind_[i]);
          sampled_.push_back(SamplePoint{target, inter_kind_[i], i});
          path_q_.push_back(q_curr);
          continue;
        }
      }
      sampled_.push_back(SamplePoint{target, inter_kind_[i], i});
      path_q_.push_back(q_next);
      q_curr = q_next;
    }
    path_q_.push_back(q_goal_);

    // 打印关节路径（度）
    auto to_deg = [](double r){ return r*180.0/M_PI; };
    for (size_t t=0; t<path_q_.size(); ++t) {
      const auto& q = path_q_[t];
      std::ostringstream ss; ss.setf(std::ios::fixed); ss<<std::setprecision(2);
      ss<<"[t="<<std::setw(2)<<t<<"] ";
      for (int k=0;k<6;++k) ss<<std::setw(8)<<to_deg(q(k))<<(k==5?"":" ");
      if (t==0) ss<<"  | START";
      else if (t+1==path_q_.size()) ss<<"  | GOAL";
      else { const auto& sp = sampled_[t-1]; ss<<"  | seg "<<sp.seg_idx<<" ("<<sp.kind<<")"; }
      RCLCPP_INFO(get_logger(), "%s", ss.str().c_str());
    }
  }

  // ---------- 可视化 ----------
  void draw_hpoly_(const HPolyhedron& H, const std::string& ns,
                   double lw, double r,double g,double b,
                   MarkerArray* ma, int* id) {
    auto V = VerticesFromH3D(H); if (V.size()<4) return;
    auto E = EdgesFromActiveSets(H, V);
    Marker m = MakeLineMarker(frame_, ns, (*id)++, lw, r,g,b, 1.0);
    for (auto& e : E) AddEdge(&m, V[e.first], V[e.second]);
    ma->markers.push_back(m);
  }

  void publish_all_() {
    MarkerArray ma; int id=0;

    // 域线框
    {
      Marker m = MakeLineMarker(frame_, "domain", id++, 0.0025, 0.6,0.6,0.6, 1.0);
      Eigen::Matrix3Xd X(3,8);
      X << lb_.x(),ub_.x(),ub_.x(),lb_.x(), lb_.x(),ub_.x(),ub_.x(),lb_.x(),
           lb_.y(),lb_.y(),ub_.y(),ub_.y(), lb_.y(),lb_.y(),ub_.y(),ub_.y(),
           lb_.z(),lb_.z(),lb_.z(),lb_.z(), ub_.z(),ub_.z(),ub_.z(),ub_.z();
      DrawHullAsAabb(X, &m); ma.markers.push_back(m);
    }

    // 障碍
    const double lw_obst = 0.004;
    for (size_t oi=0; oi<obstacles_vertices_.size(); ++oi) {
      Marker m = MakeLineMarker(frame_, "obst", id++, lw_obst, 0.9,0.1,0.1, 1.0);
      DrawHullAsAabb(obstacles_vertices_[oi], &m);
      ma.markers.push_back(m);
    }
    for (size_t i=0;i<spheres_.size();++i) {
      ma.markers.push_back(MakeSphereMarker(frame_, "spheres", id++,
                                            spheres_[i].first, spheres_[i].second,
                                            0.2,1.0,0.2, 0.35));
    }

    // 起/终点
    ma.markers.push_back(MakeSphereMarker(frame_, "seed", id++, p_start_, 0.01, 1.0,0.8,0.1,1.0));
    ma.markers.push_back(MakeSphereMarker(frame_, "seed", id++, p_goal_,  0.01, 0.1,0.9,0.3,1.0));

    // 起/终 IRIS
    draw_hpoly_(P_start_, "iris_start", 0.002, 0.1,0.6,1.0, &ma, &id);
    draw_hpoly_(P_goal_,  "iris_goal",  0.002, 0.2,1.0,0.4, &ma, &id);

    // 走廊：交错
    for (size_t i=0;i<interleaved_.size(); ++i) {
      const bool is_poly = (inter_kind_[i]=='P');
      double r = is_poly? poly_rgba_[0]: inter_rgba_[0];
      double g = is_poly? poly_rgba_[1]: inter_rgba_[1];
      double b = is_poly? poly_rgba_[2]: inter_rgba_[2];
      double lw= is_poly? poly_line_width_ : inter_line_width_;
      draw_hpoly_(interleaved_[i], is_poly?"poly":"inter", lw, r,g,b, &ma, &id);
    }

    // 采样点
    for (const auto& sp : sampled_) {
      const bool is_poly = (sp.kind=='P');
      double r = is_poly ? 0.10 : 0.95;
      double g = is_poly ? 1.00 : 0.85;
      double b = is_poly ? 0.20 : 0.10;
      ma.markers.push_back(MakeSphereMarker(frame_, is_poly?"samples_poly":"samples_inter",
                                            id++, sp.p, 0.02, r,g,b, 0.95));
    }

    // 末端轨迹
    {
      Marker m; m.header.frame_id=frame_; m.ns="ee_path"; m.id=id++;
      m.type=Marker::LINE_STRIP; m.action=Marker::ADD; m.scale.x=0.01;
      m.color.a=1.0; m.color.r=0.1; m.color.g=0.9; m.color.b=0.9;
      for (const auto& q : path_q_) {
        std::array<Eigen::Isometry3d,7> Tj; fk_all(dh_, q, Tj);
        for (int j=0;j<=6;++j) Tj[j] = T_align_ * Tj[j];
        Eigen::Vector3d p = Tj[6].translation();
        Point P; P.x=p.x(); P.y=p.y(); P.z=p.z(); m.points.push_back(P);
      }
      ma.markers.push_back(m);
    }

    // 连杆包围盒
    for (size_t t=0; t<path_q_.size(); ++t) {
      auto boxes = build_link_boxes_for_q(dh_, T_align_, path_q_[t], half_W_, half_L_);
      for (const auto& b : boxes) {
        Marker m; m.header.frame_id=frame_; m.ns="links_t"+std::to_string(t); m.id=id++;
        m.type=Marker::CUBE; m.action=Marker::ADD;
        Eigen::Quaterniond q(b.T.rotation());
        m.pose.position.x=b.T.translation().x(); m.pose.position.y=b.T.translation().y(); m.pose.position.z=b.T.translation().z();
        m.pose.orientation.x=q.x(); m.pose.orientation.y=q.y(); m.pose.orientation.z=q.z(); m.pose.orientation.w=q.w();
        m.scale.x=b.size.x(); m.scale.y=b.size.y(); m.scale.z=b.size.z();
        m.color.a=0.35f; m.color.r=0.2f; m.color.g=0.8f; m.color.b=0.2f;
        ma.markers.push_back(m);
      }
    }

    pub_->publish(ma);
  }

  // ---------- 障碍（与 viewer 示例一致） ----------
  void init_obstacles_like_viewer_() {
    auto make_X = [](std::initializer_list<std::initializer_list<double>> rows) {
      Eigen::Matrix3Xd X(3,8);
      std::vector<std::vector<double>> r; for (auto& row:rows) r.emplace_back(row);
      for (int j=0;j<8;++j) { X(0,j)=r[0][j]; X(1,j)=r[1][j]; X(2,j)=r[2][j]; }
      return X;
    };
    obstacles_vertices_.clear();
    obstacles_vertices_.push_back(make_X({
      {   0,    0,    0,    0, 0.400, 0.400, 0.400, 0.400},
      {0.550,0.550,0.570,0.570,0.550,0.550,0.570,0.570},
      {   0, 0.800,    0,0.800,    0,0.800,    0,0.800}
    }));
    obstacles_vertices_.push_back(make_X({
      {-0.420,-0.420,-0.420,-0.420,-0.400,-0.400,-0.400,-0.400},
      { 0.600, 0.600, 0.800, 0.800, 0.600, 0.600, 0.800, 0.800},
      {    0, 0.800,    0, 0.800,    0, 0.800,    0, 0.800}
    }));
    obstacles_vertices_.push_back(make_X({
      { 0.400, 0.400, 0.400, 0.400, 0.420, 0.420, 0.420, 0.420},
      { 0.600, 0.600, 0.800, 0.800, 0.600, 0.600, 0.800, 0.800},
      {    0, 0.800,    0, 0.800,    0, 0.800,    0, 0.800}
    }));
    obstacles_vertices_.push_back(make_X({
      {-0.400,-0.400,-0.400,-0.400, 0.400, 0.400, 0.400, 0.400},
      { 0.600, 0.600, 0.800, 0.800, 0.600, 0.600, 0.800, 0.800},
      { 0.180, 0.200, 0.180, 0.200, 0.180, 0.200, 0.180, 0.200}
    }));
    obstacles_vertices_.push_back(make_X({
      {-0.400,-0.400,-0.400,-0.400, 0.400, 0.400, 0.400, 0.400},
      { 0.600, 0.600, 0.800, 0.800, 0.600, 0.600, 0.800, 0.800},
      { 0.600, 0.620, 0.600, 0.620, 0.600, 0.620, 0.600, 0.620}
    }));
    spheres_ = {
      {Eigen::Vector3d( 0.050, 0.500, 0.400), 0.040},
      {Eigen::Vector3d( 0.000, 0.500, 0.600), 0.050},
      {Eigen::Vector3d(-0.350, 0.500, 0.500), 0.050},
    };

    // Drake 障碍：ConvexSets 需要 shared_ptr<ConvexSet>
    // Drake 障碍：ConvexSets 需要 copyable_unique_ptr<ConvexSet>
obstacles_.clear();

// 顶点凸包障碍
for (const auto& X : obstacles_vertices_) {
  obstacles_.emplace_back(std::make_unique<VPolytope>(X));
}

// 球体外包
for (const auto& s : spheres_) {
  obstacles_.emplace_back(MakeSpherePolyApprox(s.first, s.second, 64));
}

  }

private:
  // ROS
  rclcpp::Publisher<MarkerArray>::SharedPtr pub_;
  rclcpp::TimerBase::SharedPtr timer_;
  int pub_period_ms_{500};
  std::string frame_{"map"};

  // 域/障碍
  Eigen::Vector3d lb_, ub_;
  HPolyhedron domain_;
  ConvexSets obstacles_;
  std::vector<Eigen::Matrix3Xd> obstacles_vertices_;
  std::vector<std::pair<Eigen::Vector3d,double>> spheres_;

  // 起终 & IRIS
  Eigen::Vector3d p_start_, p_goal_;
  HPolyhedron P_start_, P_goal_;

  // 双树
  Tree Ts_, Tg_;
  bool connected_{false};
  int conn_s_{-1}, conn_g_{-1};
  HPolyhedron interH_;
  int max_iter_{200}; double scale_factor_{0.7};

  // 走廊
  std::vector<HPolyhedron> final_polys_, final_inters_;
  std::vector<HPolyhedron> interleaved_;
  std::vector<char>        inter_kind_; // 'P' / 'I'

  // 机械臂与 IK
  std::vector<DH> dh_;
  Eigen::Isometry3d T_align_{Eigen::Isometry3d::Identity()};
  Eigen::Matrix<double,6,1> q_start_{}, q_goal_{};
  IkOptions ik_opt_;
  double half_W_{0.025}, half_L_{0.025};
  std::vector<Eigen::Matrix<double,6,1>, Eigen::aligned_allocator<Eigen::Matrix<double,6,1>>> path_q_;
  std::vector<SamplePoint, Eigen::aligned_allocator<SamplePoint>> sampled_;

  // 线框样式
  double poly_line_width_{0.0020}, inter_line_width_{0.0100};
  std::array<double,4> poly_rgba_{0.10,0.60,1.00,1.0};
  std::array<double,4> inter_rgba_{0.95,0.75,0.10,1.0};
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SpaceRrtIrisIkVizNode>());
  rclcpp::shutdown();
  return 0;
}

