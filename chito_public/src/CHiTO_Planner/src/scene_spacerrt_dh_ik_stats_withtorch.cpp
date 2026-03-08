#include <rclcpp/rclcpp.hpp>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <geometry_msgs/msg/point.hpp>

#include <drake/geometry/optimization/convex_set.h>
#include <drake/geometry/optimization/hpolyhedron.h>
#include <drake/geometry/optimization/vpolytope.h>
#include <drake/geometry/optimization/iris.h>
#include <drake/solvers/mathematical_program.h>
#include <drake/solvers/solve.h>

#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <optional>
#include <queue>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cctype>
#include <cstdio>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using visualization_msgs::msg::Marker;
using visualization_msgs::msg::MarkerArray;
using geometry_msgs::msg::Point;
using drake::geometry::optimization::ConvexSets;
using drake::geometry::optimization::HPolyhedron;
using drake::geometry::optimization::Iris;
using drake::geometry::optimization::VPolytope;
namespace mp = drake::solvers;
namespace fs = std::filesystem;

// ============================================================
// marker helpers
// ============================================================
static Marker MakeLine(const std::string& frame, const std::string& ns, int id,
                       double w, double r, double g, double b, double a=1.0) {
  Marker m;
  m.header.frame_id = frame;
  m.ns = ns; m.id = id;
  m.type = Marker::LINE_LIST;
  m.action = Marker::ADD;
  m.scale.x = w;
  m.color.r = r; m.color.g = g; m.color.b = b; m.color.a = a;
  m.pose.orientation.w = 1.0;
  return m;
}

static Marker MakeSphere(const std::string& frame, const std::string& ns, int id,
                         const Eigen::Vector3d& c, double rad,
                         double r, double g, double b, double a=1.0) {
  Marker m;
  m.header.frame_id = frame;
  m.ns = ns; m.id = id;
  m.type = Marker::SPHERE;
  m.action = Marker::ADD;
  m.pose.position.x = c.x();
  m.pose.position.y = c.y();
  m.pose.position.z = c.z();
  m.pose.orientation.w = 1.0;
  m.scale.x = 2*rad; m.scale.y = 2*rad; m.scale.z = 2*rad;
  m.color.r = r; m.color.g = g; m.color.b = b; m.color.a = a;
  return m;
}

static void AddEdge(Marker* m, const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
  Point p0, p1;
  p0.x = a.x(); p0.y = a.y(); p0.z = a.z();
  p1.x = b.x(); p1.y = b.y(); p1.z = b.z();
  m->points.push_back(p0);
  m->points.push_back(p1);
}

// ============================================================
// geometry helpers
// ============================================================
static std::string trim(const std::string& s) {
  size_t a = s.find_first_not_of(" \t\r\n");
  size_t b = s.find_last_not_of(" \t\r\n");
  if (a == std::string::npos) return {};
  return s.substr(a, b-a+1);
}

static std::vector<std::string> split_ws(const std::string& s) {
  std::stringstream ss(s);
  std::vector<std::string> out;
  std::string x;
  while (ss >> x) out.push_back(x);
  return out;
}

static std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>
VerticesFromH3D(const HPolyhedron& H, double feas_tol=1e-9, double dup_tol=1e-8) {
  const auto& A = H.A();
  const auto& b = H.b();
  const int m = A.rows();
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> V;
  for (int i=0;i<m;++i) for (int j=i+1;j<m;++j) for (int k=j+1;k<m;++k) {
    Eigen::Matrix3d M;
    M.row(0)=A.row(i); M.row(1)=A.row(j); M.row(2)=A.row(k);
    if (std::abs(M.determinant()) < 1e-12) continue;
    Eigen::Vector3d rhs(b(i), b(j), b(k));
    Eigen::Vector3d x = M.fullPivLu().solve(rhs);
    if ((A*x - b).maxCoeff() <= feas_tol) {
      bool dup=false;
      for (const auto& q : V) if ((q-x).norm() <= dup_tol) { dup=true; break; }
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
  const int m=A.rows(); const int n=(int)V.size();
  std::vector<std::vector<int>> act(n);
  for (int i=0;i<n;++i)
    for (int r=0;r<m;++r)
      if (std::abs(A.row(r).dot(V[i]) - b(r)) <= tol) act[i].push_back(r);

  auto share_two = [&](int i, int j)->bool{
    int c=0;
    for (int a:act[i]) for (int bb:act[j]) if (a==bb) if (++c>=2) return true;
    return false;
  };

  std::vector<std::pair<int,int>> E;
  for (int i=0;i<n;++i) for (int j=i+1;j<n;++j) {
    if ((V[i]-V[j]).norm() > 1e-9 && share_two(i,j)) E.emplace_back(i,j);
  }
  return E;
}

static Eigen::Vector3d CentroidFromVerts(
  const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>& V) {
  Eigen::Vector3d c = Eigen::Vector3d::Zero();
  if (V.empty()) return c;
  for (const auto& v : V) c += v;
  return c / double(V.size());
}

static bool SafeIsFlat(
  const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>& V,
  double ratio_thresh=0.13) {
  if (V.size() < 4) return true;
  Eigen::Matrix3Xd M(3, V.size());
  for (size_t i=0;i<V.size();++i) M.col(i)=V[i];
  Eigen::Vector3d mu = M.rowwise().mean();
  M.colwise() -= mu;
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(M, Eigen::ComputeThinU | Eigen::ComputeThinV);
  auto s = svd.singularValues();
  if (s.size() < 3 || s(0) <= 1e-12) return true;
  return (s(2) / s(0) < ratio_thresh);
}

static std::optional<HPolyhedron> IntersectIfNonempty(const HPolyhedron& P, const HPolyhedron& Q, double feas_tol=1e-9) {
  const int m1=P.A().rows(), m2=Q.A().rows();
  Eigen::MatrixXd A(m1+m2, 3); Eigen::VectorXd b(m1+m2);
  A << P.A(), Q.A();
  b << P.b(), Q.b();
  mp::MathematicalProgram prog;
  auto x = prog.NewContinuousVariables<3>("x");
  prog.AddLinearConstraint(A*x <= b + Eigen::VectorXd::Constant(b.size(), feas_tol));
  prog.AddQuadraticCost(Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero(), x);
  auto res = mp::Solve(prog);
  if (!res.is_success()) return std::nullopt;
  return HPolyhedron(A,b);
}

static std::optional<Eigen::Vector3d> ChebyshevPoint(const HPolyhedron& H) {
  const auto& A = H.A(); const auto& b = H.b();
  mp::MathematicalProgram prog;
  auto x = prog.NewContinuousVariables<3>("x");
  auto r = prog.NewContinuousVariables<1>("r")(0);
  Eigen::VectorXd an(A.rows());
  for (int i=0;i<A.rows();++i) an(i) = A.row(i).norm();
  for (int i=0;i<A.rows();++i) prog.AddLinearConstraint(A.row(i)*x + an(i)*r <= b(i));
  prog.AddLinearCost(-r);
  auto res = mp::Solve(prog);
  if (!res.is_success()) return std::nullopt;
  return res.GetSolution(x);
}

static Eigen::Vector3d SamplePointIn(const HPolyhedron& H,
                                     std::mt19937& rng,
                                     const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>* V_cached=nullptr) {
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> V_local;
  const auto* V = V_cached;
  if (!V || V->empty()) {
    V_local = VerticesFromH3D(H);
    V = &V_local;
  }
  if (V && !V->empty()) {
    std::gamma_distribution<double> gamma(1.0, 1.0);
    std::vector<double> w(V->size());
    double s=0.0;
    for (auto& wi : w) { wi = std::max(1e-9, gamma(rng)); s += wi; }
    for (auto& wi : w) wi /= s;
    Eigen::Vector3d x = Eigen::Vector3d::Zero();
    for (size_t i=0;i<V->size();++i) x += w[i] * (*V)[i];
    return x;
  }
  if (auto c = ChebyshevPoint(H)) return *c;
  return Eigen::Vector3d::Zero();
}

static double PolygonArea3D(const std::vector<Eigen::Vector3d>& poly, const Eigen::Vector3d& n_unit) {
  if (poly.size() < 3) return 0.0;
  Eigen::Vector3d acc = Eigen::Vector3d::Zero();
  for (size_t i=0; i<poly.size(); ++i) {
    const auto& p = poly[i];
    const auto& q = poly[(i+1) % poly.size()];
    acc += p.cross(q);
  }
  return 0.5 * std::abs(acc.dot(n_unit));
}

static double ComputeConvexPolytopeVolume(const HPolyhedron& H, double plane_tol=1e-7) {
  auto V = VerticesFromH3D(H);
  if (V.size() < 4) return 0.0;

  Eigen::Vector3d c = CentroidFromVerts(V);
  const auto& A = H.A();
  const auto& b = H.b();

  double volume = 0.0;
  for (int r=0; r<A.rows(); ++r) {
    Eigen::Vector3d n = A.row(r).transpose();
    const double nn = n.norm();
    if (nn < 1e-12) continue;
    Eigen::Vector3d n_unit = n / nn;

    std::vector<Eigen::Vector3d> face_pts;
    for (const auto& v : V) {
      if (std::abs(n.dot(v) - b(r)) <= plane_tol) face_pts.push_back(v);
    }
    if (face_pts.size() < 3) continue;

    Eigen::Vector3d fc = Eigen::Vector3d::Zero();
    for (const auto& p : face_pts) fc += p;
    fc /= static_cast<double>(face_pts.size());

    Eigen::Vector3d t1 = (std::abs(n_unit.x()) < 0.9) ? Eigen::Vector3d::UnitX() : Eigen::Vector3d::UnitY();
    t1 = (t1 - t1.dot(n_unit) * n_unit).normalized();
    Eigen::Vector3d t2 = n_unit.cross(t1).normalized();

    std::vector<std::pair<double,Eigen::Vector3d>> ordered;
    ordered.reserve(face_pts.size());
    for (const auto& p : face_pts) {
      Eigen::Vector3d d = p - fc;
      double ang = std::atan2(d.dot(t2), d.dot(t1));
      ordered.push_back({ang, p});
    }
    std::sort(ordered.begin(), ordered.end(),
              [](const auto& a, const auto& b){ return a.first < b.first; });

    std::vector<Eigen::Vector3d> poly;
    poly.reserve(ordered.size());
    for (const auto& ap : ordered) poly.push_back(ap.second);

    const double area = PolygonArea3D(poly, n_unit);
    const double h = std::abs(n_unit.dot(fc - c));
    volume += area * h / 3.0;
  }

  return volume;
}

static double ComputeCorridorVolumeExact(const std::vector<HPolyhedron>& polys) {
  double vol = 0.0;
  for (const auto& H : polys) vol += ComputeConvexPolytopeVolume(H);
  return vol;
}

// ============================================================
// simple .scene parser
// ============================================================
struct SceneBox {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  std::string name;
  Eigen::Vector3d pos{0,0,0};
  Eigen::Quaterniond quat{1,0,0,0};
  Eigen::Vector3d size{1,1,1};
};

struct SceneSphere {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  std::string name;
  Eigen::Vector3d pos{0,0,0};
  double radius{0.05};
};

struct SceneData {
  std::vector<SceneBox> boxes;
  std::vector<SceneSphere> spheres;
};

static bool ParseSceneFile(const std::string& path, SceneData& out, rclcpp::Logger logger) {
  std::ifstream fin(path);
  if (!fin.is_open()) {
    RCLCPP_ERROR(logger, "Failed to open scene file: %s", path.c_str());
    return false;
  }

  std::vector<std::string> lines;
  std::string line;
  while (std::getline(fin, line)) {
    line = trim(line);
    if (!line.empty()) lines.push_back(line);
  }

  for (size_t i=0; i<lines.size();) {
    if (lines[i].empty() || lines[i][0] != '*') { ++i; continue; }

    SceneBox box;
    SceneSphere sph;
    std::string name = trim(lines[i].substr(1));
    if (name.empty()) name = "obj_" + std::to_string(i);
    box.name = name;
    sph.name = name;

    size_t j = i + 1;
    if (j < lines.size()) {
      auto t = split_ws(lines[j]);
      if (t.size() >= 3) {
        box.pos = Eigen::Vector3d(std::stod(t[0]), std::stod(t[1]), std::stod(t[2]));
        sph.pos = box.pos;
      }
      ++j;
    }

    Eigen::Quaterniond qtmp(1,0,0,0);
    bool found_type = false;
    std::string type;
    while (j < lines.size() && !lines[j].empty() && lines[j][0] != '*') {
      const auto tok = split_ws(lines[j]);
      if (tok.size() == 1 && (tok[0] == "box" || tok[0] == "sphere")) {
        type = tok[0];
        found_type = true;
        ++j;
        break;
      }
      if (tok.size() == 4) {
        try {
          double a = std::stod(tok[0]);
          double b = std::stod(tok[1]);
          double c = std::stod(tok[2]);
          double d = std::stod(tok[3]);
          qtmp = Eigen::Quaterniond(d, a, b, c);
          qtmp.normalize();
        } catch (...) {}
      }
      ++j;
    }
    box.quat = qtmp;

    if (!found_type) {
      i = j;
      continue;
    }

    if (type == "box") {
      if (j < lines.size()) {
        auto tok = split_ws(lines[j]);
        if (tok.size() >= 3) {
          box.size = Eigen::Vector3d(std::stod(tok[0]), std::stod(tok[1]), std::stod(tok[2]));
          out.boxes.push_back(box);
        }
      }
    } else if (type == "sphere") {
      if (j < lines.size()) {
        auto tok = split_ws(lines[j]);
        if (!tok.empty()) {
          sph.radius = std::stod(tok[0]);
          out.spheres.push_back(sph);
        }
      }
    }

    i = j + 1;
  }

  RCLCPP_INFO(logger, "Scene parsed: boxes=%zu spheres=%zu", out.boxes.size(), out.spheres.size());
  return !out.boxes.empty() || !out.spheres.empty();
}

static Eigen::Matrix3Xd BoxVertices(const SceneBox& b) {
  Eigen::Matrix3Xd X(3, 8);
  const Eigen::Vector3d h = 0.5 * b.size;
  std::array<Eigen::Vector3d,8> loc = {
    Eigen::Vector3d(-h.x(), -h.y(), -h.z()),
    Eigen::Vector3d( h.x(), -h.y(), -h.z()),
    Eigen::Vector3d( h.x(),  h.y(), -h.z()),
    Eigen::Vector3d(-h.x(),  h.y(), -h.z()),
    Eigen::Vector3d(-h.x(), -h.y(),  h.z()),
    Eigen::Vector3d( h.x(), -h.y(),  h.z()),
    Eigen::Vector3d( h.x(),  h.y(),  h.z()),
    Eigen::Vector3d(-h.x(),  h.y(),  h.z())
  };
  Eigen::Matrix3d R = b.quat.toRotationMatrix();
  for (int i=0;i<8;++i) X.col(i) = R * loc[i] + b.pos;
  return X;
}

static std::unique_ptr<HPolyhedron> MakeSpherePolyApprox(const Eigen::Vector3d& c, double r, int num_dirs=64) {
  Eigen::MatrixXd A(num_dirs, 3);
  Eigen::VectorXd b(num_dirs);
  const double phi = (1.0 + std::sqrt(5.0)) / 2.0;
  for (int i=0;i<num_dirs;++i) {
    double t = (i + 0.5) / num_dirs;
    double z = 1.0 - 2.0*t;
    double ang = 2.0 * M_PI * (i / phi - std::floor(i / phi));
    double rxy = std::sqrt(std::max(0.0, 1.0 - z*z));
    Eigen::Vector3d n(rxy*std::cos(ang), rxy*std::sin(ang), z);
    A.row(i) = n.transpose();
    b(i) = n.dot(c) + r;
  }
  return std::make_unique<HPolyhedron>(A, b);
}

// ============================================================
// Save helpers
// ============================================================
static bool SaveCorridorTxt(
    const std::string& filepath,
    const std::string& frame,
    const Eigen::Vector3d& p_start,
    const Eigen::Vector3d& p_goal,
    const std::vector<HPolyhedron>& polys) {
  std::ofstream fout(filepath);
  if (!fout.is_open()) return false;

  fout << "# CORRIDOR_POLYS_ONLY v1\n";
  fout << "frame: " << frame << "\n";
  fout << std::setprecision(16);
  fout << "start: " << p_start.x() << " " << p_start.y() << " " << p_start.z() << "\n";
  fout << "goal:  " << p_goal.x()  << " " << p_goal.y()  << " " << p_goal.z()  << "\n";
  fout << "segments: " << polys.size() << "\n\n";

  for (size_t i = 0; i < polys.size(); ++i) {
    const auto& H = polys[i];
    fout << "SEG " << i << " TYPE POLY ROWS " << H.A().rows() << "\n";
    for (int r = 0; r < H.A().rows(); ++r) {
      fout << H.A()(r, 0) << " "
           << H.A()(r, 1) << " "
           << H.A()(r, 2) << " "
           << H.b()(r) << "\n";
    }
    fout << "ENDSEG\n\n";
  }

  return true;
}

static bool SaveJointPathCsv(
    const std::string& filepath,
    const std::vector<Eigen::Matrix<double,6,1>,
                      Eigen::aligned_allocator<Eigen::Matrix<double,6,1>>>& path_q,
    bool save_deg = true,
    int precision = 6) {
  std::ofstream fout(filepath);
  if (!fout.is_open()) return false;

  fout << std::fixed << std::setprecision(precision);
  fout << "idx,q1_" << (save_deg ? "deg" : "rad")
       << ",q2_" << (save_deg ? "deg" : "rad")
       << ",q3_" << (save_deg ? "deg" : "rad")
       << ",q4_" << (save_deg ? "deg" : "rad")
       << ",q5_" << (save_deg ? "deg" : "rad")
       << ",q6_" << (save_deg ? "deg" : "rad")
       << "\n";

  for (size_t i = 0; i < path_q.size(); ++i) {
    fout << i;
    for (int k = 0; k < 6; ++k) {
      double v = path_q[i](k);
      if (save_deg) v = v * 180.0 / M_PI;
      fout << "," << v;
    }
    fout << "\n";
  }
  return true;
}

// ============================================================
// Space-RRT on convex IRIS regions
// ============================================================
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

static Eigen::Vector3d GuidedSampleTowardCenter(
  const Tree& T, const Eigen::Vector3d& lb, const Eigen::Vector3d& ub,
  double scale_factor, std::mt19937& rng) {
  std::uniform_real_distribution<double> ux(lb.x(), ub.x());
  std::uniform_real_distribution<double> uy(lb.y(), ub.y());
  std::uniform_real_distribution<double> uz(lb.z(), ub.z());
  Eigen::Vector3d p_sample(ux(rng), uy(rng), uz(rng));
  if (T.nodes.empty()) return p_sample;

  int best=-1; double bd=1e300;
  for (size_t i=0;i<T.nodes.size();++i) {
    const double d = (T.nodes[i].center - p_sample).norm();
    if (d < bd) { bd = d; best = (int)i; }
  }
  Eigen::Vector3d c_best = T.nodes[best].center;
  Eigen::Vector3d v = c_best - p_sample;
  const double n = v.norm();
  if (n < 1e-9) return c_best;
  v /= n;

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

// ============================================================
// UR5 DH + DLS IK + TORCH TIP
// ============================================================
struct DH { double alpha, a, d, theta_offset; };

static std::vector<DH> MakeUr5DH() {
  std::vector<DH> dh(6);
  dh[0] = {  M_PI/2,   0.0,      0.089159,  0.0 };
  dh[1] = {  0.0,     -0.425,    0.0,       0.0 };
  dh[2] = {  0.0,     -0.39225,  0.0,       0.0 };
  dh[3] = {  M_PI/2,   0.0,      0.10915,   0.0 };
  dh[4] = { -M_PI/2,   0.0,      0.09465,   0.0 };
  dh[5] = {  0.0,      0.0,      0.0823,    0.0 };
  return dh;
}

static Eigen::Matrix4d A_i(double th, double d, double a, double al) {
  const double ct = std::cos(th), st = std::sin(th), ca = std::cos(al), sa = std::sin(al);
  Eigen::Matrix4d T;
  T << ct, -st*ca,  st*sa, a*ct,
       st,  ct*ca, -ct*sa, a*st,
        0,     sa,     ca,    d,
        0,      0,      0,    1;
  return T;
}

static void fk_all(const std::vector<DH>& dh,
                   const Eigen::Matrix<double,6,1>& q,
                   std::array<Eigen::Isometry3d,7>& Tj) {
  Tj[0].setIdentity();
  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  for (int i=0;i<6;++i) {
    T = T * A_i(q(i) + dh[i].theta_offset, dh[i].d, dh[i].a, dh[i].alpha);
    Tj[i+1].matrix() = T;
  }
}

static Eigen::Matrix<double,3,6> positionJacobianAtPoint(
  const std::array<Eigen::Isometry3d,7>& Tj,
  const Eigen::Vector3d& p_world) {
  Eigen::Matrix<double,3,6> J; J.setZero();
  for (int i=0;i<6;++i) {
    const Eigen::Vector3d oi = Tj[i].translation();
    const Eigen::Vector3d zi = Tj[i].linear().col(2);
    J.col(i) = zi.cross(p_world - oi);
  }
  return J;
}

struct IkOptions {
  int max_iters{200};
  double pos_tol{1e-4};
  double step_limit{0.2};
  double lambda{3e-2};
};

static Eigen::Isometry3d tcp_world_pose(
    const Eigen::Isometry3d& T_align,
    const std::array<Eigen::Isometry3d,7>& Tj)
{
  return T_align * Tj[6];
}

static void compute_torch_keypoints(
    const Eigen::Isometry3d& T_world_tcp,
    double torch_cyl1_len,
    double torch_tilt_deg,
    double torch_cyl2_len,
    Eigen::Vector3d& p_tcp,
    Eigen::Vector3d& p2,
    Eigen::Vector3d& p4)
{
  p_tcp = T_world_tcp.translation();
  const Eigen::Matrix3d R0 = T_world_tcp.linear();

  const Eigen::Vector3d dir1 = R0 * Eigen::Vector3d::UnitZ();
  p2 = p_tcp + torch_cyl1_len * dir1;

  const double tilt = -torch_tilt_deg * M_PI / 180.0;
  const Eigen::Matrix3d R_tilt =
      Eigen::AngleAxisd(tilt, Eigen::Vector3d::UnitX()).toRotationMatrix();

  const Eigen::Vector3d dir2 = R0 * (R_tilt * Eigen::Vector3d::UnitZ());
  p4 = p2 + torch_cyl2_len * dir2;
}

static Eigen::Vector3d torch_tip_world_pos(
    const Eigen::Isometry3d& T_align,
    const std::array<Eigen::Isometry3d,7>& Tj,
    double torch_cyl1_len,
    double torch_tilt_deg,
    double torch_cyl2_len)
{
  const Eigen::Isometry3d T_world_tcp = tcp_world_pose(T_align, Tj);
  Eigen::Vector3d p_tcp, p2, p_tip;
  compute_torch_keypoints(T_world_tcp, torch_cyl1_len, torch_tilt_deg, torch_cyl2_len,
                          p_tcp, p2, p_tip);
  return p_tip;
}

static bool ik_pos_solve_torch_tip(
    const std::vector<DH>& dh,
    const Eigen::Isometry3d& T_align,
    const Eigen::Vector3d& target_world,
    Eigen::Matrix<double,6,1>& q_io,
    const IkOptions& opt,
    double torch_cyl1_len,
    double torch_tilt_deg,
    double torch_cyl2_len)
{
  for (int it=0; it<opt.max_iters; ++it) {
    std::array<Eigen::Isometry3d,7> Tj;
    fk_all(dh, q_io, Tj);

    Eigen::Vector3d p = torch_tip_world_pos(
        T_align, Tj,
        torch_cyl1_len, torch_tilt_deg, torch_cyl2_len);

    Eigen::Vector3d e = target_world - p;
    if (e.norm() < opt.pos_tol) return true;

    auto J = positionJacobianAtPoint(Tj, p);
    Eigen::Matrix3d JJt = J * J.transpose();
    JJt += (opt.lambda * opt.lambda) * Eigen::Matrix3d::Identity();
    Eigen::Vector3d y = JJt.ldlt().solve(e);
    Eigen::Matrix<double,6,1> dq = J.transpose() * y;

    for (int k=0; k<6; ++k) dq(k) = std::clamp(dq(k), -opt.step_limit, opt.step_limit);
    q_io += dq;
  }

  std::array<Eigen::Isometry3d,7> Tj;
  fk_all(dh, q_io, Tj);

  Eigen::Vector3d p = torch_tip_world_pos(
      T_align, Tj,
      torch_cyl1_len, torch_tilt_deg, torch_cyl2_len);

  return ((target_world - p).norm() < opt.pos_tol);
}

// ============================================================
// Joint interpolation helpers
// ============================================================
static std::vector<Eigen::Matrix<double,6,1>, Eigen::aligned_allocator<Eigen::Matrix<double,6,1>>>
InterpolateJointPath(
    const std::vector<Eigen::Matrix<double,6,1>, Eigen::aligned_allocator<Eigen::Matrix<double,6,1>>>& path_q,
    int interp_steps)
{
  using QVec = std::vector<Eigen::Matrix<double,6,1>, Eigen::aligned_allocator<Eigen::Matrix<double,6,1>>>;
  if (path_q.size() <= 1 || interp_steps <= 0) return path_q;

  QVec out;
  for (size_t i = 0; i + 1 < path_q.size(); ++i) {
    const auto& q0 = path_q[i];
    const auto& q1 = path_q[i+1];

    out.push_back(q0);
    for (int k = 1; k <= interp_steps; ++k) {
      double alpha = static_cast<double>(k) / static_cast<double>(interp_steps + 1);
      Eigen::Matrix<double,6,1> q = (1.0 - alpha) * q0 + alpha * q1;
      out.push_back(q);
    }
  }
  out.push_back(path_q.back());
  return out;
}

// ============================================================
// Build IK seed path using corridor cells + intersections
// ============================================================
static std::vector<Eigen::Matrix<double,6,1>, Eigen::aligned_allocator<Eigen::Matrix<double,6,1>>>
BuildSeedPathLikeCorridorViz(
    const std::vector<HPolyhedron>& polys,
    const Eigen::Vector3d& start_xyz,
    const Eigen::Vector3d& goal_xyz,
    const std::vector<DH>& dh,
    const Eigen::Isometry3d& T_align,
    const Eigen::Matrix<double,6,1>& q_start,
    const Eigen::Matrix<double,6,1>& q_goal,
    const IkOptions& ik_opt,
    std::mt19937& rng,
    bool enable_intersection_ik,
    bool enable_joint_interp,
    int joint_interp_steps,
    double torch_cyl1_len,
    double torch_tilt_deg,
    double torch_cyl2_len,
    std::vector<Eigen::Vector3d>* used_samples = nullptr)
{
  using QVec = std::vector<Eigen::Matrix<double,6,1>, Eigen::aligned_allocator<Eigen::Matrix<double,6,1>>>;
  QVec out;

  const size_t N = polys.size();
  if (N == 0) {
    out.push_back(q_start);
    out.push_back(q_goal);
    return out;
  }

  std::vector<std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>> V_list(N);
  for (size_t i = 0; i < N; ++i) {
    V_list[i] = VerticesFromH3D(polys[i]);
  }

  out.push_back(q_start);
  Eigen::Matrix<double,6,1> q_curr = q_start;

  for (size_t i = 0; i < N; ++i) {
    Eigen::Vector3d sample_poly = SamplePointIn(polys[i], rng, &V_list[i]);

    bool ok_poly = false;
    Eigen::Matrix<double,6,1> q_poly = q_curr;

    for (int rep = 0; rep < 12 && !ok_poly; ++rep) {
      q_poly = q_curr;
      ok_poly = ik_pos_solve_torch_tip(
          dh, T_align, sample_poly, q_poly, ik_opt,
          torch_cyl1_len, torch_tilt_deg, torch_cyl2_len);
      if (!ok_poly) sample_poly = SamplePointIn(polys[i], rng, &V_list[i]);
    }

    if (!ok_poly) {
      double alpha = (N <= 1) ? 0.5 : static_cast<double>(i + 1) / static_cast<double>(N + 1);
      q_poly = (1.0 - alpha) * q_start + alpha * q_goal;
    }

    if (used_samples) used_samples->push_back(sample_poly);
    out.push_back(q_poly);
    q_curr = q_poly;

    if (enable_intersection_ik && i + 1 < N) {
      auto H_inter = IntersectIfNonempty(polys[i], polys[i + 1]);
      if (H_inter.has_value()) {
        auto V_inter = VerticesFromH3D(*H_inter);
        if (V_inter.size() >= 4 && !SafeIsFlat(V_inter)) {
          Eigen::Vector3d sample_inter = SamplePointIn(*H_inter, rng, &V_inter);

          bool ok_inter = false;
          Eigen::Matrix<double,6,1> q_inter = q_curr;

          for (int rep = 0; rep < 12 && !ok_inter; ++rep) {
            q_inter = q_curr;
            ok_inter = ik_pos_solve_torch_tip(
                dh, T_align, sample_inter, q_inter, ik_opt,
                torch_cyl1_len, torch_tilt_deg, torch_cyl2_len);
            if (!ok_inter) sample_inter = SamplePointIn(*H_inter, rng, &V_inter);
          }

          if (ok_inter) {
            if (used_samples) used_samples->push_back(sample_inter);
            out.push_back(q_inter);
            q_curr = q_inter;
          }
        }
      }
    }
  }

  out.push_back(q_goal);

  if (enable_joint_interp && joint_interp_steps > 0) {
    out = InterpolateJointPath(out, joint_interp_steps);
  }

  return out;
}

// ============================================================
// Main node
// ============================================================
class SceneSpaceRrtDhIkStatsWithTorchNode : public rclcpp::Node {
public:
  SceneSpaceRrtDhIkStatsWithTorchNode() : Node("scene_spacerrt_dh_ik_stats_withtorch") {
    frame_id_ = declare_parameter<std::string>("frame_id", "world");
    scene_file_ = declare_parameter<std::string>("scene_file", "");
    out_dir_ = declare_parameter<std::string>("out_dir", "/tmp/scene_spacerrt_stats_withtorch");

    auto lb = declare_parameter<std::vector<double>>("domain_lb", {-1.0,-1.0,0.0});
    auto ub = declare_parameter<std::vector<double>>("domain_ub", { 1.0, 1.0,1.0});
    lb_ = Eigen::Vector3d(lb[0], lb[1], lb[2]);
    ub_ = Eigen::Vector3d(ub[0], ub[1], ub[2]);
    domain_ = HPolyhedron::MakeBox(lb_, ub_);

    derive_xyz_from_q_ = declare_parameter<bool>("derive_xyz_from_q", true);
    auto s = declare_parameter<std::vector<double>>("start_xyz", {0.351, 0.341, 0.419});
    auto g = declare_parameter<std::vector<double>>("goal_xyz", {-0.200, 0.724, 0.280});
    start_xyz_ = Eigen::Vector3d(s[0], s[1], s[2]);
    goal_xyz_  = Eigen::Vector3d(g[0], g[1], g[2]);

    max_iter_ = declare_parameter<int>("max_iter", 200);
    scale_factor_ = declare_parameter<double>("scale_factor", 0.7);
    batch_trials_ = declare_parameter<int>("batch_trials", 50);
    rng_seed_ = declare_parameter<int>("rng_seed", 0);
    save_q_in_degree_ = declare_parameter<bool>("save_q_in_degree", true);

    enable_intersection_ik_ = declare_parameter<bool>("enable_intersection_ik", true);
    enable_joint_interp_    = declare_parameter<bool>("enable_joint_interp", false);
    joint_interp_steps_     = declare_parameter<int>("joint_interp_steps", 5);

    auto q_start_deg = declare_parameter<std::vector<double>>("q_start_deg", {27, -90.0, 90.0, 0.0, 75.6, 0.0});
    auto q_goal_deg  = declare_parameter<std::vector<double>>("q_goal_deg",  {97, -46.8, 50.4, -3.6, 99.0, 0.0});
    for (int i=0;i<6;++i) {
      q_start_(i) = q_start_deg[i] * M_PI / 180.0;
      q_goal_(i)  = q_goal_deg[i]  * M_PI / 180.0;
    }

    double align_yaw_deg = declare_parameter<double>("align_yaw_deg", 180.0);
    auto align_xyz = declare_parameter<std::vector<double>>("align_xyz", {0.0,0.0,0.0});
    T_align_.setIdentity();
    T_align_.linear() = Eigen::AngleAxisd(align_yaw_deg * M_PI / 180.0, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    T_align_.translation() = Eigen::Vector3d(align_xyz[0], align_xyz[1], align_xyz[2]);

    // torch geometry: match qp_traj_opt
    torch_enable_   = declare_parameter<bool>("torch_enable", true);
    torch_cyl1_len_ = declare_parameter<double>("torch_cyl1_len", 0.36);
    torch_cyl1_dia_ = declare_parameter<double>("torch_cyl1_dia", 0.025);
    torch_tilt_deg_ = declare_parameter<double>("torch_tilt_deg", 45.0);
    torch_cyl2_len_ = declare_parameter<double>("torch_cyl2_len", 0.11);
    torch_cyl2_dia_ = declare_parameter<double>("torch_cyl2_dia", 0.018);

    ik_opt_.max_iters  = declare_parameter<int>("ik_max_iters", 200);
    ik_opt_.pos_tol    = declare_parameter<double>("ik_pos_tol", 1e-4);
    ik_opt_.step_limit = declare_parameter<double>("ik_step_limit", 0.2);
    ik_opt_.lambda     = declare_parameter<double>("ik_lambda", 3e-2);

    topic_ = declare_parameter<std::string>("topic", "/scene_spacerrt_stats_withtorch_markers");
    pub_ = create_publisher<MarkerArray>(topic_, rclcpp::QoS(rclcpp::KeepLast(10)).reliable().transient_local());

    fs::create_directories(out_dir_);

    if (!ParseSceneFile(scene_file_, scene_, get_logger())) {
      RCLCPP_FATAL(get_logger(), "Scene parse failed.");
      rclcpp::shutdown();
      return;
    }

    build_obstacles_from_scene_();
    dh_ = MakeUr5DH();

    if (derive_xyz_from_q_) {
      std::array<Eigen::Isometry3d,7> Tj_start, Tj_goal;
      fk_all(dh_, q_start_, Tj_start);
      fk_all(dh_, q_goal_,  Tj_goal);

      start_xyz_ = torch_tip_world_pos(
          T_align_, Tj_start,
          torch_cyl1_len_, torch_tilt_deg_, torch_cyl2_len_);

      goal_xyz_  = torch_tip_world_pos(
          T_align_, Tj_goal,
          torch_cyl1_len_, torch_tilt_deg_, torch_cyl2_len_);

      RCLCPP_INFO(get_logger(),
                  "derive_xyz_from_q=true -> torch start=(%.6f, %.6f, %.6f), torch goal=(%.6f, %.6f, %.6f)",
                  start_xyz_.x(), start_xyz_.y(), start_xyz_.z(),
                  goal_xyz_.x(), goal_xyz_.y(), goal_xyz_.z());
    }

    RCLCPP_INFO(get_logger(),
                "torch params: cyl1_len=%.6f dia1=%.6f tilt_deg=%.3f cyl2_len=%.6f dia2=%.6f",
                torch_cyl1_len_, torch_cyl1_dia_, torch_tilt_deg_,
                torch_cyl2_len_, torch_cyl2_dia_);

    run_batch_();
    timer_ = create_wall_timer(std::chrono::milliseconds(500), [this](){ publish_last_markers_(); });
  }

private:
  void build_obstacles_from_scene_() {
    obstacles_.clear();
    obstacle_boxes_vertices_.clear();

    for (const auto& b : scene_.boxes) {
      auto X = BoxVertices(b);
      obstacle_boxes_vertices_.push_back(X);
      obstacles_.emplace_back(std::make_unique<VPolytope>(X));
    }
    for (const auto& s : scene_.spheres) {
      obstacles_.emplace_back(MakeSpherePolyApprox(s.pos, s.radius, 64));
    }
    RCLCPP_INFO(get_logger(), "Obstacle sets built: convex_sets=%zu", obstacles_.size());
  }

  size_t add_node_(Tree& T, const HPolyhedron& H) {
    PolyNode node;
    node.H = H;
    node.V = VerticesFromH3D(H);
    node.center = node.V.empty() ? Eigen::Vector3d::Zero() : CentroidFromVerts(node.V);
    T.nodes.push_back(std::move(node));
    const size_t N = T.nodes.size();
    T.adj.resize(N);
    for (auto& row : T.adj) row.resize(N, 0);
    if (N >= 2) {
      const size_t id = N - 1;
      for (size_t j=0; j+1<N; ++j) {
        if (IntersectIfNonempty(T.nodes[id].H, T.nodes[j].H)) {
          T.adj[id][j] = T.adj[j][id] = 1;
        }
      }
    }
    return N - 1;
  }

  std::optional<size_t> expand_once_(Tree& T, std::mt19937& rng) {
    const Eigen::Vector3d q = GuidedSampleTowardCenter(T, lb_, ub_, scale_factor_, rng);
    HPolyhedron H;
    try {
      H = Iris(obstacles_, q, domain_);
    } catch (const std::exception& e) {
      RCLCPP_WARN(get_logger(), "IRIS exception: %s", e.what());
      return std::nullopt;
    }
    const auto V = VerticesFromH3D(H);
    if (V.size() < 4 || SafeIsFlat(V)) return std::nullopt;
    return add_node_(T, H);
  }

  static std::vector<int> BfsPath(const std::vector<std::vector<int>>& adj, int s, int t) {
    std::vector<int> prev(adj.size(), -1);
    std::queue<int> q; q.push(s); prev[s] = -2;
    while (!q.empty()) {
      int u = q.front(); q.pop();
      if (u == t) break;
      for (size_t v=0; v<adj.size(); ++v) {
        if (adj[u][v] && prev[v] == -1) { prev[v] = u; q.push((int)v); }
      }
    }
    if (prev[t] == -1) return {};
    std::vector<int> path;
    for (int v=t; v!=-2; v=prev[v]) path.push_back(v);
    std::reverse(path.begin(), path.end());
    return path;
  }

  bool plan_once_(int trial_idx,
                  double& time_sec,
                  double& corridor_volume,
                  int& poly_count,
                  std::vector<HPolyhedron>* final_polys,
                  std::vector<Eigen::Matrix<double,6,1>, Eigen::aligned_allocator<Eigen::Matrix<double,6,1>>>* path_q) {
    std::mt19937 rng(rng_seed_ == 0 ? (unsigned)std::random_device{}() + trial_idx : (unsigned)rng_seed_ + trial_idx);

    auto t0 = std::chrono::steady_clock::now();

    Tree Ts, Tg;
    HPolyhedron P_start = Iris(obstacles_, start_xyz_, domain_);
    HPolyhedron P_goal  = Iris(obstacles_, goal_xyz_, domain_);
    add_node_(Ts, P_start);
    add_node_(Tg, P_goal);

    bool connected = false;
    int conn_s = -1, conn_g = -1;
    if (IntersectIfNonempty(Ts.nodes[0].H, Tg.nodes[0].H)) {
      connected = true; conn_s = 0; conn_g = 0;
    }

    for (int it=1; !connected && it<=max_iter_; ++it) {
      auto id_s = expand_once_(Ts, rng);
      auto id_g = expand_once_(Tg, rng);
      if (id_s) {
        for (size_t j=0; j<Tg.nodes.size(); ++j) {
          if (IntersectIfNonempty(Ts.nodes[*id_s].H, Tg.nodes[j].H)) {
            connected = true; conn_s = (int)*id_s; conn_g = (int)j; break;
          }
        }
      }
      if (!connected && id_g) {
        for (size_t i=0; i<Ts.nodes.size(); ++i) {
          if (IntersectIfNonempty(Ts.nodes[i].H, Tg.nodes[*id_g].H)) {
            connected = true; conn_s = (int)i; conn_g = (int)*id_g; break;
          }
        }
      }
    }

    if (!connected) {
      auto t1 = std::chrono::steady_clock::now();
      time_sec = std::chrono::duration<double>(t1 - t0).count();
      corridor_volume = 0.0;
      poly_count = 0;
      return false;
    }

    auto path_s = BfsPath(Ts.adj, 0, conn_s);
    auto path_g = BfsPath(Tg.adj, 0, conn_g);
    std::reverse(path_g.begin(), path_g.end());

    final_polys->clear();
    for (int idx : path_s) final_polys->push_back(Ts.nodes[idx].H);
    for (int idx : path_g) final_polys->push_back(Tg.nodes[idx].H);

    poly_count = static_cast<int>(final_polys->size());
    corridor_volume = ComputeCorridorVolumeExact(*final_polys);

    std::vector<Eigen::Vector3d> used_samples;
    *path_q = BuildSeedPathLikeCorridorViz(
        *final_polys,
        start_xyz_,
        goal_xyz_,
        dh_,
        T_align_,
        q_start_,
        q_goal_,
        ik_opt_,
        rng,
        enable_intersection_ik_,
        enable_joint_interp_,
        joint_interp_steps_,
        torch_cyl1_len_,
        torch_tilt_deg_,
        torch_cyl2_len_,
        &used_samples);

    auto t1 = std::chrono::steady_clock::now();
    time_sec = std::chrono::duration<double>(t1 - t0).count();
    return true;
  }

  void run_batch_() {
    int success_count = 0;
    double sum_time = 0.0;
    double sum_vol = 0.0;
    double sum_poly = 0.0;

    std::ofstream csv((fs::path(out_dir_) / "summary.csv").string());
    csv << "trial,success,time_sec,corridor_volume,poly_count\n";

    for (int t=0; t<batch_trials_; ++t) {
      std::vector<HPolyhedron> polys;
      std::vector<Eigen::Matrix<double,6,1>, Eigen::aligned_allocator<Eigen::Matrix<double,6,1>>> path_q;
      double time_sec = 0.0, volume = 0.0;
      int poly_count = 0;
      bool ok = plan_once_(t, time_sec, volume, poly_count, &polys, &path_q);
      csv << t << "," << (ok?1:0) << "," << time_sec << "," << volume << "," << poly_count << "\n";

      if (ok) {
        ++success_count;
        sum_time += time_sec;
        sum_vol += volume;
        sum_poly += poly_count;
        last_polys_ = polys;
        last_path_q_ = path_q;

        char buf_corr[512], buf_ik[512];
        std::snprintf(buf_corr, sizeof(buf_corr), "%s/run_%03d_corridor.txt", out_dir_.c_str(), t);
        std::snprintf(buf_ik,   sizeof(buf_ik),   "%s/run_%03d_ik.csv",      out_dir_.c_str(), t);
        const bool ok_corr = SaveCorridorTxt(buf_corr, frame_id_, start_xyz_, goal_xyz_, polys);
        const bool ok_ik   = SaveJointPathCsv(buf_ik, path_q, save_q_in_degree_, 6);
        if (!ok_corr || !ok_ik) {
          RCLCPP_WARN(get_logger(), "Save run files failed: corridor=%d ik=%d", ok_corr?1:0, ok_ik?1:0);
        }
      }
      RCLCPP_INFO(get_logger(), "trial=%d success=%d time=%.4f vol=%.6f polys=%d",
                  t, ok?1:0, time_sec, volume, poly_count);
    }
    csv.close();

    const double success_rate = batch_trials_ > 0 ? double(success_count) / double(batch_trials_) : 0.0;
    const double mean_time_all = batch_trials_ > 0 ? sum_time / batch_trials_ : 0.0;
    const double mean_vol_success  = success_count > 0 ? sum_vol / success_count  : 0.0;
    const double mean_poly_success = success_count > 0 ? sum_poly / success_count : 0.0;

    std::ofstream txt((fs::path(out_dir_) / "metrics.txt").string());
    txt << std::fixed << std::setprecision(6);
    txt << "Success rate: " << success_rate << "\n";
    txt << "Average time per run: " << mean_time_all << "\n";
    txt << "Average corridor volume sum (successful runs): " << mean_vol_success << "\n";
    txt << "Average corridor polytope count (successful runs): " << mean_poly_success << "\n";
    txt.close();

    RCLCPP_INFO(get_logger(), "==============================");
    RCLCPP_INFO(get_logger(), "Success rate = %.4f", success_rate);
    RCLCPP_INFO(get_logger(), "Avg time/run = %.6f s", mean_time_all);
    RCLCPP_INFO(get_logger(), "Avg corridor volume sum = %.6f", mean_vol_success);
    RCLCPP_INFO(get_logger(), "Avg corridor poly count = %.2f", mean_poly_success);
  }

  void publish_last_markers_() {
    MarkerArray ma;
    int id = 0;

    for (const auto& X : obstacle_boxes_vertices_) {
      Marker m = MakeLine(frame_id_, "scene_box", id++, 0.004, 0.9, 0.1, 0.1, 1.0);
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
      AddEdge(&m,p000,p100); AddEdge(&m,p100,p110); AddEdge(&m,p110,p010); AddEdge(&m,p010,p000);
      AddEdge(&m,p001,p101); AddEdge(&m,p101,p111); AddEdge(&m,p111,p011); AddEdge(&m,p011,p001);
      AddEdge(&m,p000,p001); AddEdge(&m,p100,p101); AddEdge(&m,p110,p111); AddEdge(&m,p010,p011);
      ma.markers.push_back(m);
    }
    for (const auto& s : scene_.spheres) {
      ma.markers.push_back(MakeSphere(frame_id_, "scene_sphere", id++, s.pos, s.radius, 0.2, 1.0, 0.2, 0.35));
    }

    ma.markers.push_back(MakeSphere(frame_id_, "seed", id++, start_xyz_, 0.01, 1.0, 0.8, 0.1, 1.0));
    ma.markers.push_back(MakeSphere(frame_id_, "seed", id++, goal_xyz_,  0.01, 0.1, 0.9, 0.3, 1.0));

    for (size_t i=0;i<last_polys_.size();++i) {
      auto V = VerticesFromH3D(last_polys_[i]);
      auto E = EdgesFromActiveSets(last_polys_[i], V);
      Marker m = MakeLine(frame_id_, "corridor", id++, 0.002, 0.1, 0.5, 1.0, 1.0);
      for (auto& e : E) AddEdge(&m, V[e.first], V[e.second]);
      ma.markers.push_back(m);
    }

    for (size_t t=0; t<last_path_q_.size(); ++t) {
      std::array<Eigen::Isometry3d,7> Tj;
      fk_all(dh_, last_path_q_[t], Tj);

      Eigen::Isometry3d T_world_tcp = tcp_world_pose(T_align_, Tj);

      Eigen::Vector3d p_tcp, p2, p_tip;
      compute_torch_keypoints(
          T_world_tcp,
          torch_cyl1_len_, torch_tilt_deg_, torch_cyl2_len_,
          p_tcp, p2, p_tip);

      ma.markers.push_back(MakeSphere(frame_id_, "tcp", id++, p_tcp, 0.008, 0.8, 0.8, 0.8, 0.60));
      ma.markers.push_back(MakeSphere(frame_id_, "torch_joint", id++, p2, 0.009, 1.0, 0.6, 0.0, 0.85));
      ma.markers.push_back(MakeSphere(frame_id_, "torch_tip", id++, p_tip, 0.012, 0.1, 0.9, 0.9, 0.95));

      Marker link1 = MakeLine(frame_id_, "torch_seg1", id++, 0.0020, 0.9, 0.4, 0.1, 0.90);
      AddEdge(&link1, p_tcp, p2);
      ma.markers.push_back(link1);

      Marker link2 = MakeLine(frame_id_, "torch_seg2", id++, 0.0020, 0.2, 0.8, 0.9, 0.90);
      AddEdge(&link2, p2, p_tip);
      ma.markers.push_back(link2);
    }

    pub_->publish(ma);
  }

private:
  std::string frame_id_;
  std::string scene_file_;
  std::string out_dir_;
  std::string topic_;
  int batch_trials_{50};
  int max_iter_{200};
  int rng_seed_{0};
  double scale_factor_{0.7};

  SceneData scene_;
  std::vector<Eigen::Matrix3Xd> obstacle_boxes_vertices_;

  Eigen::Vector3d lb_, ub_;
  HPolyhedron domain_;
  ConvexSets obstacles_;

  Eigen::Vector3d start_xyz_, goal_xyz_;
  bool derive_xyz_from_q_{true};

  std::vector<DH> dh_;
  bool save_q_in_degree_{true};
  bool enable_intersection_ik_{true};
  bool enable_joint_interp_{false};
  int joint_interp_steps_{5};

  bool   torch_enable_{true};
  double torch_cyl1_len_{0.36};
  double torch_cyl1_dia_{0.025};
  double torch_tilt_deg_{45.0};
  double torch_cyl2_len_{0.11};
  double torch_cyl2_dia_{0.018};

  Eigen::Isometry3d T_align_{Eigen::Isometry3d::Identity()};
  Eigen::Matrix<double,6,1> q_start_{}, q_goal_{};
  IkOptions ik_opt_;

  std::vector<HPolyhedron> last_polys_;
  std::vector<Eigen::Matrix<double,6,1>, Eigen::aligned_allocator<Eigen::Matrix<double,6,1>>> last_path_q_;

  rclcpp::Publisher<MarkerArray>::SharedPtr pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SceneSpaceRrtDhIkStatsWithTorchNode>());
  rclcpp::shutdown();
  return 0;
}
