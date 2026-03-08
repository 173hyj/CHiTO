#include <chrono>
#include <fstream>
#include <optional>
#include <random>
#include <string>
#include <vector>
#include <algorithm>
#include <filesystem>
#include <sstream>
#include <array>
#include <limits>
#include <queue>
#include <cmath>

namespace fs = std::filesystem;

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "rclcpp/rclcpp.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"

#include <yaml-cpp/yaml.h>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit/planning_scene_monitor/planning_scene_monitor.h>
#include <moveit/collision_detection/collision_common.h>

#include "drake/geometry/optimization/hpolyhedron.h"
#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/solve.h"

using visualization_msgs::msg::Marker;
using visualization_msgs::msg::MarkerArray;
using geometry_msgs::msg::Point;
using drake::geometry::optimization::HPolyhedron;
namespace mp = drake::solvers;

static inline double rad2deg(double x){ return x * 180.0 / M_PI; }
static inline double deg2rad(double x){ return x * M_PI / 180.0; }
static inline std::vector<double> vec_rad2deg(const std::vector<double>& v){
  std::vector<double> out; out.reserve(v.size());
  for(double x: v) out.push_back(rad2deg(x));
  return out;
}
static inline std::vector<double> vec_deg2rad(const std::vector<double>& v){
  std::vector<double> out; out.reserve(v.size());
  for(double x: v) out.push_back(deg2rad(x));
  return out;
}

static double weightedJointDist2(const std::vector<double>& a,
                                 const std::vector<double>& b,
                                 const std::vector<double>& w) {
  if (a.size()!=b.size()) return 1e18;
  double s = 0.0;
  for (size_t i=0;i<a.size();++i) {
    const double wi = (i < w.size()) ? w[i] : 1.0;
    const double d  = a[i] - b[i];
    s += wi * d * d;
  }
  return s;
}

static bool closeQ(const std::vector<double>& a,
                   const std::vector<double>& b,
                   double tol_norm = 1e-2,
                   double tol_max  = 5e-3) {
  if (a.size() != b.size()) return false;
  double s = 0.0;
  double m = 0.0;
  for (size_t i=0;i<a.size();++i) {
    const double d = std::abs(a[i]-b[i]);
    s += d*d;
    m = std::max(m, d);
  }
  return std::sqrt(s) <= tol_norm || m <= tol_max;
}

static Marker MakeLine(const std::string& frame, const std::string& ns, int id,
                       double w, double r, double g, double b, double a=1.0) {
  Marker m; m.header.frame_id=frame; m.ns=ns; m.id=id;
  m.type=Marker::LINE_LIST; m.action=Marker::ADD;
  m.scale.x=w; m.color.r=r; m.color.g=g; m.color.b=b; m.color.a=a;
  m.pose.orientation.w=1.0; return m;
}
static Marker MakeSphere(const std::string& frame, const std::string& ns, int id,
                         const Eigen::Vector3d& c, double rad,
                         double r, double g, double b, double a=1.0) {
  Marker m; m.header.frame_id=frame; m.ns=ns; m.id=id;
  m.type=Marker::SPHERE; m.action=Marker::ADD;
  m.pose.position.x=c.x(); m.pose.position.y=c.y(); m.pose.position.z=c.z();
  m.pose.orientation.w=1.0; m.scale.x=m.scale.y=m.scale.z=2*rad;
  m.color.r=r; m.color.g=g; m.color.b=b; m.color.a=a; return m;
}

static Marker makeCube(const std::string& frame, const std::string& ns, int id,
                       const Eigen::Isometry3d& T, const Eigen::Vector3d& size,
                       double r, double g, double b, double a) {
  Marker m; m.header.frame_id = frame; m.ns = ns; m.id = id;
  m.type = Marker::CUBE; m.action = Marker::ADD;
  m.pose.position.x = T.translation().x();
  m.pose.position.y = T.translation().y();
  m.pose.position.z = T.translation().z();
  Eigen::Quaterniond q(T.rotation());
  m.pose.orientation.x = q.x();
  m.pose.orientation.y = q.y();
  m.pose.orientation.z = q.z();
  m.pose.orientation.w = q.w();
  m.scale.x = size.x(); m.scale.y = size.y(); m.scale.z = size.z();
  m.color.r = r; m.color.g = g; m.color.b = b; m.color.a = a; return m;
}
static Marker makeLinkBoxBetween(const std::string& frame, const std::string& ns, int id,
                                 const Eigen::Vector3d& p1, const Eigen::Vector3d& p2,
                                 double box_w, double r, double g, double b, double a) {
  Eigen::Vector3d z = p2 - p1; double L = z.norm();
  if (L < 1e-9) { z = Eigen::Vector3d::UnitZ(); L = 1e-6; } else { z.normalize(); }
  Eigen::Vector3d x = (std::fabs(z.x()) > 0.9) ? Eigen::Vector3d::UnitY() : Eigen::Vector3d::UnitX();
  Eigen::Vector3d y = z.cross(x).normalized(); x = y.cross(z).normalized();
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.linear().col(0)=x; T.linear().col(1)=y; T.linear().col(2)=z;
  T.translation() = 0.5*(p1+p2);
  Eigen::Vector3d size(box_w, box_w, L);
  return makeCube(frame, ns, id, T, size, r,g,b,a);
}
static void AddEdge(Marker* m, const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
  Point pa; pa.x=a.x(); pa.y=a.y(); pa.z=a.z();
  Point pb; pb.x=b.x(); pb.y=b.y(); pb.z=b.z();
  m->points.push_back(pa); m->points.push_back(pb);
}
static void AddAxes(MarkerArray& ma, int& id, const std::string& frame, const std::string& ns,
                    const Eigen::Isometry3d& T, double len, double rad) {
  auto add_arrow = [&](const Eigen::Vector3d& dir, double r, double g, double b){
    Marker a; a.header.frame_id = frame; a.ns = ns; a.id = id++;
    a.type = Marker::ARROW; a.action = Marker::ADD;
    a.scale.x = rad; a.scale.y = 2.0*rad; a.scale.z = 0.25*len;
    a.color.r = r; a.color.g = g; a.color.b = b; a.color.a = 1.0;
    geometry_msgs::msg::Point p0, p1;
    Eigen::Vector3d P0 = T.translation();
    Eigen::Vector3d P1 = P0 + T.linear()*dir*len;
    p0.x=P0.x(); p0.y=P0.y(); p0.z=P0.z();
    p1.x=P1.x(); p1.y=P1.y(); p1.z=P1.z();
    a.points.push_back(p0); a.points.push_back(p1);
    ma.markers.push_back(a);
  };
  add_arrow(Eigen::Vector3d::UnitX(), 1.0,0.2,0.2);
  add_arrow(Eigen::Vector3d::UnitY(), 0.2,1.0,0.2);
  add_arrow(Eigen::Vector3d::UnitZ(), 0.2,0.4,1.0);
}
static void AddText(MarkerArray& ma, int& id, const std::string& frame, const std::string& ns,
                    const Eigen::Vector3d& p, const std::string& text, double scale=0.035) {
  Marker m; m.header.frame_id=frame; m.ns=ns; m.id=id++;
  m.type=Marker::TEXT_VIEW_FACING; m.action=Marker::ADD;
  m.pose.position.x=p.x(); m.pose.position.y=p.y(); m.pose.position.z=p.z();
  m.scale.z=scale; m.color.r=1.0; m.color.g=1.0; m.color.b=1.0; m.color.a=0.95;
  m.text=text; ma.markers.push_back(m);
}

static std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>
VerticesFromH3D(const HPolyhedron& H, double feas_tol=1e-9, double dup_tol=1e-8) {
  const auto& A = H.A(); const auto& b = H.b(); const int m = A.rows();
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> V;
  for (int i=0;i<m;++i) for (int j=i+1;j<m;++j) for (int k=j+1;k<m;++k) {
    Eigen::Matrix3d M; M.row(0)=A.row(i); M.row(1)=A.row(j); M.row(2)=A.row(k);
    double det=M.determinant(); if (std::abs(det)<1e-12) continue;
    Eigen::Vector3d rhs(b(i),b(j),b(k));
    Eigen::Vector3d x = M.fullPivLu().solve(rhs);
    if ((A*x - b).maxCoeff() <= feas_tol) {
      bool dup=false; for (const auto& q:V) if ((x-q).norm()<=dup_tol){dup=true;break;}
      if (!dup) V.push_back(x);
    }
  }
  return V;
}
static std::vector<std::pair<int,int>>
EdgesFromActiveSets(const HPolyhedron& H,
                    const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>& V,
                    double tol=1e-7) {
  const auto& A = H.A(); const auto& B = H.b();
  const int m=A.rows(); const int n=(int)V.size();
  std::vector<std::vector<int>> act(n);
  for (int i=0;i<n;++i)
    for (int r=0;r<m;++r)
      if (std::abs(A.row(r).dot(V[i]) - B(r)) <= tol) act[i].push_back(r);
  auto share_two = [&](int i,int j)->bool{
    int c=0; for(int r1:act[i]) for(int r2:act[j]) if(r1==r2) if(++c>=2) return true;
    return false;
  };
  std::vector<std::pair<int,int>> E;
  for (int i=0;i<n;++i) for (int j=i+1;j<n;++j) {
    if ((V[i]-V[j]).norm()<=1e-9) continue;
    if (share_two(i,j)) E.emplace_back(i,j);
  }
  return E;
}
static std::optional<Eigen::Vector3d> ChebyshevPoint(const HPolyhedron& H) {
  const auto& A = H.A(); const auto& b = H.b();
  mp::MathematicalProgram prog;
  auto x = prog.NewContinuousVariables<3>("x");
  auto r = prog.NewContinuousVariables<1>("r")(0);
  Eigen::VectorXd Anorm(A.rows());
  for (int i=0;i<A.rows();++i) Anorm(i)=A.row(i).norm();
  for (int i=0;i<A.rows();++i)
    prog.AddLinearConstraint(A.row(i)*x + Anorm(i)*r <= b(i));
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
    double s = 0.0;
    for (auto& wi : w) { wi = std::max(1e-9, gamma(rng)); s += wi; }
    for (auto& wi : w) wi /= s;
    Eigen::Vector3d x = Eigen::Vector3d::Zero();
    for (size_t i=0;i<V->size();++i) x += w[i] * (*V)[i];
    return x;
  }
  Eigen::Vector3d c = Eigen::Vector3d::Zero();
  if (auto cp = ChebyshevPoint(H)) c = *cp;
  else {
    const auto& A=H.A(); const auto& b=H.b();
    mp::MathematicalProgram prog;
    auto x = prog.NewContinuousVariables<3>("x");
    prog.AddLinearConstraint(A*x <= b);
    prog.AddQuadraticCost(Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero(), x);
    auto res = mp::Solve(prog);
    if (res.is_success()) c = res.GetSolution(x);
  }
  std::normal_distribution<double> norm01(0.0, 1.0);
  Eigen::Vector3d y = c + 0.1 * Eigen::Vector3d(norm01(rng), norm01(rng), norm01(rng));
  const auto& A=H.A(); const auto& b=H.b();
  mp::MathematicalProgram prog;
  auto x = prog.NewContinuousVariables<3>("x");
  prog.AddLinearConstraint(A*x <= b);
  Eigen::Matrix3d Q = Eigen::Matrix3d::Identity();
  prog.AddQuadraticCost(Q, -Q*y, x);
  auto res = mp::Solve(prog);
  if (res.is_success()) return res.GetSolution(x);
  return c;
}

struct CorridorData {
  std::string frame_id{"world"};
  std::vector<HPolyhedron> interleaved;
  std::vector<HPolyhedron> polys;
  std::vector<HPolyhedron> inters;
};
static HPolyhedron ParseH(const YAML::Node& n) {
  const auto& A_node = n["A"]; const auto& b_node = n["b"];
  if (!A_node || !b_node) throw std::runtime_error("H node missing A/b");
  int m = (int)A_node.size(); int ncols = (int)A_node[0].size();
  Eigen::MatrixXd A(m, ncols); Eigen::VectorXd b(m);
  for (int i=0;i<m;++i) {
    for (int j=0;j<ncols;++j) A(i,j)=A_node[i][j].as<double>();
    b(i)=b_node[i].as<double>();
  }
  return HPolyhedron(A,b);
}
static bool LoadCorridorYaml(const std::string& file, CorridorData& out) {
  YAML::Node root = YAML::LoadFile(file);
  if (root["frame_id"]) out.frame_id = root["frame_id"].as<std::string>();
  if (root["corridor_interleaved"]) for (auto h : root["corridor_interleaved"]) out.interleaved.push_back(ParseH(h));
  if (root["final_path_polys"])    for (auto h : root["final_path_polys"])    out.polys.push_back(ParseH(h));
  if (root["final_path_inters"])   for (auto h : root["final_path_inters"])   out.inters.push_back(ParseH(h));
  return !out.interleaved.empty() || !out.polys.empty();
}

struct IKCandidate {
  std::vector<double> q;
  Eigen::Vector3d tcp_xyz{0,0,0};
  Eigen::Vector3d rpy_deg{0,0,0};
  bool collision_free{false};
  int source_segment{-1};
  int resample_id{0};
  int ori_id{0};
};

struct InitIKResult {
  bool success{false};
  std::vector<Eigen::Vector3d> anchors;
  std::vector<int> anchor_source;
  std::vector<std::vector<IKCandidate>> layers;
  std::vector<int> best_index_per_layer;
  std::vector<std::vector<double>> q_rep;
  std::vector<std::vector<double>> q_seed_dense;
  std::vector<int> sigma_dense;
};

class CorridorVizMoveItInitIKNoTorch : public rclcpp::Node {
public:
  CorridorVizMoveItInitIKNoTorch(): Node("corridor_viz_moveit_initik_no_torch") {
    pub_ = this->create_publisher<MarkerArray>("/corridor_initik_markers", 1);
    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

    frame_id_      = declare_parameter<std::string>("frame_id", "world");
    group_name_    = declare_parameter<std::string>("group_name", "arm");
    tip_link_      = declare_parameter<std::string>("tip_link", "tool0");
    corridor_files_str_ = declare_parameter<std::string>("corridor_files", "");
    select_index_  = declare_parameter<int>("select_index", -1);
    run_all_       = declare_parameter<bool>("run_all", false);

    q_start_ = declare_parameter<std::vector<double>>("q_start", std::vector<double>{});
    q_goal_  = declare_parameter<std::vector<double>>("q_goal",  std::vector<double>{});

    ik_timeout_       = declare_parameter<double>("ik_timeout", 0.05);
    ik_attempts_      = declare_parameter<int>("ik_attempts", 8);
    target_rpy_deg_   = declare_parameter<std::vector<double>>("target_rpy_deg", {0.0,180.0,0.0});
    ik_check_collision_ = declare_parameter<bool>("ik_check_collision", true);

    yaw_sweep_deg_    = declare_parameter<double>("yaw_sweep_deg", 120.0);
    yaw_step_deg_     = declare_parameter<double>("yaw_step_deg", 5.0);
    roll_tol_deg_     = declare_parameter<double>("roll_tol_deg", 0.0);
    roll_step_deg_    = declare_parameter<double>("roll_step_deg", 0.0);
    pitch_tol_deg_    = declare_parameter<double>("pitch_tol_deg", 0.0);
    pitch_step_deg_   = declare_parameter<double>("pitch_step_deg", 0.0);
    max_resample_per_segment_ = declare_parameter<int>("max_resample_per_segment", 8);
    allow_colliding_fallback_ = declare_parameter<bool>("allow_colliding_fallback", true);
    rng_seed_         = declare_parameter<int>("rng_seed", 0);

    q_param_in_degree_ = declare_parameter<bool>("q_param_in_degree", true);
    save_q_in_degree_  = declare_parameter<bool>("save_q_in_degree",  true);

    use_intersection_anchors_ = declare_parameter<bool>("use_intersection_anchors", true);
    candidates_per_segment_   = declare_parameter<int>("candidates_per_segment", 8);
    ik_seed_trials_per_pose_  = declare_parameter<int>("ik_seed_trials_per_pose", 4);
    interpolate_per_pair_     = declare_parameter<int>("interpolate_per_pair", 3);
    ik_duplicate_tol_norm_    = declare_parameter<double>("ik_duplicate_tol_norm", 1e-2);
    ik_duplicate_tol_max_     = declare_parameter<double>("ik_duplicate_tol_max", 5e-3);
    joint_cost_weights_       = declare_parameter<std::vector<double>>("joint_cost_weights", std::vector<double>{});

    line_w_ = declare_parameter<double>("line_width", 0.002);
    pt_r_   = declare_parameter<double>("sample_point_r", 0.01);

    link_box_w_ = declare_parameter<double>("link_box_width", 0.025);
    link_rgba_  = declare_parameter<std::vector<double>>("link_rgba", {0.20,0.80,0.25,0.65});
    if (link_rgba_.size()!=4) link_rgba_={0.20,0.80,0.25,0.65};

    show_link_frames_start_ = declare_parameter<bool>("show_link_frames_start", true);
    show_link_frames_goal_  = declare_parameter<bool>("show_link_frames_goal",  true);
    show_link_names_start_  = declare_parameter<bool>("show_link_names_start", true);
    show_link_names_goal_   = declare_parameter<bool>("show_link_names_goal",  true);
    link_frame_len_ = declare_parameter<double>("link_frame_len", 0.08);
    link_frame_rad_ = declare_parameter<double>("link_frame_rad", 0.004);

    output_dir_ = declare_parameter<std::string>("output_dir", "");

    parseFiles_();
  }

  bool init() {
    if (rng_seed_ == 0) {
      std::seed_seq seed{(unsigned)std::chrono::high_resolution_clock::now().time_since_epoch().count()};
      rng_ = std::mt19937(seed);
    } else {
      rng_ = std::mt19937((unsigned)rng_seed_);
    }

    robot_model_loader::RobotModelLoader loader(this->shared_from_this(), "robot_description");
    robot_model_ = loader.getModel();
    if (!robot_model_) { RCLCPP_FATAL(get_logger(), "Failed to load RobotModel."); return false; }
    jmg_ = robot_model_->getJointModelGroup(group_name_);
    if (!jmg_) { RCLCPP_FATAL(get_logger(), "JointModelGroup '%s' not found.", group_name_.c_str()); return false; }
    robot_state_ = std::make_shared<moveit::core::RobotState>(robot_model_);
    robot_state_->setToDefaultValues();

    if (ik_check_collision_) {
      psm_ = std::make_shared<planning_scene_monitor::PlanningSceneMonitor>(shared_from_this(), "robot_description");
      if (!psm_ || !psm_->getPlanningScene()) {
        RCLCPP_WARN(get_logger(), "PlanningSceneMonitor unavailable; collision check disabled.");
        ik_check_collision_ = false;
      } else {
        psm_->startSceneMonitor();
        psm_->startWorldGeometryMonitor();
        psm_->startStateMonitor();
        RCLCPP_INFO(get_logger(), "PlanningSceneMonitor started for IK collision checking.");
      }
    }

    if (q_param_in_degree_) {
      if (!q_start_.empty()) q_start_ = vec_deg2rad(q_start_);
      if (!q_goal_.empty())  q_goal_  = vec_deg2rad(q_goal_);
    }

    if (files_.empty()) { RCLCPP_ERROR(get_logger(), "No corridor_files."); return false; }
    if (joint_cost_weights_.empty()) joint_cost_weights_.assign(jmg_->getVariableCount(), 1.0);
    if (joint_cost_weights_.size() != jmg_->getVariableCount()) joint_cost_weights_.assign(jmg_->getVariableCount(), 1.0);

    if (run_all_ || select_index_ < 0) {
      int success = 0;
      for (size_t i=0;i<files_.size();++i) if (processOne_(i)) ++success;
      RCLCPP_INFO(get_logger(), "Batch done. success=%d / %zu", success, files_.size());
    } else {
      processOne_((size_t)select_index_);
    }

    timer_ = this->create_wall_timer(std::chrono::milliseconds(500), [this](){ publishMarkers_(); });
    return true;
  }

private:
  void parseFiles_() {
    files_.clear();
    auto trim = [](const std::string& x)->std::string{
      size_t a=x.find_first_not_of(" \t\r\n"); size_t b=x.find_last_not_of(" \t\r\n");
      if (a==std::string::npos) return std::string(); return x.substr(a,b-a+1);
    };
    std::stringstream ss(corridor_files_str_);
    std::string path;
    while (std::getline(ss, path, ';')) {
      path = trim(path);
      if (!path.empty()) files_.push_back(path);
    }
    RCLCPP_INFO(get_logger(), "Parsed %zu corridor files.", files_.size());
  }

  std::vector<Eigen::Vector3d> buildOrientationCandidatesDeg_() const {
    std::vector<Eigen::Vector3d> cand;
    if (target_rpy_deg_.size()!=3) return cand;
    const double r0 = target_rpy_deg_[0], p0 = target_rpy_deg_[1], y0 = target_rpy_deg_[2];

    auto arange = [](double a, double b, double step){
      std::vector<double> v;
      if (step <= 0) { v.push_back(0.0); return v; }
      for (double x=a; x<=b+1e-9; x+=step) v.push_back(x);
      return v;
    };

    auto yaw_list = arange(-yaw_sweep_deg_, yaw_sweep_deg_, std::max(1e-6, yaw_step_deg_));
    std::vector<double> roll_list{0.0}, pitch_list{0.0};
    if (roll_tol_deg_ > 0.0 && roll_step_deg_ > 0.0) roll_list = arange(-roll_tol_deg_, roll_tol_deg_, roll_step_deg_);
    if (pitch_tol_deg_ > 0.0 && pitch_step_deg_ > 0.0) pitch_list = arange(-pitch_tol_deg_, pitch_tol_deg_, pitch_step_deg_);

    cand.emplace_back(r0, p0, y0);
    for (double dy : yaw_list) {
      if (std::abs(dy) < 1e-9) continue;
      cand.emplace_back(r0, p0, y0 + dy);
    }
    for (double dr : roll_list) {
      for (double dp : pitch_list) {
        if (std::abs(dr)<1e-9 && std::abs(dp)<1e-9) continue;
        for (double dy : yaw_list) cand.emplace_back(r0+dr, p0+dp, y0+dy);
      }
    }
    return cand;
  }

  bool tryIKAtTipPose_(const geometry_msgs::msg::Pose& tip_pose,
                       std::vector<double>& qsol_out,
                       bool& collision_free_out,
                       const moveit::core::GroupStateValidityCallbackFn& validity) {
    for (int att=0; att<ik_attempts_; ++att) {
      bool ok = validity
        ? robot_state_->setFromIK(jmg_, tip_pose, tip_link_, ik_timeout_, validity)
        : robot_state_->setFromIK(jmg_, tip_pose, tip_link_, ik_timeout_);
      if (ok) {
        qsol_out.resize(jmg_->getVariableCount());
        robot_state_->copyJointGroupPositions(jmg_, qsol_out);
        collision_free_out = true;
        return true;
      }
    }
    if (allow_colliding_fallback_) {
      for (int att=0; att<ik_attempts_; ++att) {
        if (robot_state_->setFromIK(jmg_, tip_pose, tip_link_, ik_timeout_)) {
          qsol_out.resize(jmg_->getVariableCount());
          robot_state_->copyJointGroupPositions(jmg_, qsol_out);
          collision_free_out = false;
          return true;
        }
      }
    }
    return false;
  }

  bool solveAtTip_(const Eigen::Vector3d& tip_pos,
                   const Eigen::Vector3d& rpy_deg,
                   std::vector<double>& qsol_out,
                   bool& collision_free_out,
                   const moveit::core::GroupStateValidityCallbackFn& validity) {
    const double rr=rpy_deg.x()*M_PI/180.0, pp=rpy_deg.y()*M_PI/180.0, yy=rpy_deg.z()*M_PI/180.0;
    Eigen::Matrix3d R_tip =
      Eigen::AngleAxisd(yy, Eigen::Vector3d::UnitZ()).toRotationMatrix() *
      Eigen::AngleAxisd(pp, Eigen::Vector3d::UnitY()).toRotationMatrix() *
      Eigen::AngleAxisd(rr, Eigen::Vector3d::UnitX()).toRotationMatrix();

    geometry_msgs::msg::Pose tip_pose;
    tip_pose.position.x = tip_pos.x();
    tip_pose.position.y = tip_pos.y();
    tip_pose.position.z = tip_pos.z();
    Eigen::Quaterniond q(R_tip);
    tip_pose.orientation.x = q.x();
    tip_pose.orientation.y = q.y();
    tip_pose.orientation.z = q.z();
    tip_pose.orientation.w = q.w();

    return tryIKAtTipPose_(tip_pose, qsol_out, collision_free_out, validity);
  }

  std::vector<IKCandidate> generateCandidatesForSegment_(const HPolyhedron& sample_poly,
                                                         const Eigen::Vector3d& first_anchor,
                                                         int seg_idx,
                                                         const std::vector<Eigen::Vector3d>& ori_cands_deg,
                                                         const moveit::core::GroupStateValidityCallbackFn& validity,
                                                         const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>& V_cached) {
    std::vector<IKCandidate> out;
    auto push_unique = [&](const IKCandidate& c){
      for (const auto& e : out) {
        if (closeQ(e.q, c.q, ik_duplicate_tol_norm_, ik_duplicate_tol_max_)) return;
      }
      out.push_back(c);
    };

    std::vector<Eigen::Vector3d> tip_trials;
    tip_trials.push_back(first_anchor);
    for (int rs=0; rs<max_resample_per_segment_; ++rs) tip_trials.push_back(SamplePointIn(sample_poly, rng_, &V_cached));

    for (int t=0; t<(int)tip_trials.size(); ++t) {
      for (int oi=0; oi<(int)ori_cands_deg.size(); ++oi) {
        for (int k=0; k<ik_seed_trials_per_pose_; ++k) {
          std::vector<double> qtmp;
          bool cfree = false;
          bool ok = solveAtTip_(tip_trials[t], ori_cands_deg[oi], qtmp, cfree, validity);
          if (ok) {
            IKCandidate c;
            c.q = std::move(qtmp);
            c.tcp_xyz = tip_trials[t];
            c.rpy_deg = ori_cands_deg[oi];
            c.collision_free = cfree;
            c.source_segment = seg_idx;
            c.resample_id = t;
            c.ori_id = oi;
            push_unique(c);
            if ((int)out.size() >= candidates_per_segment_) return out;
          }
        }
      }
    }
    return out;
  }

  InitIKResult runInitIK_(const CorridorData& C) {
    InitIKResult R;
    const size_t N = C.polys.size();
    if (N == 0) return R;

    std::vector<std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>> V_poly(N);
    for (size_t i=0;i<N;++i) V_poly[i] = VerticesFromH3D(C.polys[i]);

    std::vector<Eigen::Vector3d> ori_cands_deg = buildOrientationCandidatesDeg_();

    moveit::core::GroupStateValidityCallbackFn validity;
    if (ik_check_collision_ && psm_ && psm_->getPlanningScene()) {
      validity = [this](moveit::core::RobotState* state,
                        const moveit::core::JointModelGroup* group,
                        const double* ik_solution)->bool {
        state->setJointGroupPositions(group, ik_solution);
        state->update();
        planning_scene_monitor::LockedPlanningSceneRO ls(psm_);
        collision_detection::CollisionRequest req;
        collision_detection::CollisionResult res;
        ls->checkCollision(req, res, *state);
        return !res.collision;
      };
    }

    R.anchors.resize(N, Eigen::Vector3d::Zero());
    R.anchor_source.assign(N, 0);
    R.layers.resize(N);
    R.best_index_per_layer.assign(N, -1);

    for (size_t i=0;i<N;++i) {
      Eigen::Vector3d anchor = Eigen::Vector3d::Zero();
      bool got_anchor = false;
      if (use_intersection_anchors_ && i < C.inters.size()) {
        auto cp = ChebyshevPoint(C.inters[i]);
        if (cp) {
          anchor = *cp;
          got_anchor = true;
          R.anchor_source[i] = 1;
        }
      }
      if (!got_anchor) {
        auto cp = ChebyshevPoint(C.polys[i]);
        if (cp) {
          anchor = *cp;
          got_anchor = true;
        } else {
          anchor = SamplePointIn(C.polys[i], rng_, &V_poly[i]);
          got_anchor = true;
        }
        R.anchor_source[i] = 0;
      }
      R.anchors[i] = anchor;
    }

    if (!q_start_.empty()) {
      IKCandidate c0; c0.q = q_start_; c0.tcp_xyz = R.anchors.front(); c0.collision_free = true; c0.source_segment = 0;
      R.layers.front().push_back(c0);
    }
    if (!q_goal_.empty()) {
      IKCandidate cN; cN.q = q_goal_; cN.tcp_xyz = R.anchors.back(); cN.collision_free = true; cN.source_segment = (int)N-1;
      R.layers.back().push_back(cN);
    }

    for (size_t i=0;i<N;++i) {
      const bool fixed_start = (i==0 && !q_start_.empty());
      const bool fixed_goal  = (i==N-1 && !q_goal_.empty());
      if (fixed_start || fixed_goal) continue;

      const HPolyhedron& sample_poly = (use_intersection_anchors_ && i < C.inters.size()) ? C.inters[i] : C.polys[i];
      auto cand = generateCandidatesForSegment_(sample_poly, R.anchors[i], (int)i, ori_cands_deg, validity, V_poly[i]);
      R.layers[i] = std::move(cand);
      if (R.layers[i].empty()) {
        RCLCPP_WARN(get_logger(), "InitIK layer %zu has no candidate.", i);
        return R;
      }
    }

    const int L = (int)N;
    std::vector<std::vector<double>> dp(L);
    std::vector<std::vector<int>> parent(L);
    for (int i=0;i<L;++i) {
      dp[i].assign(R.layers[i].size(), 1e18);
      parent[i].assign(R.layers[i].size(), -1);
    }
    for (size_t j=0;j<R.layers[0].size();++j) dp[0][j] = 0.0;

    for (int i=1;i<L;++i) {
      for (size_t b=0;b<R.layers[i].size();++b) {
        for (size_t a=0;a<R.layers[i-1].size();++a) {
          const double w = weightedJointDist2(R.layers[i-1][a].q, R.layers[i][b].q, joint_cost_weights_);
          const double penalty = (R.layers[i][b].collision_free ? 0.0 : 1e-3);
          const double v = dp[i-1][a] + w + penalty;
          if (v < dp[i][b]) {
            dp[i][b] = v;
            parent[i][b] = (int)a;
          }
        }
      }
    }

    int best_last = -1;
    double best_cost = 1e18;
    for (size_t j=0;j<dp[L-1].size();++j) {
      if (dp[L-1][j] < best_cost) {
        best_cost = dp[L-1][j];
        best_last = (int)j;
      }
    }
    if (best_last < 0) return R;

    R.q_rep.resize(L);
    int cur = best_last;
    for (int i=L-1;i>=0;--i) {
      R.best_index_per_layer[i] = cur;
      R.q_rep[i] = R.layers[i][cur].q;
      cur = parent[i][cur];
    }

    R.q_seed_dense.clear();
    R.sigma_dense.clear();
    if (!R.q_rep.empty()) {
      R.q_seed_dense.push_back(R.q_rep.front());
      R.sigma_dense.push_back(0);
      for (int i=0;i<L-1;++i) {
        const auto& qa = R.q_rep[i];
        const auto& qb = R.q_rep[i+1];
        for (int t=1; t<=interpolate_per_pair_; ++t) {
          double alpha = static_cast<double>(t) / static_cast<double>(interpolate_per_pair_ + 1);
          std::vector<double> qi(qa.size(), 0.0);
          for (size_t j=0;j<qa.size();++j) qi[j] = (1.0-alpha)*qa[j] + alpha*qb[j];
          R.q_seed_dense.push_back(qi);
          R.sigma_dense.push_back(i);
        }
        R.q_seed_dense.push_back(qb);
        R.sigma_dense.push_back(i+1);
      }
    }

    R.success = true;
    return R;
  }

  bool processOne_(size_t idx) {
    if (idx >= files_.size()) return false;
    const std::string& file = files_[idx];
    RCLCPP_INFO(get_logger(), "Loading corridor[%zu]: %s", idx, file.c_str());

    CorridorData C;
    if (!LoadCorridorYaml(file, C)) {
      RCLCPP_ERROR(get_logger(), "Failed to load corridor from %s", file.c_str());
      return false;
    }
    frame_id_ = C.frame_id;

    if (C.polys.empty()) {
      RCLCPP_ERROR(get_logger(), "No final_path_polys in corridor yaml.");
      return false;
    }

    initik_result_ = runInitIK_(C);
    if (!initik_result_.success) {
      RCLCPP_ERROR(get_logger(), "InitIK failed on %s", file.c_str());
      return false;
    }

    cached_lines_.clear();
    for (size_t i=0;i<C.interleaved.size();++i) {
      auto V = VerticesFromH3D(C.interleaved[i]);
      auto E = EdgesFromActiveSets(C.interleaved[i], V);
      cached_lines_.push_back({std::move(V), std::move(E)});
    }

    q_mid_list_.clear();
    used_sample_xyz_.clear();
    for (size_t i=0;i<initik_result_.layers.size();++i) {
      int best = initik_result_.best_index_per_layer[i];
      if (best >= 0 && best < (int)initik_result_.layers[i].size()) {
        q_mid_list_.push_back(initik_result_.layers[i][best].q);
        used_sample_xyz_.push_back(initik_result_.layers[i][best].tcp_xyz);
      } else {
        q_mid_list_.push_back({});
        used_sample_xyz_.push_back(initik_result_.anchors[i]);
      }
    }
    samples_ = initik_result_.anchors;

    q_paths5_.clear();
    if (!initik_result_.q_rep.empty()) {
      const size_t N = initik_result_.q_rep.size();
      size_t mid_idx = (N-1)/2;
      const auto& q_mid = initik_result_.q_rep[mid_idx];
      std::vector<double> q0 = initik_result_.q_rep.front();
      std::vector<double> q4 = initik_result_.q_rep.back();
      std::vector<double> q2 = q_mid;
      std::vector<double> q1(q2.size()), q3(q2.size());
      for (size_t j=0;j<q2.size();++j) {
        q1[j] = 0.5*(q0[j]+q2[j]);
        q3[j] = 0.5*(q2[j]+q4[j]);
      }
      q_paths5_ = {q0,q1,q2,q3,q4};
    }

    if (!output_dir_.empty()) {
      fs::create_directories(output_dir_);
      const std::string stem = fs::path(file).stem().string();
      auto to_out = [&](const std::vector<double>& q)->std::vector<double>{
        if (q.empty()) return q;
        return save_q_in_degree_ ? vec_rad2deg(q) : q;
      };

      YAML::Node out;
      out["source_corridor"] = file;
      out["group_name"] = group_name_;
      out["tip_link"] = tip_link_;
      out["joint_units"] = save_q_in_degree_ ? "deg" : "rad";
      out["use_intersection_anchors"] = use_intersection_anchors_;
      out["candidates_per_segment"] = candidates_per_segment_;
      out["interpolate_per_pair"] = interpolate_per_pair_;
      out["joint_cost_weights"] = joint_cost_weights_;

      YAML::Node poly_node;
      for (size_t i=0;i<C.polys.size();++i) {
        YAML::Node item;
        item["poly_index"] = (int)i;
        item["anchor_xyz"] = std::vector<double>{initik_result_.anchors[i].x(), initik_result_.anchors[i].y(), initik_result_.anchors[i].z()};
        item["anchor_source"] = initik_result_.anchor_source[i]==1 ? "intersection" : "poly";
        item["has_intersection_poly"] = (i < C.inters.size());
        poly_node.push_back(item);
      }
      out["poly_info"] = poly_node;

      YAML::Node layers_node;
      for (size_t i=0;i<initik_result_.layers.size();++i) {
        YAML::Node layer;
        layer["poly_index"] = (int)i;
        layer["best_index"] = initik_result_.best_index_per_layer[i];
        YAML::Node cand_list;
        for (size_t j=0;j<initik_result_.layers[i].size();++j) {
          const auto& c = initik_result_.layers[i][j];
          YAML::Node cj;
          cj["candidate_index"] = (int)j;
          cj["q"] = to_out(c.q);
          cj["tcp_xyz"] = std::vector<double>{c.tcp_xyz.x(), c.tcp_xyz.y(), c.tcp_xyz.z()};
          cj["rpy_deg"] = std::vector<double>{c.rpy_deg.x(), c.rpy_deg.y(), c.rpy_deg.z()};
          cj["collision_free"] = c.collision_free;
          cj["resample_id"] = c.resample_id;
          cj["ori_id"] = c.ori_id;
          cand_list.push_back(cj);
        }
        layer["candidates"] = cand_list;
        layers_node.push_back(layer);
      }
      out["ik_layers"] = layers_node;

      YAML::Node rep_node;
      for (const auto& q : initik_result_.q_rep) rep_node.push_back(to_out(q));
      out["q_rep"] = rep_node;

      YAML::Node dense_node;
      for (const auto& q : initik_result_.q_seed_dense) dense_node.push_back(to_out(q));
      out["q_seed_dense"] = dense_node;
      out["sigma_dense"] = initik_result_.sigma_dense;

      YAML::Node paths5_node;
      for (const auto& q : q_paths5_) paths5_node.push_back(to_out(q));
      out["q_paths5"] = paths5_node;

      const std::string yaml_out = (fs::path(output_dir_) / (stem + "_initik.yaml")).string();
      std::ofstream fy(yaml_out);
      fy << out;
      fy.close();
      RCLCPP_INFO(get_logger(), "Saved InitIK YAML -> %s", yaml_out.c_str());

      const std::string csv_out = (fs::path(output_dir_) / (stem + "_seed.csv")).string();
      std::ofstream fc(csv_out);
      const auto& var_names = jmg_->getVariableNames();
      fc << "idx,poly_index";
      for (const auto& n : var_names) fc << "," << n << (save_q_in_degree_ ? "_deg" : "_rad");
      fc << ",anchor_x,anchor_y,anchor_z\n";
      for (size_t i=0;i<initik_result_.q_seed_dense.size();++i) {
        fc << i << "," << initik_result_.sigma_dense[i];
        auto qq = to_out(initik_result_.q_seed_dense[i]);
        for (double v : qq) fc << "," << v;
        const int si = std::clamp(initik_result_.sigma_dense[i], 0, (int)initik_result_.anchors.size()-1);
        const auto& p = initik_result_.anchors[si];
        fc << "," << p.x() << "," << p.y() << "," << p.z() << "\n";
      }
      fc.close();
      RCLCPP_INFO(get_logger(), "Saved seed CSV -> %s", csv_out.c_str());
    }

    publishMarkers_();
    return true;
  }

  void drawRobotSkeleton_(MarkerArray& ma, int& id,
                          const std::vector<double>& q,
                          const std::string& ns,
                          double jr, double r,double g,double b,double a,
                          bool draw_frames=false, bool draw_names=false) {
    if (q.empty()) return;
    const auto& vars = jmg_->getVariableNames();
    if (q.size()!=vars.size()) return;

    robot_state_->setJointGroupPositions(jmg_, q);
    robot_state_->updateLinkTransforms();

    const auto& link_names = jmg_->getLinkModelNames();
    std::vector<Eigen::Vector3d> origins; origins.reserve(link_names.size());
    for (const auto& ln : link_names) {
      const auto* lm = robot_model_->getLinkModel(ln);
      Eigen::Isometry3d T = robot_state_->getGlobalLinkTransform(lm);
      origins.push_back(T.translation());
      ma.markers.push_back(MakeSphere(frame_id_, ns+"_j", id++, T.translation(), jr, r,g,b,a));
      if (draw_frames) AddAxes(ma, id, frame_id_, ns+"_axes_"+ln, T, link_frame_len_, link_frame_rad_);
      if (draw_names) AddText(ma, id, frame_id_, ns+"_txt", T.translation()+Eigen::Vector3d(0,0,link_frame_len_*0.6), ln);
    }
    for (size_t i=0;i+1<origins.size();++i) {
      ma.markers.push_back(makeLinkBoxBetween(frame_id_, ns+"_l", id++, origins[i], origins[i+1], link_box_w_, r,g,b,a));
    }
  }

  void publishMarkers_() {
    MarkerArray ma; int id=0;
    Marker kill; kill.header.frame_id=frame_id_; kill.action=Marker::DELETEALL; kill.ns="all"; kill.id=0; ma.markers.push_back(kill);

    for (size_t i=0;i<cached_lines_.size();++i) {
      double r = (i%2==0)? 0.2 : 0.9;
      double g = (i%2==0)? 0.5 : 0.3;
      double b = (i%2==0)? 1.0 : 0.3;
      Marker m = MakeLine(frame_id_, (i%2==0)?"poly":"inter", id++, line_w_, r,g,b, 1.0);
      const auto& V = cached_lines_[i].first;
      const auto& E = cached_lines_[i].second;
      for (auto& e : E) AddEdge(&m, V[e.first], V[e.second]);
      ma.markers.push_back(m);
    }

    for (size_t i=0;i<samples_.size();++i) {
      const bool from_inter = (i < initik_result_.anchor_source.size() && initik_result_.anchor_source[i] == 1);
      ma.markers.push_back(MakeSphere(frame_id_, from_inter ? "anchor_inter" : "anchor_poly", id++, samples_[i], pt_r_,
                                      from_inter ? 0.2 : 0.3, from_inter ? 1.0 : 0.8, 0.3, 0.85));
    }
    for (size_t i=0;i<used_sample_xyz_.size();++i) {
      ma.markers.push_back(MakeSphere(frame_id_, "sample_used", id++, used_sample_xyz_[i], pt_r_*1.05, 0.1,0.9,0.9,0.95));
    }

    if (!q_start_.empty()) {
      drawRobotSkeleton_(ma, id, q_start_, "robot_start", 0.012, 0.1,0.8,0.2,1.0,
                         show_link_frames_start_, show_link_names_start_);
    }

    for (size_t i=0;i<q_mid_list_.size();++i) {
      if (!q_mid_list_[i].empty()) {
        drawRobotSkeleton_(ma, id, q_mid_list_[i], "robot_rep", 0.010, 0.85,0.7,0.1,0.88);
      }
    }

    for (const auto& q : initik_result_.q_seed_dense) {
      if (!q.empty()) drawRobotSkeleton_(ma, id, q, "robot_seed", 0.006, 0.2,0.6,1.0,0.18);
    }

    if (!q_goal_.empty()) {
      drawRobotSkeleton_(ma, id, q_goal_, "robot_goal", 0.012, 0.1,0.8,0.2,1.0,
                         show_link_frames_goal_, show_link_names_goal_);
    }

    pub_->publish(ma);
  }

private:
  rclcpp::Publisher<MarkerArray>::SharedPtr pub_;
  std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  rclcpp::TimerBase::SharedPtr timer_;

  std::string frame_id_, group_name_, tip_link_, corridor_files_str_, output_dir_;
  int select_index_{-1}; bool run_all_{false};

  moveit::core::RobotModelPtr robot_model_;
  const moveit::core::JointModelGroup* jmg_{nullptr};
  moveit::core::RobotStatePtr robot_state_;
  planning_scene_monitor::PlanningSceneMonitorPtr psm_;
  bool ik_check_collision_{true};

  double ik_timeout_{0.05}; int ik_attempts_{8};
  std::vector<double> target_rpy_deg_{0.0,180.0,0.0};

  double yaw_sweep_deg_{120.0}, yaw_step_deg_{5.0};
  double roll_tol_deg_{0.0}, roll_step_deg_{0.0};
  double pitch_tol_deg_{0.0}, pitch_step_deg_{0.0};
  int    max_resample_per_segment_{8};
  bool   allow_colliding_fallback_{true};
  int    rng_seed_{0};
  std::mt19937 rng_;

  bool q_param_in_degree_{true};
  bool save_q_in_degree_{true};

  bool use_intersection_anchors_{true};
  int candidates_per_segment_{8};
  int ik_seed_trials_per_pose_{4};
  int interpolate_per_pair_{3};
  double ik_duplicate_tol_norm_{1e-2};
  double ik_duplicate_tol_max_{5e-3};
  std::vector<double> joint_cost_weights_;

  double line_w_{0.002}, pt_r_{0.01};

  double link_box_w_{0.025};
  std::vector<double> link_rgba_{0.20,0.80,0.25,0.65};

  bool show_link_frames_start_{true};
  bool show_link_frames_goal_{true};
  bool show_link_names_start_{true};
  bool show_link_names_goal_{true};
  double link_frame_len_{0.08};
  double link_frame_rad_{0.004};

  std::vector<std::string> files_;
  std::vector<std::pair<
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>,
    std::vector<std::pair<int,int>>
  >> cached_lines_;
  std::vector<Eigen::Vector3d> samples_;
  std::vector<Eigen::Vector3d> used_sample_xyz_;
  std::vector<std::vector<double>> q_mid_list_;
  std::vector<std::vector<double>> q_paths5_;
  InitIKResult initik_result_;

  std::vector<double> q_start_, q_goal_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<CorridorVizMoveItInitIKNoTorch>();
  if (!node->init()) {
    RCLCPP_ERROR(node->get_logger(), "Initialization failed.");
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
