// corridor_viz_moveit_ik.cpp
// 多个 corridor_*.yaml 顺序读取 -> 可视化 H-走廊 -> 每段采样点
// IK：MoveIt RobotState::setFromIK（优先无碰撞；不行则允许碰撞兜底）
// 中段：位置=采样点，姿态=target_rpy_deg 的“候选列表”（可扫描 yaw/roll/pitch）
// 若均失败：在对应多面体中再次采样若干次；仍失败则取碰撞 IK；还不行就置空
//
// 依赖：rclcpp, visualization_msgs, geometry_msgs, Eigen, yaml-cpp
//       MoveIt (robot_model_loader, robot_state, planning_scene_monitor), TF2
//       Drake (HPolyhedron)
//
// 关键参数（新增*）：
//   - yaw_sweep_deg (默认 120), yaw_step_deg (默认 5)
//   - roll_tol_deg/pitch_tol_deg（默认 0），roll_step_deg/pitch_step_deg（默认 0）
//   - max_resample_per_segment (默认 8)
//   - allow_colliding_fallback (默认 true)
//   - rng_seed (默认 0=使用时间种子)
//   - *q_param_in_degree (默认 true)：外部传入的 q_start/q_goal 是否为角度
//   - *save_q_in_degree  (默认 true)：导出的关节是否以角度保存
//   - *show_link_frames_start/goal：是否在起/终态为每个 link 画坐标轴与名字
//   - *link_frame_len, link_frame_rad：坐标轴长度/半径
//
// 用法示例：
// ros2 run iris_rviz_cpp corridor_viz_moveit_ik --ros-args \
//   -p corridor_files:="/path/corridor_01.yaml" \
//   -p select_index:=0 \
//   -p group_name:=arm -p tip_link:=tool0 -p frame_id:=world \
//   -p fix_orientation:=true -p target_rpy_deg:="[0.0,180.0,0.0]" \
//   -p yaw_sweep_deg:=120 -p yaw_step_deg:=5 \
//   -p ik_timeout:=0.08 -p ik_attempts:=10 -p ik_check_collision:=true \
//   -p max_resample_per_segment:=8 -p allow_colliding_fallback:=true \
//   -p q_param_in_degree:=true -p save_q_in_degree:=true \
//   -p show_link_frames_start:=true -p show_link_frames_goal:=true \
//   -p output_dir:="/home/hyj/iris_rviz_ws/src/iris_rviz_cpp/src/corridors_out1/ik_results"

#include <chrono>
#include <fstream>
#include <optional>
#include <random>
#include <string>
#include <vector>
#include <algorithm>
#include <filesystem>
#include <sstream>
#include <array>   // ★ 新增：为 std::array

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

// MoveIt
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit/planning_scene_monitor/planning_scene_monitor.h>
#include <moveit/collision_detection/collision_common.h>

using visualization_msgs::msg::Marker;
using visualization_msgs::msg::MarkerArray;
using geometry_msgs::msg::Point;

#include "drake/geometry/optimization/hpolyhedron.h"
#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/solve.h"
using drake::geometry::optimization::HPolyhedron;
namespace mp = drake::solvers;

// --------------------- 角度/弧度工具 ---------------------
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

// --------------------- 画图工具 ---------------------
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

// —— 连杆盒子 & 两段圆柱焊枪 —— //
static inline Eigen::Quaterniond quatFromZ(const Eigen::Vector3d& z_dir_world) {
  Eigen::Vector3d z = z_dir_world.normalized();
  return Eigen::Quaterniond::FromTwoVectors(Eigen::Vector3d::UnitZ(), z);
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
static Marker makeCylinder(const std::string& frame, const std::string& ns, int id,
                           const Eigen::Vector3d& p_center,
                           const Eigen::Vector3d& axis_dir_world,
                           double length, double diameter,
                           double r, double g, double b, double a) {
  Marker m; m.header.frame_id = frame; m.ns = ns; m.id = id;
  m.type = Marker::CYLINDER; m.action = Marker::ADD;
  m.scale.x = diameter; m.scale.y = diameter; m.scale.z = length;
  m.pose.position.x = p_center.x();
  m.pose.position.y = p_center.y();
  m.pose.position.z = p_center.z();
  Eigen::Quaterniond q = quatFromZ(axis_dir_world);
  m.pose.orientation.x = q.x();
  m.pose.orientation.y = q.y();
  m.pose.orientation.z = q.z();
  m.pose.orientation.w = q.w();
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

// 画坐标轴（3 个 ARROW）
static void AddAxes(MarkerArray& ma, int& id, const std::string& frame, const std::string& ns,
                    const Eigen::Isometry3d& T, double len, double rad) {
  auto add_arrow = [&](const Eigen::Vector3d& dir, double r, double g, double b){
    Marker a; a.header.frame_id = frame; a.ns = ns; a.id = id++;
    a.type = Marker::ARROW; a.action = Marker::ADD;
    a.scale.x = rad;        // shaft diameter
    a.scale.y = 2.0*rad;    // head diameter
    a.scale.z = 0.25*len;   // head length
    a.color.r = r; a.color.g = g; a.color.b = b; a.color.a = 1.0;
    geometry_msgs::msg::Point p0, p1;
    Eigen::Vector3d P0 = T.translation();
    Eigen::Vector3d P1 = P0 + T.linear()*dir*len;
    p0.x=P0.x(); p0.y=P0.y(); p0.z=P0.z();
    p1.x=P1.x(); p1.y=P1.y(); p1.z=P1.z();
    a.points.push_back(p0); a.points.push_back(p1);
    ma.markers.push_back(a);
  };
  add_arrow(Eigen::Vector3d::UnitX(), 1.0,0.2,0.2); // X 红
  add_arrow(Eigen::Vector3d::UnitY(), 0.2,1.0,0.2); // Y 绿
  add_arrow(Eigen::Vector3d::UnitZ(), 0.2,0.4,1.0); // Z 蓝
}

// 文字标签
static void AddText(MarkerArray& ma, int& id, const std::string& frame, const std::string& ns,
                    const Eigen::Vector3d& p, const std::string& text, double scale=0.035) {
  Marker m; m.header.frame_id=frame; m.ns=ns; m.id=id++;
  m.type=Marker::TEXT_VIEW_FACING; m.action=Marker::ADD;
  m.pose.position.x=p.x(); m.pose.position.y=p.y(); m.pose.position.z=p.z();
  m.scale.z=scale; m.color.r=1.0; m.color.g=1.0; m.color.b=1.0; m.color.a=0.95;
  m.text=text; ma.markers.push_back(m);
}

// --------------------- H 多面体工具 ---------------------
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
// 在 H 内采样一个点：优先用“顶点凸组合”，否则用“最近可行点”投影
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
  // 顶点不可得：取 Chebyshev 或最小范数点，再做一次“最近可行点”QP
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
  // 投影随机扰动到可行域
  std::normal_distribution<double> norm01(0.0, 1.0);
  Eigen::Vector3d y = c + 0.1 * Eigen::Vector3d(norm01(rng), norm01(rng), norm01(rng));
  const auto& A=H.A(); const auto& b=H.b();
  mp::MathematicalProgram prog;
  auto x = prog.NewContinuousVariables<3>("x");
  prog.AddLinearConstraint(A*x <= b);
  Eigen::Matrix3d Q = Eigen::Matrix3d::Identity();
  prog.AddQuadraticCost(Q, -Q*y, x); // ||x - y||^2 = x'Qx - 2y'Qx + const
  auto res = mp::Solve(prog);
  if (res.is_success()) return res.GetSolution(x);
  return c;
}

// --------------------- corridor YAML 解析 ---------------------
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
  return !out.interleaved.empty();
}

// --------------------- 主节点 ---------------------
class CorridorVizMoveItIK : public rclcpp::Node {
public:
  CorridorVizMoveItIK(): Node("corridor_viz_moveit_ik") {
    pub_ = this->create_publisher<MarkerArray>("/corridor_ik_markers", 1);
    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

    // 基本参数
    frame_id_      = declare_parameter<std::string>("frame_id", "world");
    group_name_    = declare_parameter<std::string>("group_name", "arm");
    tip_link_      = declare_parameter<std::string>("tip_link", "tool0");
    corridor_files_str_ = declare_parameter<std::string>("corridor_files", "");
    select_index_  = declare_parameter<int>("select_index", -1);
    run_all_       = declare_parameter<bool>("run_all", false);

    // 起止（关节角可用角度输入）
    q_start_ = declare_parameter<std::vector<double>>("q_start", std::vector<double>{});
    q_goal_  = declare_parameter<std::vector<double>>("q_goal",  std::vector<double>{});

    // IK 相关
    ik_timeout_       = declare_parameter<double>("ik_timeout", 0.05);
    ik_attempts_      = declare_parameter<int>("ik_attempts", 8);
    fix_ori_          = declare_parameter<bool>("fix_orientation", true);
    target_rpy_deg_   = declare_parameter<std::vector<double>>("target_rpy_deg", {0.0,180.0,0.0});
    ik_check_collision_ = declare_parameter<bool>("ik_check_collision", true);

    // 新增：姿态扫描与再采样参数
    yaw_sweep_deg_    = declare_parameter<double>("yaw_sweep_deg", 120.0);
    yaw_step_deg_     = declare_parameter<double>("yaw_step_deg", 5.0);
    roll_tol_deg_     = declare_parameter<double>("roll_tol_deg", 0.0);
    roll_step_deg_    = declare_parameter<double>("roll_step_deg", 0.0);
    pitch_tol_deg_    = declare_parameter<double>("pitch_tol_deg", 0.0);
    pitch_step_deg_   = declare_parameter<double>("pitch_step_deg", 0.0);
    max_resample_per_segment_ = declare_parameter<int>("max_resample_per_segment", 8);
    allow_colliding_fallback_ = declare_parameter<bool>("allow_colliding_fallback", true);
    rng_seed_         = declare_parameter<int>("rng_seed", 0);

    // 角度/弧度 I/O 约定（新增）
    q_param_in_degree_ = declare_parameter<bool>("q_param_in_degree", true);
    save_q_in_degree_  = declare_parameter<bool>("save_q_in_degree",  true);

    // 采样点显示
    line_w_ = declare_parameter<double>("line_width", 0.002);
    pt_r_   = declare_parameter<double>("sample_point_r", 0.01);

    // link box + torch
    link_box_w_ = declare_parameter<double>("link_box_width", 0.025);
    link_rgba_  = declare_parameter<std::vector<double>>("link_rgba", {0.20,0.80,0.25,0.65});
    if (link_rgba_.size()!=4) link_rgba_={0.20,0.80,0.25,0.65};
    torch_enable_   = declare_parameter<bool>("torch_enable", true);
    torch_cyl1_len_ = declare_parameter<double>("torch_cyl1_len", 0.36);
    torch_cyl1_dia_ = declare_parameter<double>("torch_cyl1_dia", 0.025);
    torch_cyl2_len_ = declare_parameter<double>("torch_cyl2_len", 0.11);
    torch_cyl2_dia_ = declare_parameter<double>("torch_cyl2_dia", 0.018);
    torch_tilt_deg_ = declare_parameter<double>("torch_tilt_deg", 45.0);
    auto xyz = declare_parameter<std::vector<double>>("torch_offset_xyz", {0,0,0});
    if (xyz.size()!=3) xyz={0,0,0};
    torch_off_xyz_ = Eigen::Vector3d(xyz[0],xyz[1],xyz[2]);
    auto rpy = declare_parameter<std::vector<double>>("torch_offset_rpy_deg", {0,0,0});
    if (rpy.size()!=3) rpy={0,0,0};
    torch_off_rpy_deg_ = Eigen::Vector3d(rpy[0],rpy[1],rpy[2]);
    torch1_rgba_ = declare_parameter<std::vector<double>>("torch1_rgba", {0.95,0.30,0.30,1.0});
    if (torch1_rgba_.size()!=4) torch1_rgba_={0.95,0.30,0.30,1.0};
    torch2_rgba_ = declare_parameter<std::vector<double>>("torch2_rgba", {0.95,0.60,0.20,1.0});
    if (torch2_rgba_.size()!=4) torch2_rgba_={0.95,0.60,0.20,1.0};

    // 起/终状态 link 坐标轴与命名可视化（新增）
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
    // RNG
    if (rng_seed_ == 0) {
      std::seed_seq seed{(unsigned)std::chrono::high_resolution_clock::now().time_since_epoch().count()};
      rng_ = std::mt19937(seed);
    } else {
      rng_ = std::mt19937((unsigned)rng_seed_);
    }

    // MoveIt
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

    // —— 角度入参转弧度 ——
    if (q_param_in_degree_) {
      if (!q_start_.empty()) q_start_ = vec_deg2rad(q_start_);
      if (!q_goal_.empty())  q_goal_  = vec_deg2rad(q_goal_);
    }

    if (files_.empty()) { RCLCPP_ERROR(get_logger(), "No corridor_files."); return false; }
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

  // tip->tcp 偏置（局部 RPY，先旋再平移）
  Eigen::Isometry3d buildLocalOffset_() const {
    const double r = torch_off_rpy_deg_.x() * M_PI/180.0;
    const double p = torch_off_rpy_deg_.y() * M_PI/180.0;
    const double y = torch_off_rpy_deg_.z() * M_PI/180.0;
    Eigen::Matrix3d R_off =
      Eigen::AngleAxisd(y, Eigen::Vector3d::UnitZ()).toRotationMatrix() *
      Eigen::AngleAxisd(p, Eigen::Vector3d::UnitY()).toRotationMatrix() *
      Eigen::AngleAxisd(r, Eigen::Vector3d::UnitX()).toRotationMatrix();
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    T.linear() = R_off;
    T.translation() = R_off * torch_off_xyz_;
    return T;
  }

  // 构造姿态候选（度）
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
    if (roll_tol_deg_ > 0.0 && roll_step_deg_ > 0.0)
      roll_list = arange(-roll_tol_deg_, roll_tol_deg_, roll_step_deg_);
    if (pitch_tol_deg_ > 0.0 && pitch_step_deg_ > 0.0)
      pitch_list = arange(-pitch_tol_deg_, pitch_tol_deg_, pitch_step_deg_);

    // 优先中心
    cand.emplace_back(r0, p0, y0);
    // yaw 侧向
    for (double dy : yaw_list) {
      if (std::abs(dy) < 1e-9) continue;
      cand.emplace_back(r0, p0, y0 + dy);
    }
    // roll/pitch 小网格
    for (double dr : roll_list) {
      for (double dp : pitch_list) {
        if (std::abs(dr)<1e-9 && std::abs(dp)<1e-9) continue;
        for (double dy : yaw_list) {
          cand.emplace_back(r0+dr, p0+dp, y0+dy);
        }
      }
    }
    return cand;
  }

  // 尝试 IK（优先无碰撞，其次允许碰撞）
  bool tryIKAtTipPose_(const geometry_msgs::msg::Pose& tip_pose,
                       std::vector<double>& qsol_out,
                       const moveit::core::GroupStateValidityCallbackFn& validity) {
    for (int att=0; att<ik_attempts_; ++att) {
      bool ok = validity
        ? robot_state_->setFromIK(jmg_, tip_pose, tip_link_, ik_timeout_, validity)
        : robot_state_->setFromIK(jmg_, tip_pose, tip_link_, ik_timeout_);
      if (ok) {
        qsol_out.resize(jmg_->getVariableCount());
        robot_state_->copyJointGroupPositions(jmg_, qsol_out);
        return true;
      }
    }
    if (allow_colliding_fallback_) {
      for (int att=0; att<ik_attempts_; ++att) {
        if (robot_state_->setFromIK(jmg_, tip_pose, tip_link_, ik_timeout_)) {
          qsol_out.resize(jmg_->getVariableCount());
          robot_state_->copyJointGroupPositions(jmg_, qsol_out);
          return true; // 可能碰撞
        }
      }
    }
    return false;
  }

  // 从 TCP 位置 + RPY（度）构造 tip 目标 Pose，并尝试 IK
  bool solveAtTcp_(const Eigen::Vector3d& tcp_pos,
                   const Eigen::Vector3d& rpy_deg,
                   std::vector<double>& qsol_out,
                   const Eigen::Isometry3d& T_tip2tcp_local,
                   const moveit::core::GroupStateValidityCallbackFn& validity) {
    const double rr=rpy_deg.x()*M_PI/180.0, pp=rpy_deg.y()*M_PI/180.0, yy=rpy_deg.z()*M_PI/180.0;
    Eigen::Matrix3d R_tcp =
      Eigen::AngleAxisd(yy, Eigen::Vector3d::UnitZ()).toRotationMatrix() *
      Eigen::AngleAxisd(pp, Eigen::Vector3d::UnitY()).toRotationMatrix() *
      Eigen::AngleAxisd(rr, Eigen::Vector3d::UnitX()).toRotationMatrix();

    Eigen::Isometry3d T_world_tcp = Eigen::Isometry3d::Identity();
    T_world_tcp.linear() = R_tcp;
    T_world_tcp.translation() = tcp_pos;

    // 反推 tip： T_tip = T_tcp * T_off^{-1}
    Eigen::Isometry3d T_world_tip = T_world_tcp * T_tip2tcp_local.inverse();

    geometry_msgs::msg::Pose tip_pose;
    tip_pose.position.x = T_world_tip.translation().x();
    tip_pose.position.y = T_world_tip.translation().y();
    tip_pose.position.z = T_world_tip.translation().z();
    Eigen::Quaterniond q(T_world_tip.rotation());
    tip_pose.orientation.x = q.x();
    tip_pose.orientation.y = q.y();
    tip_pose.orientation.z = q.z();
    tip_pose.orientation.w = q.w();

    return tryIKAtTipPose_(tip_pose, qsol_out, validity);
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

    // *** NEW: 只在 final_path_polys（安全走廊）上采样，不在交集多面体里采样 ***
    const size_t N = C.polys.size();
    if (N == 0) {
      RCLCPP_ERROR(get_logger(), "No final_path_polys in corridor yaml.");
      return false;
    }

    // 预计算各段顶点（用于快速采样）
    std::vector<std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>> V_list(N);
    for (size_t i=0;i<N;++i) {
      V_list[i] = VerticesFromH3D(C.polys[i]);
    }

    // 每段一个初始采样点（Chebyshev / 最小范数）
    std::vector<Eigen::Vector3d> samples(N, Eigen::Vector3d::Zero());
    for (size_t i=0;i<N;++i) {
      auto p = ChebyshevPoint(C.polys[i]);
      if (!p) {
        const auto& A=C.polys[i].A(); const auto& b=C.polys[i].b();
        mp::MathematicalProgram prog;
        auto x = prog.NewContinuousVariables<3>("x");
        prog.AddLinearConstraint(A*x <= b);
        prog.AddQuadraticCost(Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero(), x);
        auto res = mp::Solve(prog);
        if (res.is_success()) p = res.GetSolution(x);
      }
      if (p) samples[i]=*p;
    }

    // IK：中间段求解；首尾不 IK
    const auto& vars = jmg_->getVariableNames();

    // 初始姿态
    if (!q_start_.empty() && q_start_.size()==vars.size()) {
      robot_state_->setJointGroupPositions(jmg_, q_start_);
      robot_state_->updateLinkTransforms();
    } else {
      robot_state_->setToDefaultValues();
      robot_state_->updateLinkTransforms();
    }

    // 偏置 & 姿态候选
    const Eigen::Isometry3d T_tip2tcp_local = buildLocalOffset_();
    const std::vector<Eigen::Vector3d> ori_cands_deg = buildOrientationCandidatesDeg_();

    // 碰撞回调
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
      RCLCPP_INFO(get_logger(), "IK collision checking enabled.");
    } else {
      RCLCPP_INFO(get_logger(), "IK collision checking disabled.");
    }

    // 结果集合
    std::vector<std::vector<double>> q_mid_list(N);
    std::vector<Eigen::Vector3d> used_sample_xyz(N, Eigen::Vector3d::Zero());

    // 逐段
    for (size_t i=0;i<N; ++i) {
      if (i==0 || i==N-1) { // 首尾：不 IK
        q_mid_list[i].clear();
        used_sample_xyz[i] = samples[i];
        continue;
      }

      bool found = false;
      std::vector<double> qsol_candidate;
      Eigen::Vector3d chosen_sample = samples[i];

      // 第 0 次尝试：用原 sample + 姿态候选
      auto try_candidates = [&](const Eigen::Vector3d& tcp){
        for (const auto& rpy_deg : ori_cands_deg) {
          std::vector<double> qtmp;
          bool ok = solveAtTcp_(tcp, rpy_deg, qtmp, T_tip2tcp_local, validity);
          if (ok) { qsol_candidate = std::move(qtmp); return true; }
        }
        return false;
      };

      found = try_candidates(samples[i]);

      // 若未找到则在该段多面体里再采样（至多 max_resample_per_segment 次）
      for (int rs=0; !found && rs<max_resample_per_segment_; ++rs) {
        Eigen::Vector3d sp = SamplePointIn(C.polys[i], rng_, &V_list[i]);
        if (try_candidates(sp)) { chosen_sample = sp; found = true; break; }
      }

      if (found) {
        q_mid_list[i] = qsol_candidate;
        used_sample_xyz[i] = chosen_sample;
      } else {
        used_sample_xyz[i] = samples[i];
        RCLCPP_WARN(get_logger(), "Segment %zu: IK not found even after resampling.", i);
        q_mid_list[i].clear();
      }
    }

    // q_goal 兜底
    std::vector<double> qstart_out = q_start_;
    std::vector<double> qgoal_out  = q_goal_;
    if (q_goal_.empty() && N>=2 && !q_mid_list[N-2].empty())
      qgoal_out = q_mid_list[N-2];

    // *** NEW: 以当前 IK 结果为“中间那条” q_center，并对 start / goal 做插值，构造 5 条路径 ***
    // q_center[i]：第 i 段的中心轨迹关节向量（首尾用 q_start / q_goal， 中间用 q_mid_list）
    std::vector<std::vector<double>> q_center(N);

    for (size_t i=0; i<N; ++i) {
      if (i == 0 && !qstart_out.empty()) {
        q_center[i] = qstart_out;
      } else if (i == N-1 && !qgoal_out.empty()) {
        q_center[i] = qgoal_out;
      } else if (!q_mid_list[i].empty()) {
        q_center[i] = q_mid_list[i];
      } else if (!qstart_out.empty() && !qgoal_out.empty()) {
        // 若 IK 失败，则在 start / goal 之间做线性插值兜底
        double alpha = (N<=1) ? 0.0 : static_cast<double>(i) / static_cast<double>(N-1);
        q_center[i].resize(qstart_out.size());
        for (size_t j=0; j<qstart_out.size(); ++j) {
          q_center[i][j] = (1.0 - alpha)*qstart_out[j] + alpha*qgoal_out[j];
        }
      } else {
        // 实在没有信息，就留空
        q_center[i].clear();
      }
    }

    // ===== NEW: 从 q_center 中抽一条“代表性中间姿态”，构造 5 个关键关节向量 =====
    q_paths5_.clear();
    q_paths5_.resize(5);  // 5 个 [ndof] 向量

    // 记住中间段索引，后面写 CSV 的 used_sample_xyz 时复用
    size_t mid_idx = 0;

    if (!q_center.empty()) {
      // 选走廊中间段的 q_center 作为 q_mid
      mid_idx = (N <= 1) ? 0 : (N - 1) / 2;
      if (mid_idx >= q_center.size() || q_center[mid_idx].empty()) {
        // 兜底：找第一个非空的 q_center
        bool found = false;
        for (size_t i = 0; i < q_center.size(); ++i) {
          if (!q_center[i].empty()) {
            mid_idx = i;
            found = true;
            break;
          }
        }
        if (!found) {
          RCLCPP_WARN(get_logger(), "No valid q_center found, skip building q_paths5.");
        }
      }

      if (!q_center[mid_idx].empty()) {
        const std::vector<double>& q_mid = q_center[mid_idx];
        const size_t ndof = q_mid.size();

        // q0 = q_start, q4 = q_goal，如果没有就用 q_mid 兜底
        std::vector<double> q0 = q_mid;
        std::vector<double> q4 = q_mid;

        if (!qstart_out.empty()) q0 = qstart_out;
        if (!qgoal_out.empty())  q4 = qgoal_out;

        std::vector<double> q2 = q_mid;              // mid
        std::vector<double> q1(ndof), q3(ndof);      // 插值

        for (size_t j = 0; j < ndof; ++j) {
          q1[j] = 0.5 * (q0[j] + q2[j]);  // (start + mid) / 2
          q3[j] = 0.5 * (q2[j] + q4[j]);  // (mid + goal) / 2
        }

        // 顺序：0..4
        q_paths5_[0] = q0;
        q_paths5_[1] = q1;
        q_paths5_[2] = q2;
        q_paths5_[3] = q3;
        q_paths5_[4] = q4;
      }
    }

    // —— 保存到 YAML/CSV（单位可选，默认角度） —— //
    if (!output_dir_.empty()) {
      fs::create_directories(output_dir_);
      const std::string stem = fs::path(file).stem().string();

      YAML::Node out;
      out["source_corridor"] = file;
      out["group_name"]      = group_name_;
      out["tip_link"]        = tip_link_;
      out["fix_orientation"] = fix_ori_;
      out["target_rpy_deg"]  = target_rpy_deg_;
      out["yaw_sweep_deg"]   = yaw_sweep_deg_;
      out["yaw_step_deg"]    = yaw_step_deg_;
      out["roll_tol_deg"]    = roll_tol_deg_;
      out["roll_step_deg"]   = roll_step_deg_;
      out["pitch_tol_deg"]   = pitch_tol_deg_;
      out["pitch_step_deg"]  = pitch_step_deg_;
      out["max_resample_per_segment"] = max_resample_per_segment_;
      out["allow_colliding_fallback"] = allow_colliding_fallback_;
      out["ik_check_collision"]       = ik_check_collision_;
      out["tcp_offset_xyz"]  = std::vector<double>{torch_off_xyz_.x(), torch_off_xyz_.y(), torch_off_xyz_.z()};
      out["tcp_offset_rpy_deg"] = std::vector<double>{torch_off_rpy_deg_.x(), torch_off_rpy_deg_.y(), torch_off_rpy_deg_.z()};
      out["joint_units"] = save_q_in_degree_ ? "deg" : "rad";

      auto to_out = [&](const std::vector<double>& q)->std::vector<double>{
        if (q.empty()) return q;
        return save_q_in_degree_ ? vec_rad2deg(q) : q;
      };

      if (!qstart_out.empty()) out["q_start"] = to_out(qstart_out);
      if (!qgoal_out.empty())  out["q_goal"]  = to_out(qgoal_out);

      YAML::Node mid;
      YAML::Node used_pts;
      for (size_t i=0;i<N;++i) {
        if (q_mid_list[i].empty()) mid.push_back(YAML::Node());
        else                       mid.push_back(to_out(q_mid_list[i]));
        used_pts.push_back(std::vector<double>{used_sample_xyz[i].x(), used_sample_xyz[i].y(), used_sample_xyz[i].z()});
      }
      out["q_mid_list"]      = mid;
      out["used_sample_xyz"] = used_pts;

      // NEW: q_paths5_ 只存 5 个关键向量 [q0..q4]
      YAML::Node paths5_node;
      for (int k = 0; k < 5; ++k) {
        if (k < static_cast<int>(q_paths5_.size()) && !q_paths5_[k].empty()) {
          paths5_node.push_back(to_out(q_paths5_[k]));
        } else {
          paths5_node.push_back(YAML::Node());
        }
      }
      out["q_paths5"] = paths5_node;

      const std::string yaml_out = (fs::path(output_dir_) / (stem + "_ik.yaml")).string();
      std::ofstream fy(yaml_out);
      fy << out;
      fy.close();
      RCLCPP_INFO(get_logger(), "Saved IK YAML -> %s", yaml_out.c_str());

      const std::string csv_out = (fs::path(output_dir_) / (stem + "_ik.csv")).string();
      std::ofstream fc(csv_out);

      // 表头：idx, q..., used_sample_xyz
      fc << "idx";
      const auto& var_names = jmg_->getVariableNames();
      for (const auto& n : var_names) {
        fc << "," << n << (save_q_in_degree_ ? "_deg" : "_rad");
      }
      fc << ",used_sample_x,used_sample_y,used_sample_z\n";

      auto write_row = [&](const std::vector<double>& q){
        auto qq = save_q_in_degree_ ? vec_rad2deg(q) : q;
        for (double v : qq) fc << "," << v;
      };

      // 为了让 used_sample_xyz 看起来合理一点，简单用首、中、尾段的采样点插值
      Eigen::Vector3d p0 = used_sample_xyz.empty() ? Eigen::Vector3d::Zero() : used_sample_xyz.front();
      Eigen::Vector3d p4 = used_sample_xyz.empty() ? Eigen::Vector3d::Zero() : used_sample_xyz.back();
      Eigen::Vector3d p2 = (mid_idx < used_sample_xyz.size()) ? used_sample_xyz[mid_idx] : 0.5*(p0 + p4);
      Eigen::Vector3d p1 = 0.5*(p0 + p2);
      Eigen::Vector3d p3 = 0.5*(p2 + p4);
      std::array<Eigen::Vector3d,5> used_for_paths{p0,p1,p2,p3,p4};

      const size_t num_rows = q_paths5_.size();   // 理论上是 5
      for (size_t i = 0; i < num_rows; ++i) {
        fc << i;
        if (i < q_paths5_.size() && !q_paths5_[i].empty()) {
          write_row(q_paths5_[i]);
        } else {
          // 兜底：这一行没有 q，就填空
          for (size_t k = 0; k < var_names.size(); ++k) {
            fc << ",";
          }
        }
        const Eigen::Vector3d& p = (i < used_for_paths.size()) ? used_for_paths[i]
                                                                : used_for_paths.back();
        fc << "," << p.x() << "," << p.y() << "," << p.z() << "\n";
      }

      fc.close();
      RCLCPP_INFO(get_logger(), "Saved IK CSV (q_paths5) -> %s", csv_out.c_str());
    } // end if (!output_dir_.empty())

    // —— 可视化缓存 —— //
    cached_lines_.clear();
    for (size_t i=0;i<C.interleaved.size();++i) {
      auto V = VerticesFromH3D(C.interleaved[i]);
      auto E = EdgesFromActiveSets(C.interleaved[i], V);
      cached_lines_.push_back({std::move(V), std::move(E)});
    }
    samples_         = std::move(samples);
    q_mid_list_      = std::move(q_mid_list);
    used_sample_xyz_ = std::move(used_sample_xyz);

    publishMarkers_();
    return true;
  }

  void drawRobotSkeleton_(MarkerArray& ma, int& id,
                          const std::vector<double>& q,
                          const std::string& ns,
                          double jr, double /*lr*/, double r,double g,double b,double a,
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
      if (draw_frames) {
        AddAxes(ma, id, frame_id_, ns+"_axes_"+ln, T, link_frame_len_, link_frame_rad_);
      }
      if (draw_names) {
        AddText(ma, id, frame_id_, ns+"_txt", T.translation()+Eigen::Vector3d(0,0,link_frame_len_*0.6), ln);
      }
    }
    for (size_t i=0;i+1<origins.size();++i) {
      ma.markers.push_back(makeLinkBoxBetween(frame_id_, ns+"_l", id++,
                        origins[i], origins[i+1], link_box_w_, r,g,b,a));
    }

    // tip->tcp 偏置 + 焊枪
    const auto* tip = robot_model_->getLinkModel(tip_link_);
    if (!tip) return;
    Eigen::Isometry3d T_world_tip = robot_state_->getGlobalLinkTransform(tip);
    Eigen::Isometry3d T_world_tcp = T_world_tip * buildLocalOffset_();
    Eigen::Vector3d p_tcp = T_world_tcp.translation();
    Eigen::Matrix3d R0 = T_world_tcp.linear();

    if (torch_enable_) {
      Eigen::Vector3d dir1_world = R0 * Eigen::Vector3d::UnitZ();
      Eigen::Vector3d c1 = p_tcp + 0.5 * torch_cyl1_len_ * dir1_world;
      ma.markers.push_back(makeCylinder(frame_id_, ns+"_torch1", id++,
                        c1, dir1_world, torch_cyl1_len_, torch_cyl1_dia_,
                        torch1_rgba_[0], torch1_rgba_[1], torch1_rgba_[2], torch1_rgba_[3]));
      const double tilt = -torch_tilt_deg_ * M_PI/180.0;
      Eigen::Matrix3d R_tilt_localX =
          Eigen::AngleAxisd(tilt, Eigen::Vector3d::UnitX()).toRotationMatrix();
      Eigen::Vector3d dir2_world = R0 * (R_tilt_localX * Eigen::Vector3d::UnitZ());
      Eigen::Vector3d p1_end = p_tcp + torch_cyl1_len_ * dir1_world;
      Eigen::Vector3d c2 = p1_end + 0.5 * torch_cyl2_len_ * dir2_world;
      ma.markers.push_back(makeCylinder(frame_id_, ns+"_torch2", id++,
                        c2, dir2_world, torch_cyl2_len_, torch_cyl2_dia_,
                        torch2_rgba_[0], torch2_rgba_[1], torch2_rgba_[2], torch2_rgba_[3]));
    }
  }

  void publishMarkers_() {
    MarkerArray ma; int id=0;

    // 清屏
    {
      Marker kill; kill.header.frame_id=frame_id_;
      kill.action=Marker::DELETEALL; kill.ns="all"; kill.id=0; ma.markers.push_back(kill);
    }

    // 走廊线框
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

    // 原始 Chebyshev 采样点（浅色）
    for (size_t i=0;i<samples_.size();++i) {
      ma.markers.push_back(MakeSphere(frame_id_, "sample_init", id++, samples_[i], pt_r_, 0.3,0.8,0.3,0.5));
    }
    // 实际使用的采样点（深色）
    for (size_t i=0;i<used_sample_xyz_.size();++i) {
      ma.markers.push_back(MakeSphere(frame_id_, "sample_used", id++, used_sample_xyz_[i], pt_r_*1.05, 0.1,0.9,0.9,0.95));
    }

    // 起始状态：骨架 + 每个 link 的坐标轴/名字（按开关）
    if (!q_start_.empty()) {
      drawRobotSkeleton_(ma, id, q_start_, "robot_start", 0.012, 0.006, 0.1,0.8,0.2,1.0,
                         show_link_frames_start_, show_link_names_start_);
    }

    // 中间 IK 骨架
    for (size_t i=0;i<q_mid_list_.size();++i) {
      if (i==0 || i+1==q_mid_list_.size()) continue;
      if (!q_mid_list_[i].empty()) {
        drawRobotSkeleton_(ma, id, q_mid_list_[i], "robot_mid", 0.010, 0.005, 0.85,0.7,0.1,0.9);
      } else {
        ma.markers.push_back(MakeSphere(frame_id_, "ik_fail", id++, used_sample_xyz_[i], pt_r_*1.3, 1.0,0.2,0.2,0.95));
      }
    }

    // 终态：骨架 + 每个 link 的坐标轴/名字（按开关）
    if (!q_goal_.empty()) {
      drawRobotSkeleton_(ma, id, q_goal_, "robot_goal", 0.012, 0.006, 0.1,0.8,0.2,1.0,
                         show_link_frames_goal_, show_link_names_goal_);
    }

    pub_->publish(ma);
  }

private:
  // ROS
  rclcpp::Publisher<MarkerArray>::SharedPtr pub_;
  std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  rclcpp::TimerBase::SharedPtr timer_;

  // Params
  std::string frame_id_, group_name_, tip_link_, corridor_files_str_, output_dir_;
  int select_index_{-1}; bool run_all_{false};

  // MoveIt
  moveit::core::RobotModelPtr robot_model_;
  const moveit::core::JointModelGroup* jmg_{nullptr};
  moveit::core::RobotStatePtr robot_state_;
  planning_scene_monitor::PlanningSceneMonitorPtr psm_;
  bool ik_check_collision_{true};

  // IK 设置
  double ik_timeout_{0.05}; int ik_attempts_{8};
  bool fix_ori_{true};
  std::vector<double> target_rpy_deg_{0.0,180.0,0.0};

  // 姿态扫描 / 再采样
  double yaw_sweep_deg_{120.0}, yaw_step_deg_{5.0};
  double roll_tol_deg_{0.0}, roll_step_deg_{0.0};
  double pitch_tol_deg_{0.0}, pitch_step_deg_{0.0};
  int    max_resample_per_segment_{8};
  bool   allow_colliding_fallback_{true};
  int    rng_seed_{0};
  std::mt19937 rng_;

  // 角度/弧度 I/O 约定
  bool q_param_in_degree_{true};
  bool save_q_in_degree_{true};

  // 采样/显示
  double line_w_{0.002}, pt_r_{0.01};

  // link box & torch
  double link_box_w_{0.025};
  std::vector<double> link_rgba_{0.20,0.80,0.25,0.65};
  bool   torch_enable_{true};
  double torch_cyl1_len_{0.36}, torch_cyl1_dia_{0.025};
  double torch_cyl2_len_{0.11}, torch_cyl2_dia_{0.018};
  double torch_tilt_deg_{45.0};
  Eigen::Vector3d torch_off_xyz_{0,0,0};
  Eigen::Vector3d torch_off_rpy_deg_{0,0,0};
  std::vector<double> torch1_rgba_{0.95,0.30,0.30,1.0};
  std::vector<double> torch2_rgba_{0.95,0.60,0.20,1.0};

  // 起/终状态 link 坐标轴/名字显示
  bool show_link_frames_start_{true};
  bool show_link_frames_goal_{true};
  bool show_link_names_start_{true};
  bool show_link_names_goal_{true};
  double link_frame_len_{0.08};
  double link_frame_rad_{0.004};

  // 数据
  std::vector<std::string> files_;
  std::vector<std::pair<
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>,
    std::vector<std::pair<int,int>>
  >> cached_lines_;
  std::vector<Eigen::Vector3d> samples_;
  std::vector<Eigen::Vector3d> used_sample_xyz_;
  std::vector<std::vector<double>> q_mid_list_;
  // 5 个关键 q：q0=start, q1=(start+mid)/2, q2=mid, q3=(mid+goal)/2, q4=goal
  std::vector<std::vector<double>> q_paths5_; // [5][ndof]

  // 起止
  std::vector<double> q_start_, q_goal_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<CorridorVizMoveItIK>();
  if (!node->init()) {
    RCLCPP_ERROR(node->get_logger(), "Initialization failed.");
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}

