
#include <cstdio>   // for FILE*, popen, pclose
#include <cstdlib>  // for std::getenv
#include <map>
#include <random>
#include <fcl/geometry/shape/convex.h>
#include <numeric>  // for std::iota

#include <rclcpp/rclcpp.hpp>
#include <rcl_interfaces/msg/parameter_descriptor.hpp>

#include <yaml-cpp/yaml.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <cctype>
#include <unordered_map>


#include <fcl/fcl.h>
#include <gurobi_c++.h>

#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <std_msgs/msg/header.hpp>

#include <filesystem>
#include <fstream>
#include <regex>
#include <set>
#include <chrono>
#include <memory>
#include <array>
#include <vector>
#include <string>
#include <limits>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <cmath>

using Clock = std::chrono::steady_clock;
using visualization_msgs::msg::Marker;
using visualization_msgs::msg::MarkerArray;

// ---------------- 基础类型 ----------------
enum class ObType { BOX, SPHERE, CYLINDER };

struct Obstacle {
  ObType type;
  Eigen::Vector3d center;     // world
  Eigen::Vector3d size;       // BOX:(sx,sy,sz)  SPHERE:(2r,2r,2r)  CYL:(2r,2r,h)
  Eigen::Vector3d rpy_deg;    // 仅对 CYL 可视化用；BOX 在碰撞里已带姿态（可视化仍以AABB样式）
  std::shared_ptr<fcl::CollisionObjectd> obj;
};

struct LinkBox {
  std::shared_ptr<fcl::CollisionObjectd> obj;
  Eigen::Isometry3d T;
  Eigen::Vector3d   size;  // (sx,sy,sz)
};
struct ConvexSetGuideData {
  bool loaded{false};

  std::vector<Eigen::Matrix<double,6,1>> q_seed_dense;
  std::vector<Eigen::Matrix<double,6,1>> q_rep;
  std::vector<Eigen::Matrix<double,6,1>> q_paths5;
  std::vector<int> sigma_dense;
  std::vector<Eigen::Vector3d> anchor_xyz;

  // NEW: each poly is A p <= b, p in R^3
  std::vector<Eigen::MatrixXd> poly_A;
  std::vector<Eigen::VectorXd> poly_b;
};

// 由两点 + 横截参数 生成细长盒（用于 link 段 & 焊枪近似）
static LinkBox make_link_box(const Eigen::Vector3d& p1, const Eigen::Vector3d& p2,
                             double half_W, double half_L) {
  Eigen::Vector3d z = p2 - p1;
  double len = z.norm();
  if (len < 1e-9) { z = {0,0,1}; len = 1e-6; } else { z /= len; }
  Eigen::Vector3d x = (std::fabs(z.x())>0.9)?Eigen::Vector3d(0,1,0):Eigen::Vector3d(1,0,0);
  Eigen::Vector3d y = z.cross(x).normalized(); x = y.cross(z).normalized();
  Eigen::Vector3d size(2*half_W, 2*half_L, len);
  Eigen::Vector3d c = 0.5*(p1+p2);

  Eigen::Isometry3d Tw = Eigen::Isometry3d::Identity();
  Tw.linear().col(0) = x; Tw.linear().col(1) = y; Tw.linear().col(2) = z;
  Tw.translation() = c;

  auto geom = std::make_shared<fcl::Boxd>(size.x(), size.y(), size.z());
  auto obj  = std::make_shared<fcl::CollisionObjectd>(geom);
  obj->setTransform(fcl::Transform3d(Tw.matrix()));
  obj->computeAABB();
  return {obj, Tw, size};
}
static LinkBox make_box_from_pose(const Eigen::Isometry3d& Tw,
                                  double sx, double sy, double sz) {
  auto geom = std::make_shared<fcl::Boxd>(sx, sy, sz);
  auto obj  = std::make_shared<fcl::CollisionObjectd>(geom);
  obj->setTransform(fcl::Transform3d(Tw.matrix()));
  obj->computeAABB();

  LinkBox lb;
  lb.obj  = obj;
  lb.T    = Tw;
  lb.size = Eigen::Vector3d(sx, sy, sz);
  return lb;
}
// ---------------- 节点 ----------------
class QPTrajOptNode : public rclcpp::Node {
public:
  explicit QPTrajOptNode(
      const rclcpp::NodeOptions& options =
        rclcpp::NodeOptions()
          .allow_undeclared_parameters(false)
          .automatically_declare_parameters_from_overrides(false))
  : Node("qp_traj_opt", options)
  {
    // ... 你原来的构造代码保持不变 ...
    print_final_path_ = this->declare_parameter<bool>("print_final_path", true);
    final_path_out_   = this->declare_parameter<std::string>("final_path_out", "");
// 例：final_path_out:="/tmp/qp_final_path.txt"  (不填则只打印不落盘)

    // ---- Debug 参数 ----
    debug_verify_viz_     = this->declare_parameter<bool>("debug_verify_viz", true);
    debug_dump_sizes_     = this->declare_parameter<bool>("debug_dump_sizes", true);
    debug_log_sub_count_  = this->declare_parameter<bool>("debug_log_sub_count", true);
    qos_transient_local_  = this->declare_parameter<bool>("qos_transient_local", false);
        // === 局部 QP 参数（连续修补用） ===
    local_trust_s_   = this->declare_parameter<double>("local_trust_s", 0.10);
    local_alpha_     = this->declare_parameter<double>("local_alpha",   1.0);
    local_mu_scale_  = this->declare_parameter<double>("local_mu_scale",1.0);
  // NEW: 局部 QP 可视化 / 日志
    debug_viz_local_qp_   = this->declare_parameter<bool>("debug_viz_local_qp", false);
    debug_log_local_qp_   = this->declare_parameter<bool>("debug_log_local_qp", true);

    // NEW:
    debug_print_q_        = this->declare_parameter<bool>("debug_print_q", false);
    


    clock_ = this->get_clock();



    // ---------------- 通用参数 ----------------
    frame_id_ = this->declare_parameter<std::string>("frame_id", "world");
    steps_    = this->declare_parameter<int>("steps", 7);

    alpha_    = this->declare_parameter<double>("alpha", 2.5);
    mu_       = this->declare_parameter<double>("mu", 0.8);
    d_safe_   = this->declare_parameter<double>("d_safe", 0.05);
    trust_s_  = this->declare_parameter<double>("trust_s", 0.2);
    // ===== Convex-set guidance =====
use_convexset_guidance_ =
    this->declare_parameter<bool>("use_convexset_guidance", false);

convexset_result_yaml_ =
    this->declare_parameter<std::string>("convexset_result_yaml", "");

convexset_q_in_degree_ =
    this->declare_parameter<bool>("convexset_q_in_degree", true);

lambda_corr_ =
    this->declare_parameter<double>("lambda_corr", 0.20);

use_anchor_pull_ =
    this->declare_parameter<bool>("use_anchor_pull", true);

use_seed_pull_ =
    this->declare_parameter<bool>("use_seed_pull", false);

lambda_seed_q_ =
    this->declare_parameter<double>("lambda_seed_q", 0.05);
use_poly_membership_penalty_ =
    this->declare_parameter<bool>("use_poly_membership_penalty", false);

lambda_poly_membership_ =
    this->declare_parameter<double>("lambda_poly_membership", 0.10);
// ===== Final global smoothing =====
enable_final_global_smooth_ =
    this->declare_parameter<bool>("enable_final_global_smooth", false);

final_global_smooth_iters_ =
    this->declare_parameter<int>("final_global_smooth_iters", 8);

final_global_smooth_alpha_ =
    this->declare_parameter<double>("final_global_smooth_alpha", 6.0);

final_global_smooth_mu_scale_ =
    this->declare_parameter<double>("final_global_smooth_mu_scale", 0.25);

final_global_smooth_lambda_corr_ =
    this->declare_parameter<double>("final_global_smooth_lambda_corr", 0.05);

final_global_smooth_trust_s_ =
    this->declare_parameter<double>("final_global_smooth_trust_s", 0.08);
// ===== NEW: 碰撞权重自适应放大 =====
    mu_scale_on_unsafe_ = this->declare_parameter<double>("mu_scale_on_unsafe", 1.0);
    mu_max_             = this->declare_parameter<double>("mu_max", 8.0);
    boost_mu_only_after_warmup_ =
        this->declare_parameter<bool>("boost_mu_only_after_warmup", true);
    // ---- 全局参数加载附近加上这几个 ----
mu_local_ = this->declare_parameter<double>("mu_local", 1.0);
mu_local_max_ = this->declare_parameter<double>("mu_local_max", 10.0);
mu_scale_on_unsafe_local_ =
    this->declare_parameter<double>("mu_scale_on_unsafe_local", 1.3);
// 如果你也想要“只在 warmup 之后才放大 local mu”，可以加一个布尔参数
boost_mu_only_after_warmup_local_ =
    this->declare_parameter<bool>("boost_mu_only_after_warmup_local", false);

// （可选）局部 warmup 轮数
warmup_safe_iters_local_ =
    this->declare_parameter<int>("warmup_safe_iters_local", 0);
local_safe_iters_ = 0;  // 记一下连续“安全局部QP”的轮数

    max_iters_        = this->declare_parameter<int>("max_iters", 20);
    iter_period_ms_   = this->declare_parameter<int>("iter_period_ms", 0);
    stop_when_safe_   = this->declare_parameter<bool>("stop_when_safe", true);

    trust_s_min_      = this->declare_parameter<double>("trust_s_min", 0.05);
    trust_s_max_      = this->declare_parameter<double>("trust_s_max", 1.0);
    tau_plus_         = this->declare_parameter<double>("tau_plus", 0.1);
    tau_minus_        = this->declare_parameter<double>("tau_minus", 0.05);
    xtol_             = this->declare_parameter<double>("xtol", 1e-3);
    mi_thresh_        = this->declare_parameter<double>("mi_thresh", 1e-6);
    max_trust_attempts_= this->declare_parameter<int>("max_trust_attempts", 3);
    warmup_safe_iters_= this->declare_parameter<int>("warmup_safe_iters", 3);

    // 提前停止（基于全局最小距离）
    stop_min_d_           = this->declare_parameter<double>("stop_min_d", 0.0);
    stop_when_min_d_ge_   = this->declare_parameter<bool>("stop_when_min_d_ge", true);
    min_d_ignore_warmup_  = this->declare_parameter<bool>("min_d_ignore_warmup", true);

    // 批量模式
    batch_in_dir_   = this->declare_parameter<std::string>("batch_in_dir", "");
    batch_glob_     = this->declare_parameter<std::string>("batch_glob", "run_*_q.txt");
    csv_out_        = this->declare_parameter<std::string>("csv_out", "qp_batch_metrics.csv");
    q_file_is_deg_default_ = this->declare_parameter<bool>("q_file_is_deg_default", true);
    batch_visualize_= this->declare_parameter<bool>("batch_visualize", false);

    // 连续安全（扫掠体）
    cont_min_d_safe_            = this->declare_parameter<double>("cont_min_d_safe", 0.0);
    cont_after_global_only_     = this->declare_parameter<bool>("cont_after_global_only", true);
    use_continuity_check_log_only_ = this->declare_parameter<bool>("use_continuity_check_log_only", true);
    local_seg_max_iters_        = this->declare_parameter<int>("local_seg_max_iters", 20);


    // 末关节固定
    fix_last_joint_to_zero_ = this->declare_parameter<bool>("fix_last_joint_to_zero", true);
    last_joint_index_       = this->declare_parameter<int>("last_joint_index", 5);
    last_joint_fixed_value_ = this->declare_parameter<double>("last_joint_fixed_value", 0.0);
    if (last_joint_index_ < 0 || last_joint_index_ > 5) {
      RCLCPP_FATAL(get_logger(), "last_joint_index must be in [0,5]");
      rclcpp::shutdown(); return;
    }

    // 关节限位
    {
      auto vmin = this->declare_parameter<std::vector<double>>(
        "joint_min", std::vector<double>(6, -2*M_PI));
      auto vmax = this->declare_parameter<std::vector<double>>(
        "joint_max", std::vector<double>(6,  2*M_PI));
      if (vmin.size()!=6 || vmax.size()!=6) {
        RCLCPP_FATAL(get_logger(), "joint_min/joint_max must have 6 elements");
        rclcpp::shutdown(); return;
      }
      for (int i=0;i<6;++i){ qmin6_(i)=vmin[i]; qmax6_(i)=vmax[i]; }
    }
    // === 基座偏置：world -> base 绕 Z 轴旋转 base_yaw_deg_（默认 180°） ===
    base_yaw_deg_ = this->declare_parameter<double>("base_yaw_deg", 180.0);

    T_world_base_.setIdentity();
    {
      double yaw = base_yaw_deg_ * M_PI / 180.0;
      T_world_base_.linear() =
          Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    }

    // 焊枪参数
    torch_enable_ = this->declare_parameter<bool>("torch_enable", true);
    torch_in_collision_ = this->declare_parameter<bool>("torch_in_collision", true);
    viz_torch_only_last_ = this->declare_parameter<bool>("viz_torch_only_last", true);
    torch_cyl1_len_ = this->declare_parameter<double>("torch_cyl1_len", 0.36);
    torch_cyl1_dia_ = this->declare_parameter<double>("torch_cyl1_dia", 0.025);
    torch_tilt_deg_ = this->declare_parameter<double>("torch_tilt_deg", 45.0);
    torch_cyl2_len_ = this->declare_parameter<double>("torch_cyl2_len", 0.11);
    torch_cyl2_dia_ = this->declare_parameter<double>("torch_cyl2_dia", 0.018);
    torch1_rgba_ = this->declare_parameter<std::vector<double>>("torch1_rgba",{0.9,0.4,0.1,0.9});
    torch2_rgba_ = this->declare_parameter<std::vector<double>>("torch2_rgba",{0.2,0.8,0.9,0.9});


    // ===== Gripper params =====
    gripper_enable_ = this->declare_parameter<bool>("gripper_enable", false);
    gripper_in_collision_ = this->declare_parameter<bool>("gripper_in_collision", true);

    // mount offset in TCP local frame
    gripper_mount_x_ = this->declare_parameter<double>("gripper_mount_x", 0.0);
    gripper_mount_y_ = this->declare_parameter<double>("gripper_mount_y", 0.0);
    gripper_mount_z_ = this->declare_parameter<double>("gripper_mount_z", 0.0);

    // mount RPY in TCP local frame
    gripper_mount_roll_deg_  = this->declare_parameter<double>("gripper_mount_roll_deg", 0.0);
    gripper_mount_pitch_deg_ = this->declare_parameter<double>("gripper_mount_pitch_deg", 0.0);
    gripper_mount_yaw_deg_   = this->declare_parameter<double>("gripper_mount_yaw_deg", 90.0);

    // simplified 3-box gripper
    gripper_palm_len_   = this->declare_parameter<double>("gripper_palm_len",   0.060);
    gripper_palm_width_ = this->declare_parameter<double>("gripper_palm_width", 0.080);
    gripper_palm_thick_ = this->declare_parameter<double>("gripper_palm_thick", 0.030);

    gripper_finger_len_   = this->declare_parameter<double>("gripper_finger_len",   0.090);
    gripper_finger_width_ = this->declare_parameter<double>("gripper_finger_width", 0.012);
    gripper_finger_thick_ = this->declare_parameter<double>("gripper_finger_thick", 0.020);
    gripper_opening_      = this->declare_parameter<double>("gripper_opening", 0.080);
    // FK-only 标志（说明：即使为 true，也会加载障碍物供度量/可视化）
    fk_only_ = this->declare_parameter<bool>("fk_only", false);



    // 初始化路径：优先文件
    init_q_file_ = this->declare_parameter<std::string>("init_q_file", "");
    bool init_from_file_ok=false;
    if (!init_q_file_.empty()) {
      std::vector<Eigen::Matrix<double,6,1>> qpath; bool is_deg=false;
      if (load_q_path_from_file_(init_q_file_, qpath, &is_deg) && qpath.size()>=2) {
        path_ = qpath; steps_=(int)path_.size(); q_start_=path_.front(); q_goal_=path_.back(); init_from_file_ok=true;
        RCLCPP_INFO(get_logger(), "[INIT] path from '%s' (%zu pts, unit=%s)", init_q_file_.c_str(), path_.size(), is_deg?"deg->rad":"rad");
      }
    }
    if (!init_from_file_ok) {
      auto qs = this->declare_parameter<std::vector<double>>("q_start_deg",{39,-67,63,-179,-97,-215});
      auto qg = this->declare_parameter<std::vector<double>>("q_goal_deg",{67,-98,117,-134,-77,-212});
      bool as_deg = this->declare_parameter<bool>("start_goal_in_degree", true);
      for(int k=0;k<6;++k){ q_start_(k)=as_deg? qs[k]*M_PI/180.0:qs[k]; q_goal_(k)=as_deg? qg[k]*M_PI/180.0:qg[k]; }
      path_.assign(steps_, Eigen::Matrix<double,6,1>::Zero());
      for(int t=0;t<steps_;++t){ double u=(steps_==1)?0.0:double(t)/(steps_-1); for(int k=0;k<6;++k) path_[t](k)=(1-u)*q_start_(k)+u*q_goal_(k); }
      RCLCPP_INFO(get_logger(), "[INIT] linear path %d pts", steps_);
    }
    // ===== Load convex-set guide if enabled =====
if (use_convexset_guidance_) {
  bool ok = load_convexset_guide_from_yaml_(convexset_result_yaml_);
  if (!ok) {
    RCLCPP_WARN(get_logger(),
      "[GUIDE] use_convexset_guidance=true but guide file load failed. Disable guidance.");
    use_convexset_guidance_ = false;
  }
}

RCLCPP_INFO(get_logger(),
  "[SWITCH] use_convexset_guidance=%s | enable_final_global_smooth=%s",
  use_convexset_guidance_ ? "true" : "false",
  enable_final_global_smooth_ ? "true" : "false");
    // ========== NEW: 手动输入一段关节角，仅用于 FK 验证 ==========

    fk_manual_q_enable_ = this->declare_parameter<bool>("fk_manual_q_enable", false);
    fk_manual_q_deg_    = this->declare_parameter<std::vector<double>>(
        "fk_manual_q_deg", std::vector<double>{});

    if (fk_manual_q_enable_) {
      if (fk_manual_q_deg_.empty() || fk_manual_q_deg_.size() % 6 != 0) {
        RCLCPP_FATAL(get_logger(),
          "[FK-MANUAL] fk_manual_q_enable=true but fk_manual_q_deg has invalid size=%zu (must be 6*N)",
          fk_manual_q_deg_.size());
        rclcpp::shutdown();
        return;
      }

      const size_t N = fk_manual_q_deg_.size() / 6;
      path_.clear();
      path_.reserve(N);

      for (size_t i = 0; i < N; ++i) {
        Eigen::Matrix<double,6,1> q;
        for (int k = 0; k < 6; ++k) {
          double q_deg = fk_manual_q_deg_[i*6 + k];
          q(k) = q_deg * M_PI / 180.0;  // deg -> rad
        }
        path_.push_back(q);
      }

      steps_  = static_cast<int>(N);
      q_start_ = path_.front();
      q_goal_  = path_.back();

      // 手动 FK 验证时，默认只做 FK，不跑 QP
      fk_only_ = true;

      RCLCPP_WARN(get_logger(),
        "[FK-MANUAL] using fk_manual_q_deg (%zu points) as path; fk_only forced true; steps=%d",
        N, steps_);
    }
    // ==========================================================

    // ---------------- 读取障碍物（自动识别：YAML 优先；失败则尝试 .scene 盒子） ----------------
    // ---------------- 读取障碍物（自动识别：YAML 优先；失败则尝试 .scene 盒子） ----------------
{
  // 默认必须读到 ≥1 个障碍物；否则直接 FATAL 退出
  const bool require_obstacles = this->declare_parameter<bool>("require_obstacles", true);

  std::string convex_scene_yaml = this->declare_parameter<std::string>(
      "convex_scene_yaml",
      "/home/hyj/iris_rviz_ws/src/iris_rviz_cpp/src/convex_box/convex_scene.yaml");
// 保存当前参数作为“基线”，便于 reset_global_state_on_new_package_() 恢复
mu_base_        = mu_;
trust_s_base_   = trust_s_;
alpha_base_     = alpha_;
max_iters_base_ = max_iters_;

  obstacles_ = build_obstacles_auto_(convex_scene_yaml);

  if (obstacles_.empty()) {
    if (require_obstacles) {
      RCLCPP_FATAL(get_logger(),
        "[OBST] failed to load obstacles from '%s' (YAML/.scene). "
        "Set require_obstacles:=false to bypass (not recommended).",
        convex_scene_yaml.c_str());
      rclcpp::shutdown();
      return;
    } else {
      RCLCPP_WARN(get_logger(),
        "[OBST] no obstacles parsed from '%s' but continue (require_obstacles=false)",
        convex_scene_yaml.c_str());
    }
  } else {
    RCLCPP_INFO(get_logger(), "[OBST] loaded %zu obstacle(s) from '%s'",
                obstacles_.size(), convex_scene_yaml.c_str());
  }
}


    // 可视化
        rclcpp::QoS qos(rclcpp::KeepLast(100));
    qos.reliable();
    if (qos_transient_local_) qos.transient_local();  // 可选：开启历史持久化

       pub_ = this->create_publisher<MarkerArray>("/qp_opt", qos);



    // 初始可视化
    {
      int idc = base_id_for_iter(0);
      publish_markers_for_iter(path_, 0, idc);
      RCLCPP_INFO(get_logger(), "[INIT] min distance = %.6f", min_true_distance(path_));
    }

    // 批量 or 单次
    do_batch_mode_ = !batch_in_dir_.empty();
    plan_start_tp_ = Clock::now();

 if (!do_batch_mode_) {
      // ========== FK-ONLY 开关：只调试 FK，不跑 QP ==========
      if (fk_only_) {
        RCLCPP_WARN(get_logger(),
          "[MODE] fk_only=true: FK-only debug mode, no QP optimization will be run.");
        // 不创建 timer，不调用 one_iter_step；上面 INIT 的一次可视化够你在 RViz 里调 DH。
      } else {
        timer_ = this->create_wall_timer(
          std::chrono::milliseconds(iter_period_ms_),
          std::bind(&QPTrajOptNode::one_iter_step, this));
      }
      // ======================================================
    } else {
      RCLCPP_INFO(get_logger(), "[BATCH] dir='%s' glob='%s' out='%s'",
                  batch_in_dir_.c_str(), batch_glob_.c_str(), csv_out_.c_str());
    }
  }
  // -------- 批量入口（main 用）--------
  bool do_batch_mode() const { return do_batch_mode_; }
  void run_batch_and_write_csv() { run_batch_and_write_csv_(); }

private:
// ====== DH 参数与实用函数 ======
// —— 放在 QPTrajOptNode 的 private: 里，任意合适位置 ——
// 简单 CSV 转义：若包含逗号/引号/换行，则用双引号包裹，并把内部引号重复一遍
static bool save_path_csv_(rclcpp::Logger logger,
                           const std::vector<Eigen::Matrix<double,6,1>>& path,
                           const std::string& filepath)
{
  if (path.empty()) {
    RCLCPP_WARN(logger, "[SAVE] path empty, skip saving: %s", filepath.c_str());
    return false;
  }
  std::ofstream ofs(filepath);
  if (!ofs.is_open()) {
    RCLCPP_ERROR(logger, "[SAVE] cannot open file: %s", filepath.c_str());
    return false;
  }

  ofs << "t,q1,q2,q3,q4,q5,q6\n";
  for (size_t t = 0; t < path.size(); ++t) {
    ofs << t;
    for (int k = 0; k < 6; ++k) ofs << "," << std::setprecision(16) << path[t](k);
    ofs << "\n";
  }
  ofs.close();
  RCLCPP_INFO(logger, "[SAVE] wrote %zu points to %s", path.size(), filepath.c_str());
  return true;
}

static void dump_final_path_(rclcpp::Logger logger,
                             const std::vector<Eigen::Matrix<double,6,1>>& path,
                             const std::string& tag = "FINAL-PATH")
{
  if (path.empty()) {
    RCLCPP_WARN(logger, "[%s] path empty", tag.c_str());
    return;
  }

  std::ostringstream oss;
  oss << "[" << tag << "] steps=" << path.size() << "\n";
  oss << "format: t | q(deg)[0..5] | q(rad)[0..5]\n";

  for (size_t t = 0; t < path.size(); ++t) {
    oss << "t=" << t << " deg=[";
    for (int k = 0; k < 6; ++k) {
      const double qdeg = path[t](k) * 180.0 / M_PI;
      oss << std::fixed << std::setprecision(3) << qdeg << (k < 5 ? ", " : "");
    }
    oss << "] rad=[";
    for (int k = 0; k < 6; ++k) {
      oss << std::fixed << std::setprecision(6) << path[t](k) << (k < 5 ? ", " : "");
    }
    oss << "]\n";
  }

  RCLCPP_INFO(logger, "%s", oss.str().c_str());
}

bool print_final_path_{true};
std::string final_path_out_;
static void dump_path_deg_rad_(rclcpp::Logger logger,
                               const std::vector<Eigen::Matrix<double,6,1>>& path,
                               const std::string& tag,
                               const std::string& out_file = "")
{
  if (path.empty()) {
    RCLCPP_WARN(logger, "[%s] path empty", tag.c_str());
    return;
  }

  std::ostringstream oss;
  oss << "[" << tag << "] steps=" << path.size() << "\n";
  oss << "format: t, q0..q5 (deg), q0..q5 (rad)\n";

  for (size_t t = 0; t < path.size(); ++t) {
    oss << "t=" << t << " deg=[";
    for (int k = 0; k < 6; ++k) {
      double qdeg = path[t](k) * 180.0 / M_PI;
      oss << std::fixed << std::setprecision(3) << qdeg << (k<5?", ":"");
    }
    oss << "] rad=[";
    for (int k = 0; k < 6; ++k) {
      oss << std::fixed << std::setprecision(6) << path[t](k) << (k<5?", ":"");
    }
    oss << "]\n";
  }

  RCLCPP_INFO(logger, "%s", oss.str().c_str());

  if (!out_file.empty()) {
    std::ofstream fout(out_file);
    if (!fout.is_open()) {
      RCLCPP_WARN(logger, "[%s] cannot open out_file='%s'", tag.c_str(), out_file.c_str());
      return;
    }
    // 写一个更适合后处理的CSV-ish文本
    fout << "# " << tag << "\n";
    fout << "# columns: t,q0_deg,q1_deg,q2_deg,q3_deg,q4_deg,q5_deg,q0_rad,q1_rad,q2_rad,q3_rad,q4_rad,q5_rad\n";
    for (size_t t = 0; t < path.size(); ++t) {
      fout << t;
      for (int k = 0; k < 6; ++k) fout << "," << (path[t](k) * 180.0 / M_PI);
      for (int k = 0; k < 6; ++k) fout << "," << path[t](k);
      fout << "\n";
    }
    fout.close();
    RCLCPP_INFO(logger, "[%s] final path saved to '%s'", tag.c_str(), out_file.c_str());
  }
}

static std::string csv_escape_(const std::string& s) {
  bool need = (s.find_first_of(",\"\n") != std::string::npos);
  if (!need) return s;
  std::string out; out.reserve(s.size() + 2);
  out.push_back('"');
  for (char c : s) {
    if (c == '"') out.push_back('"'); // 双引号转义为两个双引号
    out.push_back(c);
  }
  out.push_back('"');
  return out;
}

// ==== Baseline copies for "reset on new package" ====
double mu_base_{0.8};
double trust_s_base_{0.2};
double alpha_base_{2.5};
int    max_iters_base_{20};

// ==== Global stats / caches (safe defaults) ====
double best_cost_{std::numeric_limits<double>::infinity()};
double best_min_d_{-std::numeric_limits<double>::infinity()};
int    global_iter_counter_{0};
int    total_safe_iters_{0};
int    total_unsafe_iters_{0};

// FK cache (key 为 hash，可按需使用；这里先占位不用也没关系)
std::unordered_map<size_t, std::array<Eigen::Isometry3d,7>> fk_cache_;

// World “stamp” 用于潜在的场景变更标记（先占位）
uint64_t collision_world_stamp_{0};

// RNG（用于将来需要的随机数；这里也先占位）
std::mt19937 rng_{123456u};
uint32_t rng_seed_base_{123456u};

// 日志节流 / 批量 CSV 计数（占位）
Clock::time_point last_log_ts_{Clock::now()};
size_t csv_row_count_{0};

// 若以后想复用一个持久 Gurobi 模型，这里留一个占位。
// 目前 solve_qp_oldstyle() 每次都新建局部模型，所以为空也OK。
std::unique_ptr<GRBModel> gurobi_model_;

// 全局“不安全边”缓存（和 per-iter 的 unsafe_edges_cache_ 区分）
std::vector<std::pair<Eigen::Matrix<double,6,1>, Eigen::Matrix<double,6,1>>> unsafe_edges_global_cache_;

  double local_trust_s_{0.10};
  double local_alpha_{1.0};
  double local_mu_scale_{1.0};


// === Auto-build robot_description/_semantic when missing ===
bool auto_build_when_missing_{true};
std::string urdf_path_, srdf_path_, xacro_file_;
std::vector<std::string> xacro_args_;

// ==== utils: epsilon ====
static constexpr double EPS_PT = 1e-6;
static constexpr double EPS_N  = 1e-6;  // 法向聚类
static constexpr double EPS_D  = 1e-6;  // 平面偏移聚类

struct PlaneKey {
  // 通过量化后的(nx,ny,nz,d)聚类共面
  int nx, ny, nz, nd;
  bool operator<(const PlaneKey& o) const {
    if (nx!=o.nx) return nx<o.nx;
    if (ny!=o.ny) return ny<o.ny;
    if (nz!=o.nz) return nz<o.nz;
    return nd<o.nd;
  }
};

static PlaneKey make_plane_key(const Eigen::Vector3d& n_raw, double d_raw){
  Eigen::Vector3d n = n_raw;
  double nlen = n.norm();
  if (nlen < 1e-12) n = Eigen::Vector3d::UnitZ(), nlen=1.0;
  n /= nlen;

  // 统一法向朝向（取第一个非零分量为正）
  if (std::fabs(n.x())>EPS_N) { if (n.x()<0) { n=-n; d_raw=-d_raw; } }
  else if (std::fabs(n.y())>EPS_N){ if (n.y()<0) { n=-n; d_raw=-d_raw; } }
  else if (n.z()<0){ n=-n; d_raw=-d_raw; }

  auto Q = [](double v, double eps){ return (int)std::llround(v/eps); };
  PlaneKey key{ Q(n.x(),EPS_N), Q(n.y(),EPS_N), Q(n.z(),EPS_N), Q(d_raw,EPS_D) };
  return key;
}

// 去重（合并极近点），返回新顶点和原->新 映射
static std::pair<std::vector<Eigen::Vector3d>, std::vector<int>>
dedup_points(const std::vector<Eigen::Vector3d>& pts) {
  std::vector<Eigen::Vector3d> uniq;
  std::vector<int> map(pts.size(), -1);
  for (size_t i=0;i<pts.size();++i) {
    int id=-1;
    for (size_t j=0;j<uniq.size();++j) {
      if ((pts[i]-uniq[j]).squaredNorm()<EPS_PT*EPS_PT){ id=(int)j; break; }
    }
    if (id<0){ id=(int)uniq.size(); uniq.push_back(pts[i]); }
    map[i]=id;
  }
  return {uniq,map};
}

// 在平面(n,d)上取一个正交基(u,v)，满足 {u,v,n} 右手
static void plane_basis(const Eigen::Vector3d& n, Eigen::Vector3d& u, Eigen::Vector3d& v){
  Eigen::Vector3d t = (std::fabs(n.x())>0.8)? Eigen::Vector3d::UnitY():Eigen::Vector3d::UnitX();
  u = (t - n*(t.dot(n))).normalized();
  v = n.cross(u).normalized();
}

// 2D 凸包（Graham-Scan / Andrew），输入平面内点的(u,v)坐标，输出外轮廓索引（逆时针）
// 输入: pts2=[(u,v)], 输出: hull_idx 是 pts2 的索引序列
static std::vector<int> convex_hull_2d(const std::vector<Eigen::Vector2d>& pts2){
  const int n = (int)pts2.size();
  if (n<=2){ std::vector<int> idx(n); std::iota(idx.begin(),idx.end(),0); return idx; }

  // 带原索引的排序
  std::vector<std::pair<Eigen::Vector2d,int>> P; P.reserve(n);
  for (int i=0;i<n;++i) P.push_back({pts2[i], i});
  std::sort(P.begin(), P.end(), [](auto& a, auto& b){
    if (a.first.x()==b.first.x()) return a.first.y()<b.first.y();
    return a.first.x()<b.first.x();
  });

  auto cross = [](const Eigen::Vector2d& a, const Eigen::Vector2d& b, const Eigen::Vector2d& c){
    return (b.x()-a.x())*(c.y()-a.y()) - (b.y()-a.y())*(c.x()-a.x());
  };

  std::vector<int> H(2*n);
  int k=0;
  // lower
  for (int i=0;i<n;++i){
    while (k>=2 && cross( P[H[k-2]].first, P[H[k-1]].first, P[i].first) <= 0 ) --k;
    H[k++]=i;
  }
  // upper
  for (int i=n-2, t=k+1; i>=0; --i){
    while (k>=t && cross( P[H[k-2]].first, P[H[k-1]].first, P[i].first) <= 0 ) --k;
    H[k++]=i;
  }
  H.resize(k-1);

  // 转回原索引
  std::vector<int> out; out.reserve(H.size());
  for (int h : H) out.push_back(P[h].second);
  return out;
}
// === [WATERTIGHT FIX] polygonal convex hull types ===
struct HullPoly { Eigen::Vector3d n; double d; std::vector<int> vidx; }; // one face: outward normal + polygon (CCW)
struct WatertightHull {
  std::vector<Eigen::Vector3d> verts; // deduped vertices
  std::vector<HullPoly> faces;        // polygonal faces
};

// === [WATERTIGHT FIX] build watertight convex from points (plane clustering + 2D hull per plane) ===
static WatertightHull build_watertight_convex_from_points(const std::vector<Eigen::Vector3d>& pts_in){
  WatertightHull H;
  auto [pts, map_id] = dedup_points(pts_in);
  const int n = (int)pts.size();
  H.verts = pts;
  if (n < 4) return H;

  // centroid
  Eigen::Vector3d C=Eigen::Vector3d::Zero(); for (auto& p:pts) C+=p; C/=double(n);

  std::map<PlaneKey, std::vector<int>> plane_to_vids;
  std::map<PlaneKey, Eigen::Vector4d>  plane_to_eq;
  const double epsA=1e-12, epsS=1e-9;

  for (int i=0;i<n;i++) for (int j=i+1;j<n;j++) for (int k=j+1;k<n;k++) {
    Eigen::Vector3d p0=pts[i], p1=pts[j], p2=pts[k];
    Eigen::Vector3d nrm=(p1-p0).cross(p2-p0); double an=nrm.norm();
    if (an<epsA) continue; nrm/=an;
    double d = nrm.dot(p0);

    double mn=1e18, mx=-1e18;
    for (int m=0;m<n;m++) { double s=nrm.dot(pts[m])-d; mn=std::min(mn,s); mx=std::max(mx,s); if (mn<-epsS && mx>epsS){ mn=-1; mx=1; break; } }
    if (!((mx<=epsS) || (mn>=-epsS))) continue;

    if (nrm.dot(C)-d > 0) { nrm=-nrm; d=-d; } // outward

    PlaneKey key = make_plane_key(nrm, d);
    auto& vec = plane_to_vids[key];
    if (plane_to_eq.find(key)==plane_to_eq.end()) plane_to_eq[key]=Eigen::Vector4d(nrm.x(),nrm.y(),nrm.z(),d);

    auto push=[&](int id){ if (std::find(vec.begin(),vec.end(),id)==vec.end()) vec.push_back(id); };
    push(i); push(j); push(k);
    for (int t=0;t<n;++t){ double s=nrm.dot(pts[t])-d; if (std::fabs(s)<=5e-6) push(t); }
  }

  for (auto& kv : plane_to_vids) {
    const auto& eq = plane_to_eq[kv.first];
    Eigen::Vector3d n(eq[0],eq[1],eq[2]); double d=eq[3];
    if (kv.second.size()<3) continue;

    Eigen::Vector3d u,v; plane_basis(n,u,v);

    std::vector<Eigen::Vector2d> pts2; pts2.reserve(kv.second.size());
    for (int id: kv.second){ const auto& P=pts[id]; pts2.emplace_back(P.dot(u), P.dot(v)); }
    auto hull_local = convex_hull_2d(pts2);
    if (hull_local.size()<3) continue;

    HullPoly poly; poly.n=n; poly.d=d; poly.vidx.reserve(hull_local.size());
    for (int h : hull_local) poly.vidx.push_back(kv.second[h]);
    H.faces.push_back(std::move(poly));
  }
  return H;
}




  // ---------------- YAML/SCENE -> Obstacles ----------------
  std::vector<Obstacle> build_obstacles_from_yaml_(const std::string& yaml_path) {
    std::vector<Obstacle> obs;
    try {
      YAML::Node root = YAML::LoadFile(yaml_path);
      auto V3 = [](const YAML::Node& n){ return Eigen::Vector3d(n[0].as<double>(), n[1].as<double>(), n[2].as<double>()); };

      if (root["boxes"] && root["boxes"].IsSequence()) {
        for (const auto& b : root["boxes"]) {
          Eigen::Vector3d mn = V3(b["min"]), mx = V3(b["max"]);
          Eigen::Vector3d sz = (mx - mn).cwiseAbs();
          Eigen::Vector3d c  = 0.5*(mx + mn);
          auto geom = std::make_shared<fcl::Boxd>(sz.x(), sz.y(), sz.z());
          auto obj  = std::make_shared<fcl::CollisionObjectd>(geom);
          Eigen::Isometry3d T=Eigen::Isometry3d::Identity(); T.translation()=c;
          obj->setTransform(fcl::Transform3d(T.matrix())); obj->computeAABB();
          obs.push_back({ObType::BOX, c, sz, Eigen::Vector3d::Zero(), obj});
        }
      }
      if (root["spheres"] && root["spheres"].IsSequence()) {
        for (const auto& s : root["spheres"]) {
          Eigen::Vector3d c = V3(s["center"]); double r=s["r"].as<double>();
          auto geom=std::make_shared<fcl::Sphered>(r);
          auto obj =std::make_shared<fcl::CollisionObjectd>(geom);
          Eigen::Isometry3d T=Eigen::Isometry3d::Identity(); T.translation()=c;
          obj->setTransform(fcl::Transform3d(T.matrix())); obj->computeAABB();
          obs.push_back({ObType::SPHERE, c, Eigen::Vector3d(2*r,2*r,2*r), Eigen::Vector3d::Zero(), obj});
        }
      }
      if (root["cylinders"] && root["cylinders"].IsSequence()) {
        auto deg2rad=[](double d){return d*M_PI/180.0;};
        for (const auto& cy : root["cylinders"]) {
          Eigen::Vector3d c = V3(cy["center"]);
          double r = cy["r"].as<double>(), h = cy["h"].as<double>();
          Eigen::Vector3d rpy = cy["rpy_deg"] ? V3(cy["rpy_deg"]) : Eigen::Vector3d::Zero();

          auto geom = std::make_shared<fcl::Cylinderd>(r, h);
          auto obj  = std::make_shared<fcl::CollisionObjectd>(geom);

          Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
          T.linear() =
            (Eigen::AngleAxisd(deg2rad(rpy.z()),Eigen::Vector3d::UnitZ())*
             Eigen::AngleAxisd(deg2rad(rpy.y()),Eigen::Vector3d::UnitY())*
             Eigen::AngleAxisd(deg2rad(rpy.x()),Eigen::Vector3d::UnitX())).toRotationMatrix();
          T.translation() = c;

          obj->setTransform(fcl::Transform3d(T.matrix())); obj->computeAABB();
          obs.push_back({ObType::CYLINDER, c, Eigen::Vector3d(2*r,2*r,h), rpy, obj});
        }
      }
      RCLCPP_INFO(get_logger(), "[OBST] %zu loaded from YAML '%s'", obs.size(), yaml_path.c_str());
    } catch (const std::exception& e) {
      // 让上层自动切换到 .scene
      RCLCPP_WARN(get_logger(), "[OBST] YAML load failed: %s (will try legacy .scene)", e.what());
    }
    return obs;
  }

  // 旧 .scene 格式（仅 box）：示例块
  // * Box_0
  // cx cy cz
  // qx qy qz qw
  // 1
  // box
  // sx sy sz          (整尺寸)
  // 0 0 0
  // 0 0 0 1
  // 0 0 0 0
  // 0
  // .                 （可选分隔符，忽略）
  std::vector<Obstacle> build_obstacles_from_legacy_scene_boxes_(const std::string& scene_path) {
  std::vector<Obstacle> obs;
  std::ifstream in(scene_path);
  if (!in.is_open()) {
    RCLCPP_ERROR(get_logger(), "[OBST] cannot open scene: %s", scene_path.c_str());
    return obs;
  }

  auto quatToR = [](double x,double y,double z,double w)->Eigen::Matrix3d {
    Eigen::Quaterniond q(w,x,y,z);  // 文件里是 x y z w，我们这里构造 q(w,x,y,z)
    q.normalize();
    return q.toRotationMatrix();
  };

  auto skip_rest_of_line = [&](void){ std::string dummy; std::getline(in, dummy); };
  auto skip_n_lines      = [&](int n){ for (int i=0;i<n;++i){ std::string dummy; std::getline(in, dummy); } };

  std::string tok;
  while (in >> tok) {
    if (tok == "*") {
      std::string name; 
      in >> name;      // Box_0 / Cylinder_0 等

      // 1) 位置
      double cx,cy,cz; 
      in >> cx >> cy >> cz;

      // 2) 四元数 (x y z w)
      double qx,qy,qz,qw; 
      in >> qx >> qy >> qz >> qw;

      // 3) 占位 1
      int one; 
      in >> one;

      // 4) 类型
      std::string typ; 
      in >> typ;

      if (typ == "box") {
        // 5) 尺寸（整尺寸）
        double sx,sy,sz; 
        in >> sx >> sy >> sz;

        // 后面 4 行占位
        skip_rest_of_line();
        skip_n_lines(4);

        Eigen::Vector3d c(cx,cy,cz);
        Eigen::Matrix3d R = quatToR(qx,qy,qz,qw);

        auto geom = std::make_shared<fcl::Boxd>(sx, sy, sz);
        auto obj  = std::make_shared<fcl::CollisionObjectd>(geom);

        Eigen::Isometry3d Tw = Eigen::Isometry3d::Identity();
        Tw.linear()      = R;
        Tw.translation() = c;
        obj->setTransform(fcl::Transform3d(Tw.matrix()));
        obj->computeAABB();

        Obstacle ob;
        ob.type     = ObType::BOX;
        ob.center   = c;
        ob.size     = Eigen::Vector3d(sx,sy,sz);
        ob.rpy_deg  = Eigen::Vector3d::Zero();   // 可视化时我们会从 obj 里读旋转
        ob.obj      = obj;
        obs.push_back(std::move(ob));
      }
      else if (typ == "cylinder") {
        // 5) 半径 + 高度
        double r, h;
        in >> r >> h;

        // 后面 4 行占位
        skip_rest_of_line();
        skip_n_lines(4);

        Eigen::Vector3d c(cx,cy,cz);
        Eigen::Matrix3d R = quatToR(qx,qy,qz,qw);

        // FCL cylinder: radius r, height h
        auto geom = std::make_shared<fcl::Cylinderd>(r, h);
        auto obj  = std::make_shared<fcl::CollisionObjectd>(geom);

        Eigen::Isometry3d Tw = Eigen::Isometry3d::Identity();
        Tw.linear()      = R;
        Tw.translation() = c;
        obj->setTransform(fcl::Transform3d(Tw.matrix()));
        obj->computeAABB();

        Obstacle ob;
        ob.type    = ObType::CYLINDER;
        ob.center  = c;
        ob.size    = Eigen::Vector3d(2*r, 2*r, h); // scale.x/y = 直径, scale.z = 高度
        ob.rpy_deg = Eigen::Vector3d::Zero();      // 实际姿态从 obj->getTransform() 读
        ob.obj     = obj;
        obs.push_back(std::move(ob));
      }
      else {
        // 其他未知类型：按原格式跳过
        skip_rest_of_line();
        skip_n_lines(4);
      }
    }
    else if (tok == ".") {
      // 分隔符，忽略
      continue;
    }
    else {
      // 其他 token：丢弃本行
      skip_rest_of_line();
    }
  }

  RCLCPP_INFO(get_logger(),
              "[OBST] %zu object(s) loaded from legacy scene '%s'",
              obs.size(), scene_path.c_str());
  return obs;
}


  // 自动选择：YAML 优先；若 0 或 YAML 解析失败则尝试旧 .scene 盒子
  std::vector<Obstacle> build_obstacles_auto_(const std::string& path) {
    auto ends_with = [](const std::string& s, const std::string& suf){
      if (s.size() < suf.size()) return false;
      return std::equal(suf.rbegin(), suf.rend(), s.rbegin(),
                        [](char a, char b){ return std::tolower(a)==std::tolower(b); });
    };

    std::vector<Obstacle> obs;
    bool tried_yaml = false;

    if (ends_with(path, ".yaml") || ends_with(path, ".yml")) {
      tried_yaml = true;
      obs = build_obstacles_from_yaml_(path);
      if (!obs.empty()) return obs;
    } else {
      // 即便不是 .yaml，先试 YAML（兼容有人误填后缀）
      obs = build_obstacles_from_yaml_(path);
      if (!obs.empty()) return obs;
    }

    // YAML 没读到 → 旧 .scene 仅 box
    auto obs2 = build_obstacles_from_legacy_scene_boxes_(path);
    if (!obs2.empty()) return obs2;

    if (tried_yaml) {
      RCLCPP_WARN(get_logger(), "[OBST] Neither YAML nor legacy scene produced obstacles: %s", path.c_str());
    }
    return {};
  }




 // ======= DH kinematics (pure, unified) =======
struct DH { double a, d, alpha, theta_off; }; // 标准DH: RotZ(theta)*TransZ(d)*TransX(a)*RotX(alpha)

// 默认 UR5（请与你的 URDF 对齐；必要时调整 a/d/alpha 符号或 theta_off）
std::array<DH,6> dh_{{
  { 0.000000, 0.089159,  +M_PI/2.0, 0.0 },  // J1
  {-0.425000, 0.000000,   0.0,       0.0 },  // J2
  {-0.392250, 0.000000,   0.0,       0.0 },  // J3
  { 0.000000, 0.109150,  +M_PI/2.0,  0.0 },  // J4
  { 0.000000, 0.094650,  -M_PI/2.0,  0.0 },  // J5
  { 0.000000, 0.082300,   0.0,       0.0 }   // J6（法兰前）
}};

// 统一只用世界->基座、法兰->TCP
Eigen::Isometry3d T_world_base_  = Eigen::Isometry3d::Identity();
Eigen::Isometry3d T_flange_tcp_  = Eigen::Isometry3d::Identity();

// 统一版 T_dh（参数顺序固定为 a, d, alpha, theta）
static inline Eigen::Isometry3d T_dh(double a, double d, double alpha, double theta)
{
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  const double ca = std::cos(alpha), sa = std::sin(alpha);
  const double ct = std::cos(theta), st = std::sin(theta);

  // 旋转部分：已经是标准 DH 的 R = Rz(theta)*Rx(alpha) 对应的 3x3
  T.linear() <<
      ct, -st*ca,  st*sa,
      st,  ct*ca, -ct*sa,
       0,     sa,     ca;

  // **关键修正：平移用 (a*ct, a*st, d)，而不是 (a,0,d)**
  T.translation() = Eigen::Vector3d(a * ct, a * st, d);

  return T;
}


// 唯一正解：Tj[0]=base系；Tj[6]=TCP（已乘法兰->TCP）
void compute_fk_chain_(const Eigen::Matrix<double,6,1>& q,
                       std::array<Eigen::Isometry3d,7>& Tj) const
{
  Eigen::Isometry3d T = T_world_base_;
  Tj[0] = T;                                   // world->base
  for (int i=0; i<6; ++i) {
    const double th = q(i) + dh_[i].theta_off;
    T = T * T_dh(dh_[i].a, dh_[i].d, dh_[i].alpha, th);
    Tj[i+1] = T;                               // world->link(i)末
  }
  // 末尾乘 flange->tcp，使 Tj[6] 直接是 world->tcp
  Tj[6] = Tj[6] * T_flange_tcp_;
}


// DH 版点位置雅可比（世界系）
Eigen::Matrix<double,3,6>
jacobian_point_world_dh_(const Eigen::Matrix<double,6,1>& q,
                         int link_seg_index,
                         const Eigen::Vector3d& p_world) const
{
  std::array<Eigen::Isometry3d,7> Tj;
  compute_fk_chain_(q, Tj);

  Eigen::Matrix<double,3,6> J; J.setZero();
  // 关节 j 的轴/原点：以“关节前”的坐标系为基准（即 Tj[j] 之前的那一帧）
  for (int j = 0; j < 6; ++j) {
    const Eigen::Vector3d o_j = Tj[j].translation();
    const Eigen::Vector3d z_j = Tj[j].linear() * Eigen::Vector3d::UnitZ();
    J.col(j) = z_j.cross(p_world - o_j);
  }
  // 保留到当前段（清零远端列）
  for (int j = link_seg_index + 1; j < 6; ++j) J.col(j).setZero();
  return J;
}

// 供现有代码调用的包装：现阶段用 DH 兜底
Eigen::Matrix<double,3,6>
compute_position_jacobian_moveit_base_(const Eigen::Matrix<double,6,1>& q,
                                       int link_seg_index,
                                       const Eigen::Vector3d& p_world) const
{
  // 未来如果 robot_state_ 就绪，可切换到 MoveIt 真·Jacobian：
  // if (robot_state_ && jmg_) { ... return J_from_moveit; }
  return jacobian_point_world_dh_(q, link_seg_index, p_world);
}
Eigen::Isometry3d get_tcp_pose_world_(const Eigen::Matrix<double,6,1>& q) const
{
  std::array<Eigen::Isometry3d,7> Tj;
  compute_fk_chain_(q, Tj);
  return Tj[6];
}
Eigen::Isometry3d make_gripper_mount_pose_(const Eigen::Isometry3d& T_world_tcp) const
{
  const double rr = gripper_mount_roll_deg_  * M_PI / 180.0;
  const double rp = gripper_mount_pitch_deg_ * M_PI / 180.0;
  const double ry = gripper_mount_yaw_deg_   * M_PI / 180.0;

  Eigen::Isometry3d T_mount = Eigen::Isometry3d::Identity();
  T_mount.linear() =
      (Eigen::AngleAxisd(ry, Eigen::Vector3d::UnitZ()) *
       Eigen::AngleAxisd(rp, Eigen::Vector3d::UnitY()) *
       Eigen::AngleAxisd(rr, Eigen::Vector3d::UnitX())).toRotationMatrix();

  T_mount.translation() =
      Eigen::Vector3d(gripper_mount_x_, gripper_mount_y_, gripper_mount_z_);

  return T_world_tcp * T_mount;
}
std::vector<LinkBox>
build_gripper_boxes_for_tcp_(const Eigen::Isometry3d& T_world_tcp) const
{
  std::vector<LinkBox> out;
  out.reserve(3);

  const Eigen::Isometry3d T_g = make_gripper_mount_pose_(T_world_tcp);

  const Eigen::Matrix3d R = T_g.linear();
  const Eigen::Vector3d p = T_g.translation();

  const Eigen::Vector3d ey = R.col(1);
  const Eigen::Vector3d ez = R.col(2);

  // ===== 1) palm =====
  Eigen::Isometry3d T_palm = Eigen::Isometry3d::Identity();
  T_palm.linear() = R;
  T_palm.translation() = p + 0.5 * gripper_palm_len_ * ez;

  out.push_back(make_box_from_pose(
      T_palm,
      gripper_palm_thick_,
      gripper_palm_width_,
      gripper_palm_len_
  ));

  // ===== 2) left finger =====
  Eigen::Isometry3d T_l = Eigen::Isometry3d::Identity();
  T_l.linear() = R;
  T_l.translation() =
      p
      + (gripper_palm_len_ + 0.5 * gripper_finger_len_) * ez
      + 0.5 * gripper_opening_ * ey;

  out.push_back(make_box_from_pose(
      T_l,
      gripper_finger_thick_,
      gripper_finger_width_,
      gripper_finger_len_
  ));

  // ===== 3) right finger =====
  Eigen::Isometry3d T_r = Eigen::Isometry3d::Identity();
  T_r.linear() = R;
  T_r.translation() =
      p
      + (gripper_palm_len_ + 0.5 * gripper_finger_len_) * ez
      - 0.5 * gripper_opening_ * ey;

  out.push_back(make_box_from_pose(
      T_r,
      gripper_finger_thick_,
      gripper_finger_width_,
      gripper_finger_len_
  ));

  return out;
}

    // ---------------- Link boxes ----------------
  std::array<Eigen::Vector3d,8> box_world_corners_(const LinkBox& lb) const {
    const double hx = lb.size.x()*0.5, hy = lb.size.y()*0.5, hz = lb.size.z()*0.5;
    Eigen::Vector3d v[8] = {
      {+hx,+hy,+hz}, {+hx,+hy,-hz}, {+hx,-hy,-hz}, {+hx,-hy,+hz},
      {-hx,+hy,+hz}, {-hx,+hy,-hz}, {-hx,-hy,-hz}, {-hx,-hy,+hz}
    };
    std::array<Eigen::Vector3d,8> out;
    for (int i=0;i<8;++i) out[i] = lb.T * v[i];
    return out;
  }
  // 由两帧 box 的 16 个角点，构造 swept mesh（三角网），实现 A(t)=convhull(A,A')
  std::shared_ptr<fcl::CollisionObjectd>
  make_swept_box_mesh_(const LinkBox& lbA, const LinkBox& lbB) const {
    using BVH = fcl::BVHModel<fcl::OBBRSSd>;
    auto model = std::make_shared<BVH>();

    // 顶点：两端各 8 个（索引 0..7 是 A，8..15 是 B）
    auto A = box_world_corners_(lbA);
    auto B = box_world_corners_(lbB);

    std::vector<Eigen::Vector3d> P(16);
    for (int i=0;i<8;++i)  P[i]   = A[i];
    for (int i=0;i<8;++i)  P[8+i] = B[i];

    // 小工具：把 Eigen::Vector3d 三角加进 BVH
    auto addTri = [&](int i, int j, int k){
      const auto& a = P[i];
      const auto& b = P[j];
      const auto& c = P[k];
      fcl::Vector3d pa(a.x(), a.y(), a.z());
      fcl::Vector3d pb(b.x(), b.y(), b.z());
      fcl::Vector3d pc(c.x(), c.y(), c.z());
      model->addTriangle(pa, pb, pc);
    };

    model->beginModel();

    // (1) 两端端盖：A 盒子 + B 盒子
    //   注意，这个面序必须和 box_world_corners_ / append_box_as_triangles 一致
    int f[12][3] = {
      {0,1,2},{0,2,3},  // +X 面
      {4,6,5},{4,7,6},  // -X 面
      {0,4,5},{0,5,1},  // +Y 面
      {3,2,6},{3,6,7},  // -Y 面
      {0,3,7},{0,7,4},  // +Z 面
      {1,5,6},{1,6,2},  // -Z 面
    };
    // A 端
    for (auto& tri : f) {
      addTri(tri[0], tri[1], tri[2]);
    }
    // B 端（索引整体 +8）
    for (auto& tri : f) {
      addTri(8 + tri[0], 8 + tri[1], 8 + tri[2]);
    }

    // (2) 侧面：对 A 的 6 个面做“挤出”，把每个四边形连接到 B 的对应角点
    // faces4 定义每个面的 4 个顶点环（对应上面的 f 面组）
    int faces4[6][4] = {
      {0,1,2,3}, // 面组1（对应 f[0],f[1]）
      {4,5,6,7}, // 面组2（对应 f[2],f[3]）
      {0,1,5,4}, // 面组3（对应 f[4],f[5]）
      {3,2,6,7}, // 面组4（对应 f[6],f[7]）
      {0,3,7,4}, // 面组5（对应 f[8],f[9]）
      {1,5,6,2}, // 面组6（对应 f[10],f[11]）
    };

    auto addQuadExtrude = [&](int a0, int a1){
      // A 边 (a0,a1)，B 边 (a0+8, a1+8)，四边形拆成两个三角
      // 顺序：(A[a0], A[a1], B[a1]) + (A[a0], B[a1], B[a0])
      addTri(a0,   a1,   8 + a1);
      addTri(a0,   8+a1, 8 + a0);
    };

    for (int fi=0; fi<6; ++fi) {
      int v[4] = {faces4[fi][0], faces4[fi][1], faces4[fi][2], faces4[fi][3]};
      // 四条边都挤出一次
      addQuadExtrude(v[0], v[1]);
      addQuadExtrude(v[1], v[2]);
      addQuadExtrude(v[2], v[3]);
      addQuadExtrude(v[3], v[0]);
    }

    model->endModel();

    auto obj = std::make_shared<fcl::CollisionObjectd>(model);
    // 顶点已经是 world 坐标，所以 transform 设 Identity
    obj->setTransform(fcl::Transform3d(Eigen::Isometry3d::Identity().matrix()));
    obj->computeAABB();
    return obj;
  }

std::vector<LinkBox> build_link_boxes_for_q(const Eigen::Matrix<double,6,1>& q) const {
  std::array<Eigen::Isometry3d,7> Tj; 
  compute_fk_chain_(q, Tj);

  std::array<Eigen::Vector3d,7> P;
  for (int i = 0; i <= 6; ++i) P[i] = Tj[i].translation();

  std::vector<LinkBox> boxes;
  boxes.reserve(16);

  // 机器人 6 段本体
  for (int i = 0; i < 6; ++i) {
    boxes.push_back(make_link_box(P[i], P[i+1], 0.025, 0.025));
  }

  const Eigen::Isometry3d T_world_tcp = Tj[6];

  // ===== end-effector switch =====
  // gripper_enable_ == true  -> use gripper
  // otherwise                -> use original torch
  if (gripper_enable_) {
    if (gripper_in_collision_) {
      auto gboxes = build_gripper_boxes_for_tcp_(T_world_tcp);
      boxes.insert(boxes.end(), gboxes.begin(), gboxes.end());
    }
  } else {
    if (torch_enable_ && torch_in_collision_) {
      Eigen::Vector3d p_tcp = T_world_tcp.translation();
      Eigen::Matrix3d R0    = T_world_tcp.linear();

      // torch segment 1
      Eigen::Vector3d dir1 = R0 * Eigen::Vector3d::UnitZ();
      Eigen::Vector3d p1 = p_tcp;
      Eigen::Vector3d p2 = p_tcp + torch_cyl1_len_ * dir1;
      double r1 = 0.5 * torch_cyl1_dia_;
      boxes.push_back(make_link_box(p1, p2, r1, r1));

      // torch segment 2
      const double tilt = -torch_tilt_deg_ * M_PI / 180.0;
      Eigen::Matrix3d R_tilt =
          Eigen::AngleAxisd(tilt, Eigen::Vector3d::UnitX()).toRotationMatrix();
      Eigen::Vector3d dir2 = R0 * (R_tilt * Eigen::Vector3d::UnitZ());
      Eigen::Vector3d p3 = p2;
      Eigen::Vector3d p4 = p3 + torch_cyl2_len_ * dir2;
      double r2 = 0.5 * torch_cyl2_dia_;
      boxes.push_back(make_link_box(p3, p4, r2, r2));
    }
  }

  return boxes;
}

  // ---------------- 连续安全：扫掠凸包 ----------------
  struct ConvexHullData {
    std::vector<Eigen::Vector3d> verts;
    std::vector<Eigen::Vector3i> faces; // 三角面
  };

 


  // 用 watertight hull + 扇形三角化 生成可视化用的三角网
ConvexHullData make_link_swept_hull_data_(const LinkBox& lbA, const LinkBox& lbB) const {
  auto A = box_world_corners_(lbA);
  auto B = box_world_corners_(lbB);
  std::vector<Eigen::Vector3d> pts; pts.reserve(16);
  for (int i=0;i<8;++i) pts.push_back(A[i]);
  for (int i=0;i<8;++i) pts.push_back(B[i]);

  // 生成 watertight 多边形面
  WatertightHull H = build_watertight_convex_from_points(pts);

  ConvexHullData out;
  out.verts = H.verts;

  // 扇形三角化每个多边形面 [v0, v1, v2, ..., v(m-1)] -> (v0,v1,v2), (v0,v2,v3), ...
  for (const auto& face : H.faces) {
    if (face.vidx.size() < 3) continue;
    for (size_t i = 1; i + 1 < face.vidx.size(); ++i) {
      out.faces.emplace_back(
        Eigen::Vector3i(face.vidx[0], face.vidx[i], face.vidx[i+1])
      );
    }
  }

  // 如果退化（极少数数值边界），兜底一个 AABB 盒，避免空三角
  if (out.verts.size() < 4 || out.faces.empty()) {
    Eigen::Vector3d mn = out.verts.empty()? Eigen::Vector3d::Zero() : out.verts[0];
    Eigen::Vector3d mx = mn;
    for (auto& p : out.verts) { mn = mn.cwiseMin(p); mx = mx.cwiseMax(p); }
    out.verts.clear(); out.faces.clear();

    // 构建一个细 AABB 盒
    Eigen::Vector3d sz = (mx - mn).cwiseMax(Eigen::Vector3d::Constant(1e-6));
    Eigen::Vector3d c  = 0.5*(mn + mx);
    std::array<Eigen::Vector3d,8> vv = {
      c + Eigen::Vector3d(+0.5*sz.x(), +0.5*sz.y(), +0.5*sz.z()),
      c + Eigen::Vector3d(+0.5*sz.x(), +0.5*sz.y(), -0.5*sz.z()),
      c + Eigen::Vector3d(+0.5*sz.x(), -0.5*sz.y(), -0.5*sz.z()),
      c + Eigen::Vector3d(+0.5*sz.x(), -0.5*sz.y(), +0.5*sz.z()),
      c + Eigen::Vector3d(-0.5*sz.x(), +0.5*sz.y(), +0.5*sz.z()),
      c + Eigen::Vector3d(-0.5*sz.x(), +0.5*sz.y(), -0.5*sz.z()),
      c + Eigen::Vector3d(-0.5*sz.x(), -0.5*sz.y(), -0.5*sz.z()),
      c + Eigen::Vector3d(-0.5*sz.x(), -0.5*sz.y(), +0.5*sz.z())
    };
    out.verts.assign(vv.begin(), vv.end());
    int f[12][3]={{0,1,2},{0,2,3},{4,6,5},{4,7,6},{0,4,5},{0,5,1},{3,2,6},{3,6,7},{0,3,7},{0,7,4},{1,5,6},{1,6,2}};
    for (auto& tri : f) out.faces.emplace_back(Eigen::Vector3i(tri[0],tri[1],tri[2]));
  }

  return out;
}


  void append_hull_triangles_to_marker_(const ConvexHullData& hull, Marker& tri) const {
    auto toPoint=[&](const Eigen::Vector3d& p){ geometry_msgs::msg::Point P; P.x=p.x(); P.y=p.y(); P.z=p.z(); return P; };
    for (auto& f : hull.faces) {
      tri.points.push_back(toPoint(hull.verts[f[0]]));
      tri.points.push_back(toPoint(hull.verts[f[1]]));
      tri.points.push_back(toPoint(hull.verts[f[2]]));
    }
  }





  bool edge_continuous_safe_convexbox_(const Eigen::Matrix<double,6,1>& qA,
                                     const Eigen::Matrix<double,6,1>& qB) const {
  if (obstacles_.empty()) return true;

  auto boxesA = build_link_boxes_for_q(qA);
  auto boxesB = build_link_boxes_for_q(qB);

  const int K_all = static_cast<int>(boxesA.size()); // 可能是 6 或 8

  for (int k = 0; k < K_all; ++k) {
    auto swept = make_swept_box_mesh_(boxesA[k], boxesB[k]);

    double best = std::numeric_limits<double>::infinity();
    for (const auto& ob : obstacles_) {
      fcl::DistanceRequestd req;
      req.enable_signed_distance = true;
      fcl::DistanceResultd res;

      double d = fcl::distance(swept.get(), ob.obj.get(), req, res);
      if (d < best) best = d;
      if (best < cont_min_d_safe_) break;
    }
    if (best < cont_min_d_safe_) {
      return false;
    }
  }
  return true;
}




  // ---------------- 违规项/真实代价 ----------------
  struct Violation {
    int t{-1}, k{-1}; double d{1e9};
    Eigen::Vector3d p_robot, p_obs, n;   // world
    Eigen::RowVectorXd wn;               // 1x6
  };

  std::vector<Violation> collect_violations(const std::vector<Eigen::Matrix<double,6,1>>& path) const {
  std::vector<Violation> vios;
  if (obstacles_.empty()) return vios;

  const int T = steps_;
  for (int t = 0; t < T; ++t) {
    // 构造 6 段连杆盒 +（可选）焊枪两段盒（用于并入 k=5 的检测）
    auto link_boxes = build_link_boxes_for_q(path[t]);
    const int K_link = 6;                    // 只遍历 0..5
    const int K_all  = (int)link_boxes.size(); // 可能是 6 或 8（含焊枪）

    for (int k = 0; k < K_link; ++k) {
      double best_d = std::numeric_limits<double>::infinity();
      Eigen::Vector3d best_pr, best_po;
      int best_source = k;  // 记录来自哪一个盒（k 或 焊枪）

      // ---- 对所有障碍，求距离最小值 ----
      for (const auto& ob : obstacles_) {
        // 1) 先看第 k 段本体
        {
          fcl::DistanceRequestd req; req.enable_nearest_points = true; req.enable_signed_distance = true;
          fcl::DistanceResultd  res;
          double d = fcl::distance(link_boxes[k].obj.get(), ob.obj.get(), req, res);
          if (d < best_d) { best_d = d; best_pr = res.nearest_points[0]; best_po = res.nearest_points[1]; best_source = k; }
        }
        // 2) 如果是最后一段（k==5），把两段焊枪也并入取最小距离
        if (k == 5 && K_all > K_link) {
          for (int tk = K_link; tk < K_all; ++tk) {
            fcl::DistanceRequestd req; req.enable_nearest_points = true; req.enable_signed_distance = true;
            fcl::DistanceResultd  res;
            double d = fcl::distance(link_boxes[tk].obj.get(), ob.obj.get(), req, res);
            if (d < best_d) { best_d = d; best_pr = res.nearest_points[0]; best_po = res.nearest_points[1]; best_source = tk; }
          }
        }
      } // end obstacles loop

      // 违反才加入
      if (best_d < d_safe_) {
        Violation vio; vio.t = t; vio.k = k;  // 注意：k 仍然是 0..5（不把焊枪作为新段）
        vio.d = best_d; vio.p_robot = best_pr; vio.p_obs = best_po;

        Eigen::Vector3d n_world = (best_d >= 0.0) ? (best_pr - best_po) : (best_po - best_pr);
        if (n_world.norm() < 1e-12) n_world = Eigen::Vector3d::UnitX(); else n_world.normalize();
        vio.n = n_world;

        // 雅可比基准：若 k<5，用第 k 段基准；若 k==5（即末段及焊枪并入），一律用第 5 段
        const int base_seg_for_J = (k < 5) ? k : 5;
        Eigen::Matrix<double,3,6> Jp = compute_position_jacobian_moveit_base_(path[t], base_seg_for_J, best_pr);
        vio.wn = (n_world.transpose() * Jp);

        // 对本体段继续执行“清零远端列”的策略；末段 k=5 无需清零；焊枪被并入 k=5 也不清零
        if (k < 5) {
          for (int j = k + 1; j < 6; ++j) vio.wn(0, j) = 0.0;
        }
        vios.push_back(vio);
      }
    } // end k
  } // end t
  return vios;
}


  static double smooth_cost(const std::vector<Eigen::Matrix<double,6,1>>& path) {
    double s=0.0; for (size_t t=0;t+1<path.size();++t) s += (path[t+1]-path[t]).squaredNorm(); return s;
  }
  double true_penalty(const std::vector<Eigen::Matrix<double,6,1>>& path) const {
    double pen=0.0; auto vios=collect_violations(path);
    for (const auto& v : vios) pen += std::max(0.0, mu_ * (d_safe_ - v.d));
    return pen;
  }
  // 单点 q 的 true penalty（仿照 true_penalty(path)，但只对一个 q）
  double true_penalty_single_q_(const Eigen::Matrix<double,6,1>& q) const {
    double pen = 0.0;
    auto vios = collect_violations_single_q_(q);
    for (const auto& v : vios) {
      pen += std::max(0.0, mu_ * (d_safe_ - v.d));
    }
    return pen;
  }

  // 局部 QP 的 smooth 项：和你局部 QP 里用的一致
  double smooth_cost_local_(const Eigen::Matrix<double,6,1>& qM,
                            const Eigen::Matrix<double,6,1>& q_mid) const {
    if (local_alpha_ <= 0.0) return 0.0;
    return 0.5 * local_alpha_ * (qM - q_mid).squaredNorm();
  }

  // 局部 QP 的 true cost = smooth_local + penalty_single
  double true_cost_local_(const Eigen::Matrix<double,6,1>& qM,
                          const Eigen::Matrix<double,6,1>& q_mid) const {
    double s = smooth_cost_local_(qM, q_mid);
    double p = true_penalty_single_q_(qM);
    return s + p;
  }

  // 通用：对任意路径计算 min_true_distance（后面全局 min_true_distance 也用它）
  double min_true_distance_path_(const std::vector<Eigen::Matrix<double,6,1>>& path) const {
    if (obstacles_.empty()) return std::numeric_limits<double>::infinity();

    double best = std::numeric_limits<double>::infinity();
    const int K_link = 6;

    for (size_t t = 0; t < path.size(); ++t) {
      auto lbs = build_link_boxes_for_q(path[t]);
      const int K_all = static_cast<int>(lbs.size());

      // 0..4 段
      for (int k = 0; k < K_link - 1; ++k) {
        for (const auto& ob : obstacles_) {
          fcl::DistanceRequestd req;
          req.enable_nearest_points    = true;
          req.enable_signed_distance   = true;
          fcl::DistanceResultd  res;
          double d = fcl::distance(lbs[k].obj.get(), ob.obj.get(), req, res);
          if (d < best) best = d;
        }
      }
      // 第 5 段 + 焊枪
      for (const auto& ob : obstacles_) {
        {
          fcl::DistanceRequestd req;
          req.enable_nearest_points    = true;
          req.enable_signed_distance   = true;
          fcl::DistanceResultd  res;
          double d = fcl::distance(lbs[5].obj.get(), ob.obj.get(), req, res);
          if (d < best) best = d;
        }
        if (K_all > K_link) {
          for (int tk = K_link; tk < K_all; ++tk) {
            fcl::DistanceRequestd req;
            req.enable_nearest_points  = true;
            req.enable_signed_distance = true;
            fcl::DistanceResultd  res;
            double d = fcl::distance(lbs[tk].obj.get(), ob.obj.get(), req, res);
            if (d < best) best = d;
          }
        }
      }
    }
    return best;
  }

  // 原来的接口改成调用通用版本
  double min_true_distance(const std::vector<Eigen::Matrix<double,6,1>>& path) const {
    return min_true_distance_path_(path);
  }



  // ---------------- QP 构造/求解 ----------------
  static double model_value(const Eigen::MatrixXd& Q, const Eigen::VectorXd& c, const Eigen::VectorXd& x){
    return 0.5 * x.dot(Q * x) + c.dot(x);
  }
  static double inf_norm(const Eigen::VectorXd& v){
    double m=0.0; for (int i=0;i<v.size();++i) m=std::max(m,std::abs(v(i))); return m;
  }
  // ---------------- QP 调试输出工具 ----------------
  
Eigen::VectorXd build_collision_linear_term_(
    const std::vector<Eigen::Matrix<double,6,1>>& path_old) const
{
  const int K = 6;
  const int T = static_cast<int>(path_old.size());
  const int N = T * K;

  Eigen::VectorXd c0 = Eigen::VectorXd::Zero(N);

  auto vios = collect_violations(path_old);
  for (const auto& vio : vios) {
    double gap = (d_safe_ - vio.d);
    int base = vio.t * K;
    for (int k = 0; k < K; ++k) {
      c0(base + k) += -gap * vio.wn(0, k);
    }
  }

  double c0max = c0.cwiseAbs().maxCoeff();
  if (c0max > 1e-12) c0 /= c0max;

  return c0;
}
Eigen::VectorXd build_convexset_guidance_c_(
    const std::vector<Eigen::Matrix<double,6,1>>& path_old) const
{
  const int K = 6;
  const int T = static_cast<int>(path_old.size());
  const int N = T * K;
  Eigen::VectorXd c_corr = Eigen::VectorXd::Zero(N);

  if (!use_convexset_guidance_) return c_corr;
  if (!guide_.loaded) return c_corr;
  if (!use_anchor_pull_) return c_corr;
  if (guide_.sigma_dense.empty() || guide_.anchor_xyz.empty()) return c_corr;

  for (int t = 0; t < T; ++t) {
    int sigma = 0;
    if (t < (int)guide_.sigma_dense.size()) sigma = guide_.sigma_dense[t];
    sigma = std::max(0, std::min(sigma, (int)guide_.anchor_xyz.size() - 1));

    const Eigen::Vector3d& anchor = guide_.anchor_xyz[sigma];

    Eigen::Isometry3d Ttcp = get_tcp_pose_world_(path_old[t]);
    Eigen::Vector3d p_tcp = Ttcp.translation();

    Eigen::Vector3d dir = anchor - p_tcp;
    if (dir.norm() < 1e-12) continue;
    dir.normalize();

    Eigen::Matrix<double,3,6> Jp =
        compute_position_jacobian_moveit_base_(path_old[t], 5, p_tcp);

    Eigen::RowVectorXd wn = -(dir.transpose() * Jp);

    int base = t * K;
    for (int k = 0; k < K; ++k) {
      c_corr(base + k) += wn(k);
    }
  }

  double m = c_corr.cwiseAbs().maxCoeff();
  if (m > 1e-12) c_corr /= m;

  return c_corr;
}


Eigen::VectorXd build_poly_membership_linear_term_(
    const std::vector<Eigen::Matrix<double,6,1>>& path_old) const
{
  const int K = 6;
  const int T = static_cast<int>(path_old.size());
  const int N = T * K;

  Eigen::VectorXd c_poly = Eigen::VectorXd::Zero(N);

  if (!use_convexset_guidance_) return c_poly;
  if (!use_poly_membership_penalty_) return c_poly;
  if (!guide_.loaded) return c_poly;
  if (guide_.sigma_dense.empty()) return c_poly;
  if (guide_.poly_A.empty() || guide_.poly_b.empty()) return c_poly;

  for (int t = 0; t < T; ++t) {
    if (t >= static_cast<int>(guide_.sigma_dense.size())) continue;

    int sigma = guide_.sigma_dense[t];
    sigma = std::max(0, std::min(sigma, static_cast<int>(guide_.poly_A.size()) - 1));

    const Eigen::MatrixXd& A = guide_.poly_A[sigma];
    const Eigen::VectorXd& b = guide_.poly_b[sigma];

    if (A.rows() == 0 || A.cols() != 3 || b.size() != A.rows()) continue;

    Eigen::Isometry3d Ttcp = get_tcp_pose_world_(path_old[t]);
    Eigen::Vector3d p_tcp = Ttcp.translation();

    Eigen::Matrix<double,3,6> Jp =
        compute_position_jacobian_moveit_base_(path_old[t], 5, p_tcp);

    Eigen::RowVectorXd grad_sum = Eigen::RowVectorXd::Zero(K);

    for (int i = 0; i < A.rows(); ++i) {
      Eigen::RowVector3d ai = A.row(i);
      double viol = ai.dot(p_tcp) - b(i);   // >0 means outside

      if (viol > 0.0) {
        // linearized penalty contribution:
        // d/dq [ max(0, ai*p-b) ] ≈ ai * Jp
        grad_sum += viol * (ai * Jp);
      }
    }

    int base = t * K;
    for (int k = 0; k < K; ++k) {
      c_poly(base + k) += grad_sum(k);
    }
  }

  double m = c_poly.cwiseAbs().maxCoeff();
  if (m > 1e-12) c_poly /= m;

  return c_poly;
}
void add_seed_pull_qp_terms_(Eigen::MatrixXd& Q, Eigen::VectorXd& c) const
{
  if (!use_convexset_guidance_) return;
  if (!use_seed_pull_) return;
  if (!guide_.loaded) return;
  if (guide_.q_seed_dense.empty()) return;

  const int K = 6;
  const int T = std::min((int)guide_.q_seed_dense.size(), steps_);

  for (int t = 0; t < T; ++t) {
    for (int k = 0; k < K; ++k) {
      int idx = t * K + k;
      Q(idx, idx) += lambda_seed_q_;
      c(idx)      += -lambda_seed_q_ * guide_.q_seed_dense[t](k);
    }
  }
}
void build_Q_c_oldstyle_from(const std::vector<Eigen::Matrix<double,6,1>>& path_old,
                             Eigen::MatrixXd& Q, Eigen::VectorXd& c) const {
  const int K = 6, T = steps_, N = T * K;
  constexpr double eps = 1e-12;

  // 1) smooth backbone
  Eigen::MatrixXd Q0 = Eigen::MatrixXd::Zero(N, N);
  for (int t = 0; t < T - 1; ++t) {
    for (int k = 0; k < K; ++k) {
      int id1 = t * K + k;
      int id2 = (t + 1) * K + k;
      Q0(id1, id1) += 1.0;
      Q0(id2, id2) += 1.0;
      Q0(id1, id2) += -1.0;
      Q0(id2, id1) += -1.0;
    }
  }

  double q0max = Q0.cwiseAbs().maxCoeff();
  if (q0max > eps) Q0 /= q0max;

// 2) collision linear term
Eigen::VectorXd c_col = build_collision_linear_term_(path_old);

// 3) anchor guidance term
Eigen::VectorXd c_corr = Eigen::VectorXd::Zero(N);
if (use_convexset_guidance_ && guide_.loaded) {
  c_corr = build_convexset_guidance_c_(path_old);
}

// 4) NEW: poly half-space membership penalty term
Eigen::VectorXd c_poly = Eigen::VectorXd::Zero(N);
if (use_convexset_guidance_ && use_poly_membership_penalty_ && guide_.loaded) {
  c_poly = build_poly_membership_linear_term_(path_old);
}

// 5) assemble
Q = alpha_ * Q0;
c = mu_ * c_col
  + lambda_corr_ * c_corr
  + lambda_poly_membership_ * c_poly;

// 6) optional seed pull
add_seed_pull_qp_terms_(Q, c);
// 7) rescale if necessary
double Mq = Q.cwiseAbs().maxCoeff();
double Mc = c.cwiseAbs().maxCoeff();
double S  = std::max(Mq, Mc);
if (S > 1e6) {
  Q /= S;
  c /= S;
}
}
  struct QPResult { bool solved{false}; Eigen::VectorXd xnew; double obj_new{0.0}; };

  QPResult solve_qp_oldstyle(const Eigen::MatrixXd& Q,
                             const Eigen::VectorXd& c,
                             const Eigen::VectorXd& lb,
                             const Eigen::VectorXd& ub,
                             const Eigen::VectorXd& x0) const {
    const int N = x0.size();
    QPResult out;
    try {
      GRBEnv env = GRBEnv(true);
      env.set("LogToConsole","0"); env.start();
      GRBModel model(env);

      std::vector<GRBVar> xvars; xvars.reserve(N);
      for (int i=0;i<N;++i) {
        auto v = model.addVar(lb(i), ub(i), 0.0, GRB_CONTINUOUS);
        v.set(GRB_DoubleAttr_Start, x0(i));
        xvars.push_back(v);
      }
      model.update();

      GRBQuadExpr quad=0.0;
      for (int i=0;i<N;++i) for (int j=0;j<N;++j) {
        double qij = Q(i,j); if (std::fabs(qij)>1e-16) quad += 0.5*qij*xvars[i]*xvars[j];
      }
      GRBLinExpr lin=0.0; for (int i=0;i<N;++i) if (std::fabs(c(i))>1e-16) lin += c(i)*xvars[i];
      model.setObjective(quad+lin, GRB_MINIMIZE);
      model.optimize();

      if (model.get(GRB_IntAttr_Status)==GRB_OPTIMAL) {
        out.solved=true; out.xnew.resize(N);
        for (int i=0;i<N;++i) out.xnew(i)=xvars[i].get(GRB_DoubleAttr_X);
        out.obj_new=model.get(GRB_DoubleAttr_ObjVal);
      } else {
        RCLCPP_WARN(rclcpp::get_logger("QP"), "QP not optimal. status=%d", model.get(GRB_IntAttr_Status));
      }
    } catch (GRBException& e) {
      RCLCPP_ERROR(rclcpp::get_logger("QP"), "Gurobi error %d: %s", e.getErrorCode(), e.getMessage().c_str());
    } catch (std::exception& e) {
      RCLCPP_ERROR(rclcpp::get_logger("QP"), "Exception: %s", e.what());
    }
    return out;
  }

  // ---------------- 迭代一步（单次模式） ----------------
  // ---------------- 迭代一步（单次模式） ----------------

// ---------------- 迭代一步（单次模式） ----------------
void one_iter_step() {
  reset_local_state_all_();
  // 1) 最大迭代数检查
  if (cur_iter_ >= max_iters_) {
    finalize_and_maybe_stop_("max_iters");
    return;
  }

  const auto path_old = path_;
  const int K = 6;
  const int T = steps_;
  const int N = T * K;

  // 2) 构造 QP
  Eigen::MatrixXd Q;
  Eigen::VectorXd c;
  build_Q_c_oldstyle_from(path_old, Q, c);

  Eigen::VectorXd x0(N);
  for (int t = 0; t < T; ++t) {
    for (int k = 0; k < K; ++k) {
      x0(t * K + k) = path_old[t](k);
    }
  }



  // 4) 旧模型代价（QP 模型的 m_old）+ 真实轨迹代价 t_old = smooth + penalty
  const double m_old       = model_value(Q, c, x0);
  const double smooth_old  = smooth_cost(path_old);
  const double penalty_old = true_penalty(path_old);
  const double t_old       = smooth_old + penalty_old;

  // 3) trust-region 边界生成器
  auto make_bounds = [&](double trust) {
    Eigen::VectorXd lb = x0.array() - trust;
    Eigen::VectorXd ub = x0.array() + trust;

    // 关节限位
    for (int t = 0; t < T; ++t) {
      for (int k = 0; k < K; ++k) {
        int idx = t * K + k;
        lb(idx) = std::max(lb(idx), qmin6_(k));
        ub(idx) = std::min(ub(idx), qmax6_(k));
      }
    }

    // 末关节固定
    if (fix_last_joint_to_zero_) {
      for (int t = 0; t < T; ++t) {
        int idx = t * K + last_joint_index_;
        lb(idx) = last_joint_fixed_value_;
        ub(idx) = last_joint_fixed_value_;
      }
    }

    // 首末点固定
    for (int k = 0; k < K; ++k) {
      int i0 = 0 * K + k;
      int i1 = (T - 1) * K + k;
      if (!(fix_last_joint_to_zero_ && k == last_joint_index_)) {
        lb(i0) = x0(i0);
        ub(i0) = x0(i0);
        lb(i1) = x0(i1);
        ub(i1) = x0(i1);
      }
    }

    return std::pair<Eigen::VectorXd, Eigen::VectorXd>(lb, ub);
  };

  // 5) trust-region 外层循环
  bool accepted = false;
  QPResult last_sol;
  std::vector<Eigen::Matrix<double,6,1>> path_candidate_last;

  for (int attempt = 0; attempt < max_trust_attempts_; ++attempt) {
    auto [lb, ub] = make_bounds(trust_s_);
    ++total_global_attempts_;

    auto sol = solve_qp_oldstyle(Q, c, lb, ub, x0);
    if (!sol.solved) {
      // QP 失败：减小 trust，重来
      trust_s_ = std::max(trust_s_min_, trust_s_ - tau_minus_);
      RCLCPP_WARN(get_logger(),
        "[ITER %d] attempt %d/%d QP fail -> trust=%.6f",
        cur_iter_ + 1, attempt + 1, max_trust_attempts_, trust_s_);
      if (trust_s_ < xtol_) {
        break;
      }
      continue;
    }

    // 解出新路径
    std::vector<Eigen::Matrix<double,6,1>> path_new = path_old;
    for (int t = 0; t < T; ++t) {
      for (int k = 0; k < K; ++k) {
        path_new[t](k) = sol.xnew(t * K + k);
      }
    }

    // === ModelImprove: 使用 QP 模型 ===
    const double m_new      = model_value(Q, c, sol.xnew);
    const double MI         = m_old - m_new;               // ModelImprove
    const double step_inf   = inf_norm(sol.xnew - x0);
    const double min_d_new  = min_true_distance(path_new);

    // === TrueImprove: 用真实轨迹代价 t = smooth + penalty ===
    const double smooth_new  = smooth_cost(path_new);
    const double penalty_new = true_penalty(path_new);
    const double t_new       = smooth_new + penalty_new;
    const double TI          = t_old - t_new;              // TrueImprove

   // 当前全局 mu
double mu_global = mu_;
// “局部 mu” 就按比例算一下，便于对比
double mu_local  = local_mu_scale_ * mu_;

RCLCPP_INFO(get_logger(),
  "[ITER %d] attempt %d/%d | trust=%.6f | MI(model)=%.6f | "
  "mu=%.6f | mu_local=%.6f | "
  "TI(true)=%.6f | step_inf=%.3e | min_d=%.6f | "
  "smooth_new=%.6f | penalty_new=%.6f",
  cur_iter_ + 1, attempt + 1, max_trust_attempts_,
  trust_s_, MI,
  mu_global, mu_local,
  TI, step_inf, min_d_new,
  smooth_new, penalty_new);


    // 如果模型预测改进足够大 -> 正常接受
    if (MI > mi_thresh_) {
      path_    = std::move(path_new);
      trust_s_ = std::min(trust_s_max_, trust_s_ + tau_plus_);
      accepted = true;
      RCLCPP_INFO(get_logger(),
        "[ITER %d] accept -> trust=%.6f",
        cur_iter_ + 1, trust_s_);
      break;
    }

    // 否则记录“最后一次可用的解”，并减少 trust
    last_sol            = std::move(sol);
    path_candidate_last = std::move(path_new);

    const bool is_last = (attempt == max_trust_attempts_ - 1);
    const bool at_min  = (trust_s_ <= trust_s_min_ + 1e-12);

    if (!is_last && !at_min) {
      trust_s_ = std::max(trust_s_min_, trust_s_ - tau_minus_);
      RCLCPP_INFO(get_logger(),
        "[ITER %d] reject -> trust=%.6f",
        cur_iter_ + 1, trust_s_);
      continue;
    }

    // 已经到最后一次 / 已经在 trust 最小值附近：如果有可行解就强制接受
    if (last_sol.solved) {
      path_    = std::move(path_candidate_last);
      accepted = true;
      trust_s_ = std::min(
        trust_s_max_,
        std::max(trust_s_, trust_s_min_ + 0.05)
      );
      RCLCPP_WARN(get_logger(),
        "[ITER %d] force-accept -> trust=%.6f",
        cur_iter_ + 1, trust_s_);
    }
    break;
  } // end for attempts

  if (!accepted) {
    RCLCPP_INFO(get_logger(),
      "[ITER %d] no accepted step",
      cur_iter_ + 1);
    return;
  }

  // 6) 连续性修补 & 可视化
  ++cur_iter_;
  if (!cont_after_global_only_) {
    (void)run_continuity_pass_once_();
  }

  int idc = base_id_for_iter(cur_iter_);
  publish_markers_for_iter(path_, cur_iter_, idc);

  // 7) 早停条件：min_d / 违规 / fk_only
  const double min_d_now = min_true_distance(path_);
  auto vios_now          = collect_violations(path_);
 // ===== NEW: 若这一轮之后仍然不安全 → 放大 mu =====
  {
    bool unsafe_now = !vios_now.empty();
    // 你也可以改成基于 min_d_now 判断，比如：min_d_now < d_safe_
    // bool unsafe_now = (min_d_now < d_safe_);

    bool can_boost = true;
    if (boost_mu_only_after_warmup_) {
      // 前 warmup_safe_iters_ 轮不动 mu，避免一上来就炸
      can_boost = (cur_iter_ >= warmup_safe_iters_);
    }

    if (unsafe_now && can_boost && mu_scale_on_unsafe_ > 1.0 && mu_ < mu_max_) {
      double mu_old = mu_;
      mu_ = std::min(mu_max_, mu_ * mu_scale_on_unsafe_);

      RCLCPP_INFO(get_logger(),
        "[ITER %d] still unsafe -> scale mu: %.6f -> %.6f (scale=%.3f, max=%.3f)",
        cur_iter_, mu_old, mu_, mu_scale_on_unsafe_, mu_max_);
    }
  }
  bool should_stop = false;
  std::string reason;

  const bool pass_min_d =
    stop_when_min_d_ge_ ? (min_d_now >= stop_min_d_)
                        : (min_d_now <= stop_min_d_);
  const bool warmup_ok_for_min_d =
    min_d_ignore_warmup_ || (cur_iter_ >= warmup_safe_iters_);

  if ((pass_min_d && warmup_ok_for_min_d) ||
      (vios_now.empty() && cur_iter_ >= warmup_safe_iters_) ||
      (fk_only_ && cur_iter_ >= max_iters_)) {
    if (pass_min_d) {
      reason = "min_d threshold";
    } else if (vios_now.empty()) {
      reason = "no violations";
    } else if (fk_only_) {
      reason = "fk_only reached max_iters";
    } else {
      reason = "other";
    }
    should_stop = true;
  }

  if (should_stop) {
    finalize_and_maybe_stop_(reason);
  }
}

struct LocalQPOutcome {
    bool accepted{false};
    bool safe_now{false};
  };

  // ---------------- 连续性 pass（插中点 + 局部QP简版） ----------------
    // ===== 连续性修补：对单条 (qA -- qM -- qB) 做 6维局部QP，只调整 qM =====
// ===== 连续性修补：对单条 (qA -- qM -- qB) 做 6维局部QP，只调整 qM =====
LocalQPOutcome run_local_qp_single_(const Eigen::Matrix<double,6,1>& qA,
                                    Eigen::Matrix<double,6,1>&       qM_io,
                                    const Eigen::Matrix<double,6,1>& qB)
{
  // ♻️ 每条边的局部QP开始前都重置“局部状态”
  reset_local_state_all_();
  LocalQPOutcome out;
  Eigen::Matrix<double,6,1> qM = qM_io;

  // 0) 先看看原始是不是已经连续安全
  bool s01 = edge_continuous_safe_convexbox_(qA, qM);
  bool s12 = edge_continuous_safe_convexbox_(qM, qB);
  if (s01 && s12) {
    out.accepted = true;
    out.safe_now = true;
    // 这轮 local 是“安全”的，可以记一次
    local_safe_iters_++;
    return out;
  }

  const Eigen::Matrix<double,6,1> q_mid = 0.5 * (qA + qB);

  // === 局部 trust-region SQP 外层循环 ===
  double trust_loc = local_trust_s_;    // 你原来全局用的那个初值
  const int N      = 6;

  // 🔴 localQP 自己的一套 mu：从成员 mu_local_ 拷一份出来用
  double mu_local = mu_local_;

  // 这条 localQP 是“修补不安全”才进来的，所以视为 unsafe 起步
  bool this_segment_is_unsafe = true;

  for (int outer = 0; outer < local_seg_max_iters_; ++outer) {

    // 1) 计算当前 qM 的违规（含焊枪）
    auto vios = collect_violations_single_q_(qM);

    // 如果已经没违规了，只看连续安全
    if (vios.empty()) {
      bool s01_ok = edge_continuous_safe_convexbox_(qA, qM);
      bool s12_ok = edge_continuous_safe_convexbox_(qM, qB);

      qM_io        = qM;
      out.accepted = true;
      out.safe_now = (s01_ok && s12_ok);

      // 根据结果更新 local 的“安全计数”
      if (out.safe_now) local_safe_iters_++;
      else              local_safe_iters_ = 0;

      // 👉 把本轮用完的 mu_local 写回成员，形成真正的“local mu 状态”
      mu_local_ = mu_local;
      return out;
    }

    // ----- 2) 构造局部 QP 模型 Q, c -----
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(N, N);
    Eigen::VectorXd c = Eigen::VectorXd::Zero(N);

    // 2.1 平滑项：0.5 * local_alpha * ||qM - q_mid||^2
    if (local_alpha_ > 0.0) {
      Q = local_alpha_ * Eigen::MatrixXd::Identity(N, N);
      c = -local_alpha_ * q_mid;
    }

    // 2.2 碰撞处罚线性项：⚠️ 这里用 local mu，不再用全局 mu_
    Eigen::RowVectorXd grad_row = Eigen::RowVectorXd::Zero(N);
    for (const auto& vio : vios) {
      double gap = (d_safe_ - vio.d);
      if (gap <= 0.0) continue;
      grad_row += -gap * vio.wn;  // vio.wn = 1x6，由 DH 雅可比算
    }
    if (grad_row.norm() > 0.0) {
      Eigen::VectorXd grad = grad_row.transpose();
      c += mu_local * grad;   // ✅ 核心：这里是 mu_local
    }

    // 3) 当前点 x0 以及 trust 范围内的 box 约束 + 限位 + 末关节固定
    Eigen::VectorXd x0(N);
    for (int i = 0; i < N; ++i) x0(i) = qM(i);

    auto make_bounds_local = [&](double trust)
    {
      Eigen::VectorXd lb(N), ub(N);
      for (int k = 0; k < N; ++k) {
        double lo = qM(k) - trust;
        double hi = qM(k) + trust;
        lo = std::max(lo, qmin6_(k));
        hi = std::min(hi, qmax6_(k));
        lb(k) = lo;
        ub(k) = hi;
      }
      if (fix_last_joint_to_zero_) {
        int idx = last_joint_index_;
        if (idx >= 0 && idx < N) {
          lb(idx) = last_joint_fixed_value_;
          ub(idx) = last_joint_fixed_value_;
        }
      }
      return std::pair<Eigen::VectorXd, Eigen::VectorXd>(lb, ub);
    };

    const double m_old = model_value(Q, c, x0);
    const double t_old = true_cost_local_(qM, q_mid);   // 平滑 + true penalty(单点)

    // 4) 内层 trust-region 尝试循环
    bool       step_accepted = false;
    QPResult   last_sol;
    Eigen::Matrix<double,6,1> qM_candidate = qM;

    for (int attempt = 0; attempt < max_trust_attempts_; ++attempt) {

      auto [lb, ub] = make_bounds_local(trust_loc);
      auto sol      = solve_qp_oldstyle(Q, c, lb, ub, x0);

      if (!sol.solved) {
        trust_loc = std::max(trust_s_min_, trust_loc - tau_minus_);
        RCLCPP_WARN(get_logger(),
          "[LOCAL-QP] outer=%d attempt=%d/%d QP fail -> trust_loc=%.6f",
          outer, attempt + 1, max_trust_attempts_, trust_loc);
        if (trust_loc < xtol_) break;
        continue;
      }

      Eigen::Matrix<double,6,1> qM_new;
      for (int i = 0; i < N; ++i) qM_new(i) = sol.xnew(i);

      const double m_new    = model_value(Q, c, sol.xnew);
      const double MI       = m_old - m_new;
      const double t_new    = true_cost_local_(qM_new, q_mid);
      const double TI       = t_old - t_new;
      const double step_inf = inf_norm(sol.xnew - x0);

      std::vector<Eigen::Matrix<double,6,1>> local_path{qA, qM_new, qB};
      const double min_d_now = min_true_distance_path_(local_path);

      if (debug_log_local_qp_) {
        RCLCPP_INFO(get_logger(),
          "[LOCAL-QP] outer=%d attempt=%d/%d | trust_loc=%.6f | "
          "MI=%.6e | TI=%.6e | mu_local=%.6f | step_inf=%.3e | min_d=%.6f",
          outer, attempt + 1, max_trust_attempts_,
          trust_loc,
          MI, TI,
          mu_local,
          step_inf, min_d_now);
      }

      // ==== 判“unsafe”并按和全局一样的逻辑放大 mu_local ====
      // 这里我用 min_d_now < d_safe_ 作为 unsafe 判据，
      // 你若全局用的是 cont_min_d_safe_ 或别的 flag，可以改成同一个条件。
      bool unsafe_now = (min_d_now < d_safe_);

      if (unsafe_now &&
          mu_scale_on_unsafe_local_ > 1.0 &&
          (!boost_mu_only_after_warmup_local_ ||
           local_safe_iters_ >= warmup_safe_iters_local_))
      {
        double old_mu_local = mu_local;
        mu_local = std::min(mu_local_max_, mu_local * mu_scale_on_unsafe_local_);

        if (debug_log_local_qp_) {
          RCLCPP_INFO(get_logger(),
            "[LOCAL-QP] scale mu_local: %.6f -> %.6f (min_d=%.6f)",
            old_mu_local, mu_local, min_d_now);
        }
      }

      // ===== 模型预测改进足够好 -> 接受这一步 =====
      if (MI > mi_thresh_) {
        qM            = qM_new;
        step_accepted = true;
        trust_loc     = std::min(trust_s_max_, trust_loc + tau_plus_);

        if (!unsafe_now) local_safe_iters_++;
        else             local_safe_iters_ = 0;

        if (debug_log_local_qp_) {
          RCLCPP_INFO(get_logger(),
            "[LOCAL-QP] outer=%d accept -> trust_loc=%.6f | safe_iters_local=%d",
            outer, trust_loc, local_safe_iters_);
        }
        break;
      }

      // 否则，缩 trust 再试
      last_sol      = std::move(sol);
      qM_candidate  = qM_new;

      const bool is_last = (attempt == max_trust_attempts_ - 1);
      const bool at_min  = (trust_loc <= trust_s_min_ + 1e-12);

      if (!is_last && !at_min) {
        trust_loc = std::max(trust_s_min_, trust_loc - tau_minus_);
        if (debug_log_local_qp_) {
          RCLCPP_INFO(get_logger(),
            "[LOCAL-QP] outer=%d reject -> trust_loc=%.6f",
            outer, trust_loc);
        }
        continue;
      }

      // 最后一试 / 已到底，仍然有解：强行接受
      if (last_sol.solved) {
        qM            = qM_candidate;
        step_accepted = true;
        trust_loc     = std::min(
          trust_s_max_,
          std::max(trust_loc, trust_s_min_ + 0.05)
        );
        RCLCPP_WARN(get_logger(),
          "[LOCAL-QP] outer=%d force-accept -> trust_loc=%.6f",
          outer, trust_loc);
      }
      break;
    } // attempt

    if (!step_accepted) {
      RCLCPP_WARN(get_logger(),
        "[LOCAL-QP] outer=%d no accepted step -> stop local SQP",
        outer);
      out.accepted = false;
      out.safe_now = false;
      qM_io        = qM;
      // 失败不算“安全轮次”，清零
      local_safe_iters_ = 0;
      mu_local_         = mu_local;   // 仍然把当前 mu_local 写回成员
      return out;
    }

    // 5) 这一步接受了 qM，检查连续安全
    bool s01_new = edge_continuous_safe_convexbox_(qA, qM);
    bool s12_new = edge_continuous_safe_convexbox_(qM, qB);

    if (s01_new && s12_new) {
      qM_io        = qM;
      out.accepted = true;
      out.safe_now = true;

      local_safe_iters_++;
      mu_local_ = mu_local;   // 更新回全局的 local 状态

      return out;
    }

    // 还不完全安全 -> outer++ 再来一轮
  }

  // outer 用完：接受当前 qM（可能仍不完全连续安全）
  qM_io        = qM;
  out.accepted = true;
  out.safe_now =
    edge_continuous_safe_convexbox_(qA, qM) &&
    edge_continuous_safe_convexbox_(qM, qB);

  if (out.safe_now) local_safe_iters_++;
  else              local_safe_iters_ = 0;

  mu_local_ = mu_local;    // 最后写回
  return out;
}





  bool run_continuity_pass_once_() {
  unsafe_edges_cache_.clear();

  if (use_continuity_check_log_only_) {
    int unsafe_edges = 0;
    for (size_t i = 0; i + 1 < path_.size(); ++i) {
      if (!edge_continuous_safe_convexbox_(path_[i], path_[i+1])) {
        ++unsafe_edges;
        unsafe_edges_cache_.emplace_back(path_[i], path_[i+1]);
      }
    }
    RCLCPP_INFO(get_logger(), "[CONT-LOG] unsafe edges = %d", unsafe_edges);
    return false;
  }

  std::vector<Eigen::Matrix<double,6,1>> new_path;
  new_path.reserve(path_.size() * 2);
  int inserted = 0;

  for (size_t i = 0; i + 1 < path_.size(); ++i) {
    const auto& qA = path_[i];
    const auto& qB = path_[i+1];
    new_path.push_back(qA);

    if (!edge_continuous_safe_convexbox_(qA, qB)) {
      unsafe_edges_cache_.emplace_back(qA, qB);

      Eigen::Matrix<double,6,1> qM = 0.5 * (qA + qB);

      // ✅ 这里直接调用“有外层 SQP 的局部 QP”，不再套 rep 循环
      auto out = run_local_qp_single_(qA, qM, qB);

      // out.accepted == false：就保持原来的 qM（0.5*(qA+qB)）插进去
      new_path.push_back(qM);
      ++inserted;
    }
  }

  new_path.push_back(path_.back());

  if (inserted > 0) {
    path_.swap(new_path);
    steps_ = static_cast<int>(path_.size());
    RCLCPP_INFO(get_logger(),
                "[CONT] inserted %d midpoints -> steps=%d",
                inserted, steps_);
    return true;
  } else {
    RCLCPP_INFO(get_logger(), "[CONT] all edges continuously safe.");
    return false;
  }
}


  // ---------------- 批量评估 ----------------
  static bool match_glob_(const std::string& name, const std::string& glob) {
    std::string re="^"; for (char c:glob){ if(c=='*') re+=".*";
      else if (std::isalnum((unsigned char)c)||c=='_'||c=='.'||c=='-') re+=c;
      else { re+='\\'; re+=c; } } re+="$";
    return std::regex_match(name, std::regex(re));
  }

  struct Metrics { int success{0}; double time_ms{0}; int steps{0}; double normed_len{0}; double min_d{0}; };

  bool load_q_path_from_file_(const std::string& path,
                              std::vector<Eigen::Matrix<double,6,1>>& out,
                              bool* is_deg_out) {
    std::ifstream fin(path);
    if (!fin.is_open()) return false;
    bool is_deg = q_file_is_deg_default_;
    out.clear();
    std::string line; bool header_checked=false;
    while (std::getline(fin,line)) {
      auto trim=[](std::string s){ size_t a=s.find_first_not_of(" \t\r\n"), b=s.find_last_not_of(" \t\r\n");
        if(a==std::string::npos) return std::string(); return s.substr(a,b-a+1); };
      line=trim(line); if(line.empty()) continue;
      if (!header_checked) {
        header_checked=true;
        if (line.rfind("#",0)==0) {
          auto u=line; std::transform(u.begin(),u.end(),u.begin(),::tolower);
          if (u.find("deg")!=std::string::npos) is_deg=true;
          if (u.find("rad")!=std::string::npos) is_deg=false;
          continue;
        }
        if (line.find("t,")==0 || line.find("q0")!=std::string::npos) continue;
      }
      std::vector<double> vals; { std::string s=line;
        for (char& ch:s){ if (!((ch>='0'&&ch<='9')||ch=='+'||ch=='-'||ch=='.'||ch=='e'||ch=='E')) ch=' '; }
        std::stringstream ss(s); double v; while (ss>>v) vals.push_back(v); }
      if (vals.size()>=7) {
  Eigen::Matrix<double,6,1> q;
  // 假设布局：t, q0..q5, x, y, z
  for (int k=0;k<6;++k)
    q(k) = vals[1 + k];     // 索引 1..6 -> q0..q5
  out.push_back(q);
} else if (vals.size()==6) {
  Eigen::Matrix<double,6,1> q;
  for (int k=0;k<6;++k) q(k)=vals[k];
  out.push_back(q);
}

    }
    if (out.empty()) return false;
    if (is_deg) for (auto& q:out) for (int k=0;k<6;++k) q(k)*=M_PI/180.0;
    if (is_deg_out) *is_deg_out=is_deg;
    return true;
  }
  bool load_convexset_guide_from_yaml_(const std::string& file) {
  guide_ = ConvexSetGuideData{};

  if (file.empty()) {
    RCLCPP_WARN(get_logger(), "[GUIDE] empty convexset_result_yaml");
    return false;
  }

  try {
    YAML::Node root = YAML::LoadFile(file);

    auto read_q_list = [&](const YAML::Node& node,
                           std::vector<Eigen::Matrix<double,6,1>>& out) {
      out.clear();
      if (!node || !node.IsSequence()) return;

      for (const auto& item : node) {
        if (!item || !item.IsSequence() || item.size() != 6) continue;

        Eigen::Matrix<double,6,1> q;
        for (int k = 0; k < 6; ++k) {
          double v = item[k].as<double>();
          q(k) = convexset_q_in_degree_ ? (v * M_PI / 180.0) : v;
        }
        out.push_back(q);
      }
    };

    read_q_list(root["q_seed_dense"], guide_.q_seed_dense);
    read_q_list(root["q_rep"],        guide_.q_rep);
    read_q_list(root["q_paths5"],     guide_.q_paths5);

    if (root["sigma_dense"] && root["sigma_dense"].IsSequence()) {
      for (const auto& s : root["sigma_dense"]) {
        guide_.sigma_dense.push_back(s.as<int>());
      }
    }

    if (root["poly_info"] && root["poly_info"].IsSequence()) {
  for (const auto& item : root["poly_info"]) {
    // anchor
    if (item["anchor_xyz"] && item["anchor_xyz"].IsSequence() && item["anchor_xyz"].size() == 3) {
      guide_.anchor_xyz.emplace_back(
        item["anchor_xyz"][0].as<double>(),
        item["anchor_xyz"][1].as<double>(),
        item["anchor_xyz"][2].as<double>()
      );
    } else {
      guide_.anchor_xyz.emplace_back(Eigen::Vector3d::Zero());
    }

    // NEW: read A,b
    Eigen::MatrixXd A;
    Eigen::VectorXd b;

    if (item["A"] && item["A"].IsSequence() && item["b"] && item["b"].IsSequence()) {
      const int m = static_cast<int>(item["A"].size());
      if (m > 0 && static_cast<int>(item["b"].size()) == m) {
        A.resize(m, 3);
        b.resize(m);

        bool ok_ab = true;
        for (int i = 0; i < m; ++i) {
          if (!item["A"][i].IsSequence() || item["A"][i].size() != 3) {
            ok_ab = false;
            break;
          }
          A(i,0) = item["A"][i][0].as<double>();
          A(i,1) = item["A"][i][1].as<double>();
          A(i,2) = item["A"][i][2].as<double>();
          b(i)   = item["b"][i].as<double>();
        }

        if (!ok_ab) {
          A.resize(0,3);
          b.resize(0);
        }
      }
    }

    guide_.poly_A.push_back(A);
    guide_.poly_b.push_back(b);
  }
}

    guide_.loaded = !guide_.q_seed_dense.empty();

    RCLCPP_INFO(get_logger(),
      "[GUIDE] loaded '%s' | q_seed_dense=%zu | sigma=%zu | anchors=%zu | q_rep=%zu | q_paths5=%zu",
      file.c_str(),
      guide_.q_seed_dense.size(),
      guide_.sigma_dense.size(),
      guide_.anchor_xyz.size(),
      guide_.q_rep.size(),
      guide_.q_paths5.size());

    return guide_.loaded;
  }
  catch (const std::exception& e) {
    RCLCPP_ERROR(get_logger(), "[GUIDE] failed to load '%s': %s", file.c_str(), e.what());
    guide_ = ConvexSetGuideData{};
    return false;
  }
}

  static double path_length_normed(const std::vector<Eigen::Matrix<double,6,1>>& path) {
    if (path.size()<2) return 0.0;
    double L=0.0; for (size_t t=0;t+1<path.size();++t) L += (path[t+1]-path[t]).norm(); return L;
  }

  void mean_std_(const std::vector<double>& v, double& m, double& s) {
    if (v.empty()) { m=0; s=0; return; }
    m=0; for (auto x:v) m+=x; m/=v.size();
    s=0; for (auto x:v) s+=(x-m)*(x-m); s = std::sqrt(s/v.size());
  }

  // —— 用下面这个版本“完整替换”你当前的 run_batch_and_write_csv_() —— 
void run_batch_and_write_csv_() {
  namespace fs = std::filesystem;
  if (!fs::exists(batch_in_dir_) || !fs::is_directory(batch_in_dir_)) {
    RCLCPP_FATAL(get_logger(),"batch_in_dir '%s' invalid", batch_in_dir_.c_str()); return;
  }
  std::vector<fs::path> files;
  for (auto& p : fs::directory_iterator(batch_in_dir_)) {
    if (!p.is_regular_file()) continue;
    if (match_glob_(p.path().filename().string(), batch_glob_)) files.push_back(p.path());
  }
  std::sort(files.begin(), files.end());
  if (files.empty()) { RCLCPP_WARN(get_logger(),"no files matched '%s'", batch_glob_.c_str()); return; }

  std::ofstream fout(csv_out_);
  if (!fout.is_open()) { RCLCPP_ERROR(get_logger(),"cannot open csv_out: %s", csv_out_.c_str()); return; }

  // 头部：在最后新增 file 列
  fout << "idx,success,time_ms,steps,normed_len,min_d,file\n";

  std::vector<double> v_succ, v_time, v_steps, v_len, v_md;

  for (size_t i=0;i<files.size();++i) {
    reset_local_state_all_();
    reset_global_state_on_new_package_(); // 新包：全局状态清零

    std::vector<Eigen::Matrix<double,6,1>> qpath; bool isdeg=false;
    if (!load_q_path_from_file_(files[i].string(), qpath, &isdeg)) {
      RCLCPP_WARN(get_logger(),"[%zu/%zu] load fail: %s", i+1, files.size(), files[i].c_str());
      continue;
    }
    path_ = qpath; steps_=(int)path_.size(); q_start_=path_.front(); q_goal_=path_.back();

    mu_ = this->get_parameter("mu").as_double();
    trust_s_ = this->get_parameter("trust_s").as_double();
    cur_iter_ = 0; total_global_attempts_ = 0;

    Metrics M = run_to_completion_collect_metrics_(batch_visualize_);

    // 这里在末尾加文件名列（用基础文件名；如需完整路径可改为 files[i].string()）
    const std::string file_col = csv_escape_(files[i].filename().string());
    fout << (i+1) << "," << M.success << "," << std::fixed << std::setprecision(3)
     << M.time_ms << "," << M.steps << "," << std::setprecision(6)
     << M.normed_len << "," << M.min_d << ","
     << std::quoted(files[i].string()) << "\n";

    v_succ.push_back(M.success);
    v_time.push_back(M.time_ms);
    v_steps.push_back(M.steps);
    v_len.push_back(M.normed_len);
    v_md.push_back(M.min_d);

    RCLCPP_INFO(get_logger(),
      "[%zu/%zu] ok file='%s' | success=%d time_ms=%.3f steps=%d normed_len=%.6f min_d=%.6f",
      i+1, files.size(), files[i].c_str(), M.success, M.time_ms, M.steps, M.normed_len, M.min_d);
  }

  auto ms_pair=[&](const std::vector<double>& v){ double m,s; mean_std_(v,m,s); return std::pair<double,double>(m,s); };
  auto S=ms_pair(v_succ), T=ms_pair(v_time), P=ms_pair(v_steps), L=ms_pair(v_len), D=ms_pair(v_md);

  // 统计行在 file 列补空即可
  fout << "mean," << S.first << "," << std::fixed << std::setprecision(3)
       << T.first << "," << P.first << "," << std::setprecision(6)
       << L.first << "," << D.first << ",\n";
  fout << "std," << S.second << "," << std::fixed << std::setprecision(3)
       << T.second << "," << P.second << "," << std::setprecision(6)
       << L.second << "," << D.second << ",\n";
  fout.close();
  RCLCPP_INFO(get_logger(), "CSV written: %s", csv_out_.c_str());
}


  Metrics run_to_completion_collect_metrics_(bool visualize) {
    Metrics M; plan_start_tp_=Clock::now(); cur_iter_=0;
    if (visualize) { int idc=base_id_for_iter(0); publish_markers_for_iter(path_,0,idc); }
    for (;;) {
      const auto path_old = path_;
      const int K=6,T=steps_,N=T*K;
      Eigen::MatrixXd Q; Eigen::VectorXd c; build_Q_c_oldstyle_from(path_old,Q,c);
      Eigen::VectorXd x0(N); for(int t=0;t<T;++t)for(int k=0;k<K;++k) x0(t*K+k)=path_old[t](k);

      auto make_bounds=[&](double trust){
        Eigen::VectorXd lb=x0.array()-trust, ub=x0.array()+trust;
        for(int t=0;t<T;++t)for(int k=0;k<K;++k){int idx=t*K+k; lb(idx)=std::max(lb(idx), qmin6_(k)); ub(idx)=std::min(ub(idx), qmax6_(k));}
        if (fix_last_joint_to_zero_) for(int t=0;t<T;++t){int idx=t*K+last_joint_index_; lb(idx)=last_joint_fixed_value_; ub(idx)=last_joint_fixed_value_;}
        for(int k=0;k<K;++k){int i0=0*K+k,i1=(T-1)*K+k;
          if(!(fix_last_joint_to_zero_&&k==last_joint_index_)){ lb(i0)=x0(i0);ub(i0)=x0(i0);lb(i1)=x0(i1);ub(i1)=x0(i1);}}
        return std::pair<Eigen::VectorXd,Eigen::VectorXd>(lb,ub);
      };

      const double m_old=model_value(Q,c,x0);
      bool accepted=false; QPResult last_sol; std::vector<Eigen::Matrix<double,6,1>> last_path;

      for (int attempt=0; attempt<max_trust_attempts_; ++attempt) {
        ++total_global_attempts_;
        auto [lb,ub]=make_bounds(trust_s_);
        auto sol=solve_qp_oldstyle(Q,c,lb,ub,x0);
        if (!sol.solved){ trust_s_=std::max(trust_s_min_,trust_s_-tau_minus_); if(trust_s_<xtol_) break; continue; }
        std::vector<Eigen::Matrix<double,6,1>> path_new=path_old;
        for(int t=0;t<T;++t)for(int k=0;k<K;++k) path_new[t](k)=sol.xnew(t*K+k);
        double MI = m_old - model_value(Q,c,sol.xnew);
        if (MI > mi_thresh_) { path_=std::move(path_new); trust_s_=std::min(trust_s_max_,trust_s_+tau_plus_); accepted=true; break; }
        last_sol=std::move(sol); last_path=std::move(path_new);
        const bool is_last=(attempt==max_trust_attempts_-1), at_min=(trust_s_<=trust_s_min_+1e-12);
        if (!is_last && !at_min){ trust_s_=std::max(trust_s_min_,trust_s_-tau_minus_); continue; }
        if (last_sol.solved){ path_=std::move(last_path); accepted=true; trust_s_=std::min(trust_s_max_, std::max(trust_s_, trust_s_min_+0.05)); }
        break;
      }

      ++cur_iter_;
      if (!cont_after_global_only_) (void)run_continuity_pass_once_();
      if (visualize){ int idc=base_id_for_iter(cur_iter_); publish_markers_for_iter(path_,cur_iter_,idc); }

      const double min_d_now=min_true_distance(path_);
      auto vios_now = collect_violations(path_);
      bool stop=false;
      const bool pass_min_d = stop_when_min_d_ge_ ? (min_d_now>=stop_min_d_) : (min_d_now<=stop_min_d_);
      const bool warmup_ok = min_d_ignore_warmup_ || (cur_iter_>=warmup_safe_iters_);
      if ((pass_min_d && warmup_ok) || (vios_now.empty() && cur_iter_>=warmup_safe_iters_) || (cur_iter_>=max_iters_))
        stop=true;

      if (stop) {
        // ******** 多轮连续修补（非 log-only）循环补丁：开始 ********
        if (cont_after_global_only_) {
          for (int pass = 0; pass < 5; ++pass) {
            bool changed = run_continuity_pass_once_();
            if (visualize){ int idc=base_id_for_iter(cur_iter_); publish_markers_for_iter(path_,cur_iter_,idc); }
            if (!changed) break;
          }
        }
        // ******** 多轮连续修补（非 log-only）循环补丁：结束 ********
        if (enable_final_global_smooth_) {
  run_final_global_smooth_();

  if (cont_after_global_only_) {
    (void)run_continuity_pass_once_();
  }

  if (visualize) {
    int idc = base_id_for_iter(cur_iter_);
    publish_markers_for_iter(path_, cur_iter_, idc);
  }
}

        const auto elapsed = std::chrono::duration<double,std::milli>(Clock::now()-plan_start_tp_).count();
        M.time_ms=elapsed; M.steps=(int)path_.size(); M.normed_len=path_length_normed(path_); M.min_d=min_true_distance(path_);
        M.success = ((stop_when_min_d_ge_ ? (M.min_d >= stop_min_d_) : (M.min_d <= stop_min_d_)) ? 1 : 0);
        RCLCPP_INFO(get_logger(),"[PLAN DONE][BATCH] steps=%d len=%.6f min_d=%.6f time=%.3fms", M.steps, M.normed_len, M.min_d, M.time_ms);
        break;
      }
    }
    return M;
  }
void run_final_global_smooth_() {
  if (!enable_final_global_smooth_) return;
  if (path_.size() < 3) return;

  const int K = 6;
  const int T = steps_;
  const int N = T * K;

  RCLCPP_INFO(get_logger(),
    "[FINAL-SMOOTH] start | iters=%d | alpha=%.3f | mu_scale=%.3f | trust=%.3f",
    final_global_smooth_iters_,
    final_global_smooth_alpha_,
    final_global_smooth_mu_scale_,
    final_global_smooth_trust_s_);

  for (int it = 0; it < final_global_smooth_iters_; ++it) {
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(N, N);
    Eigen::VectorXd c = Eigen::VectorXd::Zero(N);

    // 1) stronger smoothness
    Eigen::MatrixXd Q0 = Eigen::MatrixXd::Zero(N, N);
    for (int t = 0; t < T - 1; ++t) {
      for (int k = 0; k < K; ++k) {
        int i1 = t * K + k;
        int i2 = (t + 1) * K + k;
        Q0(i1, i1) += 1.0;
        Q0(i2, i2) += 1.0;
        Q0(i1, i2) += -1.0;
        Q0(i2, i1) += -1.0;
      }
    }

    double q0max = Q0.cwiseAbs().maxCoeff();
    if (q0max > 1e-12) Q0 /= q0max;

    Q += final_global_smooth_alpha_ * Q0;

    // 2) weak collision
    Eigen::VectorXd c_col = build_collision_linear_term_(path_);
    c += final_global_smooth_mu_scale_ * mu_ * c_col;

    // 3) weak convex-set guidance
    if (use_convexset_guidance_ && guide_.loaded) {
      Eigen::VectorXd c_corr = build_convexset_guidance_c_(path_);
      c += final_global_smooth_lambda_corr_ * c_corr;
    }
    if (use_convexset_guidance_ && use_poly_membership_penalty_ && guide_.loaded) {
  Eigen::VectorXd c_poly = build_poly_membership_linear_term_(path_);
  c += lambda_poly_membership_ * c_poly;
}

    // 4) optional seed pull
    if (use_convexset_guidance_ && use_seed_pull_) {
      add_seed_pull_qp_terms_(Q, c);
    }

    // 5) x0
    Eigen::VectorXd x0(N);
    for (int t = 0; t < T; ++t)
      for (int k = 0; k < K; ++k)
        x0(t * K + k) = path_[t](k);

    // 6) trust-region bounds
    Eigen::VectorXd lb = x0.array() - final_global_smooth_trust_s_;
    Eigen::VectorXd ub = x0.array() + final_global_smooth_trust_s_;

    for (int t = 0; t < T; ++t) {
      for (int k = 0; k < K; ++k) {
        int idx = t * K + k;
        lb(idx) = std::max(lb(idx), qmin6_(k));
        ub(idx) = std::min(ub(idx), qmax6_(k));
      }
    }

    // fix endpoints
    for (int k = 0; k < K; ++k) {
      int i0 = 0 * K + k;
      int i1 = (T - 1) * K + k;
      lb(i0) = ub(i0) = x0(i0);
      lb(i1) = ub(i1) = x0(i1);
    }

    // fix last joint if needed
    if (fix_last_joint_to_zero_) {
      for (int t = 0; t < T; ++t) {
        int idx = t * K + last_joint_index_;
        lb(idx) = last_joint_fixed_value_;
        ub(idx) = last_joint_fixed_value_;
      }
    }

    auto sol = solve_qp_oldstyle(Q, c, lb, ub, x0);
    if (!sol.solved) {
      RCLCPP_WARN(get_logger(), "[FINAL-SMOOTH] QP failed at iter=%d", it);
      break;
    }

    for (int t = 0; t < T; ++t) {
      for (int k = 0; k < K; ++k) {
        path_[t](k) = sol.xnew(t * K + k);
      }
    }
  }

  RCLCPP_INFO(get_logger(),
    "[FINAL-SMOOTH] done | steps=%zu | min_d=%.6f | len=%.6f",
    path_.size(), min_true_distance(path_), path_length_normed(path_));
}
  // ---------------- 可视化 ----------------
  void append_box_as_triangles(const LinkBox& lb, Marker& tri) const {
    const double hx=lb.size.x()*0.5, hy=lb.size.y()*0.5, hz=lb.size.z()*0.5;
    Eigen::Vector3d v[8]={{+hx,+hy,+hz},{+hx,+hy,-hz},{+hx,-hy,-hz},{+hx,-hy,+hz},
                          {-hx,+hy,+hz},{-hx,+hy,-hz},{-hx,-hy,-hz},{-hx,-hy,+hz}};
    auto toPoint=[&](const Eigen::Vector3d& p){ Eigen::Vector3d pw=lb.T*p; geometry_msgs::msg::Point P; P.x=pw.x(); P.y=pw.y(); P.z=pw.z(); return P; };
    int f[12][3]={{0,1,2},{0,2,3},{4,6,5},{4,7,6},{0,4,5},{0,5,1},{3,2,6},{3,6,7},{0,3,7},{0,7,4},{1,5,6},{1,6,2}};
    for (auto& tri_idx:f){ tri.points.push_back(toPoint(v[tri_idx[0]])); tri.points.push_back(toPoint(v[tri_idx[1]])); tri.points.push_back(toPoint(v[tri_idx[2]])); }
  }

  void append_obstacles_markers(MarkerArray& arr, const std_msgs::msg::Header& header) const {
  int base=90000000, id=0;
  for (const auto& ob : obstacles_) {
    Marker m; 
    m.header = header; 
    m.ns     = "obstacle"; 
    m.id     = base+id++; 
    m.action = Marker::ADD;

    // === 从 FCL 对象里取世界变换：统一转成 Eigen::Isometry3d ===
    Eigen::Isometry3d Tw = Eigen::Isometry3d::Identity();
    {
      const auto& Tf = ob.obj->getTransform();      // fcl::Transform3d
      Tw.matrix() = Tf.matrix();
    }

    Eigen::Vector3d p = Tw.translation();
    Eigen::Quaterniond q(Tw.linear());

    m.pose.position.x = p.x();
    m.pose.position.y = p.y();
    m.pose.position.z = p.z();
    m.pose.orientation.x = q.x();
    m.pose.orientation.y = q.y();
    m.pose.orientation.z = q.z();
    m.pose.orientation.w = q.w();

    if (ob.type==ObType::BOX) {
      m.type = Marker::CUBE;
      m.scale.x = ob.size.x();
      m.scale.y = ob.size.y();
      m.scale.z = ob.size.z();
      m.color.a = 0.6f; m.color.r=1.0f; m.color.g=0.3f; m.color.b=0.3f;
    } 
    else if (ob.type==ObType::SPHERE) {
      m.type = Marker::SPHERE;
      m.scale.x = ob.size.x();
      m.scale.y = ob.size.y();
      m.scale.z = ob.size.z();
      m.color.a = 0.6f; m.color.r=0.2f; m.color.g=0.6f; m.color.b=1.0f;
    } 
    else { // CYLINDER
      m.type = Marker::CYLINDER;
      m.scale.x = ob.size.x();   // 直径
      m.scale.y = ob.size.y();
      m.scale.z = ob.size.z();   // 高度
      m.color.a = 0.6f; m.color.r=0.9f; m.color.g=0.7f; m.color.b=0.2f;
    }

    arr.markers.push_back(m);
  }
}


  inline int  base_id_for_iter(int iter) const { return iter*10000; }
  inline void color_from_ratio(double r, float& R,float& G,float& B) const { r=std::clamp(r,0.0,1.0); R=float(1.0-r); G=float(r); B=0.2f; }
  inline std::string ns_iter(int iter, const std::string& tag) const { char buf[32]; std::snprintf(buf,sizeof(buf),"iter_%03d",iter); return std::string(buf)+"/"+tag; }

  void publish_markers_for_iter(const std::vector<Eigen::Matrix<double,6,1>>& path_iter,
                              int iter, int& id_cursor) const {
  // 先准备 header/arr
  MarkerArray arr;
  std_msgs::msg::Header header;
  header.frame_id = frame_id_;
  header.stamp = this->now();  // const 成员内可用
    // ... 你现在的 header / base cube 等代码在前面 ...

  // === NEW: 打印当前这一轮可视化使用的路径关节角（deg） ===
  if (debug_print_q_ && !path_iter.empty()) {
    std::ostringstream oss;
    oss << "[VIZ-q] iter=" << iter
        << " | steps=" << path_iter.size()
        << " | q (deg):\n";

    for (size_t t = 0; t < path_iter.size(); ++t) {
      oss << "  t=" << t << " : [";
      for (int k = 0; k < 6; ++k) {
        double q_deg = path_iter[t](k) * 180.0 / M_PI;
        oss << std::fixed << std::setprecision(3) << q_deg;
        if (k < 5) oss << ", ";
      }
      oss << "]\n";
    }

    RCLCPP_INFO(get_logger(), "%s", oss.str().c_str());

    // 如果你只想打印一次，可以在外面开个 debug_print_q_once_ 开关：
    // debug_print_q_ = !debug_print_q_once_;
  }

  // === 从这里开始，保持你原来的逻辑 ===
  double ratio = (max_iters_>1)? std::min(1.0, std::max(0.0, double(iter)/double(std::max(1,max_iters_)))):0.0;
  float R,G,B; color_from_ratio(ratio,R,G,B);

  // 末端轨迹 / 连杆TRIANGLES / 违规点 / 扫掠壳 / 焊枪 / 障碍物 ...
  // （你现有的代码保持不变）

    // 末端轨迹
    {
      Marker m; m.header=header; m.id=id_cursor++; m.ns=ns_iter(iter,"ee_path"); m.type=Marker::LINE_STRIP; m.action=Marker::ADD;
      m.scale.x=0.01; m.color.a=1.0; m.color.r=R; m.color.g=G; m.color.b=B;
      for (auto& q : path_iter) { std::array<Eigen::Isometry3d,7> Tj; compute_fk_chain_(q,Tj);
        auto p=Tj[6].translation(); geometry_msgs::msg::Point P; P.x=p.x(); P.y=p.y(); P.z=p.z(); m.points.push_back(P); }
      arr.markers.push_back(m);
    }
    // === Start / Goal TCP markers ===
    if (!path_iter.empty()) {
      // 计算 start TCP
      std::array<Eigen::Isometry3d,7> T_start, T_goal;
      compute_fk_chain_(path_iter.front(), T_start);
      compute_fk_chain_(path_iter.back(),  T_goal);

      Eigen::Vector3d p_start = T_start[6].translation();
      Eigen::Vector3d p_goal  = T_goal[6].translation();

      // Start: 绿色球
      {
        Marker m;
        m.header = header;
        m.ns     = ns_iter(iter, "start_tcp");
        m.id     = id_cursor++;
        m.type   = Marker::SPHERE;
        m.action = Marker::ADD;
        m.pose.position.x = p_start.x();
        m.pose.position.y = p_start.y();
        m.pose.position.z = p_start.z();
        m.pose.orientation.w = 1.0;
        m.scale.x = m.scale.y = m.scale.z = 0.04;   // 球的大小
        m.color.a = 1.0;
        m.color.r = 0.1f;
        m.color.g = 0.9f;   // 绿色
        m.color.b = 0.1f;
        arr.markers.push_back(m);
      }

      // Goal: 蓝色球
      {
        Marker m;
        m.header = header;
        m.ns     = ns_iter(iter, "goal_tcp");
        m.id     = id_cursor++;
        m.type   = Marker::SPHERE;
        m.action = Marker::ADD;
        m.pose.position.x = p_goal.x();
        m.pose.position.y = p_goal.y();
        m.pose.position.z = p_goal.z();
        m.pose.orientation.w = 1.0;
        m.scale.x = m.scale.y = m.scale.z = 0.04;
        m.color.a = 1.0;
        m.color.r = 0.1f;
        m.color.g = 0.1f;
        m.color.b = 0.9f;   // 蓝色
        arr.markers.push_back(m);
      }
    }

    // 合并 link 盒三角
    {
      Marker tri; tri.header=header; tri.id=id_cursor++; tri.ns=ns_iter(iter,"links_all"); tri.type=Marker::TRIANGLE_LIST; tri.action=Marker::ADD;
      tri.scale.x=tri.scale.y=tri.scale.z=1.0; tri.color.a=0.45f; tri.color.r=R; tri.color.g=G; tri.color.b=B;
      for (const auto& q : path_iter) {
        auto boxes = build_link_boxes_for_q(q);
        for (const auto& lb : boxes) append_box_as_triangles(lb, tri);
      }
      arr.markers.push_back(tri);
    }

    // 违规最近点/法向线
    auto vios = collect_violations(path_iter);
    for (const auto& vio : vios) {
      Marker pr; pr.header=header; pr.ns=ns_iter(iter,"near_p_robot"); pr.id=id_cursor++; pr.type=Marker::SPHERE; pr.action=Marker::ADD;
      pr.scale.x=pr.scale.y=pr.scale.z=0.03; pr.color.a=1.0; pr.color.r=1.0; pr.color.g=0.0; pr.color.b=0.0;
      pr.pose.position.x=vio.p_robot.x(); pr.pose.position.y=vio.p_robot.y(); pr.pose.position.z=vio.p_robot.z(); pr.pose.orientation.w=1.0;
      arr.markers.push_back(pr);

      Marker po=pr; po.ns=ns_iter(iter,"near_p_obs"); po.id=id_cursor++; po.color.r=0.0; po.color.g=0.2; po.color.b=1.0;
      po.pose.position.x=vio.p_obs.x(); po.pose.position.y=vio.p_obs.y(); po.pose.position.z=vio.p_obs.z();
      arr.markers.push_back(po);

      Marker ln; ln.header=header; ln.ns=ns_iter(iter,"near_normal"); ln.id=id_cursor++; ln.type=Marker::LINE_LIST; ln.action=Marker::ADD;
      ln.scale.x=0.01; ln.color.a=1.0; ln.color.r=0.0; ln.color.g=1.0; ln.color.b=0.0;
      geometry_msgs::msg::Point A,Bp; A.x=vio.p_robot.x();A.y=vio.p_robot.y();A.z=vio.p_robot.z(); Bp.x=vio.p_obs.x();Bp.y=vio.p_obs.y();Bp.z=vio.p_obs.z();
      ln.points.push_back(A); ln.points.push_back(Bp); arr.markers.push_back(ln);
    }

    // 连续安全：不安全边的扫掠 hull
    if (!unsafe_edges_cache_.empty()) {
      for (const auto& e : unsafe_edges_cache_) {
        const auto& qA=e.first; const auto& qB=e.second;
        Marker tri; tri.header=header; tri.id=id_cursor++; tri.ns=ns_iter(iter,"swept_unsafe"); tri.type=Marker::TRIANGLE_LIST; tri.action=Marker::ADD;
        tri.scale.x=tri.scale.y=tri.scale.z=1.0; tri.color.a=0.35f; tri.color.r=0.1f; tri.color.g=0.9f; tri.color.b=0.9f;
        auto boxA=build_link_boxes_for_q(qA), boxB=build_link_boxes_for_q(qB);
       
  const int K_all = static_cast<int>(boxA.size());  // 6 或 8
  for (int k = 0; k < K_all; ++k) {
    auto hull = make_link_swept_hull_data_(boxA[k], boxB[k]);
    append_hull_triangles_to_marker_(hull, tri);
  }

        arr.markers.push_back(tri);

        Marker edge; edge.header=header; edge.id=id_cursor++; edge.ns=ns_iter(iter,"swept_edges"); edge.type=Marker::LINE_LIST; edge.action=Marker::ADD;
        edge.scale.x=0.005; edge.color.a=1.0; edge.color.r=1.0; edge.color.g=0.2; edge.color.b=0.2;
        std::array<Eigen::Isometry3d,7> TA,TB; compute_fk_chain_(qA,TA); compute_fk_chain_(qB,TB);
        auto pA=TA[6].translation(), pB=TB[6].translation();
        geometry_msgs::msg::Point PA,PB; PA.x=pA.x();PA.y=pA.y();PA.z=pA.z(); PB.x=pB.x();PB.y=pB.y();PB.z=pB.z();
        edge.points.push_back(PA); edge.points.push_back(PB);
        arr.markers.push_back(edge);
      }
    }

    // 焊枪可视化
    if (!gripper_enable_ && torch_enable_) {
      auto emit=[&](const Eigen::Matrix<double,6,1>& q, int& idc, const std::string& ns){
        Eigen::Isometry3d T_world = get_tcp_pose_world_(q);
        Eigen::Vector3d p_tcp=T_world.translation(); Eigen::Matrix3d R0=T_world.linear();

        auto makeCyl=[&](const Eigen::Vector3d& center, const Eigen::Vector3d& dir, double len, double dia,
                         double r,double g,double b,double a, const std::string& _ns, int& id)->Marker{
          Marker m; m.header=header; m.ns=_ns; m.id=id++; m.type=Marker::CYLINDER; m.action=Marker::ADD;
          m.scale.x=dia; m.scale.y=dia; m.scale.z=len; m.color.r=r; m.color.g=g; m.color.b=b; m.color.a=a;
          Eigen::Vector3d d=dir; if (d.norm()<1e-12) d=Eigen::Vector3d::UnitZ();
          Eigen::Quaterniond qz = Eigen::Quaterniond::FromTwoVectors(Eigen::Vector3d::UnitZ(), d.normalized());
          m.pose.position.x=center.x(); m.pose.position.y=center.y(); m.pose.position.z=center.z();
          m.pose.orientation.x=qz.x(); m.pose.orientation.y=qz.y(); m.pose.orientation.z=qz.z(); m.pose.orientation.w=qz.w();
          return m;
        };

        Eigen::Vector3d dir1 = R0*Eigen::Vector3d::UnitZ();
        Eigen::Vector3d c1 = p_tcp + 0.5*torch_cyl1_len_*dir1;
        arr.markers.push_back(makeCyl(c1,dir1,torch_cyl1_len_,torch_cyl1_dia_,
                                      torch1_rgba_[0],torch1_rgba_[1],torch1_rgba_[2],torch1_rgba_[3],
                                      ns+"_torch1", idc));
        const double tilt=-torch_tilt_deg_*M_PI/180.0;
        Eigen::Matrix3d Rtilt = Eigen::AngleAxisd(tilt,Eigen::Vector3d::UnitX()).toRotationMatrix();
        Eigen::Vector3d dir2 = R0*(Rtilt*Eigen::Vector3d::UnitZ());
        Eigen::Vector3d p2 = p_tcp + torch_cyl1_len_*dir1;
        Eigen::Vector3d c2 = p2 + 0.5*torch_cyl2_len_*dir2;
        arr.markers.push_back(makeCyl(c2,dir2,torch_cyl2_len_,torch_cyl2_dia_,
                                      torch2_rgba_[0],torch2_rgba_[1],torch2_rgba_[2],torch2_rgba_[3],
                                      ns+"_torch2", idc));
      };
      if (viz_torch_only_last_) { if (!path_iter.empty()) emit(path_iter.back(), id_cursor, ns_iter(iter,"torch")); }
      else { for (const auto& q: path_iter) emit(q, id_cursor, ns_iter(iter,"torch")); }
    }

    append_obstacles_markers(arr, header);
    
        // ====== 发布前统计内容（可选）======
    size_t tri_points = 0, line_points = 0, spheres = 0, cubes = 0, cylinders = 0, textn = 0, totals = arr.markers.size();
    if (debug_dump_sizes_) {
      for (const auto& m : arr.markers) {
        switch (m.type) {
          case Marker::TRIANGLE_LIST: tri_points += m.points.size(); break;
          case Marker::LINE_LIST:
          case Marker::LINE_STRIP:    line_points += m.points.size(); break;
          case Marker::SPHERE:        spheres++;   break;
          case Marker::CUBE:          cubes++;     break;
          case Marker::CYLINDER:      cylinders++; break;
          case Marker::TEXT_VIEW_FACING: textn++;  break;
          default: break;
        }
      }
    }

    // ====== 正式发布 ======
    pub_->publish(arr);

    // ====== 发布后验证 ======
    if (debug_verify_viz_) {
      // 1) 打印这次实际发送的marker数量/点数（证明消息不是空的）
      RCLCPP_INFO(get_logger(),
        "[VIZ] iter=%d published %zu markers | TRI pts=%zu | LINE pts=%zu | S=%zu C=%zu CYL=%zu TEXT=%zu",
        iter, totals, tri_points, line_points, spheres, cubes, cylinders, textn);

      // 2) 打印当前订阅数量（RViz是否连上）
      size_t cur = pub_->get_subscription_count();
      if (cur != last_sub_count_) {
        if (debug_log_sub_count_) {
          RCLCPP_INFO(get_logger(), "[SUB] /qp_opt/markers subscriptions: %zu", cur);
        }
        last_sub_count_ = cur;
      }

      // 3) 若订阅为0，发出一次性告警（帮助定位话题/QoS/RViz配置问题）
      if (cur == 0) {
        RCLCPP_WARN_THROTTLE(get_logger(), *clock_, 3000,
  "[VIZ] No subscribers on /qp_opt/markers. "
  "Check RViz 'Add->By display type: MarkerArray' topic='/qp_opt/markers' "
  "and Fixed Frame='%s' (qos_transient_local=%s).",
  frame_id_.c_str(), qos_transient_local_ ? "true" : "false");


      }
    }

  }

  // ---------------- 结束/统计 ----------------
void finalize_and_maybe_stop_(const std::string& reason) {
  if (cont_after_global_only_) {
    int idc = base_id_for_iter(cur_iter_);
    for (int pass = 0; pass < 5; ++pass) {
      bool changed = run_continuity_pass_once_();  // use_continuity_check_log_only_=false 时会插点
      publish_markers_for_iter(path_, cur_iter_, idc);
      if (!changed) break;
    }
  } else {
    int idc = base_id_for_iter(cur_iter_);
    publish_markers_for_iter(path_, cur_iter_, idc);
  }
// ===== optional final global smoothing =====
if (enable_final_global_smooth_) {
  run_final_global_smooth_();

  // optional: re-check continuity once after final smoothing
  if (cont_after_global_only_) {
    (void)run_continuity_pass_once_();
  }

  int idc = base_id_for_iter(cur_iter_);
  publish_markers_for_iter(path_, cur_iter_, idc);
}
  const auto elapsed = std::chrono::duration<double,std::milli>(Clock::now()-plan_start_tp_).count();
  const double L = path_length_normed(path_), md = min_true_distance(path_);
  RCLCPP_INFO(get_logger(), "[PLAN DONE] reason=%s | iters=%d | len=%.6f | min_d=%.6f | time=%.3fms",
              reason.c_str(), cur_iter_, L, md, elapsed);

  // ======= 在这里打印最终成功路径（只打印一次）=======
  // 你也可以换成更严谨的 success 判定（比如 min_d >= stop_min_d_ 或 violations empty）
  bool success = false;

// 1) 用 min_d 判
if (stop_when_min_d_ge_) success = (md >= stop_min_d_);
else                     success = (md <= stop_min_d_);
  if (success) {
    dump_final_path_(get_logger(), path_, "FINAL-PATH");
    save_path_csv_(get_logger(), path_, "/home/hyj/iris_rviz_ws/src/iris_rviz_cpp/src/tmp/final_path.csv");   // 你也可以换成你项目目录
  } else {
    RCLCPP_INFO(get_logger(), "[FINAL-PATH] not printed because not marked success (reason=%s)", reason.c_str());
  }
  // ===============================================

  if (stop_when_safe_) timer_->cancel();
}


private:
// —— 将“本轮/本段”会被复用的局部状态全部清零 ——
// 注意：这是“局部QP与连续性修补”的运行时状态，
// 在 private: 增加

void reset_global_state_on_new_package_() {
  // 1) 恢复超参到“基线默认值”
  mu_          = mu_base_;          // 基线值在构造时保存
  trust_s_     = trust_s_base_;
  alpha_       = alpha_base_;
  max_iters_   = max_iters_base_;

  // 2) 清空跨包统计与缓存
  best_cost_           = std::numeric_limits<double>::infinity();
  best_min_d_          = -std::numeric_limits<double>::infinity();
  global_iter_counter_ = 0;
  total_safe_iters_    = 0;
  total_unsafe_iters_  = 0;
  fk_cache_.clear();
  collision_world_stamp_++;

  // 3) 日志/随机数/计时器
  rng_.seed(rng_seed_base_);
  last_log_ts_ = Clock::now();
  csv_row_count_ = 0;

  // 4) 若有外部求解器/场景，做“软清理”
  if (gurobi_model_) gurobi_model_->reset(); // 或者重建模型
  unsafe_edges_global_cache_.clear();
}

// 不会修改全局超参（如 joint limits / mu_ / trust_s_ 等）
void reset_local_state_all_()
{
  // 连续性修补相关
  unsafe_edges_cache_.clear();
  local_safe_iters_        = 0;     // 连续“安全边”计数
  local_viz_iter_counter_  = 0;

  // 局部QP相关（从参数恢复“默认值”，避免上一段/上一轮残留）
  if (this->has_parameter("mu_local"))
    mu_local_ = this->get_parameter("mu_local").as_double();
  if (this->has_parameter("local_trust_s"))
    local_trust_s_ = this->get_parameter("local_trust_s").as_double();
  if (this->has_parameter("local_alpha"))
    local_alpha_ = this->get_parameter("local_alpha").as_double();
  if (this->has_parameter("mu_scale_on_unsafe_local"))
    mu_scale_on_unsafe_local_ = this->get_parameter("mu_scale_on_unsafe_local").as_double();
  if (this->has_parameter("mu_local_max"))
    mu_local_max_ = this->get_parameter("mu_local_max").as_double();
  if (this->has_parameter("warmup_safe_iters_local"))
    warmup_safe_iters_local_ = this->get_parameter("warmup_safe_iters_local").as_int();

  // 订阅计数“脏值”清理（仅影响日志打印节流逻辑）
  last_sub_count_ = std::numeric_limits<size_t>::max();
}

  // ====== 单个关节向量 q 的违规收集（局部QP用） ======
  std::vector<Violation>
  collect_violations_single_q_(const Eigen::Matrix<double,6,1>& q) const {
    std::vector<Violation> vios;
    if (obstacles_.empty()) return vios;

    auto link_boxes = build_link_boxes_for_q(q);
    const int K_link = 6;
    const int K_all  = (int)link_boxes.size(); // 可能含焊枪

    for (int k = 0; k < K_link; ++k) {
      double best_d = std::numeric_limits<double>::infinity();
      Eigen::Vector3d best_pr, best_po;

      // 先本体段
      for (const auto& ob : obstacles_) {
        {
          fcl::DistanceRequestd req; req.enable_nearest_points = true; req.enable_signed_distance = true;
          fcl::DistanceResultd  res;
          double d = fcl::distance(link_boxes[k].obj.get(), ob.obj.get(), req, res);
          if (d < best_d) { best_d = d; best_pr = res.nearest_points[0]; best_po = res.nearest_points[1]; }
        }
        // 若是末段，把焊枪并入
        if (k == 5 && K_all > K_link) {
          for (int tk = K_link; tk < K_all; ++tk) {
            fcl::DistanceRequestd req; req.enable_nearest_points = true; req.enable_signed_distance = true;
            fcl::DistanceResultd  res;
            double d = fcl::distance(link_boxes[tk].obj.get(), ob.obj.get(), req, res);
            if (d < best_d) { best_d = d; best_pr = res.nearest_points[0]; best_po = res.nearest_points[1]; }
          }
        }
      }

      if (best_d < d_safe_) {
        Violation vio;
        vio.t = 0;      // 单点版本，这里随便填
        vio.k = k;
        vio.d = best_d;
        vio.p_robot = best_pr;
        vio.p_obs   = best_po;

        Eigen::Vector3d n_world = (best_d >= 0.0) ? (best_pr - best_po) : (best_po - best_pr);
        if (n_world.norm() < 1e-12) n_world = Eigen::Vector3d::UnitX(); else n_world.normalize();
        vio.n = n_world;

        const int base_seg_for_J = (k < 5) ? k : 5;
        Eigen::Matrix<double,3,6> Jp = compute_position_jacobian_moveit_base_(q, base_seg_for_J, best_pr);
        vio.wn = (n_world.transpose() * Jp);

        if (k < 5) {
          for (int j = k + 1; j < 6; ++j) vio.wn(0, j) = 0.0;
        }

        vios.push_back(vio);
      }
    }
    return vios;
  }

// —— QPTrajOptNode private: （成员区）
  // 路径
  Eigen::Matrix<double,6,1> qmin6_, qmax6_;
  Eigen::Matrix<double,6,1> q_start_, q_goal_;
  std::vector<Eigen::Matrix<double,6,1>> path_;
  std::string init_q_file_;
// ===== Convex-set guidance =====
bool use_convexset_guidance_{false};
std::string convexset_result_yaml_;
bool convexset_q_in_degree_{true};

double lambda_corr_{0.20};
bool use_anchor_pull_{true};
bool use_seed_pull_{false};
double lambda_seed_q_{0.05};
bool   use_poly_membership_penalty_{false};
double lambda_poly_membership_{0.10};
// ===== Final global smoothing =====
bool enable_final_global_smooth_{false};
int    final_global_smooth_iters_{8};
double final_global_smooth_alpha_{6.0};
double final_global_smooth_mu_scale_{0.25};
double final_global_smooth_lambda_corr_{0.05};
double final_global_smooth_trust_s_{0.08};

// loaded guide data
ConvexSetGuideData guide_;
  // NEW: 手动 FK 验证用
  bool fk_manual_q_enable_{false};
  std::vector<double> fk_manual_q_deg_;

// === Simple 3D Quickhull for up to few dozen points (here 16) ===


  // NEW: 是否打印当前可视化用的整条路径 q（单位：deg）
  bool  debug_print_q_{false};
  
  // 如果你只想打印一次，可以再加一个：
  bool  debug_print_q_once_{false};
 // NEW: 是否在每次构造 QP 时输出 Q/c

  mutable size_t last_sub_count_{std::numeric_limits<size_t>::max()};

// ======= DH kinematics (pure) =======
double base_yaw_deg_{180.0};  // world->base 绕Z的偏置，默认 180°


  // params/state
    // === Debug / Diagnostics ===
  // === Viz Debug (no heartbeat) ===
  bool  debug_verify_viz_{true};          // 是否开启每次发布的可视化验证日志
  bool  debug_dump_sizes_{true};          // 统计每次Marker内容（点数/数量）
  bool  debug_log_sub_count_{true};       // 订阅数量变化时打印
  bool  qos_transient_local_{false};      // 可选：把发布改成transient_local（RViz后连也能看历史）


  // 局部 QP 的可视化与日志
  bool debug_viz_local_qp_{false};
  bool debug_log_local_qp_{true};
  int  local_viz_iter_counter_{0};

  std::string frame_id_;
  int steps_{7};
  double alpha_{2.5}, mu_{0.8}, d_safe_{0.05}, trust_s_{0.2};
  int max_iters_{20}, iter_period_ms_{0}, cur_iter_{0};
  bool stop_when_safe_{true};
// NEW: unsafe 时自动放大 mu
  double mu_scale_on_unsafe_{1.0};
  double mu_max_{8.0};
  bool   boost_mu_only_after_warmup_{true};
  double trust_s_min_{0.05}, trust_s_max_{1.0};
  double tau_plus_{0.1}, tau_minus_{0.05}, xtol_{1e-3};
  double mi_thresh_{1e-6};
  int    max_trust_attempts_{3};
  int    warmup_safe_iters_{3};
double mu_local_{1.0};
double mu_local_max_{10.0};
double mu_scale_on_unsafe_local_{1.3};
bool   boost_mu_only_after_warmup_local_{false};
int    warmup_safe_iters_local_{0};
int    local_safe_iters_{0};  // 连续安全的局部修补次数

  bool   fk_only_{false};

  double stop_min_d_{0.0};
  bool   stop_when_min_d_ge_{true};
  bool   min_d_ignore_warmup_{true};

  bool cont_after_global_only_{true};
  double cont_min_d_safe_{0.0};
  int    local_seg_max_iters_{20};
  bool   use_continuity_check_log_only_{true};

  bool   fix_last_joint_to_zero_{true};
  int    last_joint_index_{5};
  double last_joint_fixed_value_{0.0};



  // 焊枪
  bool torch_enable_{true}, torch_in_collision_{true}, viz_torch_only_last_{true};
  double torch_cyl1_len_{0.36}, torch_cyl1_dia_{0.025};
  double torch_cyl2_len_{0.11}, torch_cyl2_dia_{0.018};
  double torch_tilt_deg_{45.0};
  std::vector<double> torch1_rgba_{0.9,0.4,0.1,0.9}, torch2_rgba_{0.2,0.8,0.9,0.9};

  // ===== Gripper =====
  bool gripper_enable_{false};
  bool gripper_in_collision_{true};

  double gripper_mount_x_{0.0};
  double gripper_mount_y_{0.0};
  double gripper_mount_z_{0.0};

  double gripper_mount_roll_deg_{0.0};
  double gripper_mount_pitch_deg_{0.0};
  double gripper_mount_yaw_deg_{90.0};

  double gripper_palm_len_{0.060};
  double gripper_palm_width_{0.080};
  double gripper_palm_thick_{0.030};

  double gripper_finger_len_{0.090};
  double gripper_finger_width_{0.012};
  double gripper_finger_thick_{0.020};
  double gripper_opening_{0.080};

  // 障碍物
  std::vector<Obstacle> obstacles_;

  // 连续性缓存（不安全的边）
  std::vector<std::pair<Eigen::Matrix<double,6,1>, Eigen::Matrix<double,6,1>>> unsafe_edges_cache_;

  // 统计
  int total_global_attempts_{0};
  Clock::time_point plan_start_tp_{Clock::now()};

  // ROS
  rclcpp::Publisher<MarkerArray>::SharedPtr pub_;
  rclcpp::TimerBase::SharedPtr timer_;
rclcpp::Clock::SharedPtr clock_;

  // 批量
  bool do_batch_mode_{false};
  std::string batch_in_dir_, batch_glob_, csv_out_;
  bool q_file_is_deg_default_{true};
  bool batch_visualize_{false};
};

// ---------------- main ----------------
// ---------------- main ----------------
int main(int argc, char** argv) {
  rclcpp::init(argc, argv);

  rclcpp::NodeOptions opts;
  opts.allow_undeclared_parameters(false);
  opts.automatically_declare_parameters_from_overrides(false);

  auto node = std::make_shared<QPTrajOptNode>(opts);

  if (node->do_batch_mode()) {
    node->run_batch_and_write_csv();
    rclcpp::shutdown();
    return 0;
  }
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}



