
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

#include "chito_planner/core/convex_hull.hpp"
#include "chito_planner/core/robot_geometry.hpp"
#include "chito_planner/optimization/qp_solver.hpp"


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
using chito_planner::core::ConvexSetGuideData;
using chito_planner::core::LinkBox;
using chito_planner::core::ObType;
using chito_planner::core::Obstacle;
using chito_planner::core::make_box_from_pose;
using chito_planner::core::make_link_box;

// ---------------- 鍩虹绫诲瀷 ----------------
class QPTrajOptNode : public rclcpp::Node {
public:
  explicit QPTrajOptNode(
      const rclcpp::NodeOptions& options =
        rclcpp::NodeOptions()
          .allow_undeclared_parameters(false)
          .automatically_declare_parameters_from_overrides(false))
  : Node("qp_traj_opt", options)
  {
    // ... 浣犲師鏉ョ殑鏋勯€犱唬鐮佷繚鎸佷笉鍙?...
    print_final_path_ = this->declare_parameter<bool>("print_final_path", true);
    final_path_out_   = this->declare_parameter<std::string>("final_path_out", "");
// 渚嬶細final_path_out:="/tmp/qp_final_path.txt"  (涓嶅～鍒欏彧鎵撳嵃涓嶈惤鐩?

    // ---- Debug 鍙傛暟 ----
    debug_verify_viz_     = this->declare_parameter<bool>("debug_verify_viz", true);
    debug_dump_sizes_     = this->declare_parameter<bool>("debug_dump_sizes", true);
    debug_log_sub_count_  = this->declare_parameter<bool>("debug_log_sub_count", true);
    qos_transient_local_  = this->declare_parameter<bool>("qos_transient_local", false);
        // === 灞€閮?QP 鍙傛暟锛堣繛缁慨琛ョ敤锛?===
    local_trust_s_   = this->declare_parameter<double>("local_trust_s", 0.10);
    local_alpha_     = this->declare_parameter<double>("local_alpha",   1.0);
    local_mu_scale_  = this->declare_parameter<double>("local_mu_scale",1.0);
  // NEW: 灞€閮?QP 鍙鍖?/ 鏃ュ織
    debug_viz_local_qp_   = this->declare_parameter<bool>("debug_viz_local_qp", false);
    debug_log_local_qp_   = this->declare_parameter<bool>("debug_log_local_qp", true);

    // NEW:
    debug_print_q_        = this->declare_parameter<bool>("debug_print_q", false);
    


    clock_ = this->get_clock();



    // ---------------- 閫氱敤鍙傛暟 ----------------
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
// ===== NEW: 纰版挒鏉冮噸鑷€傚簲鏀惧ぇ =====
    mu_scale_on_unsafe_ = this->declare_parameter<double>("mu_scale_on_unsafe", 1.0);
    mu_max_             = this->declare_parameter<double>("mu_max", 8.0);
    boost_mu_only_after_warmup_ =
        this->declare_parameter<bool>("boost_mu_only_after_warmup", true);
    // ---- 鍏ㄥ眬鍙傛暟鍔犺浇闄勮繎鍔犱笂杩欏嚑涓?----
mu_local_ = this->declare_parameter<double>("mu_local", 1.0);
mu_local_max_ = this->declare_parameter<double>("mu_local_max", 10.0);
mu_scale_on_unsafe_local_ =
    this->declare_parameter<double>("mu_scale_on_unsafe_local", 1.3);
// 濡傛灉浣犱篃鎯宠鈥滃彧鍦?warmup 涔嬪悗鎵嶆斁澶?local mu鈥濓紝鍙互鍔犱竴涓竷灏斿弬鏁?
boost_mu_only_after_warmup_local_ =
    this->declare_parameter<bool>("boost_mu_only_after_warmup_local", false);

// 锛堝彲閫夛級灞€閮?warmup 杞暟
warmup_safe_iters_local_ =
    this->declare_parameter<int>("warmup_safe_iters_local", 0);
local_safe_iters_ = 0;  // 璁颁竴涓嬭繛缁€滃畨鍏ㄥ眬閮≦P鈥濈殑杞暟

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

    // 鎻愬墠鍋滄锛堝熀浜庡叏灞€鏈€灏忚窛绂伙級
    stop_min_d_           = this->declare_parameter<double>("stop_min_d", 0.0);
    stop_when_min_d_ge_   = this->declare_parameter<bool>("stop_when_min_d_ge", true);
    min_d_ignore_warmup_  = this->declare_parameter<bool>("min_d_ignore_warmup", true);

    // 鎵归噺妯″紡
    batch_in_dir_   = this->declare_parameter<std::string>("batch_in_dir", "");
    batch_glob_     = this->declare_parameter<std::string>("batch_glob", "run_*_q.txt");
    csv_out_        = this->declare_parameter<std::string>("csv_out", "qp_batch_metrics.csv");
    q_file_is_deg_default_ = this->declare_parameter<bool>("q_file_is_deg_default", true);
    batch_visualize_= this->declare_parameter<bool>("batch_visualize", false);

    // 杩炵画瀹夊叏锛堟壂鎺犱綋锛?
    cont_min_d_safe_            = this->declare_parameter<double>("cont_min_d_safe", 0.0);
    cont_after_global_only_     = this->declare_parameter<bool>("cont_after_global_only", true);
    use_continuity_check_log_only_ = this->declare_parameter<bool>("use_continuity_check_log_only", true);
    local_seg_max_iters_        = this->declare_parameter<int>("local_seg_max_iters", 20);


    // 鏈叧鑺傚浐瀹?
    fix_last_joint_to_zero_ = this->declare_parameter<bool>("fix_last_joint_to_zero", true);
    last_joint_index_       = this->declare_parameter<int>("last_joint_index", 5);
    last_joint_fixed_value_ = this->declare_parameter<double>("last_joint_fixed_value", 0.0);
    if (last_joint_index_ < 0 || last_joint_index_ > 5) {
      RCLCPP_FATAL(get_logger(), "last_joint_index must be in [0,5]");
      rclcpp::shutdown(); return;
    }

    // 鍏宠妭闄愪綅
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
    // === 鍩哄骇鍋忕疆锛歸orld -> base 缁?Z 杞存棆杞?base_yaw_deg_锛堥粯璁?180掳锛?===
    base_yaw_deg_ = this->declare_parameter<double>("base_yaw_deg", 180.0);

    T_world_base_.setIdentity();
    {
      double yaw = base_yaw_deg_ * M_PI / 180.0;
      T_world_base_.linear() =
          Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    }

    // 鐒婃灙鍙傛暟
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
    // FK-only 鏍囧織锛堣鏄庯細鍗充娇涓?true锛屼篃浼氬姞杞介殰纰嶇墿渚涘害閲?鍙鍖栵級
    fk_only_ = this->declare_parameter<bool>("fk_only", false);



    // 鍒濆鍖栬矾寰勶細浼樺厛鏂囦欢
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
    // ========== NEW: 鎵嬪姩杈撳叆涓€娈靛叧鑺傝锛屼粎鐢ㄤ簬 FK 楠岃瘉 ==========

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

      // 鎵嬪姩 FK 楠岃瘉鏃讹紝榛樿鍙仛 FK锛屼笉璺?QP
      fk_only_ = true;

      RCLCPP_WARN(get_logger(),
        "[FK-MANUAL] using fk_manual_q_deg (%zu points) as path; fk_only forced true; steps=%d",
        N, steps_);
    }
    // ==========================================================

    // ---------------- 璇诲彇闅滅鐗╋紙鑷姩璇嗗埆锛歒AML 浼樺厛锛涘け璐ュ垯灏濊瘯 .scene 鐩掑瓙锛?----------------
    // ---------------- 璇诲彇闅滅鐗╋紙鑷姩璇嗗埆锛歒AML 浼樺厛锛涘け璐ュ垯灏濊瘯 .scene 鐩掑瓙锛?----------------
{
  // 榛樿蹇呴』璇诲埌 鈮? 涓殰纰嶇墿锛涘惁鍒欑洿鎺?FATAL 閫€鍑?
  const bool require_obstacles = this->declare_parameter<bool>("require_obstacles", true);

  std::string convex_scene_yaml = this->declare_parameter<std::string>(
      "convex_scene_yaml",
      "/home/hyj/iris_rviz_ws/src/iris_rviz_cpp/src/convex_box/convex_scene.yaml");
// 淇濆瓨褰撳墠鍙傛暟浣滀负鈥滃熀绾库€濓紝渚夸簬 reset_global_state_on_new_package_() 鎭㈠
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


    // 鍙鍖?
        rclcpp::QoS qos(rclcpp::KeepLast(100));
    qos.reliable();
    if (qos_transient_local_) qos.transient_local();  // 鍙€夛細寮€鍚巻鍙叉寔涔呭寲

       pub_ = this->create_publisher<MarkerArray>("/qp_opt", qos);



    // 鍒濆鍙鍖?
    {
      int idc = base_id_for_iter(0);
      publish_markers_for_iter(path_, 0, idc);
      RCLCPP_INFO(get_logger(), "[INIT] min distance = %.6f", min_true_distance(path_));
    }

    // 鎵归噺 or 鍗曟
    do_batch_mode_ = !batch_in_dir_.empty();
    plan_start_tp_ = Clock::now();

 if (!do_batch_mode_) {
      // ========== FK-ONLY 寮€鍏筹細鍙皟璇?FK锛屼笉璺?QP ==========
      if (fk_only_) {
        RCLCPP_WARN(get_logger(),
          "[MODE] fk_only=true: FK-only debug mode, no QP optimization will be run.");
        // 涓嶅垱寤?timer锛屼笉璋冪敤 one_iter_step锛涗笂闈?INIT 鐨勪竴娆″彲瑙嗗寲澶熶綘鍦?RViz 閲岃皟 DH銆?
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
  // -------- 鎵归噺鍏ュ彛锛坢ain 鐢級--------
  bool do_batch_mode() const { return do_batch_mode_; }
  void run_batch_and_write_csv() { run_batch_and_write_csv_(); }

private:
// ====== DH 鍙傛暟涓庡疄鐢ㄥ嚱鏁?======
// 鈥斺€?鏀惧湪 QPTrajOptNode 鐨?private: 閲岋紝浠绘剰鍚堥€備綅缃?鈥斺€?
// 绠€鍗?CSV 杞箟锛氳嫢鍖呭惈閫楀彿/寮曞彿/鎹㈣锛屽垯鐢ㄥ弻寮曞彿鍖呰９锛屽苟鎶婂唴閮ㄥ紩鍙烽噸澶嶄竴閬?
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
    // 鍐欎竴涓洿閫傚悎鍚庡鐞嗙殑CSV-ish鏂囨湰
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
    if (c == '"') out.push_back('"'); // 鍙屽紩鍙疯浆涔変负涓や釜鍙屽紩鍙?
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

// FK cache (key 涓?hash锛屽彲鎸夐渶浣跨敤锛涜繖閲屽厛鍗犱綅涓嶇敤涔熸病鍏崇郴)
std::unordered_map<size_t, std::array<Eigen::Isometry3d,7>> fk_cache_;

// World 鈥渟tamp鈥?鐢ㄤ簬娼滃湪鐨勫満鏅彉鏇存爣璁帮紙鍏堝崰浣嶏級
uint64_t collision_world_stamp_{0};

// RNG锛堢敤浜庡皢鏉ラ渶瑕佺殑闅忔満鏁帮紱杩欓噷涔熷厛鍗犱綅锛?
std::mt19937 rng_{123456u};
uint32_t rng_seed_base_{123456u};

// 鏃ュ織鑺傛祦 / 鎵归噺 CSV 璁℃暟锛堝崰浣嶏級
Clock::time_point last_log_ts_{Clock::now()};
size_t csv_row_count_{0};

// 鑻ヤ互鍚庢兂澶嶇敤涓€涓寔涔?Gurobi 妯″瀷锛岃繖閲岀暀涓€涓崰浣嶃€?
// 鐩墠 solve_qp_oldstyle() 姣忔閮芥柊寤哄眬閮ㄦā鍨嬶紝鎵€浠ヤ负绌轰篃OK銆?
std::unique_ptr<GRBModel> gurobi_model_;

// 鍏ㄥ眬鈥滀笉瀹夊叏杈光€濈紦瀛橈紙鍜?per-iter 鐨?unsafe_edges_cache_ 鍖哄垎锛?
std::vector<std::pair<Eigen::Matrix<double,6,1>, Eigen::Matrix<double,6,1>>> unsafe_edges_global_cache_;

  double local_trust_s_{0.10};
  double local_alpha_{1.0};
  double local_mu_scale_{1.0};


// === Auto-build robot_description/_semantic when missing ===
bool auto_build_when_missing_{true};
std::string urdf_path_, srdf_path_, xacro_file_;
std::vector<std::string> xacro_args_;

// ==== continuous-time swept-volume geometry lives in core/convex_hull.* ====

#include "nodes/hierarchical_opt_obstacle_loading.ipp"

#include "nodes/hierarchical_opt_geometry.ipp"

#include "nodes/hierarchical_opt_global_optimization.ipp"

#include "nodes/hierarchical_opt_continuity.ipp"

#include "nodes/hierarchical_opt_batch_io.ipp"

#include "nodes/hierarchical_opt_visualization.ipp"

#include "nodes/hierarchical_opt_state.ipp"

// ======= DH kinematics (pure) =======
double base_yaw_deg_{180.0};  // world->base 缁昛鐨勫亸缃紝榛樿 180掳


  // params/state
    // === Debug / Diagnostics ===
  // === Viz Debug (no heartbeat) ===
  bool  debug_verify_viz_{true};          // 鏄惁寮€鍚瘡娆″彂甯冪殑鍙鍖栭獙璇佹棩蹇?
  bool  debug_dump_sizes_{true};          // 缁熻姣忔Marker鍐呭锛堢偣鏁?鏁伴噺锛?
  bool  debug_log_sub_count_{true};       // 璁㈤槄鏁伴噺鍙樺寲鏃舵墦鍗?
  bool  qos_transient_local_{false};      // 鍙€夛細鎶婂彂甯冩敼鎴恡ransient_local锛圧Viz鍚庤繛涔熻兘鐪嬪巻鍙诧級


  // 灞€閮?QP 鐨勫彲瑙嗗寲涓庢棩蹇?
  bool debug_viz_local_qp_{false};
  bool debug_log_local_qp_{true};
  int  local_viz_iter_counter_{0};

  std::string frame_id_;
  int steps_{7};
  double alpha_{2.5}, mu_{0.8}, d_safe_{0.05}, trust_s_{0.2};
  int max_iters_{20}, iter_period_ms_{0}, cur_iter_{0};
  bool stop_when_safe_{true};
// NEW: unsafe 鏃惰嚜鍔ㄦ斁澶?mu
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
int    local_safe_iters_{0};  // 杩炵画瀹夊叏鐨勫眬閮ㄤ慨琛ユ鏁?

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



  // 鐒婃灙
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

  // 闅滅鐗?
  std::vector<Obstacle> obstacles_;

  // 杩炵画鎬х紦瀛橈紙涓嶅畨鍏ㄧ殑杈癸級
  std::vector<std::pair<Eigen::Matrix<double,6,1>, Eigen::Matrix<double,6,1>>> unsafe_edges_cache_;

  // 缁熻
  int total_global_attempts_{0};
  Clock::time_point plan_start_tp_{Clock::now()};

  // ROS
  rclcpp::Publisher<MarkerArray>::SharedPtr pub_;
  rclcpp::TimerBase::SharedPtr timer_;
rclcpp::Clock::SharedPtr clock_;

  // 鎵归噺
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



