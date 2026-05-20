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

