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
  using QPResult = chito_planner::optimization::QPResult;

  QPResult solve_qp_oldstyle(const Eigen::MatrixXd& Q,
                             const Eigen::VectorXd& c,
                             const Eigen::VectorXd& lb,
                             const Eigen::VectorXd& ub,
                             const Eigen::VectorXd& x0) const {
    return chito_planner::optimization::solve_box_qp(Q, c, lb, ub, x0);
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

