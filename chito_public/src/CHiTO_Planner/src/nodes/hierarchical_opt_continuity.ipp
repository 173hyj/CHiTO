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
