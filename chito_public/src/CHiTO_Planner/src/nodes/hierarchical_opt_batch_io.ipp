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
