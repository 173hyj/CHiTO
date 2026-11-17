  bool repair_variable_window_(std::vector<Eigen::Matrix<double,6,1>>& work,int center){
    using chito_planner::core::collision_free_window;
    std::vector<double> distances;for(const auto& q:work)distances.push_back(min_true_distance_path_({q}));
    const auto window=collision_free_window(distances,center);
    std::ofstream trace;if(const char* f=std::getenv("CHITO_WINDOW_TRACE"))trace.open(f,std::ios::app);
    auto vector_json=[&](const auto& values){trace<<'[';bool first=true;for(const auto& v:values){if(!first)trace<<',';first=false;trace<<v;}trace<<']';};
    if(trace){trace<<std::setprecision(17)<<"{\"event\":\"selection\",\"center\":"<<center<<",\"left\":"<<window.left<<",\"right\":"<<window.right<<",\"valid\":"<<int(window.valid)<<",\"nodes\":"<<work.size()<<",\"point_distances\":";vector_json(distances);trace<<"}\n";}
    if(!window.valid){RCLCPP_ERROR(get_logger(),"[FREE-ANCHOR] invalid/colliding global endpoint or nonfinite path; no local movement");return false;}
    const int L=window.left,R=window.right,P=R-L-1,N=6*P;
    if(L==center-1&&R==center+1){const auto before=work[center];(void)run_local_qp_single_(work[center-1],work[center],work[center+1]);return (work[center]-before).norm()>1e-12;}
    reset_local_state_all_();const double target=std::max(cont_min_d_safe_,1e-4),weight=mu_local_;
    const auto original=work; // Whole accepted window is committed atomically.
    Eigen::VectorXd x(N),reference(N);for(int j=0;j<P;++j){x.segment<6>(6*j)=work[L+1+j];reference.segment<6>(6*j)=original[L]+double(j+1)/(P+1)*(original[R]-original[L]);}
    auto decode=[&](const Eigen::VectorXd& v){auto p=original;for(int j=0;j<P;++j)p[L+1+j]=v.segment<6>(6*j);return p;};
    auto clearance=[&](const Eigen::VectorXd& v){auto p=decode(v);std::vector<double> d;for(int j=L+1;j<R;++j)d.push_back(min_true_distance_path_({p[j]}));for(int j=L;j<R;++j)d.push_back(edge_swept_clearance_(p[j],p[j+1]));return d;};
    auto merit=[&](const Eigen::VectorXd& v){auto p=decode(v);double value=.5*local_alpha_*(v-reference).squaredNorm();for(int j=L+1;j<R;++j)for(const auto& vio:collect_violations_single_q_(p[j]))value+=.5*weight*std::pow(std::max(0.,d_safe_-vio.d),2);for(int j=L;j<R;++j)value+=.5*weight*std::pow(std::max(0.,target-edge_swept_clearance_(p[j],p[j+1])),2);return value;};
    auto safe=[&](const Eigen::VectorXd& v){auto p=decode(v);for(int j=L;j<R;++j)if(!edge_continuous_safe_convexbox_(p[j],p[j+1]))return false;return true;};
    bool changed=false;double trust=local_trust_s_;
    for(int iter=0;iter<local_seg_max_iters_;++iter){
      if(safe(x))break;
      const auto old=decode(x);const auto old_d=clearance(x);const double old_merit=merit(x);
      Eigen::MatrixXd Q=local_alpha_*Eigen::MatrixXd::Identity(N,N);Eigen::VectorXd c=-local_alpha_*reference;
      for(int j=0;j<P;++j)for(const auto& vio:collect_violations_single_q_(old[L+1+j])){
        const double gap=d_safe_-vio.d;if(gap<=0.)continue;Eigen::VectorXd g=Eigen::VectorXd::Zero(N);g.segment<6>(6*j)=-vio.wn.transpose();chito_planner::core::add_linearized_squared_hinge(gap,g,x,weight,Q,c);
      }
      for(int edge=L;edge<R;++edge){double h=std::max(0.,target-old_d[P+edge-L]);if(h<=0.)continue;Eigen::VectorXd g=Eigen::VectorXd::Zero(N);
        for(int node=std::max(edge,L+1);node<=std::min(edge+1,R-1);++node)for(int k=0;k<6;++k){auto plus=old,minus=old;plus[node][k]+=1e-4;minus[node][k]-=1e-4;double a=std::max(0.,target-edge_swept_clearance_(plus[edge],plus[edge+1])),b=std::max(0.,target-edge_swept_clearance_(minus[edge],minus[edge+1]));g[6*(node-L-1)+k]=(a-b)/2e-4;}
        chito_planner::core::add_linearized_squared_hinge(h,g,x,weight,Q,c);
      }
      bool accepted=false;
      for(int attempt=0;attempt<max_trust_attempts_;++attempt){
        Eigen::VectorXd lb=x.array()-trust,ub=x.array()+trust;for(int j=0;j<P;++j)for(int k=0;k<6;++k){int ix=6*j+k;lb[ix]=std::max(lb[ix],qmin6_[k]);ub[ix]=std::min(ub[ix],qmax6_[k]);if(fix_last_joint_to_zero_&&k==last_joint_index_)lb[ix]=ub[ix]=last_joint_fixed_value_;}
        auto sol=solve_qp_oldstyle(Q,c,lb,ub,x);
        if(sol.solved&&sol.xnew.allFinite()){
          auto next_d=clearance(sol.xnew);double next_merit=merit(sol.xnew);bool take=chito_planner::core::window_step_acceptable(old_d,next_d,P,target,old_merit,next_merit);
          if(trace){trace<<std::setprecision(17)<<"{\"event\":\"candidate\",\"left\":"<<L<<",\"right\":"<<R<<",\"points\":"<<P<<",\"iter\":"<<iter<<",\"attempt\":"<<attempt<<",\"accepted\":"<<int(take)<<",\"target\":"<<target<<",\"merit_before\":"<<old_merit<<",\"merit_after\":"<<next_merit<<",\"before\":";vector_json(old_d);trace<<",\"after\":";vector_json(next_d);trace<<",\"x_before\":";vector_json(x);trace<<",\"x_candidate\":";vector_json(sol.xnew);trace<<"}\n";}
          if(take){x=sol.xnew;accepted=changed=true;break;}
        }
        trust*=.5;
      }
      if(!accepted)break;
    }
    if(changed)for(int j=0;j<P;++j)work[L+1+j]=x.segment<6>(6*j);
    if(trace){trace<<"{\"event\":\"exit\",\"left\":"<<L<<",\"right\":"<<R<<",\"changed\":"<<int(changed)<<",\"x_final\":";vector_json(x);trace<<"}\n";}
    return changed;
  }

  bool run_free_anchor_pass_(){
    unsafe_edges_cache_.clear();
    // User endpoints are never released: fail closed if either endpoint collides.
    if(path_.size()<2)return false;
    if(min_true_distance_path_({path_.front()})<0.||min_true_distance_path_({path_.back()})<0.){RCLCPP_ERROR(get_logger(),"[FREE-ANCHOR] trajectory start/goal collides; endpoints preserved");return false;}
    bool changed=false;
    if(repair_revisit_existing_)for(int j=1;j+1<int(path_.size());++j){if(edge_continuous_safe_convexbox_(path_[j-1],path_[j])&&edge_continuous_safe_convexbox_(path_[j],path_[j+1]))continue;changed=repair_variable_window_(path_,j)||changed;}
    const int initial_size=path_.size();int inserted=0;
    for(int i=0;i+1<int(path_.size());){
      if(edge_continuous_safe_convexbox_(path_[i],path_[i+1])){++i;continue;}
      unsafe_edges_cache_.emplace_back(path_[i],path_[i+1]);
      int count=chito_planner::core::repair_insertion_budget(repair_interior_points_,initial_size,inserted,repair_waypoint_limit_);
      if(count==0){++i;continue;}
      auto local=chito_planner::core::repair_interpolation(path_[i],path_[i+1],count);
      path_.insert(path_.begin()+i+1,local.begin()+1,local.end()-1);steps_=path_.size();inserted+=count;changed=true;
      // Optimize against the current whole path so released neighbors and their
      // connecting edges are retained, including nodes processed earlier.
      for(int j=1;j<=count;++j)changed=repair_variable_window_(path_,i+j)||changed;
      i+=count+1;
    }
    RCLCPP_INFO(get_logger(),"[FREE-ANCHOR] inserted=%d nodes=%zu changed=%d",inserted,path_.size(),int(changed));return changed;
  }

struct LocalQPOutcome {
    bool accepted{false};
    bool safe_now{false};
  };

  LocalQPOutcome run_swept_local_qp_(const Eigen::Matrix<double,6,1>& a, Eigen::Matrix<double,6,1>& m, const Eigen::Matrix<double,6,1>& b) {
    RCLCPP_INFO(get_logger(), "[SWEPT-QP] enabled midpoint optimization");
    reset_local_state_all_();
    Eigen::Matrix<double,6,1> center=0.5*(a+b);
    double margin=std::max(cont_min_d_safe_,1e-4),weight=mu_local_;
    auto gaps=[&](const Eigen::Matrix<double,6,1>& x) { Eigen::Vector2d h; h<<std::max(0.,margin-edge_swept_clearance_(a,x)),std::max(0.,margin-edge_swept_clearance_(x,b)); return h; };
    auto clearance=[&](const Eigen::Matrix<double,6,1>& x){return chito_planner::core::RepairClearances{min_true_distance_path_({x}),edge_swept_clearance_(a,x),edge_swept_clearance_(x,b)};};
    std::ofstream acceptance_trace;if(const char* f=std::getenv("CHITO_ACCEPT_TRACE"))acceptance_trace.open(f,std::ios::app);
    const bool need_clearances=local_feasibility_first_||acceptance_trace.is_open();
    // Match the QP's point and two-hull squared penalties, thresholds and weight.
    // Global/discrete objectives and the separate safety acceptance gates are unchanged.
    auto merit=[&](const Eigen::Matrix<double,6,1>& x) {
      double point_squared=0.;
      for(const auto& v:collect_violations_single_q_(x))
        point_squared+=std::pow(std::max(0.,d_safe_-v.d),2);
      return smooth_cost_local_(x,center)+0.5*weight*(point_squared+gaps(x).squaredNorm());
    };
    // Diagnostic observations only. Optimizer, bounds and acceptance are unchanged.
    static int call_counter=0; const int call=++call_counter;
    std::ofstream trace; if(const char* f=std::getenv("CHITO_QP_TRACE"))trace.open(f,std::ios::app);
    if(!trace.is_open())trace.setstate(std::ios::failbit);
    auto array=[&](const auto& v){trace<<'[';for(int k=0;k<v.size();++k){if(k)trace<<',';trace<<v(k);}trace<<']';};
    auto state=[&](const char* tag,int iter,int attempt,const Eigen::Matrix<double,6,1>& x,bool solved,bool accept,double trust) {
      if(!trace)return;
      trace<<std::setprecision(17)<<"{\"event\":\""<<tag<<"\",\"call\":"<<call<<",\"iter\":"<<iter<<",\"attempt\":"<<attempt
        <<",\"solved\":"<<int(solved)<<",\"accepted\":"<<int(accept)<<",\"trust\":"<<trust
        <<",\"d_a\":"<<min_true_distance_path_({a})<<",\"d_m\":"<<min_true_distance_path_({x})<<",\"d_b\":"<<min_true_distance_path_({b})
        <<",\"hull_left\":"<<edge_swept_clearance_(a,x)<<",\"hull_right\":"<<edge_swept_clearance_(x,b)
        <<",\"safe_left\":"<<int(edge_continuous_safe_convexbox_(a,x))<<",\"safe_right\":"<<int(edge_continuous_safe_convexbox_(x,b))
        <<",\"smooth\":"<<smooth_cost_local_(x,center)<<",\"point_linear\":"<<true_penalty_single_q_(x)
        <<",\"edge_squared\":"<<0.5*weight*gaps(x).squaredNorm()<<",\"merit\":"<<merit(x)
        <<",\"mu_global\":"<<mu_<<",\"mu_local\":"<<weight<<",\"alpha\":"<<local_alpha_<<",\"d_safe\":"<<d_safe_
        <<",\"margin\":"<<margin<<",\"stop_min_d\":"<<stop_min_d_<<",\"a\":";array(a);trace<<",\"m\":";array(x);trace<<",\"b\":";array(b);
      trace<<"}\n";trace.flush();
    };
    auto point_squared=[&](const Eigen::Matrix<double,6,1>& x){double v=0.;for(const auto& vio:collect_violations_single_q_(x))v+=0.5*weight*std::pow(std::max(0.,d_safe_-vio.d),2);return v;};
    bool changed=false; double trust=local_trust_s_;
    state("entry",-1,-1,m,false,false,trust);
    for(int iter=0;iter<local_seg_max_iters_;++iter) {
      if(edge_continuous_safe_convexbox_(a,m)&&edge_continuous_safe_convexbox_(m,b)){state("exit_safe",iter,-1,m,true,true,trust);return {true,true};}
      auto h=gaps(m); Eigen::MatrixXd Q=local_alpha_*Eigen::MatrixXd::Identity(6,6); Eigen::VectorXd c=-local_alpha_*center;
      for(const auto& v:collect_violations_single_q_(m)) if(d_safe_>v.d) chito_planner::core::add_linearized_squared_hinge(d_safe_-v.d,-v.wn.transpose(),m,weight,Q,c);
      Eigen::Matrix<double,6,2> edge_grad=Eigen::Matrix<double,6,2>::Zero();
      for(int edge=0;edge<2;++edge) if(h(edge)>0.) { Eigen::VectorXd g(6); for(int k=0;k<6;++k) {auto plus=m,minus=m;plus(k)+=1e-4;minus(k)-=1e-4;g(k)=(gaps(plus)(edge)-gaps(minus)(edge))/2e-4;} edge_grad.col(edge)=g; chito_planner::core::add_linearized_squared_hinge(h(edge),g,m,weight,Q,c); }
      if(trace) {
        trace<<std::setprecision(17)<<"{\"event\":\"model\",\"call\":"<<call<<",\"iter\":"<<iter<<",\"model_grad\":";array(Q*m+c);
        trace<<",\"edge_grad_left\":";array(edge_grad.col(0));trace<<",\"edge_grad_right\":";array(edge_grad.col(1));
        if(call<=8) {
          Eigen::Matrix<double,6,1> fd_merit,fd_squared;
          for(int k=0;k<6;++k){auto plus=m,minus=m;plus(k)+=1e-4;minus(k)-=1e-4;
            fd_merit(k)=(merit(plus)-merit(minus))/2e-4;
            fd_squared(k)=(point_squared(plus)-point_squared(minus))/2e-4;}
          trace<<",\"fd_merit_grad\":";array(fd_merit);trace<<",\"fd_point_squared_grad\":";array(fd_squared);
        }
        trace<<"}\n";trace.flush();
      }
      bool accepted=false; double old=merit(m),oldedge=h.squaredNorm();
      const auto old_clearance=need_clearances?clearance(m):chito_planner::core::RepairClearances{};
      for(int attempt=0;attempt<max_trust_attempts_;++attempt) {
        Eigen::VectorXd lb=(m.array()-trust).matrix().cwiseMax(qmin6_),ub=(m.array()+trust).matrix().cwiseMin(qmax6_);
        if(fix_last_joint_to_zero_&&last_joint_index_>=0&&last_joint_index_<6)lb(last_joint_index_)=ub(last_joint_index_)=last_joint_fixed_value_;
        auto sol=solve_qp_oldstyle(Q,c,lb,ub,m);
        if(sol.solved&&sol.xnew.allFinite()) {Eigen::Matrix<double,6,1> candidate=sol.xnew;
          const double candidate_merit=merit(candidate);
          const auto new_clearance=need_clearances?clearance(candidate):chito_planner::core::RepairClearances{};
          bool take=local_feasibility_first_?chito_planner::core::local_repair_acceptable(old_clearance,new_clearance,margin,stop_min_d_,old,candidate_merit,true):
            (candidate_merit<old-1e-12&&gaps(candidate).squaredNorm()<=oldedge+1e-12&&min_true_distance_path_({candidate})>=std::min(stop_min_d_,min_true_distance_path_({m}))-1e-10);
          if(acceptance_trace){
            acceptance_trace<<std::setprecision(17)<<"{\"call\":"<<call<<",\"iter\":"<<iter<<",\"attempt\":"<<attempt<<",\"fixed\":"<<int(local_feasibility_first_)<<",\"accepted\":"<<int(take)<<",\"target\":"<<margin<<",\"point_floor\":"<<stop_min_d_<<",\"merit_before\":"<<old<<",\"merit_after\":"<<candidate_merit<<",\"before\":[";
            for(int k=0;k<3;++k){if(k)acceptance_trace<<',';acceptance_trace<<old_clearance[k];}acceptance_trace<<"],\"after\":[";
            for(int k=0;k<3;++k){if(k)acceptance_trace<<',';acceptance_trace<<new_clearance[k];}acceptance_trace<<"]}\n";
          }
          state("candidate",iter,attempt,candidate,true,take,trust);
          if(trace){trace<<std::setprecision(17)<<"{\"event\":\"step\",\"call\":"<<call<<",\"iter\":"<<iter<<",\"attempt\":"<<attempt
            <<",\"accepted\":"<<int(take)<<",\"model_improvement\":"<<model_value(Q,c,m)-model_value(Q,c,candidate)
            <<",\"merit_improvement\":"<<old-merit(candidate)<<",\"edge_improvement\":"<<oldedge-gaps(candidate).squaredNorm()
            <<",\"point_old\":"<<min_true_distance_path_({m})<<",\"point_new\":"<<min_true_distance_path_({candidate})
            <<",\"step_inf\":"<<(candidate-m).cwiseAbs().maxCoeff()<<"}\n";trace.flush();}
          if(call==1&&iter==0&&attempt==2&&trace) {
            for(double scale:{0.5,0.25,0.125,0.0625,0.03125,0.015625}) {
              Eigen::Matrix<double,6,1> x=m+scale*(candidate-m);
              state("probe_qp_ray",iter,int(std::round(-std::log2(scale))),x,true,false,trust);
            }
            Eigen::Matrix<double,6,1> fd;
            for(int k=0;k<6;++k){auto plus=m,minus=m;plus(k)+=1e-4;minus(k)-=1e-4;fd(k)=(merit(plus)-merit(minus))/2e-4;}
            if(fd.cwiseAbs().maxCoeff()>1e-12) for(double size:{0.025,0.0125,0.00625,0.003125}) {
              Eigen::Matrix<double,6,1> x=m-size*fd/fd.cwiseAbs().maxCoeff();
              state("probe_merit_descent",iter,int(std::round(-std::log2(size))),x,true,false,size);
            }
          }
          if(take) {m=candidate;changed=true;accepted=true;break;}}
        else state("qp_fail",iter,attempt,m,false,false,trust);
        trust*=0.5;
      }
      if(!accepted){state("exit_no_step",iter,-1,m,false,false,trust);break;}
    }
    state("exit",-1,-1,m,true,changed,trust);
    return {changed,edge_continuous_safe_convexbox_(a,m)&&edge_continuous_safe_convexbox_(m,b)};
  }
  // ---------------- 连续性 pass（插中点 + 局部QP简版） ----------------
    // ===== 连续性修补：对单条 (qA -- qM -- qB) 做 6维局部QP，只调整 qM =====
// ===== 连续性修补：对单条 (qA -- qM -- qB) 做 6维局部QP，只调整 qM =====
LocalQPOutcome run_local_qp_single_(const Eigen::Matrix<double,6,1>& qA,
                                    Eigen::Matrix<double,6,1>&       qM_io,
                                    const Eigen::Matrix<double,6,1>& qB)
{
  if(continuous_convexification_) return run_swept_local_qp_(qA,qM_io,qB);
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
  if(split_corridor_nodes_&&continuous_convexification_&&!use_continuity_check_log_only_)return run_split_corridor_pass_();
  if(corridor_moving_endpoints_&&continuous_convexification_&&!use_continuity_check_log_only_)return run_corridor_endpoint_pass_();
  if(free_colliding_neighbors_&&continuous_convexification_&&!use_continuity_check_log_only_)return run_free_anchor_pass_();
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

  int revisited=0,moved=0;
  if(repair_revisit_existing_ && continuous_convexification_) {
    // Coordinate sweep: only internal nodes move; latest neighbors define both hulls.
    // Existing failed insertion nodes no longer become permanently fixed endpoints.
    for(size_t j=1;j+1<path_.size();++j) {
      if(edge_continuous_safe_convexbox_(path_[j-1],path_[j]) &&
         edge_continuous_safe_convexbox_(path_[j],path_[j+1])) continue;
      const auto before=path_[j];
      (void)run_local_qp_single_(path_[j-1],path_[j],path_[j+1]);
      ++revisited;
      moved+=(path_[j]-before).cwiseAbs().maxCoeff()>1e-12;
    }
    RCLCPP_INFO(get_logger(),"[REVISIT] existing=%d moved=%d nodes=%zu",revisited,moved,path_.size());
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

      const int count=chito_planner::core::repair_insertion_budget(
        repair_interior_points_,static_cast<int>(path_.size()),inserted,repair_waypoint_limit_);
      if(count==0)continue;
      auto local=chito_planner::core::repair_interpolation(qA,qB,count);
      // Fixed original endpoints; a left-to-right local-QP sweep uses the
      // latest left neighbor and the current right neighbor for BOTH hulls.
      for(int j=1;j<=count;++j)
        (void)run_local_qp_single_(local[j-1],local[j],local[j+1]);
      new_path.insert(new_path.end(),local.begin()+1,local.end()-1);
      inserted+=count;
    }
  }

  new_path.push_back(path_.back());

  if (inserted > 0) {
    path_.swap(new_path);
    steps_ = static_cast<int>(path_.size());
    RCLCPP_INFO(get_logger(),
                "[CONT] inserted %d interior points -> steps=%d",
                inserted, steps_);
    return true;
  } else {
    if(!unsafe_edges_cache_.empty())
      RCLCPP_WARN(get_logger(), "[CONT] no insertion: waypoint budget exhausted; %zu unsafe edges remain",unsafe_edges_cache_.size());
    else RCLCPP_INFO(get_logger(), "[CONT] all edges continuously safe.");
    return moved>0;
  }
}


  // ---------------- 批量评估 ----------------
