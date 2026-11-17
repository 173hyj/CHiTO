  bool corridor_moving_endpoints_{false};
  std::vector<int> corridor_original_sigma_;
  std::vector<double> corridor_provenance_;

  bool prepare_corridor_provenance_(){
    if(corridor_provenance_.empty()){
      if(!guide_.loaded||guide_.sigma_dense.size()!=path_.size())return false;
      corridor_original_sigma_=guide_.sigma_dense;for(size_t j=0;j<path_.size();++j)corridor_provenance_.push_back(double(j));
    }
    return corridor_provenance_.size()==path_.size();
  }
  bool sync_corridor_cells_(){
    if(corridor_provenance_.size()!=path_.size())return false;
    guide_.sigma_dense.clear();for(double u:corridor_provenance_){int id=chito_planner::core::corridor_cell(corridor_original_sigma_,u);if(id<0)return false;guide_.sigma_dense.push_back(id);}return true;
  }

  bool repair_corridor_endpoints_(int center){
    using namespace chito_planner::core;
    std::vector<double> distances;for(const auto& q:path_)distances.push_back(min_true_distance_path_({q}));
    const auto w=moving_endpoint_window(distances,center);if(!w.valid||!prepare_corridor_provenance_())return false;
    const int first=std::max(1,w.left),last=std::min(int(path_.size())-2,w.right),P=last-first+1,N=6*P;
    if(P<1)return false;
    const int haloL=first-1,haloR=last+1;
    const auto original=path_;std::vector<int> constrained,cells;
    if(split_corridor_nodes_&&split_corridor_initialized_&&split_copy_roles_.size()!=path_.size())return false;
    for(int node:{w.left,w.right})if(node>0&&node+1<int(path_.size())&&(!(split_corridor_nodes_&&split_corridor_initialized_)||split_copy_roles_[node])){
      int id=corridor_cell(corridor_original_sigma_,corridor_provenance_[node]);
      if(id<0||id>=int(guide_.poly_A.size())||id>=int(guide_.poly_b.size()))return false;
      Eigen::MatrixXd A;Eigen::VectorXd b;if(!normalized_corridor(guide_.poly_A[id],guide_.poly_b[id],A,b))return false;
      constrained.push_back(node);cells.push_back(id);
    }
    if(split_corridor_nodes_&&split_corridor_initialized_){
      if(split_copy_roles_.size()!=path_.size())return false;
      for(int node=first;node<=last;++node)if(split_copy_roles_[node]&&std::find(constrained.begin(),constrained.end(),node)==constrained.end()){
        int id=corridor_cell(corridor_original_sigma_,corridor_provenance_[node]);if(id<0||id>=int(guide_.poly_A.size())||id>=int(guide_.poly_b.size()))return false;
        Eigen::MatrixXd A;Eigen::VectorXd b;if(!normalized_corridor(guide_.poly_A[id],guide_.poly_b[id],A,b))return false;constrained.push_back(node);cells.push_back(id);
      }
    }
    reset_local_state_all_();const double target=std::max(cont_min_d_safe_,1e-4),weight=mu_local_;
    Eigen::VectorXd x(N);for(int j=0;j<P;++j)x.segment<6>(6*j)=path_[first+j];
    auto decode=[&](const Eigen::VectorXd& z){auto p=original;for(int j=0;j<P;++j)p[first+j]=z.segment<6>(6*j);return p;};
    auto gaps=[&](const Eigen::VectorXd& z){auto p=decode(z);std::vector<double> d;for(int j=first;j<=last;++j)d.push_back(min_true_distance_path_({p[j]}));for(int j=haloL;j<haloR;++j)d.push_back(edge_swept_clearance_(p[j],p[j+1]));return d;};
    auto membership=[&](const Eigen::VectorXd& z){auto p=decode(z);std::vector<double> residual;for(size_t k=0;k<constrained.size();++k)residual.push_back(corridor_residual(guide_.poly_A[cells[k]],guide_.poly_b[cells[k]],get_tcp_pose_world_(p[constrained[k]]).translation()));return residual;};
    auto in_corridor=[&](const Eigen::VectorXd& z){for(double r:membership(z))if(!std::isfinite(r)||r>1e-6)return false;return true;};
    auto safe=[&](const Eigen::VectorXd& z){if(!in_corridor(z))return false;auto p=decode(z);for(int j=haloL;j<haloR;++j)if(!edge_continuous_safe_convexbox_(p[j],p[j+1]))return false;return true;};
    // Original three-node local smoothness, summed over each movable node.
    // Fixed halo nodes enter the constant term. No pull to the seed trajectory.
    Eigen::MatrixXd L=Eigen::MatrixXd::Zero(N,N);Eigen::VectorXd offset=Eigen::VectorXd::Zero(N);
    for(int j=first;j<=last;++j)for(int k=j-1;k<=j+1;++k){double a=k==j?1.:-.5;if(k>=first&&k<=last)L.block<6,6>(6*(j-first),6*(k-first))+=a*Eigen::Matrix<double,6,6>::Identity();else offset.segment<6>(6*(j-first))+=a*original[k];}
    auto merit=[&](const Eigen::VectorXd& z){auto p=decode(z);double v=.5*local_alpha_*(L*z+offset).squaredNorm();for(int j=first;j<=last;++j)for(const auto& vio:collect_violations_single_q_(p[j]))v+=.5*weight*std::pow(std::max(0.,d_safe_-vio.d),2);for(int j=haloL;j<haloR;++j)v+=.5*weight*std::pow(std::max(0.,target-edge_swept_clearance_(p[j],p[j+1])),2);return v;};
    std::ofstream trace;if(const char* f=std::getenv("CHITO_CORRIDOR_TRACE"))trace.open(f,std::ios::app);
    auto array=[&](const auto& v){trace<<'[';bool sep=false;for(const auto& a:v){if(sep)trace<<',';sep=true;trace<<a;}trace<<']';};
    if(trace){trace<<std::setprecision(17)<<"{\"event\":\"selection\",\"center\":"<<center<<",\"left\":"<<w.left<<",\"right\":"<<w.right<<",\"first\":"<<first<<",\"last\":"<<last<<",\"nodes\":"<<path_.size()<<",\"point_distances\":";array(distances);trace<<",\"constrained\":";array(constrained);trace<<",\"cells\":";array(cells);trace<<",\"provenance\":";array(corridor_provenance_);trace<<",\"original_cells\":";array(corridor_original_sigma_);trace<<",\"copy_roles\":";array(split_copy_roles_);trace<<",\"x_initial\":";array(x);trace<<",\"membership_initial\":";array(membership(x));trace<<"}\n";}
    bool changed=false;double trust=local_trust_s_;
    for(int iter=0;iter<local_seg_max_iters_;++iter){
      if(safe(x))break;
      auto old=decode(x);auto old_d=gaps(x);double old_merit=merit(x);
      Eigen::MatrixXd Q=local_alpha_*L.transpose()*L;Eigen::VectorXd c=local_alpha_*L.transpose()*offset;
      for(int j=first;j<=last;++j)for(const auto& vio:collect_violations_single_q_(old[j])){double h=d_safe_-vio.d;if(h<=0.)continue;Eigen::VectorXd g=Eigen::VectorXd::Zero(N);g.segment<6>(6*(j-first))=-vio.wn.transpose();add_linearized_squared_hinge(h,g,x,weight,Q,c);}
      for(int edge=haloL;edge<haloR;++edge){double h=std::max(0.,target-old_d[P+edge-haloL]);if(h<=0.)continue;Eigen::VectorXd g=Eigen::VectorXd::Zero(N);
        for(int node=std::max(edge,first);node<=std::min(edge+1,last);++node)for(int k=0;k<6;++k){auto plus=old,minus=old;plus[node][k]+=1e-4;minus[node][k]-=1e-4;double a=std::max(0.,target-edge_swept_clearance_(plus[edge],plus[edge+1])),b=std::max(0.,target-edge_swept_clearance_(minus[edge],minus[edge+1]));g[6*(node-first)+k]=(a-b)/2e-4;}add_linearized_squared_hinge(h,g,x,weight,Q,c);
      }
      std::vector<Eigen::MatrixXd> rows;std::vector<Eigen::VectorXd> uppers;int plane_count=0;
      for(size_t k=0;k<constrained.size();++k){int node=constrained[k],id=cells[k];Eigen::Vector3d p=get_tcp_pose_world_(old[node]).translation();Eigen::Matrix<double,3,6> J;
        // Differentiate the actual FK used by exact membership validation.
        for(int a=0;a<6;++a){auto plus=old[node],minus=old[node];plus[a]+=1e-6;minus[a]-=1e-6;J.col(a)=(get_tcp_pose_world_(plus).translation()-get_tcp_pose_world_(minus).translation())/2e-6;}
        Eigen::MatrixXd C;Eigen::VectorXd u;if(!linearized_corridor(guide_.poly_A[id],guide_.poly_b[id],p,J,old[node],C,u))return false;
        Eigen::MatrixXd row=Eigen::MatrixXd::Zero(C.rows(),N);row.block(0,6*(node-first),C.rows(),6)=C;plane_count+=C.rows();rows.push_back(row);uppers.push_back(u);
      }
      bool accepted=false;
      for(int attempt=0;attempt<max_trust_attempts_;++attempt){
        Eigen::MatrixXd A=Eigen::MatrixXd::Zero(N+plane_count,N);A.topRows(N).setIdentity();Eigen::VectorXd lower=Eigen::VectorXd::Constant(N+plane_count,-INFINITY),upper(N+plane_count);
        for(int j=0;j<P;++j)for(int k=0;k<6;++k){int ix=6*j+k;lower[ix]=std::max(x[ix]-trust,qmin6_[k]);upper[ix]=std::min(x[ix]+trust,qmax6_[k]);if(fix_last_joint_to_zero_&&k==last_joint_index_)lower[ix]=upper[ix]=last_joint_fixed_value_;}
        int off=N;for(size_t k=0;k<rows.size();++k){A.middleRows(off,rows[k].rows())=rows[k];upper.segment(off,rows[k].rows())=uppers[k];off+=rows[k].rows();}
        ++qp_attempts_;auto sol=chito_planner::optimization::solve_linear_qp(Q,c,A,lower,upper,x);if(sol.solved)++qp_solved_;
        if(sol.solved&&sol.xnew.allFinite()){
          auto next_d=gaps(sol.xnew);double next_merit=merit(sol.xnew);bool in=in_corridor(sol.xnew);bool feasible_step=window_step_acceptable(old_d,next_d,P,target,old_merit,next_merit);bool take=in&&feasible_step;
          if(trace){trace<<std::setprecision(17)<<"{\"event\":\"candidate\",\"iter\":"<<iter<<",\"attempt\":"<<attempt<<",\"points\":"<<P<<",\"accepted\":"<<int(take)<<",\"in_corridor\":"<<int(in)<<",\"clearance_step\":"<<int(feasible_step)<<",\"trust\":"<<trust<<",\"target\":"<<target<<",\"merit_before\":"<<old_merit<<",\"merit_after\":"<<next_merit<<",\"before\":";array(old_d);trace<<",\"after\":";array(next_d);trace<<",\"x_before\":";array(x);trace<<",\"x_candidate\":";array(sol.xnew);trace<<",\"membership_after\":";array(membership(sol.xnew));trace<<",\"tcp_after\":[";auto next=decode(sol.xnew);for(size_t k=0;k<constrained.size();++k){if(k)trace<<',';array(get_tcp_pose_world_(next[constrained[k]]).translation());}trace<<"]}\n";}
          if(take){x=sol.xnew;changed=accepted=true;break;}
        }else if(trace)trace<<"{\"event\":\"qp_fail\",\"iter\":"<<iter<<",\"attempt\":"<<attempt<<",\"trust\":"<<trust<<"}\n";
        trust*=.5;
      }
      if(!accepted)break;
    }
    if(changed)for(int j=0;j<P;++j)path_[first+j]=x.segment<6>(6*j);
    if(trace){trace<<std::setprecision(17)<<"{\"event\":\"exit\",\"changed\":"<<int(changed)<<",\"x_final\":";array(x);trace<<",\"membership_final\":";array(membership(x));trace<<"}\n";}
    return changed;
  }

  bool run_corridor_endpoint_pass_(){
    if(path_.size()<2||!prepare_corridor_provenance_()){RCLCPP_ERROR(get_logger(),"[CORRIDOR-ENDS] missing or inconsistent corridor provenance; no repair");return false;}
    if(min_true_distance_path_({path_.front()})<0.||min_true_distance_path_({path_.back()})<0.){RCLCPP_ERROR(get_logger(),"[CORRIDOR-ENDS] colliding true start/goal; no repair");return false;}
    unsafe_edges_cache_.clear();bool changed=false;
    if(repair_revisit_existing_)for(int j=1;j+1<int(path_.size());++j){if(edge_continuous_safe_convexbox_(path_[j-1],path_[j])&&edge_continuous_safe_convexbox_(path_[j],path_[j+1])&&split_copy_corridor_safe_(j-1,j+1))continue;changed=repair_corridor_endpoints_(j)||changed;}
    const int initial_size=path_.size();int inserted=0;
    for(int i=0;i+1<int(path_.size());){
      if(edge_continuous_safe_convexbox_(path_[i],path_[i+1])){++i;continue;}
      unsafe_edges_cache_.emplace_back(path_[i],path_[i+1]);int count=chito_planner::core::repair_insertion_budget(repair_interior_points_,initial_size,inserted,repair_waypoint_limit_);
      if(count==0){++i;continue;}
      auto local=chito_planner::core::repair_interpolation(path_[i],path_[i+1],count);
      if(!chito_planner::core::insert_provenance(corridor_provenance_,i,count))return changed;
      if(split_corridor_nodes_&&split_corridor_initialized_){split_copy_roles_.insert(split_copy_roles_.begin()+i+1,count,0);split_source_nodes_.insert(split_source_nodes_.begin()+i+1,count,-1);}
      path_.insert(path_.begin()+i+1,local.begin()+1,local.end()-1);steps_=path_.size();inserted+=count;changed=true;if(!sync_corridor_cells_())return changed;
      for(int j=1;j<=count;++j)changed=repair_corridor_endpoints_(i+j)||changed;
      i+=count+1;
    }
    RCLCPP_INFO(get_logger(),"[CORRIDOR-ENDS] inserted=%d nodes=%zu changed=%d",inserted,path_.size(),int(changed));return changed;
  }
