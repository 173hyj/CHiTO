  bool split_corridor_nodes_{false},split_corridor_initialized_{false};
  std::vector<int> split_copy_roles_,split_source_nodes_;

  bool split_copy_corridor_safe_(int first=0,int last=-1)const{
    if(!split_corridor_nodes_)return true;
    if(!split_corridor_initialized_)return false;
    if(split_copy_roles_.size()!=path_.size()||corridor_provenance_.size()!=path_.size())return false;
    if(last<0)last=int(path_.size())-1;
    for(int j=std::max(1,first);j<=std::min(last,int(path_.size())-2);++j)if(split_copy_roles_[j]){
      int id=chito_planner::core::corridor_cell(corridor_original_sigma_,corridor_provenance_[j]);
      if(id<0||id>=int(guide_.poly_A.size())||id>=int(guide_.poly_b.size())||chito_planner::core::corridor_residual(guide_.poly_A[id],guide_.poly_b[id],get_tcp_pose_world_(path_[j]).translation())>1e-6)return false;
    }return true;
  }

  std::array<Eigen::Matrix<double,6,1>,2> solve_split_piece_(std::array<Eigen::Matrix<double,6,1>,2> piece,int edge,int count,const std::vector<int>& original_cells){
    using namespace chito_planner::core;const auto initial=piece;const bool fixed[2]={edge==0,edge+2==count};
    std::vector<int> movable;for(int k=0;k<2;++k)if(!fixed[k])movable.push_back(k);const int N=6*movable.size();if(N==0)return piece;
    std::vector<int> cells{original_cells[edge],original_cells[edge+1]};for(int id:cells)if(id<0||id>=int(guide_.poly_A.size())||id>=int(guide_.poly_b.size()))return piece;
    reset_local_state_all_();const double target=std::max(cont_min_d_safe_,1e-4),weight=mu_local_;
    Eigen::VectorXd x(N);for(size_t k=0;k<movable.size();++k)x.segment<6>(6*k)=piece[movable[k]];
    auto decode=[&](const Eigen::VectorXd& z){auto p=initial;for(size_t k=0;k<movable.size();++k)p[movable[k]]=z.segment<6>(6*k);return p;};
    auto clearance=[&](const Eigen::VectorXd& z){auto p=decode(z);return std::vector<double>{min_true_distance_path_({p[0]}),min_true_distance_path_({p[1]}),edge_swept_clearance_(p[0],p[1])};};
    auto membership=[&](const Eigen::VectorXd& z){auto p=decode(z);std::vector<double> v;for(int k:movable)v.push_back(corridor_residual(guide_.poly_A[cells[k]],guide_.poly_b[cells[k]],get_tcp_pose_world_(p[k]).translation()));return v;};
    auto inside=[&](const Eigen::VectorXd& z){for(double a:membership(z))if(!std::isfinite(a)||a>1e-6)return false;return true;};
    auto merit=[&](const Eigen::VectorXd& z){auto p=decode(z);double v=.5*local_alpha_*(p[1]-p[0]).squaredNorm();for(const auto& q:p)for(const auto& vio:collect_violations_single_q_(q))v+=.5*weight*std::pow(std::max(0.,d_safe_-vio.d),2);v+=.5*weight*std::pow(std::max(0.,target-edge_swept_clearance_(p[0],p[1])),2);return v;};
    std::ofstream trace;if(const char* f=std::getenv("CHITO_SPLIT_TRACE"))trace.open(f,std::ios::app);
    auto array=[&](const auto& v){trace<<'[';bool sep=false;for(const auto& a:v){if(sep)trace<<',';sep=true;trace<<a;}trace<<']';};
    if(trace){trace<<std::setprecision(17)<<"{\"event\":\"piece_start\",\"edge\":"<<edge<<",\"original_nodes\":"<<count<<",\"cells\":";array(cells);trace<<",\"movable\":";array(movable);trace<<",\"initial\":[";array(piece[0]);trace<<',';array(piece[1]);trace<<"],\"x_initial\":";array(x);trace<<"}\n";}
    double trust=local_trust_s_;int accepts=0;
    for(int iter=0;iter<local_seg_max_iters_;++iter){
      auto old=decode(x);auto d=clearance(x);if(inside(x)&&std::min(d[0],d[1])>=stop_min_d_&&edge_continuous_safe_convexbox_(old[0],old[1]))break;
      double old_merit=merit(x);Eigen::MatrixXd D=Eigen::MatrixXd::Zero(6,N);Eigen::Matrix<double,6,1> offset=Eigen::Matrix<double,6,1>::Zero();
      for(int k=0;k<2;++k){double sign=k==0?-1.:1.;auto it=std::find(movable.begin(),movable.end(),k);if(it==movable.end())offset+=sign*initial[k];else D.block<6,6>(0,6*(it-movable.begin()))=sign*Eigen::Matrix<double,6,6>::Identity();}
      Eigen::MatrixXd Q=local_alpha_*D.transpose()*D;Eigen::VectorXd c=local_alpha_*D.transpose()*offset;
      for(size_t k=0;k<movable.size();++k)for(const auto& vio:collect_violations_single_q_(old[movable[k]])){double h=d_safe_-vio.d;if(h<=0.)continue;Eigen::VectorXd g=Eigen::VectorXd::Zero(N);g.segment<6>(6*k)=-vio.wn.transpose();add_linearized_squared_hinge(h,g,x,weight,Q,c);}
      double gap=std::max(0.,target-d[2]);if(gap>0){Eigen::VectorXd g(N);for(int k=0;k<N;++k){auto plus=x,minus=x;plus[k]+=1e-4;minus[k]-=1e-4;g[k]=(std::max(0.,target-clearance(plus)[2])-std::max(0.,target-clearance(minus)[2]))/2e-4;}add_linearized_squared_hinge(gap,g,x,weight,Q,c);}
      std::vector<Eigen::MatrixXd> rows;std::vector<Eigen::VectorXd> uppers;int nplanes=0;bool valid=true;
      for(size_t k=0;k<movable.size();++k){int side=movable[k],id=cells[side];Eigen::Matrix<double,3,6> J;for(int a=0;a<6;++a){auto plus=old[side],minus=old[side];plus[a]+=1e-6;minus[a]-=1e-6;J.col(a)=(get_tcp_pose_world_(plus).translation()-get_tcp_pose_world_(minus).translation())/2e-6;}
        Eigen::MatrixXd C;Eigen::VectorXd u;if(!linearized_corridor(guide_.poly_A[id],guide_.poly_b[id],get_tcp_pose_world_(old[side]).translation(),J,old[side],C,u)){valid=false;break;}
        Eigen::MatrixXd row=Eigen::MatrixXd::Zero(C.rows(),N);row.block(0,6*k,C.rows(),6)=C;rows.push_back(row);uppers.push_back(u);nplanes+=C.rows();}
      if(!valid)break;bool accepted=false;
      for(int attempt=0;attempt<max_trust_attempts_;++attempt){
        Eigen::MatrixXd A=Eigen::MatrixXd::Zero(N+nplanes,N);A.topRows(N).setIdentity();Eigen::VectorXd lo=Eigen::VectorXd::Constant(N+nplanes,-INFINITY),hi(N+nplanes);
        for(int k=0;k<N;++k){lo[k]=std::max(x[k]-trust,qmin6_[k%6]);hi[k]=std::min(x[k]+trust,qmax6_[k%6]);if(fix_last_joint_to_zero_&&k%6==last_joint_index_)lo[k]=hi[k]=last_joint_fixed_value_;}
        int off=N;for(size_t k=0;k<rows.size();++k){A.middleRows(off,rows[k].rows())=rows[k];hi.segment(off,rows[k].rows())=uppers[k];off+=rows[k].rows();}
        ++qp_attempts_;auto sol=chito_planner::optimization::solve_linear_qp(Q,c,A,lo,hi,x);if(sol.solved)++qp_solved_;
        if(sol.solved&&sol.xnew.allFinite()){
          auto next=clearance(sol.xnew);auto residual=membership(sol.xnew);double next_merit=merit(sol.xnew);bool in=inside(sol.xnew),take=in&&split_piece_step(d,next,2,target,old_merit,next_merit);
          if(trace){trace<<std::setprecision(17)<<"{\"event\":\"piece_candidate\",\"edge\":"<<edge<<",\"iter\":"<<iter<<",\"attempt\":"<<attempt<<",\"accepted\":"<<int(take)<<",\"target\":"<<target<<",\"trust\":"<<trust<<",\"merit_before\":"<<old_merit<<",\"merit_after\":"<<next_merit<<",\"before\":";array(d);trace<<",\"after\":";array(next);trace<<",\"x_before\":";array(x);trace<<",\"x_candidate\":";array(sol.xnew);trace<<",\"membership_after\":";array(residual);trace<<",\"tcp_after\":[";auto candidate=decode(sol.xnew);for(size_t k=0;k<movable.size();++k){if(k)trace<<',';array(get_tcp_pose_world_(candidate[movable[k]]).translation());}trace<<"]}\n";}
          if(take){x=sol.xnew;accepted=true;++accepts;break;}
        }else if(trace)trace<<"{\"event\":\"piece_qp_fail\",\"edge\":"<<edge<<",\"iter\":"<<iter<<",\"attempt\":"<<attempt<<"}\n";
        trust*=.5;
      }if(!accepted)break;
    }
    piece=decode(x);if(trace){trace<<std::setprecision(17)<<"{\"event\":\"piece_end\",\"edge\":"<<edge<<",\"accepted_steps\":"<<accepts<<",\"x_final\":";array(x);trace<<",\"final\":[";array(piece[0]);trace<<',';array(piece[1]);trace<<"],\"clearances\":";array(clearance(x));trace<<",\"membership\":";array(membership(x));trace<<"}\n";}return piece;
  }

  bool initialize_split_corridor_(){
    if(!prepare_corridor_provenance_())return false;const auto original=path_;const auto ids=corridor_original_sigma_;
    auto layout=chito_planner::core::split_corridor_layout(ids,repair_waypoint_limit_);if(!layout.valid||ids.size()!=path_.size())return false;
    std::vector<std::array<Eigen::Matrix<double,6,1>,2>> pieces;for(int j=0;j+1<int(original.size());++j)pieces.push_back(solve_split_piece_({original[j],original[j+1]},j,original.size(),ids));
    std::vector<Eigen::Matrix<double,6,1>> built;for(size_t j=0;j<layout.source.size();++j){int source=layout.source[j];bool incoming=j>0&&layout.source[j-1]!=source;built.push_back(source==0?pieces.front()[0]:(source==int(original.size())-1?pieces.back()[1]:(incoming?pieces[source-1][1]:pieces[source][0])));}
    path_=built;steps_=path_.size();corridor_provenance_.clear();for(int src:layout.source)corridor_provenance_.push_back(double(src));split_source_nodes_=layout.source;split_copy_roles_.assign(path_.size(),1);split_copy_roles_.front()=split_copy_roles_.back()=0;split_corridor_initialized_=true;if(!sync_corridor_cells_())return false;
    if(const char* f=std::getenv("CHITO_SPLIT_TRACE")){std::ofstream out(f,std::ios::app);out<<std::setprecision(17)<<"{\"event\":\"assembly_vectors\",\"q\":[";for(size_t j=0;j<path_.size();++j){if(j)out<<',';out<<'[';for(int k=0;k<6;++k){if(k)out<<',';out<<path_[j][k];}out<<']';}out<<"]}\n";}
    log_split_connectors_("assembled");return true;
  }
  void log_split_connectors_(const char* stage){
    std::ofstream trace;if(const char* f=std::getenv("CHITO_SPLIT_TRACE"))trace.open(f,std::ios::app);if(!trace)return;
    trace<<std::setprecision(17)<<"{\"event\":\"connectors\",\"stage\":\""<<stage<<"\",\"nodes\":"<<path_.size()<<",\"copy_roles\":[";for(size_t k=0;k<split_copy_roles_.size();++k){if(k)trace<<',';trace<<split_copy_roles_[k];}trace<<"],\"source_nodes\":[";for(size_t k=0;k<split_source_nodes_.size();++k){if(k)trace<<',';trace<<split_source_nodes_[k];}trace<<"],\"connectors\":[";
    bool sep=false;std::map<int,int> prior;for(int j=1;j+1<int(path_.size());++j)if(split_copy_roles_[j]){int id=split_source_nodes_[j];auto it=prior.find(id);if(it!=prior.end()){int left=it->second;double d=INFINITY;bool safe=true;for(int k=left;k<j;++k){d=std::min(d,edge_swept_clearance_(path_[k],path_[k+1]));safe=edge_continuous_safe_convexbox_(path_[k],path_[k+1])&&safe;}if(sep)trace<<',';sep=true;trace<<"{\"source\":"<<id<<",\"left\":"<<left<<",\"right\":"<<j<<",\"joint_gap\":"<<(path_[left]-path_[j]).norm()<<",\"minimum\":"<<d<<",\"safe\":"<<int(safe)<<"}";}else prior[id]=j;}
    trace<<"],\"copy_corridor_safe\":"<<int(split_copy_corridor_safe_())<<"}\n";
  }
  bool run_split_corridor_pass_(){
    if(path_.size()<2||!path_.front().allFinite()||!path_.back().allFinite()||min_true_distance_path_({path_.front()})<0||min_true_distance_path_({path_.back()})<0)return false;
    bool initialized=false;if(!split_corridor_initialized_){if(!initialize_split_corridor_())return false;initialized=true;}
    bool changed=run_corridor_endpoint_pass_();log_split_connectors_("stitched");return changed||initialized;
  }
