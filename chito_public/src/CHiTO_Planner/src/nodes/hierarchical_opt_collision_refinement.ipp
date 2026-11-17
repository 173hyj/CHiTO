// Additional final layer, preserving the preceding pipeline unchanged.
std::string refine_remaining_collisions_(const chito_planner::OptimizerRequest& request){
 RCLCPP_INFO(get_logger(),"[FINAL-COLLISION-QP] residual edge refinement");
 auto bad_count=[&](const auto& p){int n=0;for(size_t i=1;i<p.size();++i)n+=!edge_continuous_safe_convexbox_(p[i-1],p[i]);return n;};
 const int before=bad_count(path_);int insertions=0,accepted=0,solves=0;
 const double margin=std::max(request.continuous_clearance,1e-4);
 for(int pass=0;pass<10;++pass){
  if(bad_count(path_)==0)break;
  bool changed=false;
  for(int edge=0;edge+1<int(path_.size());++edge){
   if(edge_continuous_safe_convexbox_(path_[edge],path_[edge+1]))continue;
   // First resolve a conservative coarse envelope when both halves certify safe.
   chito_planner::JointVector mid=.5*(path_[edge]+path_[edge+1]);
   if(int(path_.size())<request.repair_waypoint_limit &&
      min_true_distance_path_({mid})>=request.minimum_clearance &&
      edge_continuous_safe_convexbox_(path_[edge],mid)&&edge_continuous_safe_convexbox_(mid,path_[edge+1])){
    path_.insert(path_.begin()+edge+1,mid);steps_=path_.size();++insertions;changed=true;continue;
   }
   std::vector<int> points;
   if(edge>0)points.push_back(edge);
   if(edge+1<int(path_.size())-1)points.push_back(edge+1);
   if(points.empty())continue;
   std::vector<int> edges;
   for(int e=std::max(0,edge-1);e<=std::min(edge+1,int(path_.size())-2);++e)edges.push_back(e);
   const int n=6*points.size(),m=edges.size(),N=n+m;
   auto distances=[&](const auto& p){Eigen::VectorXd d(m);for(int j=0;j<m;++j)d[j]=edge_swept_clearance_(p[edges[j]],p[edges[j]+1]);return d;};
   auto gaps=[&](const Eigen::VectorXd& d){return (margin-d.array()).max(0.).matrix().eval();};
   const auto old_dist=distances(path_);const double old_penalty=gaps(old_dist).squaredNorm();
   Eigen::MatrixXd A=Eigen::MatrixXd::Zero(N+m,N);A.topRows(N).setIdentity();
   for(int k=0;k<n;++k){auto plus=path_,minus=path_;plus[points[k/6]][k%6]+=1e-4;minus[points[k/6]][k%6]-=1e-4;
    A.block(N,k,m,1)=(distances(plus)-distances(minus))/2e-4;}
   if(!A.allFinite())continue;
   Eigen::MatrixXd Q=Eigen::MatrixXd::Identity(N,N);Q.diagonal().tail(m).setConstant(10000.);
   Eigen::VectorXd c=Eigen::VectorXd::Zero(N),lo=Eigen::VectorXd::Constant(N+m,-INFINITY),hi=Eigen::VectorXd::Constant(N+m,INFINITY);
   for(int k=0;k<n;++k){lo[k]=std::max(-.1,qmin6_[k%6]-path_[points[k/6]][k%6]);hi[k]=std::min(.1,qmax6_[k%6]-path_[points[k/6]][k%6]);}
   for(int j=0;j<m;++j){lo[n+j]=0.;A(N+j,n+j)=1.;lo[N+j]=margin-old_dist[j];}
   auto sol=chito_planner::optimization::solve_linear_qp(Q,c,A,lo,hi,Eigen::VectorXd::Zero(N));
   if(!sol.solved)continue;++solves;
   const double point_floor=std::min(request.minimum_clearance,min_true_distance_path_(path_));
   for(double scale=1.;scale>=1./64.;scale*=.5){auto candidate=path_;
    for(int k=0;k<n;++k)candidate[points[k/6]][k%6]+=scale*sol.xnew[k];
    const double point_min=min_true_distance_path_(candidate);
    if(!std::isfinite(point_min)||point_min<point_floor)continue;
    const auto fresh=distances(candidate);if(!fresh.allFinite()||gaps(fresh).squaredNorm()>=old_penalty-1e-12)continue;
    bool preserve=true;for(int e:edges)if(edge_continuous_safe_convexbox_(path_[e],path_[e+1])&&!edge_continuous_safe_convexbox_(candidate[e],candidate[e+1]))preserve=false;
    if(!preserve)continue;
    path_=std::move(candidate);++accepted;changed=true;break;
   }
  }
  if(!changed)break;
 }
 return "; residual_edges_before="+std::to_string(before)+", residual_edges_after="+std::to_string(bad_count(path_))+", lower_QPs="+std::to_string(solves)+", lower_updates="+std::to_string(accepted)+", lower_insertions="+std::to_string(insertions);
}
