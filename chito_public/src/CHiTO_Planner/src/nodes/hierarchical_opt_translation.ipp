// Envelope uses both actual endpoint orientations. Only their center
// translations are linearized; orientations are frozen within each QP.
struct TranslationPlane {
  int t,k; Eigen::Vector3d normal; double gap0,gap1;
  Eigen::Matrix<double,1,6> j0,j1;
};
std::vector<TranslationPlane> translation_planes_(
 const std::vector<Eigen::Matrix<double,6,1>>& path) const {
 std::vector<TranslationPlane> rows;
 for(int t=0;t+1<int(path.size());++t){
  const auto left=build_link_boxes_for_q(path[t]),right=build_link_boxes_for_q(path[t+1]);
  for(int k=0;k<int(left.size());++k){
   const auto R=left[k].T.linear();const Eigen::Vector3d half=.5*left[k].size;
   const auto R1=right[k].T.linear();const Eigen::Vector3d half1=.5*right[k].size;
   const Eigen::Vector3d p0=left[k].T.translation(),p1=right[k].T.translation();
   for(const auto& ob:obstacles_){
    const Eigen::Matrix3d O=ob.obj->getTransform().linear();
    const Eigen::Vector3d center=ob.obj->getTransform().translation();
    std::vector<Eigen::Vector3d> normals;
    for(int a=0;a<3;++a){normals.push_back(R.col(a));normals.push_back(R1.col(a));normals.push_back(O.col(a));
     normals.push_back((p1-p0).cross(O.col(a)));
     for(int b=0;b<3;++b){normals.push_back(R.col(a).cross(O.col(b)));normals.push_back(R1.col(a).cross(O.col(b)));normals.push_back(R.col(a).cross(R1.col(b)));}}
    normals.push_back(p0-center);normals.push_back(p1-center);normals.push_back(.5*(p0+p1)-center);
    auto support=[&](const Eigen::Vector3d& n){
     double radius=0.;
     if(ob.type==ObType::SPHERE)radius=.5*ob.size.x();
     else if(ob.type==ObType::CYLINDER){double z=n.dot(O.col(2));radius=.5*ob.size.z()*std::abs(z)+.5*ob.size.x()*std::sqrt(std::max(0.,1-z*z));}
     else for(int a=0;a<3;++a)radius+=.5*ob.size[a]*std::abs(n.dot(O.col(a)));
     return n.dot(center)+radius;
    };
    double best=-std::numeric_limits<double>::infinity();TranslationPlane row;row.t=t;row.k=k;
    for(auto n:normals){if(n.norm()<1e-10)continue;n.normalize();
     for(int sign:{-1,1}){Eigen::Vector3d axis=sign*n;double radius=0.,radius1=0.;for(int a=0;a<3;++a){radius+=half[a]*std::abs(axis.dot(R.col(a)));radius1+=half1[a]*std::abs(axis.dot(R1.col(a)));}
      const double obstacle=support(axis),g0=axis.dot(p0)-radius-obstacle,g1=axis.dot(p1)-radius1-obstacle;
      if(std::min(g0,g1)>best){best=std::min(g0,g1);row.normal=axis;row.gap0=g0;row.gap1=g1;}}
    }
    if(!std::isfinite(best))throw std::runtime_error("No finite translation separating plane");
    row.j0=row.normal.transpose()*compute_position_jacobian_moveit_base_(path[t],std::min(k,5),p0);
    row.j1=row.normal.transpose()*compute_position_jacobian_moveit_base_(path[t+1],std::min(k,5),p1);
    rows.push_back(row);
   }
  }
 }
 return rows;
}
chito_planner::RefinementResult refine_translation_(const chito_planner::OptimizerRequest& request){
 chito_planner::RefinementResult result;const auto begin=Clock::now();
 RCLCPP_INFO(get_logger(),"[TRANSLATION-SCP] global two-endpoint plane constraints");
 // Retain the original discrete optimizer as a feasible warm start instead
 // of replacing it with an unrelated solve starting directly from raw IK.
 run_to_completion_collect_metrics_(false);
 const double baseline_point_min=min_true_distance_path_(path_);
 const double margin=std::max({request.continuous_clearance,request.minimum_clearance,1e-4}),weight=10000.;
 const int n=int(path_.size())*6;double trust=.1;int solved=0;double initial_gap=INFINITY;
 auto score=[&](const auto& path){double loss=0.;for(const auto& row:translation_planes_(path))loss+=std::pow(std::max(0.,margin-row.gap0),2)+std::pow(std::max(0.,margin-row.gap1),2);return loss;};
 for(int iter=0;iter<request.optimization_max_iterations;++iter){
  const auto planes=translation_planes_(path_);std::vector<TranslationPlane> active;
  if(iter==0)for(const auto& row:planes)initial_gap=std::min({initial_gap,row.gap0,row.gap1});
  for(const auto& row:planes)if(std::min(row.gap0,row.gap1)<margin+.2)active.push_back(row);
  double old=score(path_);if(old<1e-16)break;
  const int m=int(active.size())*2,N=n+m;
  Eigen::MatrixXd Q=Eigen::MatrixXd::Zero(N,N);Eigen::VectorXd c=Eigen::VectorXd::Zero(N),q(n);
  for(int t=0;t<int(path_.size());++t)q.segment<6>(6*t)=path_[t];
  for(int t=0;t+1<int(path_.size());++t)for(int k=0;k<6;++k){int a=6*t+k,b=a+6;Q(a,a)+=2.;Q(b,b)+=2.;Q(a,b)-=2.;Q(b,a)-=2.;}
  c.head(n)=Q.topLeftCorner(n,n)*q;Q.diagonal().head(n).array()+=.01;Q.diagonal().tail(m).setConstant(weight);
  Eigen::MatrixXd A=Eigen::MatrixXd::Zero(N+m,N);A.topRows(N).setIdentity();
  Eigen::VectorXd lower=Eigen::VectorXd::Constant(N+m,-INFINITY),upper=Eigen::VectorXd::Constant(N+m,INFINITY),initial=Eigen::VectorXd::Zero(N);
  for(int i=0;i<n;++i){lower[i]=std::max(-trust,qmin6_[i%6]-q[i]);upper[i]=std::min(trust,qmax6_[i%6]-q[i]);if(i<6||i>=n-6)lower[i]=upper[i]=0.;}
  for(int j=0;j<m;++j)lower[n+j]=0.;
  for(int j=0;j<int(active.size());++j){const auto& row=active[j];
   A.block<1,6>(N+2*j,6*row.t)=row.j0;A(N+2*j,n+2*j)=1.;lower[N+2*j]=margin-row.gap0;
   A.block<1,6>(N+2*j+1,6*(row.t+1))=row.j1;A(N+2*j+1,n+2*j+1)=1.;lower[N+2*j+1]=margin-row.gap1;
  }
  auto solution=chito_planner::optimization::solve_linear_qp(Q,c,A,lower,upper,initial);
  if(!solution.solved){trust*=.5;if(trust<1e-4)break;continue;}++solved;
  bool accepted=false;
  for(double scale=1.;scale>=1./64.;scale*=.5){auto candidate=path_;for(int t=1;t+1<int(path_.size());++t)candidate[t]+=scale*solution.xnew.segment<6>(6*t);
   const double candidate_point_min=min_true_distance_path_(candidate);
   const double current_point_min=min_true_distance_path_(path_);
   if(score(candidate)<old-1e-12 && std::isfinite(candidate_point_min) &&
      candidate_point_min>=std::min(request.minimum_clearance,current_point_min)){
     path_=std::move(candidate);accepted=true;break;}}
  if(!accepted){trust*=.5;if(trust<1e-4)break;}else trust=std::min(.2,trust*1.2);
 }
 std::string lower_stage;
 if(request.continuous_collision_refinement)lower_stage=refine_remaining_collisions_(request);
 result.optimization_ms=std::chrono::duration<double,std::milli>(Clock::now()-begin).count();
 const auto check=Clock::now();double minimum=INFINITY;
 for(const auto& row:translation_planes_(path_))minimum=std::min({minimum,row.gap0,row.gap1});
 result.minimum_distance=minimum;result.continuous_safe=std::isfinite(minimum)&&minimum>=margin-1e-7;
 result.waypoint_safe=result.continuous_safe;result.success=result.continuous_safe;
 if(request.accept_approximate_hull){
  bool hull_safe=true;
  for(size_t t=1;t<path_.size();++t)hull_safe=edge_continuous_safe_convexbox_(path_[t-1],path_[t])&&hull_safe;
  result.waypoint_safe=result.continuous_safe=result.success=hull_safe;
 }
 result.status=result.success?"success":(solved?"validation_failed":"optimization_failed");
 result.message="endpoint_box_envelope: initial_gap="+std::to_string(initial_gap)+", final_gap="+std::to_string(minimum)+", solved_QPs="+std::to_string(solved)+"; distinct endpoint orientations; translation-only Jacobians; no intervening rotation guarantee";
 result.message+="; baseline_point_min="+std::to_string(baseline_point_min)+", final_point_min="+std::to_string(min_true_distance_path_(path_));
 result.message+=lower_stage;
 if(request.accept_approximate_hull)result.message+="; acceptance solely by FCL approximate endpoint hulls (plane gap is diagnostic only)";
 result.trajectory=path_;result.normalized_length=path_length_normed(path_);
 result.validation_ms=std::chrono::duration<double,std::milli>(Clock::now()-check).count();return result;
}
