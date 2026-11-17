#define CHITO_STAGE_LIBRARY
#define CHITO_GLOBAL_DIAGNOSTICS
#include "../src/hierarchical_opt.cpp"
#include "chito_planner/config.hpp"
#include <filesystem>
namespace fs=std::filesystem;
static chito_planner::JointPath read_path(const fs::path& p){
 std::ifstream in(p);std::string line,cell;std::getline(in,line);chito_planner::JointPath path;
 while(std::getline(in,line)){std::istringstream row(line);std::getline(row,cell,',');chito_planner::JointVector q;for(int k=0;k<6;++k){std::getline(row,cell,',');q[k]=std::stod(cell);}path.push_back(q);}return path;
}
static void save_path(const fs::path& p,const chito_planner::JointPath& path){std::ofstream o(p);o<<std::setprecision(17)<<"idx,q1_rad,q2_rad,q3_rad,q4_rad,q5_rad,q6_rad\n";for(size_t i=0;i<path.size();++i){o<<i;for(int k=0;k<6;++k)o<<','<<path[i][k];o<<'\n';}}
int main(int argc,char** argv){
 if(argc<5)return 2;rclcpp::init(0,nullptr);const fs::path input=argv[1],out=argv[2];fs::create_directories(out);
 auto request=chito_planner::read_optimizer_config((input/"effective_config.yaml").string());request.formal_method=false;auto generation=chito_planner::GenerationResult{};generation.success=true;generation.seed=read_path(input/"seed.csv");
 const bool fixed=std::string(argv[3])=="fixed";const std::string mode=argv[4];
 rclcpp::NodeOptions options;options.parameter_overrides({rclcpp::Parameter("torch_enable",false),rclcpp::Parameter("torch_in_collision",false),rclcpp::Parameter("gripper_enable",false),rclcpp::Parameter("global_actual_acceptance",fixed),rclcpp::Parameter("split_corridor_nodes",std::getenv("CHITO_SPLIT_CORRIDOR")&&std::string(std::getenv("CHITO_SPLIT_CORRIDOR"))=="1"),rclcpp::Parameter("corridor_moving_endpoints",std::getenv("CHITO_CORRIDOR_ENDS")&&std::string(std::getenv("CHITO_CORRIDOR_ENDS"))=="1"),rclcpp::Parameter("free_colliding_neighbors",std::getenv("CHITO_FREE_NEIGHBORS")&&std::string(std::getenv("CHITO_FREE_NEIGHBORS"))=="1"),rclcpp::Parameter("local_feasibility_first",std::getenv("CHITO_LOCAL_FEASIBILITY")&&std::string(std::getenv("CHITO_LOCAL_FEASIBILITY"))=="1")});
 QPTrajOptNode node(options,&request,&generation);node.convexset_q_in_degree_=false;if(!node.load_convexset_guide_from_yaml_((input/"guide.yaml").string()))return 3;node.use_convexset_guidance_=true;
 // The loaded YAML reader and exact csv describe the same seed in radians.
 if((node.guide_.q_seed_dense[1]-generation.seed[1]).norm()>1e-12)return 4;
 if(mode=="direction"){
  auto probes=read_path(input/"probes.csv");std::ofstream report(out/"directions.jsonl");report<<std::setprecision(17);
  auto vec=[&](const auto& a){report<<'[';for(int k=0;k<a.size();++k){if(k)report<<',';report<<a(k);}report<<']';};
  for(size_t p=0;p<probes.size();++p){const auto q=probes[p];auto boxes=node.build_link_boxes_for_q(q);
   for(const auto& vio:node.collect_violations_single_q_(q)){
    int obstacle=-1;double best=1e9;fcl::DistanceResultd witness;
    auto dist=[&](const Eigen::Matrix<double,6,1>& x,int ob){auto b=node.build_link_boxes_for_q(x);fcl::DistanceRequestd req;req.enable_signed_distance=true;req.enable_nearest_points=true;fcl::DistanceResultd res;return fcl::distance(b[vio.k].obj.get(),node.obstacles_[ob].obj.get(),req,res);};
    for(size_t h=0;h<node.obstacles_.size();++h){double v=dist(q,h);if(v<best){best=v;obstacle=h;}}
    Eigen::Matrix<double,6,1> g=vio.wn.transpose();
    auto sat=[&](const Eigen::Matrix<double,6,1>& x){auto b=node.build_link_boxes_for_q(x);const auto& robot=b[vio.k];const auto& ob=node.obstacles_[obstacle];const Eigen::Matrix3d A=robot.T.linear(),B=ob.obj->getRotation();Eigen::Vector3d delta=robot.T.translation()-ob.obj->getTranslation();double gap=-1e9;
     std::vector<Eigen::Vector3d> axes;for(int k=0;k<3;++k){axes.push_back(A.col(k));axes.push_back(B.col(k));for(int j=0;j<3;++j)axes.push_back(A.col(k).cross(B.col(j)));}
     for(auto axis:axes){if(axis.norm()<1e-8)continue;axis.normalize();double extent=0.;for(int k=0;k<3;++k)extent+=.5*robot.size[k]*std::abs(axis.dot(A.col(k)))+.5*ob.size[k]*std::abs(axis.dot(B.col(k)));gap=std::max(gap,std::abs(axis.dot(delta))-extent);}return gap;};
    report<<"{\"probe\":"<<p<<",\"link\":"<<vio.k<<",\"obstacle\":"<<obstacle<<",\"d\":"<<best<<",\"sat_gap\":"<<sat(q)<<",\"q\":";vec(q);report<<",\"gradient\":";vec(g);report<<",\"normal\":";vec(vio.n);report<<",\"robot_witness\":";vec(vio.p_robot);report<<",\"obstacle_witness\":";vec(vio.p_obs);
    Eigen::Vector3d material=boxes[vio.k].T.inverse()*vio.p_robot;auto J=node.compute_position_jacobian_moveit_base_(q,vio.k,vio.p_robot);Eigen::Matrix<double,3,6> Jfd;
    for(int k=0;k<6;++k){auto a=q,b=q;a(k)+=1e-6;b(k)-=1e-6;auto ba=node.build_link_boxes_for_q(a),bb=node.build_link_boxes_for_q(b);Jfd.col(k)=((ba[vio.k].T*material)-(bb[vio.k].T*material))/2e-6;}
    report<<",\"material_jacobian_error\":"<<(J-Jfd).norm()/std::max(1e-12,Jfd.norm())<<",\"derivatives\":[";
    bool first=true;for(double eps:{1e-3,1e-4,1e-5,1e-6,1e-7}){Eigen::Matrix<double,6,1> fd,fd_sat;for(int k=0;k<6;++k){auto a=q,b=q;a(k)+=eps;b(k)-=eps;fd(k)=(dist(a,obstacle)-dist(b,obstacle))/(2*eps);fd_sat(k)=(sat(a)-sat(b))/(2*eps);}if(!first)report<<',';first=false;report<<"{\"eps\":"<<eps<<",\"fd\":";vec(fd);report<<",\"sat_fd\":";vec(fd_sat);report<<",\"relative\":"<<(g-fd).norm()/std::max(1e-12,fd.norm())<<",\"cosine\":"<<g.dot(fd)/std::max(1e-12,g.norm()*fd.norm())<<'}';}
    report<<"],\"steps\":[";first=true;
    if(g.cwiseAbs().maxCoeff()>1e-12)for(double step:{1e-4,1e-3,1e-2})for(int sign:{1,-1}){auto x=(q+sign*step*g/g.cwiseAbs().maxCoeff()).eval();double newd=dist(x,obstacle);double nearest=1e9;int nearob=-1;for(size_t h=0;h<node.obstacles_.size();++h){double v=dist(x,h);if(v<nearest){nearest=v;nearob=h;}}if(!first)report<<',';first=false;report<<"{\"step\":"<<step<<",\"sign\":"<<sign<<",\"d\":"<<newd<<",\"sat_gap\":"<<sat(x)<<",\"nearest_obstacle\":"<<nearob<<",\"nearest_d\":"<<nearest<<'}';}
    report<<"]}\n";
   }
  }rclcpp::shutdown();return 0;
 }

 if(mode=="split-public-gate-check"){
  // An all-fixed static safe path isolates the final-validator constraint gate.
  node.path_={request.q_start,request.q_start};request.q_goal=request.q_start;
  chito_planner::RefinementResult candidate;candidate.trajectory=node.path_;candidate.success=false;candidate.status="corridor_constraint_failed";
  auto validated=chito_planner::detail::validate_original_trajectory(request,generation,candidate,options);
  if(validated.success||validated.status!="corridor_constraint_failed")return 61;
  std::cout<<"SPLIT_PUBLIC_GATE_CHECK geometry revalidation preserves corridor failure"<<std::endl;rclcpp::shutdown();return 0;
 }
 if(mode=="split-check"){
  const auto before=node.path_;node.split_corridor_nodes_=true;
  if(node.split_copy_corridor_safe_())return 55;
  const int oldlimit=node.repair_waypoint_limit_;node.repair_waypoint_limit_=5;
  if(node.run_split_corridor_pass_()||node.split_corridor_initialized_||node.path_.size()!=before.size())return 56;
  for(size_t j=0;j<before.size();++j)if((node.path_[j]-before[j]).norm()!=0.)return 57;
  node.repair_waypoint_limit_=oldlimit;

  for(size_t id=0;id<node.guide_.poly_A.size();++id){node.guide_.poly_A[id].resize(6,3);node.guide_.poly_A[id]<<1,0,0,-1,0,0,0,1,0,0,-1,0,0,0,1,0,0,-1;node.guide_.poly_b[id]=Eigen::VectorXd::Constant(6,2.);}
  if(!node.initialize_split_corridor_()||node.path_.size()!=8||node.split_copy_roles_.size()!=8||node.split_source_nodes_!=std::vector<int>({0,1,1,2,2,3,3,4}))return 50;
  if((node.path_.front()-before.front()).norm()>1e-12||(node.path_.back()-before.back()).norm()>1e-12)return 51;
  double maxgap=0;for(int j:{1,3,5})maxgap=std::max(maxgap,(node.path_[j]-node.path_[j+1]).norm());if(maxgap<1e-8)return 52;
  const auto b=node.guide_.poly_b;for(auto& value:node.guide_.poly_b)value.setConstant(-100.);if(node.split_copy_corridor_safe_())return 53;node.guide_.poly_b=b;
  node.reset_global_state_on_new_package_();node.path_=before;if(node.split_corridor_initialized_||!node.split_copy_roles_.empty()||!node.split_source_nodes_.empty()||!node.prepare_corridor_provenance_())return 54;
  std::cout<<"SPLIT_CHECK real arm, synthetic broad cells; side copies diverged; endpoints and new-task reset preserved; max_gap="<<maxgap<<std::endl;rclcpp::shutdown();return 0;
 }
 if(mode=="corridor-reset-check"){
  if(!node.prepare_corridor_provenance_())return 39;
  const auto original_path=node.path_;const auto original_sigma=node.guide_.sigma_dense;
  if(!chito_planner::core::insert_provenance(node.corridor_provenance_,0,1))return 44;
  node.path_.insert(node.path_.begin()+1,.5*(node.path_[0]+node.path_[1]));
  if(!node.sync_corridor_cells_())return 44;
  node.reset_global_state_on_new_package_();node.path_=original_path;
  if(node.guide_.sigma_dense!=original_sigma||!node.prepare_corridor_provenance_())return 45;
  node.reset_global_state_on_new_package_();
  if(!node.corridor_provenance_.empty()||!node.corridor_original_sigma_.empty())return 40;
  if(!node.prepare_corridor_provenance_())return 41;
  if(!node.load_convexset_guide_from_yaml_((input/"guide.yaml").string()))return 42;
  if(!node.corridor_provenance_.empty()||!node.corridor_original_sigma_.empty())return 43;
  std::cout<<"CORRIDOR_RESET_CHECK provenance cleared for new package and reloaded guide"<<std::endl;rclcpp::shutdown();return 0;
 }
 if(mode=="corridor-check"){
  auto before=node.path_;if(!node.prepare_corridor_provenance_())return 30;
  // A real-arm fixture with deliberately broad cells tests endpoint release;
  // this synthetic cell replacement is not used in paired experiments.
  for(size_t id=0;id<node.guide_.poly_A.size();++id){node.guide_.poly_A[id].resize(6,3);node.guide_.poly_A[id]<<1,0,0,-1,0,0,0,1,0,0,-1,0,0,0,1,0,0,-1;node.guide_.poly_b[id]=Eigen::VectorXd::Constant(6,2.);}
  bool changed=node.repair_corridor_endpoints_(2);
  double moved=(node.path_[1]-before[1]).norm()+(node.path_[3]-before[3]).norm();
  if(!changed||(node.path_[1]-before[1]).norm()<1e-8||(node.path_[3]-before[3]).norm()<1e-8||(node.path_.front()-before.front()).norm()>1e-12||(node.path_.back()-before.back()).norm()>1e-12)return 31;
  for(int j:{1,3})if(chito_planner::core::corridor_residual(node.guide_.poly_A[node.guide_.sigma_dense[j]],node.guide_.poly_b[node.guide_.sigma_dense[j]],node.get_tcp_pose_world_(node.path_[j]).translation())>1e-6)return 32;
  std::cout<<"CORRIDOR_CHECK real-arm synthetic-cell fixture released both local endpoint variables; displacement_sum="<<moved<<std::endl;
  node.path_=before;
  for(auto& b:node.guide_.poly_b)b.setConstant(-100.); // Contradictory opposite planes.
  if(node.repair_corridor_endpoints_(2))return 35;
  for(size_t j=0;j<before.size();++j)if((before[j]-node.path_[j]).norm()!=0.)return 36;
  std::cout<<"CORRIDOR_CHECK infeasible endpoint cells rejected with path unchanged"<<std::endl;
  node.path_.front()=before[3];auto invalid=node.path_;if(node.run_corridor_endpoint_pass_())return 33;
  for(size_t j=0;j<invalid.size();++j)if((invalid[j]-node.path_[j]).norm()!=0.)return 34;
  std::cout<<"CORRIDOR_CHECK colliding true start rejected with path unchanged"<<std::endl;rclcpp::shutdown();return 0;
 }
 if(mode=="window-check"){
  auto before=node.path_;std::vector<double> d;for(const auto& q:before)d.push_back(node.min_true_distance_path_({q}));
  auto w=chito_planner::core::collision_free_window(d,2);if(!w.valid||!(w.left<3&&3<w.right)||d[3]>=0.)return 21;
  (void)node.repair_variable_window_(node.path_,2);
  if((node.path_.front()-before.front()).norm()>1e-12||(node.path_.back()-before.back()).norm()>1e-12)return 22;
  std::cout<<"WINDOW_CHECK released S3; displacement="<<(node.path_[3]-before[3]).norm()<<" before="<<d[3]<<" after="<<node.min_true_distance_path_({node.path_[3]})<<std::endl;
  node.path_.front()=before[3];auto invalid=node.path_;if(node.run_free_anchor_pass_())return 23;
  for(size_t i=0;i<invalid.size();++i)if((invalid[i]-node.path_[i]).norm()!=0.)return 24;
  std::cout<<"WINDOW_CHECK invalid global endpoint preserved; local repair rejected"<<std::endl;rclcpp::shutdown();return 0;
 }
 if(mode=="reject-check"){
   if(!fixed)return 11;
   if(node.global_candidate_acceptable_(1.,-1.)||node.global_candidate_acceptable_(-1.,1.)||node.global_candidate_acceptable_(NAN,1.)||!node.global_candidate_acceptable_(1.,1.))return 12;
   auto before=node.path_;node.mi_thresh_=1e100;node.cont_after_global_only_=true;
   node.one_iter_step();
   for(size_t i=0;i<before.size();++i)if((before[i]-node.path_[i]).norm()!=0.)return 13;
   if(node.cur_iter_!=1)return 14;
   std::cout<<"REJECT_CHECK unchanged path and bounded iteration passed"<<std::endl;rclcpp::shutdown();return 0;
 }
 if(mode=="initial"){std::ofstream view(out/"frames.json");node.export_collision_view(view);rclcpp::shutdown();return 0;}
 if(mode=="scaled-gradient"){node.alpha_*=1e7;node.mu_*=1e7;node.lambda_corr_*=1e7;node.lambda_poly_membership_*=1e7;}
 if(const char* component=std::getenv("CHITO_COMPONENT")){
   const std::string part=component;
   if(part!="smooth")node.alpha_=0.;if(part!="collision")node.mu_=0.;if(part!="anchor")node.lambda_corr_=0.;if(part!="poly")node.lambda_poly_membership_=0.;
 }
 const double fd_step=std::getenv("CHITO_FD_STEP")?std::stod(std::getenv("CHITO_FD_STEP")):1e-7;
 Eigen::MatrixXd Q;Eigen::VectorXd c;node.build_Q_c_oldstyle_from(node.path_,Q,c);auto old=node.path_;Eigen::VectorXd x(6*old.size());for(size_t i=0;i<old.size();++i)x.segment<6>(6*i)=old[i];
 const auto model_gradient=(Q*x+c).eval();Eigen::VectorXd numerical=model_gradient;
 for(size_t t=1;t+1<old.size();++t)for(int k=0;k<6;++k){auto a=old,b=old;a[t][k]+=fd_step;b[t][k]-=fd_step;numerical[6*t+k]=(node.global_actual_merit_(a)-node.global_actual_merit_(b))/(2*fd_step);}
 const int free_count=6*(old.size()-2);
 double relative=(numerical-model_gradient).segment(6,free_count).norm()/std::max(1e-12,numerical.segment(6,free_count).norm());
 std::ofstream audit(out/"gradient.json");audit<<std::setprecision(17)<<"{\"relative_error\":"<<relative<<",\"model_gradient\":[";for(int k=0;k<x.size();++k){if(k)audit<<',';audit<<model_gradient[k];}audit<<"],\"numerical_gradient\":[";for(int k=0;k<x.size();++k){if(k)audit<<',';audit<<numerical[k];}audit<<"]}";audit.close();
 std::cout<<"GRADIENT "<<std::setprecision(17)<<relative<<std::endl;
 if(mode=="gradient"||mode=="scaled-gradient"){rclcpp::shutdown();return relative<1e-3?0:10;}
 // Save the actual final global iterate before any local repairs.
 if(mode=="global"){
   node.cont_after_global_only_=true;node.global_capture_only_=true;
   node.run_to_completion_collect_metrics_(false);
   save_path(out/"trajectory.csv",node.path_);std::ofstream view(out/"frames.json");node.export_collision_view(view);rclcpp::shutdown();return 0;
 }
 auto result=node.refine(request);save_path(out/"trajectory.csv",result.trajectory);
 if(node.split_corridor_initialized_){node.log_split_connectors_("final");std::ofstream meta(out/"split-metadata.json");meta<<"{\"copy_roles\":[";for(size_t j=0;j<node.split_copy_roles_.size();++j){if(j)meta<<',';meta<<node.split_copy_roles_[j];}meta<<"],\"source_nodes\":[";for(size_t j=0;j<node.split_source_nodes_.size();++j){if(j)meta<<',';meta<<node.split_source_nodes_[j];}meta<<"],\"provenance\":[";for(size_t j=0;j<node.corridor_provenance_.size();++j){if(j)meta<<',';meta<<node.corridor_provenance_[j];}meta<<"],\"copy_corridor_safe\":"<<int(node.split_copy_corridor_safe_())<<"}";}
 std::ofstream result_file(out/"result.json");result_file<<std::setprecision(17)<<"{\"success\":"<<int(result.success)<<",\"waypoint_safe\":"<<int(result.waypoint_safe)<<",\"continuous_safe\":"<<int(result.continuous_safe)<<",\"nodes\":"<<result.trajectory.size()<<",\"minimum_distance\":"<<result.minimum_distance<<",\"optimization_ms\":"<<result.optimization_ms<<",\"validation_ms\":"<<result.validation_ms<<",\"joint_length\":"<<result.normalized_length<<",\"global_steps\":"<<node.cur_iter_<<"}";result_file.close();
 std::ofstream view(out/"frames.json");node.export_collision_view(view);
 if((result.trajectory.front()-request.q_start).norm()>1e-9||(result.trajectory.back()-request.q_goal).norm()>1e-9)return 5;
 rclcpp::shutdown();
}
