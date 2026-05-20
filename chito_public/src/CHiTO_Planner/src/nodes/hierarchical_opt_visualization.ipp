  void append_box_as_triangles(const LinkBox& lb, Marker& tri) const {
    const double hx=lb.size.x()*0.5, hy=lb.size.y()*0.5, hz=lb.size.z()*0.5;
    Eigen::Vector3d v[8]={{+hx,+hy,+hz},{+hx,+hy,-hz},{+hx,-hy,-hz},{+hx,-hy,+hz},
                          {-hx,+hy,+hz},{-hx,+hy,-hz},{-hx,-hy,-hz},{-hx,-hy,+hz}};
    auto toPoint=[&](const Eigen::Vector3d& p){ Eigen::Vector3d pw=lb.T*p; geometry_msgs::msg::Point P; P.x=pw.x(); P.y=pw.y(); P.z=pw.z(); return P; };
    int f[12][3]={{0,1,2},{0,2,3},{4,6,5},{4,7,6},{0,4,5},{0,5,1},{3,2,6},{3,6,7},{0,3,7},{0,7,4},{1,5,6},{1,6,2}};
    for (auto& tri_idx:f){ tri.points.push_back(toPoint(v[tri_idx[0]])); tri.points.push_back(toPoint(v[tri_idx[1]])); tri.points.push_back(toPoint(v[tri_idx[2]])); }
  }

  void append_obstacles_markers(MarkerArray& arr, const std_msgs::msg::Header& header) const {
  int base=90000000, id=0;
  for (const auto& ob : obstacles_) {
    Marker m; 
    m.header = header; 
    m.ns     = "obstacle"; 
    m.id     = base+id++; 
    m.action = Marker::ADD;

    // === 从 FCL 对象里取世界变换：统一转成 Eigen::Isometry3d ===
    Eigen::Isometry3d Tw = Eigen::Isometry3d::Identity();
    {
      const auto& Tf = ob.obj->getTransform();      // fcl::Transform3d
      Tw.matrix() = Tf.matrix();
    }

    Eigen::Vector3d p = Tw.translation();
    Eigen::Quaterniond q(Tw.linear());

    m.pose.position.x = p.x();
    m.pose.position.y = p.y();
    m.pose.position.z = p.z();
    m.pose.orientation.x = q.x();
    m.pose.orientation.y = q.y();
    m.pose.orientation.z = q.z();
    m.pose.orientation.w = q.w();

    if (ob.type==ObType::BOX) {
      m.type = Marker::CUBE;
      m.scale.x = ob.size.x();
      m.scale.y = ob.size.y();
      m.scale.z = ob.size.z();
      m.color.a = 0.6f; m.color.r=1.0f; m.color.g=0.3f; m.color.b=0.3f;
    } 
    else if (ob.type==ObType::SPHERE) {
      m.type = Marker::SPHERE;
      m.scale.x = ob.size.x();
      m.scale.y = ob.size.y();
      m.scale.z = ob.size.z();
      m.color.a = 0.6f; m.color.r=0.2f; m.color.g=0.6f; m.color.b=1.0f;
    } 
    else { // CYLINDER
      m.type = Marker::CYLINDER;
      m.scale.x = ob.size.x();   // 直径
      m.scale.y = ob.size.y();
      m.scale.z = ob.size.z();   // 高度
      m.color.a = 0.6f; m.color.r=0.9f; m.color.g=0.7f; m.color.b=0.2f;
    }

    arr.markers.push_back(m);
  }
}


  inline int  base_id_for_iter(int iter) const { return iter*10000; }
  inline void color_from_ratio(double r, float& R,float& G,float& B) const { r=std::clamp(r,0.0,1.0); R=float(1.0-r); G=float(r); B=0.2f; }
  inline std::string ns_iter(int iter, const std::string& tag) const { char buf[32]; std::snprintf(buf,sizeof(buf),"iter_%03d",iter); return std::string(buf)+"/"+tag; }

  void publish_markers_for_iter(const std::vector<Eigen::Matrix<double,6,1>>& path_iter,
                              int iter, int& id_cursor) const {
  // 先准备 header/arr
  MarkerArray arr;
  std_msgs::msg::Header header;
  header.frame_id = frame_id_;
  header.stamp = this->now();  // const 成员内可用
    // ... 你现在的 header / base cube 等代码在前面 ...

  // === NEW: 打印当前这一轮可视化使用的路径关节角（deg） ===
  if (debug_print_q_ && !path_iter.empty()) {
    std::ostringstream oss;
    oss << "[VIZ-q] iter=" << iter
        << " | steps=" << path_iter.size()
        << " | q (deg):\n";

    for (size_t t = 0; t < path_iter.size(); ++t) {
      oss << "  t=" << t << " : [";
      for (int k = 0; k < 6; ++k) {
        double q_deg = path_iter[t](k) * 180.0 / M_PI;
        oss << std::fixed << std::setprecision(3) << q_deg;
        if (k < 5) oss << ", ";
      }
      oss << "]\n";
    }

    RCLCPP_INFO(get_logger(), "%s", oss.str().c_str());

    // 如果你只想打印一次，可以在外面开个 debug_print_q_once_ 开关：
    // debug_print_q_ = !debug_print_q_once_;
  }

  // === 从这里开始，保持你原来的逻辑 ===
  double ratio = (max_iters_>1)? std::min(1.0, std::max(0.0, double(iter)/double(std::max(1,max_iters_)))):0.0;
  float R,G,B; color_from_ratio(ratio,R,G,B);

  // 末端轨迹 / 连杆TRIANGLES / 违规点 / 扫掠壳 / 焊枪 / 障碍物 ...
  // （你现有的代码保持不变）

    // 末端轨迹
    {
      Marker m; m.header=header; m.id=id_cursor++; m.ns=ns_iter(iter,"ee_path"); m.type=Marker::LINE_STRIP; m.action=Marker::ADD;
      m.scale.x=0.01; m.color.a=1.0; m.color.r=R; m.color.g=G; m.color.b=B;
      for (auto& q : path_iter) { std::array<Eigen::Isometry3d,7> Tj; compute_fk_chain_(q,Tj);
        auto p=Tj[6].translation(); geometry_msgs::msg::Point P; P.x=p.x(); P.y=p.y(); P.z=p.z(); m.points.push_back(P); }
      arr.markers.push_back(m);
    }
    // === Start / Goal TCP markers ===
    if (!path_iter.empty()) {
      // 计算 start TCP
      std::array<Eigen::Isometry3d,7> T_start, T_goal;
      compute_fk_chain_(path_iter.front(), T_start);
      compute_fk_chain_(path_iter.back(),  T_goal);

      Eigen::Vector3d p_start = T_start[6].translation();
      Eigen::Vector3d p_goal  = T_goal[6].translation();

      // Start: 绿色球
      {
        Marker m;
        m.header = header;
        m.ns     = ns_iter(iter, "start_tcp");
        m.id     = id_cursor++;
        m.type   = Marker::SPHERE;
        m.action = Marker::ADD;
        m.pose.position.x = p_start.x();
        m.pose.position.y = p_start.y();
        m.pose.position.z = p_start.z();
        m.pose.orientation.w = 1.0;
        m.scale.x = m.scale.y = m.scale.z = 0.04;   // 球的大小
        m.color.a = 1.0;
        m.color.r = 0.1f;
        m.color.g = 0.9f;   // 绿色
        m.color.b = 0.1f;
        arr.markers.push_back(m);
      }

      // Goal: 蓝色球
      {
        Marker m;
        m.header = header;
        m.ns     = ns_iter(iter, "goal_tcp");
        m.id     = id_cursor++;
        m.type   = Marker::SPHERE;
        m.action = Marker::ADD;
        m.pose.position.x = p_goal.x();
        m.pose.position.y = p_goal.y();
        m.pose.position.z = p_goal.z();
        m.pose.orientation.w = 1.0;
        m.scale.x = m.scale.y = m.scale.z = 0.04;
        m.color.a = 1.0;
        m.color.r = 0.1f;
        m.color.g = 0.1f;
        m.color.b = 0.9f;   // 蓝色
        arr.markers.push_back(m);
      }
    }

    // 合并 link 盒三角
    {
      Marker tri; tri.header=header; tri.id=id_cursor++; tri.ns=ns_iter(iter,"links_all"); tri.type=Marker::TRIANGLE_LIST; tri.action=Marker::ADD;
      tri.scale.x=tri.scale.y=tri.scale.z=1.0; tri.color.a=0.45f; tri.color.r=R; tri.color.g=G; tri.color.b=B;
      for (const auto& q : path_iter) {
        auto boxes = build_link_boxes_for_q(q);
        for (const auto& lb : boxes) append_box_as_triangles(lb, tri);
      }
      arr.markers.push_back(tri);
    }

    // 违规最近点/法向线
    auto vios = collect_violations(path_iter);
    for (const auto& vio : vios) {
      Marker pr; pr.header=header; pr.ns=ns_iter(iter,"near_p_robot"); pr.id=id_cursor++; pr.type=Marker::SPHERE; pr.action=Marker::ADD;
      pr.scale.x=pr.scale.y=pr.scale.z=0.03; pr.color.a=1.0; pr.color.r=1.0; pr.color.g=0.0; pr.color.b=0.0;
      pr.pose.position.x=vio.p_robot.x(); pr.pose.position.y=vio.p_robot.y(); pr.pose.position.z=vio.p_robot.z(); pr.pose.orientation.w=1.0;
      arr.markers.push_back(pr);

      Marker po=pr; po.ns=ns_iter(iter,"near_p_obs"); po.id=id_cursor++; po.color.r=0.0; po.color.g=0.2; po.color.b=1.0;
      po.pose.position.x=vio.p_obs.x(); po.pose.position.y=vio.p_obs.y(); po.pose.position.z=vio.p_obs.z();
      arr.markers.push_back(po);

      Marker ln; ln.header=header; ln.ns=ns_iter(iter,"near_normal"); ln.id=id_cursor++; ln.type=Marker::LINE_LIST; ln.action=Marker::ADD;
      ln.scale.x=0.01; ln.color.a=1.0; ln.color.r=0.0; ln.color.g=1.0; ln.color.b=0.0;
      geometry_msgs::msg::Point A,Bp; A.x=vio.p_robot.x();A.y=vio.p_robot.y();A.z=vio.p_robot.z(); Bp.x=vio.p_obs.x();Bp.y=vio.p_obs.y();Bp.z=vio.p_obs.z();
      ln.points.push_back(A); ln.points.push_back(Bp); arr.markers.push_back(ln);
    }

    // 连续安全：不安全边的扫掠 hull
    if (!unsafe_edges_cache_.empty()) {
      for (const auto& e : unsafe_edges_cache_) {
        const auto& qA=e.first; const auto& qB=e.second;
        Marker tri; tri.header=header; tri.id=id_cursor++; tri.ns=ns_iter(iter,"swept_unsafe"); tri.type=Marker::TRIANGLE_LIST; tri.action=Marker::ADD;
        tri.scale.x=tri.scale.y=tri.scale.z=1.0; tri.color.a=0.35f; tri.color.r=0.1f; tri.color.g=0.9f; tri.color.b=0.9f;
        auto boxA=build_link_boxes_for_q(qA), boxB=build_link_boxes_for_q(qB);
       
  const int K_all = static_cast<int>(boxA.size());  // 6 或 8
  for (int k = 0; k < K_all; ++k) {
    auto hull = make_link_swept_hull_data_(boxA[k], boxB[k]);
    append_hull_triangles_to_marker_(hull, tri);
  }

        arr.markers.push_back(tri);

        Marker edge; edge.header=header; edge.id=id_cursor++; edge.ns=ns_iter(iter,"swept_edges"); edge.type=Marker::LINE_LIST; edge.action=Marker::ADD;
        edge.scale.x=0.005; edge.color.a=1.0; edge.color.r=1.0; edge.color.g=0.2; edge.color.b=0.2;
        std::array<Eigen::Isometry3d,7> TA,TB; compute_fk_chain_(qA,TA); compute_fk_chain_(qB,TB);
        auto pA=TA[6].translation(), pB=TB[6].translation();
        geometry_msgs::msg::Point PA,PB; PA.x=pA.x();PA.y=pA.y();PA.z=pA.z(); PB.x=pB.x();PB.y=pB.y();PB.z=pB.z();
        edge.points.push_back(PA); edge.points.push_back(PB);
        arr.markers.push_back(edge);
      }
    }

    // 焊枪可视化
    if (!gripper_enable_ && torch_enable_) {
      auto emit=[&](const Eigen::Matrix<double,6,1>& q, int& idc, const std::string& ns){
        Eigen::Isometry3d T_world = get_tcp_pose_world_(q);
        Eigen::Vector3d p_tcp=T_world.translation(); Eigen::Matrix3d R0=T_world.linear();

        auto makeCyl=[&](const Eigen::Vector3d& center, const Eigen::Vector3d& dir, double len, double dia,
                         double r,double g,double b,double a, const std::string& _ns, int& id)->Marker{
          Marker m; m.header=header; m.ns=_ns; m.id=id++; m.type=Marker::CYLINDER; m.action=Marker::ADD;
          m.scale.x=dia; m.scale.y=dia; m.scale.z=len; m.color.r=r; m.color.g=g; m.color.b=b; m.color.a=a;
          Eigen::Vector3d d=dir; if (d.norm()<1e-12) d=Eigen::Vector3d::UnitZ();
          Eigen::Quaterniond qz = Eigen::Quaterniond::FromTwoVectors(Eigen::Vector3d::UnitZ(), d.normalized());
          m.pose.position.x=center.x(); m.pose.position.y=center.y(); m.pose.position.z=center.z();
          m.pose.orientation.x=qz.x(); m.pose.orientation.y=qz.y(); m.pose.orientation.z=qz.z(); m.pose.orientation.w=qz.w();
          return m;
        };

        Eigen::Vector3d dir1 = R0*Eigen::Vector3d::UnitZ();
        Eigen::Vector3d c1 = p_tcp + 0.5*torch_cyl1_len_*dir1;
        arr.markers.push_back(makeCyl(c1,dir1,torch_cyl1_len_,torch_cyl1_dia_,
                                      torch1_rgba_[0],torch1_rgba_[1],torch1_rgba_[2],torch1_rgba_[3],
                                      ns+"_torch1", idc));
        const double tilt=-torch_tilt_deg_*M_PI/180.0;
        Eigen::Matrix3d Rtilt = Eigen::AngleAxisd(tilt,Eigen::Vector3d::UnitX()).toRotationMatrix();
        Eigen::Vector3d dir2 = R0*(Rtilt*Eigen::Vector3d::UnitZ());
        Eigen::Vector3d p2 = p_tcp + torch_cyl1_len_*dir1;
        Eigen::Vector3d c2 = p2 + 0.5*torch_cyl2_len_*dir2;
        arr.markers.push_back(makeCyl(c2,dir2,torch_cyl2_len_,torch_cyl2_dia_,
                                      torch2_rgba_[0],torch2_rgba_[1],torch2_rgba_[2],torch2_rgba_[3],
                                      ns+"_torch2", idc));
      };
      if (viz_torch_only_last_) { if (!path_iter.empty()) emit(path_iter.back(), id_cursor, ns_iter(iter,"torch")); }
      else { for (const auto& q: path_iter) emit(q, id_cursor, ns_iter(iter,"torch")); }
    }

    append_obstacles_markers(arr, header);
    
        // ====== 发布前统计内容（可选）======
    size_t tri_points = 0, line_points = 0, spheres = 0, cubes = 0, cylinders = 0, textn = 0, totals = arr.markers.size();
    if (debug_dump_sizes_) {
      for (const auto& m : arr.markers) {
        switch (m.type) {
          case Marker::TRIANGLE_LIST: tri_points += m.points.size(); break;
          case Marker::LINE_LIST:
          case Marker::LINE_STRIP:    line_points += m.points.size(); break;
          case Marker::SPHERE:        spheres++;   break;
          case Marker::CUBE:          cubes++;     break;
          case Marker::CYLINDER:      cylinders++; break;
          case Marker::TEXT_VIEW_FACING: textn++;  break;
          default: break;
        }
      }
    }

    // ====== 正式发布 ======
    pub_->publish(arr);

    // ====== 发布后验证 ======
    if (debug_verify_viz_) {
      // 1) 打印这次实际发送的marker数量/点数（证明消息不是空的）
      RCLCPP_INFO(get_logger(),
        "[VIZ] iter=%d published %zu markers | TRI pts=%zu | LINE pts=%zu | S=%zu C=%zu CYL=%zu TEXT=%zu",
        iter, totals, tri_points, line_points, spheres, cubes, cylinders, textn);

      // 2) 打印当前订阅数量（RViz是否连上）
      size_t cur = pub_->get_subscription_count();
      if (cur != last_sub_count_) {
        if (debug_log_sub_count_) {
          RCLCPP_INFO(get_logger(), "[SUB] /qp_opt/markers subscriptions: %zu", cur);
        }
        last_sub_count_ = cur;
      }

      // 3) 若订阅为0，发出一次性告警（帮助定位话题/QoS/RViz配置问题）
      if (cur == 0) {
        RCLCPP_WARN_THROTTLE(get_logger(), *clock_, 3000,
  "[VIZ] No subscribers on /qp_opt/markers. "
  "Check RViz 'Add->By display type: MarkerArray' topic='/qp_opt/markers' "
  "and Fixed Frame='%s' (qos_transient_local=%s).",
  frame_id_.c_str(), qos_transient_local_ ? "true" : "false");


      }
    }

  }

  // ---------------- 结束/统计 ----------------
void finalize_and_maybe_stop_(const std::string& reason) {
  if (cont_after_global_only_) {
    int idc = base_id_for_iter(cur_iter_);
    for (int pass = 0; pass < 5; ++pass) {
      bool changed = run_continuity_pass_once_();  // use_continuity_check_log_only_=false 时会插点
      publish_markers_for_iter(path_, cur_iter_, idc);
      if (!changed) break;
    }
  } else {
    int idc = base_id_for_iter(cur_iter_);
    publish_markers_for_iter(path_, cur_iter_, idc);
  }
// ===== optional final global smoothing =====
if (enable_final_global_smooth_) {
  run_final_global_smooth_();

  // optional: re-check continuity once after final smoothing
  if (cont_after_global_only_) {
    (void)run_continuity_pass_once_();
  }

  int idc = base_id_for_iter(cur_iter_);
  publish_markers_for_iter(path_, cur_iter_, idc);
}
  const auto elapsed = std::chrono::duration<double,std::milli>(Clock::now()-plan_start_tp_).count();
  const double L = path_length_normed(path_), md = min_true_distance(path_);
  RCLCPP_INFO(get_logger(), "[PLAN DONE] reason=%s | iters=%d | len=%.6f | min_d=%.6f | time=%.3fms",
              reason.c_str(), cur_iter_, L, md, elapsed);

  // ======= 在这里打印最终成功路径（只打印一次）=======
  // 你也可以换成更严谨的 success 判定（比如 min_d >= stop_min_d_ 或 violations empty）
  bool success = false;

// 1) 用 min_d 判
if (stop_when_min_d_ge_) success = (md >= stop_min_d_);
else                     success = (md <= stop_min_d_);
  if (success) {
    dump_final_path_(get_logger(), path_, "FINAL-PATH");
    save_path_csv_(get_logger(), path_, "/home/hyj/iris_rviz_ws/src/iris_rviz_cpp/src/tmp/final_path.csv");   // 你也可以换成你项目目录
  } else {
    RCLCPP_INFO(get_logger(), "[FINAL-PATH] not printed because not marked success (reason=%s)", reason.c_str());
  }
  // ===============================================

  if (stop_when_safe_) timer_->cancel();
}


private:
// —— 将“本轮/本段”会被复用的局部状态全部清零 ——
// 注意：这是“局部QP与连续性修补”的运行时状态，
// 在 private: 增加

