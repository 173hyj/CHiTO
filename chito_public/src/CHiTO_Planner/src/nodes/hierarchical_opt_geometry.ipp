 // ======= DH kinematics (pure, unified) =======
struct DH { double a, d, alpha, theta_off; }; // 标准DH: RotZ(theta)*TransZ(d)*TransX(a)*RotX(alpha)

// 默认 UR5（请与你的 URDF 对齐；必要时调整 a/d/alpha 符号或 theta_off）
std::array<DH,6> dh_{{
  { 0.000000, 0.089159,  +M_PI/2.0, 0.0 },  // J1
  {-0.425000, 0.000000,   0.0,       0.0 },  // J2
  {-0.392250, 0.000000,   0.0,       0.0 },  // J3
  { 0.000000, 0.109150,  +M_PI/2.0,  0.0 },  // J4
  { 0.000000, 0.094650,  -M_PI/2.0,  0.0 },  // J5
  { 0.000000, 0.082300,   0.0,       0.0 }   // J6（法兰前）
}};

// 统一只用世界->基座、法兰->TCP
Eigen::Isometry3d T_world_base_  = Eigen::Isometry3d::Identity();
Eigen::Isometry3d T_flange_tcp_  = Eigen::Isometry3d::Identity();

// 统一版 T_dh（参数顺序固定为 a, d, alpha, theta）
static inline Eigen::Isometry3d T_dh(double a, double d, double alpha, double theta)
{
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  const double ca = std::cos(alpha), sa = std::sin(alpha);
  const double ct = std::cos(theta), st = std::sin(theta);

  // 旋转部分：已经是标准 DH 的 R = Rz(theta)*Rx(alpha) 对应的 3x3
  T.linear() <<
      ct, -st*ca,  st*sa,
      st,  ct*ca, -ct*sa,
       0,     sa,     ca;

  // **关键修正：平移用 (a*ct, a*st, d)，而不是 (a,0,d)**
  T.translation() = Eigen::Vector3d(a * ct, a * st, d);

  return T;
}


// 唯一正解：Tj[0]=base系；Tj[6]=TCP（已乘法兰->TCP）
void compute_fk_chain_(const Eigen::Matrix<double,6,1>& q,
                       std::array<Eigen::Isometry3d,7>& Tj) const
{
  Eigen::Isometry3d T = T_world_base_;
  Tj[0] = T;                                   // world->base
  for (int i=0; i<6; ++i) {
    const double th = q(i) + dh_[i].theta_off;
    T = T * T_dh(dh_[i].a, dh_[i].d, dh_[i].alpha, th);
    Tj[i+1] = T;                               // world->link(i)末
  }
  // 末尾乘 flange->tcp，使 Tj[6] 直接是 world->tcp
  Tj[6] = Tj[6] * T_flange_tcp_;
}


// DH 版点位置雅可比（世界系）
Eigen::Matrix<double,3,6>
jacobian_point_world_dh_(const Eigen::Matrix<double,6,1>& q,
                         int link_seg_index,
                         const Eigen::Vector3d& p_world) const
{
  std::array<Eigen::Isometry3d,7> Tj;
  compute_fk_chain_(q, Tj);

  Eigen::Matrix<double,3,6> J; J.setZero();
  // 关节 j 的轴/原点：以“关节前”的坐标系为基准（即 Tj[j] 之前的那一帧）
  for (int j = 0; j < 6; ++j) {
    const Eigen::Vector3d o_j = Tj[j].translation();
    const Eigen::Vector3d z_j = Tj[j].linear() * Eigen::Vector3d::UnitZ();
    J.col(j) = z_j.cross(p_world - o_j);
  }
  // 保留到当前段（清零远端列）
  for (int j = link_seg_index + 1; j < 6; ++j) J.col(j).setZero();
  return J;
}

// 供现有代码调用的包装：现阶段用 DH 兜底
Eigen::Matrix<double,3,6>
compute_position_jacobian_moveit_base_(const Eigen::Matrix<double,6,1>& q,
                                       int link_seg_index,
                                       const Eigen::Vector3d& p_world) const
{
  // 未来如果 robot_state_ 就绪，可切换到 MoveIt 真·Jacobian：
  // if (robot_state_ && jmg_) { ... return J_from_moveit; }
  return jacobian_point_world_dh_(q, link_seg_index, p_world);
}
Eigen::Isometry3d get_tcp_pose_world_(const Eigen::Matrix<double,6,1>& q) const
{
  std::array<Eigen::Isometry3d,7> Tj;
  compute_fk_chain_(q, Tj);
  return Tj[6];
}
Eigen::Isometry3d make_gripper_mount_pose_(const Eigen::Isometry3d& T_world_tcp) const
{
  const double rr = gripper_mount_roll_deg_  * M_PI / 180.0;
  const double rp = gripper_mount_pitch_deg_ * M_PI / 180.0;
  const double ry = gripper_mount_yaw_deg_   * M_PI / 180.0;

  Eigen::Isometry3d T_mount = Eigen::Isometry3d::Identity();
  T_mount.linear() =
      (Eigen::AngleAxisd(ry, Eigen::Vector3d::UnitZ()) *
       Eigen::AngleAxisd(rp, Eigen::Vector3d::UnitY()) *
       Eigen::AngleAxisd(rr, Eigen::Vector3d::UnitX())).toRotationMatrix();

  T_mount.translation() =
      Eigen::Vector3d(gripper_mount_x_, gripper_mount_y_, gripper_mount_z_);

  return T_world_tcp * T_mount;
}
std::vector<LinkBox>
build_gripper_boxes_for_tcp_(const Eigen::Isometry3d& T_world_tcp) const
{
  std::vector<LinkBox> out;
  out.reserve(3);

  const Eigen::Isometry3d T_g = make_gripper_mount_pose_(T_world_tcp);

  const Eigen::Matrix3d R = T_g.linear();
  const Eigen::Vector3d p = T_g.translation();

  const Eigen::Vector3d ey = R.col(1);
  const Eigen::Vector3d ez = R.col(2);

  // ===== 1) palm =====
  Eigen::Isometry3d T_palm = Eigen::Isometry3d::Identity();
  T_palm.linear() = R;
  T_palm.translation() = p + 0.5 * gripper_palm_len_ * ez;

  out.push_back(make_box_from_pose(
      T_palm,
      gripper_palm_thick_,
      gripper_palm_width_,
      gripper_palm_len_
  ));

  // ===== 2) left finger =====
  Eigen::Isometry3d T_l = Eigen::Isometry3d::Identity();
  T_l.linear() = R;
  T_l.translation() =
      p
      + (gripper_palm_len_ + 0.5 * gripper_finger_len_) * ez
      + 0.5 * gripper_opening_ * ey;

  out.push_back(make_box_from_pose(
      T_l,
      gripper_finger_thick_,
      gripper_finger_width_,
      gripper_finger_len_
  ));

  // ===== 3) right finger =====
  Eigen::Isometry3d T_r = Eigen::Isometry3d::Identity();
  T_r.linear() = R;
  T_r.translation() =
      p
      + (gripper_palm_len_ + 0.5 * gripper_finger_len_) * ez
      - 0.5 * gripper_opening_ * ey;

  out.push_back(make_box_from_pose(
      T_r,
      gripper_finger_thick_,
      gripper_finger_width_,
      gripper_finger_len_
  ));

  return out;
}

    // ---------------- Link boxes ----------------
  std::array<Eigen::Vector3d,8> box_world_corners_(const LinkBox& lb) const {
    const double hx = lb.size.x()*0.5, hy = lb.size.y()*0.5, hz = lb.size.z()*0.5;
    Eigen::Vector3d v[8] = {
      {+hx,+hy,+hz}, {+hx,+hy,-hz}, {+hx,-hy,-hz}, {+hx,-hy,+hz},
      {-hx,+hy,+hz}, {-hx,+hy,-hz}, {-hx,-hy,-hz}, {-hx,-hy,+hz}
    };
    std::array<Eigen::Vector3d,8> out;
    for (int i=0;i<8;++i) out[i] = lb.T * v[i];
    return out;
  }
  // 由两帧 box 的 16 个角点，构造 swept mesh（三角网），实现 A(t)=convhull(A,A')
  std::shared_ptr<fcl::CollisionObjectd>
  make_swept_box_mesh_(const LinkBox& lbA, const LinkBox& lbB) const {
    using BVH = fcl::BVHModel<fcl::OBBRSSd>;
    auto model = std::make_shared<BVH>();

    // 顶点：两端各 8 个（索引 0..7 是 A，8..15 是 B）
    auto A = box_world_corners_(lbA);
    auto B = box_world_corners_(lbB);

    std::vector<Eigen::Vector3d> P(16);
    for (int i=0;i<8;++i)  P[i]   = A[i];
    for (int i=0;i<8;++i)  P[8+i] = B[i];

    // 小工具：把 Eigen::Vector3d 三角加进 BVH
    auto addTri = [&](int i, int j, int k){
      const auto& a = P[i];
      const auto& b = P[j];
      const auto& c = P[k];
      fcl::Vector3d pa(a.x(), a.y(), a.z());
      fcl::Vector3d pb(b.x(), b.y(), b.z());
      fcl::Vector3d pc(c.x(), c.y(), c.z());
      model->addTriangle(pa, pb, pc);
    };

    model->beginModel();

    // (1) 两端端盖：A 盒子 + B 盒子
    //   注意，这个面序必须和 box_world_corners_ / append_box_as_triangles 一致
    int f[12][3] = {
      {0,1,2},{0,2,3},  // +X 面
      {4,6,5},{4,7,6},  // -X 面
      {0,4,5},{0,5,1},  // +Y 面
      {3,2,6},{3,6,7},  // -Y 面
      {0,3,7},{0,7,4},  // +Z 面
      {1,5,6},{1,6,2},  // -Z 面
    };
    // A 端
    for (auto& tri : f) {
      addTri(tri[0], tri[1], tri[2]);
    }
    // B 端（索引整体 +8）
    for (auto& tri : f) {
      addTri(8 + tri[0], 8 + tri[1], 8 + tri[2]);
    }

    // (2) 侧面：对 A 的 6 个面做“挤出”，把每个四边形连接到 B 的对应角点
    // faces4 定义每个面的 4 个顶点环（对应上面的 f 面组）
    int faces4[6][4] = {
      {0,1,2,3}, // 面组1（对应 f[0],f[1]）
      {4,5,6,7}, // 面组2（对应 f[2],f[3]）
      {0,1,5,4}, // 面组3（对应 f[4],f[5]）
      {3,2,6,7}, // 面组4（对应 f[6],f[7]）
      {0,3,7,4}, // 面组5（对应 f[8],f[9]）
      {1,5,6,2}, // 面组6（对应 f[10],f[11]）
    };

    auto addQuadExtrude = [&](int a0, int a1){
      // A 边 (a0,a1)，B 边 (a0+8, a1+8)，四边形拆成两个三角
      // 顺序：(A[a0], A[a1], B[a1]) + (A[a0], B[a1], B[a0])
      addTri(a0,   a1,   8 + a1);
      addTri(a0,   8+a1, 8 + a0);
    };

    for (int fi=0; fi<6; ++fi) {
      int v[4] = {faces4[fi][0], faces4[fi][1], faces4[fi][2], faces4[fi][3]};
      // 四条边都挤出一次
      addQuadExtrude(v[0], v[1]);
      addQuadExtrude(v[1], v[2]);
      addQuadExtrude(v[2], v[3]);
      addQuadExtrude(v[3], v[0]);
    }

    model->endModel();

    auto obj = std::make_shared<fcl::CollisionObjectd>(model);
    // 顶点已经是 world 坐标，所以 transform 设 Identity
    obj->setTransform(fcl::Transform3d(Eigen::Isometry3d::Identity().matrix()));
    obj->computeAABB();
    return obj;
  }

std::vector<LinkBox> build_link_boxes_for_q(const Eigen::Matrix<double,6,1>& q) const {
  std::array<Eigen::Isometry3d,7> Tj; 
  compute_fk_chain_(q, Tj);

  std::array<Eigen::Vector3d,7> P;
  for (int i = 0; i <= 6; ++i) P[i] = Tj[i].translation();

  std::vector<LinkBox> boxes;
  boxes.reserve(16);

  // 机器人 6 段本体
  for (int i = 0; i < 6; ++i) {
    boxes.push_back(make_link_box(P[i], P[i+1], 0.025, 0.025));
  }

  const Eigen::Isometry3d T_world_tcp = Tj[6];

  // ===== end-effector switch =====
  // gripper_enable_ == true  -> use gripper
  // otherwise                -> use original torch
  if (gripper_enable_) {
    if (gripper_in_collision_) {
      auto gboxes = build_gripper_boxes_for_tcp_(T_world_tcp);
      boxes.insert(boxes.end(), gboxes.begin(), gboxes.end());
    }
  } else {
    if (torch_enable_ && torch_in_collision_) {
      Eigen::Vector3d p_tcp = T_world_tcp.translation();
      Eigen::Matrix3d R0    = T_world_tcp.linear();

      // torch segment 1
      Eigen::Vector3d dir1 = R0 * Eigen::Vector3d::UnitZ();
      Eigen::Vector3d p1 = p_tcp;
      Eigen::Vector3d p2 = p_tcp + torch_cyl1_len_ * dir1;
      double r1 = 0.5 * torch_cyl1_dia_;
      boxes.push_back(make_link_box(p1, p2, r1, r1));

      // torch segment 2
      const double tilt = -torch_tilt_deg_ * M_PI / 180.0;
      Eigen::Matrix3d R_tilt =
          Eigen::AngleAxisd(tilt, Eigen::Vector3d::UnitX()).toRotationMatrix();
      Eigen::Vector3d dir2 = R0 * (R_tilt * Eigen::Vector3d::UnitZ());
      Eigen::Vector3d p3 = p2;
      Eigen::Vector3d p4 = p3 + torch_cyl2_len_ * dir2;
      double r2 = 0.5 * torch_cyl2_dia_;
      boxes.push_back(make_link_box(p3, p4, r2, r2));
    }
  }

  return boxes;
}

  // ---------------- 连续安全：扫掠凸包 ----------------
  struct ConvexHullData {
    std::vector<Eigen::Vector3d> verts;
    std::vector<Eigen::Vector3i> faces; // 三角面
  };

 


  // 用 watertight hull + 扇形三角化 生成可视化用的三角网
ConvexHullData make_link_swept_hull_data_(const LinkBox& lbA, const LinkBox& lbB) const {
  auto A = box_world_corners_(lbA);
  auto B = box_world_corners_(lbB);
  std::vector<Eigen::Vector3d> pts; pts.reserve(16);
  for (int i=0;i<8;++i) pts.push_back(A[i]);
  for (int i=0;i<8;++i) pts.push_back(B[i]);

  // 生成 watertight 多边形面
  auto H = chito_planner::core::build_watertight_convex_hull(pts);

  ConvexHullData out;
  out.verts = H.vertices;

  // 扇形三角化每个多边形面 [v0, v1, v2, ..., v(m-1)] -> (v0,v1,v2), (v0,v2,v3), ...
  for (const auto& face : H.faces) {
    if (face.vertex_indices.size() < 3) continue;
    for (size_t i = 1; i + 1 < face.vertex_indices.size(); ++i) {
      out.faces.emplace_back(
        Eigen::Vector3i(face.vertex_indices[0], face.vertex_indices[i], face.vertex_indices[i+1])
      );
    }
  }

  // 如果退化（极少数数值边界），兜底一个 AABB 盒，避免空三角
  if (out.verts.size() < 4 || out.faces.empty()) {
    Eigen::Vector3d mn = out.verts.empty()? Eigen::Vector3d::Zero() : out.verts[0];
    Eigen::Vector3d mx = mn;
    for (auto& p : out.verts) { mn = mn.cwiseMin(p); mx = mx.cwiseMax(p); }
    out.verts.clear(); out.faces.clear();

    // 构建一个细 AABB 盒
    Eigen::Vector3d sz = (mx - mn).cwiseMax(Eigen::Vector3d::Constant(1e-6));
    Eigen::Vector3d c  = 0.5*(mn + mx);
    std::array<Eigen::Vector3d,8> vv = {
      c + Eigen::Vector3d(+0.5*sz.x(), +0.5*sz.y(), +0.5*sz.z()),
      c + Eigen::Vector3d(+0.5*sz.x(), +0.5*sz.y(), -0.5*sz.z()),
      c + Eigen::Vector3d(+0.5*sz.x(), -0.5*sz.y(), -0.5*sz.z()),
      c + Eigen::Vector3d(+0.5*sz.x(), -0.5*sz.y(), +0.5*sz.z()),
      c + Eigen::Vector3d(-0.5*sz.x(), +0.5*sz.y(), +0.5*sz.z()),
      c + Eigen::Vector3d(-0.5*sz.x(), +0.5*sz.y(), -0.5*sz.z()),
      c + Eigen::Vector3d(-0.5*sz.x(), -0.5*sz.y(), -0.5*sz.z()),
      c + Eigen::Vector3d(-0.5*sz.x(), -0.5*sz.y(), +0.5*sz.z())
    };
    out.verts.assign(vv.begin(), vv.end());
    int f[12][3]={{0,1,2},{0,2,3},{4,6,5},{4,7,6},{0,4,5},{0,5,1},{3,2,6},{3,6,7},{0,3,7},{0,7,4},{1,5,6},{1,6,2}};
    for (auto& tri : f) out.faces.emplace_back(Eigen::Vector3i(tri[0],tri[1],tri[2]));
  }

  return out;
}


  void append_hull_triangles_to_marker_(const ConvexHullData& hull, Marker& tri) const {
    auto toPoint=[&](const Eigen::Vector3d& p){ geometry_msgs::msg::Point P; P.x=p.x(); P.y=p.y(); P.z=p.z(); return P; };
    for (auto& f : hull.faces) {
      tri.points.push_back(toPoint(hull.verts[f[0]]));
      tri.points.push_back(toPoint(hull.verts[f[1]]));
      tri.points.push_back(toPoint(hull.verts[f[2]]));
    }
  }





  bool edge_continuous_safe_convexbox_(const Eigen::Matrix<double,6,1>& qA,
                                     const Eigen::Matrix<double,6,1>& qB) const {
  if (obstacles_.empty()) return true;

  auto boxesA = build_link_boxes_for_q(qA);
  auto boxesB = build_link_boxes_for_q(qB);

  const int K_all = static_cast<int>(boxesA.size()); // 可能是 6 或 8

  for (int k = 0; k < K_all; ++k) {
    auto swept = make_swept_box_mesh_(boxesA[k], boxesB[k]);

    double best = std::numeric_limits<double>::infinity();
    for (const auto& ob : obstacles_) {
      fcl::DistanceRequestd req;
      req.enable_signed_distance = true;
      fcl::DistanceResultd res;

      double d = fcl::distance(swept.get(), ob.obj.get(), req, res);
      if (d < best) best = d;
      if (best < cont_min_d_safe_) break;
    }
    if (best < cont_min_d_safe_) {
      return false;
    }
  }
  return true;
}




  // ---------------- 违规项/真实代价 ----------------
  struct Violation {
    int t{-1}, k{-1}; double d{1e9};
    Eigen::Vector3d p_robot, p_obs, n;   // world
    Eigen::RowVectorXd wn;               // 1x6
  };

  std::vector<Violation> collect_violations(const std::vector<Eigen::Matrix<double,6,1>>& path) const {
  std::vector<Violation> vios;
  if (obstacles_.empty()) return vios;

  const int T = steps_;
  for (int t = 0; t < T; ++t) {
    // 构造 6 段连杆盒 +（可选）焊枪两段盒（用于并入 k=5 的检测）
    auto link_boxes = build_link_boxes_for_q(path[t]);
    const int K_link = 6;                    // 只遍历 0..5
    const int K_all  = (int)link_boxes.size(); // 可能是 6 或 8（含焊枪）

    for (int k = 0; k < K_link; ++k) {
      double best_d = std::numeric_limits<double>::infinity();
      Eigen::Vector3d best_pr, best_po;
      int best_source = k;  // 记录来自哪一个盒（k 或 焊枪）

      // ---- 对所有障碍，求距离最小值 ----
      for (const auto& ob : obstacles_) {
        // 1) 先看第 k 段本体
        {
          fcl::DistanceRequestd req; req.enable_nearest_points = true; req.enable_signed_distance = true;
          fcl::DistanceResultd  res;
          double d = fcl::distance(link_boxes[k].obj.get(), ob.obj.get(), req, res);
          if (d < best_d) { best_d = d; best_pr = res.nearest_points[0]; best_po = res.nearest_points[1]; best_source = k; }
        }
        // 2) 如果是最后一段（k==5），把两段焊枪也并入取最小距离
        if (k == 5 && K_all > K_link) {
          for (int tk = K_link; tk < K_all; ++tk) {
            fcl::DistanceRequestd req; req.enable_nearest_points = true; req.enable_signed_distance = true;
            fcl::DistanceResultd  res;
            double d = fcl::distance(link_boxes[tk].obj.get(), ob.obj.get(), req, res);
            if (d < best_d) { best_d = d; best_pr = res.nearest_points[0]; best_po = res.nearest_points[1]; best_source = tk; }
          }
        }
      } // end obstacles loop

      // 违反才加入
      if (best_d < d_safe_) {
        Violation vio; vio.t = t; vio.k = k;  // 注意：k 仍然是 0..5（不把焊枪作为新段）
        vio.d = best_d; vio.p_robot = best_pr; vio.p_obs = best_po;

        Eigen::Vector3d n_world = (best_d >= 0.0) ? (best_pr - best_po) : (best_po - best_pr);
        if (n_world.norm() < 1e-12) n_world = Eigen::Vector3d::UnitX(); else n_world.normalize();
        vio.n = n_world;

        // 雅可比基准：若 k<5，用第 k 段基准；若 k==5（即末段及焊枪并入），一律用第 5 段
        const int base_seg_for_J = (k < 5) ? k : 5;
        Eigen::Matrix<double,3,6> Jp = compute_position_jacobian_moveit_base_(path[t], base_seg_for_J, best_pr);
        vio.wn = (n_world.transpose() * Jp);

        // 对本体段继续执行“清零远端列”的策略；末段 k=5 无需清零；焊枪被并入 k=5 也不清零
        if (k < 5) {
          for (int j = k + 1; j < 6; ++j) vio.wn(0, j) = 0.0;
        }
        vios.push_back(vio);
      }
    } // end k
  } // end t
  return vios;
}


