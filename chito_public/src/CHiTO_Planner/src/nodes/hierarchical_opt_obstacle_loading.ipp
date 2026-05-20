  // ---------------- YAML/SCENE -> Obstacles ----------------
  std::vector<Obstacle> build_obstacles_from_yaml_(const std::string& yaml_path) {
    std::vector<Obstacle> obs;
    try {
      YAML::Node root = YAML::LoadFile(yaml_path);
      auto V3 = [](const YAML::Node& n){ return Eigen::Vector3d(n[0].as<double>(), n[1].as<double>(), n[2].as<double>()); };

      if (root["boxes"] && root["boxes"].IsSequence()) {
        for (const auto& b : root["boxes"]) {
          Eigen::Vector3d mn = V3(b["min"]), mx = V3(b["max"]);
          Eigen::Vector3d sz = (mx - mn).cwiseAbs();
          Eigen::Vector3d c  = 0.5*(mx + mn);
          auto geom = std::make_shared<fcl::Boxd>(sz.x(), sz.y(), sz.z());
          auto obj  = std::make_shared<fcl::CollisionObjectd>(geom);
          Eigen::Isometry3d T=Eigen::Isometry3d::Identity(); T.translation()=c;
          obj->setTransform(fcl::Transform3d(T.matrix())); obj->computeAABB();
          obs.push_back({ObType::BOX, c, sz, Eigen::Vector3d::Zero(), obj});
        }
      }
      if (root["spheres"] && root["spheres"].IsSequence()) {
        for (const auto& s : root["spheres"]) {
          Eigen::Vector3d c = V3(s["center"]); double r=s["r"].as<double>();
          auto geom=std::make_shared<fcl::Sphered>(r);
          auto obj =std::make_shared<fcl::CollisionObjectd>(geom);
          Eigen::Isometry3d T=Eigen::Isometry3d::Identity(); T.translation()=c;
          obj->setTransform(fcl::Transform3d(T.matrix())); obj->computeAABB();
          obs.push_back({ObType::SPHERE, c, Eigen::Vector3d(2*r,2*r,2*r), Eigen::Vector3d::Zero(), obj});
        }
      }
      if (root["cylinders"] && root["cylinders"].IsSequence()) {
        auto deg2rad=[](double d){return d*M_PI/180.0;};
        for (const auto& cy : root["cylinders"]) {
          Eigen::Vector3d c = V3(cy["center"]);
          double r = cy["r"].as<double>(), h = cy["h"].as<double>();
          Eigen::Vector3d rpy = cy["rpy_deg"] ? V3(cy["rpy_deg"]) : Eigen::Vector3d::Zero();

          auto geom = std::make_shared<fcl::Cylinderd>(r, h);
          auto obj  = std::make_shared<fcl::CollisionObjectd>(geom);

          Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
          T.linear() =
            (Eigen::AngleAxisd(deg2rad(rpy.z()),Eigen::Vector3d::UnitZ())*
             Eigen::AngleAxisd(deg2rad(rpy.y()),Eigen::Vector3d::UnitY())*
             Eigen::AngleAxisd(deg2rad(rpy.x()),Eigen::Vector3d::UnitX())).toRotationMatrix();
          T.translation() = c;

          obj->setTransform(fcl::Transform3d(T.matrix())); obj->computeAABB();
          obs.push_back({ObType::CYLINDER, c, Eigen::Vector3d(2*r,2*r,h), rpy, obj});
        }
      }
      RCLCPP_INFO(get_logger(), "[OBST] %zu loaded from YAML '%s'", obs.size(), yaml_path.c_str());
    } catch (const std::exception& e) {
      // 让上层自动切换到 .scene
      RCLCPP_WARN(get_logger(), "[OBST] YAML load failed: %s (will try legacy .scene)", e.what());
    }
    return obs;
  }

  // 旧 .scene 格式（仅 box）：示例块
  // * Box_0
  // cx cy cz
  // qx qy qz qw
  // 1
  // box
  // sx sy sz          (整尺寸)
  // 0 0 0
  // 0 0 0 1
  // 0 0 0 0
  // 0
  // .                 （可选分隔符，忽略）
  std::vector<Obstacle> build_obstacles_from_legacy_scene_boxes_(const std::string& scene_path) {
  std::vector<Obstacle> obs;
  std::ifstream in(scene_path);
  if (!in.is_open()) {
    RCLCPP_ERROR(get_logger(), "[OBST] cannot open scene: %s", scene_path.c_str());
    return obs;
  }

  auto quatToR = [](double x,double y,double z,double w)->Eigen::Matrix3d {
    Eigen::Quaterniond q(w,x,y,z);  // 文件里是 x y z w，我们这里构造 q(w,x,y,z)
    q.normalize();
    return q.toRotationMatrix();
  };

  auto skip_rest_of_line = [&](void){ std::string dummy; std::getline(in, dummy); };
  auto skip_n_lines      = [&](int n){ for (int i=0;i<n;++i){ std::string dummy; std::getline(in, dummy); } };

  std::string tok;
  while (in >> tok) {
    if (tok == "*") {
      std::string name; 
      in >> name;      // Box_0 / Cylinder_0 等

      // 1) 位置
      double cx,cy,cz; 
      in >> cx >> cy >> cz;

      // 2) 四元数 (x y z w)
      double qx,qy,qz,qw; 
      in >> qx >> qy >> qz >> qw;

      // 3) 占位 1
      int one; 
      in >> one;

      // 4) 类型
      std::string typ; 
      in >> typ;

      if (typ == "box") {
        // 5) 尺寸（整尺寸）
        double sx,sy,sz; 
        in >> sx >> sy >> sz;

        // 后面 4 行占位
        skip_rest_of_line();
        skip_n_lines(4);

        Eigen::Vector3d c(cx,cy,cz);
        Eigen::Matrix3d R = quatToR(qx,qy,qz,qw);

        auto geom = std::make_shared<fcl::Boxd>(sx, sy, sz);
        auto obj  = std::make_shared<fcl::CollisionObjectd>(geom);

        Eigen::Isometry3d Tw = Eigen::Isometry3d::Identity();
        Tw.linear()      = R;
        Tw.translation() = c;
        obj->setTransform(fcl::Transform3d(Tw.matrix()));
        obj->computeAABB();

        Obstacle ob;
        ob.type     = ObType::BOX;
        ob.center   = c;
        ob.size     = Eigen::Vector3d(sx,sy,sz);
        ob.rpy_deg  = Eigen::Vector3d::Zero();   // 可视化时我们会从 obj 里读旋转
        ob.obj      = obj;
        obs.push_back(std::move(ob));
      }
      else if (typ == "cylinder") {
        // 5) 半径 + 高度
        double r, h;
        in >> r >> h;

        // 后面 4 行占位
        skip_rest_of_line();
        skip_n_lines(4);

        Eigen::Vector3d c(cx,cy,cz);
        Eigen::Matrix3d R = quatToR(qx,qy,qz,qw);

        // FCL cylinder: radius r, height h
        auto geom = std::make_shared<fcl::Cylinderd>(r, h);
        auto obj  = std::make_shared<fcl::CollisionObjectd>(geom);

        Eigen::Isometry3d Tw = Eigen::Isometry3d::Identity();
        Tw.linear()      = R;
        Tw.translation() = c;
        obj->setTransform(fcl::Transform3d(Tw.matrix()));
        obj->computeAABB();

        Obstacle ob;
        ob.type    = ObType::CYLINDER;
        ob.center  = c;
        ob.size    = Eigen::Vector3d(2*r, 2*r, h); // scale.x/y = 直径, scale.z = 高度
        ob.rpy_deg = Eigen::Vector3d::Zero();      // 实际姿态从 obj->getTransform() 读
        ob.obj     = obj;
        obs.push_back(std::move(ob));
      }
      else {
        // 其他未知类型：按原格式跳过
        skip_rest_of_line();
        skip_n_lines(4);
      }
    }
    else if (tok == ".") {
      // 分隔符，忽略
      continue;
    }
    else {
      // 其他 token：丢弃本行
      skip_rest_of_line();
    }
  }

  RCLCPP_INFO(get_logger(),
              "[OBST] %zu object(s) loaded from legacy scene '%s'",
              obs.size(), scene_path.c_str());
  return obs;
}


  // 自动选择：YAML 优先；若 0 或 YAML 解析失败则尝试旧 .scene 盒子
  std::vector<Obstacle> build_obstacles_auto_(const std::string& path) {
    auto ends_with = [](const std::string& s, const std::string& suf){
      if (s.size() < suf.size()) return false;
      return std::equal(suf.rbegin(), suf.rend(), s.rbegin(),
                        [](char a, char b){ return std::tolower(a)==std::tolower(b); });
    };

    std::vector<Obstacle> obs;
    bool tried_yaml = false;

    if (ends_with(path, ".yaml") || ends_with(path, ".yml")) {
      tried_yaml = true;
      obs = build_obstacles_from_yaml_(path);
      if (!obs.empty()) return obs;
    } else {
      // 即便不是 .yaml，先试 YAML（兼容有人误填后缀）
      obs = build_obstacles_from_yaml_(path);
      if (!obs.empty()) return obs;
    }

    // YAML 没读到 → 旧 .scene 仅 box
    auto obs2 = build_obstacles_from_legacy_scene_boxes_(path);
    if (!obs2.empty()) return obs2;

    if (tried_yaml) {
      RCLCPP_WARN(get_logger(), "[OBST] Neither YAML nor legacy scene produced obstacles: %s", path.c_str());
    }
    return {};
  }




