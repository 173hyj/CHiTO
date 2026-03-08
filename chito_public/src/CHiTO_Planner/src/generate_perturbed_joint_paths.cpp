#include <array>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>

namespace fs = std::filesystem;

using Joint6 = std::array<double, 6>;

// ============================================================
// helpers
// ============================================================
static bool SaveJointPathCsv(
    const std::string& filepath,
    const std::vector<Joint6>& path_q,
    bool save_deg = true,
    int precision = 6)
{
    std::ofstream fout(filepath);
    if (!fout.is_open()) return false;

    fout << std::fixed << std::setprecision(precision);
    fout << "idx,q1_" << (save_deg ? "deg" : "rad")
         << ",q2_" << (save_deg ? "deg" : "rad")
         << ",q3_" << (save_deg ? "deg" : "rad")
         << ",q4_" << (save_deg ? "deg" : "rad")
         << ",q5_" << (save_deg ? "deg" : "rad")
         << ",q6_" << (save_deg ? "deg" : "rad")
         << "\n";

    for (size_t i = 0; i < path_q.size(); ++i) {
        fout << i;
        for (int k = 0; k < 6; ++k) {
            fout << "," << path_q[i][k];
        }
        fout << "\n";
    }
    return true;
}

static Joint6 LerpJoint(const Joint6& q0, const Joint6& q1, double alpha)
{
    Joint6 q{};
    for (int i = 0; i < 6; ++i) {
        q[i] = (1.0 - alpha) * q0[i] + alpha * q1[i];
    }
    return q;
}

static std::vector<Joint6> GenerateInterpolatedPath(
    const Joint6& q_start_deg,
    const Joint6& q_goal_deg,
    int num_interp_points)
{
    const int num_waypoints = num_interp_points + 2;

    std::vector<Joint6> path;
    path.reserve(num_waypoints);

    for (int i = 0; i < num_waypoints; ++i) {
        double alpha = static_cast<double>(i) / static_cast<double>(num_waypoints - 1);
        path.push_back(LerpJoint(q_start_deg, q_goal_deg, alpha));
    }
    return path;
}

static void AddPerturbationToMiddlePoints(
    std::vector<Joint6>& path_deg,
    std::mt19937& rng,
    double perturb_min_deg,
    double perturb_max_deg,
    bool positive_only)
{
    std::uniform_real_distribution<double> dist_signed(-perturb_max_deg, perturb_max_deg);
    std::uniform_real_distribution<double> dist_pos(perturb_min_deg, perturb_max_deg);

    if (path_deg.size() <= 2) return;

    for (size_t p = 1; p + 1 < path_deg.size(); ++p) {
        for (int j = 0; j < 6; ++j) {
            double delta = positive_only ? dist_pos(rng) : dist_signed(rng);
            path_deg[p][j] += delta;
        }
    }
}

static bool ParseJoint6FromString(const std::string& s, Joint6& q)
{
    std::stringstream ss(s);
    std::string item;
    std::vector<double> vals;

    while (std::getline(ss, item, ',')) {
        try {
            vals.push_back(std::stod(item));
        } catch (...) {
            return false;
        }
    }

    if (vals.size() != 6) return false;
    for (int i = 0; i < 6; ++i) q[i] = vals[i];
    return true;
}

static void PrintUsage(const char* prog)
{
    std::cout << "Usage:\n";
    std::cout << "  " << prog << " [out_dir] [num_interp_points] [start_deg_csv] [goal_deg_csv]\n\n";
    std::cout << "Example:\n";
    std::cout << "  " << prog
              << " /home/hyj/iris_rviz_ws/src/iris_rviz_cpp/src/generated_joint_paths"
              << " 5"
              << " \"-111,-141,-81,-136,-107,0\""
              << " \"-69,-116,-83,-158,-65,0\"\n";
}

// ============================================================
// main
// ============================================================
int main(int argc, char** argv)
{
    // 默认值
    Joint6 q_start_deg = {-111, -141, -81, -136, -107, 0};
    Joint6 q_goal_deg  = { -69, -116, -83, -158,  -65, 0};

    int num_groups = 50;
    int num_interp_points = 5;

    double perturb_min_deg = 0.0;
    double perturb_max_deg = 30.0;
    bool positive_only = false;

    std::string out_dir = "/home/hyj/iris_rviz_ws/src/iris_rviz_cpp/src/generated_joint_paths";

    // 命令行参数覆盖
    // argv[1] = out_dir
    // argv[2] = num_interp_points
    // argv[3] = start_deg_csv
    // argv[4] = goal_deg_csv
    if (argc >= 2) {
        out_dir = argv[1];
    }
    if (argc >= 3) {
        num_interp_points = std::atoi(argv[2]);
        if (num_interp_points < 0) {
            std::cerr << "num_interp_points must be >= 0\n";
            return 1;
        }
    }
    if (argc >= 4) {
        if (!ParseJoint6FromString(argv[3], q_start_deg)) {
            std::cerr << "Failed to parse start_deg_csv\n";
            PrintUsage(argv[0]);
            return 1;
        }
    }
    if (argc >= 5) {
        if (!ParseJoint6FromString(argv[4], q_goal_deg)) {
            std::cerr << "Failed to parse goal_deg_csv\n";
            PrintUsage(argv[0]);
            return 1;
        }
    }

    fs::create_directories(out_dir);

    const std::vector<Joint6> base_path =
        GenerateInterpolatedPath(q_start_deg, q_goal_deg, num_interp_points);

    {
        const std::string base_file = (fs::path(out_dir) / "base_ik.csv").string();
        if (!SaveJointPathCsv(base_file, base_path, true, 6)) {
            std::cerr << "Failed to save base path: " << base_file << std::endl;
            return 1;
        }
    }

    std::random_device rd;
    std::mt19937 rng(rd());

    for (int i = 0; i < num_groups; ++i) {
        std::vector<Joint6> path = base_path;

        AddPerturbationToMiddlePoints(
            path,
            rng,
            perturb_min_deg,
            perturb_max_deg,
            positive_only);

        std::ostringstream oss;
        oss << "run_" << std::setw(3) << std::setfill('0') << i << "_ik.csv";
        const std::string filepath = (fs::path(out_dir) / oss.str()).string();

        if (!SaveJointPathCsv(filepath, path, true, 6)) {
            std::cerr << "Failed to save: " << filepath << std::endl;
            return 1;
        }
    }

    {
        std::ofstream fout((fs::path(out_dir) / "info.txt").string());
        if (fout.is_open()) {
            fout << "q_start_deg: ";
            for (int i = 0; i < 6; ++i) fout << q_start_deg[i] << (i < 5 ? ", " : "\n");

            fout << "q_goal_deg: ";
            for (int i = 0; i < 6; ++i) fout << q_goal_deg[i] << (i < 5 ? ", " : "\n");

            fout << "num_groups: " << num_groups << "\n";
            fout << "num_interp_points: " << num_interp_points << "\n";
            fout << "num_waypoints_total: " << (num_interp_points + 2) << "\n";
            fout << "perturb_min_deg: " << perturb_min_deg << "\n";
            fout << "perturb_max_deg: " << perturb_max_deg << "\n";
            fout << "positive_only: " << (positive_only ? "true" : "false") << "\n";
            fout << "out_dir: " << out_dir << "\n";
        }
    }

    std::cout << "Done. Generated " << num_groups
              << " IK csv files in: " << out_dir << std::endl;
    std::cout << "Middle interpolation points = " << num_interp_points
              << ", total waypoints = " << (num_interp_points + 2) << std::endl;

    return 0;
}
