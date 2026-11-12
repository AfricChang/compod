// Command line front end for the official PSDR C++ core.
//
//   psdr_cli <pointcloud.ply> --out <refined.npz>
//            [--epsilon-ratio 0.01] [--min-inliers 120] [--knn 10]
//            [--normal-th 0.9] [--max-iter -1] [--max-seconds -1]
//            [--detected <detected.npz>] [--no-refine]
//
// The .npz written by Shape_Container::save() has exactly the keys COMPOD's
// VertexGroup expects: points, normals, classes, group_parameters,
// group_colors, group_num_points, group_points.

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#include <spdlog/spdlog.h>

#include "shape_container.h"
#include "shape_detector.h"

// boost::program_options is not installed in this vcpkg tree; the option set
// is tiny, so parse argv directly.
int main(int argc, char** argv) {
    std::string input, out, detected, ply_out;
    double eps_ratio = 0.01, normal_th = 0.9;
    int min_inliers = 120, knn = 10, max_iter = -1, max_seconds = -1;
    bool no_refine = false;

    try {
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            auto next = [&]() -> std::string {
                if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
                return std::string(argv[++i]);
            };
            if (a == "--help") {
                std::cout << "usage: psdr_cli <cloud.ply> --out <refined.npz> "
                             "[--detected f.npz] [--ply f.ply] [--epsilon-ratio 0.01] "
                             "[--min-inliers 120] [--knn 10] [--normal-th 0.9] "
                             "[--max-iter -1] [--max-seconds -1] [--no-refine]\n";
                return 0;
            } else if (a == "--no-refine") no_refine = true;
            else if (a == "--out") out = next();
            else if (a == "--detected") detected = next();
            else if (a == "--ply") ply_out = next();
            else if (a == "--epsilon-ratio") eps_ratio = std::stod(next());
            else if (a == "--min-inliers") min_inliers = std::stoi(next());
            else if (a == "--knn") knn = std::stoi(next());
            else if (a == "--normal-th") normal_th = std::stod(next());
            else if (a == "--max-iter") max_iter = std::stoi(next());
            else if (a == "--max-seconds") max_seconds = std::stoi(next());
            else if (a.rfind("--", 0) == 0) throw std::runtime_error("unknown option " + a);
            else if (input.empty()) input = a;
            else throw std::runtime_error("unexpected argument " + a);
        }
        if (input.empty() || out.empty())
            throw std::runtime_error("need <cloud.ply> and --out <file.npz>");
    } catch (const std::exception& e) {
        std::cerr << "bad arguments: " << e.what() << std::endl;
        return 2;
    }

    auto t0 = std::chrono::steady_clock::now();
    Shape_Detector SD;
    SD._logger->set_level(spdlog::level::info);

    if (SD.load_points(input) != 0) {
        std::cerr << "failed to load point cloud: " << input << std::endl;
        return 1;
    }
    const double diag = SD.get_bbox_diagonal();
    const double eps = eps_ratio * diag;
    std::cout << "PSDR_CLI points=" << SD.points.size()
              << " bbox_diag=" << diag << " epsilon=" << eps
              << " min_inliers=" << min_inliers << " knn=" << knn
              << " normal_th=" << normal_th << std::endl;

    SD.set_detection_parameters(min_inliers, eps, knn, normal_th);
    SD.set_discretization_parameters(0.5, eps / 2.0);

    Shape_Container SC(&SD);
    if (SC.detect() != 0) {
        std::cerr << "detect() failed" << std::endl;
        return 1;
    }
    std::cout << "PSDR_CLI detected_primitives=" << SC.get_number_of_primitives()
              << " inliers=" << SC.get_total_number_of_inliers() << std::endl;
    if (!detected.empty()) SC.save(detected);

    if (!no_refine) {
        SD.set_constraint(false);
        SD.set_weight_m(0);
        SD.set_metric_type(0);  // hybrid
        SD.set_lambda_complexity(1.0);
        SD.set_lambda_completeness(1.0);
        SD.set_lambda_regularity(1.0);
        SD.set_lambda_fidelity(1.0);
        SD.set_discretization_parameters(0.5, eps / 2.0);
        if (SC.refine(max_iter, max_seconds) != 0) {
            std::cerr << "refine() failed" << std::endl;
            return 1;
        }
    }

    int rc = SC.save(out);
    if (rc != 0) {
        std::cerr << "save(" << out << ") failed" << std::endl;
        return 1;
    }
    if (!ply_out.empty()) {
        try {
            SC.to_ply(ply_out, "convex");
            std::cout << "PSDR_CLI ply_out=" << ply_out << std::endl;
        } catch (const std::exception& ex) {
            std::cerr << "to_ply failed: " << ex.what() << std::endl;
        }
    }
    std::cout << "PSDR_CLI refined_primitives=" << SC.get_number_of_primitives()
              << " inliers=" << SC.get_total_number_of_inliers()
              << " out=" << out
              << " seconds=" << std::chrono::duration<double>(
                     std::chrono::steady_clock::now() - t0).count()
              << std::endl;
    return 0;
}
