#include "Distortion.h"
#include "IO.h"
#include <iostream>
#include <Eigen/Core>
#include "Config.h"
#include <igl/doublearea.h>
#include <unordered_map>



double calculate_distortion_area(
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXi &F,
    const Eigen::MatrixXd &UV,
    Tree *tree,
    int root)
{
    Eigen::VectorXd A3, A2;
    igl::doublearea(V, F, A3);
    igl::doublearea(UV, F, A2);

    const double total_area_3D = std::abs(A3.sum()) * 0.5;
    const double total_area_2D = std::abs(A2.sum()) * 0.5;

    if (total_area_3D == 0.0) {
        std::cerr << "Warning: Total 3D area is 0. Returning 1 distortion.\n";
        return 1.0;
    }

    const double total_ratio = total_area_2D / total_area_3D;
    double distortion_sum = 0.0;
    const int num_faces = F.rows();
    double max_distortion = 0.0;

    std::vector<FaceAreaData> face_data(num_faces);

    for(int i = 0; i < num_faces; ++i) {
        double face_area_3D = std::abs(A3(i) * 0.5);
        double face_area_2D = std::abs(A2(i) * 0.5);
        double ratio = 0.0;

        if(face_area_3D != 0.0) {
            ratio = face_area_2D / face_area_3D;
        }
        if (tree != nullptr)
            face_data[i] = {face_area_2D, face_area_3D, ratio};

        if(face_area_3D == 0.0) continue;

        double distortion = ratio / total_ratio;

        if(CONFIG_pipelineThreshold >= 1) {
            distortion = distortion < 1.0 ? 1.0 / distortion : distortion;
        } else {
            distortion = distortion < 1.0 ? 1.0 - distortion : 1.0 - 1.0 / distortion;
        }

        if (distortion > max_distortion) {
            max_distortion = distortion;
        }
        distortion_sum += distortion;
        if (distortion_sum > static_cast<double>(std::numeric_limits<int>::max())) {
            return static_cast<double>(std::numeric_limits<int>::max());
        }

    }
    /*
        ratio_n = face_area_2D_n / face_area_3D_n

        distortion_sum = sum(ratio_n / total_ratio) = sum(face_area_2D_n / face_area_3D_n) / (sum(face_area_2D_n) / sum(face_area_3D_n))
    }
    */

    if (tree != nullptr) {
        tree->update_distortion_norm(face_data, root == -1 ? tree->root() : root);
    }

    // return distortion_sum / static_cast<double>(num_faces);
    double avg_distortion = distortion_sum / static_cast<double>(num_faces);
    if (std::isnan(avg_distortion)) {
        return static_cast<double>(std::numeric_limits<int>::max());
    }
    return avg_distortion;
    // return max_distortion;

}
