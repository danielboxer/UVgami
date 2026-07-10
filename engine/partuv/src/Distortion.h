#ifndef DISTORTION_MEASURE_H
#define DISTORTION_MEASURE_H

#include <Eigen/Core>

#include "IO.h"




double calculate_distortion_area(
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXi &F,
    const Eigen::MatrixXd &UV,
    Tree *tree = nullptr,
    int root = -1);

#endif // DISTORTION_MEASURE_H
