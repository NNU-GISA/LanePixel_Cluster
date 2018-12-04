//
// Created by lightol on 18-12-4.
//

#ifndef HOUGH_TRANSFORM_SOLVER_H
#define HOUGH_TRANSFORM_SOLVER_H

struct costFunctor {
    costFunctor(double _x, double _y): x(_x), y(_y) {}

    template <typename T>
    bool operator()(const T* const a, const T* const b, T* residual) const {
        residual[0] = T(y) - (a[0]*x + b[0]);

        return true;
    }

    const double x;
    const double y;
};

#endif //HOUGH_TRANSFORM_SOLVER_H
