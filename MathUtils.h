#ifndef MATHUTILS_H
#define MATHUTILS_H

#include <array>
#include <cmath>

namespace MathUtils {
    inline double dotProduct(const std::array<double, 3>& a, const std::array<double, 3>& b) {
        return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
    }
    
    inline double vectorNorm(const std::array<double, 3>& a) {
        return std::sqrt(dotProduct(a, a));
    }
    
    inline std::array<double, 3> hatVector(const std::array<double, 3>& a) {
        double norm = vectorNorm(a);
        if (norm == 0.0) return {0.0, 0.0, 0.0};
        return {a[0]/norm, a[1]/norm, a[2]/norm};
    }
    inline  double vectorNormSquared(const std::array<double, 3>& vec) {
        return vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2];
    }
}

#endif // MATHUTILS_H
