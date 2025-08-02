#ifndef BOUNDARIES_H
#define BOUNDARIES_H

#include <vector>
#include "Particle.h"
#include "VariableConverter.h"
#include "EquationOfState.h"

enum class BoundaryType {
    PERIODIC,
    FIXED,
    OPEN
};

class Boundaries {
public:
    Boundaries(double x_min, double x_max, BoundaryType type);
    void apply(std::vector<Particle>& particles);

private:
    double x_min_;
    double x_max_;
    BoundaryType type_;
    void applyPeriodic(std::vector<Particle>& particles);
    void applyFixed(std::vector<Particle>& particles);
    void applyOpen(std::vector<Particle>& particles);
};

#endif // BOUNDARIES_H