#ifndef DENSITYUPDATER_H
#define DENSITYUPDATER_H

#include "Particle.h"
#include "Kernel.h"
#include <vector>
#include <array>
#include <cmath>
#include <utility>
#include <omp.h>

class DensityUpdater {
public:
    DensityUpdater(double eta, double tol, bool use_fixed_h, double fixed_h);

    // Actualiza la densidad de la partícula (y su h) usando los vecinos y el kernel.
    void updateDensity(Particle& particle, const std::vector<Particle>& neighbors, const Kernel& kernel);
    double calculateDensity(const Particle& particle, const std::vector<Particle>& neighbors, double h, const Kernel& kernel) const;
    double calculateOmega(const Particle& particle, const std::vector<Particle>& neighbors, double h, double rho, const Kernel& kernel) const;
    std::pair<double, double> computeFunctionAndDerivative(const Particle& particle, const std::vector<Particle>& neighbors, double h, const Kernel& kernel) const;

    double newtonRaphson(const Particle& particle, const std::vector<Particle>& neighbors, double h_guess, const Kernel& kernel) const;
    double bisectionMethod(const Particle& particle, const std::vector<Particle>& neighbors, 
                             double h_left, double h_right, const Kernel& kernel) const;
    double findConvergedH(const Particle& particle, const std::vector<Particle>& neighbors, double h_guess, const Kernel& kernel) const;

    double eta_;
    double tol_;
    bool use_fixed_h_;
    double fixed_h_;
};

#endif // DENSITYUPDATER_H
