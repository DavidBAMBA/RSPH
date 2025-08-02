// ShockTubeIC.h — concrete implementation of a 1‑D Riemann‑problem IC
// =============================================================================
#ifndef SHOCKTUBEIC_H
#define SHOCKTUBEIC_H

#include "IInitialCondition.h"
#include "InitialConditions.h"   // for InitialConditionType enum
#include <array>

// Light‑weight parameter pack describing a two‑state shock tube. You can load
// this from YAML/JSON later and pass it straight to the constructor.
struct ShockICParams {
    double rhoL = 1.0, rhoR = 1.0;             // densities
    double pL   = 1.0, pR   = 1.0;             // pressures
    std::array<double,3> vL{0,0,0}, vR{0,0,0}; // velocities
    double hL = 1.2, hR = 1.2;                 // h‑multipliers
    // optional perturbation (added on right state)
    double pertAmp  = 0.0;                     // amplitude
    double pertFreq = 0.0;                     // k
};

/**
 * Concrete strategy that generates classic 1‑D shock‑tube setups
 * (relativistic or Newtonian) used for code validation.
 */
class ShockTubeIC final : public IInitialCondition {
public:
    // Create from pre‑defined enum (legacy path from InitialConditions::setType)
    explicit ShockTubeIC(InitialConditionType legacyType);

    // Create directly from a parameter structure (future YAML/JSON path)
    explicit ShockTubeIC(const ShockICParams& prm) : params(prm) {}

    // IInitialCondition interface
    void generate(std::vector<Particle>&           particles,
                  std::shared_ptr<Kernel>          kernel,
                  std::shared_ptr<EquationOfState> eos,
                  int                              N,
                  double                           x_min,
                  double                           x_max) override;

private:
    ShockICParams params;
};

#endif // SHOCKTUBEIC_H
