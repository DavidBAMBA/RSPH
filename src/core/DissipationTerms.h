// DissipationTerms.h
#ifndef DISSIPATIONTERMS_H
#define DISSIPATIONTERMS_H

#include "Particle.h"
#include "Kernel.h"
#include "EquationOfState.h"
#include <array>
#include <cmath>
#include <utility>

class DissipationTerms {
public:
    DissipationTerms(double alpha_value, double beta_value);
    
/*     // MONAGAN 1997 (viscous dissipation)
    std::pair<double, double> mon97_art_vis(const Particle& a, const Particle& b, const Kernel& kernel, const EquationOfState& equationOfState) const;

    // PRICE 2010 (conductivity dissipation)
    double price08_therm_cond(double p_i, double p_j, double rho_mean, double u_i, double u_j) const;

    // PHANTOM
    std::pair<double, double> PriceLondato2010Viscosity(const Particle& a, const Particle& b, const Kernel& kernel, const EquationOfState& equationOfState) const;
    double viscousShockHeating(const Particle& a, const Particle& b, const Kernel& kernel, const EquationOfState& equationOfState) const;
    double artificialThermalConductivity(const Particle& a, const Particle& b, const Kernel& kernel, const EquationOfState& equationOfState) const;
 */
    std::pair<double, double> Chow1997Viscosity(const Particle& a, const Particle& b, const Kernel& kernel, const EquationOfState& equationOfState) const;
    std::pair<double, double> Viscosity_LiptaiPrice2018(const Particle& a, const Particle& b, const Kernel& kernel, const EquationOfState& equationOfState) const;
    std::pair<double, double> Conductivity_LiptaiPrice2018(const Particle& a, const Particle& b, const Kernel& kernel, const EquationOfState& equationOfState) const;

    double alphaAV;
    double betaAV;
    
private:

};

#endif // DISSIPATIONTERMS_H
