#include "Particle.h"
#include "DensityUpdater.h"
#include "EquationOfState.h"
#include "DissipationTerms.h"
#include "Kernel.h"
#include "MathUtils.h"
#include "Globals.h"
#include "Logger.h"
#include <cmath>
#include <iostream>
#include <omp.h>


Particle::Particle(const std::array<double, 3>& pos,
                   const std::array<double, 3>& vel,
                   double m,
                   double sInE,
                   bool ghost)
    : isGhost(ghost),
      //primitives n P u v
      density(0.0),
      pressure(0.0),
      specificInternalEnergy(sInE),
      velocity(vel),
      //conservatives  N S e
      baryonDensity(0.0),
      specificMomentum({0.0, 0.0, 0.0}),
      specificEnergy(0.0),
      //sph
      mass(m),
      totalSpecificEnergy(0.0),
      h(0.0),
      Omega(1.0),
      position(pos),
      acceleration({0.0, 0.0, 0.0}),
      energyChangeRate(0.0),
      conv_h("con")

{
}

// --------------------------------------------------------------------------------------
// 1) Actualiza la "baryonDensity" ~ N
// --------------------------------------------------------------------------------------
void Particle::updateDensity(const std::vector<Particle>& neighbors,
                             DensityUpdater& densityUpdater,
                             const Kernel& kernel){

    densityUpdater.updateDensity(*this, neighbors, kernel);
}

// --------------------------------------------------------------------------------------
// 2) Usa la Ecuación de Estado para calcular la presión P(n, u), 
// --------------------------------------------------------------------------------------
void Particle::updatePressure(const EquationOfState& eos)
{
    pressure = eos.calculatePressure(density, specificInternalEnergy);
}

// --------------------------------------------------------------------------------------
// 4) Calcula dq/dt y d eHat/dt. Interpretamos "acceleration" = dq/dt, "energyChangeRate" = d eHat/dt
// --------------------------------------------------------------------------------------
void Particle::calculateAccelerationAndEnergyChangeRate(const std::vector<Particle>& neighbors,
                                                          const Kernel& kernel,
                                                          const DissipationTerms& dissipation,
                                                          const EquationOfState& eos)
{
    double dS_dt_x = 0.0, dS_dt_y = 0.0, dS_dt_z = 0.0;
    double d_eHat_dt = 0.0;  

    #pragma omp parallel for reduction(+:dS_dt_x, dS_dt_y, dS_dt_z, d_eHat_dt)
    for (size_t i = 0; i < neighbors.size(); ++i) {
        const auto& nb = neighbors[i];
        if (&nb == this) continue;

        std::array<double, 3> r_ab = {
            position[0] - nb.position[0],
            position[1] - nb.position[1],
            position[2] - nb.position[2]
        };
        std::array<double, 3> r_ab_hat = MathUtils::hatVector(r_ab);
        std::array<double, 3> gradW_a = kernel.gradW(r_ab, h);
        std::array<double, 3> gradW_b = kernel.gradW(r_ab, nb.h);

        std::array<double, 3> AvgGradW = {
           0.5 * (gradW_a[0] + gradW_b[0]),
           0.5 * (gradW_a[1] + gradW_b[1]),
           0.5 * (gradW_a[2] + gradW_b[2])
        };

        double nu_b = nb.mass;
        double Omeg_a = Omega;
        double Omeg_b = nb.Omega;
        double P_a = pressure; 
        double P_b = nb.pressure;
        double N_a = baryonDensity; 
        double N_b = nb.baryonDensity;
        //double fact_a = P_a / (Omeg_a * N_a * N_a);
        //double fact_b = P_b / (Omeg_b * N_b * N_b);

        std::array<double, 3> v_a = velocity;
        std::array<double, 3> v_b = nb.velocity;


        auto [q_a, q_b] = dissipation.Viscosity_LiptaiPrice2018(*this, nb, kernel, eos);
        double fact_a = (P_a + q_a) / (Omeg_a * N_a * N_a);
        double fact_b = (P_b + q_b) / (Omeg_b * N_b * N_b);

        //auto [Pi_ab, Omega_ab] = dissipation.Chow1997Viscosity(*this, nb, kernel, eos);
        //dS_dt_x -= nu_b * (fact_a * gradW_a[0] + fact_b * gradW_b[0] + Pi_ab * AvgGradW[0]);
        //dS_dt_y -= nu_b * (fact_a * gradW_a[1] + fact_b * gradW_b[1] + Pi_ab * AvgGradW[1]);
        //dS_dt_z -= nu_b * (fact_a * gradW_a[2] + fact_b * gradW_b[2] + Pi_ab * AvgGradW[2]);

        dS_dt_x -= nu_b * (fact_a * gradW_a[0] + fact_b * gradW_b[0] );
        dS_dt_y -= nu_b * (fact_a * gradW_a[1] + fact_b * gradW_b[1] );
        dS_dt_z -= nu_b * (fact_a * gradW_a[2] + fact_b * gradW_b[2] );

        std::array<double, 3> sum_v_a = {
            fact_a * v_b[0],
            fact_a * v_b[1],
            fact_a * v_b[2]
        };
        std::array<double, 3> sum_v_b = {
            fact_b * v_a[0],
            fact_b * v_a[1],
            fact_b * v_a[2]
        };
        
        auto [Condu_a, Condu_b] = dissipation.Conductivity_LiptaiPrice2018(*this, nb, kernel, eos);
        double v_sum_dot_gradW_a = MathUtils::dotProduct(sum_v_a, gradW_a);
        double v_sum_dot_gradW_b = MathUtils::dotProduct(sum_v_b, gradW_b);

        //d_eHat_dt -= nu_b * (v_sum_dot_gradW_a + v_sum_dot_gradW_b + Omega_ab * MathUtils::dotProduct(AvgGradW, r_ab_hat));

        double G_a = MathUtils::dotProduct(gradW_a, r_ab_hat) / Omeg_a;
        double G_b = MathUtils::dotProduct(gradW_b, r_ab_hat) / Omeg_b;
        double Conductivity = Condu_a * G_a + Condu_b * G_b;

        d_eHat_dt -= nu_b * (v_sum_dot_gradW_a + v_sum_dot_gradW_b - Conductivity);
    }

    acceleration = {dS_dt_x, dS_dt_y, dS_dt_z};
    energyChangeRate = d_eHat_dt;

}

