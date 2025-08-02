#ifndef PARTICLE_H
#define PARTICLE_H

#include <array>
#include <string>
#include <vector>
#include <memory>
#include <omp.h>

// Forward declarations
class DensityUpdater;
class EquationOfState;
class DissipationTerms;
class Kernel;

class Particle {
public:

    bool isGhost;

    Particle(const std::array<double, 3>& pos,
             const std::array<double, 3>& vel,
             double m,
             double sIntE,
             bool ghost = false);

    // -----------------------------------------------
    // Métodos de actualización para la simulación SPH
    // -----------------------------------------------
    
    void updateDensity(const std::vector<Particle>& neighbors,DensityUpdater& densityUpdater,const Kernel& kernel);

    void updatePressure(const EquationOfState& eos);

    /**
     * @brief Calcula la aceleración (dv/dt) y el cambio de energía interna (du/dt).
     * @param neighbors     Vecinos de la partícula.
     * @param kernel        Kernel SPH (función de suavizado).
     * @param dissipation   Objeto con los términos disipativos (viscosidad, etc.).
     * @param equationOfState  Ecuación de estado para cálculos de presión, etc.
     */
    void calculateAccelerationAndEnergyChangeRate(const std::vector<Particle>& neighbors,
                                                  const Kernel& kernel,
                                                  const DissipationTerms& dissipation,
                                                  const EquationOfState& equationOfState);

    void updateTotalSpecificEnergy();

    // Variables primitivas
    double density;                    ///< Densidad de la partícula (no relativista).
    double pressure;                   ///< Presión de la partícula.
    double specificInternalEnergy;     ///< Energía interna específica u.
    std::array<double, 3> velocity;    ///< Velocidad (vx, vy, vz).

    // Variables conservadas (para compatibilidad con solvers relativistas)
    double baryonDensity;                   ///< Baryon density (gamma * n).
    std::array<double, 3> specificMomentum; ///< Momento específico (S_i).
    double specificEnergy;                  ///< Energía específica conservada (\hat{e}).

    //Otras variables típicas del SPH
    double mass;                        //< Masa de la partícula.
    double totalSpecificEnergy;         //< Energía específica total (u + KE).
    double h;                           //< Longitud de suavizado (kernel radius).
    double Omega;                       //< Factor para corrección en SPH (adimensional).
    std::array<double, 3> position;     //< Posición de la partícula.
    std::array<double, 3> acceleration; //< Aceleración (dv/dt).
    double energyChangeRate;            //< Tasa de cambio de la energía interna (du/dt).
    mutable std::string conv_h;

    /**
     * @brief Sobrecarga del operador == para comparar partículas.
     * @param other  Otra partícula con la que comparar.
     * @return true si las partículas tienen la misma posición.
     */
    bool operator==(const Particle& other) const {
        return position == other.position;
    }

};

#endif // PARTICLE_H
