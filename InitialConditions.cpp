
#include "InitialConditions.h"
#include <cmath>
#include "DensityUpdater.h"
#include <iostream>
#include "VariableConverter.h"


InitialConditions::InitialConditions() : icType(InitialConditionType::TEST_SB) {}

void InitialConditions::setInitialConditionType(InitialConditionType type) {
    icType = type;
}

void InitialConditions::initializeParticles(std::vector<Particle>& particles,
                                            std::shared_ptr<Kernel> kernel,
                                            std::shared_ptr<EquationOfState> eos,
                                            int N,
                                            double x_min,
                                            double x_max) {
    double GAMMA = eos->getGamma();
    DensityUpdater densityUpdater(1.2, 1e-6, false, 0.1);
    VariableConverter converter(1e-10);

    switch (icType) {
         
     case InitialConditionType::TEST_RSOD: {
        // ---- PARÁMETROS DEL PROBLEMA SOD ----
        const double density_left = 10.0;
        const double density_right = 1.0;
        const double pressure_left = 40.0 / 3.0;
        const double pressure_right = 10.0e-6;
        const double x_discontinuity = 0.5 * (x_min + x_max);

        // ---- EXTENDER DOMINIO ----
        const double x_min_extended = x_min - 0.2 * (x_max - x_min);
        const double x_max_extended = x_max + 0.2 * (x_max - x_min);
        const double extended_volume_left = x_discontinuity - x_min_extended;
        const double extended_volume_right = x_max_extended - x_discontinuity;

        // ---- CÁLCULO DE DISTRIBUCIÓN ----
        const double mass_left = density_left * extended_volume_left;
        const double mass_right = density_right * extended_volume_right;
        const double total_mass = mass_left + mass_right;

        int N_left = static_cast<int>(N * mass_left / total_mass);
        int N_right = N - N_left;

        // Ajuste de un 20% adicional en cada lado
        N_left  += static_cast<int>(0.2*N_left);
        N_right += static_cast<int>(0.2*N_right);

        const double dx_left = extended_volume_left / N_left;
        const double dx_right = extended_volume_right / N_right;
        const double mass_per_particle = total_mass / (N_left + N_right);
        std::cout << " spacing left: " << dx_left << "\n spacing right: " << dx_right << std::endl;

        // Elegimos un ancho de transición
        double avg_dx = 0.5 * (dx_left + dx_right);
        double transition_length = 1.0 * avg_dx;

        // Función suave para densidad:
        auto smoothDensity = [&](double x) {
            return (density_left - density_right)
                / (1.0 + std::exp((x - x_discontinuity) / transition_length))
                + density_right;
        };

        // Función suave para presión:
        auto smoothPressure = [&](double x) {
            return (pressure_left - pressure_right)
                / (1.0 + std::exp((x - x_discontinuity) / transition_length))
                + pressure_right;
        };


        // ---- CREAR TODAS LAS PARTICULAS COMO REALES ----
        auto createParticle = [&](double x, double density, double pressure, double h) {
            std::array<double, 3> position = {x, 0.0, 0.0};
            std::array<double, 3> velocity = {0.0, 0.0, 0.0};
            double u = pressure / ((GAMMA - 1.0) * density);
            Particle p(position, velocity, mass_per_particle, u);
            p.density = density;
            p.pressure = pressure;
            p.h = h;
            p.baryonDensity = density;
            //p.specificEnergy = 1.0 + u;
            return p;
        };

        // Partículas en el lado izquierdo (incluyendo dominio extendido)
        for (int i = 0; i < N_left; ++i) {
            double x = x_min_extended + (i + 0.5) * dx_left;
            double dens = smoothDensity(x); //density_left;//(x);
            double pres = smoothPressure(x); //pressure_left;//(x);

            particles.push_back(createParticle(x, dens, pres, 1.4 * mass_per_particle/density_left));
        }

        // Partículas en el lado derecho (incluyendo dominio extendido)
        for (int i = 0; i < N_right; ++i) {
            double x = x_discontinuity + (i + 0.5) * dx_right;
            double dens = smoothDensity(x);
            double pres = smoothPressure(x);

            particles.push_back(createParticle(x, dens, pres, 1.4 * mass_per_particle/density_right));
        }

        // ---- CÁLCULO DE PROPIEDADES FÍSICAS ----
        //#pragma omp parallel for
        for (size_t i = 0; i < particles.size(); ++i) {
            auto& p = particles[i];
            p.updateDensity(particles, densityUpdater, *kernel);
            p.updatePressure(*eos);
            converter.primitivesToConserved(p, *eos);
        }

        // ---- MARCAR PARTICULAS COMO GHOST ----
        const int numGhostLeft = static_cast<int>(std::ceil(N_left * 0.2));
        const int numGhostRight = static_cast<int>(std::ceil(N_right * 0.2));

        // Marcar partículas fantasma desde el extremo izquierdo
        for (int i = 0; i < numGhostLeft; ++i) {
            particles[i].isGhost = true;
        }
        // Marcar partículas fantasma desde el extremo derecho
        for (int i = 0; i < numGhostRight; ++i) {
            particles[particles.size() - 1 - i].isGhost = true;
        }

        break;
    }  

     case InitialConditionType::TEST_RSOD2: {
        // ---- PARÁMETROS DEL PROBLEMA SOD ----
        const double density_left = 10.0;
        const double density_right = 1.0;
        const double pressure_left = 4000.0 / 3.0;
        const double pressure_right = 10.0e-6;
        const double x_discontinuity = 0.5 * (x_min + x_max);

        // ---- EXTENDER DOMINIO ----
        const double x_min_extended = x_min - 0.2 * (x_max - x_min);
        const double x_max_extended = x_max + 0.2 * (x_max - x_min);
        const double extended_volume_left = x_discontinuity - x_min_extended;
        const double extended_volume_right = x_max_extended - x_discontinuity;

        // ---- CÁLCULO DE DISTRIBUCIÓN ----
        const double mass_left = density_left * extended_volume_left;
        const double mass_right = density_right * extended_volume_right;
        const double total_mass = mass_left + mass_right;

        int N_left = static_cast<int>(N * mass_left / total_mass);
        int N_right = N - N_left;

        // Ajuste de un 20% adicional en cada lado
        N_left  += static_cast<int>(0.2*N_left);
        N_right += static_cast<int>(0.2*N_right);

        const double dx_left = extended_volume_left / N_left;
        const double dx_right = extended_volume_right / N_right;
        const double mass_per_particle = total_mass / (N_left + N_right);
        std::cout << " spacing left: " << dx_left << "\n spacing right: " << dx_right << std::endl;

        // Elegimos un ancho de transición, por ejemplo la mitad
        double avg_dx = 0.5 * (dx_left + dx_right);
        double transition_length = 1.0* avg_dx; // Ajusta según te convenga

        // Función suave para densidad:
        auto smoothDensity = [&](double x) {
            return (density_left - density_right)
                / (1.0 + std::exp((x - x_discontinuity) / transition_length))
                + density_right;
        };

        // Función suave para presión:
        auto smoothPressure = [&](double x) {
            return (pressure_left - pressure_right)
                / (1.0 + std::exp((x - x_discontinuity) / transition_length))
                + pressure_right;
        };


        // ---- CREAR TODAS LAS PARTICULAS COMO REALES ----
        auto createParticle = [&](double x, double density, double pressure, double h) {
            std::array<double, 3> position = {x, 0.0, 0.0};
            std::array<double, 3> velocity = {0.0, 0.0, 0.0};
            double u = pressure / ((GAMMA - 1.0) * density);
            Particle p(position, velocity, mass_per_particle, u);
            p.density = density;
            p.pressure = pressure;
            p.h = h;
            p.baryonDensity = density;
            //p.specificEnergy = 1.0 + u;
            return p;
        };

        // Partículas en el lado izquierdo (incluyendo dominio extendido)
        for (int i = 0; i < N_left; ++i) {
            double x = x_min_extended + (i + 0.5) * dx_left;
            double dens = smoothDensity(x); //density_left;//(x);
            double pres = smoothPressure(x); //pressure_left;//(x);

            particles.push_back(createParticle(x, dens, pres, 1.2 * mass_per_particle/density_left));
        }

        // Partículas en el lado derecho (incluyendo dominio extendido)
        for (int i = 0; i < N_right; ++i) {
            double x = x_discontinuity + (i + 0.5) * dx_right;
            double dens = smoothDensity(x);
            double pres = smoothPressure(x);

            particles.push_back(createParticle(x, dens, pres, 1.2 * mass_per_particle/density_right));
        }

        // ---- CÁLCULO DE PROPIEDADES FÍSICAS ----
        //#pragma omp parallel for
        for (size_t i = 0; i < particles.size(); ++i) {
            auto& p = particles[i];
            p.updateDensity(particles, densityUpdater, *kernel);
            p.updatePressure(*eos);
            converter.primitivesToConserved(p, *eos);
        }

        // ---- MARCAR PARTICULAS COMO GHOST ----
        const int numGhostLeft = static_cast<int>(std::ceil(N_left * 0.2));
        const int numGhostRight = static_cast<int>(std::ceil(N_right * 0.2));

        // Marcar partículas fantasma desde el extremo izquierdo
        for (int i = 0; i < numGhostLeft; ++i) {
            particles[i].isGhost = true;
        }
        // Marcar partículas fantasma desde el extremo derecho
        for (int i = 0; i < numGhostRight; ++i) {
            particles[particles.size() - 1 - i].isGhost = true;
        }

        break;
    }  

     case InitialConditionType::TEST_RSOD3: {
        // ---- PARÁMETROS DEL PROBLEMA SOD ----
        const double density_left = 10.0;
        const double density_right = 1.0;
        const double pressure_left = 40000.0 / 3.0;
        const double pressure_right = 10.0e-6;
        const double x_discontinuity = 0.5 * (x_min + x_max);

        // ---- EXTENDER DOMINIO ----
        const double x_min_extended = x_min - 0.2 * (x_max - x_min);
        const double x_max_extended = x_max + 0.2 * (x_max - x_min);
        const double extended_volume_left = x_discontinuity - x_min_extended;
        const double extended_volume_right = x_max_extended - x_discontinuity;

        // ---- CÁLCULO DE DISTRIBUCIÓN ----
        const double mass_left = density_left * extended_volume_left;
        const double mass_right = density_right * extended_volume_right;
        const double total_mass = mass_left + mass_right;

        int N_left = static_cast<int>(N * mass_left / total_mass);
        int N_right = N - N_left;

        // Ajuste de un 20% adicional en cada lado
        N_left  += static_cast<int>(0.2*N_left);
        N_right += static_cast<int>(0.2*N_right);

        const double dx_left = extended_volume_left / N_left;
        const double dx_right = extended_volume_right / N_right;
        const double mass_per_particle = total_mass / (N_left + N_right);
        std::cout << " spacing left: " << dx_left << "\n spacing right: " << dx_right << std::endl;

        // Elegimos un ancho de transición, por ejemplo la mitad
        double avg_dx = 0.5 * (dx_left + dx_right);
        double transition_length = 1.0 * avg_dx; // Ajusta según te convenga

        // Función suave para densidad:
        auto smoothDensity = [&](double x) {
            return (density_left - density_right)
                / (1.0 + std::exp((x - x_discontinuity) / transition_length))
                + density_right;
        };

        // Función suave para presión:
        auto smoothPressure = [&](double x) {
            return (pressure_left - pressure_right)
                / (1.0 + std::exp((x - x_discontinuity) / transition_length))
                + pressure_right;
        };


        // ---- CREAR TODAS LAS PARTICULAS COMO REALES ----
        auto createParticle = [&](double x, double density, double pressure, double h) {
            std::array<double, 3> position = {x, 0.0, 0.0};
            std::array<double, 3> velocity = {0.0, 0.0, 0.0};
            double u = pressure / ((GAMMA - 1.0) * density);
            Particle p(position, velocity, mass_per_particle, u);
            p.density = density;
            p.pressure = pressure;
            p.h = h;
            p.baryonDensity = density;
            //p.specificEnergy = 1.0 + u;
            return p;
        };

        // Partículas en el lado izquierdo (incluyendo dominio extendido)
        for (int i = 0; i < N_left; ++i) {
            double x = x_min_extended + (i + 0.5) * dx_left;
            double dens = smoothDensity(x); //density_left;//(x);
            double pres = smoothPressure(x); //pressure_left;//(x);

            particles.push_back(createParticle(x, dens, pres, 1.2 * mass_per_particle/density_left));
        }

        // Partículas en el lado derecho (incluyendo dominio extendido)
        for (int i = 0; i < N_right; ++i) {
            double x = x_discontinuity + (i + 0.5) * dx_right;
            double dens = smoothDensity(x);
            double pres = smoothPressure(x);

            particles.push_back(createParticle(x, dens, pres, 1.2 * mass_per_particle/density_right));
        }

        // ---- CÁLCULO DE PROPIEDADES FÍSICAS ----
        //#pragma omp parallel for
        for (size_t i = 0; i < particles.size(); ++i) {
            auto& p = particles[i];
            p.updateDensity(particles, densityUpdater, *kernel);
            p.updatePressure(*eos);
            converter.primitivesToConserved(p, *eos);
        }

        // ---- MARCAR PARTICULAS COMO GHOST ----
        const int numGhostLeft = static_cast<int>(std::ceil(N_left * 0.2));
        const int numGhostRight = static_cast<int>(std::ceil(N_right * 0.2));

        // Marcar partículas fantasma desde el extremo izquierdo
        for (int i = 0; i < numGhostLeft; ++i) {
            particles[i].isGhost = true;
        }
        // Marcar partículas fantasma desde el extremo derecho
        for (int i = 0; i < numGhostRight; ++i) {
            particles[particles.size() - 1 - i].isGhost = true;
        }

        break;
    }  

     case InitialConditionType::TEST_SB: {
        // ---- PARÁMETROS DEL PROBLEMA SOD ----
        const double density_left = 1.0;
        const double density_right = 1.0;
        const double pressure_left = 1000.0;
        const double pressure_right = 0.01;
        const double x_discontinuity = 0.5 * (x_min + x_max);

        // ---- EXTENDER DOMINIO ----
        const double x_min_extended = x_min - 0.2 * (x_max - x_min);
        const double x_max_extended = x_max + 0.2 * (x_max - x_min);
        const double extended_volume_left = x_discontinuity - x_min_extended;
        const double extended_volume_right = x_max_extended - x_discontinuity;

        // ---- CÁLCULO DE DISTRIBUCIÓN ----
        const double mass_left = density_left * extended_volume_left;
        const double mass_right = density_right * extended_volume_right;
        const double total_mass = mass_left + mass_right;

        int N_left = static_cast<int>(N * mass_left / total_mass);
        int N_right = N - N_left;

        // Ajuste de un 20% adicional en cada lado
        N_left  += static_cast<int>(0.2*N_left);
        N_right += static_cast<int>(0.2*N_right);

        const double dx_left = extended_volume_left / N_left;
        const double dx_right = extended_volume_right / N_right;
        const double mass_per_particle = total_mass / (N_left + N_right);
        std::cout << " spacing left: " << dx_left << "\n spacing right: " << dx_right << std::endl;

        // Elegimos un ancho de transición, por ejemplo la mitad
        double avg_dx = 0.5 * (dx_left + dx_right);
        double transition_length = 1.0 * avg_dx; 

        // Función suave para densidad:
        auto smoothDensity = [&](double x) {
            return (density_left - density_right)
                / (1.0 + std::exp((x - x_discontinuity) / transition_length))
                + density_right;
        };

        // Función suave para presión:
        auto smoothPressure = [&](double x) {
            return (pressure_left - pressure_right)
                / (1.0 + std::exp((x - x_discontinuity) / transition_length))
                + pressure_right;
        };


        // ---- CREAR TODAS LAS PARTICULAS COMO REALES ----
        auto createParticle = [&](double x, double density, double pressure, double h) {
            std::array<double, 3> position = {x, 0.0, 0.0};
            std::array<double, 3> velocity = {0.0, 0.0, 0.0};
            double u = pressure / ((GAMMA - 1.0) * density);
            Particle p(position, velocity, mass_per_particle, u);
            p.density = density;
            p.pressure = pressure;
            p.h = h;
            p.baryonDensity = density;
            //p.specificEnergy = 1.0 + u;
            return p;
        };

        // Partículas en el lado izquierdo (incluyendo dominio extendido)
        for (int i = 0; i < N_left; ++i) {
            double x = x_min_extended + (i + 0.5) * dx_left;
            double dens = smoothDensity(x); //density_left;//(x);
            double pres = smoothPressure(x); //pressure_left;//(x);

            particles.push_back(createParticle(x, dens, pres, 1.4 * mass_per_particle/density_left));
        }

        // Partículas en el lado derecho (incluyendo dominio extendido)
        for (int i = 0; i < N_right; ++i) {
            double x = x_discontinuity + (i + 0.5) * dx_right;
            double dens = smoothDensity(x);
            double pres = smoothPressure(x);

            particles.push_back(createParticle(x, dens, pres, 1.4 * mass_per_particle/density_right));
        }

        // ---- CÁLCULO DE PROPIEDADES FÍSICAS ----
        //#pragma omp parallel for
        for (size_t i = 0; i < particles.size(); ++i) {
            auto& p = particles[i];
            p.updateDensity(particles, densityUpdater, *kernel);
            p.updatePressure(*eos);
            converter.primitivesToConserved(p, *eos);
        }

        // ---- MARCAR PARTICULAS COMO GHOST ----
        const int numGhostLeft = static_cast<int>(std::ceil(N_left * 0.2));
        const int numGhostRight = static_cast<int>(std::ceil(N_right * 0.2));

        // Marcar partículas fantasma desde el extremo izquierdo
        for (int i = 0; i < numGhostLeft; ++i) {
            particles[i].isGhost = true;
        }
        // Marcar partículas fantasma desde el extremo derecho
        for (int i = 0; i < numGhostRight; ++i) {
            particles[particles.size() - 1 - i].isGhost = true;
        }

        break;
    }  
         
     case InitialConditionType::TEST_PERTUB_SIN: {
        // ---- PARÁMETROS DEL PROBLEMA SOD ----
        const double density_left = 5.0;
        const double density_right = 2.3;
        const double pressure_left = 50.0;
        const double pressure_right = 5.0;
        const double x_discontinuity = 0.5 * (x_min + x_max);

        // ---- EXTENDER DOMINIO ----
        const double x_min_extended = x_min - 0.2 * (x_max - x_min);
        const double x_max_extended = x_max + 0.2 * (x_max - x_min);
        const double extended_volume_left = x_discontinuity - x_min_extended;
        const double extended_volume_right = x_max_extended - x_discontinuity;

        // ---- CÁLCULO DE DISTRIBUCIÓN ----
        const double mass_left = density_left * extended_volume_left;
        const double mass_right = density_right * extended_volume_right;
        const double total_mass = mass_left + mass_right;

        int N_left = static_cast<int>(N * mass_left / total_mass);
        int N_right = N - N_left;

        // Ajuste de un 20% adicional en cada lado
        N_left  += static_cast<int>(0.2*N_left);
        N_right += static_cast<int>(0.2*N_right);

        const double dx_left = extended_volume_left / N_left;
        const double dx_right = extended_volume_right / N_right;
        const double mass_per_particle = total_mass / (N_left + N_right);
        std::cout << " spacing left: " << dx_left << "\n spacing right: " << dx_right << std::endl;

        // Elegimos un ancho de transición, por ejemplo la mitad
        double avg_dx = 0.5 * (dx_left + dx_right);
        double transition_length = 1.0* avg_dx; // Ajusta según te convenga

        // Función suave para densidad:
/*         auto smoothDensity = [&](double x) {
            return (density_left - density_right)
                / (1.0 + std::exp((x - x_discontinuity) / transition_length))
                + density_right;
        }; */

        // Función suave para presión:
/*         auto smoothPressure = [&](double x) {
            return (pressure_left - pressure_right)
                / (1.0 + std::exp((x - x_discontinuity) / transition_length))
                + pressure_right;
        };
 */

        // ---- CREAR TODAS LAS PARTICULAS COMO REALES ----
        auto createParticle = [&](double x, double density, double pressure, double h) {
            std::array<double, 3> position = {x, 0.0, 0.0};
            std::array<double, 3> velocity = {0.0, 0.0, 0.0};
            double u = pressure / ((GAMMA - 1.0) * density);
            Particle p(position, velocity, mass_per_particle, u);
            p.density = density;
            p.pressure = pressure;
            p.h = h;
            p.baryonDensity = density;
            //p.specificEnergy = 1.0 + u;
            return p;
        };

        // Partículas en el lado izquierdo (incluyendo dominio extendido)
        for (int i = 0; i < N_left; ++i) {
            double x = x_min_extended + (i + 0.5) * dx_left;
            double dens = density_left;//smoothDensity(x);
            double pres = pressure_left; //pressure_left;//(x);

            particles.push_back(createParticle(x, dens, pres, 1.2 * mass_per_particle/density_left));
        }

        // Partículas en el lado derecho (incluyendo dominio extendido)
        for (int i = 0; i < N_right; ++i) {
            double x = x_discontinuity + (i + 0.5) * dx_right;
            double dens = 2.0 + 0.3 * std::sin( 50.0 * x);
            double pres = pressure_right;

            particles.push_back(createParticle(x, dens, pres, 1.2 * mass_per_particle/density_right));
        }

        // ---- CÁLCULO DE PROPIEDADES FÍSICAS ----
        //#pragma omp parallel for
        for (size_t i = 0; i < particles.size(); ++i) {
            auto& p = particles[i];
            p.updateDensity(particles, densityUpdater, *kernel);
            p.updatePressure(*eos);
            converter.primitivesToConserved(p, *eos);
        }

        // ---- MARCAR PARTICULAS COMO GHOST ----
        const int numGhostLeft = static_cast<int>(std::ceil(N_left * 0.2));
        const int numGhostRight = static_cast<int>(std::ceil(N_right * 0.2));

        // Marcar partículas fantasma desde el extremo izquierdo
        for (int i = 0; i < numGhostLeft; ++i) {
            particles[i].isGhost = true;
        }
        // Marcar partículas fantasma desde el extremo derecho
        for (int i = 0; i < numGhostRight; ++i) {
            particles[particles.size() - 1 - i].isGhost = true;
        }

        break;
    }  
      
     case InitialConditionType::TEST_TRANS_VEL: {
        // ---- PARÁMETROS DEL PROBLEMA SOD ----
        const double density_left = 1.0;
        const double density_right = 1.0;
        const double pressure_left = 1000.0;
        const double pressure_right = 10.0e-2;
        const double x_discontinuity = 0.5 * (x_min + x_max);

        // ---- EXTENDER DOMINIO ----
        const double x_min_extended = x_min - 0.2 * (x_max - x_min);
        const double x_max_extended = x_max + 0.2 * (x_max - x_min);
        const double extended_volume_left = x_discontinuity - x_min_extended;
        const double extended_volume_right = x_max_extended - x_discontinuity;

        // ---- CÁLCULO DE DISTRIBUCIÓN ----
        const double mass_left = density_left * extended_volume_left;
        const double mass_right = density_right * extended_volume_right;
        const double total_mass = mass_left + mass_right;

        int N_left = static_cast<int>(N * mass_left / total_mass);
        int N_right = N - N_left;

        // Ajuste de un 20% adicional en cada lado
        N_left  += static_cast<int>(0.2*N_left);
        N_right += static_cast<int>(0.2*N_right);

        const double dx_left = extended_volume_left / N_left;
        const double dx_right = extended_volume_right / N_right;
        const double mass_per_particle = total_mass / (N_left + N_right);
        std::cout << " spacing left: " << dx_left << "\n spacing right: " << dx_right << std::endl;

        // Elegimos un ancho de transición, por ejemplo la mitad
        double avg_dx = 0.5 * (dx_left + dx_right);
        double transition_length = 0.5* avg_dx; // Ajusta según te convenga

        // Función suave para densidad:
         auto smoothDensity = [&](double x) {
            return (density_left - density_right)
                / (1.0 + std::exp((x - x_discontinuity) / transition_length))
                + density_right;
        }; 

        // Función suave para presión:
         auto smoothPressure = [&](double x) {
            return (pressure_left - pressure_right)
                / (1.0 + std::exp((x - x_discontinuity) / transition_length))
                + pressure_right;
        };
 

        // ---- CREAR TODAS LAS PARTICULAS COMO REALES ----
        auto createParticle = [&](double x, double density, double pressure, std::array<double, 3> vel, double h) {
            std::array<double, 3> position = {x, 0.0, 0.0};
            std::array<double, 3> velocity = vel;
            double u = pressure / ((GAMMA - 1.0) * density);
            Particle p(position, velocity, mass_per_particle, u);
            p.density = density;
            p.pressure = pressure;
            p.h = h;
            p.baryonDensity = density;
            //p.specificEnergy = 1.0 + u;
            return p;
        };

        // Partículas en el lado izquierdo (incluyendo dominio extendido)
        for (int i = 0; i < N_left; ++i) {
            double x = x_min_extended + (i + 0.5) * dx_left;
            double dens = smoothDensity(x);//smoothDensity(x);
            double pres = smoothPressure(x); //pressure_left;//(x);
            std::array<double, 3> v = {0.0 , 0.0, 0.0 }; 
            particles.push_back(createParticle(x, dens, pres, v,1.2 * mass_per_particle/density_left));
        }

        // Partículas en el lado derecho (incluyendo dominio extendido)
        for (int i = 0; i < N_right; ++i) {
            double x = x_discontinuity + (i + 0.5) * dx_right;
            double dens = smoothDensity(x);
            double pres = smoothPressure(x);
            std::array<double, 3> v = {0.0 , 0.99, 0.0 }; 
            particles.push_back(createParticle(x, dens, pres, v, 1.2 * mass_per_particle/density_right));
        }

        // ---- CÁLCULO DE PROPIEDADES FÍSICAS ----
        //#pragma omp parallel for
        for (size_t i = 0; i < particles.size(); ++i) {
            auto& p = particles[i];
            p.updateDensity(particles, densityUpdater, *kernel);
            p.updatePressure(*eos);
            converter.primitivesToConserved(p, *eos);
        }

        // ---- MARCAR PARTICULAS COMO GHOST ----
        const int numGhostLeft = static_cast<int>(std::ceil(N_left * 0.2));
        const int numGhostRight = static_cast<int>(std::ceil(N_right * 0.2));

        // Marcar partículas fantasma desde el extremo izquierdo
        for (int i = 0; i < numGhostLeft; ++i) {
            particles[i].isGhost = true;
        }
        // Marcar partículas fantasma desde el extremo derecho
        for (int i = 0; i < numGhostRight; ++i) {
            particles[particles.size() - 1 - i].isGhost = true;
        }

        break;
    }  
         
     case InitialConditionType::NR_SOD: {
        // ---- PARÁMETROS DEL PROBLEMA SOD ----
        const double density_left = 1.0;
        const double density_right = 0.125;
        const double pressure_left = 1.0;
        const double pressure_right = 0.1;
        const double x_discontinuity = 0.5 * (x_min + x_max);

        // ---- EXTENDER DOMINIO ----
        const double x_min_extended = x_min - 0.2 * (x_max - x_min);
        const double x_max_extended = x_max + 0.2 * (x_max - x_min);
        const double extended_volume_left = x_discontinuity - x_min_extended;
        const double extended_volume_right = x_max_extended - x_discontinuity;

        // ---- CÁLCULO DE DISTRIBUCIÓN ----
        const double mass_left = density_left * extended_volume_left;
        const double mass_right = density_right * extended_volume_right;
        const double total_mass = mass_left + mass_right;

        int N_left = static_cast<int>(N * mass_left / total_mass);
        int N_right = N - N_left;

        // Ajuste de un 20% adicional en cada lado
        N_left  += static_cast<int>(0.2*N_left);
        N_right += static_cast<int>(0.2*N_right);

        const double dx_left = extended_volume_left / N_left;
        const double dx_right = extended_volume_right / N_right;
        const double mass_per_particle = total_mass / (N_left + N_right);
        std::cout << " spacing left: " << dx_left << "\n spacing right: " << dx_right << std::endl;

        // Elegimos un ancho de transición, por ejemplo la mitad
        double avg_dx = 0.5 * (dx_left + dx_right);
        double transition_length = 1.0* avg_dx; // Ajusta según te convenga

        // Función suave para densidad:
        auto smoothDensity = [&](double x) {
            return (density_left - density_right)
                / (1.0 + std::exp((x - x_discontinuity) / transition_length))
                + density_right;
        };

        // Función suave para presión:
        auto smoothPressure = [&](double x) {
            return (pressure_left - pressure_right)
                / (1.0 + std::exp((x - x_discontinuity) / transition_length))
                + pressure_right;
        };


        // ---- CREAR TODAS LAS PARTICULAS COMO REALES ----
        auto createParticle = [&](double x, double density, double pressure, double h) {
            std::array<double, 3> position = {x, 0.0, 0.0};
            std::array<double, 3> velocity = {0.0, 0.0, 0.0};
            double u = pressure / ((GAMMA - 1.0) * density);
            Particle p(position, velocity, mass_per_particle, u);
            p.density = density;
            p.pressure = pressure;
            p.h = h;
            p.baryonDensity = density;
            //p.specificEnergy = 1.0 + u;
            return p;
        };

        // Partículas en el lado izquierdo (incluyendo dominio extendido)
        for (int i = 0; i < N_left; ++i) {
            double x = x_min_extended + (i + 0.5) * dx_left;
            double dens = smoothDensity(x); //density_left;//(x);
            double pres = smoothPressure(x); //pressure_left;//(x);

            particles.push_back(createParticle(x, dens, pres, 1.2 * mass_per_particle/density_left));
        }

        // Partículas en el lado derecho (incluyendo dominio extendido)
        for (int i = 0; i < N_right; ++i) {
            double x = x_discontinuity + (i + 0.5) * dx_right;
            double dens = smoothDensity(x);
            double pres = smoothPressure(x);

            particles.push_back(createParticle(x, dens, pres, 1.2 * mass_per_particle/density_right));
        }

        // ---- CÁLCULO DE PROPIEDADES FÍSICAS ----
        //#pragma omp parallel for
        for (size_t i = 0; i < particles.size(); ++i) {
            auto& p = particles[i];
            p.updateDensity(particles, densityUpdater, *kernel);
            p.updatePressure(*eos);
            converter.primitivesToConserved(p, *eos);
        }

        // ---- MARCAR PARTICULAS COMO GHOST ----
        const int numGhostLeft = static_cast<int>(std::ceil(N_left * 0.2));
        const int numGhostRight = static_cast<int>(std::ceil(N_right * 0.2));

        // Marcar partículas fantasma desde el extremo izquierdo
        for (int i = 0; i < numGhostLeft; ++i) {
            particles[i].isGhost = true;
        }
        // Marcar partículas fantasma desde el extremo derecho
        for (int i = 0; i < numGhostRight; ++i) {
            particles[particles.size() - 1 - i].isGhost = true;
        }

        break;
    }  

}

}
