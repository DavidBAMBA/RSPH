// DissipationTerms.cpp
#include "DissipationTerms.h"
#include "MathUtils.h"
#include <algorithm>
#include "Kernel.h"
#include "EquationOfState.h"

// Constructor
DissipationTerms::DissipationTerms(double alpha_value, double beta_value)
    : alphaAV(alpha_value), betaAV(beta_value) {}


std::pair<double, double> DissipationTerms::Chow1997Viscosity(const Particle& a, const Particle& b, const Kernel& kernel, const EquationOfState& equationOfState) const {
    // Vector de posición entre partículas
    std::array<double, 3> r_ab = {
        b.position[0] - a.position[0],
        b.position[1] - a.position[1],
        b.position[2] - a.position[2]
    };

    std::array<double, 3> v_ab = {
        a.velocity[0] - b.velocity[0],
        a.velocity[1] - b.velocity[1],
        a.velocity[2] - b.velocity[2]
    };

    std::array<double, 3> r_ab_hat = MathUtils::hatVector(r_ab); // Dirección normalizada

    // Variables conservadas de las partículas
    double N_a = a.baryonDensity;
    double N_b = b.baryonDensity;
    double N_mean = 0.5 * (N_a + N_b);

    // Proyección de la velocidad en la dirección de la línea de visión
    double v_star_a = MathUtils::dotProduct(a.velocity, r_ab_hat);
    double v_star_b = MathUtils::dotProduct(b.velocity, r_ab_hat);

    // Calcular el factor de Lorentz en la dirección de la línea de visión
    double gamma_star_a = 1.0 / std::sqrt(1.0 - v_star_a * v_star_a);
    double gamma_star_b = 1.0 / std::sqrt(1.0 - v_star_b * v_star_b);

    // Calcular q* = \gamma^* / \gamma q
    double entalpy_a = 1.0 + a.specificInternalEnergy + a.pressure / a.density;
    double entalpy_b  = 1.0 + b.specificInternalEnergy + b.pressure / b.density;
    std::array<double, 3 > q_a_star = { gamma_star_a * a.velocity[0] * entalpy_a, 
                                        gamma_star_a * a.velocity[1] * entalpy_a,  
                                        gamma_star_a * a.velocity[2] * entalpy_a};

    std::array<double, 3 > q_b_star = { gamma_star_b * b.velocity[0] * entalpy_b, 
                                        gamma_star_b * b.velocity[1] * entalpy_b,  
                                        gamma_star_b * b.velocity[2] * entalpy_b};

    // Verificar si las partículas se están acercando
    std::array<double, 3 > q_ab_star = {
        q_a_star[0] - q_b_star[0],
        q_a_star[1] - q_b_star[1],
        q_a_star[2] - q_b_star[2]
    };

    double q_ab_star_dot_j = MathUtils::dotProduct(q_ab_star, r_ab_hat);
    
    if (q_ab_star_dot_j <= 0.0) {
        return {0.0, 0.0}; // No hay disipación si no se acercan
    }

    // Velocidad de señal (promedio de las velocidades del sonido)
    double cs_a = equationOfState.calculateSoundSpeed(a.density, a.specificInternalEnergy);
    double cs_b = equationOfState.calculateSoundSpeed(b.density, b.specificInternalEnergy);
    double K = 1.0;

    double v_ab_star = - MathUtils::dotProduct(v_ab, r_ab_hat);

    double v_sig_a = ( cs_a + std::fabs(v_ab_star) ) / (1.0 + cs_a * std::fabs(v_ab_star)) ;
    double v_sig_b = ( cs_b + std::fabs(v_ab_star) ) / (1.0 + cs_b * std::fabs(v_ab_star)) ;
    double v_sig   = v_sig_a + v_sig_b + std::fabs(v_ab_star);

    // Término disipativo Pi_ab
    double Pi_ab = - K * v_sig * q_ab_star_dot_j / N_mean;
    
    
    // Calcular e* para cada partícula
    double e_a_star = gamma_star_a * entalpy_a - a.pressure / N_a;
    double e_b_star = gamma_star_b * entalpy_b - b.pressure / N_b;
    double Ome_ab   = - K * v_sig * (e_a_star - e_b_star) / N_mean;
    
    return {Pi_ab, Ome_ab};
}


std::pair<double, double> DissipationTerms::Viscosity_LiptaiPrice2018(const Particle& a, const Particle& b, const Kernel& kernel, const EquationOfState& equationOfState) const {
    // Vector de posición entre partículas
    std::array<double, 3> r_ab = {
        b.position[0] - a.position[0],
        b.position[1] - a.position[1],
        b.position[2] - a.position[2]
    };

    std::array<double, 3> v_ab = {
        a.velocity[0] - b.velocity[0],
        a.velocity[1] - b.velocity[1],
        a.velocity[2] - b.velocity[2]
    };

    std::array<double, 3> r_ab_hat = MathUtils::hatVector(r_ab); // Dirección normalizada

    double v_ab_hat = MathUtils::dotProduct(v_ab, r_ab_hat);

    if (v_ab_hat >= 0.0) {
        return {0.0, 0.0}; // No hay disipación si no se acercan
    }

    // Variables conservadas de las partículas
    double N_a = a.baryonDensity;
    double N_b = b.baryonDensity;

    // Proyección de la velocidad en la dirección de la línea de visión
    double v_star_a = MathUtils::dotProduct(a.velocity, r_ab_hat);
    double v_star_b = MathUtils::dotProduct(b.velocity, r_ab_hat);

    // Calcular el factor de Lorentz en la dirección de la línea de visión
    double gamma_star_a = 1.0 / std::sqrt(1.0 - v_star_a * v_star_a);
    double gamma_star_b = 1.0 / std::sqrt(1.0 - v_star_b * v_star_b);

    // Calcular q* = \gamma^* / \gamma q
    double entalpy_a = 1.0 + a.specificInternalEnergy + a.pressure / a.density;
    double entalpy_b  = 1.0 + b.specificInternalEnergy + b.pressure / b.density;

    double S_a_start = gamma_star_a * v_star_a;
    double S_b_start = gamma_star_b * v_star_b;
    double S_ab_start = S_a_start - S_b_start;

    // Velocidad de señal (promedio de las velocidades del sonido)
    //double cs_a = equationOfState.calculateSoundSpeed(a.density, a.specificInternalEnergy);
    //double cs_b = equationOfState.calculateSoundSpeed(b.density, b.specificInternalEnergy);
    double cs_a = std::sqrt(a.pressure*(5.0/3.0) / (a.density* entalpy_a));
    double cs_b = std::sqrt(b.pressure*(5.0/3.0) / (b.density* entalpy_b));
    
    double v_ab_star =  (v_star_a - v_star_b) / (1.0 - v_star_a*v_star_b);
    double v_sig_a = ( cs_a + std::fabs(v_ab_star) ) / (1.0 + cs_a * std::fabs(v_ab_star)) ;
    double v_sig_b = ( cs_b + std::fabs(v_ab_star) ) / (1.0 + cs_b * std::fabs(v_ab_star)) ;

    // Término disipativo Pi_ab
    double ALPHA = 0.5;
    double q_a =  0.5 * ALPHA * N_a * v_sig_a * entalpy_a * S_ab_start;
    double q_b =  0.5 * ALPHA * N_b * v_sig_b * entalpy_b * S_ab_start;
        
    return {q_a, q_b};
}


std::pair<double, double> DissipationTerms::Conductivity_LiptaiPrice2018(const Particle& a, const Particle& b, const Kernel& kernel, const EquationOfState& equationOfState) const {
    // Vector de posición entre partículas
    std::array<double, 3> r_ab = {
        b.position[0] - a.position[0],
        b.position[1] - a.position[1],
        b.position[2] - a.position[2]
    };

    std::array<double, 3> v_ab = {
        a.velocity[0] - b.velocity[0],
        a.velocity[1] - b.velocity[1],
        a.velocity[2] - b.velocity[2]
    };

    std::array<double, 3> r_ab_hat = MathUtils::hatVector(r_ab); // Dirección normalizada

    double v_ab_dot_r_ab_hat = MathUtils::dotProduct(v_ab, r_ab_hat);

    if (v_ab_dot_r_ab_hat >= 0.0) {
        return {0.0, 0.0}; // No hay disipación si no se acercan
    }

    // Variables conservadas de las partículas
    double N_a = a.baryonDensity;
    double N_b = b.baryonDensity;

    // Proyección de la velocidad en la dirección de la línea de visión
    double v_star_a = MathUtils::dotProduct(a.velocity, r_ab_hat);
    double v_star_b = MathUtils::dotProduct(b.velocity, r_ab_hat);

    // Calcular el factor de Lorentz en la dirección de la línea de visión
    double gamma_a = 1.0 / std::sqrt(1.0 - MathUtils::vectorNormSquared(a.velocity));
    double gamma_b = 1.0 / std::sqrt(1.0 - MathUtils::vectorNormSquared(b.velocity));
    double u_moña_a = a.specificInternalEnergy / gamma_a;
    double u_moña_b = b.specificInternalEnergy / gamma_b;
    double u_moña_ab = u_moña_a - u_moña_b;

    // Velocidad de señal (promedio de las velocidades del sonido)
    double cs_a = equationOfState.calculateSoundSpeed(a.density, a.specificInternalEnergy);
    double cs_b = equationOfState.calculateSoundSpeed(b.density, b.specificInternalEnergy);
    double v_ab_star =  (v_star_a - v_star_b) / (1.0 - v_star_a*v_star_b);
    double v_sig_a = ( cs_a + std::fabs(v_ab_star) ) / (1.0 + cs_a * std::fabs(v_ab_star)) ;
    double v_sig_b = ( cs_b + std::fabs(v_ab_star) ) / (1.0 + cs_b * std::fabs(v_ab_star)) ;

    double v_sig_u = std::min(1.0, std::sqrt( std::fabs(a.pressure - b.pressure)/ (0.5 * (a.density + b.density))) );
    double ALPHA_u = 0.2;
    double Omega_a = 0.5 * ALPHA_u * u_moña_ab * v_sig_a / N_a ;
    double Omega_b = 0.5 * ALPHA_u * u_moña_ab * v_sig_b / N_b ;

        
    return {Omega_a, Omega_b};
}

