#include "EquationOfState.h"
#include <cmath>
#include <algorithm>
#include <iostream>

EquationOfState::EquationOfState(double gammaVal)
    : GAMMA(gammaVal)
{}

double EquationOfState::getGamma() const {
    return GAMMA;
}

// ----------------------------------------------------------
// 1) Presión: P = (Gamma - 1) * rho * u
// ----------------------------------------------------------
double EquationOfState::calculatePressure(double density,
                                          double specificInternalEnergy) const
{
    double P = (GAMMA - 1.0) * density * specificInternalEnergy;
    if (P <= 0.0 ){
        std::cerr << "\t\t[calculatePressure: ] PRESION NEGATIVA-------- \tn: "<< density << "   \te:  "<< specificInternalEnergy <<"\n";
    }
    return P;
}

// ----------------------------------------------------------
// 2) Velocidad del sonido relativista (simplificada)
//    c_s^2 = Gamma * P / [ rho*(1 + u) + P ]
//    donde 'rho' = density, 'u' = specificInternalEnergy
// ----------------------------------------------------------
double EquationOfState::calculateSoundSpeed(double density,
                                            double specificInternalEnergy) const
{
    double numerator   = GAMMA * (GAMMA - 1.0) * specificInternalEnergy;
    double denominator = 1.0 + specificInternalEnergy * GAMMA;

    // Evitar divisiones por cero o valores negativos
    if (denominator < 1e-14) {
        denominator = 1e-14;
        std::cerr << "[EquationOfState::calculateSoundSpeed] Advertencia: denominador ajustado.\n";
    }

    double cs2 = numerator / denominator;
    if (cs2 < 0.0) {
        // Ajuste de seguridad si por redondeo cs2 sale negativo
        std::cerr << "[EquationOfState::calculateSoundSpeed] Advertencia: cs^2 negativo, ajustado a 1e-14.\n";
        cs2 = 1e-14;
        //std::cout <<"\tP : " << P << std::endl;

    }

    return std::sqrt(cs2);
}
