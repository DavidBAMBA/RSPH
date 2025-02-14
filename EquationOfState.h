#ifndef EQUATIONOFSTATE_H
#define EQUATIONOFSTATE_H

class EquationOfState {
public:
    // Constructor
    explicit EquationOfState(double gammaVal);

    // Devuelve el índice adiabático
    double getGamma() const;

    // Calcula la presión: P = (Gamma - 1) * density * specificInternalEnergy
    // (Forma clásica politrópica; ajústala si tienes una EOS distinta)
    double calculatePressure(double density, double specificInternalEnergy) const;

    // **Nuevo**: Calcula la velocidad del sonido relativista (aprox).
    // c_s^2 = Gamma * P / [ density * (1 + u + P/density ]
    double calculateSoundSpeed(double density, double specificInternalEnergy) const;

    double GAMMA; // Índice adiabático
};

#endif // EQUATIONOFSTATE_H
