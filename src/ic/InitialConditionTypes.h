#ifndef INITIALCONDITIONTYPES_H
#define INITIALCONDITIONTYPES_H

/**
 * Enumeración centralizada de todas las condiciones
 * iniciales disponibles.  Añade aquí nuevas entradas
 * cuando implementes más clases.
 */
enum class InitialConditionType
{
    // ――― 1‑D shock tubes ―――
    TEST_RSOD,        ///< Sod relativista (moderado)
    TEST_RSOD2,       ///< Sod relativista (fuerte)
    TEST_RSOD3,       ///< Sod relativista (muy fuerte)
    TEST_SB,          ///< Blast wave 1‑D (pL » pR)
    TEST_PERTUB_SIN,  ///< Perturbación sinusoidal en densidad
    TEST_TRANS_VEL,   ///< Shock tube con velocidad transversal
    NR_SOD,           ///< Sod no‑relativista

    // ――― 2‑D/3‑D (place‑holders) ―――
    KELVIN_HELMHOLTZ  ///< Cizalla Kelvin–Helmholtz (a implementar)
};

#endif // INITIALCONDITIONTYPES_H
