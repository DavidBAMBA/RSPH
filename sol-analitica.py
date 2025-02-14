#!/usr/bin/env python3
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq  # Usamos brentq en lugar de newton

# =============================================================================
# Funciones auxiliares para hidrodinámica relativista
# =============================================================================
def lorentz_factor(v):
    """Factor de Lorentz: W = 1/sqrt(1-v^2)."""
    return 1.0/np.sqrt(1 - v**2)

def specific_enthalpy(rho, p, gamma):
    """
    Entalpía específica del gas ideal relativista:
      h = 1 + (γ/(γ-1)) (p/ρ)
    """
    return 1.0 + (gamma/(gamma-1))*p/rho

def sound_speed(rho, p, gamma):
    """
    Velocidad del sonido relativista:
      c_s² = γ p/(ρ h)
    donde h es la entalpía específica.
    """
    h = specific_enthalpy(rho, p, gamma)
    return np.sqrt(gamma*p/(rho*h))

# =============================================================================
# Funciones f(p) para cada lado del discontinuo (izquierdo o derecho)
# =============================================================================
def f_state(p, p_i, rho_i, v_i, gamma):
    """
    Función f_i(p) para el estado i (izquierdo o derecho).
    Si p <= p_i se tiene onda de rarefacción; si p > p_i, onda de choque.
    """
    W = lorentz_factor(v_i)
    h_i = specific_enthalpy(rho_i, p_i, gamma)
    c_si = sound_speed(rho_i, p_i, gamma)
    
    if p <= p_i:  # Onda de rarefacción
        term = (p/p_i)**((gamma-1)/(2*gamma))
        return (2 * c_si/(gamma-1)) * (term - 1) / W
    else:  # Onda de choque
        A_i = 2.0/((gamma+1)*rho_i)
        B_i = (gamma-1)/(gamma+1)*p_i
        return (p - p_i)*np.sqrt(A_i/(p + B_i)) / W

def pressure_function(p, pL, rhoL, vL, pR, rhoR, vR, gamma):
    """
    Función de la que se anula la suma de las contribuciones de los dos lados
    y la diferencia de velocidades:
      f_total(p) = f_state(p, pL, rhoL, vL, gamma) + f_state(p, pR, rhoR, vR, gamma) + (vR - vL)
    """
    return f_state(p, pL, rhoL, vL, gamma) + f_state(p, pR, rhoR, vR, gamma) + (vR - vL)

# =============================================================================
# Solución exacta relativista del problema de Riemann
# =============================================================================
def exact_riemann_solution_relativistic(rhoL, vL, pL, rhoR, vR, pR, gamma=5.0/3.0, x0=0.5, t=0.2):
    """
    Calcula la solución exacta del problema de Riemann relativista para condiciones
    iniciales arbitrarias (con v en [-1,1] y c=1). Se resuelve numéricamente la
    ecuación trascendental para p* y se calculan las velocidades en la región estrella,
    además de las posiciones de las ondas (rarefacción o choque).

    Devuelve un diccionario con:
      - p_star: presión en la región intermedia
      - v_star: velocidad en la región intermedia
      - rho_starL, rho_starR: densidades en la región intermedia (izquierda y derecha)
      - cL, cR: velocidades del sonido en los estados iniciales
      - Posiciones de las ondas:
            Para el lado izquierdo: 'x_head' y 'x_tail' (si rarefacción) o 'x_shock_L'
            Para el lado derecho: 'x_tail_R' (si rarefacción) o 'x_shock_R'
            y 'x_contact' para la discontinuidad de contacto.
    """
    # Velocidades del sonido en los estados iniciales:
    cL = sound_speed(rhoL, pL, gamma)
    cR = sound_speed(rhoR, pR, gamma)
    
    # --- Resolver la ecuación trascendental para p* ---
    # En lugar de usar newton (que falló), definimos un intervalo amplio
    p_min = 1e-8
    p_max = max(pL, pR) * 10.0  # Por ejemplo, 10 veces el mayor de pL y pR
    # Verificamos que la función cambie de signo en [p_min, p_max]
    f_min = pressure_function(p_min, pL, rhoL, vL, pR, rhoR, vR, gamma)
    f_max = pressure_function(p_max, pL, rhoL, vL, pR, rhoR, vR, gamma)
    if f_min * f_max > 0:
        raise RuntimeError(f"No se encontró cambio de signo en pressure_function en el intervalo [{p_min}, {p_max}]. f(p_min)={f_min}, f(p_max)={f_max}")
    # Se usa el método de Brent, que es robusto para ecuaciones trascendentales
    p_star = brentq(pressure_function, p_min, p_max, args=(pL, rhoL, vL, pR, rhoR, vR, gamma))
    
    # --- Calcular v* a partir de cualquiera de los lados ---
    fL = f_state(p_star, pL, rhoL, vL, gamma)
    fR = f_state(p_star, pR, rhoR, vR, gamma)
    v_star = 0.5*(vL + vR + fR - fL)
    
    # --- Calcular las densidades en la región "estrella" ---
    if p_star <= pL:
        rho_starL = rhoL*(p_star/pL)**(1/gamma)
    else:
        A_L = 2.0/((gamma+1)*rhoL)
        B_L = (gamma-1)/(gamma+1)*pL
        rho_starL = rhoL * ((p_star/pL + B_L/A_L) / (1 + B_L/A_L))
    
    if p_star <= pR:
        rho_starR = rhoR*(p_star/pR)**(1/gamma)
    else:
        A_R = 2.0/((gamma+1)*rhoR)
        B_R = (gamma-1)/(gamma+1)*pR
        rho_starR = rhoR * ((p_star/pR + B_R/A_R) / (1 + B_R/A_R))
    
    # --- Cálculo de las velocidades de propagación de las ondas ---
    wave_speeds = {}
    # LADO IZQUIERDO
    if p_star <= pL:
        # Onda de rarefacción izquierda
        s_head_L = (vL - cL)/(1 - vL*cL)
        h_starL  = specific_enthalpy(rho_starL, p_star, gamma)
        c_starL  = np.sqrt(gamma*p_star/(rho_starL*h_starL))
        s_tail_L = (v_star - c_starL)/(1 - v_star*c_starL)
        wave_speeds['x_head'] = x0 + s_head_L*t
        wave_speeds['x_tail'] = x0 + s_tail_L*t
    else:
        # Onda de choque izquierda
        W_L = lorentz_factor(vL)
        S_L = (rhoL**2 * W_L**2 * vL - 
               np.sqrt((p_star - pL)*((p_star - pL) + rhoL**2 * W_L**2 * (1-vL**2))) ) / \
              (rhoL**2 * W_L**2 + (p_star - pL))
        wave_speeds['x_shock_L'] = x0 + S_L*t

    # CONTINUIDAD
    wave_speeds['x_contact'] = x0 + v_star*t

    # LADO DERECHO
    if p_star <= pR:
        s_head_R = (vR + cR)/(1 + vR*cR)
        h_starR = specific_enthalpy(rho_starR, p_star, gamma)
        c_starR = np.sqrt(gamma*p_star/(rho_starR*h_starR))
        s_tail_R = (v_star + c_starR)/(1 + v_star*c_starR)
        wave_speeds['x_tail_R'] = x0 + s_head_R*t  # Usamos s_head_R como frontera
    else:
        W_R = lorentz_factor(vR)
        S_R = (rhoR**2 * W_R**2 * vR + 
               np.sqrt((p_star - pR)*((p_star - pR) + rhoR**2 * W_R**2 * (1-vR**2))) ) / \
              (rhoR**2 * W_R**2 + (p_star - pR))
        wave_speeds['x_shock_R'] = x0 + S_R*t

    return {
        'p_star': p_star,
        'v_star': v_star,
        'rho_starL': rho_starL,
        'rho_starR': rho_starR,
        'cL': cL,
        'cR': cR,
        **wave_speeds
    }

# =============================================================================
# Solución completa del tubo de choque relativista
# =============================================================================
def general_shock_tube_solution_relativistic(x, t, rhoL, vL, pL, rhoR, vR, pR, gamma=5.0/3.0, x0=0.5):
    """
    Construye la solución analítica (exacta) para el problema del tubo de choque
    en el régimen relativista en función de la posición x y el tiempo t.
    Devuelve: rho, v, p y la energía interna (e_int = p/[(γ-1)ρ]).
    """
    sol = exact_riemann_solution_relativistic(rhoL, vL, pL, rhoR, vR, pR, gamma, x0, t)
    p_star    = sol['p_star']
    v_star    = sol['v_star']
    rho_starL = sol['rho_starL']
    rho_starR = sol['rho_starR']
    cL        = sol['cL']
    cR        = sol['cR']
    
    # Determinar las posiciones de las ondas
    if 'x_shock_L' in sol:
        x_wave_L = sol['x_shock_L']
    else:
        x_wave_L = sol['x_head']
        x_wave_L_tail = sol['x_tail']
    
    x_contact = sol['x_contact']
    
    if 'x_shock_R' in sol:
        x_wave_R = sol['x_shock_R']
    else:
        x_wave_R = sol['x_tail_R']
    
    # Inicializar arrays solución
    rho = np.zeros_like(x)
    v   = np.zeros_like(x)
    p   = np.zeros_like(x)
    
    for i, xi in enumerate(x):
        if xi < x_wave_L:
            rho[i] = rhoL
            v[i]   = vL
            p[i]   = pL
        elif ('x_shock_L' not in sol) and (xi < x_wave_L_tail):
            # Dentro del fan de rarefacción izquierda (interpolación exacta en ξ)
            frac = (xi - x_wave_L)/(x_wave_L_tail - x_wave_L)
            v[i]   = vL + frac*(v_star - vL)
            p[i]   = pL + frac*(p_star - pL)
            rho[i] = rhoL*(p[i]/pL)**(1/gamma)
        elif xi < x_contact:
            rho[i] = rho_starL
            v[i]   = v_star
            p[i]   = p_star
        elif xi < x_wave_R:
            rho[i] = rho_starR
            v[i]   = v_star
            p[i]   = p_star
        else:
            rho[i] = rhoR
            v[i]   = vR
            p[i]   = pR

    e_int = p/(rho*(gamma-1))
    return rho, v, p, e_int

# =============================================================================
# Visualización de datos (compara SPH con la solución analítica relativista)
# =============================================================================

def plot_data(filename):
    """Genera gráficos comparando los datos SPH con la solución analítica relativista."""
    data = pd.read_csv(filename, sep='\t', skipinitialspace=True)
    
    # Limpiar nombres de columnas eliminando espacios en blanco al inicio y al final
    data.columns = data.columns.str.strip()
    
    time = data['t'].iloc[0]
    step = os.path.splitext(os.path.basename(filename))[0].split('_')[-1]
    
    # Condiciones iniciales (ejemplo típico)
    rhoL, vL, pL = 1.0, 0.0, 100000.0
    rhoR, vR, pR = 1.0, 0.0, 1.0
    gamma = 5.0/3.0

    x_analytic = np.linspace(0.0, 1.0, 1000)
    rho_an, v_an, p_an, eint_an = general_shock_tube_solution_relativistic(
        x_analytic, time, rhoL, vL, pL, rhoR, vR, pR, gamma
    )

    # Filtrar partículas reales y ghost
    real   = data[data['IsGhost'] == 0]
    ghosts = data[data['IsGhost'] == 1]

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    axs[0,0].scatter(real['x'], real['vx'], s=2, c='r', label='SPH Real')
    axs[0,0].scatter(ghosts['x'], ghosts['vx'], s=2, c='k', label='SPH Ghost')
    axs[0,0].plot(x_analytic, v_an, 'k-', lw=2, label='Analítico Relativista')
    axs[0,0].set_title(f'Velocidad (t={time:.4f})')
    axs[0,0].legend()

    axs[0,1].scatter(real['x'], real['P'], s=2, c='r')
    axs[0,1].scatter(ghosts['x'], ghosts['P'], s=2, c='k')
    axs[0,1].plot(x_analytic, p_an, 'k-', lw=2)
    axs[0,1].set_title('Presión')

    axs[1,0].scatter(real['x'], real['u'], s=2, c='r')
    axs[1,0].scatter(ghosts['x'], ghosts['u'], s=2, c='k')
    axs[1,0].plot(x_analytic, eint_an, 'k-', lw=2)
    axs[1,0].set_title('Energía Interna')

    axs[1,1].scatter(real['x'], real['N'], s=2, c='r')
    axs[1,1].scatter(ghosts['x'], ghosts['N'], s=2, c='k')
    axs[1,1].plot(x_analytic, rho_an, 'k-', lw=2)
    axs[1,1].set_title('Densidad')

    plt.tight_layout()
    plt.savefig(f'plot_step_{step}.png')
    plt.close()

# =============================================================================
# Función principal
# =============================================================================
def main():
    csv_files = sorted(glob.glob('outputs/output_step_*.csv'),
                       key=lambda x: int(x.split('_')[-1].split('.')[0]))
    os.makedirs('plots', exist_ok=True)
    os.chdir('plots')
    for file in csv_files:
        print(f'Procesando: {file}')
        plot_data(os.path.join('..', file))
    print("Todos los gráficos generados en la carpeta 'plots'")

if __name__ == "__main__":
    main()
