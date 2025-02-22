import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import srrp

############################################################
# Función para calcular la métrica L2 (versión adimensional)
############################################################
def compute_L2_error(x_num, y_num, x_ana, y_ana):
    """
    Calcula el error L2 adimensional entre la solución numérica y la analítica.
    
    Parámetros
    ----------
    x_num : array
        Posiciones de las partículas numéricas.
    y_num : array
        Valores numéricos (ej. velocidad, densidad...) en dichas posiciones.
    x_ana : array
        Posiciones donde se conoce la solución analítica.
    y_ana : array
        Valores analíticos correspondientes a x_ana.
    
    Retorna
    -------
    L2 : float
        Error L2 adimensional.
    """
    # Interpolamos la solución analítica en las posiciones numéricas
    y_ana_interp = np.interp(x_num, x_ana, y_ana)
    
    # Evitar divisiones por cero si max es cero o muy pequeño
    max_exact = np.max(np.abs(y_ana_interp))
    if max_exact < 1e-14:
        return np.nan  # O devuelve 0 si prefieres
    
    # Fórmula L2
    N = len(y_num)
    suma_cuadrados = np.sum((y_num - y_ana_interp)**2)
    L2 = np.sqrt((1.0 / (N * max_exact)) * suma_cuadrados)
    
    return L2

def solve_relativistic_riemann(rho_L, p_L, v_L, rho_R, p_R, v_R, Gamma, t, x0=0.0, npts=1000):
    """Función envoltorio para el solver analítico relativista."""
    if t <= 0:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    
    stateL = srrp.State(rho=rho_L, pressure=p_L, vx=v_L, vt=0)
    stateR = srrp.State(rho=rho_R, pressure=p_R, vx=v_R, vt=0)
    solver = srrp.Solver()
    solution = solver.solve(stateL, stateR, Gamma)
    
    xs = np.linspace(-0.5, 0.5, npts)
    xis = (xs - x0)/t if t != 0 else np.zeros_like(xs)
    
    try:
        states = solution.getState(xis)
    except:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    
    # Calcular energía interna específica (ε = p/((Γ-1)ρ))
    eint_an = states.pressure/((Gamma - 1.0)*states.rho)
    
    return xs, states.rho, states.vx, states.pressure, eint_an

def plot_data(filename):
    # Leer los datos del CSV con tabulación como separador
    data = pd.read_csv(filename, sep='\\t')
    
    # Extraer el tiempo del primer valor (asumiendo que es constante)
    time = data['t'].iloc[0]
    if time <= 0:
        print(f'Archivo {filename} tiene tiempo={time} - omitiendo')
        return
    
    # Extraer el número de paso desde el nombre del archivo
    step = os.path.splitext(os.path.basename(filename))[0].split('_')[-1]

    # Crear figura con subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Separar partículas reales y fantasma
    real_particles = data[data['        IsGhost'] == 0]
    ghost_particles = data[data['        IsGhost'] == 1]

    # Condiciones para solver analítico
    rhoL, pL, vL = 1.0, 1000.0, 0.0   
    rhoR, pR, vR = 1.0, 0.001, 0.0  
    Gamma = 5.0/3.0

    # Solución analítica
    x_an, rho_an, vel_an, pres_an, eint_an = solve_relativistic_riemann(
        rhoL, pL, vL, rhoR, pR, vR, Gamma, time
    )

    # === Gráfico 1: Velocidad vs Posición ===
    axs[0, 0].scatter(real_particles['  x'], real_particles['        vx'], 
                     s=1, c='blue', label='SPH (reales)')
    axs[0, 0].scatter(ghost_particles['  x'], ghost_particles['        vx'], 
                     s=1, c='black', label='SPH (fantasma)')
    if len(x_an) > 0:
        axs[0, 0].plot(x_an, vel_an, 'k', lw=2, label='Analítico')
    axs[0, 0].set_title(f'Velocidad vs Posición (Paso {step}, t={time:.4f})')
    axs[0, 0].set_xlabel('x')
    axs[0, 0].set_ylabel('vx')
    axs[0, 0].set_xlim(-0.5, 0.5)
    axs[0, 0].legend()

    # === Gráfico 2: Presión vs Posición ===
    axs[0, 1].scatter(real_particles['  x'], real_particles['        P'], 
                     s=1, c='red', label='SPH (reales)')
    axs[0, 1].scatter(ghost_particles['  x'], ghost_particles['        P'], 
                     s=1, c='black', label='SPH (fantasma)')
    if len(x_an) > 0:
        axs[0, 1].plot(x_an, pres_an, 'k', lw=2, label='Analítico')
    axs[0, 1].set_title(f'Presión vs Posición (Paso {step}, t={time:.4f})')
    axs[0, 1].set_xlabel('x')
    axs[0, 1].set_xlim(-0.5, 0.5)

    axs[0, 1].set_ylabel('P')
    axs[0, 1].legend()


    # === Gráfico 3: Energía Interna vs Posición ===
    axs[1, 0].scatter(real_particles['  x'], real_particles['        u'], 
                     s=1, c='green', label='SPH (reales)')
    axs[1, 0].scatter(ghost_particles['  x'], ghost_particles['        u'], 
                     s=1, c='black', label='SPH (fantasma)')
    if len(x_an) > 0:
        axs[1, 0].plot(x_an, eint_an, 'k', lw=2, label='Analítico')
    axs[1, 0].set_title(f'Energía Interna vs Posición (Paso {step}, t={time:.4f})')
    axs[1, 0].set_xlabel('x')
    axs[1, 0].set_xlim(-0.5, 0.5)

    axs[1, 0].set_ylabel('u')
    axs[1, 0].legend()

    # === Gráfico 4: Densidad vs Posición ===
    axs[1, 1].scatter(real_particles['  x'], real_particles['        N'], 
                     s=1, c='purple', label='SPH (reales)')
    axs[1, 1].scatter(ghost_particles['  x'], ghost_particles['        N'], 
                     s=1, c='black', label='SPH (fantasma)')
    if len(x_an) > 0:
        axs[1, 1].plot(x_an, rho_an, 'k', lw=2, label='Analítico')
    axs[1, 1].set_title(f'Densidad vs Posición (Paso {step}, t={time:.4f})')
    axs[1, 1].set_xlabel('x')
    axs[1, 1].set_xlim(-0.5, 0.5)

    axs[1, 1].set_ylabel('rho')
    axs[1, 1].legend()

    plt.tight_layout()
    plt.savefig(f'results_step_{step}.png')
    plt.close()

    ###########################################################
    # Cálculo y reporte del error L2 (ejemplo en cada variable)
    ###########################################################
    # Solo partículas reales para comparar con la solución exacta
    x_r = real_particles['  x'].to_numpy()
    
    # Velocidad
    vx_r = real_particles['        vx'].to_numpy()
    L2_vx = compute_L2_error(x_r, vx_r, x_an, vel_an) if len(x_an) else np.nan
    
    # Presión
    p_r = real_particles['        P'].to_numpy()
    L2_p = compute_L2_error(x_r, p_r, x_an, pres_an) if len(x_an) else np.nan
    
    # Energía interna
    u_r = real_particles['        u'].to_numpy()
    L2_u = compute_L2_error(x_r, u_r, x_an, eint_an) if len(x_an) else np.nan
    
    # Densidad (ten en cuenta que en tus datos se llama 'N', ajústalo si es la densidad real)
    # Si 'N' es la densidad numérica, procede, si no, ajusta a la columna correspondiente
    rho_r = real_particles['        N'].to_numpy()
    L2_rho = compute_L2_error(x_r, rho_r, x_an, rho_an) if len(x_an) else np.nan
    
    print(f"Paso {step} | t={time:.4f}:")
    print(f"  -> L2(vx)   = {L2_vx:.6f}")
    print(f"  -> L2(P)    = {L2_p:.6f}")
    print(f"  -> L2(u)    = {L2_u:.6f}")
    print(f"  -> L2(rho)  = {L2_rho:.6f}")
    
def main():
    # Obtener lista de archivos CSV que coinciden con el patrón
    csv_files = glob.glob('outputs/output_step_*.csv')
    
    # Ordenar por número de paso
    csv_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[-1]))
    
    # Filtrar archivos cuyos pasos son múltiplos de 100 (ajusta si gustas)
    csv_files = [f for f in csv_files if int(os.path.splitext(os.path.basename(f))[0].split('_')[-1]) % 100 == 0]
    
    # Crear carpeta para guardar las imágenes
    os.makedirs('plots', exist_ok=True)
    os.chdir('plots')
    
    # Generar gráficos y errores
    for filename in csv_files:
        print(f'Procesando {filename}...')
        plot_data(os.path.join('..', filename))
    
    print('Gráficos generados exitosamente en la carpeta \"plots\".')


if __name__ == '__main__':
    main()
