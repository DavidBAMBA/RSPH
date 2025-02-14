
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_data(filename):
    # Leer los datos del CSV con tabulación como separador
    data = pd.read_csv(filename, sep="\t")
    
    # Extraer el tiempo del primer valor (asumiendo que es constante)
    time = data['t'].iloc[0]
    
    # Extraer el número de paso desde el nombre del archivo
    step = os.path.splitext(os.path.basename(filename))[0].split('_')[-1]

    # Crear una figura con 4 subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Rango x para solución analítica (si aplica)
    x_analytic = np.linspace(0.2, 0.8, 500)

    # Separar partículas reales y fantasma
    real_particles = data[data['        IsGhost'] == 0]
    ghost_particles = data[data['        IsGhost'] == 1]

    # Gráfico 1: Velocidad vx vs Posición x
    axs[0, 0].scatter(real_particles['  x'], real_particles['        vx'], s=1, c='blue', label='SPH (reales)')
    axs[0, 0].scatter(ghost_particles['  x'], ghost_particles['        vx'], s=1, c='black', label='SPH (fantasma)')
    axs[0, 0].set_title(f'Velocidad vs Posición\nPaso {step}, Tiempo {time:.4f}')
    axs[0, 0].set_xlabel('Posición x')
    axs[0, 0].set_ylabel('Velocidad vx')
    axs[0, 0].set_xlim(0.0, 1.0)
    #axs[0, 0].set_xlim(0.45, 0.55)
    axs[0, 0].legend()

    # Gráfico 2: Presión vs Posición x
    axs[0, 1].scatter(real_particles['  x'], real_particles['        P'], s=1, c='red', label='SPH (reales)')
    axs[0, 1].scatter(ghost_particles['  x'], ghost_particles['        P'], s=1, c='black', label='SPH (fantasma)')
    axs[0, 1].set_title(f'Presión vs Posición\nPaso {step}, Tiempo {time:.4f}')
    axs[0, 1].set_xlabel('Posición x')
    axs[0, 1].set_ylabel('Presión')
    axs[0, 1].set_xlim(0.0, 1.0)
    #axs[0, 1].set_xlim(1.0, 2.0)
    #axs[0, 1].set_ylim(0.5, 1.5)

    axs[0, 1].legend()

    # Gráfico 3: Energía Interna vs Posición x
    axs[1, 0].scatter(real_particles['  x'], real_particles['        vy'], s=1, c='green', label='SPH (reales)')
    axs[1, 0].scatter(ghost_particles['  x'], ghost_particles['        vy'], s=1, c='black', label='SPH (fantasma)')
    axs[1, 0].set_title(f'Energía Interna vs Posición\nPaso {step}, Tiempo {time:.4f}')
    axs[1, 0].set_xlabel('Posición x')
    axs[1, 0].set_ylabel('Energía Interna')
    #axs[1, 0].set_xlim(1.0, 2.0)
    axs[1, 0].set_xlim(0.0, 1.0)
    axs[1, 0].legend()

    # Gráfico 4: Densidad vs Posición x
    axs[1, 1].scatter(real_particles['  x'], real_particles['        N'], s=1, c='purple', label='SPH (reales)')
    axs[1, 1].scatter(ghost_particles['  x'], ghost_particles['        N'], s=1, c='black', label='SPH (fantasma)')
    axs[1, 1].set_title(f'Densidad vs Posición\nPaso {step}, Tiempo {time:.4f}')
    axs[1, 1].set_xlabel('Posición x')
    axs[1, 1].set_ylabel('Densidad')
    axs[1, 1].set_xlim(0.0, 1.0)
    axs[1, 1].legend()

    # Ajustar el layout
    plt.tight_layout()
    
    # Guardar la figura
    plt.savefig(f'results_step_{step}.png')
    plt.close()

def main():
    # Obtener una lista de todos los archivos CSV que coinciden con el patrón
    csv_files = glob.glob('outputs/output_step_*.csv')
    
    # Ordenar los archivos por número de paso
    csv_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[-1]))
    
    # Filtrar archivos cuyos números de paso son múltiplos de 100
    csv_files = [f for f in csv_files if int(os.path.splitext(os.path.basename(f))[0].split('_')[-1]) % 100 == 0]
    
    # Crear una carpeta para guardar las imágenes si no existe
    os.makedirs('plots', exist_ok=True)
    
    # Cambiar al directorio de plots
    os.chdir('plots')
    
    # Iterar sobre cada archivo CSV filtrado y generar los gráficos
    for filename in csv_files:
        print(f'Procesando {filename}...')
        plot_data(os.path.join('..', filename))
    
    print('Gráficos generados exitosamente en la carpeta "plots".')

if __name__ == '__main__':
    main()
