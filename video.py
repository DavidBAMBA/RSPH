import os
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

def create_video_from_plots(folder='plots', output_video='Relativistic-Sod-Shock-Tube.mp4', fps=10):
    """
    Crea un video a partir de las gráficas generadas en la carpeta `folder`.
    
    Parámetros:
        folder (str): Carpeta donde se encuentran las imágenes.
        output_video (str): Nombre del archivo de salida.
        fps (int): Cuadros por segundo del video.
    """
    # Verificar que la carpeta exista
    if not os.path.exists(folder):
        raise FileNotFoundError(f"La carpeta '{folder}' no existe.")
    
    # Listar las imágenes en orden y filtrar por múltiplos de 100
    images = sorted(
        [os.path.join(folder, img) for img in os.listdir(folder) if img.endswith('.png')],
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[-1])
    )
    
    # Filtrar solo los archivos que tienen un número múltiplo de 100
    images = [img for img in images if int(os.path.splitext(os.path.basename(img))[0].split('_')[-1]) % 100 == 0]
    
    # Verificar que haya imágenes
    if not images:
        raise ValueError(f"No se encontraron imágenes con múltiplos de 100 en la carpeta '{folder}'.")
    
    # Crear el video
    clip = ImageSequenceClip(images, fps=fps)
    clip.write_videofile(output_video, codec="libx264", audio=False)
    print(f"Video guardado como '{output_video}'.")

if __name__ == "__main__":
    create_video_from_plots()
