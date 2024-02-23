# Imports necesarios
from flask import Flask, render_template, request, redirect, url_for, session, jsonify

import os
import torch
from torchvision import transforms
from torchvision.transforms import ToPILImage


from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from notebooks.Model.unet import *

app = Flask(__name__)

app.secret_key = 'your_secret_key'

# Ruta a donde se almacenan las imagenes que se cargan
upload_directory='static/uploads'

# Ruta a donde se almacenan las imagenes resultado
results_directory='static/results'

nombre_model = 'model/best_model_results.pth'

# Transformación para convertir un Tensor en una imagen PIL
to_pil = ToPILImage()

# Si no existe ese path lo crea
if(not os.path.exists(upload_directory)):
    os.mkdir(upload_directory)

if(not os.path.exists(results_directory)):
    os.mkdir(results_directory)

# Transformación sobre la imagen para que sea interpretada por el modelo
def process_image(image_path):
    try:
        image = Image.open(image_path)
        preprocess = transforms.Compose([
            transforms.Resize(256),  

            # Recorta la imagen al centro para obtener una imagen de 256x256
            transforms.CenterCrop(256),  
            transforms.ToTensor(),
        ])
        image = preprocess(image)
        return image.unsqueeze(0)

    # Gestion de errores en caso de que no se pueda procesar la imagen
    except Exception as e:
        print(f"Error al procesar la imagen: {e}")
        return None

# Realiza la predicción del modelo
def make_prediction(model, image):
    try:
        # Podemos el modelo en modo evaluación
        model.eval()

        # Predicción de nuestro modelo para la imagen
        with torch.no_grad():
            output = model(image)
            outputs = torch.softmax(output, dim=1)  

        print("Fin predicción, cargando resultados...")

        return outputs

    # Si falla la predicción capturamos el error
    except Exception as e:
        print(f"Error al hacer la predicción: {e}")
        return None


# Binarizamos los mapas aplicando el criterio de mayor probabilidad para los pixeles multiclase
def bin_prediction(outputs, thresholds):
    try:
        # Binarización de los mapas de probabilidad, creamos un tensor con la misma forma que la salida
        binary_maps = torch.zeros_like(outputs)

        # Creamos un tensor para almacenar las probabilidades máximas
        max_probs, max_classes = torch.max(outputs, dim=1)

        # Generamos un plano binarizado para cada clase, si superan el umbral 1 sino 0
        for i in range(outputs.size(1)):

            # Transformamos los tensores, de todo 0 a, 1's donde se supera el umbral y 0's donde no
            binary_map = (outputs[:, i, :, :] > thresholds[i]).float()

            # Aplicamos la clase de mayor probabilidad obtenida en max_classes
            # Para cada una de las clases generamos el mapa binarizado, pero ahora los pixeles tienen 0's o el valor de la clase correspondiente
            binary_maps[:, i, :, :] = binary_map * (max_classes == i).float()

        return binary_maps

    # Si falla la binarización capturamos el error
    except Exception as e:
        print(f"Error al binarizar la predicción: {e}")
        return None

# Aplicamos la máscara de color elegida a los mapas binarizados
def apply_mask_color(outputs, binary_maps, class_colors):
    try:
        # Inicializa un tensor de salida para la imagen final
        final_image = torch.zeros((1, 3, outputs.size(2), outputs.size(3)))

        # Recorremos las claves del diccionario de colores
        for class_idx in class_colors:
            
            # Creamos un tensor que actua de máscara para cada color
            class_mask = (binary_maps[:, class_idx, :, :] == 1).float()
            
            # Transforma cada color a un tensor de pytorch, normalizando
            color = torch.tensor(class_colors[class_idx], dtype=torch.float32) / 255.0

            # Cambiamos la forma del tensor
            color = color.view(1, 3, 1, 1)
            
            # class_mask*color permite uqe los pixeles que pertencen a la clase tengan el color correspondiente, y 0 en el resto
            # En cada ejecución del bucle se suma un nuevo plano, se hace porque son excluyentes unos de otros, 
            # y un mismo pixel unicamente puede tomar un valor
            final_image += class_mask * color

        # Convierte el tensor a un arreglo NumPy y muestra la imagen
        final_image_np = final_image.squeeze().cpu().numpy()

        return final_image_np

    # Si no puede aplicar la máscara de colores capturamos el error
    except Exception as e:
        print(f"Error al aplicar la máscara de color: {e}")
        return None

# Cuenta los pixeles negros de la imagen predicha
def count_black_pixels(final_image_np):
    try:
        # Convierte la imagen a escala de grises
        imagen_gris = final_image_np.mean(axis=0)

        # Cuenta el número total de píxeles en la imagen
        total_pixeles = int(imagen_gris.shape[0] * imagen_gris.shape[1])

        # Cuenta el número de píxeles negros en la imagen
        pixeles_negros = int(np.sum(imagen_gris == 0))

        porcentaje_negros = (pixeles_negros/total_pixeles)*100

        return porcentaje_negros,pixeles_negros

    except Exception as e:
        print(f"Error al contar los píxeles negros: {e}")
        return None, None


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':

        # guarda el archivo
        f = request.files['input']
        filename = f.filename

        # Almacenamos la imagen original
        filepath = os.path.join(upload_directory, filename)
        f.save(filepath)
        print(f"Image saved at {filepath}")
        print(f"Filename: {filename}")        

        print("Procesando imagen...")
        image = process_image(filepath)

        # Almacenamos la imagen procesada
        pil_image = to_pil(image.squeeze(0))  
        resized_filename = f'resized_{filename}'
        resized_filepath = os.path.join(upload_directory, resized_filename)
        
        try:
            # Guardamos la imagen
            pil_image.save(resized_filepath)
            print(f"Resized image saved at {resized_filepath}")
            print(f"Resized Filename: {resized_filename}")
        except Exception as e:
            print(f'Ocurrió un error al guardar la imagen: {e}')

        # Creamos instancia del modelo
        model = UNet(5)

        # Cargamos el modelo con el que hemos obtenido los mejores resultados
        model = torch.load(nombre_model,torch.device('cpu'))
        
        print("Modelo cargado correctamente")

        # Obtenemos la prediccion del modelo
        outputs = make_prediction(model, image)

        # Umbrales para la binarizacion
        thresholds = [0.50, 0.50, 0.50, 0.50, 0.50]

        # Define el mapeo de clases a colores
        class_colors = {
            0: [155, 38, 182],
            1: [14, 135, 204],
            2: [124, 252, 0],
            3: [255, 20, 147],
            4: [169, 169, 169]
        }

        print("- Optimizando resultados -")
        # Obtenemos las predicciones binarizadas
        bin_maps = bin_prediction(outputs, thresholds)

        print("Aplicando mascara de colores")
        # Obtenemos la imagen aplicando los colores
        final_image_np = apply_mask_color(outputs, bin_maps, class_colors)
        
        # Generamos el plt.figure del tamño adecuado para la imagen final
        plt.figure(figsize=(256 / plt.gcf().dpi, 256 / plt.gcf().dpi))
        plt.imshow(final_image_np.transpose(1, 2, 0))
        plt.axis('off')

        print("Contando pixeles en negro")
        porcentaje_negros, pixeles_negros = count_black_pixels(final_image_np)
        

        # Guarda la imagen en la ruta especificada
        base_filename = os.path.splitext(filename)[0]  # Obtiene el nombre del archivo sin la extensión
        feature_map_filename = f'{base_filename}_prediction.png'

        print(f'Dimensiones de la imagen antes de guardar: {final_image_np.shape}')
    
        try:
            plt.savefig(f'{results_directory}/{feature_map_filename}', bbox_inches='tight', pad_inches=0)
            print(f'Imagen guardada correctamente en {results_directory}/{feature_map_filename}')
        except Exception as e:
            print(f'Ocurrió un error al guardar la imagen: {e}')
        finally:
            plt.close()


        # Guardamos en la sesion lo que es de nuestro interes
        session['feature_map_filename'] = feature_map_filename
        session['original_image_filepath'] = 'uploads/' + resized_filename
        session['pixeles_negros'] = pixeles_negros
        session['porcentaje_negros'] = round(porcentaje_negros,2)

        print("Mostrando resultados obtenidos...")
        # redirige a la página de procesamiento
        return redirect(url_for('processing'))

    return render_template('interfaz.html')

@app.route('/processing')
def processing():
    # Obtén los nombres de los archivos de los mapas de características desde la sesión
    feature_map_filename = session.get('feature_map_filename', None)

    # Obtén la ruta completa de la imagen original desde la sesión
    original_image_filepath = session.get('original_image_filepath', None)

    # Obtenemos los pixeles negross y el porcentaje
    pixeles_negros = session.get('pixeles_negros', None)
    porcentaje_negros = session.get('porcentaje_negros', None)

    # Pasa los nombres de los archivos y la ruta de la imagen original a la plantilla
    return render_template('resultados.html', feature_map_filename=feature_map_filename, original_image_filepath=original_image_filepath, pixeles_negros=pixeles_negros, porcentaje_negros=porcentaje_negros)

@app.route('/interfaz')
def interfaz():
    print("Redirigiendo a interfaz HTML, ahora cargue su imagen.")
    return render_template('interfaz.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
