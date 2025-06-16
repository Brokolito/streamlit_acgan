import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Activation, Dense, Input
from tensorflow.keras.layers import Conv2D, Flatten
from tensorflow.keras.layers import Reshape, Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization

from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

import os
import math
import numpy as np
import matplotlib.pyplot as plt
# --- Función de Ploteo (Modificada para recibir etiquetas numéricas) ---
def plot_images_acgan(generator,
                      noise_input,
                      noise_class_labels, # Recibe etiquetas numéricas
                      show=False,
                      step=0,
                      model_name="gan"):
    """Generate fake images and plot them (ACGAN version)"""
    os.makedirs(model_name, exist_ok=True)
    filename = os.path.join(model_name, f"{step:05d}.png")

    # Generator espera etiquetas con shape (batch_size, 1)
    noise_class_input = noise_class_labels.reshape(-1, 1)
    images = generator.predict([noise_input, noise_class_input])

    print(f"{model_name} labels for generated images: {noise_class_labels}")

    plt.figure(figsize=(4.4, 4.4)) # Aumentado para más claridad
    num_images = images.shape[0]
    image_size = images.shape[1]
    rows = int(math.sqrt(num_images))
    cols = int(math.ceil(num_images / rows))

    for i in range(num_images):
        plt.subplot(rows, cols, i + 1)
        image = np.reshape(images[i], [image_size, image_size])
        plt.imshow(image, cmap='gray')
        # Añadir la etiqueta como título
        plt.title(f"Label: {noise_class_labels[i]}")
        plt.axis('off')

    plt.tight_layout() # Ajustar espaciado
    plt.savefig(filename)
    if show:
        plt.show()
    else:
        plt.close('all')
# --- Función de Test (Modificada para pasar etiquetas numéricas) ---
def test_generator_acgan(generator, latent_size, num_classes, class_label=None):
    noise_input = np.random.uniform(-1.0, 1.0, size=[16, latent_size])
    step = 0
    if class_label is None:
        # Generar ejemplos de clases aleatorias
        noise_class_labels = np.random.randint(0, num_classes, 16)
    else:
        # Generar ejemplos de una clase específica
        noise_class_labels = np.ones(16, dtype='int32') * class_label
        step = class_label

    plot_images_acgan(generator,
                      noise_input=noise_input,
                      noise_class_labels=noise_class_labels, # Pasar etiquetas numéricas
                      show=True,
                      step=step,
                      model_name="test_outputs_acgan")
trained_generator = load_model("acgan_mnist.h5")
etiqueta = int(input("Ingresa el numero a generar (0-9): "))
test_generator_acgan(trained_generator, 100, 10, class_label=etiqueta)