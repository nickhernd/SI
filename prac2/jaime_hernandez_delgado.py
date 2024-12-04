# Nombre : Jaime Hernández Delgado
# DNI : 48776654W
# PRÁCTICA 2: Visión artificial y aprendizaje

# Configuración de warnings y logging
import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Librerías básicas
import numpy as np
import time
import os

# TensorFlow y Keras
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Visualización
import matplotlib.pyplot as plt
import seaborn as sns

# Procesamiento de imágenes
from PIL import Image

# Métricas y evaluación
from sklearn.metrics import confusion_matrix, classification_report


# Carga y preprocesa el dataset CIFAR10
# Devuelve los conjuntos de datos normalizados y preprocesados
def cargar_y_preprocesar_cifar10():
    (X_train, Y_train), (X_test, Y_test) = keras.datasets.cifar10.load_data()

    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    Y_train = keras.utils.to_categorical(Y_train, 10)
    Y_test = keras.utils.to_categorical(Y_test, 10)

    return X_train, Y_train, X_test, Y_test

# Visualiza las curvas de pérdida y precisión durante el entrenamiento
# Parámetros:
#   - historia: objeto History devuelto por model.fit()
#   - titulo: título para las gráficas
def visualizar_historia(historia, titulo="Evolución del entrenamiento"):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    ax1.plot(historia.history['accuracy'], label='train')
    ax1.plot(historia.history['val_accuracy'], label='validation')
    ax1.set_title(f'{titulo} - Precisión')
    ax1.set_ylabel('Precisión')
    ax1.set_xlabel('Época')
    ax1.legend()

    ax2.plot(historia.history['loss'], label='train')
    ax2.plot(historia.history['val_loss'], label='validation')
    ax2.set_title(f'{titulo} - Pérdida')
    ax2.set_ylabel('Pérdida')
    ax2.set_xlabel('Época')
    ax2.legend()

    plt.tight_layout()
    plt.show()

# Crea y devuelve un modelo MLP básico con las capas especificadas
def crear_modelo_mlp():
    modelo = Sequential([
        Flatten(input_shape=(32, 32, 3)),
        Dense(32, activation='sigmoid'),
        Dense(10, activation='softmax')
    ])

    modelo.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return modelo

# Realiza experimentos con diferentes números de épocas
# Entrena modelos y compara sus resultados
# Parámetros:
#   - X_train, Y_train: datos de entrenamiento
#   - X_test, Y_test: datos de prueba
#   - epocas_list: lista con los diferentes valores de épocas a probar
def experimentar_epocas(X_train, Y_train, X_test, Y_test, epocas_list=[5, 10, 20, 50, 100]):
    resultados = []

    for epocas in epocas_list:
        print(f"\nEntrenando modelo con {epocas} épocas...")
        modelo = crear_modelo_mlp()

        tiempo_inicio = time.time()
        historia = modelo.fit(
            X_train, Y_train,
            epochs=epocas,
            batch_size=32,
            validation_split=0.1,
            verbose=1
        )
        tiempo_total = time.time() - tiempo_inicio

        test_loss, test_acc = modelo.evaluate(X_test, Y_test, verbose=0)

        resultados.append({
            'epocas': epocas,
            'historia': historia,
            'test_acc': test_acc,
            'test_loss': test_loss,
            'tiempo': tiempo_total
        })

        print(f"Test accuracy: {test_acc:.4f}")
        print(f"Tiempo: {tiempo_total:.2f} segundos")

    return resultados

# Visualiza los resultados de los experimentos con diferentes épocas
# Genera gráficas de accuracy y tiempo de entrenamiento
def visualizar_resultados_epocas(resultados):
    epocas = [r['epocas'] for r in resultados]
    accuracies = [r['test_acc'] for r in resultados]
    tiempos = [r['tiempo'] for r in resultados]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    ax1.plot(epocas, accuracies, 'bo-')
    ax1.set_xlabel('Número de épocas')
    ax1.set_ylabel('Test Accuracy')
    ax1.set_title('Accuracy vs Número de épocas')
    ax1.grid(True)

    ax2.plot(epocas, tiempos, 'ro-')
    ax2.set_xlabel('Número de épocas')
    ax2.set_ylabel('Tiempo de entrenamiento (s)')
    ax2.set_title('Tiempo vs Número de épocas')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    mejor_idx = np.argmax(accuracies)
    mejor_historia = resultados[mejor_idx]['historia']

    plt.figure(figsize=(10, 6))
    plt.plot(mejor_historia.history['accuracy'], label='train')
    plt.plot(mejor_historia.history['val_accuracy'], label='validation')
    plt.title(f'Curvas de aprendizaje para {epocas[mejor_idx]} épocas')
    plt.xlabel('Época')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

# Visualiza la matriz de confusión del modelo
# Parámetros:
#   - Y_true: etiquetas verdaderas
#   - Y_pred: predicciones del modelo
def visualizar_matriz_confusion(Y_true, Y_pred, titulo="Matriz de Confusión"):
    if len(Y_true.shape) > 1:
        Y_true = np.argmax(Y_true, axis=1)
    if len(Y_pred.shape) > 1:
        Y_pred = np.argmax(Y_pred, axis=1)

    cm = confusion_matrix(Y_true, Y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(titulo)
    plt.ylabel('Verdadero')
    plt.xlabel('Predicho')
    plt.show()

# Función principal que ejecuta los experimentos de la Tarea B
def tarea_b():
    print("Cargando y preprocesando datos...")
    X_train, Y_train, X_test, Y_test = cargar_y_preprocesar_cifar10()

    print("Iniciando experimentos...")
    epocas_list = [5, 10, 20, 50, 100]
    resultados = experimentar_epocas(X_train, Y_train, X_test, Y_test, epocas_list)

    print("\nVisualizando resultados...")
    visualizar_resultados_epocas(resultados)

    return resultados

# Realiza experimentos con diferentes tamaños de batch
def experimentar_batch_sizes(X_train, Y_train, X_test, Y_test, batch_sizes=[16, 32, 64, 128, 256]):
    resultados = []

    for batch_size in batch_sizes:
        print(f"\nEntrenando modelo con batch_size={batch_size}...")
        modelo = crear_modelo_mlp()

        tiempo_inicio = time.time()
        historia = modelo.fit(
            X_train, Y_train,
            epochs=50,  # Usamos 50 épocas basándonos en los resultados de la Tarea B
            batch_size=batch_size,
            validation_split=0.1,
            verbose=1
        )
        tiempo_total = time.time() - tiempo_inicio

        test_loss, test_acc = modelo.evaluate(X_test, Y_test, verbose=0)

        resultados.append({
            'batch_size': batch_size,
            'historia': historia,
            'test_acc': test_acc,
            'test_loss': test_loss,
            'tiempo': tiempo_total
        })

        print(f"Test accuracy: {test_acc:.4f}")
        print(f"Tiempo: {tiempo_total:.2f} segundos")

    return resultados

def visualizar_resultados_batch(resultados):
    batch_sizes = [r['batch_size'] for r in resultados]
    accuracies = [r['test_acc'] for r in resultados]
    tiempos = [r['tiempo'] for r in resultados]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    ax1.semilogx(batch_sizes, accuracies, 'bo-')
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Test Accuracy')
    ax1.set_title('Accuracy vs Batch Size')
    ax1.grid(True)

    ax2.semilogx(batch_sizes, tiempos, 'ro-')
    ax2.set_xlabel('Batch Size')
    ax2.set_ylabel('Tiempo de entrenamiento (s)')
    ax2.set_title('Tiempo vs Batch Size')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    mejor_idx = np.argmax(accuracies)
    mejor_historia = resultados[mejor_idx]['historia']

    plt.figure(figsize=(10, 6))
    plt.plot(mejor_historia.history['accuracy'], label='train')
    plt.plot(mejor_historia.history['val_accuracy'], label='validation')
    plt.title(f'Curvas de aprendizaje para batch_size={batch_sizes[mejor_idx]}')
    plt.xlabel('Época')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

# Función principal que ejecuta los experimentos de la Tarea C
def tarea_c():
    print("Cargando y preprocesando datos...")
    X_train, Y_train, X_test, Y_test = cargar_y_preprocesar_cifar10()

    print("Iniciando experimentos...")
    batch_sizes = [16, 32, 64, 128, 256]
    resultados = experimentar_batch_sizes(X_train, Y_train, X_test, Y_test, batch_sizes)

    print("\nVisualizando resultados...")
    visualizar_resultados_batch(resultados)

    return resultados

# Crea y devuelve un modelo MLP con la función de activación especificada
def crear_modelo_mlp_con_activacion(activacion):
    modelo = Sequential([
        Flatten(input_shape=(32, 32, 3)),
        Dense(32, activation=activacion),
        Dense(10, activation='softmax')  # Capa de salida siempre con softmax para clasificación
    ])

    modelo.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return modelo

# Realiza experimentos con diferentes funciones de activación
def experimentar_activaciones(X_train, Y_train, X_test, Y_test, activaciones=['sigmoid', 'relu', 'tanh', 'elu']):
    resultados = []

    for activacion in activaciones:
        print(f"\nEntrenando modelo con activación {activacion}...")
        modelo = crear_modelo_mlp_con_activacion(activacion)

        tiempo_inicio = time.time()
        historia = modelo.fit(
            X_train, Y_train,
            epochs=50,  # Usamos 50 épocas de la Tarea B
            batch_size=256,  # Usamos el mejor batch_size de la Tarea C
            validation_split=0.1,
            verbose=1
        )
        tiempo_total = time.time() - tiempo_inicio

        test_loss, test_acc = modelo.evaluate(X_test, Y_test, verbose=0)

        resultados.append({
            'activacion': activacion,
            'historia': historia,
            'test_acc': test_acc,
            'test_loss': test_loss,
            'tiempo': tiempo_total
        })

        print(f"Test accuracy: {test_acc:.4f}")
        print(f"Tiempo: {tiempo_total:.2f} segundos")

    return resultados

def visualizar_resultados_activacion(resultados):
    activaciones = [r['activacion'] for r in resultados]
    accuracies = [r['test_acc'] for r in resultados]
    tiempos = [r['tiempo'] for r in resultados]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    # Gráfica de accuracy
    x = range(len(activaciones))
    ax1.bar(x, accuracies, align='center')
    ax1.set_xticks(x)
    ax1.set_xticklabels(activaciones)
    ax1.set_ylabel('Test Accuracy')
    ax1.set_title('Accuracy vs Función de Activación')
    ax1.grid(True)

    # Gráfica de tiempo
    ax2.bar(x, tiempos, align='center', color='red')
    ax2.set_xticks(x)
    ax2.set_xticklabels(activaciones)
    ax2.set_ylabel('Tiempo de entrenamiento (s)')
    ax2.set_title('Tiempo vs Función de Activación')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    # Mostrar curvas de aprendizaje del mejor modelo
    mejor_idx = np.argmax(accuracies)
    mejor_historia = resultados[mejor_idx]['historia']

    plt.figure(figsize=(10, 6))
    plt.plot(mejor_historia.history['accuracy'], label='train')
    plt.plot(mejor_historia.history['val_accuracy'], label='validation')
    plt.title(f'Curvas de aprendizaje para activación {activaciones[mejor_idx]}')
    plt.xlabel('Época')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

def tarea_d():
    print("Cargando y preprocesando datos...")
    X_train, Y_train, X_test, Y_test = cargar_y_preprocesar_cifar10()

    print("Iniciando experimentos...")
    activaciones = ['sigmoid', 'relu', 'tanh', 'elu']
    resultados = experimentar_activaciones(X_train, Y_train, X_test, Y_test, activaciones)

    print("\nVisualizando resultados...")
    visualizar_resultados_activacion(resultados)

    return resultados

# Tarea E: Ajuste del número de neuronas
def crear_modelo_mlp_neuronas(num_neuronas):
    modelo = Sequential([
        Flatten(input_shape=(32, 32, 3)),
        Dense(num_neuronas, activation='relu'),  # Usamos relu basándonos en resultados de la tarea D
        Dense(10, activation='softmax')
    ])

    modelo.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return modelo

def experimentar_neuronas(X_train, Y_train, X_test, Y_test, neuronas_list=[16, 32, 64, 128, 256]):
    resultados = []

    for neuronas in neuronas_list:
        print(f"\nEntrenando modelo con {neuronas} neuronas...")
        modelo = crear_modelo_mlp_neuronas(neuronas)

        tiempo_inicio = time.time()
        historia = modelo.fit(
            X_train, Y_train,
            epochs=50,
            batch_size=256,
            validation_split=0.1,
            verbose=1
        )
        tiempo_total = time.time() - tiempo_inicio

        test_loss, test_acc = modelo.evaluate(X_test, Y_test, verbose=0)

        resultados.append({
            'neuronas': neuronas,
            'historia': historia,
            'test_acc': test_acc,
            'test_loss': test_loss,
            'tiempo': tiempo_total
        })

        print(f"Test accuracy: {test_acc:.4f}")
        print(f"Tiempo: {tiempo_total:.2f} segundos")

    return resultados

def visualizar_resultados_neuronas(resultados):
    neuronas = [r['neuronas'] for r in resultados]
    accuracies = [r['test_acc'] for r in resultados]
    tiempos = [r['tiempo'] for r in resultados]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    # Gráfica de accuracy
    ax1.semilogx(neuronas, accuracies, 'bo-')
    ax1.set_xlabel('Número de neuronas')
    ax1.set_ylabel('Test Accuracy')
    ax1.set_title('Accuracy vs Número de neuronas')
    ax1.grid(True)

    # Gráfica de tiempo
    ax2.semilogx(neuronas, tiempos, 'ro-')
    ax2.set_xlabel('Número de neuronas')
    ax2.set_ylabel('Tiempo de entrenamiento (s)')
    ax2.set_title('Tiempo vs Número de neuronas')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    # Curvas de aprendizaje del mejor modelo
    mejor_idx = np.argmax(accuracies)
    mejor_historia = resultados[mejor_idx]['historia']

    plt.figure(figsize=(10, 6))
    plt.plot(mejor_historia.history['accuracy'], label='train')
    plt.plot(mejor_historia.history['val_accuracy'], label='validation')
    plt.title(f'Curvas de aprendizaje para {neuronas[mejor_idx]} neuronas')
    plt.xlabel('Época')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

def tarea_e():
    print("Cargando y preprocesando datos...")
    X_train, Y_train, X_test, Y_test = cargar_y_preprocesar_cifar10()

    print("Iniciando experimentos...")
    neuronas_list = [16, 32, 64, 128, 256]
    resultados = experimentar_neuronas(X_train, Y_train, X_test, Y_test, neuronas_list)

    print("\nVisualizando resultados...")
    visualizar_resultados_neuronas(resultados)

    return resultados

# Tarea F: MLP multicapa
def crear_modelo_mlp_multicapa(arquitectura):
    modelo = Sequential([Flatten(input_shape=(32, 32, 3))])

    # Añadir capas ocultas según la arquitectura especificada
    for neuronas in arquitectura:
        modelo.add(Dense(neuronas, activation='relu'))

    # Capa de salida
    modelo.add(Dense(10, activation='softmax'))

    modelo.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return modelo

def experimentar_arquitecturas(X_train, Y_train, X_test, Y_test):
    # Lista de arquitecturas a probar
    arquitecturas = [
        [64],              # Una capa de 64
        [32, 32],         # Dos capas de 32
        [64, 32],         # Dos capas decrecientes
        [32, 64],         # Dos capas crecientes
        [64, 64],         # Dos capas iguales
        [128, 64, 32]     # Tres capas decrecientes
    ]

    resultados = []

    for arquitectura in arquitecturas:
        nombre_arq = ' -> '.join(str(n) for n in arquitectura)
        print(f"\nEntrenando modelo con arquitectura: {nombre_arq}")

        modelo = crear_modelo_mlp_multicapa(arquitectura)

        tiempo_inicio = time.time()
        historia = modelo.fit(
            X_train, Y_train,
            epochs=50,
            batch_size=256,
            validation_split=0.1,
            verbose=1
        )
        tiempo_total = time.time() - tiempo_inicio

        test_loss, test_acc = modelo.evaluate(X_test, Y_test, verbose=0)

        resultados.append({
            'arquitectura': nombre_arq,
            'historia': historia,
            'test_acc': test_acc,
            'test_loss': test_loss,
            'tiempo': tiempo_total
        })

        print(f"Test accuracy: {test_acc:.4f}")
        print(f"Tiempo: {tiempo_total:.2f} segundos")

    return resultados

def visualizar_resultados_arquitecturas(resultados):
    arquitecturas = [r['arquitectura'] for r in resultados]
    accuracies = [r['test_acc'] for r in resultados]
    tiempos = [r['tiempo'] for r in resultados]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    # Gráfica de accuracy
    x = range(len(arquitecturas))
    ax1.bar(x, accuracies, align='center')
    ax1.set_xticks(x)
    ax1.set_xticklabels(arquitecturas, rotation=45)
    ax1.set_ylabel('Test Accuracy')
    ax1.set_title('Accuracy vs Arquitectura')
    ax1.grid(True)

    # Gráfica de tiempo
    ax2.bar(x, tiempos, align='center', color='red')
    ax2.set_xticks(x)
    ax2.set_xticklabels(arquitecturas, rotation=45)
    ax2.set_ylabel('Tiempo de entrenamiento (s)')
    ax2.set_title('Tiempo vs Arquitectura')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    # Curvas de aprendizaje del mejor modelo
    mejor_idx = np.argmax(accuracies)
    mejor_historia = resultados[mejor_idx]['historia']

    plt.figure(figsize=(10, 6))
    plt.plot(mejor_historia.history['accuracy'], label='train')
    plt.plot(mejor_historia.history['val_accuracy'], label='validation')
    plt.title(f'Curvas de aprendizaje para arquitectura {arquitecturas[mejor_idx]}')
    plt.xlabel('Época')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

def tarea_f():
    print("Cargando y preprocesando datos...")
    X_train, Y_train, X_test, Y_test = cargar_y_preprocesar_cifar10()

    print("Iniciando experimentos...")
    resultados = experimentar_arquitecturas(X_train, Y_train, X_test, Y_test)

    print("\nVisualizando resultados...")
    visualizar_resultados_arquitecturas(resultados)

    return resultados

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Tarea G: CNN básica
def crear_cnn_basica(incluir_maxpool=False):
    modelo = Sequential()

    # Primera capa convolucional
    modelo.add(Conv2D(16, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    if incluir_maxpool:
        modelo.add(MaxPooling2D((2, 2)))

    # Segunda capa convolucional
    modelo.add(Conv2D(32, (3, 3), activation='relu'))
    if incluir_maxpool:
        modelo.add(MaxPooling2D((2, 2)))

    # Capas de clasificación
    modelo.add(Flatten())
    modelo.add(Dense(10, activation='softmax'))

    modelo.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return modelo

def experimentar_cnn_basica(X_train, Y_train, X_test, Y_test):
    resultados = []

    for incluir_maxpool in [False, True]:
        nombre = "CNN con MaxPool" if incluir_maxpool else "CNN sin MaxPool"
        print(f"\nEntrenando {nombre}...")

        modelo = crear_cnn_basica(incluir_maxpool)

        tiempo_inicio = time.time()
        historia = modelo.fit(
            X_train, Y_train,
            epochs=50,
            batch_size=256,
            validation_split=0.1,
            verbose=1
        )
        tiempo_total = time.time() - tiempo_inicio

        test_loss, test_acc = modelo.evaluate(X_test, Y_test, verbose=0)

        resultados.append({
            'tipo': nombre,
            'historia': historia,
            'test_acc': test_acc,
            'test_loss': test_loss,
            'tiempo': tiempo_total
        })

        print(f"Test accuracy: {test_acc:.4f}")
        print(f"Tiempo: {tiempo_total:.2f} segundos")

    return resultados

# Tarea H: Ajuste de kernel_size
def crear_cnn_kernel(kernel_size):
    modelo = Sequential([
        Conv2D(16, kernel_size, activation='relu', input_shape=(32, 32, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(32, kernel_size, activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(10, activation='softmax')
    ])

    modelo.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return modelo

def experimentar_kernel_sizes(X_train, Y_train, X_test, Y_test, kernel_sizes=[(2,2), (3,3), (4,4), (5,5)]):
    resultados = []

    for kernel_size in kernel_sizes:
        print(f"\nEntrenando CNN con kernel_size={kernel_size}...")
        modelo = crear_cnn_kernel(kernel_size)

        tiempo_inicio = time.time()
        historia = modelo.fit(
            X_train, Y_train,
            epochs=50,
            batch_size=256,
            validation_split=0.1,
            verbose=1
        )
        tiempo_total = time.time() - tiempo_inicio

        test_loss, test_acc = modelo.evaluate(X_test, Y_test, verbose=0)

        resultados.append({
            'kernel_size': f"{kernel_size[0]}x{kernel_size[1]}",
            'historia': historia,
            'test_acc': test_acc,
            'test_loss': test_loss,
            'tiempo': tiempo_total
        })

        print(f"Test accuracy: {test_acc:.4f}")
        print(f"Tiempo: {tiempo_total:.2f} segundos")

    return resultados

# Tarea I: Optimización de arquitectura CNN
def crear_cnn_optimizada(arquitectura):
    modelo = Sequential()

    # Primera capa convolucional (siempre presente)
    modelo.add(Conv2D(arquitectura[0], (3, 3), activation='relu', input_shape=(32, 32, 3)))
    modelo.add(MaxPooling2D((2, 2)))

    # Capas convolucionales adicionales
    for filtros in arquitectura[1:]:
        modelo.add(Conv2D(filtros, (3, 3), activation='relu'))
        modelo.add(MaxPooling2D((2, 2)))

    # Capas de clasificación
    modelo.add(Flatten())
    modelo.add(Dense(128, activation='relu'))
    modelo.add(Dense(10, activation='softmax'))

    modelo.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return modelo

def experimentar_arquitecturas_cnn(X_train, Y_train, X_test, Y_test):
    arquitecturas = [
        [32],                  # Una capa conv
        [32, 64],             # Dos capas conv
        [32, 64, 128],        # Tres capas conv
        [64, 128, 256],       # Tres capas con más filtros
        [16, 32, 64, 128]     # Cuatro capas incrementales
    ]

    resultados = []

    for arquitectura in arquitecturas:
        nombre = ' -> '.join(str(f) for f in arquitectura)
        print(f"\nEntrenando CNN con arquitectura: {nombre}")

        modelo = crear_cnn_optimizada(arquitectura)

        tiempo_inicio = time.time()
        historia = modelo.fit(
            X_train, Y_train,
            epochs=50,
            batch_size=256,
            validation_split=0.1,
            verbose=1
        )
        tiempo_total = time.time() - tiempo_inicio

        test_loss, test_acc = modelo.evaluate(X_test, Y_test, verbose=0)

        resultados.append({
            'arquitectura': nombre,
            'historia': historia,
            'test_acc': test_acc,
            'test_loss': test_loss,
            'tiempo': tiempo_total
        })

        print(f"Test accuracy: {test_acc:.4f}")
        print(f"Tiempo: {tiempo_total:.2f} segundos")

    return resultados

# Funciones principales de cada tarea
def tarea_g():
    print("Cargando y preprocesando datos...")
    X_train, Y_train, X_test, Y_test = cargar_y_preprocesar_cifar10()

    print("Iniciando experimentos...")
    resultados = experimentar_cnn_basica(X_train, Y_train, X_test, Y_test)

    print("\nVisualizando resultados...")
    visualizar_resultados_comparativos(resultados)

    return resultados

def tarea_h():
    print("Cargando y preprocesando datos...")
    X_train, Y_train, X_test, Y_test = cargar_y_preprocesar_cifar10()

    print("Iniciando experimentos...")
    kernel_sizes = [(2,2), (3,3), (4,4), (5,5)]
    resultados = experimentar_kernel_sizes(X_train, Y_train, X_test, Y_test, kernel_sizes)

    print("\nVisualizando resultados...")
    visualizar_resultados_comparativos(resultados)

    return resultados

def tarea_i():
    print("Cargando y preprocesando datos...")
    X_train, Y_train, X_test, Y_test = cargar_y_preprocesar_cifar10()

    print("Iniciando experimentos...")
    resultados = experimentar_arquitecturas_cnn(X_train, Y_train, X_test, Y_test)

    print("\nVisualizando resultados...")
    visualizar_resultados_comparativos(resultados)

    return resultados

def cargar_imagen_y_redimensionar(ruta_imagen):
    try:
        # Cargar imagen
        imagen = Image.open(ruta_imagen)

        # Convertir a RGB si es necesario
        if imagen.mode != 'RGB':
            imagen = imagen.convert('RGB')

        # Redimensionar a 32x32
        imagen = imagen.resize((32, 32), Image.Resampling.LANCZOS)

        # Convertir a array numpy
        array_imagen = np.array(imagen)

        return array_imagen
    except Exception as e:
        print(f"Error procesando {ruta_imagen}: {str(e)}")
        return None

def cargar_dataset_propio(directorio_base):
    categorias = ['avion', 'automovil', 'pajaro', 'gato', 'ciervo',
                 'perro', 'rana', 'caballo', 'barco', 'camion']

    X = []  # Imágenes
    Y = []  # Etiquetas

    for idx, categoria in enumerate(categorias):
        directorio_categoria = os.path.join(directorio_base, categoria)

        # Verificar que existe el directorio
        if not os.path.exists(directorio_categoria):
            print(f"No se encontró el directorio para {categoria}")
            continue

        # Cargar todas las imágenes de la categoría
        for archivo in os.listdir(directorio_categoria):
            if archivo.lower().endswith(('.png', '.jpg', '.jpeg')):
                ruta_completa = os.path.join(directorio_categoria, archivo)
                imagen_procesada = cargar_imagen_y_redimensionar(ruta_completa)

                if imagen_procesada is not None:
                    X.append(imagen_procesada)
                    Y.append(idx)

    # Convertir a arrays numpy
    X = np.array(X)
    Y = np.array(Y)

    # Normalizar valores de píxeles a [0,1]
    X = X.astype('float32') / 255.0

    # Convertir etiquetas a one-hot encoding
    Y = keras.utils.to_categorical(Y, 10)

    return X, Y

def evaluar_modelo_con_dataset_propio(modelo, X, Y):
    # Evaluar el modelo
    test_loss, test_acc = modelo.evaluate(X, Y, verbose=0)

    # Obtener predicciones
    Y_pred = modelo.predict(X)

    # Crear matriz de confusión
    visualizar_matriz_confusion(Y, Y_pred, "Matriz de Confusión - Dataset Propio")

    return test_loss, test_acc, Y_pred

def comparar_modelos_dataset_propio(modelos, X, Y):
    resultados = []

    for nombre, modelo in modelos.items():
        print(f"\nEvaluando modelo: {nombre}")
        test_loss, test_acc, _ = evaluar_modelo_con_dataset_propio(modelo, X, Y)

        resultados.append({
            'nombre': nombre,
            'test_acc': test_acc,
            'test_loss': test_loss
        })

        print(f"Test accuracy: {test_acc:.4f}")
        print(f"Test loss: {test_loss:.4f}")

    # Visualizar comparación
    nombres = [r['nombre'] for r in resultados]
    accuracies = [r['test_acc'] for r in resultados]

    plt.figure(figsize=(10, 6))
    plt.bar(nombres, accuracies)
    plt.title('Comparación de modelos en dataset propio')
    plt.xlabel('Modelo')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return resultados

# Función principal para la Tarea J
def tarea_j():
    print("Cargando dataset propio...")
    X, Y = cargar_dataset_propio('dataset_propio')

    print("\nForma del dataset:")
    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")

    # Mostrar algunas imágenes de ejemplo
    indices = np.random.choice(len(X), 5, replace=False)
    for idx in indices:
        plt.figure()
        plt.imshow(X[idx])
        plt.title(f"Clase: {np.argmax(Y[idx])}")
        plt.axis('off')
        plt.show()

    return X, Y

# Función principal para la Tarea K
def tarea_k(X_propio, Y_propio):
    # Definir diferentes configuraciones a probar
    configuraciones = {
        'MLP_base': crear_modelo_mlp(),
        'MLP_grande': crear_modelo_mlp_neuronas(128),
        'CNN_base': crear_cnn_basica(True),
        'CNN_optimizada': crear_cnn_optimizada([32, 64, 128])
    }

    # Entrenar y evaluar cada configuración
    resultados = comparar_modelos_dataset_propio(configuraciones, X_propio, Y_propio)

    return resultados

# Función principal para la Tarea L
def tarea_l(X_propio, Y_propio):
    # Crear modelo con mejoras
    modelo_mejorado = Sequential([
        # Aumento de datos
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.1),

        # CNN con regularización
        Conv2D(32, (3, 3), activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        # Clasificación
        Flatten(),
        Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])

    modelo_mejorado.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Entrenar y evaluar
    resultados = evaluar_modelo_con_dataset_propio(modelo_mejorado, X_propio, Y_propio)

    return resultados, modelo_mejorado

def visualizar_resultados_comparativos(resultados):
    # Extraer datos para la visualización
    nombres = []
    accuracies = []
    tiempos = []

    for r in resultados:
        nombre = r.get('tipo') or r.get('arquitectura') or r.get('kernel_size') or ''
        nombres.append(str(nombre))  # Convertir a string para evitar errores
        accuracies.append(r['test_acc'])
        tiempos.append(r['tiempo'])

    # Crear figura con dos subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Gráfica de accuracy
    x = range(len(nombres))
    ax1.bar(x, accuracies)
    ax1.set_xticks(x)
    ax1.set_xticklabels(nombres, rotation=45)
    ax1.set_ylabel('Test Accuracy')
    ax1.set_title('Comparación de Accuracy')
    ax1.grid(True)

    # Gráfica de tiempo
    ax2.bar(x, tiempos, color='red')
    ax2.set_xticks(x)
    ax2.set_xticklabels(nombres, rotation=45)
    ax2.set_ylabel('Tiempo de entrenamiento (s)')
    ax2.set_title('Comparación de Tiempos')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    # Mostrar curvas de aprendizaje del mejor modelo
    mejor_idx = np.argmax(accuracies)
    mejor_historia = resultados[mejor_idx]['historia']

    plt.figure(figsize=(10, 6))
    plt.plot(mejor_historia.history['accuracy'], label='train')
    plt.plot(mejor_historia.history['val_accuracy'], label='validation')
    plt.title(f'Curvas de aprendizaje para el mejor modelo ({nombres[mejor_idx]})')
    plt.xlabel('Época')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    tarea = input("Elegir tarea (B-L): ").upper()

    if tarea in ['J', 'K', 'L']:
        # Cargar dataset propio una vez
        X_propio, Y_propio = cargar_dataset_propio('dataset_propio')

    switches = {
        'B': tarea_b,
        'C': tarea_c,
        'D': tarea_d,
        'E': tarea_e,
        'F': tarea_f,
        'G': tarea_g,
        'H': tarea_h,
        'I': tarea_i,
        'J': lambda: tarea_j(),
        'K': lambda: tarea_k(X_propio, Y_propio),
        'L': lambda: tarea_l(X_propio, Y_propio)
    }

    if tarea in switches:
        switches[tarea]()
    else:
        print("Tarea no válida")
