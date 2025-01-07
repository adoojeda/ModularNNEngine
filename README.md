# Neural Network

Este proyecto implementa una red neuronal desde cero en Python, sin hacer uso de librerías de aprendizaje profundo como TensorFlow o PyTorch. La red neuronal se entrena y evalúa en tres conjuntos de datos clásicos: Iris, Digits y Wine.

## Introducción

Las redes neuronales artificiales son herramientas fundamentales en el aprendizaje automático, con aplicaciones en clasificación, reconocimiento de patrones y otros problemas complejos. Este proyecto aborda el diseño, entrenamiento y evaluación de una red neuronal modular, centándose en la construcción de capas, funciones de activación, algoritmos de retropropagación y optimizadores.

## Estructura del Proyecto

### Funciones de Activación (*activations.py*)
Se incluyen funciones de activación como ReLU, Sigmoid, Tanh y Softmax, junto con sus derivadas, esenciales para dar no linealidad y permitir el aprendizaje de relaciones complejas.

### Clase de Capa (*layer.py*)
Define la estructura de una capa individual en la red. Incluye atributos como dimensiones de entrada/salida y métodos como `forward` para la propagación hacia adelante.

### Red Neuronal (*neural_network.py*)
Clase principal que gestiona la red completa, incluyendo la adición de capas, feedforward y retropropagación. También incluye métricas de evaluación como precisión y pérdida.

### Preprocesamiento (*preprocessing.py*)
Incluye operaciones como:
- **Escalado** con `StandardScaler` para normalizar las características.
- **Codificación One-Hot** para etiquetas.

### Optimizadores (*optimizers.py*)
Contiene el algoritmo de descenso de gradiente y su variante Momentum, utilizados para ajustar los pesos durante el entrenamiento.

### Visualización (*visualizations.py*)
Proporciona herramientas como:
- **Matriz de confusión** para comparar predicciones con etiquetas reales.
- **Curvas ROC** para evaluar la tasa de verdaderos positivos y falsos positivos.

### Carga de Datos (*load_data.py*)
Funciones para cargar los conjuntos de datos Iris, Digits y Wine desde scikit-learn, asegurando compatibilidad con las siguientes etapas del flujo de trabajo.

### Notebook Principal (*main.ipynb*)
Centraliza todas las etapas del proyecto:
1. Carga y preprocesamiento de datos.
2. Diseño y entrenamiento de modelos.
3. Evaluación y visualización de resultados.

## Conjuntos de Datos

### Iris
- 150 muestras de tres clases de flores.
- Atributos: longitud y ancho del sépalo y pétalo.

### Digits
- 1797 imágenes de dígitos escritos a mano (resolución de 8x8).
- 10 clases correspondientes a los dígitos del 0 al 9.

### Wine
- 178 muestras de vino clasificadas en tres clases.
- 13 atributos como nivel de alcohol, fenoles totales y acidez.

## Experimentos y Resultados

Se evaluaron diferentes arquitecturas para los conjuntos de datos:

### Iris
1. **Modelo 1:** Capa oculta (5 neuronas, ReLU) y salida (3 neuronas, Softmax). Precisión: 97%.
2. **Modelo 2:** Tres capas ocultas (20, 10, 6 neuronas). Precisión: 93%.

### Digits
1. **Modelo 1:** Capa oculta (5 neuronas, Tanh) y salida (10 neuronas, Softmax). Precisión: 83.6%.
2. **Modelo 2:** Tres capas ocultas (32, 16, 8 neuronas, Sigmoid). Precisión: 90.8%.

### Wine
1. **Modelo 1:** Arquitectura básica con Sigmoid. Precisión: 100%.

## Ejecución del Proyecto

### Requisitos
- Python 3.10+
- Librerías: `numpy`, `scikit-learn`, `matplotlib`

Instala las dependencias:
```bash
pip install numpy scikit-learn matplotlib
```

### Uso
Ejecuta el notebook principal *main.ipynb* para:
1. Cargar y preprocesar datos.
2. Entrenar y evaluar diferentes arquitecturas.
3. Visualizar resultados mediante matrices de confusión y curvas ROC.

## Conclusiones
El proyecto destaca la importancia de equilibrar la complejidad del modelo y la capacidad de generalización, con arquitecturas simples logrando alto rendimiento en datos bien estructurados. Optimizar hiperparámetros y funciones de activación es crucial para mejorar la precisión.

## Trabajo Futuro
1. Incorporar regularización como Dropout o Batch Normalization.
2. Ampliar soporte a arquitecturas profundas (CNN, RNN).
3. Evaluar el rendimiento en tareas de texto, series temporales y clasificación de imágenes.
4. Implementar un dashboard para visualización interactiva.

## Colaboradores del Proyecto
- Joel Clemente López Cabrera
- Adonai Ojeda Martín
- Daniel Medina González

---

Repositorio GitHub: [Modular Neural Network Engine](https://github.com/adoojeda/ModularNNEngine).

