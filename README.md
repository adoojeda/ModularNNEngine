# Neural Network 

Este proyecto implementa una red neuronal desde cero en Python, sin hacer uso de librerias de aprendizaje profundo como puede ser TensorFlow o Pytorch. La red neural se entrena y evalúa en dos conjuntos de datos clásicos: Iris y Digits.

## Estructura del Proyecto

En el archivo *activations.py*, se definen las funciones de activación que son esenciales para dar no linealidad a la red neuronal. Entre estas funciones se encuentran relu, softmax, tanh y sigmoid. Estas funciones se utilizan en las capas de la red neuronal para transformar las señales que fluyen entre las neuronas, lo cual permite al modelo aprender representaciones complejas de los datos.

El archivo *layer.py* contiene la implementación de la clase Layer, que representa una capa individual en la red neuronal. La clase Layer incluye atributos como el tamaño de la entrada y salida, la función de activación y su derivada. También define métodos como forward, que realiza la propagación hacia adelante en una capa específica. Cada capa en la red puede configurarse con distintos tamaños y funciones de activación, permitiendo la creación de arquitecturas personalizadas.

En *neural_network.py* se encuentra la clase principal NeuralNetwork, que maneja la estructura completa de la red neuronal. La clase NeuralNetwork permite añadir múltiples capas (usando la clase Layer) y define métodos fundamentales como feedforward, para la propagación hacia adelante a través de todas las capas, y backpropagation, para ajustar los pesos de todas las capas mediante la retropropagación. Además, el método accuracy calcula la precisión del modelo comparando las etiquetas verdaderas con las predichas, lo que permite evaluar el rendimiento de la red en distintas etapas del entrenamiento.

El archivo *optimizers.py* contiene los algoritmos de optimización utilizados durante el entrenamiento de la red neuronal. En este caso, el archivo incluye la implementación de gradient_descent, una función que aplica el algoritmo de descenso de gradiente para ajustar los pesos del modelo durante múltiples épocas de entrenamiento. Este optimizador toma el modelo, los datos de entrenamiento, la tasa de aprendizaje y la función de precisión, permitiendo un entrenamiento supervisado.

Para preparar los datos antes de entrenar la red, el archivo *preprocessing.py* ofrece la función preprocess_data. Esta función se encarga de transformar los datos de entrada y las etiquetas, aplicando operaciones como normalización, que mejoran la eficiencia y precisión del modelo durante el entrenamiento.

La visualización de métricas y resultados se realiza a través del archivo *visualizations.py*, que incluye funciones como plot_confusion_matrix, para generar una matriz de confusión que muestra el rendimiento del modelo en términos de clasificaciones correctas e incorrectas. Además, roc_curve se utiliza para calcular y mostrar la curva ROC, una métrica que permite analizar la tasa de verdaderos positivos frente a los falsos positivos para cada clase en un problema de clasificación.

El archivo *load_data.py* en el directorio data se utiliza para cargar los conjuntos de datos Iris y Digits, que son ampliamente utilizados para tareas de clasificación y benchmarking en redes neuronales. Este archivo contiene funciones como load_iris_data y load_digits_data, que cargan y devuelven los datos junto con sus etiquetas, listos para ser preprocesados y utilizados en el entrenamiento de la red.

Finalmente, *main.ipynb* es el script principal que coordina la ejecución del proyecto. Este archivo de notebook contiene el flujo de trabajo completo: carga y preprocesamiento de los datos, creación y entrenamiento de diferentes arquitecturas de redes neuronales, evaluación de la precisión del modelo antes y después del entrenamiento, y visualización de resultados con curvas ROC. En el notebook se crean y evalúan distintos modelos, probando varias configuraciones de capas y funciones de activación para analizar su rendimiento en los conjuntos de datos Iris y Digits.

## Ejecución del proyecto.

Versión de Python, 3.10 en adelante.
Bibliotecas necesarias:
    numpy
    scikit-learn
    matplotlib

Puedes instalar las liberías con el comando :
pip install (nombre de la libreria)

### Uso
Para entrenar y evaluar la red neuronal, ejecuta el notebook main.ipynb, el cual:
* Carga los datos: Utiliza load_iris_data y load_digits_data para cargar y preprocesar los datos de Iris y Digits respectivamente.

Entrenamiento y Evaluación de Modelos:
* Crea dos arquitecturas de red diferentes para cada conjunto de datos.
* Imprime la precisión antes y después del entrenamiento.

Visualización:
* Muestra las curvas ROC para cada modelo y clase, lo cual ayuda a analizar el rendimiento del modelo en términos de tasa de verdaderos positivos y falsos positivos.

## Ejemplo de Arquitecturas
### Para el conjunto de datos Iris:

Modelo 1: Capa oculta con 5 neuronas (ReLU) y capa de salida con 3 neuronas (softmax).
Modelo 2: Red profunda con 3 capas ocultas y softmax en la salida.
Para el conjunto de datos Digits:

### Para el conjunto de datos Digits:
Modelo 1: Capa oculta con 5 neuronas (tanh) y capa de salida con 10 neuronas (softmax).
Modelo 2: Red profunda con 3 capas ocultas de diferentes tamaños y softmax en la salida.

# Visualización de Resultados
Al final del entrenamiento, el notebook muestra las curvas ROC para cada modelo, ayudando a evaluar su desempeño en términos de precisión y sensibilidad.

# Colaboradores del protecto
Para este proyecto han colaborado los alumnos:
* Joel Clemente López Cabrera.
* Adonai Ojeda Martín.
* Daniel Medina González

