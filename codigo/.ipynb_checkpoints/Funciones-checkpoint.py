import os
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import keras
from keras import layers
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from keras.callbacks import EarlyStopping


def buscar_imagen(directorio_padre, nombre_imagen):
    '''
    Función empleada para encontrar una imagen concreta (a partir de su nombre) dentro de cualquiera de las subcarpetas del directorio_padre.
    ---------------------------------------------------------
    Parámetros:
    - directorio_padre: ruta donde se encuentra la carpeta cada una de las subcarpetas con las imágenes de radiografías de tórax
    - nombre_imagen: nombre de la imágen a la que se desea acceder 
    ----------------------------------------------------------
    Return:
    - ruta_imagen: ruta completa de la imágen a la que se desea acceder 
    '''
    # Subcarpetas principales en las que buscar
    subcarpetas_principales = ['train', 'test', 'val']
    # Subcarpetas adicionales en las que buscar dentro de cada subcarpeta principal
    subcarpetas_adicionales = ['NORMAL', 'PNEUMONIA']

    # Se itera sobre las subcarpetas principales
    for subcarpeta_principal in subcarpetas_principales:
        # Se itera sobre las subcarpetas adicionales dentro de cada subcarpeta principal
        for subcarpeta_adicional in subcarpetas_adicionales:
            # Se obtiene la ruta completa de la imagen
            ruta_imagen = os.path.join(directorio_padre, subcarpeta_principal, subcarpeta_adicional, nombre_imagen)
            # verificar si la imagen existe en la subcarpeta actual
            if os.path.exists(ruta_imagen):
                return ruta_imagen  # devolver la ruta de la imagen si se encuentra



def redestribucion_imagenes(directorio_principal):

    '''
    Función empleada para redestribuir las imágenes ubicadas en distintas subcarpetas dentro de la carpeta data en una carpeta nueva con las
    mismas subcarpetas pero con un porcentaje distinto de imágenes en cada subcarpeta. La distribución quedaría de la siguiente manera:
    - test: 20% del total
    - train: 64% del total
    - val: 16% del total
    De igual forma, la distribución de las carpetas "PNEUMONIA" y "NORMAL" también queda de forma proporcional.
    --------------------------------------------------------------------
    Parámetros:
    - directorio_principal: ruta donde se encuentra la carpeta data con cada una de las subcarpetas con las imágenes de radiografías de tórax
    -------------------------------------------------------------------
    Return: 
    - nada
    '''

    '''
    En primer lugar, se crea un csv con dos columnas nombres_ficheros y clases compuesto por todas las imágenes existentes en el directorio_padre.
    En la columna nombres_ficheros debe aparecer el nombre de TODAS las imágenes que existen dentro de cada subcarpeta y en la columna clases debe 
    aparecer 0 o 1 en función si se trata de una imagen de la carpeta NORMAL o PNEUMONIA respectivamente.
    '''

    directorio_padre = os.path.join(directorio_principal, 'data')
    
    # Listas para almacenar los nombres de las imágenes y las clases (0 o 1 en función de si es normal o neumonía respectivamente)
    nombres_ficheros = []
    clases = []
    
    # Recorremos las carpetas de train, test y val
    for subcarpeta in ['train', 'test', 'val']:
        ruta_subcarpeta = os.path.join(directorio_padre, subcarpeta)
        for clase in ['NORMAL', 'PNEUMONIA']:
            ruta_clase = os.path.join(ruta_subcarpeta, clase)
            for nombre_fichero in os.listdir(ruta_clase):
                nombres_ficheros.append(nombre_fichero)
                clases.append(0 if clase == 'NORMAL' else 1)
    
    # Se crea el DataFrame con los datos
    df_todas = pd.DataFrame({'nombre_fichero': nombres_ficheros,'clase': clases})
    
    # Se guarda el DataFrame en un archivo CSV
    ruta_csv = os.path.join(directorio_padre, 'dataset_info.csv') #el nuevo dataframe se guarda dentro del directorio padre
    df_todas.to_csv(ruta_csv, index=False)

    '''
    A partir del csv anterior y, con ayuda de la función train_test_split de skitlearn de divide el csv anterior en dos 
    subgrupos de train y test en proporción 80, 20 para poder usar el 80% de las imágenes para train y el 20% para test.
    También se emplea el parámetro stratify para que exista una proporción de clases en cada uno de los grupos, es decir, en ''NORMAL" y "PNEUMONIA".
    '''
    
    # se emplea train_test_split para dividir el dataset en train (80%) y test (20%)
    # random_state=42 se emplea para que cada vez que se ejecute el código, se obtenga la misma división de datos. El valor 42 es un valor que se usa
    # comunmente en este caso pero se puede emplear cualquie otro valor entero
    # stratify se emplea para agrupar de manera proporcional las clases neumonia y normal en los distintos dataframes
    train_df, test_df = train_test_split(df_todas, test_size=0.2, stratify=df_todas['clase'], random_state=42)
    
    # Se guardan los nuevos conjuntos de datos en archivos CSV
    ruta_train_csv = os.path.join(directorio_padre, 'train_dataset_info.csv') #el nuevo dataframe se guarda dentro del directorio padre
    ruta_test_csv = os.path.join(directorio_padre, 'test_dataset_info.csv') #el nuevo dataframe se guarda dentro del directorio padre
    train_df.to_csv(ruta_train_csv, index=False)
    test_df.to_csv(ruta_test_csv, index=False)

    '''
    A continuación, se coge el conjunto de datos obtenido previamente de train, es decir, el csv "train_df" y se repite el mismo
    proceso pero, esta vez dividiendo este conjunto de datos para train y val en un 80% y 20% respectivamente.
    De tal forma que, finalemnte se obtenga el conjunto de test que represeneta el 20% del total (obtenido previamente), el conjunto de train
    que representa el 80% del 80% del total ya que, inicialmente nos hemos quedado con el 80% pero luego, de este 80%, el 20% va destinado al conjunto
    de validación. Por lo que finalmete quedarías distribuidos de la siguiente manera:
    - test: 20% del total
    - train: 64% del total
    - val: 16% del total
    '''

    # Se emplea train_test_split para dividir el conjunto de datos de entrenamiento en train (80%) y val (20%)
    train_def_df, val_df = train_test_split(train_df, test_size=0.2, stratify=train_df['clase'], random_state=42)
    
    # Se guardan los nuevos conjuntos de datos en archivos CSV
    ruta_train_final_csv = os.path.join(directorio_padre, 'train_final_dataset_info.csv') #el nuevo dataframe se guarda dentro del directorio padre
    ruta_val_csv = os.path.join(directorio_padre, 'val_dataset_info.csv') #el nuevo dataframe se guarda dentro del directorio padre
    train_def_df.to_csv(ruta_train_final_csv, index=False)
    val_df.to_csv(ruta_val_csv, index=False)

    '''
    Finalmente, se crea una nueva carpeta denominada data_nuevo dentro del directorio principal. Dentro de esta carpeta se crean 3 subcarpetas 
    ("train", "test" y "val") que corresponderian con los dataframes obtenidos hasta hora: train_def_df, val_df y test_df y, dentro de estas 3 
    subcarpetas, se crean 2 carpetas "NORMAL" y "PNEUMONIA" que corresponden con con las clases determinadas en cada dataframe, 0 en caso de 
    "NORMAL" y 1 para "PNEUMONIA". Dentro de estas dos carpetas para ("train", "test" y "val") se encontraran las imagenes correspondientes 
    para cada caso según los dataframes obtenidos.
    '''

    # Se crea la nueva carpeta dentro del directorio principal
    ruta_principal_nueva = os.path.join(directorio_principal, 'data_nuevo') 

    # Se crean las carpetas 'train', 'test' y 'val' dentro de la nueva carpeta principal
    for subcarpeta in ['train', 'test', 'val']:
        ruta_subcarpeta = os.path.join(ruta_principal_nueva, subcarpeta)
        os.makedirs(ruta_subcarpeta, exist_ok=True) #verifica si la carpeta ruta_subcarpeta ya existe. Si existe, no se hace nada y el programa continúa su ejecución sin lanzar un error. Si no existe, la función os.makedirs() la crea junto con cualquier carpeta intermedia necesaria en la ruta especificada
        
        # Se crean las subcarpetas 'normal' y 'neumonia' dentro de cada subcarpeta ('train', 'test' y 'val')
        for clase in ['NORMAL', 'PNEUMONIA']:
            ruta_clase = os.path.join(ruta_subcarpeta, clase)
            os.makedirs(ruta_clase, exist_ok=True)
    
                
    # Se copian los archivos CSV a las subcarpetas correspondientes
    for df, nombre_carpeta in [(train_def_df, 'train'), (val_df, 'val'), (test_df, 'test')]:
        for index, row in df.iterrows(): #se itera sobre cada dataframe fila a fila
            clase = 'NORMAL' if row['clase'] == 0 else 'PNEUMONIA'
            nombre_archivo = row['nombre_fichero']
    
            # ruta de origen donde se busca la imagen concreta a partir de la función realizada previamente
            # esta ruta se refiere a donde esta que se desea guardar en la carpeta destino originalmente para poder copiarla
            ruta_origen=buscar_imagen(directorio_padre, nombre_archivo)
            
            # ruta donde se desa guardar (y redestribuir de la forma correcta) las imágenes
            ruta_destino = os.path.join(ruta_principal_nueva, nombre_carpeta, clase, nombre_archivo)
            
            shutil.copyfile(ruta_origen, ruta_destino) # copia las imágenes de la ruta incial a la ruta final



def preparar_modelo(ruta, batch_size,target_size):

    '''
    Función que configura los generadores de datos para entrenar, validar y probar un modelo de aprendizaje automático con imágenes.
    -----------------------------------------------------------
    Parámetros:
    - ruta: str. Ruta base donde se encuentran las imágenes organizadas en subcarpetas (train, val, test)
    - batchsize: int. Tamaño del lote que se utiliza en una única iteración del algoritmo de aprendizaje
    - target_size: tupla de números enteros que representa el alto y ancho al que se van a redimensionar todas las imágenes. 
    ----------------------------------------------------
    Return:
    - nada
    '''
    
    dir_general = ruta #ubicacion donde se encuentran las imágenes organizadas en subcarpetas (train, val, test). Añadir esta carpeta a one drive en TFG

    dir_train = os.path.join(dir_general, 'train')
    dir_validation = os.path.join(dir_general, 'val')
    dir_test = os.path.join(dir_general, 'test')
    
    # Preprocesamiento de imágenes
    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)
    validation_datagen=ImageDataGenerator(rescale=1./255)
    
    #Iterador que recorre el directorio de imágenes
    train_generator = train_datagen.flow_from_directory(
        dir_train,
        target_size=target_size, #cambiar a (150,150) si no se usa como AlexNet
        batch_size=batch_size, #lo más grande posible que no cause problemas de memoria 
        color_mode='rgb',
        class_mode='binary',
        classes=['NORMAL','PNEUMONIA'], #se indican las clases
        shuffle=True) # el conjunto de datos se barajará aleatoriamente para evitar sobreajuste 
    validation_generator = validation_datagen.flow_from_directory(
        dir_validation,
        target_size=target_size, #cambiar a (150,150) si no se usa como AlexNet
        batch_size=batch_size, #lo más grande posible que no cause problemas de memoria 
        color_mode='rgb',
        class_mode='binary',
        classes=['NORMAL','PNEUMONIA'], #se indican las clases
        shuffle=False)
    
    test_generator = test_datagen.flow_from_directory(
        dir_test,
        target_size=target_size, #cambiar a (150,150) si no se usa como AlexNet
        batch_size=batch_size, #lo más grande posible que no cause problemas de memoria 
        color_mode='rgb',
        class_mode='binary',
        classes=['NORMAL','PNEUMONIA'], #se indican las clases
        shuffle=False)
    
    return train_generator, validation_generator, test_generator


def metricas(y_test, y_pred):
    '''
    Funcicón que calcula distintas métricas para la evaluación del modelo.
    -----------------------------------------------------
    Parámetros: 
    - y_test: array de etiquetas verdaderas del conjunto de prueba
    - y_pred: array de etiquetas predichas por el modelo
    ----------------------------------------
    Return: 
    - accuracy: float que indica la proporción de predicciones correctas
    - precision: float que indica la proporción de predicciones positivas correctas
    - recall: float que indica la proporción de positivos detectados
    - f1: float que indica la media armónica de precisión y exhaustividad para evaluar de una forma más equilibrada el rendimiento del modelo
    - specificity: float que indica la proporción de negativos detectados
    - fpr: float que indica la tasa de falsos positivos, es decir, la proporción de negativos incorrectamente clasificadas como positivos, 
    respecto al total de casos negativos reales.
    - fnr: float que indica la tasa de falsos negativos, es decir, la proporción de positivos incorrectamente clasificadas como negativos, 
    respecto al total de casos positivos reales.
    - auc: float que se emplea para evaluar la capacidad de distinción entre clases positivas y negativas de un modelo de clasificación 
    binaria. Un 1 significa que es capaz de distinguir perfectamente entre clases, un 0.5 significa una clasificación aleatoria y un 0 indica 
    que ninguna clase ha sido correctamente clasificada.
    '''
    
    y_pred_bin=np.where(y_pred>=0.5,1,0) #para convertirlo en un problema binario
    
    #se obtienen los verdaderos negativos, falsos positivos, falsos negativos y verdaderos positivos a partir de la matriz de confusión 
    #con .ravel() se convierte la matriz en un array unidimensional
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_bin).ravel() 

    #se calculan cada una de las métricas empleando su correspondiente fórmula
    accuracy = (tp + tn)/(tn + fp + fn + tp)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * ((precision*recall)/(precision+recall))
    specificity = tn / (tn + fp)
    fpr = fp / (fp + tn) #tasa de falsos positivos
    fnr = fn / (fn + tp) #tasa de falsos negativos
    auc = roc_auc_score(y_test, y_pred)

    
    return [accuracy, precision, recall, f1, specificity, fpr, fnr, auc] #se devuleve como una lista para poder trabajar correctmante con las métricas


    
def establecer_arquitectura_propia(tipo):

    '''
    Función que establece distintos tipos de modelos de red neuronal convolucional (CNN) según el tipo que se introduzca como parámetro.
    --------------------------------------------------------------
    Parámetros
    - tipo: str que indica el tipo de modelo al que se quiere acceder 
    -------------------------------------------------------------
    Return
    -model: modelo sequencial en keras según el tipo que se haya introducido como parámetro de entrada y que contiene toda la información necesaria 
    sobre la arquitectura del modelo
    '''
    
    input_shape=(150,150,3) # se define el tamaño de entrada de las imágenes

    '''
    El modelo Simple1, se corresponde con un modelo que posee varias capas convolucionales (con las que se obtienen características importantes
    de las imágenes) seguidas de capas de MaxPooling2D para reducir la dimensionalidad. Después del Flatten se encuentra una capa densa.
    La función de activación sigmoide en la capa de salida produce una probabilidad entre 0 y 1 para la clasificación binaria.
    Este modelo es muy simple y los resultados no van a ser buenos.
    '''
    
    
    if tipo == "Simple1":
        model = keras.Sequential(
            [
                keras.Input(shape=input_shape),
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(), #convierte imágenes en vectores
                layers.Dropout(0.2), 
                layers.Dense(1, activation="sigmoid"), #produce una probabilidad entre 0 y 1 para la clasificación binaria
            ]
        )
        
        '''
    El modelo Simple2, se corresponde con un modelo que posee varias capas convolucionales (con las que se obtienen características importantes
    de las imágenes) seguidas de capas de MaxPooling2D para reducir la dimensionalidad. Después del Flatten se encuentra una capa oculta de 
    100 unidades y una capa densa.
    La función de activación sigmoide en la capa de salida produce una probabilidad entre 0 y 1 para la clasificación binaria.
    '''

    elif tipo == "Simple2":
        model = keras.Sequential(
            [
                keras.Input(shape=input_shape),
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(), #convierte imágenes en vectores
                layers.Dense(100, activation="relu"), #100 neuronas en la primera capa
                layers.Dropout(0.2),
                layers.Dense(1, activation="sigmoid"), #produce una probabilidad entre 0 y 1 para la clasificación binaria
            ]
        )
        '''
    El modelo Simple3, se corresponde con un modelo que posee varias capas convolucionales (con las que se obtienen características importantes
    de las imágenes) seguidas de capas de MaxPooling2D para reducir la dimensionalidad. Después del Flatten se encuentra una capa se encuentra 
    una capa oculta de 100 neuronas, una segunda capa oculta de 16 neuronas y una capa densa.
    La función de activación sigmoide en la capa de salida produce una probabilidad entre 0 y 1 para la clasificación binaria.
    '''

    elif tipo == "Simple3":
        model = keras.Sequential(
            [
                keras.Input(shape=input_shape),
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(), #convierte imágenes en vectores
                layers.Dense(100, activation="relu"), #100 neuronas en la primera capa
                layers.Dropout(0.2),
                layers.Dense(16, activation="relu"), #16 neuronas en la segunda capa
                layers.Dropout(0.2),
                layers.Dense(1, activation="sigmoid"), #produce una probabilidad entre 0 y 1 para la clasificación binaria
            ]
        )
    else: #si no se cumple ninguna de las opciones anteriores, aparece un error
        raise ValueError("Tipo de arquitectura no reconocida")
    
    return model


def establecer_arquitectura_AlexaNet(tipo):

    '''
    Función que establece distintos tipos de modelos de red neuronal convolucional (CNN) según el tipo que se introduzca como parámetro.
    --------------------------------------------------------------
    Parámetros
    - tipo: str que indica el tipo de modelo al que se quiere acceder 
    -------------------------------------------------------------
    Return
    -model: modelo sequencial en keras según el tipo que se haya introducido como parámetro de entrada y que contiene toda la información necesaria 
    sobre la arquitectura del modelo
    '''
    
    input_shape=(340,340,3) # se define el tamaño de entrada de las imágenes

    '''
    El modelo Simple1, se corresponde con un modelo que posee varias capas convolucionales (con las que se obtienen características importantes
    de las imágenes) seguidas de capas de MaxPooling2D para reducir la dimensionalidad. Después del Flatten se encuentra una capa densa.
    La función de activación sigmoide en la capa de salida produce una probabilidad entre 0 y 1 para la clasificación binaria.
    Este modelo es muy simple y los resultados no van a ser buenos.
    '''

    
    
    if tipo == "Simple1":
        model = keras.Sequential(
            [
                keras.Input(shape=input_shape),
                layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), padding='valid', activation='relu'),
                layers.MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'),
                layers.BatchNormalization(),
                
                layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), padding='valid', activation='relu'),
                layers.MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'),
                layers.BatchNormalization(),
                
                layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'),
                layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'),
                layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'),
                layers.MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'),
                layers.BatchNormalization(),
                
                layers.Flatten(), #convierte imágenes en vectores
                layers.Dropout(0.2), #cambiar a menos de 0,5 
                layers.Dense(1, activation="sigmoid"), #produce una probabilidad entre 0 y 1 para la clasificación binaria
            ]
        )
        
        '''
    El modelo Simple2, se corresponde con un modelo que posee varias capas convolucionales (con las que se obtienen características importantes
    de las imágenes) seguidas de capas de MaxPooling2D para reducir la dimensionalidad. Después del Flatten se encuentra una capa oculta de 
    100 unidades y una capa densa.
    La función de activación sigmoide en la capa de salida produce una probabilidad entre 0 y 1 para la clasificación binaria.
    '''

    elif tipo == "Simple2":
        model = keras.Sequential(
            [
                keras.Input(shape=input_shape),
                layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), padding='valid', activation='relu'),
                layers.MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'),
                layers.BatchNormalization(),
                
                layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), padding='valid', activation='relu'),
                layers.MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'),
                layers.BatchNormalization(),
                
                layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'),
                layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'),
                layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'),
                layers.MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'),
                layers.BatchNormalization(),
                
                layers.Flatten(), #convierte imágenes en vectores
                layers.Dense(100, activation="relu"), #100 neuronas en la primera capa
                layers.Dropout(0.2),
                layers.Dense(1, activation="sigmoid"), #produce una probabilidad entre 0 y 1 para la clasificación binaria
            ]
        )
        '''
    El modelo Simple3, se corresponde con un modelo que posee varias capas convolucionales (con las que se obtienen características importantes
    de las imágenes) seguidas de capas de MaxPooling2D para reducir la dimensionalidad. Después del Flatten se encuentra una capa se encuentra 
    una capa oculta de 100 neuronas, una segunda capa oculta de 16 neuronas y una capa densa.
    La función de activación sigmoide en la capa de salida produce una probabilidad entre 0 y 1 para la clasificación binaria.
    '''

    elif tipo == "Simple3":
        model = keras.Sequential(
            [
                keras.Input(shape=input_shape),
                layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), padding='valid', activation='relu'),
                layers.MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'),
                layers.BatchNormalization(),
                
                layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), padding='valid', activation='relu'),
                layers.MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'),
                layers.BatchNormalization(),
                
                layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'),
                layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'),
                layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'),
                layers.MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'),
                layers.BatchNormalization(),
                
                layers.Flatten(), #convierte imágenes en vectores
                layers.Dense(100, activation="relu"), #100 neuronas en la primera capa
                layers.Dropout(0.2),
                layers.Dense(16, activation="relu"), #16 neuronas en la segunda capa
                layers.Dropout(0.2),
                layers.Dense(1, activation="sigmoid"), #produce una probabilidad entre 0 y 1 para la clasificación binaria
            ]
        )
    else: #si no se cumple ninguna de las opciones anteriores, aparece un error
        raise ValueError("Tipo de arquitectura no reconocida")
    
    return model 



def arq_batch_propia(ruta,epochs,batch_sizes,modelos,target_size, directorio_historico, nombre_historico):
    '''
    Función que devuelve una tabla comparativa para distintas arquitecturas de modelo y distintos batch size introducidos como parámetros. 
    ----------------------------------------------------
    Parámetros:
    - ruta: str. Ruta base donde se encuentran las imágenes organizadas en subcarpetas (train, val, test)
    - epochs: int. Número de épocas a entrenar 
    - batch_sizes: lista con los distintos valores de batch size para probar en cada entrenamiento
    - modelos: lista de nombres de cada uno de los modelos que se van a comparar obtenidos partir de la función realizada previamente 
    - target_size: tupla de números enteros que representa el alto y ancho al que se van a redimensionar todas las imágenes.
    "establecer_arquitectura(modelo)"
    - directorio_historico: str. Ruta general donde se va a crear la carpeta del historico
    - nombre_historico: str. Nombre de la carpeta creada para guardar los historicos de la CNN propia
    --------------------------------------------------
    Return:
    - compara_arqu_batch_def: dataframe que contiene como índice las columnas referidas al modelo de arquitectura y al valor de batch size. El dataframe 
    obtenido se observa como una tabla comparativa de diversas métricas para cada arquitectura y cada batch size.
    '''
    
    #se inicializa un dataframe vacío donde, posteriormente se van a añadir todos los componentes necesarios para comparar los distintos 
    #modelos de arquitectura para distintos batch size (comparando las métricas)
    compara_arqu_batch=pd.DataFrame()
    

    #bucle en el que se recorren cada uno de los modelos y los tamaños de batch_size 
    for modelo in modelos:
        print(f"Comparando modelo {modelo}...")
        for batch_size in batch_sizes:
            print(f"Entrenando modelo {modelo} y batch_size {batch_size}")
    
            #se emplea la función preparar_modelo para configurar los generadores de datos para entrenar, validar y probar 
            #un modelo de aprendizaje automático con imágenes
            train_generator, validation_generator, test_generator = preparar_modelo(ruta, batch_size,target_size)
            
            #se emplea la función establecer_arquitectura para determinar el modelo con el que se trabaja cada vez
            model = establecer_arquitectura_propia(modelo)
            
            #se compila el modelo y se calculan las métricas con las que se quiere trabajar
            #en este caso, en la función de pérdida "loss", se emplea la entropía cruzada binaria "binary_crossentropy" ya que se trata de 
            #un problema de clasificación binaria
            model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy","Recall","AUC"]) #cambias loss
    
            #ENTRENA
            # con callbacks se detiene el entrenamiento si la pérdida en el conjunto de validación no mejora después de 10 épocas (patience)
            history=model.fit(train_generator, epochs=epochs, validation_data=validation_generator, callbacks=EarlyStopping(monitor='val_auc', patience=10,restore_best_weights=True))
            historico = pd.DataFrame(history.history)
            print(historico) #hacer grafica val y train para auc o loss

            #se guarda el historico en un csv para guardar los valores de entrenamiento y validación (accuracy, recall, val_auc, val_los...)
            nombre_archivo = f'hist_propia_{modelo}_{batch_size}.csv' #se define el nombre que van a tener cada uno de los dataframes donde esta el historico
            ruta_historico = os.path.join(directorio_historico,nombre_historico) #se guarda dentro de una nueva carpeta 
            # Crea la carpeta 'historico_2_64' si no existe
            os.makedirs(ruta_historico, exist_ok=True)
            ruta_archivo = os.path.join(ruta_historico, nombre_archivo)
            historico.to_csv(ruta_archivo, index=False)
        
            #se calculan las métricas
            y_test=test_generator.labels
            y_pred=model.predict(test_generator)
            calculo_metricas=metricas(y_test, y_pred) #se llama a la función creada previamente para calcular las métricas de cada modelo
            #se calcula loss a partir de la evaluación del modelo
            loss=model.evaluate(test_generator, verbose=0)[0]
            
            #esto es en caso de querer meter todos estos parametros dentro de metricas (cambiando tambien la linea de arriba, en lugar de metricas loss, accuracy...)
            #metricas = f"Loss: {loss}, Accuracy: {accuracy}, Recall: {recall}, AUC: {AUC}, Precision: {precision}"
    
            #cambiar .append por .concat
            #se añaden todos los componentes necesarios para comparar los distintos modelos de arquitectura para distintos batch size 
            #(comparando las métricas)
            compara_arqu_batch=compara_arqu_batch.append({"Red": modelo, "BatchSize": batch_size, "Loss": loss, "Accuracy": calculo_metricas[0], "Precision": calculo_metricas[1], "Recall": calculo_metricas[2], "F1":calculo_metricas[3], "Specificity":calculo_metricas[4], "fpr":calculo_metricas[5], "fnr":calculo_metricas[6], "AUC": calculo_metricas[7]}, ignore_index=True)
    
    #se fijan las columnas Red y BatchSize como índices. 
    compara_arqu_batch.set_index(["Red","BatchSize"], inplace=True) #inplace=True se pone para modificar el dataframe original ya que sino, no se modifica
    compara_arqu_batch_def = compara_arqu_batch.round(2) #se redondean los decimales a 2
    return compara_arqu_batch_def


def arq_batch_AlexNet(ruta,epochs,batch_sizes,modelos,target_size, directorio_historico, nombre_historico):
    '''
    Función que devuelve una tabla comparativa para distintas arquitecturas de modelo y distintos batch size introducidos como parámetros. 
    ----------------------------------------------------
    Parámetros:
    - ruta: str. Ruta base donde se encuentran las imágenes organizadas en subcarpetas (train, val, test)
    - epochs: int. Número de épocas a entrenar 
    - batch_sizes: lista con los distintos valores de batch size para probar en cada entrenamiento
    - modelos: lista de nombres de cada uno de los modelos que se van a comparar obtenidos partir de la función realizada previamente 
    - target_size: tupla de números enteros que representa el alto y ancho al que se van a redimensionar todas las imágenes.
    "establecer_arquitectura(modelo)"
    - directorio_historico: str. Ruta general donde se va a crear la carpeta del historico
    - nombre_historico: str. Nombre de la carpeta creada para guardar los historicos de la CNN AlexNet
    --------------------------------------------------
    Return:
    - compara_arqu_batch_def: dataframe que contiene como índice las columnas referidas al modelo de arquitectura y al valor de batch size. El dataframe 
    obtenido se observa como una tabla comparativa de diversas métricas para cada arquitectura y cada batch size.
    '''
    
    #se inicializa un dataframe vacío donde, posteriormente se van a añadir todos los componentes necesarios para comparar los distintos 
    #modelos de arquitectura para distintos batch size (comparando las métricas)
    compara_arqu_batch=pd.DataFrame()
    

    #bucle en el que se recorren cada uno de los modelos y los tamaños de batch_size 
    for modelo in modelos:
        print(f"Comparando modelo {modelo}...")
        for batch_size in batch_sizes:
            print(f"Entrenando modelo {modelo} y batch_size {batch_size}")
    
            #se emplea la función preparar_modelo para configurar los generadores de datos para entrenar, validar y probar 
            #un modelo de aprendizaje automático con imágenes
            train_generator, validation_generator, test_generator = preparar_modelo(ruta, batch_size,target_size)
            
            #se emplea la función establecer_arquitectura para determinar el modelo con el que se trabaja cada vez
            model = establecer_arquitectura_AlexaNet(modelo)
            
            #se compila el modelo y se calculan las métricas con las que se quiere trabajar
            #en este caso, en la función de pérdida "loss", se emplea la entropía cruzada binaria "binary_crossentropy" ya que se trata de 
            #un problema de clasificación binaria
            model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy","Recall","AUC"]) #cambias loss
    
            #ENTRENA
            # con callbacks se detiene el entrenamiento si la pérdida en el conjunto de validación no mejora después de 10 épocas (patience)
            history= model.fit(train_generator, epochs=epochs, validation_data=validation_generator, callbacks=EarlyStopping(monitor='val_auc', patience=10,restore_best_weights=True))
            historico = pd.DataFrame(history.history)
            print(historico) #hacer grafica val y train para auc o loss

            #se guarda el historico en un csv para guardar los valores de entrenamiento y validación (accuracy, recall, val_auc, val_los...)
            nombre_archivo = f'hist_anet_{modelo}_{batch_size}.csv' #se define el nombre que van a tener cada uno de los dataframes donde esta el historico
            ruta_historico = os.path.join(directorio_historico, nombre_historico) #se guarda dentro de una nueva carpeta denominada 'historico_2_64'
            # Crea la carpeta 'historico_2_64' si no existe
            os.makedirs(ruta_historico, exist_ok=True)
            ruta_archivo = os.path.join(ruta_historico, nombre_archivo)
            historico.to_csv(ruta_archivo, index=False)
        
            #se calculan las métricas
            y_test=test_generator.labels
            y_pred=model.predict(test_generator)
            calculo_metricas=metricas(y_test, y_pred) #se llama a la función creada previamente para calcular las métricas de cada modelo
            #se calcula loss a partir de la evaluación del modelo
            loss=model.evaluate(test_generator, verbose=0)[0]
            
            #esto es en caso de querer meter todos estos parametros dentro de metricas (cambiando tambien la linea de arriba, en lugar de metricas loss, accuracy...)
            #metricas = f"Loss: {loss}, Accuracy: {accuracy}, Recall: {recall}, AUC: {AUC}, Precision: {precision}"
    
            #cambiar .append por .concat
            #se añaden todos los componentes necesarios para comparar los distintos modelos de arquitectura para distintos batch size 
            #(comparando las métricas)
            compara_arqu_batch=compara_arqu_batch.append({"Red": modelo, "BatchSize": batch_size, "Loss": loss, "Accuracy": calculo_metricas[0], "Precision": calculo_metricas[1], "Recall": calculo_metricas[2], "F1":calculo_metricas[3], "Specificity":calculo_metricas[4], "fpr":calculo_metricas[5], "fnr":calculo_metricas[6], "AUC": calculo_metricas[7]}, ignore_index=True)
    #se fijan las columnas Red y BatchSize como índices. 
    compara_arqu_batch.set_index(["Red","BatchSize"], inplace=True) #inplace=True se pone para modificar el dataframe original ya que sino, no se modifica
    compara_arqu_batch_def = compara_arqu_batch.round(2) #se redondean los decimales a 2
    return compara_arqu_batch_def



def neuronas(num_neuronas, epochs, ruta, batch_size, target_size, directorio_historico, nombre_historico):

    '''
    Función que devuelve una tabla comparativa para distintas valores de neuronas introducidos como parámetros a partir del modelo y el batch size
    seleccionado previamente.
    ------------------------------------------------------------------------
    Parámetros;
    - num_neuronas:
    - epochs:
    - ruta: str. Ruta base donde se encuentran las imágenes organizadas en subcarpetas (train, val, test)
    - batch_size: int. Tamaño del lote que se utiliza en una única iteración del algoritmo de aprendizaje. Se emplea dentro de la función
    - target_size: tupla de números enteros que representa el alto y ancho al que se van a redimensionar todas las imágenes.
    "preparar_modelo" para determinar el tamaño del lote para cada uno de los generadores (train, val y test)
    - directorio_historico: str. Ruta general donde se va a crear la carpeta del historico
    - nombre_historico: str. Nombre de la carpeta creada para guardar los historicos de la CNN AlexNet, modelo Simple3 y batch size=64
    ----------------------------------------------------------------
    Return:
    - compara_neuronas_def: dataframe que contiene como índice las columnas referidas al número de neuronas. El dataframe 
    obtenido se observa como una tabla comparativa de diversas métricas para cada número de neuronas.
    '''
    
    #se inicializa un dataframe vacío donde, posteriormente se van a añadir todos los componentes necesarios para comparar y determinar cual es el mejor
    #valor de neuronas en la capa oculta
    compara_neuronas=pd.DataFrame()
    
    input_shape=(340,340,3)

    #se emplea la función preparar_modelo para configurar los generadores de datos para entrenar, validar y probar 
    #un modelo de aprendizaje automático con imágenes
    train_generator, validation_generator, test_generator = preparar_modelo(ruta, batch_size, target_size)
    
    
    for neurona in num_neuronas:
        print(f"Modelo con {neurona} neuronas en su capa oculta...")
        #se emplea el modelo Simple2 que es el que se ha determinado previamente como "mejor"
        model = keras.Sequential(
            [
                keras.Input(shape=input_shape),
                layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), padding='valid', activation='relu'),
                layers.MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'),
                layers.BatchNormalization(),
                
                layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), padding='valid', activation='relu'),
                layers.MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'),
                layers.BatchNormalization(),
                
                layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'),
                layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'),
                layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'),
                layers.MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'),
                layers.BatchNormalization(),
                
                layers.Flatten(), #convierte imágenes en vectores
                layers.Dense(neurona, activation="relu"), 
                layers.Dropout(0.2),
                layers.Dense(neurona, activation="relu"), 
                layers.Dropout(0.2),
                layers.Dense(1, activation="sigmoid"), #produce una probabilidad entre 0 y 1 para la clasificación binaria
            ]
        )

        
        #se compila el modelo y se calculan las métricas con las que se quiere trabajar
        #en este caso, en la función de pérdida "loss", se emplea la entropía cruzada binaria "binary_crossentropy" ya que se trata de 
        #un problema de clasificación binaria
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy","Recall","AUC"]) #cambias loss
    
        #ENTRENA
        # con callbacks se detiene el entrenamiento si la pérdida en el conjunto de validación no mejora después de 10 épocas (patience)
        #se emplea un batch size de 32 que es el que ha dado mejores resultados antes
        history=model.fit(train_generator, epochs=epochs, validation_data=validation_generator, callbacks=EarlyStopping(monitor='val_auc', patience=10,restore_best_weights=True))
        historico = pd.DataFrame(history.history)
        print(historico) #hacer grafica val y train para auc o loss
        #se guarda el historico en un csv para guardar los valores de entrenamiento y validación (accuracy, recall, val_auc, val_los...)
        nombre_archivo = f'historico_{neurona}.csv' #se define el nombre que van a tener cada uno de los dataframes donde esta el historico
        ruta_historico = os.path.join(directorio_historico, nombre_historico) #se guarda dentro de una nueva carpeta denominada 'historico_2_64'
        # Crea la carpeta 'historico_3_64' si no existe
        os.makedirs(ruta_historico, exist_ok=True)
        ruta_archivo = os.path.join(ruta_historico, nombre_archivo)
        historico.to_csv(ruta_archivo, index=False)
        
        #se calculan las métricas
        y_test=test_generator.labels
        y_pred=model.predict(test_generator)
        calculo_metricas=metricas(y_test, y_pred) #se llama a la función creada previamente para calcular las métricas de cada modelo
        #se calcula loss a partir de la evaluación del modelo
        loss=model.evaluate(test_generator, verbose=0)[0]
        #esto es en caso de querer meter todos estos parametros dentro de metricas (cambiando tambien la linea de arriba, en lugar de metricas loss, accuracy...)
        #metricas = f"Loss: {loss}, Accuracy: {accuracy}, Recall: {recall}, AUC: {AUC}, Precision: {precision}"
        #cambiar .append por .concat
        #se añaden todos los componentes necesarios para comparar los distintos modelos de arquitectura para distintos batch size 
        #(comparando las métricas)
        compara_neuronas=compara_neuronas.append({"Número de neuronas": neurona, "Loss": loss, "Accuracy": calculo_metricas[0], "Precision": calculo_metricas[1], "Recall": calculo_metricas[2], "F1":calculo_metricas[3], "Specificity":calculo_metricas[4], "fpr":calculo_metricas[5], "fnr":calculo_metricas[6], "AUC": calculo_metricas[7]}, ignore_index=True)
    
    #se fija la columna "Número de neuronas" como índice. 
    compara_neuronas.set_index("Número de neuronas", inplace=True) #inplace=True se pone para modificar el dataframe original ya que sino, no se modifica
    compara_neuronas_def = compara_neuronas.round(2) #se redondean los decimales a 2
    #compara_neuronas_def['Número de neuronas'] = compara_neuronas_def['Número de neuronas'].astype(int) #para convertr la columna Numero neuronas a entero y no aparezca como decimales
    return compara_neuronas_def
    



def grafica(directorio_historico, metrica_entrenamiento, metrica_validacion):

    '''
    Función empleada para crear una gráfica a partir de uno de los csv creados previamente para observar la evolución de dos métricas 
    (loss o auc generalment) durante el entrenamiento y la validación del modelo, lo cual es útil para evaluar el rendimiento 
    del modelo a lo largo de las épocas.
    -----------------------------------------------------------------
    Parámetros:
    - directorio_historico: directorio donde se encuentra el csv del que se desea obtener las gráficas.
    - metrica_entrenamiento: metrica monitoreada durante el entrenamiento que se desea visualizar (loss o auc)
    - metrica_validacion: metrica monitoreada durante la validación que se desea visualizar (loss o auc)
    --------------------------------------------------------------
    Return:
    - nada
    '''
    # Se cargan los datos desde el archivo CSV
    df_mejor = pd.read_csv(directorio_historico) 
    
    
    # 'columna1' y 'columna2' son los nombres de las columnas que se van a graficar
    columna1 = df_mejor[metrica_entrenamiento]
    columna2 = df_mejor[metrica_validacion]
    
    # Se crea la gráfica para las dos columnas
    plt.plot(columna1, label='Entrenamiento')
    plt.plot(columna2, label='Validación')
    
    # Se añaden etiquetas del eje x, el eje y y el título de la gráfica
    plt.xlabel('Epochs')
    plt.ylabel('loss/auc') 
    plt.title('Gráfico evolución entrenamiento y validación')
    
    # Se añade la leyenda
    plt.legend()
    
    # Se muestra la gráfica
    plt.show()



def matriz_conf_inicial(ruta, batch_size, target_size, epochs):

    '''
    Función que crea una matriz de confusión a partir del modelo inicial es decir, con la CNN propia, modelo Simple1 (sin capas ocultas).
    ----------------------------------------------------------------
    Parámetros:
    - ruta: str. Ruta base donde se encuentran las imágenes organizadas en subcarpetas (train, val, test)
    - batchsize: int. Tamaño del lote que se utiliza en una única iteración del algoritmo de aprendizaje
    - target_size: tupla de números enteros que representa el alto y ancho al que se van a redimensionar todas las imágenes. En este caso,como se emplea
    el modelo Simple1 donde el input_shape=(150,150,3), target_sizetendrá que ser igual a (150,150)
    - epochs: int. Número de épocas a entrenar 
    -----------------------------------------------------------
    Return:
    - nada
    '''
    
    train_generator, validation_generator, test_generator = preparar_modelo(ruta, batch_size, target_size)

    input_shape=(150,150,3)

    # se emplea el modelo más simple con CNN propia
    model = keras.Sequential( 
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"), 
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5), 
            layers.Dense(1, activation="sigmoid"), #una unica neurona, sigmoide
        ]
    )
    
    epochs = epochs
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy","Recall","AUC"]) #cambias loss
    
    # con callbacks se detiene el entrenamiento si la pérdida en el conjunto de validación no mejora después de 5 épocas (patience)
    model.fit(train_generator, epochs=epochs, validation_data=validation_generator, callbacks=EarlyStopping(monitor='val_auc', patience=10,restore_best_weights=True)) 

    # se calcula y_test e y_pred para obtener la matriz de confusion
    y_test=test_generator.labels
    y_pred=model.predict(test_generator)
    y_pred_bin=np.where(y_pred>=0.5,1,0) #para convertirlo en un problema binario

    
    #PERCEPTRON SKLEARN
    labels=np.unique(y_test)
    
    matriz_conf = metrics.confusion_matrix(y_test, y_pred_bin,labels=labels)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = matriz_conf, display_labels = ["PNEUMONIA" , "NORMAL"])
    fig, ax = plt.subplots(figsize=(5,5))
    cm_display.plot(ax=ax)
    plt.title("Matriz inicial PNEUMONIA-NORMAL")
    plt.show()




def matriz_conf_final(ruta, batch_size, target_size, epochs):

    '''
    Función que crea una matriz de confusión a partir del modelo final es decir, con la CNN alexNet, modelo Simple3, batch size = 64 y 100 y 16 
    neuronas en las capas ocultas reespectivamente.
    ----------------------------------------------------------------
    Parámetros:
    - ruta: str. Ruta base donde se encuentran las imágenes organizadas en subcarpetas (train, val, test)
    - batchsize: int. Tamaño del lote que se utiliza en una única iteración del algoritmo de aprendizaje. En este caso el batch size será igual a 64
    - target_size: tupla de números enteros que representa el alto y ancho al que se van a redimensionar todas las imágenes. En este caso,como se emplea
    el modelo Simple1 donde el input_shape=(340,340,3), target_sizetendrá que ser igual a (340,340)
    - epochs: int. Número de épocas a entrenar 
    -----------------------------------------------------------
    Return:
    - nada
    '''
    
    train_generator, validation_generator, test_generator = preparar_modelo(ruta, batch_size, target_size)

    input_shape=(340,340,3)

    # se emplea el modelo más simple con CNN propia

    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), padding='valid', activation='relu'),
            layers.MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'),
            layers.BatchNormalization(),
                
            layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), padding='valid', activation='relu'),
            layers.MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'),
            layers.BatchNormalization(),
                
            layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'),
            layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'),
            layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'),
            layers.MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'),
            layers.BatchNormalization(),
                
            layers.Flatten(), #convierte imágenes en vectores
            layers.Dense(100, activation="relu"), #100 neuronas en la primera capa
            layers.Dropout(0.2),
            layers.Dense(16, activation="relu"), #16 neuronas en la segunda capa
            layers.Dropout(0.2),
            layers.Dense(1, activation="sigmoid"), #produce una probabilidad entre 0 y 1 para la clasificación binaria
        ]
    )

    
    epochs = epochs

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy","Recall","AUC"]) #cambias loss
    
    # con callbacks se detiene el entrenamiento si la pérdida en el conjunto de validación no mejora después de 5 épocas (patience)
    model.fit(train_generator, epochs=epochs, validation_data=validation_generator, callbacks=EarlyStopping(monitor='val_auc', patience=10,restore_best_weights=True)) 

    # se calcula y_test e y_pred para obtener la matriz de confusion
    y_test=test_generator.labels
    y_pred=model.predict(test_generator)
    y_pred_bin=np.where(y_pred>=0.5,1,0) #para convertirlo en un problema binario

    
    #PERCEPTRON SKLEARN
    labels=np.unique(y_test)
    
    matriz_conf = metrics.confusion_matrix(y_test, y_pred_bin,labels=labels)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = matriz_conf, display_labels = ["PNEUMONIA" , "NORMAL"])
    fig, ax = plt.subplots(figsize=(5,5))
    cm_display.plot(ax=ax)
    plt.title("Matriz final PNEUMONIA-NORMAL")
    plt.show()



    