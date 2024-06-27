#AUTOR: Nuria Martínez Queralt
#CURSO: Grado en Ingeniería de la Salud
#ORGANIZACIÓN: Universidad de Burgos - TFG

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
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn import metrics

def buscar_imagen(directorio_padre, nombre_imagen):
    '''
    Función empleada para encontrar una imagen concreta (a partir de su nombre) dentro de cualquiera de las subcarpetas del directorio_padre.
    ---------------------------------------------------------
    Parámetros:
    - directorio_padre: ruta donde se encuentra la carpeta principal con cada una de las subcarpetas con las imágenes de radiografías de tórax
    - nombre_imagen: nombre de la imágen a la que se desea acceder 
    ----------------------------------------------------------
    Return:
    - ruta_imagen: ruta completa de la imágen a la que se desea acceder 
    '''
    # Subcarpetas principales en las que buscar
    subcarpetas_principales = ['train', 'test', 'val']
    # Subcarpetas adicionales en las que buscar dentro de cada subcarpeta principal (donde se encuentran las imágenes)
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
    Función empleada para redistribuir las imágenes ubicadas en distintas subcarpetas dentro de la carpeta data en una carpeta nueva con las
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
    En primer lugar, se crea un csv con dos columnas ''nombres_ficheros'' y ''clases'' compuesto por todas las imágenes existentes en el directorio_padre.
    En la columna ''nombres_ficheros'', debe aparecer el nombre de TODAS las imágenes que existen dentro de cada subcarpeta y en la columna ''clases'',
    debe aparecer 0 o 1 en función si se trata de una imagen de la carpeta NORMAL o PNEUMONIA respectivamente.
    '''

    directorio_padre = os.path.join(directorio_principal, 'data') #se accede a la ruta de la carpeta data
    
    # Listas vacias para almacenar los nombres de las imágenes y las clases (0 o 1 en función de si es normal o neumonía respectivamente)
    nombres_ficheros = []
    clases = []
    
    # Se recorren las carpetas de train, test y val
    for subcarpeta in ['train', 'test', 'val']:
        ruta_subcarpeta = os.path.join(directorio_padre, subcarpeta) #ruta a cada una de las subcarpetas
        for clase in ['NORMAL', 'PNEUMONIA']:
            ruta_clase = os.path.join(ruta_subcarpeta, clase) #ruta a cada una de las clases dentro de las subcarpetas
            for nombre_fichero in os.listdir(ruta_clase):
                nombres_ficheros.append(nombre_fichero)
                clases.append(0 if clase == 'NORMAL' else 1)
    
    # Se crea el DataFrame con los datos
    df_todas = pd.DataFrame({'nombre_fichero': nombres_ficheros,'clase': clases})
    
    # Se guarda el DataFrame en un archivo CSV dentro de la carpeta creada ''Datos''
    ruta_datos=os.path.join('.', 'Datos') #se crea la ruta a la nueva carpeta ''Datos'' en el directorio actual
    os.makedirs(ruta_datos, exist_ok=True) # se crea la nueva carpeta si esta no existe
    ruta_csv = os.path.join(ruta_datos, 'dataset_info.csv') #el nuevo dataframe se guarda dentro del directorio padre
    df_todas.to_csv(ruta_csv, index=False, encoding='utf-8')

    '''
    A partir del csv anterior y, con ayuda de la función ''train_test_split'' de skitlearn, se divide el csv anterior en dos 
    subgrupos de train y test en proporción 80, 20 para poder usar el 80% de las imágenes para train y el 20% para test.
    También se emplea el parámetro ''stratify'' para que exista una proporción de clases en cada uno de los grupos, es decir, en ''NORMAL" y "PNEUMONIA".
    '''
    
    # random_state=42 se emplea para que cada vez que se ejecute el código, se obtenga la misma división de datos. El valor 42 es un valor que se usa
    # comunmente en este caso pero se puede emplear cualquie otro valor entero.
    train_df, test_df = train_test_split(df_todas, test_size=0.2, stratify=df_todas['clase'], random_state=42)
    
    # Se guardan los nuevos conjuntos de datos en archivos CSV dentro de la carpeta ''Datos''
    ruta_train_csv = os.path.join(ruta_datos, 'train_dataset_info.csv') #el nuevo dataframe se guarda dentro de la carpeta Datos
    ruta_test_csv = os.path.join(ruta_datos, 'test_dataset_info.csv') #el nuevo dataframe se guarda dentro de la carpeta Datos
    train_df.to_csv(ruta_train_csv, index=False, encoding='utf-8')
    test_df.to_csv(ruta_test_csv, index=False, encoding='utf-8')

    '''
    A continuación, se coge el conjunto de datos obtenido previamente de train, es decir, el csv "train_df" y se repite el mismo
    proceso pero, esta vez dividiendo este conjunto de datos para train y val en un 80% y 20% respectivamente.
    De tal forma que, finalemnte se obtenga el conjunto de test que represeneta el 20% del total (obtenido previamente), el conjunto de train
    que representa el 80% del 80% del total ya que, inicialmente nos hemos quedado con el 80% pero luego, de este 80%, el 20% va destinado al conjunto
    de validación. Por lo que, finalmete quedarían distribuidos de la siguiente manera:
    - test: 20% del total
    - train: 64% del total
    - val: 16% del total
    '''

    # Se emplea train_test_split para dividir el conjunto de datos de entrenamiento en train (80%) y val (20%)
    train_def_df, val_df = train_test_split(train_df, test_size=0.2, stratify=train_df['clase'], random_state=42)
    
    # Se guardan los nuevos conjuntos de datos en archivos CSV dentro de la carpeta ''Datos''
    ruta_train_final_csv = os.path.join(ruta_datos, 'train_final_dataset_info.csv') #el nuevo dataframe se guarda dentro de la carpeta ''Datos''
    ruta_val_csv = os.path.join(ruta_datos, 'val_dataset_info.csv') #el nuevo dataframe se guarda dentro de la carpeta ''Datos''
    train_def_df.to_csv(ruta_train_final_csv, index=False, encoding='utf-8')
    val_df.to_csv(ruta_val_csv, index=False, encoding='utf-8')

    '''
    Finalmente, se crea una nueva carpeta denominada data_nuevo dentro del directorio principal. Dentro de esta carpeta se crean 3 subcarpetas 
    ("train", "test" y "val") que corresponderian con los dataframes obtenidos hasta hora: train_def_df, val_df y test_df y, dentro de estas 3 
    subcarpetas, se crean 2 carpetas "NORMAL" y "PNEUMONIA" que corresponden con con las clases determinadas en cada dataframe, 0 en caso de 
    "NORMAL" y 1 para "PNEUMONIA". Dentro de estas dos carpetas para ("train", "test" y "val") se encontrarán las imagenes correspondientes 
    para cada caso según los dataframes obtenidos.
    
    La función ''os.makedirs'', verifica si la carpeta ruta_subcarpeta ya existe. Si existe, no se hace nada y el programa continúa su ejecución 
    sin lanzar un error. Si no existe, la función os.makedirs() la crea junto con cualquier carpeta intermedia necesaria en la ruta especificada.
    '''

    # Se crea la nueva carpeta dentro del directorio principal
    ruta_principal_nueva = os.path.join(directorio_principal, 'data_nuevo') 

    # Se crean las carpetas 'train', 'test' y 'val' dentro de la nueva carpeta principal
    for subcarpeta in ['train', 'test', 'val']:
        ruta_subcarpeta = os.path.join(ruta_principal_nueva, subcarpeta)
        os.makedirs(ruta_subcarpeta, exist_ok=True) 
        
        # Se crean las subcarpetas 'NORMAL' y 'PNEUMONIA' dentro de cada subcarpeta ('train', 'test' y 'val')
        for clase in ['NORMAL', 'PNEUMONIA']:
            ruta_clase = os.path.join(ruta_subcarpeta, clase)
            os.makedirs(ruta_clase, exist_ok=True)

    # Se copian los archivos CSV a las subcarpetas correspondientes
    for df, nombre_carpeta in [(train_def_df, 'train'), (val_df, 'val'), (test_df, 'test')]:
        for index, row in df.iterrows(): #se itera sobre cada dataframe fila a fila
            clase = 'NORMAL' if row['clase'] == 0 else 'PNEUMONIA'
            nombre_archivo = row['nombre_fichero']
    
            # ruta de origen donde se busca la imagen concreta a partir de la función realizada previamente
            # esta ruta se refiere a donde esta la imagen que se desea guardar en la carpeta destino originalmente para poder copiarla
            ruta_origen=buscar_imagen(directorio_padre, nombre_archivo)
            
            # ruta donde se desa guardar (y redestribuir de la forma correcta) las imágenes
            ruta_destino = os.path.join(ruta_principal_nueva, nombre_carpeta, clase, nombre_archivo)
            
            shutil.copyfile(ruta_origen, ruta_destino) # copia las imágenes de la ruta incial a la ruta final



def preparar_modelo(ruta, batch_size,target_size):

    '''
    Función que configura los generadores de datos para entrenar, validar y probar un modelo de aprendizaje automático con imágenes.
    -----------------------------------------------------------
    Parámetros:
    - ruta: str. Ruta base donde se encuentran las imágenes organizadas en subcarpetas (train, val, test). Ruta data_nuevo.
    - batchsize: int. Tamaño del lote que se utiliza en una única iteración del algoritmo de aprendizaje
    - target_size: tupla de números enteros que representa el alto y ancho al que se van a redimensionar todas las imágenes. Este valor deberá coincidir
    con el input_shape empleado posteriormente dependiendo de cada caso.
    ----------------------------------------------------
    Return:
    - nada
    '''
    
    dir_general = ruta 
    
    dir_train = os.path.join(dir_general, 'train') # se accede a la ruta concreta de la carpeta ''train'' dentro de data_nuevo
    dir_validation = os.path.join(dir_general, 'val') # se accede a la ruta concreta de la carpeta ''val'' dentro de data_nuevo
    dir_test = os.path.join(dir_general, 'test') # se accede a la ruta concreta de la carpeta ''test'' dentro de data_nuevo
    
    # Preprocesamiento de imágenes
    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)
    validation_datagen=ImageDataGenerator(rescale=1./255)
    
    #Iterador que recorre el directorio de imágenes del conjunto de entrenamiento
    train_generator = train_datagen.flow_from_directory(
        dir_train,
        target_size=target_size, 
        batch_size=batch_size, 
        color_mode='rgb',
        class_mode='binary', #clase binaria
        classes=['NORMAL','PNEUMONIA'], #se indican las clases
        shuffle=True) # el conjunto de datos se barajará aleatoriamente para evitar sobreajuste 

    #Iterador que recorre el directorio de imágenes del conjunto de validación
    validation_generator = validation_datagen.flow_from_directory(
        dir_validation,
        target_size=target_size, 
        batch_size=batch_size, 
        color_mode='rgb',
        class_mode='binary', #clase binaria
        classes=['NORMAL','PNEUMONIA'], #se indican las clases
        shuffle=False) 

    #Iterador que recorre el directorio de imágenes del conjunto de prueba
    test_generator = test_datagen.flow_from_directory(
        dir_test,
        target_size=target_size, 
        batch_size=batch_size,
        color_mode='rgb',
        class_mode='binary', #clase binaria
        classes=['NORMAL','PNEUMONIA'], #se indican las clases
        shuffle=False) 
    
    return train_generator, validation_generator, test_generator

def metricas(y_test, y_pred):
    '''
    Función que calcula distintas métricas para la evaluación del modelo.
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
    
    #Se obtienen los verdaderos negativos, falsos positivos, falsos negativos y verdaderos positivos a partir de la matriz de confusión 
    #con .ravel() se convierte la matriz en un array unidimensional
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_bin).ravel() 

    #Se calculan cada una de las métricas empleando su correspondiente fórmula
    accuracy = (tp + tn)/(tn + fp + fn + tp)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * ((precision*recall)/(precision+recall))
    specificity = tn / (tn + fp)
    fpr = fp / (fp + tn) #tasa de falsos positivos
    fnr = fn / (fn + tp) #tasa de falsos negativos
    auc = roc_auc_score(y_test, y_pred)

    
    return [accuracy, precision, recall, f1, specificity, fpr, fnr, auc] #se devuleve como una lista para poder trabajar correctamente con las métricas

def establecer_arquitectura_propia(tipo):

    '''
    Función que establece tres tipos de modelos de red neuronal convolucional (CNN) a partir de una CNN propia.
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
    Este modelo es muy simple y, se sabe de antemano que los resultados no van a ser buenos pero sirve de punto de partida.
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
                layers.Dense(100, activation="relu"), #100 neuronas en la capa oculta
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
    Función que establece distintos tipos de modelos de red neuronal convolucional (CNN) a partir de la CNN de alexNet.
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
    El modelo Simple1, se corresponde con un modelo que posee cinco capas convolucionales (con las que se obtienen características importantes
    de las imágenes). La primera, segunda y quinta están seguidas de una capa de MaxPooling2D para reducir la dimensionalidad. 
    Después del Flatten se encuentra una capa densa.
    La función de activación sigmoide en la capa de salida produce una probabilidad entre 0 y 1 para la clasificación binaria.
    Este modelo es muy simple y los resultados no van a ser buenos pero sirve de punto de partida.
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
                layers.Dropout(0.2), 
                layers.Dense(1, activation="sigmoid"), #produce una probabilidad entre 0 y 1 para la clasificación binaria
            ]
        )
        
        '''
    El modelo Simple2, se corresponde con un modelo que posee cinco capas convolucionales (con las que se obtienen características importantes
    de las imágenes). La primera, segunda y quinta están seguidas de una capa de MaxPooling2D para reducir la dimensionalidad.
    Después del Flatten se encuentra una capa oculta de 100 unidades y una capa densa.
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
                layers.Dense(100, activation="relu"), #100 neuronas en la capa oculta
                layers.Dropout(0.2),
                layers.Dense(1, activation="sigmoid"), #produce una probabilidad entre 0 y 1 para la clasificación binaria
            ]
        )
        '''
    El modelo Simple3, se corresponde con un modelo que posee cinco capas convolucionales (con las que se obtienen características importantes
    de las imágenes). La primera, segunda y quinta están seguidas de una capa de MaxPooling2D para reducir la dimensionalidad. 
    Después del Flatten se encuentra una capa se encuentra una capa oculta de 100 neuronas, una segunda capa oculta de 16 neuronas y una capa densa.
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




def arq_batch_propia(ruta,epochs,batch_sizes,modelos,target_size, nombre_carpeta_hist, nombre_carpeta_resultados, nombre_carpeta_modelos):
    '''
    Función que entrena distintas arquitecturas de modelo y distintos batch size introducidos como parámetros a partir de la CNN propia 
    establecida previamente y devuelve una tabla comparativa con los distintos valores de métricas para cada caso.
    También guarda en distintas carpetas los historicos con las métricas obtenidas tras cada entrenaminto para
    cada modelo, los distintos modelos obtenidos según la arquitectura y el batch size y el dataframe final en formato csv.
    ----------------------------------------------------
    Parámetros:
    - ruta: str. Ruta base donde se encuentran las imágenes organizadas en subcarpetas (train, val, test). Ruta data_nuevo
    - epochs: int. Número de épocas a entrenar 
    - batch_sizes: lista con los distintos valores de batch size para probar en cada entrenamiento
    - modelos: lista de nombres de cada uno de los modelos que se van a comparar obtenidos partir de la función realizada previamente 
    - target_size: tupla de números enteros que representa el alto y ancho al que se van a redimensionar todas las imágenes. En este caso deberá ser 
    (150,150) para coincidir con el input_shape del modelo de CNN propia.
    - nombre_carpeta_hist: str. Nombre de la carpeta creada para guardar los csv de los históricos
    - nombre_carpeta_resultados: str. Nombre de la carpeta creada para guardar el dataframe final en formato csv.
    - nombre_carpeta_modelos: str. Nombre de la carpeta creada para guardar los distintos modelos
    --------------------------------------------------
    Return:
    - compara_arqu_batch_def: dataframe que contiene como índice las columnas referidas al modelo de arquitectura y al valor de batch size. El dataframe 
    obtenido se observa como una tabla comparativa de diversas métricas para cada arquitectura y cada batch size.
    '''
    
    #se inicializa un dataframe vacío donde, posteriormente se van a añadir todos los componentes necesarios para comparar los distintos 
    #modelos de arquitectura para distintos batch size (comparando las métricas)
    compara_arqu_batch=pd.DataFrame()
    

    #bucle en el que se recorren cada uno de los modelos y los tamaños de batch_size introducidos como parámetro
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
            model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy","Recall","AUC"]) 
    
            #ENTRENAMIENTO DEL MODELO
            # con callbacks se detiene el entrenamiento si la pérdida en el conjunto de validación no mejora después de 10 épocas (patience)
            history=model.fit(train_generator, epochs=epochs, validation_data=validation_generator, callbacks=EarlyStopping(monitor='val_auc', patience=10,restore_best_weights=True))
            historico = pd.DataFrame(history.history)
            print(historico) 

            #SE GUARDAN LOS DISTINTOS HISTÓRICOS EN UNA CARPETA
            #se guarda el historico en un csv para guardar los valores de entrenamiento y validación (accuracy, recall, val_auc, val_los...)
            ruta_historicos = os.path.join('.', nombre_carpeta_hist) #se crea la ruta a la nueva carpeta de Historicos en el directorio actual
            os.makedirs(ruta_historicos, exist_ok=True) # se crea la nueva carpeta si esta no existe
            #se crea la ruta a la subcarpeta dentro de la nueva carpeta de modelos
            subcarpeta_historicos = os.path.join(ruta_historicos,'historico_propia_arqu_batchsize') 
            os.makedirs(subcarpeta_historicos, exist_ok=True) # se crea la nueva subcarpeta si esta no existe
            nombre_historico = f'hist_propia_{modelo}_{batch_size}.csv' #se define el nombre que van a tener cada uno de los csv donde esta el historico correspondiente
            ruta_historico = os.path.join(subcarpeta_historicos, nombre_historico) #se define la ruta donde se econtrará cada modelo
            historico.to_csv(ruta_historico, index=False, encoding='utf-8') #se crea el csv de cada historico

            #SE GUARDAN LOS DISTINTOS MODELOS EN UNA CARPETA
            ruta_modelos = os.path.join('.', nombre_carpeta_modelos) #se crea la ruta a la nueva carpeta de Modelos en el directorio actual
            os.makedirs(ruta_modelos, exist_ok=True) # se crea la nueva carpeta si esta no existe
            #se crea la ruta a la subcarpeta dentro de la nueva carpeta de modelos
            subcarpeta_modelo = os.path.join(ruta_modelos, 'modelo_propia_arqu_batchsize')
            os.makedirs(subcarpeta_modelo, exist_ok=True) # se crea la nueva subcarpeta si esta no existe
            nombre_modelo = f'modelo_propia_{modelo}_{batch_size}.h5' #se define en nombre de cada uno de los archivos que contienen los modelos
            ruta_modelo = os.path.join(subcarpeta_modelo, nombre_modelo) #se define la ruta donde se econtrará cada modelo
            model.save(ruta_modelo) #se guarda el modelo
        
            #se calculan las métricas a partir de la función creada previamente
            y_test=test_generator.labels
            y_pred=model.predict(test_generator)
            calculo_metricas=metricas(y_test, y_pred) #se llama a la función creada previamente para calcular las métricas de cada modelo
            #se calcula loss a partir de la evaluación del modelo
            loss=model.evaluate(test_generator, verbose=0)[0]
            
        
            #se añaden todos los componentes necesarios para comparar los distintos modelos de arquitectura para distintos batch size 
            #(comparando las métricas)
            compara_arqu_batch=compara_arqu_batch.append({"Red": modelo, "BatchSize": batch_size, "Loss": loss, "Accuracy": calculo_metricas[0], "Precision": calculo_metricas[1], "Recall": calculo_metricas[2], "F1":calculo_metricas[3], "Specificity":calculo_metricas[4], "fpr":calculo_metricas[5], "fnr":calculo_metricas[6], "AUC": calculo_metricas[7]}, ignore_index=True)
    
    #se fijan las columnas Red y BatchSize como índices. 
    compara_arqu_batch.set_index(["Red","BatchSize"], inplace=True) #inplace=True se pone para modificar el dataframe original ya que sino, no se modifica
    compara_arqu_batch_def = compara_arqu_batch.round(2) #se redondean los decimales a 2

    #SE GUARDA EL DATAFRAME FINAL EN UNA CARPETA EN FORMATO CSV
    ruta_resultados = os.path.join('.', nombre_carpeta_resultados) #se crea la ruta a la nueva carpeta de Resultados en el directorio actual
    os.makedirs(ruta_resultados, exist_ok=True) # se crea la nueva carpeta si esta no existe
    ruta_resultado_final = os.path.join(ruta_resultados, 'compara_propia_arqu_batch_def.csv') #se define la ruta donde se econtrará el dataframe
    compara_arqu_batch_def.to_csv(ruta_resultado_final, index=False, encoding='utf-8') #se crea el csv del dataframe


    return compara_arqu_batch_def



def arq_batch_AlexNet(ruta,epochs,batch_sizes,modelos,target_size, nombre_carpeta_hist, nombre_carpeta_resultados, nombre_carpeta_modelos):
    '''
    Función que entrena distintas arquitecturas de modelo y distintos batch size introducidos como parámetros a partir de la CNN basada en AlexNet 
    establecida previamente y devuelve una tabla comparativa con los distintos valores de métricas para cada caso. 
    También guarda en distintas carpetas los historicos con las métricas obtenidas tras cada entrenaminto para
    cada modelo, los distintos modelos obtenidos según la arquitectura y el batch size y el dataframe final en formato csv.
    ----------------------------------------------------
    Parámetros:
    - ruta: str. Ruta base donde se encuentran las imágenes organizadas en subcarpetas (train, val, test). Ruta data_nuevo
    - epochs: int. Número de épocas a entrenar 
    - batch_sizes: lista con los distintos valores de batch size para probar en cada entrenamiento
    - modelos: lista de nombres de cada uno de los modelos que se van a comparar obtenidos partir de la función realizada previamente 
    - target_size: tupla de números enteros que representa el alto y ancho al que se van a redimensionar todas las imágenes. En este caso deberá ser 
    (340,340) para coincidir con el input_shape del modelo de CNN alexNet.
    - nombre_carpeta_hist: str. Nombre de la carpeta creada para guardar los csv de los historicos
    - nombre_carpeta_resultados: str. Nombre de la carpeta creada para guardar el dataframe final en formato csv.
    - nombre_carpeta_modelos: str. Nombre de la carpeta creada para guardar los distintos modelos
    --------------------------------------------------
    Return:
    - compara_arqu_batch_def: dataframe que contiene como índice las columnas referidas al modelo de arquitectura y al valor de batch size. El dataframe 
    obtenido se observa como una tabla comparativa de diversas métricas para cada arquitectura y cada batch size para la CNN basada en alexNet.
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
            model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy","Recall","AUC"]) 
    
            #ENTRENAMIENTO
            # con callbacks se detiene el entrenamiento si la pérdida en el conjunto de validación no mejora después de 10 épocas (patience)
            history= model.fit(train_generator, epochs=epochs, validation_data=validation_generator, callbacks=EarlyStopping(monitor='val_auc', patience=10,restore_best_weights=True))
            historico = pd.DataFrame(history.history)
            print(historico) 

        
            #SE GUARDAN LOS DISTINTOS HISTÓRICOS EN UNA CARPETA
            #se guarda el historico en un csv para guardar los valores de entrenamiento y validación (accuracy, recall, val_auc, val_los...)
            ruta_historicos = os.path.join('.', nombre_carpeta_hist) #se crea la ruta a la nueva carpeta de Historicos en el directorio actual
            os.makedirs(ruta_historicos, exist_ok=True) # se crea la nueva carpeta si esta no existe
            #se crea la ruta a la subcarpeta dentro de la nueva carpeta de modelos
            subcarpeta_historicos = os.path.join(ruta_historicos,'historico_alexnet_arqu_batchsize') 
            os.makedirs(subcarpeta_historicos, exist_ok=True) # se crea la nueva subcarpeta si esta no existe
            nombre_historico = f'hist_alexNet_{modelo}_{batch_size}.csv' #se define el nombre que van a tener cada uno de los csv donde esta el historico correspondiente
            ruta_historico = os.path.join(subcarpeta_historicos, nombre_historico) #se define la ruta donde se econtrará cada modelo
            historico.to_csv(ruta_historico, index=False, encoding='utf-8') #se crea el csv de cada historico

            #SE GUARDAN LOS DISTINTOS MODELOS EN UNA CARPETA
            ruta_modelos = os.path.join('.', nombre_carpeta_modelos) #se crea la ruta a la nueva carpeta de Modelos en el directorio actual
            os.makedirs(ruta_modelos, exist_ok=True) # se crea la nueva carpeta si esta no existe
            #se crea la ruta a la subcarpeta dentro de la nueva carpeta de modelos
            subcarpeta_modelo = os.path.join(ruta_modelos, 'modelo_alexnet_arqu_batchsize')
            os.makedirs(subcarpeta_modelo, exist_ok=True) # se crea la nueva subcarpeta si esta no existe
            nombre_modelo = f'modelo_alexnet_{modelo}_{batch_size}.h5' #se define en nombre de cada uno de los archivos que contienen los modelos
            ruta_modelo = os.path.join(subcarpeta_modelo, nombre_modelo) #se define la ruta donde se econtrará cada modelo
            model.save(ruta_modelo) #se guarda el modelo

            
        
            #se calculan las métricas a partir de la la función creada previamente
            y_test=test_generator.labels
            y_pred=model.predict(test_generator)
            calculo_metricas=metricas(y_test, y_pred) #se llama a la función creada previamente para calcular las métricas de cada modelo
            #se calcula loss a partir de la evaluación del modelo
            loss=model.evaluate(test_generator, verbose=0)[0]
            
            
            #se añaden todos los componentes necesarios para comparar los distintos modelos de arquitectura para distintos batch size 
            #(comparando las métricas)
            compara_arqu_batch=compara_arqu_batch.append({"Red": modelo, "BatchSize": batch_size, "Loss": loss, "Accuracy": calculo_metricas[0], "Precision": calculo_metricas[1], "Recall": calculo_metricas[2], "F1":calculo_metricas[3], "Specificity":calculo_metricas[4], "fpr":calculo_metricas[5], "fnr":calculo_metricas[6], "AUC": calculo_metricas[7]}, ignore_index=True)
    
    #se fijan las columnas Red y BatchSize como índices. 
    compara_arqu_batch.set_index(["Red","BatchSize"], inplace=True) #inplace=True se pone para modificar el dataframe original ya que sino, no se modifica
    compara_arqu_batch_def = compara_arqu_batch.round(2) #se redondean los decimales a 2

    #SE GUARDA EL DATAFRAME FINAL EN UNA CARPETA EN FORMATO CSV
    ruta_resultados = os.path.join('.', nombre_carpeta_resultados) #se crea la ruta a la nueva carpeta de Resultados en el directorio actual
    os.makedirs(ruta_resultados, exist_ok=True) # se crea la nueva carpeta si esta no existe
    ruta_resultado_final = os.path.join(ruta_resultados, 'compara_alexNet_arqu_batch_def.csv') #se define la ruta donde se econtrará el dataframe
    compara_arqu_batch_def.to_csv(ruta_resultado_final, index=False, encoding='utf-8') #se crea el csv del dataframe


    
    return compara_arqu_batch_def



def neuronas(num_neuronas, epochs, ruta, batch_size, target_size, nombre_carpeta_hist, nombre_carpeta_resultados, nombre_carpeta_modelos):

    '''
    Función que devuelve una tabla comparativa con las diferentes métricas para distintos valores de neuronas introducidos como parámetros 
    a partir del modelo y el batch size seleccionado previamente (modelo Simple3 AlexNet batch size=64).
    También guarda en distintas carpetas los historicos con las métricas obtenidas tras cada entrenaminto para
    cada modelo, los distintos modelos obtenidos según el número de neuronas y el dataframe final en formato csv.
    ------------------------------------------------------------------------
    Parámetros:
    - num_neuronas: lista con los distintos valores de neuronas en las capas ocultas para probar en cada entrenamiento
    - epochs: int. Número de épocas a entrenar 
    - ruta: str. Ruta base donde se encuentran las imágenes organizadas en subcarpetas (train, val, test). Ruta data_nuevo
    - batch_size: int. Tamaño del lote que se utiliza en una única iteración del algoritmo de aprendizaje. Se emplea dentro de la función 
    "preparar_modelo" para determinar el tamaño del lote para cada uno de los generadores (train, val y test). En este caso será 64
    - target_size: tupla de números enteros que representa el alto y ancho al que se van a redimensionar todas las imágenes. En este caso deberá
    ser (340,340) ya que se emplea la CNN de alexNet
    - nombre_carpeta_hist: str. Nombre de la carpeta creada para guardar los csv de los historicos
    - nombre_carpeta_resultados: str. Nombre de la carpeta creada para guardar el dataframe final en formato csv.
    - nombre_carpeta_modelos: str. Nombre de la carpeta creada para guardar los distintos modelos
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
    #bucle en el que se recorren cada uno los valores de neuronas para las capas ocultas
    for neurona in num_neuronas:
        print(f"Modelo con {neurona} neuronas en su capa oculta...")
        #se emplea el modelo Simple3 y la CNN de alexNet que es con el que se han obtenido mejores resultados
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
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy","Recall","AUC"]) 
        #ENTRENAMIENTO
        # con callbacks se detiene el entrenamiento si la pérdida en el conjunto de validación no mejora después de 10 épocas (patience)
        history=model.fit(train_generator, epochs=epochs, validation_data=validation_generator, callbacks=EarlyStopping(monitor='val_auc', patience=10,restore_best_weights=True))
        historico = pd.DataFrame(history.history)
        print(historico) 
        

        #SE GUARDAN LOS DISTINTOS HISTÓRICOS EN UNA CARPETA
        #se guarda el historico en un csv para guardar los valores de entrenamiento y validación (accuracy, recall, val_auc, val_los...)
        ruta_historicos = os.path.join('.', nombre_carpeta_hist) #se crea la ruta a la nueva carpeta de Historicos en el directorio actual
        os.makedirs(ruta_historicos, exist_ok=True) # se crea la nueva carpeta si esta no existe
        #se crea la ruta a la subcarpeta dentro de la nueva carpeta de modelos
        subcarpeta_historicos = os.path.join(ruta_historicos,'historico_neuronas') 
        os.makedirs(subcarpeta_historicos, exist_ok=True) # se crea la nueva subcarpeta si esta no existe
        nombre_historico = f'historico_{neurona}.csv' #se define el nombre que van a tener cada uno de los csv donde esta el historico correspondiente
        ruta_historico = os.path.join(subcarpeta_historicos, nombre_historico) #se define la ruta donde se econtrará cada modelo
        historico.to_csv(ruta_historico, index=False, encoding='utf-8') #se crea el csv de cada historico

        #SE GUARDAN LOS DISTINTOS MODELOS EN UNA CARPETA
        ruta_modelos = os.path.join('.', nombre_carpeta_modelos) #se crea la ruta a la nueva carpeta de Modelos en el directorio actual
        os.makedirs(ruta_modelos, exist_ok=True) # se crea la nueva carpeta si esta no existe
        #se crea la ruta a la subcarpeta dentro de la nueva carpeta de modelos
        subcarpeta_modelo = os.path.join(ruta_modelos, 'modelo_neuronas')
        os.makedirs(subcarpeta_modelo, exist_ok=True) # se crea la nueva subcarpeta si esta no existe
        nombre_modelo = f'modelo_{neurona}.h5' #se define en nombre de cada uno de los archivos que contienen los modelos
        ruta_modelo = os.path.join(subcarpeta_modelo, nombre_modelo) #se define la ruta donde se econtrará cada modelo
        model.save(ruta_modelo) #se guarda el modelo

        
        
        #se calculan las métricas
        y_test=test_generator.labels
        y_pred=model.predict(test_generator)
        calculo_metricas=metricas(y_test, y_pred) #se llama a la función creada previamente para calcular las métricas de cada modelo
        #se calcula loss a partir de la evaluación del modelo
        loss=model.evaluate(test_generator, verbose=0)[0]
        
        #se añaden todos los componentes necesarios para comparar los distintos modelos de arquitectura para distintos batch size 
        #(comparando las métricas)
        compara_neuronas=compara_neuronas.append({"Número de neuronas": neurona, "Loss": loss, "Accuracy": calculo_metricas[0], "Precision": calculo_metricas[1], "Recall": calculo_metricas[2], "F1":calculo_metricas[3], "Specificity":calculo_metricas[4], "fpr":calculo_metricas[5], "fnr":calculo_metricas[6], "AUC": calculo_metricas[7]}, ignore_index=True)
    
    #se fija la columna "Número de neuronas" como índice. 
    compara_neuronas.set_index("Número de neuronas", inplace=True) #inplace=True se pone para modificar el dataframe original ya que sino, no se modifica
    compara_neuronas_def = compara_neuronas.round(2) #se redondean los decimales a 2

    #SE GUARDA EL DATAFRAME FINAL EN UNA CARPETA EN FORMATO CSV
    ruta_resultados = os.path.join('.', nombre_carpeta_resultados) #se crea la ruta a la nueva carpeta de Resultados en el directorio actual
    os.makedirs(ruta_resultados, exist_ok=True) # se crea la nueva carpeta si esta no existe
    ruta_resultado_final = os.path.join(ruta_resultados, 'compara_neuronas.csv') #se define la ruta donde se econtrará el dataframe
    compara_neuronas_def.to_csv(ruta_resultado_final, index=False, encoding='utf-8') #se crea el csv del dataframe
   
    return compara_neuronas_def



def grafica(directorio_historico, metrica_entrenamiento, metrica_validacion):

    '''
    Función empleada para la obtención de una gráfica a partir de uno de los csv (dentro de la carpeta Históricos)
    creados previamente para observar la evolución de dos métricas (loss o auc generalmente) durante el entrenamiento y la validación 
    del modelo, lo cual es útil para evaluar el rendimiento del modelo a lo largo de las épocas.
    -----------------------------------------------------------------
    Parámetros:
    - directorio_historico: directorio donde se encuentra el csv del que se desea obtener las gráficas.
    - metrica_entrenamiento: metrica monitoreada durante el entrenamiento que se desea visualizar (loss o auc)
    - metrica_validacion: metrica monitoreada durante la validación que se desea visualizar (val_loss o val_auc)
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
    
    # Se añaden etiquetas al eje x, el eje y y el título de la gráfica
    plt.xlabel('Epochs')
    plt.ylabel('loss/auc') 
    plt.title('Gráfico evolución entrenamiento y validación')
    
    # Se añade la leyenda
    plt.legend()
    
    # Se muestra la gráfica
    plt.show()



def matriz_conf(ruta, batch_size, target_size, epochs, modelo, titulo):
    
    '''
    Función que crea una matriz de confusión a partir de la carga de un modelo dentro de la carpeta ''Modelos''.
    ----------------------------------------------------------------
    Parámetros:
    - ruta: str. Ruta base donde se encuentran las imágenes organizadas en subcarpetas (train, val, test). Ruta data_nuevo
    - batchsize: int. Tamaño del lote que se utiliza en una única iteración del algoritmo de aprendizaje
    - target_size: tupla de números enteros que representa el alto y ancho al que se van a redimensionar todas las imágenes. 
    - epochs: int. Número de épocas a entrenar
    - modelo: modelo previamente entrenado y almacenado dentro de la carpeta ''Modelos'' del cual se quiere obtener la matriz de confusión.
    - titulo: str. Título que se quiere asignar a la matriz de confusión
    -----------------------------------------------------------
    Return:
    - nada
    '''

    #se genera los datos
    train_generator, validation_generator, test_generator = preparar_modelo(ruta, batch_size, target_size)
    
    # se calcula y_test e y_pred del modelo pasado como parámetro para obtener la matriz de confusion
    y_test=test_generator.labels
    y_pred=modelo.predict(test_generator)
    y_pred_bin=np.where(y_pred>=0.5,1,0) #para convertirlo en un problema binario
    
        
    # matriz de confusión visualmente
    labels=np.unique(y_test)
        
    matriz_conf = metrics.confusion_matrix(y_test, y_pred_bin,labels=labels)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = matriz_conf, display_labels = ["NORMAL" , "PNEUMONIA"])
    fig, ax = plt.subplots(figsize=(5,5))
    cm_display.plot(ax=ax)
    plt.title(titulo)
    plt.show()







