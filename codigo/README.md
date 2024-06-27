Debido al espacio limitado para emplear en GitHub, se han añadido las carpetas obtenidas tras la ejecución de código en este README a partir del link de OneDrive.

Se trata de las carpetas ''Datos'', ''Historicos'', ''Modelos'', ''Resultados'',''Historicos_Py'', ''Modelos_Py'' y ''Resultados_Py''.

La carpeta ''Datos'' incluye 5 archivos en formato csv, los cuales se corresponden con los distintos dataframes obtenidos en la primera función ''redestribucion_imagenes''. En la cual, se busca redistribuir las imágenes de la carpeta data de una forma más equitativa. Todos los csv están formados por dos columnas, ''nombres_ficheros'' que corresponde con el nombre de las imágenes y ''clases'' que puede ser un 1 o un 0 en función si la imagen tiene neumonía o no.

La carpeta ''Historicos'', incluye diversas subcarpetas, cada una de ellas perteneciente a un modelo o batch size determinado para CNN propia o CNN Alex Net y para distinto número de neuronas. Dentro de cada subcarpeta, se encuentran distintos csv obtenidos para cada uno de esos casos. En estos csv se incluyen los valores de diferentes métricas obtenidas durante el entrenamiento en cada época para ese modelo, batch size o número de neuronas concretos. Estos resultados se corresponden con la ejecución del notebook ''redes neuronales - neumonia''.

La carpeta ''Modelos'' incluye diversas subcarpetas, cada una de ellas perteneciente a un modelo o batch size determinado para CNN propia o CNN Alex Net y para distinto número de neuronas. Dentro de cada subcarpeta, se encuentran distintos archivos obtenidos para cada modelo y cada batch size de la CNN propia, de la CNN Alexnet y para distinto número de neuronas del modelo Simple3 y bacht size 64. Estos resultados se corresponden con la ejecución del notebook ''redes neuronales - neumonia''. Estos archivos, se encuentran en un formato HDF, el cual se emplea para almacenar grandes cantidades de datos (numéricos, gráficos y de texto) de una forma jerárquica, por lo que, la gestión de estos datos se realiza de una forma eficiente. Dentro de cada archivo, se encuentran los valores de pesos del modelo, su estructura completa (capas, conexiones, etc.), información sobre el optimizador y su estado y la configuración de compilación del modelo que incluye la función de pérdida y las métricas.

La carpeta ''Resultados'', incluye los distintos dataframes en formato csv donde se muestran las tablas comparativas obtenidas en este trabajo. Estos resultados se corresponden con la ejecución del notebook ''redes neuronales - neumonia''.

Las carpetas ''Historicos_Py'', ''Modelos_Py'' y ''Resultados_Py'' incluyen lo mismo que las carpetas ''Historicos'', ''Modelos'' y ''Resultados'' respectivamente, pero, tras la ejecución del archivo ''redes neuronales ejecución archivo.py''. Existe esta distinción debido a que los csv obtenidos tras cada ejecución pueden ser ligeramente distintos dado la aleatoriedad de pesos y muestras iniciales en cada ejecución (explicado en el apartado de ''Resultados'' de la memoria).

Cabe mencionar que, una vez se ejecutan los notebooks, estas carpetas aparecen automáticamente en el mismo directorio donde se encuentra el notebook pero, en este caso, para acceder a dichas carpetas se tiene que hacer a partir de los links que se incluyen a continuación y descargarlas desde OneDrive. Ya que, debido al espacio limitado de GitHub no es posible añadir estas carpetas directamente dentro de la carpeta ''codigo''.

Las carpetas se encuentran comprimidas en zip por lo que, para poder abrirlas será necesario descomprimirlas previamente.

A continuación, se incluyen los links para cada una de las carpetas mencionadas previamente:
-	Link a la carpeta ''Datos'': https://universidaddeburgos-my.sharepoint.com/:u:/g/personal/nmq1001_alu_ubu_es/EZ8LorRxSA5Iloj3xrqpNcMB3MWCLjkjo0WXN73hTDo-oA?e=prWsvA
-	Link a la carpeta ''Historicos'': https://universidaddeburgos-my.sharepoint.com/:u:/g/personal/nmq1001_alu_ubu_es/EajSsbq0mBRPuUNbaIfqgpEB7lGU9EqPl184BaLr54ok8g?e=Wg5KdE
-	Link a la carpeta ''Modelos'': https://universidaddeburgos-my.sharepoint.com/:u:/g/personal/nmq1001_alu_ubu_es/EYbNj_Y1kYpLi4qOJHMxmC8BjUwwImJvPci-o2rrmE4d-w?e=4j072G
-	Link a la carpeta ''Resultados'': https://universidaddeburgos-my.sharepoint.com/:u:/g/personal/nmq1001_alu_ubu_es/EVSmiGBpAGVEqtenA_z1w0gBEor3Uz6AeSVz7zKvowpEZg?e=DkyJYe
-	Link a la carpeta ''Historicos_Py'': https://universidaddeburgos-my.sharepoint.com/:u:/g/personal/nmq1001_alu_ubu_es/EbHEtKCBMwVHuLjSqCfZAWUB7p__xISxJ2DVbmqRHVMoRA?e=4oKL2I
-	Link a la carpeta ''Modelos_Py'': https://universidaddeburgos-my.sharepoint.com/:u:/g/personal/nmq1001_alu_ubu_es/EcGwNiLnTwRMjfC2_t1YncQBu3AGtJ77hfUklA1ZQ2vJUA?e=4ZZKM1
-	Link a la carpeta ''Resultados_Py'': https://universidaddeburgos-my.sharepoint.com/:u:/g/personal/nmq1001_alu_ubu_es/EYFFut4lK95BjCYclccB3JQBHj9oDsq_Ks64ltOKZ0KmdQ?e=fG9BjN
