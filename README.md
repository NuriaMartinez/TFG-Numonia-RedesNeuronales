<img style="float:left" width="80%" src="/codigo/pics/cabeceraSalud.jpg">

# Trabajo Fin de Grado

<h2 style="display: inline-block; padding: 4mm; padding-left: 2em; background-color: navy; line-height: 1.3em; color: white; border-radius: 10px;">NOMBRE TFG</h2>


### **Nuria Martínez Queralt**  
### Tutor 1: **Daniel Urda Muñoz**  
### Tutor 2: **Fernando Gustavo Gutierrez Herrero**  

## **Resumen**
La neumonía es una infección respiratoria aguda que afecta a las vías respiratorias y a los alvéolos. Causando que los sacos de aire, o alvéolos, en los pulmones se llenen de líquido o pus, lo que provoca su inflamación. 

Se trata de una de las principales causas de muerte tanto en niños como entre personas mayores con una tasa de mortalidad anual de 2,5 millones en los últimos años. El diagnóstico precoz es imprescindible para un correcto tratamiento.

Para su diagnóstico se emplean diversas técnicas como la resonancia magnética (RM), la radiografía de tórax (CXT por sus siglas en inglés) y la tomografía computarizada (TC). La CXT es una de las técnicas más empleadas y útiles a la hora de diagnosticar neumonía debido a la gran cantidad de información que ofrece. Aunque, también puede producir resultados confusos incluso para radiólogos especializados debido a la similitud que existe entre estas imágenes y las de otras anomalías pulmonares como el cáncer de pulmón o el exceso de líquido.

Por ello, este trabajo se enfoca en proporcionar ayuda al personal médico en el diagnóstico de neumonía a partir de la interpretación de imágenes médicas como las CXT por medio de la inteligencia artificial (IA), con el objetivo de adelantar el diagnóstico y mejorar su precisión.

## **Abstract**
Pneumonia is an acute respiratory infection that affects the airways and alveoli. It causes the air sacs, or alveoli, in the lungs to fill with fluid or pus, causing them to become inflamed. 

It is a leading cause of death in both children and the elderly with an annual mortality rate of 2.5 million in recent years. Early diagnosis is essential for correct treatment.

Various techniques such as magnetic resonance imaging (MRI), chest X-ray (CXT) and computed tomography (CT) are used for diagnosis. CXT is one of the most widely used and useful techniques for diagnosing pneumonia because of the wealth of information it provides. However, it can also produce confusing results even for specialised radiologists due to the similarity between these images and those of other lung abnormalities such as lung cancer or excess fluid.

Therefore, this work focuses on providing assistance to medical staff in the diagnosis of pneumonia through the interpretation of medical images such as CXT images by means of artificial intelligence (AI), with the aim of advancing the diagnosis and improving its accuracy.

## **Estructura del repositorio**

## **Descripción de los datos**
Debido a la gran cantidad de imágenes de radiografía de tórax con las que se trabaja, no es posible subir a GitHub la carpeta con todas las imágenes por lo que, este acceso se consigue a través de distintos enlaces a OneDrive.

En realidad, no se trata de una sola carpeta que contenga las imágenes ya que, la carpeta inicial con la que se trabaja no dispone de una buena proporcionalidad en el número de imágenes de las distintas subcarpetas (train, test y val) 
por lo que, por medio de código de python, se tuvo que crear otra carpeta con las mismas subcarpetas que la inicial, pero con una nueva distribución de las imágenes.

Por lo tanto, en este README se proporcionan dos enlaces, uno de ellos a la carpeta ''data'' la cual se corresponde con la carpeta inicial descargada de internet con las imágenes de radiografía de tórax con y sin neumonía
distribuidas en distintas subcarpetas. Y, el otro enlace lleva a la carpeta ''data_nuevo'', que es la carpeta creada con la redistribución de las imágenes y la carpeta principal que se emplea en este trabajo.

- Link a la carpeta ''data'': https://universidaddeburgos-my.sharepoint.com/:u:/g/personal/nmq1001_alu_ubu_es/ESVy_MWCxGhOjVv4diaD40QBizFKxoUYpSXz2Sdlr6Afeg?e=VbxbEj
- Link a la carpeta ''data_nuevo'': https://universidaddeburgos-my.sharepoint.com/:u:/g/personal/nmq1001_alu_ubu_es/EQdZechTq-VIsWCQq9-I-EIBKMhStRhQvBI1mi-bIw4Qfg?e=7LKcFl
