DESCRIPCIÓN DEL CÓDIGO IMPLEMENTADO:

El ejercicio consiste en inducir un árbol de decisión con la biblioteca de Spark ML, para el conjunto de datos ”Census Income Data Set” disponible en https://archive.ics.uci.edu/ml/datasets/Census+Income. Es un problema de clasificación binaria, consistente en predecir si los ingresos de un contribuyente superan un cierto umbral a partir de datos públicos del Censo de EEUU de 1994 (se puede ver la descripción en el fichero "adult.names").

Las instancias están descritas por 13 atributos entre nominales y continuos, más la información de clase. Hay valores desconocidos en los atributos de algunas instancias. Los datos ya están separados en dos conjuntos, Entrenamiento (32000 ejemplos, aprox.) y Test (12000 ejemplos, aprox.) El conjunto Test sólo se puede utilizar para estimar la tasa de error del modelo final. El conjunto Entrenamiento se puede utilizar para probar y para estimar la tasa de error de los árboles construidos con diferentes parámetros, realizando una partición del mismo en
Entrenamiento y Validación.

Se incluyen los conjuntos de Entrenamiento ("adult.data") y Test ("adult.test").

Se ha dividido el código en dos partes:

1) En la primera "evaluacionTema3_parte1.scala" se ha creado un árbol de decisión con ML, de profundidad máxima 3, con el conjunto Entrenamiento y se ha estimado su tasa de error sobre el conjunto Test. Este script incluye todas las etapas, desde la lectura de datos hasta el almacenamiento del modelo final.

2) En la segunda "evaluacionTema3_parte2.scala" se determina la profundidad óptima del árbol, entre los valores 3, 5, 7 y 9. Para ello se crean árboles de distinta profundidad y se estima su tasa de error sobre el conjunto de Validación. Se selecciona la profundidad con una menor tasa de error, para posteriormente volver a crear un árbol con todos los datos de Entrenamiento y la profundidad seleccionada, y finalmente estimar su tasa de error con el conjunto Test.

NOTA: Los scripts implementados corren sobre cualquier máquina Linux que tenga instalado Spark 2.4.0 sin más que modificar el PATH al directorio de los datos.

NOTA: Todo el código implementado está convenientemente comentado para su mejor comprensión.

