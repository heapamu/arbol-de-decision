/*****************************************************************************************************
Actividad de Evaluación del tema 3: Inducción de un árbol con ML
Parte 2
Determinar la profundidad óptima del árbol, entre los valores 3, 5, 7 y 9. Para ello crear
árboles de distinta profundidad y estimar su tasa de error.

Héctor Jesús Aparicio Muñoz
*****************************************************************************************************/

// De la parte 1 de este ejercicio de evaluación ya tenemos generados los DataFrames de los conjuntos
// de datos de Entrenamiento y de Test, en el formato necesario para que se les pueda aplicar el modelo,
// es decir, con dos columnas, features y label.

// Importamos los paquetes necesarios, aunque en realidad ya los tenemos cargados de la parte 1 del ejercicio
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.mllib.evaluation.MulticlassMetrics

// Creamos los siguientes Arrays para almacenar los resultados de los modelos que vamos a generar
val valoresMaxProf = Array(3, 5, 7, 9)
val acierto = Array(0.0, 0.0, 0.0, 0.0)
val error = Array(0.0, 0.0, 0.0, 0.0)

// Creamos una instancia del modelo de clasificación deseado, que es DecisionTreeClassifier
val DTincome = new DecisionTreeClassifier()

// Fijamos los parámetros del modelo
val impureza = "entropy"
val maxBins = 41

DTincome.setImpurity(impureza)
DTincome.setMaxBins(maxBins)

// El parámetro profundidad del modelo lo fijaremos dentro de la estructura de control siguiente
for (i <- 0 to 3)
{
	val maxProf = valoresMaxProf(i)
	DTincome.setMaxDepth(maxProf)   // Fijamos la profundidad óptima del modelo en cada iteración

	val DTincomeModel = DTincome.fit(trainRegistrosDF)   // Entrenamos los modelos
	DTincomeModel.toDebugString   // Examinamos los árboles generados en cada iteración

	// Predicciones sobre el conjunto de prueba
	val predictionsAndLabelsDF = DTincomeModel.transform(testRegistrosDF).select("prediction", "label")

	val predictionsAndLabelsRDD = predictionsAndLabelsDF.rdd.map(row => (row.getDouble(0), row.getDouble(1)))  // Pasamos de DF a RDD

	val metrics = new MulticlassMetrics(predictionsAndLabelsRDD)   // Creamos una instancia de MulticlassMetrics
	acierto(i) = metrics.accuracy   // Tasa de acierto del modelo generado en cada iteración
	error(i) = 1 - acierto(i)   // Tasa de error del modelo generado en cada iteración

	println("Con maxProf = " + maxProf.toString + " la precision es: " + acierto(i).toString + ", y la tasa de error: " + error(i).toString)
}

// Elegimos la maxProf = 9 que es el que mayor tasa de acierto tiene (menor tasa de error por lo tanto)
println("Elegimos el árbol con profundidad maxProf=9")

// Volvemos a entrenar el modelo con todos los datos de Entrenamiento
val maxProf = 9
DTincome.setMaxDepth(maxProf)
val DTincomeModel = DTincome.fit(registrosFeaturesLabelDF)

// Guardamos el modelo final
DTincomeModel.save( PATH + "DTincomeModelFinalML")

// Vamos a aplicar el modelo final sobre el conjunto de datos de Test, que ya tenemos en el formato necesario para
// poder utilizarse, DataFrame con las columnas features y label
val predictionsAndLabelsDF = DTincomeModel.transform(registrosTestFeaturesLabelDF).select("prediction", "label")

val predictionsAndLabelsRDD = predictionsAndLabelsDF.rdd.map(row => (row.getDouble(0), row.getDouble(1)))  // Pasamos de DF a RDD

val metrics = new MulticlassMetrics(predictionsAndLabelsRDD)   // Creamos una instancia de MulticlassMetrics
val aciertoTest = metrics.accuracy   // Tasa de acierto del modelo sobre el conjunto de datos de Test
val errorTest = 1 - aciertoTest   // Tasa de error del modelo sobre el conjunto de datos de Test

println("El modelo tiene una precision de: " + aciertoTest.toString + ", y una tasa de error de: " + errorTest.toString + ", para el conjunto de datos de Test " + PATH + Test)

