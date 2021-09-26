/*****************************************************************************************************
Actividad de Evaluación del tema 3: Inducción de un árbol con ML
Parte 1
Crear un árbol de decisión con ML, de profundidad máxima 3, con el conjunto Entrenamiento
y estimar su tasa de error sobre el conjunto Test.

Héctor Jesús Aparicio Muñoz
*****************************************************************************************************/

// Importamos los paquetes necesarios
import spark.implicits._
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.StringIndexerModel
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.mllib.evaluation.MulticlassMetrics

// Vamos a definir una case class denominada regCenso con el esquema del RDD y del DataFrame que vamos a generar. Otra opción
// hubiera sido crear los DataFrames con StructType
case class RegCenso (age: Int, workClass: String, fnlwgt: Int, education: String, educationNum: Int, maritalStatus: String, occupation: String, relationship: String, race: String, sex: String, capitalGain: Int, capitalLoss: Int, hoursPerWeek: Int, nativeCountry: String, income: String)

// Haremos primeramente la lectura de los datos sobre RDDs
val PATH="/home/hector/Documentos/Census+Income/"

val Entrenamiento = "adult.data"
val Test = "adult.test"

val training = sc.textFile(PATH + Entrenamiento)
val test = sc.textFile(PATH + Test)

println("Compruebo el número de registros en cada fichero y observo además el primero de cada uno de ellos, y vemos que la primera línea de test hay que eliminarla")
training.count()
test.count()
training.first()
test.first()

// Elimino la primera línea de test
val testSinPrimera = test.filter(x => ! x.contains("|1x3 Cross"))
testSinPrimera.count()
testSinPrimera.first()

// Elimino las lineas vacias y quito los registros con alguna "?"
val nonEmpty = training.filter(x => x.nonEmpty).filter(y => ! y.contains("?"))
nonEmpty.count()
val nonEmptyTest = testSinPrimera.filter(x => x.nonEmpty).filter(y => ! y.contains("?"))
nonEmptyTest.count()

val campos = nonEmpty.map{linea => linea.split(", ")}
campos.first()
val camposTest = nonEmptyTest.map{linea => linea.split(", ")}
camposTest.first()

// Convierto los RDDs anteriores en RDDs de la case class RegCenso
val registros = campos.map(x => RegCenso(x(0).toInt, x(1), x(2).toInt, x(3), x(4).toInt, x(5), x(6), x(7), x(8), x(9), x(10).toInt, x(11).toInt, x(12).toInt, x(13), x(14)))
registros.first()
val registrosTest = camposTest.map(x => RegCenso(x(0).toInt, x(1), x(2).toInt, x(3), x(4).toInt, x(5), x(6), x(7), x(8), x(9), x(10).toInt, x(11).toInt, x(12).toInt, x(13), x(14)))
registrosTest.first()

// -----------------------------------------------------------------------------------------------------------------------
// ---------------------- Descripción del conjunto de datos --------------------------------------------------------------
println("***********************************************************************************************************************")

// A pesar de que ya ha sido realizada en el ejercicio de Autoevaluación, incluyo aquí también la descripción de los datos
// ya que considero que debe realizarse en todos los proyectos que trabajen sobre datos. La haré tanto para el conjunto de
// datos de Entrenamiento como para el conjunto de datos de Test.

// -----> Descripción del conjunto de datos de Entrenamiento:
println("Descripción del conjunto de datos de Entrenamiento:")

// Calculo los valores estadísticos que describirán los datos continuos
println("Estadisticas para los distintos atributos continuos del conjunto de datos de Entrenamiento:")
val edadStatistics = registros.map(r => r.age).stats
val fnlwgtStatistics = registros.map(r => r.fnlwgt).stats
val capitalGainStatistics = registros.map(r => r.capitalGain).stats
val capitalLossStatistics = registros.map(r => r.capitalLoss).stats
val hPWStatistics = registros.map(r => r.hoursPerWeek).stats

println("age: " + edadStatistics)
println("fnlwgt: " + fnlwgtStatistics)
println("capitalGain: " + capitalGainStatistics)
println("capitalLoss: " + capitalLossStatistics)
println("hoursPerWeek: " + hPWStatistics)

// Vamos a describir ahora los datos categóricos
// Contamos primeramente el número de valores distintos que tiene cada uno de esos atributos
val numWorkClass = registros.map{r => r.workClass}.distinct().count()
val numEducation = registros.map{r => r.education.trim}.distinct().count()
val numMaritalStatus = registros.map{r => r.maritalStatus.trim}.distinct().count()
val numOccupation = registros.map{r => r.occupation.trim}.distinct().count()
val numRelationship = registros.map{r => r.relationship.trim}.distinct().count()
val numRace = registros.map{r => r.race.trim}.distinct().count()
val numSex = registros.map{r => r.sex.trim}.distinct().count()
val numNativeCountry = registros.map{r => r.nativeCountry}.distinct().count()

// Calculo el número de valores distintos que puede tomar la clase
val numClaseDistintos = registros.map(r => r.income).distinct().count()

println("Valores distintos para los atributos discretos del conjunto de datos de Entrenamiento:")
println("workClass: " + numWorkClass.toString)
println("education: " + numEducation.toString)
println("maritalStatus: " + numMaritalStatus.toString)
println("occupation: " + numOccupation.toString)
println("relationship: " + numRelationship.toString)
println("race: " + numRace.toString)
println("sex: " + numSex.toString)
println("nativeCountry: " + numNativeCountry.toString)
println("Valores para la clase income: " + numClaseDistintos.toString)

// Enumero los valores distintos que puede tomar cada atributo, ordenados además por frecuencia de aparición
val valoresWorkClass = registros.map{r => (r.workClass, 1)}.reduceByKey(_ + _).collect().sortBy(- _._2)
val valoresEducation = registros.map{r => (r.education, 1)}.reduceByKey(_ + _).collect().sortBy(- _._2)
val valoresMaritalStatus = registros.map{r => (r.maritalStatus, 1)}.reduceByKey(_ + _).collect().sortBy(- _._2)
val valoresOccupation = registros.map{r => (r.occupation, 1)}.reduceByKey(_ + _).collect().sortBy(- _._2)
val valoresRelationship = registros.map{r => (r.relationship, 1)}.reduceByKey(_ + _).collect().sortBy(- _._2)
val valoresRace = registros.map{r => (r.race, 1)}.reduceByKey(_ + _).collect().sortBy(- _._2)
val valoresSex = registros.map{r => (r.sex, 1)}.reduceByKey(_ + _).collect().sortBy(- _._2)
val valoresNativeCountry = registros.map{r => (r.nativeCountry, 1)}.reduceByKey(_ + _).collect().sortBy(- _._2)

// Valores distintos que puede tomar la clase, ordenados por frecuencia y porcentaje de aparición
val valoresClase = registros.map{r => (r.income, 1)}.reduceByKey(_ + _).collect().sortBy(-_._2)
val valoresClaseDistintos = valoresClase.map{r => (r._1, r._2, (r._2/registros.count().toDouble)*100)}

println("Distribucion de valores por frecuencia de aparicion para los atributos discretos del conjunto de datos de Entrenamiento: ")
println("WorkClass: ")
valoresWorkClass.foreach(println)

println("Education: ")
valoresEducation.foreach(println)

println("MaritalStatus: ")
valoresMaritalStatus.foreach(println)

println("Occupation: ")
valoresOccupation.foreach(println)

println("Relationship: ")
valoresRelationship.foreach(println)

println("Race: ")
valoresRace.foreach(println)

println("Sex: ")
valoresSex.foreach(println)

println("NativeCountry: ")
valoresNativeCountry.foreach(println)

println("Distribución de clases.")
println("Valores para la clase income: ")
println("(Class, N, N[%])")
valoresClaseDistintos.foreach(println)


// -----> Descripción del conjunto de datos de Test:
println("Descripción del conjunto de datos de Test:")

// Calculo los valores estadísticos que describirán los datos continuos
println("Estadisticas para los distintos atributos continuos del conjunto de datos de Test:")
val edadStatisticsTest = registrosTest.map(r => r.age).stats
val fnlwgtStatisticsTest = registrosTest.map(r => r.fnlwgt).stats
val capitalGainStatisticsTest = registrosTest.map(r => r.capitalGain).stats
val capitalLossStatisticsTest = registrosTest.map(r => r.capitalLoss).stats
val hPWStatisticsTest = registrosTest.map(r => r.hoursPerWeek).stats

println("age: " + edadStatisticsTest)
println("fnlwgt: " + fnlwgtStatisticsTest)
println("capitalGain: " + capitalGainStatisticsTest)
println("capitalLoss: " + capitalLossStatisticsTest)
println("hoursPerWeek: " + hPWStatisticsTest)

// Vamos a describir ahora los datos categóricos
// Contamos primeramente el número de valores distintos que tiene cada uno de esos atributos
val numWorkClassTest = registrosTest.map{r => r.workClass}.distinct().count()
val numEducationTest = registrosTest.map{r => r.education.trim}.distinct().count()
val numMaritalStatusTest = registrosTest.map{r => r.maritalStatus.trim}.distinct().count()
val numOccupationTest = registrosTest.map{r => r.occupation.trim}.distinct().count()
val numRelationshipTest = registrosTest.map{r => r.relationship.trim}.distinct().count()
val numRaceTest = registrosTest.map{r => r.race.trim}.distinct().count()
val numSexTest = registrosTest.map{r => r.sex.trim}.distinct().count()
val numNativeCountryTest = registrosTest.map{r => r.nativeCountry}.distinct().count()

// Calculo el número de valores distintos que puede tomar la clase
val numClaseDistintosTest = registrosTest.map(r => r.income).distinct().count()

println("Valores distintos para los atributos discretos del conjunto de datos de Test:")
println("workClass: " + numWorkClassTest.toString)
println("education: " + numEducationTest.toString)
println("maritalStatus: " + numMaritalStatusTest.toString)
println("occupation: " + numOccupationTest.toString)
println("relationship: " + numRelationshipTest.toString)
println("race: " + numRaceTest.toString)
println("sex: " + numSexTest.toString)
println("nativeCountry: " + numNativeCountryTest.toString)
println("Valores para la clase income: " + numClaseDistintosTest.toString)

// Enumero los valores distintos que puede tomar cada atributo, ordenados además por frecuencia de aparición
val valoresWorkClassTest = registrosTest.map{r => (r.workClass, 1)}.reduceByKey(_ + _).collect().sortBy(- _._2)
val valoresEducationTest = registrosTest.map{r => (r.education, 1)}.reduceByKey(_ + _).collect().sortBy(- _._2)
val valoresMaritalStatusTest = registrosTest.map{r => (r.maritalStatus, 1)}.reduceByKey(_ + _).collect().sortBy(- _._2)
val valoresOccupationTest = registrosTest.map{r => (r.occupation, 1)}.reduceByKey(_ + _).collect().sortBy(- _._2)
val valoresRelationshipTest = registrosTest.map{r => (r.relationship, 1)}.reduceByKey(_ + _).collect().sortBy(- _._2)
val valoresRaceTest = registrosTest.map{r => (r.race, 1)}.reduceByKey(_ + _).collect().sortBy(- _._2)
val valoresSexTest = registrosTest.map{r => (r.sex, 1)}.reduceByKey(_ + _).collect().sortBy(- _._2)
val valoresNativeCountryTest = registrosTest.map{r => (r.nativeCountry, 1)}.reduceByKey(_ + _).collect().sortBy(- _._2)

// Valores distintos que puede tomar la clase, ordenados por frecuencia y porcentaje de aparición
val valoresClaseTest = registrosTest.map{r => (r.income, 1)}.reduceByKey(_ + _).collect().sortBy(-_._2)
val valoresClaseDistintosTest = valoresClaseTest.map{r => (r._1, r._2, (r._2/registrosTest.count().toDouble)*100)}

println("Distribucion de valores por frecuencia de aparicion para los atributos discretos del conjunto de datos de Test: ")
println("WorkClass: ")
valoresWorkClassTest.foreach(println)

println("Education: ")
valoresEducationTest.foreach(println)

println("MaritalStatus: ")
valoresMaritalStatusTest.foreach(println)

println("Occupation: ")
valoresOccupationTest.foreach(println)

println("Relationship: ")
valoresRelationshipTest.foreach(println)

println("Race: ")
valoresRaceTest.foreach(println)

println("Sex: ")
valoresSexTest.foreach(println)

println("NativeCountry: ")
valoresNativeCountryTest.foreach(println)

println("Distribución de clases.")
println("Valores para la clase income: ")
println("(Class, N, N[%])")
valoresClaseDistintosTest.foreach(println)

println("***********************************************************************************************************************")
// -----------------------------------------------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------------------------------------------

// *** Creamos los DataFrames de Entrenamiento y de Test
val registrosDF = registros.toDF
registrosDF.show()
val registrosTestDF = registrosTest.toDF
registrosTestDF.show()

// *** Vamos a transformar todos los atributos nominales (categóricos) a índices numéricos
// mediante la función indexStringColumns la cual utiliza StringIndexer

// Definimos la función indexStringColumns
def indexStringColumns(df:DataFrame, cols:Array[String]):DataFrame =
{
	var newdf = df
	for(col <- cols)
	{
		val si = new
		StringIndexer().setInputCol(col).setOutputCol(col+"-num")
		val sm:StringIndexerModel = si.fit(newdf)
		newdf = sm.transform(newdf).drop(col)
		newdf = newdf.withColumnRenamed(col+"-num", col)
	}
	newdf
}

// La aplicamos sobre los DataFrames registrosDF y registrosTestDF
val registrosDFnumeric = indexStringColumns(registrosDF, Array("workClass", "education", "maritalStatus", "occupation", "relationship", "race", "sex", "nativeCountry"))
val registrosTestDFnumeric = indexStringColumns(registrosTestDF, Array("workClass", "education", "maritalStatus", "occupation", "relationship", "race", "sex", "nativeCountry"))

// Veamos cómo quedan los DataFrames de Entrenamiento y de Test
registrosDFnumeric.show(10)
registrosTestDFnumeric.show(10)


// *** Vamos a crear ahora la columna features con VectorAssembler y la columna label, en los DataFrames de Entrenamiento y de Test

// ----> Para el DataFrame de Entrenamiento

val va = new VectorAssembler().setOutputCol("features").setInputCols(registrosDFnumeric.columns.diff(Array("income")))
// Lo aplicamos al DataFrame
val registrosDFlpoints = va.transform(registrosDFnumeric).select("features", "income")

// A continuación etiquetamos la clase
val indiceClase= new StringIndexer().setInputCol("income").setOutputCol("label")

val registrosFeaturesLabelDF = indiceClase.fit(registrosDFlpoints).transform(registrosDFlpoints).drop("income")

// Veamos cómo queda el DataFrame preparado para aplicarle el algoritmo de aprendizaje
registrosFeaturesLabelDF.show(10)

// ----> Para el DataFrame de Test

val va = new VectorAssembler().setOutputCol("features").setInputCols(registrosTestDFnumeric.columns.diff(Array("income")))
// Lo aplicamos al DataFrame
val registrosTestDFlpoints = va.transform(registrosTestDFnumeric).select("features", "income")

// A continuación etiquetamos la clase
val indiceClase= new StringIndexer().setInputCol("income").setOutputCol("label")

val registrosTestFeaturesLabelDF = indiceClase.fit(registrosTestDFlpoints).transform(registrosTestDFlpoints).drop("income")

// Veamos cómo queda el DataFrame preparado para aplicarle el algoritmo de aprendizaje
registrosTestFeaturesLabelDF.show(10)


// *** Vamos ahora a realizar el entrenamiento del clasificador.

// Realizaremos primeramente la partición aleatoria de los datos de Entrenamiento, con el 66% para
// entrenamiento y 34% para prueba.

val dataSplits = registrosFeaturesLabelDF.randomSplit(Array(0.66, 0.34))
val trainRegistrosDF = dataSplits(0)
val testRegistrosDF = dataSplits(1)

// A continuación creamos la instancia del modelo de clasificación deseado,
// es decir, creamos una instancia de DecisionTreeClassifier, que será un estimator
// que necesitaremos entrenar

val DTincome = new DecisionTreeClassifier()

// Fijamos los parámetros del modelo

val impureza = "entropy"
val maxProf = 3
val maxBins = 41   // Le asigno este valor ya que es el mayor número de valores distintos que tiene el atributo con
                   // más valores distintos
DTincome.setImpurity(impureza)
DTincome.setMaxDepth(maxProf)
DTincome.setMaxBins(maxBins)

// Entrenamos el modelo
val DTincomeModel = DTincome.fit(trainRegistrosDF)   // Ese modelo entrenado es un transformer

// Examinamos el árbol generado
DTincomeModel.toDebugString

// Ahora predecimos sobre el conjunto de prueba
val predictionsAndLabelsDF = DTincomeModel.transform(testRegistrosDF).select("prediction", "label")
// Veamos el DataFrame obtenido
predictionsAndLabelsDF.show


// *** Evaluación y tasa de error.
// Debemos de convertir el DataFrame a RDD, ya que ML aún no soporta la clase MulticalssMetrics

val predictionsAndLabelsRDD = predictionsAndLabelsDF.rdd.map(row => (row.getDouble(0), row.getDouble(1)))

// Calculamos ahora la tasa de error.
// Para ello primero creamos una instancia de MulticlassMetrics
val metrics = new MulticlassMetrics(predictionsAndLabelsRDD)

// Tasa de acierto
val acierto = metrics.accuracy

// Tasa de error
val error = 1 - acierto


// *** Por último vamos a guardar el modelo
DTincomeModel.save( PATH + "DTincomeModelML")

