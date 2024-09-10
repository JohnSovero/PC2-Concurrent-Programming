package main

import (
    "fmt"
    //"time"
    "math/rand"
    "PC2/algorithms/randomforest"
    //"PC2/algorithms/svm"
    //"PC2/algorithms/dnn"
    //"math/rand"
    //"PC2/algorithms/fc"
    //"PC2/utils"
    //"strconv"
)


func main() {
    //df := utils.Load_dataset("dataset/delays_zurich_transport_small.csv")
	//if df == nil {
	//	fmt.Println("No se pudo cargar el dataset.")
	//	return
	//}
//
	//var xData [][]float64
	//var yData []float64
    //var headers []string
//
	//// Obtener la columna 'delay'
	//delayColumn, err := df.GetColumn("delay")
	//if err != nil {
	//	fmt.Println(err)
	//	return
	//}
//
	//// Convertir la columna 'delay' a float64 y agregarla a yData
	//for _, value := range delayColumn {
	//	floatValue, err := strconv.ParseFloat(value, 64)
	//	if err != nil {
	//		fmt.Println("Error al convertir el valor a float64:", err)
	//		return
	//	}
	//	yData = append(yData, floatValue)
	//}
//
	//// Obtener los nombres de las columnas excepto 'delay'
	//for columnName := range df.Columns {
	//	if columnName == "delay" {
	//		continue
	//	}
	//	headers = append(headers, columnName)
	//	column, err := df.GetColumn(columnName)
	//	if err != nil {
	//		fmt.Println(err)
	//		return
	//	}
//
	//	var colData []float64
	//	// Convertir cada valor de la columna a float64
	//	for _, value := range column {
	//		floatValue, err := strconv.ParseFloat(value, 64)
	//		if err != nil {
	//			fmt.Println("Error al convertir el valor a float64:", err)
	//			return
	//		}
	//		colData = append(colData, floatValue)
	//	}
	//	xData = append(xData, colData)
	//}
//
	//// Visualizar los datos en un formato legible
	//fmt.Println("Datos del DataFrame:")
	//fmt.Printf("%-10s", "delay")
	//for _, header := range headers {
	//	fmt.Printf("%-10s", header)
	//}
	//fmt.Println()
//
	//for i := 0; i < len(yData); i++ {
	//	fmt.Printf("%-10.2f", yData[i])
	//	for j := 0; j < len(headers); j++ {
	//		fmt.Printf("%-10.2f", xData[j][i])
	//	}
	//	fmt.Println()
	//}
	
    //--------------------------------------------------------------Random Forest
    
   	xData := [][]float64{}
	yData := []float64{}
	for i := 0; i < 1000; i++ {
		x := []float64{rand.Float64(), rand.Float64(), rand.Float64(), rand.Float64()}
		y := float64(x[0] + x[1] + x[2] + x[3])
		xData = append(xData, x)
		yData = append(yData, y)
	}
	forest := randomforest.RandomForest{NumTrees: 5, MaxDepth: 3}
	forest.Train(xData, yData)

	prediction := forest.Predict([]float64{1.0, 1.0, 1.0, 1.0})
	fmt.Printf("Predicción: %.2f\n", prediction)
    // Medir el tiempo de ejecución para ForestSequencial
    //utils.MeasureExecutionTime("ForestSequencial", func() {
    //    forestSequencial := randomforest.ForestSequencial{}
    //    forestSequencial.Data = randomforest.ForestDataSequencial{X: xData, Class: yData}
    //    forestSequencial.TrainSequecial(1000)
    //    vote := forestSequencial.Vote([]float64{0.1, 0.1, 0.1, 0.1})
    //    if vote[0] < vote[1] || vote[0] < vote[2] || vote[0] < vote[3] {
    //        fmt.Println("Wrong Machine Learning !")
    //    }
    //    vote = forestSequencial.Vote([]float64{0.9, 0.9, 0.9, 0.9})
    //    if vote[3] < vote[0] || vote[3] < vote[1] || vote[3] < vote[2] {
    //        fmt.Println("Wrong Machine Learning !")
    //    }
    //    fmt.Println("Vote", forestSequencial.Vote([]float64{0.1, 0.1, 0.1, 0.1}))
    //    fmt.Println("Vote", forestSequencial.Vote([]float64{0.9, 0.9, 0.9, 0.9}))
    //    fmt.Println("Predicción para [0.1, 0.1, 0.1, 0.1]:", forestSequencial.PredictSequencial([]float64{0.1, 0.1, 0.1, 0.1}))
    //    fmt.Println("Predicción para [0.9, 0.9, 0.9, 0.9]:", forestSequencial.PredictSequencial([]float64{0.9, 0.9, 0.9, 0.9}))
    //})  
    //Medir el tiempo de ejecución para ForestConcurrent
    //utils.MeasureExecutionTime("ForestConcurrent", func() {
    //    forestConcurrent := randomforest.ForestConcurrent{}
    //    forestConcurrent.Data = randomforest.ForestDataConcurrent{X: xData, Class: yData}
    //    forestConcurrent.TrainConcurrent(1000)
    //    fmt.Println("Vote", forestConcurrent.Vote([]float64{0.1, 0.1, 0.1, 0.1}))
    //    fmt.Println("Vote", forestConcurrent.Vote([]float64{0.9, 0.9, 0.9, 0.9}))
    //    fmt.Println("Predicción para [0.1, 0.1, 0.1, 0.1]:", forestConcurrent.PredictConcurrent([]float64{0.1, 0.1, 0.1, 0.1}))
    //    fmt.Println("Predicción para [0.9, 0.9, 0.9, 0.9]:", forestConcurrent.PredictConcurrent([]float64{0.9, 0.9, 0.9, 0.9}))
    //})

    //--------------------------------------------------------------SVM
    //xDataSVM := [][]float64{}
    //yDataSVM := []float64{}
//
    //for i := 0; i < 1000; i++ {
	//    x := []float64{rand.Float64(), rand.Float64(), rand.Float64(), rand.Float64()}
	//    y := float64(1)
    //    if int(x[0]+x[1]+x[2]+x[3])%2 != 0 {
	//		y = -1
	//	}
	//    xDataSVM = append(xDataSVM, x)
	//    yDataSVM = append(yDataSVM, y)
    //}
    //// Medir el tiempo de ejecución para SVMConcurrent
    //measureExecutionTime("SVMConcurrent", func() {
    //    svmConcurrent := svm.SVMConcurrent(2, 1000, 0.001)
    //    svmConcurrent.TrainConcurrent(xDataSVM, yDataSVM)
    //    fmt.Println("Predicción para [1.0, 2.0]:", svmConcurrent.PredictConcurrent([]float64{2.0, 2.0}))
    //})
    // measureExecutionTime("SVMSequencial", func() {
    //    svmSequencial := svm.SVMSequencial(2, 1000, 0.001)
    //    svmSequencial.TrainSequencial(xDataSVM, yDataSVM)
    //    fmt.Println("Predicción para [1.0, 2.0]:", svmSequencial.PredictSequencial([]float64{1.0, 2.0}))
    //})
    //--------------------------------------------------------------RNN
    //inputs  := [][]float64{}
	//targets := [][]float64{}
	//for i := 0; i < 10000; i++ {
	//	x := []float64{rand.Float64(), rand.Float64(), rand.Float64(), rand.Float64()}
    //    y := []float64{1.0}
    //    if x[0]+x[1]+x[2]+x[3] < 1 {
    //        y = []float64{-1.0}
    //    }
	//	inputs = append(inputs, x)
	//	targets = append(targets, y)
	//}
    //inputSize := 4
	//hiddenSize := 2
	//outputSize := 1
	//learningRate := 0.5
	//epochs := 1000
    //testCases := [][]float64{
	//	{0.1, 0.2, 0.3, 0.4},
	//	{0.9, 0.8, 0.7, 0.6},
	//}
    //measureExecutionTime("RNA", func() {
    //    nn := dnn.NewNeuralNetworkSequencial(inputSize, hiddenSize, outputSize, learningRate)
    //    nn.TrainSequencial(inputs, targets, epochs)
    //    // Probar la red neuronal después del entrenamiento
	//	for _, input := range testCases {
	//		output := nn.ForwardSequencial(input)
	//		fmt.Printf("Input: %v, Predicted Output: %v\n", input, output)
	//	}
    //})
    //measureExecutionTime("RNA", func() {
    //    nn := dnn.NewNeuralNetworkConcurrent(inputSize, hiddenSize, outputSize, learningRate)
    //    nn.TrainConcurrent(inputs, targets, epochs)
    //    // Probar la red neuronal después del entrenamiento
    //    for _, input := range testCases {
    //        output := nn.ForwardConcurrent(input)
    //        fmt.Printf("Input: %v, Predicted Output: %v\n", input, output)
    //    }
    //})
    //--------------------------------------------------------------Filtrado colaborativo
    //numUsers := 1000    // Número de usuarios
	//numItems := 100     // Número de elementos (reducido para aumentar probabilidad de coincidencias)
	//maxRating := 5.0    // Calificación máxima
	//minRating := 1.0    // Calificación mínima
    //// Seleccionar un usuario aleatorio para generar recomendaciones
	//user := fmt.Sprintf("User%d", rand.Intn(numUsers))
	//k := 10 // Número de recomendaciones
    //ratings1 := fc.NewRatingsSequencial()
    //ratings2 := fc.NewRatingsConcurrent()
	//// Añadir calificaciones aleatorias para cada usuario y cada elemento
	//for u := 0; u < numUsers; u++ {
	//	user := fmt.Sprintf("User%d", u)
	//	numRatings := rand.Intn(numItems/2) + numItems/2 // Cada usuario califica al menos la mitad de los items
	//	for i := 0; i < numRatings; i++ {
	//		item := fmt.Sprintf("Item%d", rand.Intn(numItems))
	//		rating := minRating + rand.Float64()*(maxRating-minRating)
	//		ratings1.AddRatingSequencial(user, item, rating)
    //        ratings2.AddRatingConcurrent(user, item, rating)
	//	}
	//}
    //measureExecutionTime("FC", func() {
	//	recommendations := fc.RecommendSequencial(ratings1, user, k)
	//	fmt.Printf("Recomendaciones para %s: %v\n", user, recommendations)
	//})
    //measureExecutionTime("FC", func() {
	//	recommendations := fc.RecommendConcurrent(ratings2, user, k)
	//	fmt.Printf("Recomendaciones para %s: %v\n", user, recommendations)
	//})
}