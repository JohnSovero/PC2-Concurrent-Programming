package main

import (
	"fmt"
	"PC2/algorithms/randomForest"
	"PC2/algorithms/svm"
	"PC2/algorithms/dnn"
	"PC2/algorithms/fc"
	"PC2/utils"
	"math/rand"
	"strconv"
)

func main() {
	//Leer archivos
	df_ratings := utils.LoadDataset("datasets/ratings.csv")	
    df_higgs := utils.LoadDataset("datasets/Higgs.csv")

	//Procesar datos para conseguir xData y yData
	xData, yData, error := utils.ProcessData(df_higgs)
	if error != nil {
		fmt.Println("Error al procesar los datos:", error)
		return
	}

	//Separar datos de entrenamiento y prueba para Random Forest continuo
	trainXRF, trainYRF, testXRF, testYRF := utils.TrainTestSplit2(xData, yData, 0.2)

    //--------------------------------------------------------------Random Forest--------------------------------------------------------------
    utils.MeasureExecutionTime("RandomForestSequencial", func() {
		rfSequencial := randomForest.ForestSequencial{}
		rfSequencial.Data = randomForest.ForestDataSequencial{X: trainXRF, Class: trainYRF}
		rfSequencial.TrainSequecial(1)
        predictions := rfSequencial.PredictSequencial(testXRF)
        accuracy := rfSequencial.Accuracy(predictions, testYRF)
        fmt.Printf("Precisión: %.2f%%\n", accuracy * 100)
    })
    utils.MeasureExecutionTime("ForestConcurrent", func() {
		rfConcurrent := randomForest.ForestConcurrent{}
		rfConcurrent.Data = randomForest.ForestDataConcurrent{X: trainXRF, Class: trainYRF}
        rfConcurrent.TrainConcurrent(1)
        predictions := rfConcurrent.PredictConcurrent(testXRF)
        accuracy := rfConcurrent.Accuracy(predictions, testYRF)
        fmt.Printf("Precisión: %.2f%%\n", accuracy * 100)
    })
    //--------------------------------------------------------------SVM--------------------------------------------------------------
    //Separar datos de entrenamiento y prueba para SVM y DNN
	trainX, trainY, testX, testY := utils.TrainTestSplit(xData, yData, 0.2)
	
	//Parámetros de entrenamiento
	epochs := 10
	lr := 0.001

    utils.MeasureExecutionTime("SVMSequencial", func() {
        svmSequencial := svm.SVMSequencial(lr, epochs)
        svmSequencial.TrainSequencial(trainX, trainY)
        predictions := svmSequencial.PredictSequencial(testX)
        accuracy := svmSequencial.AccuracySequencial(predictions, testY)
        fmt.Printf("Accuracy: %.2f%%\n", accuracy * 100)
    })
    utils.MeasureExecutionTime("SVMConcurrent", func() {
        svmConcurrent := svm.SVMConcurrent(lr, epochs)
        svmConcurrent.TrainConcurrent(trainX, trainY)
        predictions := svmConcurrent.PredictConcurrent(testX)
        accuracy := svmConcurrent.AccuracyConcurrent(predictions, testY)
        fmt.Printf("Accuracy: %.2f%%\n", accuracy * 100)
    })

    //--------------------------------------------------------------RNN--------------------------------------------------------------
    trainXFrame := make(dnn.Frame, len(trainX))
    for i, row := range trainX {
        vector := make(dnn.Vector, len(row))
        for j, val := range row {
            vector[j] = float32(val)
        }
        trainXFrame[i] = vector
    }
    trainYFrame := make(dnn.Frame, len(trainY))
    for i, val := range trainY {
        vector := make(dnn.Vector, 1)
        vector[0] = float32(val)
        trainYFrame[i] = vector
    }
    testXFrame := make(dnn.Frame, len(testX))
    for i, row := range testX {
        vector := make(dnn.Vector, len(row))
        for j, val := range row {
            vector[j] = float32(val)
        }
        testXFrame[i] = vector
    }
    testYFrame := make(dnn.Frame, len(testY))
    for i, val := range testY {
        vector := make(dnn.Vector, 1)
        vector[0] = float32(val)
        testYFrame[i] = vector
    }

	//Parámetros de entrenamiento
	inputSize := len(trainX[0])
	outputSize := 1

	utils.MeasureExecutionTime("DNN Secuencial", func() {
		nn := &dnn.MLPSequencial{
			Layers: []*dnn.LayerSequencial{
				{Name: "Input Layer", Width: inputSize},
				{Name: "Hidden Layer", Width: 10, ActivationFunction: dnn.Sigmoid, ActivationFunctionDeriv: dnn.SigmoidDerivative},
				{Name: "Output Layer", Width: outputSize, ActivationFunction: dnn.Sigmoid, ActivationFunctionDeriv: dnn.SigmoidDerivative},
			},
			LearningRate: 0.1,
			Introspect: func(step dnn.StepSequencial) {
				fmt.Printf("Epoch: %d, Loss: %f\n", step.Epoch, step.LossSequencial)
			},
		}
		loss, err := nn.TrainSequencial(epochs, trainXFrame, trainYFrame)
		if err != nil {
			fmt.Println("Error durante el entrenamiento:", err)
			return
		}
		fmt.Printf("Entrenamiento completado con pérdida final: %f\n", loss)
		predictions := nn.PredictSequencial(testXFrame)
		accuracy := dnn.CalculateAccuracy(predictions, testY)
		fmt.Printf("Precisión: %.2f%%\n", accuracy * 100)
	})
	
	utils.MeasureExecutionTime("DNN Concurrente", func() {
		nn := &dnn.MLPConcurrent{
			Layers: []*dnn.LayerConcurrent{
				{Name: "Input Layer", Width: inputSize},
				{Name: "Hidden Layer", Width: 10, ActivationFunction: dnn.Sigmoid, ActivationFunctionDeriv: dnn.SigmoidDerivative},
				{Name: "Output Layer", Width: outputSize, ActivationFunction: dnn.Sigmoid, ActivationFunctionDeriv: dnn.SigmoidDerivative},
			},
			LearningRate: 0.1,
			Introspect: func(step dnn.StepConcurrent) {
				fmt.Printf("Epoch: %d, Loss: %f\n", step.Epoch, step.LossConcurrent)
			},
		}
		loss, err := nn.TrainConcurrent(epochs, trainXFrame, trainYFrame)
		if err != nil {
			fmt.Println("Error durante el entrenamiento:", err)
			return
		}
		fmt.Printf("Entrenamiento completado con pérdida final: %f\n", loss)
		predictions := nn.PredictConcurrent(testXFrame)
		accuracy := dnn.CalculateAccuracy(predictions, testY)
		fmt.Printf("Precisión: %.2f%%\n", accuracy * 100)
	})

    //--------------------------------------------------------------Filtrado colaborativo
    ratings1 := fc.NewRatingsSequencial()
    ratings2 := fc.NewRatingsConcurrent()

	for _, record := range df_ratings[1:] {
        user := fmt.Sprintf("User%s", record[0])
        item := fmt.Sprintf("Item%s", record[1])
        rating, err := strconv.ParseFloat(record[2], 64)
        if err != nil {
            fmt.Println("Error al convertir la calificación:", err)
            return
        }
        ratings1.AddRatingSequencial(user, item, rating)
        ratings2.AddRatingConcurrent(user, item, rating)
    }

	// Parametros de recomendación
    numUsers := 103170 
    user := fmt.Sprintf("User%d", rand.Intn(numUsers))
    k := 10

	utils.MeasureExecutionTime("FCSequencial", func() {
        recommendations := fc.RecommendSequencial(ratings1, user, k)
        fmt.Printf("Recomendaciones secuenciales para %s: %v\n", user, recommendations)
    })
    utils.MeasureExecutionTime("FCConcurrent", func() {
		recommendations := fc.RecommendConcurrent(ratings2, user, k)
		fmt.Printf("Recomendaciones concurrentes para %s: %v\n", user, recommendations)
	})
}