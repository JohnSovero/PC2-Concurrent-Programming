package utils

import (
	"encoding/csv"
	"fmt"
	"os"
	"time"
	"math/rand"
	"strconv"
	"PC2/algorithms/dnn"
)

// LoadDataset carga el archivo CSV y lo convierte en un DataFrame
func LoadDataset(path string) [][] string {
	file, err := os.Open(path)
	if err != nil {
		fmt.Println("Error al abrir el archivo:", err)
		return nil
	}
	defer file.Close()
	reader := csv.NewReader(file)

	records, err := reader.ReadAll()
	if err != nil {
		fmt.Println("Error al leer el archivo:", err)
		return nil
	}
	
	if len(records) == 0 {
		fmt.Println("Archivo CSV vacío")
		return nil
	}
	return records[1:]
}

// MeasureExecutionTime mide el tiempo de ejecución de una función
func MeasureExecutionTime(name string, f func()) {
    start := time.Now()
    f()
    duration := time.Since(start)
    fmt.Printf("Tiempo de ejecución para %s: %v\n", name, duration)
}

// Función para convertir de [][]string a [][]float64
func ConvertToFloat64(data [][]string) ([][]float64, error) {
	floatData := make([][]float64, len(data))
	for i, row := range data {
		floatRow := make([]float64, len(row))
		for j, val := range row {
			floatVal, err := strconv.ParseFloat(val, 64)
			if err != nil {
				return nil, fmt.Errorf("error al convertir '%s' a float64: %w", val, err)
			}
			floatRow[j] = floatVal
		}
		floatData[i] = floatRow
	}
	return floatData, nil
}
// Función para convertir []int a []float64
func ConvertIntToFloat64(data []int) []float64 {
	floatData := make([]float64, len(data))
	for i, val := range data {
		floatData[i] = float64(val)
	}
	return floatData
}

// Función para intercalar dos listas dentro de una lista principal
func Interleave(slice [][]float64, len1, len2 int) {
	rand.Seed(time.Now().UnixNano())
	
	// Crear una nueva lista para almacenar el resultado
	interleaved := make([][]float64, 0, len1+len2)
	
	// Punteros para las dos listas
	i, j := 0, len1
	
	for i < len1 || j < len1+len2 {
		if i < len1 {
			interleaved = append(interleaved, slice[i])
			i++
		}
		if j < len1+len2 {
			interleaved = append(interleaved, slice[j])
			j++
		}
	}
	
	// Copiar los elementos intercalados de vuelta a la lista original
	copy(slice, interleaved)
}

// TrainTestSplit divides the data into training and testing sets
func TrainTestSplit(xData [][]float64, yData []int, testSize float64) (trainX [][]float64, trainY []float64, testX [][]float64, testY []float64) {
	// Seed the random number generator to ensure reproducibility
	rand.Seed(time.Now().UnixNano())

	// Calculate the number of test samples
	totalSamples := len(xData)
	numTestSamples := int(testSize * float64(totalSamples))

	// Generate a list of indices and shuffle them
	indices := rand.Perm(totalSamples)

	// Split the indices into training and testing indices
	testIndices := indices[:numTestSamples]
	trainIndices := indices[numTestSamples:]

	// Initialize slices for the output data
	trainX = make([][]float64, len(trainIndices))
	trainY = make([]float64, len(trainIndices))
	testX = make([][]float64, len(testIndices))
	testY = make([]float64, len(testIndices))

	// Fill the training data
	for i, idx := range trainIndices {
		trainX[i] = xData[idx]
		trainY[i] = float64(yData[idx])
	}

	// Fill the testing data
	for i, idx := range testIndices {
		testX[i] = xData[idx]
		testY[i] = float64(yData[idx])
	}

	return
}

// TrainTestSplit divide los datos en conjuntos de entrenamiento y prueba
func TrainTestSplit2(xData [][]float64, yData []int, testSize float64) (trainX [][]float64, trainY []int, testX [][]float64, testY []int) {
    // Seed the random number generator to ensure reproducibility
    rand.Seed(time.Now().UnixNano())

    // Calculate the number of test samples
    totalSamples := len(xData)
    numTestSamples := int(testSize * float64(totalSamples))

    // Generate a list of indices and shuffle them
    indices := rand.Perm(totalSamples)

    // Split the indices into training and testing indices
    testIndices := indices[:numTestSamples]
    trainIndices := indices[numTestSamples:]

    // Initialize slices for the output data
    trainX = make([][]float64, len(trainIndices))
    trainY = make([]int, len(trainIndices))
    testX = make([][]float64, len(testIndices))
    testY = make([]int, len(testIndices))

    // Fill the training data
    for i, idx := range trainIndices {
        trainX[i] = xData[idx]
        trainY[i] = yData[idx]
    }

    // Fill the testing data
    for i, idx := range testIndices {
        testX[i] = xData[idx]
        testY[i] = yData[idx]
    }

    return
}

// Procesa los datos del dataset y los separa en xData y yData
func ProcessData(df [][]string) ([][]float64, []int, error) {
    var xData [][]float64
    var yData []int

    for _, row := range df {
        var xRow []float64
        for i, val := range row {
            if i == len(row)-1 {
                yValue, err := strconv.Atoi(val)
                if err != nil {
                    return nil, nil, fmt.Errorf("error al convertir la etiqueta a int: %v", err)
                }
                yData = append(yData, yValue)
            } else {
                xValue, err := strconv.ParseFloat(val, 64)
                if err != nil {
                    return nil, nil, fmt.Errorf("error al convertir el valor a float64: %v", err)
                }
                xRow = append(xRow, xValue)
            }
        }
        xData = append(xData, xRow)
    }

    return xData, yData, nil
}

func ConvertToDNNFrames(trainX [][]float64, trainY []int, testX [][]float64, testY []int) (dnn.Frame, dnn.Frame, dnn.Frame, dnn.Frame) {
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

    return trainXFrame, trainYFrame, testXFrame, testYFrame
}
