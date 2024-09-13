package svm

import (
    "fmt"
    "sync"
)

// SVMC represents a simple linear SVMC model
type SVMC struct {
    Weights      []float64
    Bias         float64
    LearningRate float64
    Iterations   int
}

// SVMConcurrent creates a new SVMC model with given parameters
func SVMConcurrent(learningRate float64, iterations int) *SVMC {
    return &SVMC{
        LearningRate: learningRate,
        Iterations:   iterations,
    }
}

// TrainConcurrent trains the SVM model using goroutines
func (s *SVMC) TrainConcurrent(X [][]float64, Y []float64) {
	numSamples := len(X)
	numFeatures := len(X[0])
	s.Weights = make([]float64, numFeatures)

	var wg sync.WaitGroup
	mutex := &sync.Mutex{}

	weightUpdates := make([][]float64, numSamples)
	biasUpdates := make([]float64, numSamples)

	for i := 0; i < s.Iterations; i++ {
		for j := range weightUpdates {
			weightUpdates[j] = make([]float64, numFeatures)
		}

		for j := 0; j < numSamples; j++ {
			wg.Add(1)
			go func(index int) {
				defer wg.Done()
				dot := s.dotProduct(s.Weights, X[index])
				if Y[index]*dot <= 0 {
					// Calculate updates
					weightChange := make([]float64, numFeatures)
					for k := 0; k < numFeatures; k++ {
						weightChange[k] = s.LearningRate * Y[index] * X[index][k]
					}
					biasChange := s.LearningRate * Y[index]
					mutex.Lock()
					weightUpdates[index] = weightChange
					biasUpdates[index] = biasChange
					mutex.Unlock()
				}
			}(j)
		}
		wg.Wait()
		mutex.Lock()
		for j := 0; j < numSamples; j++ {
			if weightUpdates[j] != nil {
				for k := 0; k < numFeatures; k++ {
					s.Weights[k] += weightUpdates[j][k]
				}
				s.Bias += biasUpdates[j]
			}
		}
		mutex.Unlock()
	}
}

// PredictConcurrent predicts the class for given input data
func (s *SVMC) PredictConcurrent(X [][]float64) []float64 {
    numSamples := len(X)
    predictions := make([]float64, numSamples)

    for i := 0; i < numSamples; i++ {
        dot := s.dotProduct(s.Weights, X[i]) + s.Bias
        if dot >= 0 {
            predictions[i] = 1
        } else {
            predictions[i] = -1
        }
    }

    return predictions
}

// dotProduct calculates the dot product of two vectors
func (s *SVMC) dotProduct(vec1, vec2 []float64) float64 {
    sum := 0.0
    for i := 0; i < len(vec1); i++ {
        sum += vec1[i] * vec2[i]
    }
    return sum
}

// Accuracy calculates the accuracy of the model
func (s *SVMC) AccuracyConcurrent(predictions, labels []float64) float64 {
    if len(predictions) != len(labels) {
        fmt.Println("Error: the length of predictions and labels must be the same")
        return 0.0
    }

    correct := 0
    total := len(labels)

    for i := 0; i < total; i++ {
        if predictions[i] == labels[i] {
            correct++
        }
    }

    return float64(correct) / float64(total)
}