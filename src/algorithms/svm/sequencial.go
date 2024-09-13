package svm

import (
	"fmt"
)

// SVMS represents a simple linear SVMS model
type SVMS struct {
	Weights      []float64
	Bias         float64
	LearningRate float64
	Iterations   int
}

// SVMSequencial creates a new SVMS model with given parameters
func SVMSequencial(learningRate float64, iterations int) *SVMS {
	return &SVMS{
		LearningRate: learningRate,
		Iterations:   iterations,
	}
}

// TrainSequencial trains the SVMS model using gradient descent
func (s *SVMS) TrainSequencial(X [][]float64, Y []float64) {
	numSamples := len(X)
	numFeatures := len(X[0])
	s.Weights = make([]float64, numFeatures)

	for i := 0; i < s.Iterations; i++ {
		for j := 0; j < numSamples; j++ {
			dot := s.dotProductSequencial(s.Weights, X[j]) + s.Bias
			if Y[j]*dot <= 1 {
				// Update weights and bias using the hinge loss gradient
				for k := 0; k < numFeatures; k++ {
					s.Weights[k] += s.LearningRate * (Y[j]*X[j][k] - 2*0.01*s.Weights[k])
				}
				s.Bias += s.LearningRate * Y[j]
			} else {
				// Update weights with regularization term
				for k := 0; k < numFeatures; k++ {
					s.Weights[k] -= s.LearningRate * 2 * 0.01 * s.Weights[k]
				}
			}
		}
	}
}

// PredictSequencial predicts the class for given input data
func (s *SVMS) PredictSequencial(X [][]float64) []float64 {
	numSamples := len(X)
	predictions := make([]float64, numSamples)

	for i := 0; i < numSamples; i++ {
		dot := s.dotProductSequencial(s.Weights, X[i]) + s.Bias
		if dot >= 0 {
			predictions[i] = 1
		} else {
			predictions[i] = -1
		}
	}

	return predictions
}

// dotProductSequencial calculates the dot product of two vectors
func (s *SVMS) dotProductSequencial(vec1, vec2 []float64) float64 {
	sum := 0.0
	for i := 0; i < len(vec1); i++ {
		sum += vec1[i] * vec2[i]
	}
	return sum
}

// Accuracy calculates the accuracy of the model
func (s *SVMS) AccuracySequencial(predictions, labels []float64) float64 {
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