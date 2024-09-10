package dnn

import "math"

// Sigmoid calcula la función de activación sigmoide
func Sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// SigmoidDerivative calcula la derivada de la función de activación sigmoide
func SigmoidDerivative(x float64) float64 {
	return x * (1.0 - x)
}