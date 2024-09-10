package dnn

import (
	"math/rand"
	"time"
)

// NeuronLayerSequencial representa una capa de neuronas
type NeuronLayerSequencial struct {
	Weights [][]float64
	Biases  []float64
	Outputs []float64
}

// NeuralNetworkSequencial representa la red neuronal
type NeuralNetworkSequencial struct {
	InputLayer  NeuronLayerSequencial
	HiddenLayer NeuronLayerSequencial
	OutputLayer NeuronLayerSequencial
	LearningRate float64
}

// NewNeuronLayer inicializa una capa de neuronas con pesos y sesgos aleatorios
func NewNeuronLayer(inputSize, outputSize int) NeuronLayerSequencial {
	rand.Seed(time.Now().UnixNano())
	weights := make([][]float64, inputSize)
	for i := range weights {
		weights[i] = make([]float64, outputSize)
		for j := range weights[i] {
			weights[i][j] = rand.Float64()
		}
	}
	biases := make([]float64, outputSize)
	for i := range biases {
		biases[i] = rand.Float64()
	}
	return NeuronLayerSequencial{
		Weights: weights,
		Biases:  biases,
		Outputs: make([]float64, outputSize),
	}
}

// NewNeuralNetworkSequencial inicializa una red neuronal con una capa de entrada, una oculta y una de salida
func NewNeuralNetworkSequencial(inputSize, hiddenSize, outputSize int, learningRate float64) NeuralNetworkSequencial {
	return NeuralNetworkSequencial{
		InputLayer:  NewNeuronLayer(inputSize, hiddenSize),
		HiddenLayer: NewNeuronLayer(hiddenSize, outputSize),
		OutputLayer: NewNeuronLayer(hiddenSize, outputSize),
		LearningRate: learningRate,
	}
}

// ForwardSequencial realiza la propagación hacia adelante
func (nn *NeuralNetworkSequencial) ForwardSequencial(input []float64) []float64 {
	nn.InputLayer.Outputs = input

	// Capa oculta
	for j := range nn.HiddenLayer.Outputs {
		sum := nn.HiddenLayer.Biases[j]
		for i := range nn.InputLayer.Outputs {
			sum += nn.InputLayer.Outputs[i] * nn.InputLayer.Weights[i][j]
		}
		nn.HiddenLayer.Outputs[j] = Sigmoid(sum)
	}

	// Capa de salida
	for j := range nn.OutputLayer.Outputs {
		sum := nn.OutputLayer.Biases[j]
		for i := range nn.HiddenLayer.Outputs {
			sum += nn.HiddenLayer.Outputs[i] * nn.HiddenLayer.Weights[i][j]
		}
		nn.OutputLayer.Outputs[j] = Sigmoid(sum)
	}

	return nn.OutputLayer.Outputs
}

// Backward realiza la propagación hacia atrás y actualiza los pesos y sesgos
func (nn *NeuralNetworkSequencial) Backward(input, target []float64) {
	// Cálculo del error de salida
	outputDeltas := make([]float64, len(nn.OutputLayer.Outputs))
	for i := range outputDeltas {
		error := target[i] - nn.OutputLayer.Outputs[i]
		outputDeltas[i] = error * SigmoidDerivative(nn.OutputLayer.Outputs[i])
	}

	// Cálculo del error de la capa oculta
	hiddenDeltas := make([]float64, len(nn.HiddenLayer.Outputs))
	for i := range hiddenDeltas {
		error := 0.0
		for j := range outputDeltas {
			error += outputDeltas[j] * nn.HiddenLayer.Weights[i][j]
		}
		hiddenDeltas[i] = error * SigmoidDerivative(nn.HiddenLayer.Outputs[i])
	}

	// Actualización de los pesos y sesgos de la capa de salida
	for i := range nn.HiddenLayer.Outputs {
		for j := range nn.OutputLayer.Outputs {
			nn.HiddenLayer.Weights[i][j] += nn.LearningRate * outputDeltas[j] * nn.HiddenLayer.Outputs[i]
		}
	}
	for j := range nn.OutputLayer.Biases {
		nn.OutputLayer.Biases[j] += nn.LearningRate * outputDeltas[j]
	}

	// Actualización de los pesos y sesgos de la capa oculta
	for i := range nn.InputLayer.Outputs {
		for j := range nn.HiddenLayer.Outputs {
			nn.InputLayer.Weights[i][j] += nn.LearningRate * hiddenDeltas[j] * nn.InputLayer.Outputs[i]
		}
	}
	for j := range nn.HiddenLayer.Biases {
		nn.HiddenLayer.Biases[j] += nn.LearningRate * hiddenDeltas[j]
	}
}

// TrainSequencial entrena la red neuronal con los datos de entrada y las etiquetas esperadas
func (nn *NeuralNetworkSequencial) TrainSequencial(inputs, targets [][]float64, epochs int) {
	for epoch := 0; epoch < epochs; epoch++ {
		for i := range inputs {
			nn.ForwardSequencial(inputs[i])
			nn.Backward(inputs[i], targets[i])
		}
	}
}