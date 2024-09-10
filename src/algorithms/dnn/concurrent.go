package dnn

import (
	"math/rand"
	"sync"
	"time"
)

// NeuronLayerConcurrent representa una capa de neuronas
type NeuronLayerConcurrent struct {
	Weights [][]float64
	Biases  []float64
	Outputs []float64
}

// NeuralNetworkConcurrent representa la red neuronal
type NeuralNetworkConcurrent struct {
	InputLayer  NeuronLayerConcurrent
	HiddenLayer NeuronLayerConcurrent
	OutputLayer NeuronLayerConcurrent
	LearningRate float64
	mu sync.Mutex
}

// NewNeuronLayerConcurrent inicializa una capa de neuronas con pesos y sesgos aleatorios
func NewNeuronLayerConcurrent(inputSize, outputSize int) NeuronLayerConcurrent {
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
	return NeuronLayerConcurrent{
		Weights: weights,
		Biases:  biases,
		Outputs: make([]float64, outputSize),
	}
}

// NewNeuralNetworkConcurrent inicializa una red neuronal con una capa de entrada, una oculta y una de salida
func NewNeuralNetworkConcurrent(inputSize, hiddenSize, outputSize int, learningRate float64) NeuralNetworkConcurrent {
	return NeuralNetworkConcurrent{
		InputLayer:  NewNeuronLayerConcurrent(inputSize, hiddenSize),
		HiddenLayer: NewNeuronLayerConcurrent(hiddenSize, outputSize),
		OutputLayer: NewNeuronLayerConcurrent(hiddenSize, outputSize),
		LearningRate: learningRate,
	}
}

// ForwardConcurrent realiza la propagación hacia adelante concurrentemente
func (nn *NeuralNetworkConcurrent) ForwardConcurrent(input []float64) []float64 {
	nn.InputLayer.Outputs = input

	// Capa oculta
	var wg sync.WaitGroup
	outputsHidden := make([]float64, len(nn.HiddenLayer.Outputs))
	for j := range outputsHidden {
		wg.Add(1)
		go func(j int) {
			defer wg.Done()
			sum := nn.HiddenLayer.Biases[j]
			for i := range nn.InputLayer.Outputs {
				sum += nn.InputLayer.Outputs[i] * nn.InputLayer.Weights[i][j]
			}
			outputsHidden[j] = Sigmoid(sum)
		}(j)
	}
	wg.Wait()
	nn.mu.Lock()
	nn.HiddenLayer.Outputs = outputsHidden
	nn.mu.Unlock()

	// Capa de salida
	outputsOutput := make([]float64, len(nn.OutputLayer.Outputs))
	for j := range outputsOutput {
		wg.Add(1)
		go func(j int) {
			defer wg.Done()
			sum := nn.OutputLayer.Biases[j]
			for i := range nn.HiddenLayer.Outputs {
				sum += nn.HiddenLayer.Outputs[i] * nn.HiddenLayer.Weights[i][j]
			}
			outputsOutput[j] = Sigmoid(sum)
		}(j)
	}
	wg.Wait()
	nn.mu.Lock()
	nn.OutputLayer.Outputs = outputsOutput
	nn.mu.Unlock()

	return nn.OutputLayer.Outputs
}

// Backward realiza la propagación hacia atrás y actualiza los pesos y sesgos concurrentemente
func (nn *NeuralNetworkConcurrent) Backward(input, target []float64) {
	outputDeltas := make([]float64, len(nn.OutputLayer.Outputs))
	for i := range outputDeltas {
		error := target[i] - nn.OutputLayer.Outputs[i]
		outputDeltas[i] = error * SigmoidDerivative(nn.OutputLayer.Outputs[i])
	}

	hiddenDeltas := make([]float64, len(nn.HiddenLayer.Outputs))
	for i := range hiddenDeltas {
		error := 0.0
		for j := range outputDeltas {
			error += outputDeltas[j] * nn.HiddenLayer.Weights[i][j]
		}
		hiddenDeltas[i] = error * SigmoidDerivative(nn.HiddenLayer.Outputs[i])
	}

	var wg sync.WaitGroup
	wg.Add(2)

	go func() {
		defer wg.Done()
		for i := range nn.HiddenLayer.Outputs {
			for j := range nn.OutputLayer.Outputs {
				nn.mu.Lock()
				nn.HiddenLayer.Weights[i][j] += nn.LearningRate * outputDeltas[j] * nn.HiddenLayer.Outputs[i]
				nn.mu.Unlock()
			}
		}
		for j := range nn.OutputLayer.Biases {
			nn.mu.Lock()
			nn.OutputLayer.Biases[j] += nn.LearningRate * outputDeltas[j]
			nn.mu.Unlock()
		}
	}()

	go func() {
		defer wg.Done()
		for i := range nn.InputLayer.Outputs {
			for j := range nn.HiddenLayer.Outputs {
				nn.mu.Lock()
				nn.InputLayer.Weights[i][j] += nn.LearningRate * hiddenDeltas[j] * nn.InputLayer.Outputs[i]
				nn.mu.Unlock()
			}
		}
		for j := range nn.HiddenLayer.Biases {
			nn.mu.Lock()
			nn.HiddenLayer.Biases[j] += nn.LearningRate * hiddenDeltas[j]
			nn.mu.Unlock()
		}
	}()

	wg.Wait()
}

// TrainConcurrent entrena la red neuronal con los datos de entrada y las etiquetas esperadas concurrentemente
func (nn *NeuralNetworkConcurrent) TrainConcurrent(inputs, targets [][]float64, epochs int) {
	for epoch := 0; epoch < epochs; epoch++ {
		var wg sync.WaitGroup
		for i := range inputs {
			wg.Add(1)
			go func(i int) {
				defer wg.Done()
				nn.ForwardConcurrent(inputs[i])
				nn.Backward(inputs[i], targets[i])
			}(i)
		}
		wg.Wait()
	}
}