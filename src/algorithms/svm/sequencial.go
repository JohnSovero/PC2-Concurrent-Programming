package svm

type SVMS struct {
	weights []float64
	bias    float64
	epochs  int
	alpha   float64
}

func SVMSequencial(dim int, epochs int, alpha float64) *SVMS {
	return &SVMS{
		weights: make([]float64, dim),
		bias:    0.0,
		epochs:  epochs,
		alpha:   alpha,
	}
}

func (svm *SVMS) TrainSequencial(data [][]float64, labels []float64) {
	n := len(data)

	for epoch := 0; epoch < svm.epochs; epoch++ {
		for i := 0; i < n; i++ {
			svm.updateWeightsSequencial(data[i], labels[i])
		}
	}
}

func (svm *SVMS) updateWeightsSequencial(xi []float64, yi float64) {
	output := svm.predictRawSequencial(xi)
	if yi*output < 1 {
		for j := 0; j < len(svm.weights); j++ {
			svm.weights[j] += svm.alpha * (yi*xi[j] - 2*(1/float64(len(xi)))*svm.weights[j])
		}
		svm.bias += svm.alpha * yi
	}
}

func (svm *SVMS) predictRawSequencial(xi []float64) float64 {
	output := svm.bias
	for j := 0; j < len(svm.weights); j++ {
		output += svm.weights[j] * xi[j]
	}
	return output
}

func (svm *SVMS) PredictSequencial(xi []float64) float64 {
	if svm.predictRawSequencial(xi) >= 0 {
		return 1
	}
	return -1
}