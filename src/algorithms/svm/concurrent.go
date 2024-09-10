package svm

import (
	"sync"
)

type SVMC struct {
	weights []float64
	bias    float64
	epochs  int
	alpha   float64
}

func SVMConcurrent(dim int, epochs int, alpha float64) *SVMC {
	return &SVMC{
		weights: make([]float64, dim),
		bias:    0.0,
		epochs:  epochs,
		alpha:   alpha,
	}
}

func (svm *SVMC) TrainConcurrent(data [][]float64, labels []float64) {
	n := len(data)

	var wg sync.WaitGroup

	for epoch := 0; epoch < svm.epochs; epoch++ {
		wg.Add(n)
		for i := 0; i < n; i++ {
			go func(i int) {
				defer wg.Done()
				svm.updateWeights(data[i], labels[i])
			}(i)
		}
		wg.Wait()
	}
}

func (svm *SVMC) updateWeights(xi []float64, yi float64) {
	output := svm.predictRaw(xi)
	if yi*output < 1 {
		for j := 0; j < len(svm.weights); j++ {
			svm.weights[j] += svm.alpha * (yi*xi[j] - 2*(1/float64(len(xi)))*svm.weights[j])
		}
		svm.bias += svm.alpha * yi
	}
}

func (svm *SVMC) predictRaw(xi []float64) float64 {
	output := svm.bias
	for j := 0; j < len(svm.weights); j++ {
		output += svm.weights[j] * xi[j]
	}
	return output
}

func (svm *SVMC) PredictConcurrent(xi []float64) float64 {
	if svm.predictRaw(xi) >= 0 {
		return 1
	}
	return -1
}