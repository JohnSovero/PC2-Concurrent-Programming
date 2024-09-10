package randomforest

import ("sort")

func CalculateMean(data []float64) float64 {
	sum := 0.0
	for _, value := range data {
		sum += value
	}
	return sum / float64(len(data))
}

func CalculateVariance(data []float64, mean float64) float64 {
	sum := 0.0
	for _, value := range data {
		diff := value - mean
		sum += diff * diff
	}
	return sum / float64(len(data))
}

// calculateQuantile calcula el cuantil q de un conjunto de datos
func CalculateQuantile(data []float64, q float64) float64 {
	if q < 0 || q > 1 {
		panic("q debe estar entre 0 y 1")
	}

	n := len(data)
	if n == 0 {
		return 0 // o panico, dependiendo de cómo quieras manejar el caso de lista vacía
	}

	// Ordenar el conjunto de datos
	sort.Float64s(data)

	// Calcular la posición del cuantil
	pos := q * float64(n-1)
	lowerIndex := int(pos)
	upperIndex := lowerIndex + 1
	weight := pos - float64(lowerIndex)

	// Interpolación lineal si es necesario
	if upperIndex < n {
		return data[lowerIndex]*(1-weight) + data[upperIndex]*weight
	}
	return data[lowerIndex]
}