package randomForest

import (
	"math"
	"math/rand"
	"time"
)

// TreeNode representa un nodo en el árbol de decisión
type TreeNode struct {
	Feature    int
	Threshold  float64
	Left       *TreeNode
	Right      *TreeNode
	Prediction float64
}

// DecisionTree representa un árbol de decisión
type DecisionTree struct {
	MaxDepth int
	Root     *TreeNode
}

// RandomForest representa un bosque aleatorio
type RandomForest struct {
	Trees    []DecisionTree
	NumTrees int
	MaxDepth int
}

// Entrena un árbol de decisión
func (tree *DecisionTree) Train(X [][]float64, Y []float64) {
	tree.Root = tree.buildTree(X, Y, 0)
}

// Construye el árbol recursivamente
func (tree *DecisionTree) buildTree(X [][]float64, Y []float64, depth int) *TreeNode {
	if depth >= tree.MaxDepth || len(Y) <= 1 {
		return &TreeNode{Prediction: mean(Y)}
	}

	feature, threshold, leftX, leftY, rightX, rightY := bestSplit(X, Y)
	if feature == -1 {
		return &TreeNode{Prediction: mean(Y)}
	}

	leftNode := tree.buildTree(leftX, leftY, depth+1)
	rightNode := tree.buildTree(rightX, rightY, depth+1)

	return &TreeNode{
		Feature:   feature,
		Threshold: threshold,
		Left:      leftNode,
		Right:     rightNode,
	}
}

// Predice el valor para un punto de datos
func (tree *DecisionTree) Predict(x []float64) float64 {
	node := tree.Root
	for node.Left != nil && node.Right != nil {
		if x[node.Feature] < node.Threshold {
			node = node.Left
		} else {
			node = node.Right
		}
	}
	return node.Prediction
}

// Entrena el bosque aleatorio
func (forest *RandomForest) Train(X [][]float64, Y []float64) {
	forest.Trees = make([]DecisionTree, forest.NumTrees)
	for i := 0; i < forest.NumTrees; i++ {
		sampleX, sampleY := bootstrapSample(X, Y)
		tree := DecisionTree{MaxDepth: forest.MaxDepth}
		tree.Train(sampleX, sampleY)
		forest.Trees[i] = tree
	}
}

// Predice el valor para un punto de datos usando el bosque aleatorio
func (forest *RandomForest) Predict(x []float64) float64 {
	sum := 0.0
	for _, tree := range forest.Trees {
		sum += tree.Predict(x)
	}
	return sum / float64(forest.NumTrees)
}

// Encuentra la mejor división para un nodo
func bestSplit(X [][]float64, Y []float64) (int, float64, [][]float64, []float64, [][]float64, []float64) {
	bestFeature, bestThreshold := -1, 0.0
	bestGini := math.MaxFloat64
	var leftX, rightX [][]float64
	var leftY, rightY []float64

	for i := 0; i < len(X[0]); i++ {
		thresholds := uniqueValues(getColumn(X, i))
		for _, t := range thresholds {
			lX, lY, rX, rY := split(X, Y, i, t)
			gini := giniImpurity(lY, rY)
			if gini < bestGini {
				bestFeature = i
				bestThreshold = t
				bestGini = gini
				leftX, leftY, rightX, rightY = lX, lY, rX, rY
			}
		}
	}

	return bestFeature, bestThreshold, leftX, leftY, rightX, rightY
}

// Divide los datos en dos grupos basados en un umbral
func split(X [][]float64, Y []float64, feature int, threshold float64) ([][]float64, []float64, [][]float64, []float64) {
	var leftX, rightX [][]float64
	var leftY, rightY []float64

	for i := 0; i < len(X); i++ {
		if X[i][feature] < threshold {
			leftX = append(leftX, X[i])
			leftY = append(leftY, Y[i])
		} else {
			rightX = append(rightX, X[i])
			rightY = append(rightY, Y[i])
		}
	}

	return leftX, leftY, rightX, rightY
}

// Calcula la impureza de Gini para un conjunto de datos
func giniImpurity(leftY, rightY []float64) float64 {
	total := len(leftY) + len(rightY)
	if total == 0 {
		return 0
	}

	leftGini := 1.0
	rightGini := 1.0

	for _, v := range uniqueValues(leftY) {
		p := float64(count(leftY, v)) / float64(len(leftY))
		leftGini -= p * p
	}

	for _, v := range uniqueValues(rightY) {
		p := float64(count(rightY, v)) / float64(len(rightY))
		rightGini -= p * p
	}

	return (leftGini*float64(len(leftY)) + rightGini*float64(len(rightY))) / float64(total)
}

// Funciones auxiliares

func bootstrapSample(X [][]float64, Y []float64) ([][]float64, []float64) {
	rand.Seed(time.Now().UnixNano())
	n := len(X)
	sampleX := make([][]float64, n)
	sampleY := make([]float64, n)
	for i := 0; i < n; i++ {
		idx := rand.Intn(n)
		sampleX[i] = X[idx]
		sampleY[i] = Y[idx]
	}
	return sampleX, sampleY
}

func uniqueValues(arr []float64) []float64 {
	unique := make(map[float64]bool)
	for _, v := range arr {
		unique[v] = true
	}
	keys := make([]float64, 0, len(unique))
	for k := range unique {
		keys = append(keys, k)
	}
	return keys
}

func getColumn(X [][]float64, col int) []float64 {
	column := make([]float64, len(X))
	for i := 0; i < len(X); i++ {
		column[i] = X[i][col]
	}
	return column
}

func count(arr []float64, value float64) int {
	c := 0
	for _, v := range arr {
		if v == value {
			c++
		}
	}
	return c
}

func mean(arr []float64) float64 {
	sum := 0.0
	for _, v := range arr {
		sum += v
	}
	return sum / float64(len(arr))
}