package randomForest

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"sort"
	"sync"
)

var (
	muxConcurrent        = &sync.Mutex{}
	NumWorkersConcurrent = runtime.NumCPU() 
)

// ForestConcurrent je base class for whole forest with database, properties of ForestConcurrent and trees.
type ForestConcurrent struct {
	Data              ForestDataConcurrent // database for calculate trees
	Trees             []TreeConcurrent     // all generated trees
	Features          int        // number of attributes
	Classes           int        // number of classes
	LeafSize          int        // leaf size
	MFeatures         int        // attributes for choose proper split
	NTrees            int        // number of trees
	NSize             int        // len of data
	MaxDepth          int        // max depth of forest
	FeatureImportance []float64  //stats of FeatureImportance
}

// ForestDataConcurrent contains database
type ForestDataConcurrent struct {
	X     [][]float64 // All data are float64 numbers
	Class []int       // Result should be int numbers 0,1,2,..
}

// TreeConcurrent is one random tree in forest with BranchConcurrent and validation number
type TreeConcurrent struct {
	Root       BranchConcurrent
	Validation float64
}

// BranchConcurrent is tree structure of branches
type BranchConcurrent struct {
	Attribute        int
	Value            float64
	IsLeaf           bool
	LeafValue        []float64
	Gini             float64
	GiniGain         float64
	Size             int
	Branch0, Branch1 *BranchConcurrent
	Depth            int
}

func (forest *ForestConcurrent) buildNewTreesConcurrent(firstIndex int, trees int) {
	s := make(chan bool, NumWorkersConcurrent)
	for i := 0; i < trees; i++ {
		s <- true
		go func(j int) {
			defer func() { <-s }()
			forest.newTree(j)
		}(firstIndex + i)
	}
	for i := 0; i < NumWorkersConcurrent; i++ {
		s <- true
	}
}

// TrainConcurrent run training process. Parameter is number of calculated trees.
func (forest *ForestConcurrent) TrainConcurrent(trees int) {
	forest.defaults()
	forest.NTrees = trees
	forest.Trees = make([]TreeConcurrent, forest.NTrees)
	forest.buildNewTreesConcurrent(0, trees)
	imp := make([]float64, forest.Features)
	for i := 0; i < trees; i++ {
		z := forest.Trees[i].importance(forest)
		for i := 0; i < forest.Features; i++ {
			imp[i] += z[i]
		}
		//forest.Trees[i].Root.print()
	}
	for i := 0; i < forest.Features; i++ {
		imp[i] = imp[i] / float64(trees)
	}
	forest.FeatureImportance = imp
}

// Calculate outliers with Isolation ForestConcurrent method
func (forest *ForestConcurrent) IsolationForest() (isolations []float64, mean float64, stddev float64) {
	isolations = make([]float64, forest.NSize)
	for i, x := range forest.Data.X {
		for _, t := range forest.Trees {
			isolations[i] += float64(t.Root.depth(x))
		}
	}
	for i, is := range isolations {
		isolations[i] = is / float64(forest.NTrees)
	}
	mean = CalculateMean(isolations)
	variance := CalculateVariance(isolations, mean)
	stddev = math.Sqrt(variance)
	return
}

// AddDataRow add new data
// data: new data row
// class: result
// max: max number of data. Remove first if there is more datas. If max < 1 - unlimited
// newTrees: number of trees after add data row
// maxTress: maximum number of trees
//
// This feature support Continuous Random ForestConcurrent
func (forest *ForestConcurrent) AddDataRow(data []float64, class int, max int, newTrees int, maxTrees int) {
	forest.Data.X = append(forest.Data.X, data)
	forest.Data.Class = append(forest.Data.Class, class)
	if max > 0 && len(forest.Data.X) > max {
		forest.Data.X = forest.Data.X[1:]
		forest.Data.Class = forest.Data.Class[1:]
	}
	forest.defaults()
	index := len(forest.Trees)
	for i := 0; i < newTrees; i++ {
		forest.Trees = append(forest.Trees, TreeConcurrent{})
	}
	forest.buildNewTreesConcurrent(index, newTrees)
	//remove old trees
	if len(forest.Trees) > maxTrees && maxTrees > 0 {
		forest.Trees = forest.Trees[len(forest.Trees)-maxTrees:]
	}
	forest.NTrees = len(forest.Trees)
}

func (forest *ForestConcurrent) defaults() {
	forest.NSize = len(forest.Data.X)
	forest.Features = len(forest.Data.X[0])
	forest.Classes = 0
	for _, c := range forest.Data.Class {
		if c >= forest.Classes {
			forest.Classes = c + 1
		}
	}
	if forest.MFeatures == 0 {
		forest.MFeatures = int(math.Sqrt(float64(forest.Features)))
	}
	if forest.LeafSize == 0 {
		forest.LeafSize = forest.NSize / 20
		if forest.LeafSize <= 0 {
			forest.LeafSize = 1
		} else if forest.LeafSize > 50 {
			forest.LeafSize = 50
		}
	}
	if forest.MaxDepth == 0 {
		forest.MaxDepth = 10
	}
}

// Vote is used for calculate class in existed forest
func (forest *ForestConcurrent) Vote(x []float64) []float64 {
	votes := make([]float64, forest.Classes)
	for i := 0; i < forest.NTrees; i++ {
		v := forest.Trees[i].vote(x)
		for j := 0; j < forest.Classes && j < len(v); j++ {
			votes[j] += v[j]
		}
	}
	for j := 0; j < forest.Classes; j++ {
		votes[j] = votes[j] / float64(forest.NTrees)
	}
	return votes
}

// WeightVote use validation's weight for result
func (forest *ForestConcurrent) WeightVote(x []float64) []float64 {
	votes := make([]float64, forest.Classes)
	total := 0.0
	for i := 0; i < forest.NTrees; i++ {
		e := 1.0001 - forest.Trees[i].Validation
		w := 0.5 * math.Log(float64(forest.Classes-1)*(1-e)/e)
		if w > 0 {
			v := forest.Trees[i].vote(x)
			for j := 0; j < forest.Classes; j++ {
				votes[j] += v[j] * w
			}
			total += w
		}
	}
	for j := 0; j < forest.Classes; j++ {
		votes[j] = votes[j] / total
	}
	return votes
}

// Calculate a new tree in forest.
func (forest *ForestConcurrent) newTree(index int) {
	//data
	used := make([]bool, forest.NSize)
	x := make([][]float64, forest.NSize)
	results := make([]int, forest.NSize)
	for i := 0; i < forest.NSize; i++ {
		k := rand.Intn(forest.NSize)
		x[i] = forest.Data.X[k]
		results[i] = forest.Data.Class[k]
		used[k] = true
	}
	// build Root
	root := BranchConcurrent{}
	root.build(forest, x, results, 1)
	tree := TreeConcurrent{Root: root}
	// validation test tree
	count := 0
	e := 0.0
	for i := 0; i < forest.NSize; i++ {
		if !used[i] {
			count++
			v := root.vote(forest.Data.X[i])
			e += v[forest.Data.Class[i]]
		}
	}
	tree.Validation = e / float64(count)

	// add tree
	muxConcurrent.Lock()
	forest.Trees[index] = tree
	muxConcurrent.Unlock()
}

// PrintFeatureImportance print list of features
func (forest *ForestConcurrent) PrintFeatureImportance() {
	imp := make([]float64, forest.Features)
	for i := 0; i < forest.NTrees; i++ {
		z := forest.Trees[i].importance(forest)
		for i := 0; i < forest.Features; i++ {
			imp[i] += z[i]
		}
	}
	for i := 0; i < forest.Features; i++ {
		imp[i] = imp[i] / float64(forest.NTrees)
	}
	forest.FeatureImportance = imp

	fmt.Println("-------- feature importance")
	for i := 0; i < forest.Features; i++ {
		fmt.Println(i, forest.FeatureImportance[i])
	}
	fmt.Println("-------- cross validation")
	xs := make([]float64, 0)
	for _, tree := range forest.Trees {
		xs = append(xs, tree.Validation)
	}
	sort.Float64s(xs)
	mean := CalculateMean(xs)
	median := CalculateQuantile(xs, 0.5)
	variance := CalculateVariance(xs, mean)
	stddev := math.Sqrt(variance)

	fmt.Printf("mean=       %v\n", mean)
	fmt.Printf("median=     %v\n", median)
	fmt.Printf("variance=   %v\n", variance)
	fmt.Printf("std-dev=    %v\n", stddev)
	fmt.Printf("worst tree= %v\n", xs[0])
	fmt.Printf("best tree=  %v\n", xs[len(xs)-1])

	fmt.Println("--------")
}

func (branch *BranchConcurrent) build(forest *ForestConcurrent, x [][]float64, class []int, depth int) {
	classCount := make([]int, forest.Classes)
	for _, r := range class {
		classCount[r]++
	}
	branch.Gini = giniConcurrent(classCount)
	branch.Size = len(class)
	branch.Depth = depth

	if (len(x) <= forest.LeafSize) || (branch.Gini == 0) || branch.Depth == forest.MaxDepth {
		branch.IsLeaf = true
		branch.LeafValue = make([]float64, forest.Classes)
		for i, r := range classCount {
			if branch.Size > 0 {
				branch.LeafValue[i] = float64(r) / float64(branch.Size)
			}
		}
		return
	}
	//find best split
	attrsRandom := rand.Perm(forest.Features)[:forest.MFeatures]
	var bestAtrr int
	var bestValue float64
	var bestGini = 1.0
	for _, a := range attrsRandom {
		//sort data
		srt := make([]int, branch.Size)
		for i := 0; i < branch.Size; i++ {
			srt[i] = i
		}
		sort.Slice(srt, func(i, j int) bool {
			ii := srt[i]
			jj := srt[j]
			return x[ii][a] < x[jj][a]
		})
		//go throuh data
		v := x[srt[0]][a]
		s1 := make([]int, forest.Classes)
		s2 := make([]int, forest.Classes)
		copy(s2, classCount)
		for i := 0; i < branch.Size; i++ {
			index := srt[i]
			if x[index][a] > v {
				g1 := giniConcurrent(s1)
				g2 := giniConcurrent(s2)
				wg := (g1*float64(i) + g2*float64(branch.Size-i)) / float64(branch.Size)
				if wg < bestGini {
					bestGini = wg
					bestValue = v
					bestAtrr = a
				}
				v = x[index][a]
			}
			s1[class[index]]++
			s2[class[index]]--
		}
	}
	//split it
	branch.GiniGain = branch.Gini - bestGini
	branch.Attribute = bestAtrr
	branch.Value = bestValue
	x0 := make([][]float64, 0)
	x1 := make([][]float64, 0)
	c0 := make([]int, 0)
	c1 := make([]int, 0)
	for i := 0; i < branch.Size; i++ {
		if x[i][branch.Attribute] > branch.Value {
			x1 = append(x1, x[i])
			c1 = append(c1, class[i])
		} else {
			x0 = append(x0, x[i])
			c0 = append(c0, class[i])
		}
	}
	//create branches
	branch.Branch0 = &BranchConcurrent{}
	branch.Branch1 = &BranchConcurrent{}
	branch.Branch0.build(forest, x0, c0, depth+1)
	branch.Branch1.build(forest, x1, c1, depth+1)
}

func (tree *TreeConcurrent) vote(x []float64) []float64 {
	return tree.Root.vote(x)
}

func (tree *TreeConcurrent) importance(forest *ForestConcurrent) []float64 {
	imp := make([]float64, forest.Features)
	tree.Root.importance(imp)
	//normalize
	sum := 0.0
	for i := 0; i < forest.Features; i++ {
		sum += imp[i]
	}
	if sum > 0 {
		for i := 0; i < forest.Features; i++ {
			imp[i] = imp[i] / sum
		}
	}
	return imp
}

func (branch *BranchConcurrent) importance(imp []float64) {
	if branch.IsLeaf {
		return
	}
	imp[branch.Attribute] += float64(branch.Size) * branch.Gini
	branch.Branch0.importance(imp)
	branch.Branch1.importance(imp)

}

func (branch *BranchConcurrent) vote(x []float64) []float64 {
	if branch.IsLeaf {
		return branch.LeafValue
	}
	if x[branch.Attribute] > branch.Value {
		return branch.Branch1.vote(x)
	}
	return branch.Branch0.vote(x)
}

func (branch *BranchConcurrent) depth(x []float64) int {
	if branch.IsLeaf {
		return branch.Depth
	}
	if x[branch.Attribute] > branch.Value {
		return branch.Branch1.depth(x)
	}
	return branch.Branch0.depth(x)
}

func giniConcurrent(data []int) float64 {
	sum := 0
	for _, a := range data {
		sum += a
	}
	sumF := float64(sum)
	g := 1.0
	for _, a := range data {
		if sumF != 0 {
			g = g - (float64(a)/sumF)*(float64(a)/sumF)
		}
	}
	return g
}

func (forest *ForestConcurrent) PredictConcurrent(data [][]float64) []int {
    predictions := make([]int, len(data))
    for i, x := range data {
        probabilities := forest.Vote(x)
        maxIndex := 0
        maxValue := probabilities[0]
        for j, value := range probabilities {
            if value > maxValue {
                maxValue = value
                maxIndex = j
            }
        }
        predictions[i] = maxIndex
    }
    return predictions
}

// Accuracy calcula la precisión del modelo dado un conjunto de predicciones y sus etiquetas verdaderas
func (forest *ForestConcurrent) Accuracy(predictions []int, trueLabels []int) float64 {
    correct := 0
    for i, label := range trueLabels {
        if predictions[i] == label {
            correct++
        }
    }
    return float64(correct) / float64(len(trueLabels))
}