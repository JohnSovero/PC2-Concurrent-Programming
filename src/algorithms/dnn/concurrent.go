package dnn

import (
    "errors"
    "fmt"
    "math"
    "math/rand"
	"sync"
)

// Loss calcula la pérdida entre las predicciones y las etiquetas concurrentemente
func LossConcurrent(predictions, labels Frame) float32 {
    if len(predictions) != len(labels) {
        panic("frames must be of the same length")
    }
    var wg sync.WaitGroup
    var mu sync.Mutex
    var loss float32
    numGoroutines := 4
    chunkSize := len(predictions) / numGoroutines
    for i := 0; i < numGoroutines; i++ {
        start := i * chunkSize
        end := start + chunkSize
        if i == numGoroutines-1 {
            end = len(predictions)
        }
        wg.Add(1)
        go func(start, end int) {
            defer wg.Done()
            localLoss := float32(0)
            for j := start; j < end; j++ {
                for k := range predictions[j] {
                    diff := predictions[j][k] - labels[j][k]
                    localLoss += diff * diff
                }
            }
            mu.Lock()
            loss += localLoss
            mu.Unlock()
        }(start, end)
    }
    wg.Wait()
    return loss / float32(len(predictions))
}

// MLPConcurrent provides a Multi-LayerConcurrent Perceptron which can be configured for
// any network architecture within that paradigm.
type MLPConcurrent struct {
    Layers       []*LayerConcurrent
    LearningRate float32
    Introspect   func(step StepConcurrent)
}

// StepConcurrent captures status updates that happens within a single Epoch, for use in
// introspecting models.
type StepConcurrent struct {
    Epoch int
    LossConcurrent  float32
}

// InitializeConcurrent sets up network layers with the needed memory allocations and
// references for proper operation. It is called automatically during training,
// provided separately only to facilitate more precise use of the network from
// a performance analysis perspective.
func (n *MLPConcurrent) InitializeConcurrent() {
    var prev *LayerConcurrent
    for i, layer := range n.Layers {
        var next *LayerConcurrent
        if i < len(n.Layers)-1 {
            next = n.Layers[i+1]
        }
        layer.initializeConcurrent(n, prev, next)
        prev = layer
    }
}

// TrainConcurrent takes in a set of inputs and a set of labels and trains the network
// using backpropagation to adjust internal weights to minimize loss, over the
// specified number of epochs. The final loss value is returned after training
// completes.
func (n *MLPConcurrent) TrainConcurrent(epochs int, inputs, labels Frame) (float32, error) {
    if err := n.checkConcurrent(inputs, labels); err != nil {
        return 0, err
    }

    n.InitializeConcurrent()

    var loss float32
    for e := 0; e < epochs; e++ {
        predictions := make(Frame, len(inputs))

        for i, input := range inputs {
            activations := input
            for _, layer := range n.Layers {
                activations = layer.ForwardProp(activations)
            }
            predictions[i] = activations

            for step := range n.Layers {
                l := len(n.Layers) - (step + 1)
                layer := n.Layers[l]

                if l == 0 {
                    continue
                }

                layer.BackProp(labels[i])
            }
        }

        loss = LossConcurrent(predictions, labels)
        if n.Introspect != nil {
            n.Introspect(StepConcurrent{
                Epoch: e,
                LossConcurrent:  loss,
            })
        }
    }

    return loss, nil
}

// PredictConcurrent takes in a set of input rows with the width of the input layer, and
// returns a frame of prediction rows with the width of the output layer,
// representing the predictions of the network.
func (n *MLPConcurrent) PredictConcurrent(inputs Frame) Frame {
    preds := make(Frame, len(inputs))
    for i, input := range inputs {
        activations := input
        for _, layer := range n.Layers {
            activations = layer.ForwardProp(activations)
        }
        preds[i] = activations
    }
    return preds
}

func (n *MLPConcurrent) checkConcurrent(inputs Frame, outputs Frame) error {
    if len(n.Layers) == 0 {
        return errors.New("ann must have at least one layer")
    }

    if len(inputs) != len(outputs) {
        return fmt.Errorf(
            "inputs count %d mismatched with outputs count %d",
            len(inputs), len(outputs),
        )
    }
    return nil
}

// LayerConcurrent defines a layer in the neural network. These are presently basic
// feed-forward layers that also provide capabilities to facilitate
// backpropagatin within the MLPConcurrent structure.
type LayerConcurrent struct {
    Name                     string
    Width                    int
    ActivationFunction       func(float32) float32
    ActivationFunctionDeriv  func(float32) float32
    nn                       *MLPConcurrent
    prev                     *LayerConcurrent
    next                     *LayerConcurrent
    initialized              bool
    weights                  Frame
    biases                   Vector
    lastZ                    Vector
    lastActivations          Vector
    lastE                    Vector
    lastL                    Frame
}

// initializeConcurrent sets up the needed data structures and random initial values for
// the layer. If key values are unspecified, defaults are configured.
func (l *LayerConcurrent) initializeConcurrent(nn *MLPConcurrent, prev *LayerConcurrent, next *LayerConcurrent) {
    if l.initialized || prev == nil {
        return
    }

    l.nn = nn
    l.prev = prev
    l.next = next

    if l.ActivationFunction == nil {
        l.ActivationFunction = Sigmoid
    }
    if l.ActivationFunctionDeriv == nil {
        l.ActivationFunctionDeriv = SigmoidDerivative
    }

    l.weights = make(Frame, l.Width)
    for i := range l.weights {
        l.weights[i] = make(Vector, l.prev.Width)
        for j := range l.weights[i] {
            weight := rand.NormFloat64() * math.Pow(float64(l.prev.Width), -0.5)
            l.weights[i][j] = float32(weight)
        }
    }
    l.biases = make(Vector, l.Width)
    for i := range l.biases {
        l.biases[i] = rand.Float32()
    }
    l.lastE = make(Vector, l.Width)
    l.lastL = make(Frame, l.Width)
    for i := range l.lastL {
        l.lastL[i] = make(Vector, l.prev.Width)
    }

    l.initialized = true
}

// ForwardProp takes in a set of inputs from the previous layer and performs
// forward propagation for the current layer, returning the resulting
// activations. As a special case, if this LayerConcurrent has no previous layer and is
// thus the input layer for the network, the values are passed through
// unmodified. Internal state from the calculation is persisted for later use
// in back propagation.
func (l *LayerConcurrent) ForwardProp(input Vector) Vector {
    if l.prev == nil {
        l.lastActivations = input
        return input
    }

    Z := make(Vector, l.Width)
    activations := make(Vector, l.Width)
    for i := range activations {
        nodeWeights := l.weights[i]
        nodeBias := l.biases[i]
        Z[i] = DotProduct(input, nodeWeights) + nodeBias
        activations[i] = l.ActivationFunction(Z[i])
    }
    l.lastZ = Z
    l.lastActivations = activations
    return activations
}

// BackProp performs the training process of back propagation on the layer for
// the given set of labels. Weights and biases are updated for this layer
// according to the computed error. Internal state on the backpropagation
// process is captured for further backpropagation in earlier layers of the
// network as well.
func (l *LayerConcurrent) BackProp(label Vector) {
    if l.next == nil {
        l.lastE = l.lastActivations.Subtract(label)
    } else {
        l.lastE = make(Vector, len(l.lastE))
        for j := range l.weights {
            for jn := range l.next.lastL {
                l.lastE[j] += l.next.lastL[jn][j]
            }
        }
    }
    dLdA := l.lastE.Scalar(2)
    dAdZ := l.lastZ.Apply(l.ActivationFunctionDeriv)

    for j := range l.weights {
        l.lastL[j] = l.weights[j].Scalar(l.lastE[j])
    }

    for j := range l.weights {
        for k := range l.weights[j] {
            dZdW := l.prev.lastActivations[k]
            dLdW := dLdA[j] * dAdZ[j] * dZdW
            l.weights[j][k] -= dLdW * l.nn.LearningRate
        }
    }

    biasUpdate := dLdA.ElementwiseProduct(dAdZ)
    l.biases = l.biases.Subtract(biasUpdate.Scalar(l.nn.LearningRate))
}