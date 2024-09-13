package dnn

import (
    "errors"
    "fmt"
    "math"
    "math/rand"
)

// LossSequencial calcula la pérdida entre las predicciones y las etiquetas
func LossSequencial(predictions, labels Frame) float32 {
    if len(predictions) != len(labels) {
        panic("frames must be of the same length")
    }
    var loss float32
    for i := range predictions {
        for j := range predictions[i] {
            diff := predictions[i][j] - labels[i][j]
            loss += diff * diff
        }
    }
    return loss / float32(len(predictions))
}

// MLPSequencial provides a Multi-LayerSequencial Perceptron which can be configured for
// any network architecture within that paradigm.
type MLPSequencial struct {
    Layers       []*LayerSequencial
    LearningRate float32
    Introspect   func(step StepSequencial)
}

// StepSequencial captures status updates that happens within a single Epoch, for use in
// introspecting models.
type StepSequencial struct {
    Epoch int
    LossSequencial  float32
}

// InitializeSequencial sets up network layers with the needed memory allocations and
// references for proper operation. It is called automatically during training,
// provided separately only to facilitate more precise use of the network from
// a performance analysis perspective.
func (n *MLPSequencial) InitializeSequencial() {
    var prev *LayerSequencial
    for i, layer := range n.Layers {
        var next *LayerSequencial
        if i < len(n.Layers)-1 {
            next = n.Layers[i+1]
        }
        layer.initializeSequencial(n, prev, next)
        prev = layer
    }
}

// TrainSequencial takes in a set of inputs and a set of labels and trains the network
// using backpropagation to adjust internal weights to minimize loss, over the
// specified number of epochs. The final loss value is returned after training
// completes.
func (n *MLPSequencial) TrainSequencial(epochs int, inputs, labels Frame) (float32, error) {
    if err := n.checkSequencial(inputs, labels); err != nil {
        return 0, err
    }

    n.InitializeSequencial()

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

        loss = LossSequencial(predictions, labels)
        if n.Introspect != nil {
            n.Introspect(StepSequencial{
                Epoch: e,
                LossSequencial:  loss,
            })
        }
    }

    return loss, nil
}

// PredictSequencial takes in a set of input rows with the width of the input layer, and
// returns a frame of prediction rows with the width of the output layer,
// representing the predictions of the network.
func (n *MLPSequencial) PredictSequencial(inputs Frame) Frame {
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

func (n *MLPSequencial) checkSequencial(inputs Frame, outputs Frame) error {
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

// LayerSequencial defines a layer in the neural network. These are presently basic
// feed-forward layers that also provide capabilities to facilitate
// backpropagatin within the MLPSequencial structure.
type LayerSequencial struct {
    Name                     string
    Width                    int
    ActivationFunction       func(float32) float32
    ActivationFunctionDeriv  func(float32) float32
    nn                       *MLPSequencial
    prev                     *LayerSequencial
    next                     *LayerSequencial
    initialized              bool
    weights                  Frame
    biases                   Vector
    lastZ                    Vector
    lastActivations          Vector
    lastE                    Vector
    lastL                    Frame
}

// initializeSequencial sets up the needed data structures and random initial values for
// the layer. If key values are unspecified, defaults are configured.
func (l *LayerSequencial) initializeSequencial(nn *MLPSequencial, prev *LayerSequencial, next *LayerSequencial) {
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
// activations. As a special case, if this LayerSequencial has no previous layer and is
// thus the input layer for the network, the values are passed through
// unmodified. Internal state from the calculation is persisted for later use
// in back propagation.
func (l *LayerSequencial) ForwardProp(input Vector) Vector {
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
func (l *LayerSequencial) BackProp(label Vector) {
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