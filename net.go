package goregression

import (
	"math/rand/v2"

	"gonum.org/v1/gonum/mat"
)

type Activation struct {
	Activate   func(float64) float64
	Derivative func(float64) float64
}

type Model struct {
	Weights  []*mat.Dense
	Internal Activation
	Output   Activation
}

func NewModel(source *rand.Rand, Internal Activation, Output Activation, layers ...int) *Model {
	if len(layers) < 2 {
		panic("there should be at least 2 layers: input and output")
	}
	model := &Model{
		Internal: Internal,
		Output:   Output,
	}
	for i, c := range layers[:len(layers)-1] {
		c = c + 1 // include bias
		r := layers[i+1]
		randWeights := make([]float64, r*c)
		for j := range randWeights {
			randWeights[j] = source.NormFloat64()
		}
		model.Weights = append(model.Weights, mat.NewDense(r, c, randWeights))
	}
	return model
}

func (m Model) InputSize() int {
	_, c := m.Weights[0].Dims()
	return c - 1
}

func (m Model) OutputSize() int {
	r, _ := m.Weights[len(m.Weights)-1].Dims()
	return r
}

func (m Model) Predict(start mat.Vector) mat.Vector {
	var input *mat.VecDense
	input = mat.NewVecDense(start.Len()+1, nil)
	for i := 0; i < start.Len(); i++ {
		input.SetVec(i, m.Internal.Activate(start.AtVec(i)))
	}
	input.SetVec(start.Len(), 1)
	for _, weights := range m.Weights[:len(m.Weights)-1] {
		r, _ := weights.Dims()
		output := mat.NewVecDense(r, nil)
		output.MulVec(weights, input)
		newIn := make([]float64, 0, output.Len()+1)
		for i := 0; i < output.Len(); i++ {
			newIn = append(newIn, m.Internal.Activate(output.AtVec(i)))
		}
		newIn = append(newIn, 1)
		input = mat.NewVecDense(len(newIn), newIn)
	}
	finalw := m.Weights[len(m.Weights)-1]
	r, _ := finalw.Dims()
	output := mat.NewVecDense(r, nil)
	output.MulVec(finalw, input)
	for i := 0; i < r; i++ {
		output.SetVec(i, m.Output.Activate(output.AtVec(i)))
	}
	return output
}

func (m Model) String() string {
	return "not implemented"
}

type TrainingContext struct {
	*Model
	GeneratedNodes []*mat.VecDense
	PreNormalized  []*mat.VecDense
}

func (tc *TrainingContext) feedForward(input mat.Vector) {
	// init activation layers
	if len(tc.GeneratedNodes) != len(tc.Weights)+1 {
		tc.GeneratedNodes = make([]*mat.VecDense, 0, len(tc.Weights)+1)
		tc.PreNormalized = make([]*mat.VecDense, 0, len(tc.Weights)+1)
		for i, w := range tc.Weights {
			_, c := w.Dims()
			tc.GeneratedNodes = append(tc.GeneratedNodes, mat.NewVecDense(c, nil))
			tc.PreNormalized = append(tc.PreNormalized, mat.NewVecDense(c-1, nil))
			tc.GeneratedNodes[i].SetVec(c-1, 1.0)
		}
		r, _ := tc.Weights[len(tc.Weights)-1].Dims()
		tc.GeneratedNodes = append(tc.GeneratedNodes, mat.NewVecDense(r, nil))
		tc.PreNormalized = append(tc.PreNormalized, mat.NewVecDense(r, nil))
	}

	// read in input vector
	if input.Len()+1 != tc.GeneratedNodes[0].Len() {
		panic("incorrect input size")
	}
	for i := 0; i < input.Len(); i++ {
		tc.PreNormalized[0].SetVec(i, input.AtVec(i))
		tc.GeneratedNodes[0].SetVec(i, tc.Internal.Activate(input.AtVec(i)))
	}

	// generate next layers
	for layer := 1; layer < len(tc.GeneratedNodes)-1; layer++ {
		tc.PreNormalized[layer].MulVec(tc.Weights[layer-1], tc.GeneratedNodes[layer-1])
		for i := 0; i < tc.GeneratedNodes[layer].Len()-1; i++ {
			tc.GeneratedNodes[layer].SetVec(i, tc.Internal.Activate(tc.PreNormalized[layer].AtVec(i)))
		}
	}

	// generate output layer
	tc.PreNormalized[len(tc.PreNormalized)-1].MulVec(tc.Weights[len(tc.Weights)-1], tc.GeneratedNodes[len(tc.GeneratedNodes)-2])
	for i := 0; i < tc.GeneratedNodes[len(tc.GeneratedNodes)-1].Len(); i++ {
		tc.GeneratedNodes[len(tc.GeneratedNodes)-1].SetVec(i, tc.Output.Activate(tc.PreNormalized[len(tc.PreNormalized)-1].AtVec(i)))
	}
}

func (tc *TrainingContext) backPropogate(target mat.Vector, lrate float64) float64 {
	LearningError := 0.0
	for i := 0; i < target.Len(); i++ {
		diff := target.AtVec(i) - tc.GeneratedNodes[len(tc.GeneratedNodes)-1].AtVec(i)
		LearningError += (diff * diff) / 2
	}
	LearningError /= float64(target.Len())
	Error := LearningError
	LearningError *= lrate

	deltas := make([][]float64, len(tc.GeneratedNodes))
	outputsize := tc.OutputSize()
	for node := 0; node < outputsize; node++ {
		deltas[len(deltas)-1] = append(deltas[len(deltas)-1], (tc.GeneratedNodes[len(tc.GeneratedNodes)-1].AtVec(node)-target.AtVec(node))*tc.Output.Derivative(tc.PreNormalized[len(tc.PreNormalized)-1].AtVec(node)))
	}

	for layer := len(deltas) - 2; layer > 0; layer-- {
		deltas[layer] = make([]float64, tc.GeneratedNodes[layer].Len())
		for node := 0; node < tc.GeneratedNodes[layer].Len(); node++ {
			sum := 0.0
			for nextnode := 0; nextnode < tc.GeneratedNodes[layer+1].Len(); nextnode++ {
				sum += tc.Weights[layer].At(nextnode, node) * deltas[layer+1][nextnode]
			}
			if node < tc.PreNormalized[layer].Len() {
				deltas[layer][node] = sum * tc.Internal.Derivative(tc.PreNormalized[layer].AtVec(node))
			} else {
				deltas[layer][node] = sum
			}
		}
	}

	for layer, weights := range tc.Weights {
		R, C := weights.Dims()
		for r := 0; r < R; r++ {
			for c := 0; c < C; c++ {
				current := weights.At(r, c)
				div := tc.GeneratedNodes[layer].AtVec(c) * deltas[layer+1][r]
				if div == 0 {
					continue
				}
				change := LearningError / div
				tc.Weights[layer].Set(r, c, current-change)
			}
		}
	}
	return Error
}

func (tc *TrainingContext) Train(trainingSet [][]mat.Vector, iterations int, lrate float64, debug func(epoch int, err float64)) {
	if debug == nil {
		debug = func(epoch int, error float64) {}
	}
	for i := 0; i < iterations; i++ {
		totalerror := 0.0
		for _, set := range trainingSet {
			tc.feedForward(set[0])
			totalerror += tc.backPropogate(set[1], lrate)
		}
		debug(i, totalerror)
	}
}
