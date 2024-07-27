package goregression

import (
	"fmt"
	"math"
	"math/rand/v2"
	"strconv"
	"strings"
	"sync"

	"gonum.org/v1/gonum/mat"
)

type Model struct {
	Weights  []mat.Mutable
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
		input.SetVec(i, start.AtVec(i))
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
	var builder strings.Builder
	builder.WriteString("input\n")
	for i, weights := range m.Weights {
		if i > 0 {
			builder.WriteString(fmt.Sprintf("Hidden Layer %d\n", i))
		}
		R, C := weights.Dims()
		for r := 0; r < R; r++ {
			for c := 0; c < C; c++ {
				f := strconv.FormatFloat(weights.At(r, c), 'g', 3, 64)
				builder.WriteString(f)
				builder.WriteByte('\t')
			}
			builder.WriteByte('\n')
		}
	}
	builder.WriteString("output")
	return builder.String()
}

func (m Model) hasNaN() bool {
	for _, weights := range m.Weights {
		R, C := weights.Dims()
		for r := 0; r < R; r++ {
			for c := 0; c < C; c++ {
				if math.IsNaN(weights.At(r, c)) {
					return true
				}
			}
		}
	}
	return false
}

func (m *Model) Clone() *Model {
	weights := make([]mat.Mutable, len(m.Weights))
	for i, w := range m.Weights {
		R, C := w.Dims()
		nw := mat.NewDense(R, C, nil)
		for r := 0; r < R; r++ {
			for c := 0; c < C; c++ {
				nw.Set(r, c, w.At(r, c))
			}
		}
		weights[i] = nw
	}
	return &Model{
		Weights:  weights,
		Internal: m.Internal,
		Output:   m.Output,
	}
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
		tc.GeneratedNodes[0].SetVec(i, input.AtVec(i))
	}

	// generate next layers
	for layer := 1; layer < len(tc.GeneratedNodes)-1; layer++ {
		tc.PreNormalized[layer].MulVec(tc.Weights[layer-1], tc.GeneratedNodes[layer-1])
		for i := 0; i < tc.GeneratedNodes[layer].Len()-1; i++ {
			node := i
			tc.GeneratedNodes[layer].SetVec(node, tc.Internal.Activate(tc.PreNormalized[layer].AtVec(node)))
		}
	}

	// generate output layer
	tc.PreNormalized[len(tc.PreNormalized)-1].MulVec(tc.Weights[len(tc.Weights)-1], tc.GeneratedNodes[len(tc.GeneratedNodes)-2])
	for i := 0; i < tc.GeneratedNodes[len(tc.GeneratedNodes)-1].Len(); i++ {
		tc.GeneratedNodes[len(tc.GeneratedNodes)-1].SetVec(i, tc.Output.Activate(tc.PreNormalized[len(tc.PreNormalized)-1].AtVec(i)))
	}
}

func (tc *TrainingContext) backPropogate(target mat.Vector, lrate float64) float64 {
	return tc.backPropogateChanges(target, lrate, tc.Weights)
}

func (tc *TrainingContext) backPropogateChanges(target mat.Vector, lrate float64, changes []mat.Mutable) float64 {
	Error := 0.0
	for i := 0; i < target.Len(); i++ {
		diff := target.AtVec(i) - tc.GeneratedNodes[len(tc.GeneratedNodes)-1].AtVec(i)
		Error += (diff * diff) / 2
	}
	Error /= float64(target.Len())

	deltas := make([][]float64, len(tc.GeneratedNodes))
	outputsize := tc.OutputSize()
	for node := 0; node < outputsize; node++ {
		deltas[len(deltas)-1] = append(deltas[len(deltas)-1], (tc.GeneratedNodes[len(tc.GeneratedNodes)-1].AtVec(node)-target.AtVec(node))*tc.Output.Derivative(tc.PreNormalized[len(tc.PreNormalized)-1].AtVec(node)))
	}

	for layer := len(deltas) - 2; layer > 0; layer-- {
		deltas[layer] = make([]float64, tc.PreNormalized[layer].Len())
		for node := 0; node < tc.PreNormalized[layer].Len(); node++ {
			sum := 0.0
			for nextnode := 0; nextnode < tc.PreNormalized[layer+1].Len(); nextnode++ {
				sum += tc.Weights[layer].At(nextnode, node) * deltas[layer+1][nextnode]
			}
			deltas[layer][node] = sum * tc.Internal.Derivative(tc.PreNormalized[layer].AtVec(node))
		}
	}

	for layer, weights := range changes {
		R, C := weights.Dims()
		for r := 0; r < R; r++ {
			for c := 0; c < C; c++ {
				current := weights.At(r, c)
				change := lrate * tc.GeneratedNodes[layer].AtVec(c) * deltas[layer+1][r]
				changes[layer].Set(r, c, current-change)
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

type updateStep struct {
	*Model
	data [][]mat.Vector
}

func (tc *TrainingContext) TrainChunked(trainingSet [][]mat.Vector, iterations int, workers int, chunksize int, lrate float64, debug func(epoch int, current *Model)) {
	if debug == nil {
		debug = func(epoch int, current *Model) {}
	}
	stepch := make(chan updateStep)
	changech := make(chan []mat.Mutable)
	workerGroup := sync.WaitGroup{}
	workerGroup.Add(workers)
	for i := 0; i < workers; i++ {
		go func() {
			defer workerGroup.Done()
			local := new(TrainingContext)
			for step := range stepch {
				changes := make([]mat.Mutable, len(tc.Model.Weights))
				for i, w := range tc.Model.Weights {
					R, C := w.Dims()
					changes[i] = mat.NewDense(R, C, nil)
				}
				local.Model = step.Model
				for _, data := range step.data {
					local.feedForward(data[0])
					local.backPropogateChanges(data[1], lrate, changes)
				}
				changech <- changes
			}
		}()
	}

	NewModelCh := make(chan *Model)
	go func() {
		model := tc.Model.Clone()
		layerUpdatersCh := make([]chan mat.Matrix, len(model.Weights))
		updateFlag := sync.WaitGroup{}
		for i, weights := range model.Weights {
			updateCh := make(chan mat.Matrix)
			layerUpdatersCh[i] = updateCh
			go func() {
				R, C := weights.Dims()
				for update := range updateCh {
					for r := 0; r < R; r++ {
						for c := 0; c < C; c++ {
							weights.Set(r, c, weights.At(r, c) + update.At(r, c))
						}
					}
					updateFlag.Done()
				}
			}()
		}
		changeCounter := 0
		for change := range changech {
			updateFlag.Add(len(change))
			for i, cha := range change {
				layerUpdatersCh[i] <- cha
			}
			updateFlag.Wait()
			changeCounter++
			if changeCounter >= workers {
				NewModelCh <- model.Clone()
				changeCounter = 0
			}
		}
		NewModelCh <- model
		close(NewModelCh)
	}()

	stepCounter := 0
	for epoch := 0; epoch < iterations; epoch++ {
		for start := 0; start < len(trainingSet); start += chunksize {
			end := start + chunksize
			if end > len(trainingSet) {
				end = len(trainingSet)
			}
			stepch <- updateStep{
				Model: tc.Model,
				data:  trainingSet[start:end],
			}
			stepCounter++
			if stepCounter >= workers {
				tc.Model = <-NewModelCh
				stepCounter = 0
			}
		}
		debug(epoch, tc.Model)
	}
	close(stepch)
	workerGroup.Wait()
	close(changech)
	for tc.Model = range NewModelCh {
	}
}
