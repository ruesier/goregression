package goregression

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestMul(t *testing.T) {
	output := mat.NewVecDense(2, nil)
	input := mat.NewVecDense(2, []float64{1, 2})
	weights := mat.NewDense(2, 2, []float64{
		1, 2,
		3, 4,
	})
	output.MulVec(weights, input)
	if output.AtVec(0) != 5 {
		t.Fatalf("wrong first output: %f", output.AtVec(0))
	}
	if output.AtVec(1) != 11 {
		t.Fatalf("wrong second output: %f", output.AtVec(1))
	}
}

func TestPredict(t *testing.T) {
	model := Model{
		Weights: []mat.Mutable{
			mat.NewDense(2, 3, []float64{
				1, 2, 0,
				3, 4, 0,
			}),
			mat.NewDense(1, 3, []float64{
				1, 2, 0,
			}),
		},
		Output:   Linear(1),
		Internal: Linear(1),
	}
	for _, test := range []struct {
		input  mat.Vector
		expect mat.Vector
	}{
		{
			input:  mat.NewVecDense(2, []float64{1, 0}),
			expect: mat.NewVecDense(1, []float64{7}),
		},
		{
			input:  mat.NewVecDense(2, []float64{1, 2}),
			expect: mat.NewVecDense(1, []float64{27}),
		},
	} {
		out := model.Predict(test.input)
		if out.Len() != test.expect.Len() {
			t.Fatal("incorrect output size")
		}
		for i := 0; i < out.Len(); i++ {
			if out.AtVec(i) != test.expect.AtVec(i) {
				t.Fatalf("incorrect output value, out[%d] = %f, expect[%d] = %f", i, out.AtVec(i), i, test.expect.AtVec(i))
			}
		}
	}
}

func TestTraining(t *testing.T) {
	// AND
	andTest := [][]mat.Vector{
		{
			mat.NewVecDense(2, []float64{0, 0}),
			mat.NewVecDense(1, []float64{0}),
		},
		{
			mat.NewVecDense(2, []float64{1, 0}),
			mat.NewVecDense(1, []float64{0}),
		},
		{
			mat.NewVecDense(2, []float64{0, 1}),
			mat.NewVecDense(1, []float64{0}),
		},
		{
			mat.NewVecDense(2, []float64{1, 1}),
			mat.NewVecDense(1, []float64{1}),
		},
	}
	train := TrainingContext{
		Model: NewModel(rand.New(rand.NewPCG(20, 24)), ReLU, Sigmoid, 2, 4, 1),
	}
	train.Train(andTest, 3000, 0.6, func(epoch int, err float64) {
		if epoch%1000 == 999 {
			t.Logf("And iteration %d: error %f", epoch, err)
		}
	})
	for _, test := range andTest {
		input, expect := test[0], test[1]
		output := train.Predict(input)
		if math.Round(output.AtVec(0)) != expect.AtVec(0) {
			t.Errorf("AND test failed. Got (%f AND %f) == %f, want %f", input.AtVec(0), input.AtVec(1), output.AtVec(0), expect.AtVec(0))
		}
	}
	// XOR
	xorTest := [][]mat.Vector{
		{
			mat.NewVecDense(2, []float64{0, 0}),
			mat.NewVecDense(1, []float64{0}),
		},
		{
			mat.NewVecDense(2, []float64{1, 0}),
			mat.NewVecDense(1, []float64{1}),
		},
		{
			mat.NewVecDense(2, []float64{0, 1}),
			mat.NewVecDense(1, []float64{1}),
		},
		{
			mat.NewVecDense(2, []float64{1, 1}),
			mat.NewVecDense(1, []float64{0}),
		},
	}
	train = TrainingContext{
		Model: NewModel(rand.New(rand.NewPCG(30, 34)), ReLU, Sigmoid, 2, 4, 1),
	}
	train.Train(xorTest, 3000, 0.4, func(epoch int, err float64) {
		if epoch%1000 == 999 {
			t.Logf("XOR iteration %d: error %f", epoch, err)
		}
	})
	for _, test := range xorTest {
		input, expect := test[0], test[1]
		output := train.Predict(input)
		if math.Round(output.AtVec(0)) != expect.AtVec(0) {
			t.Errorf("XOR test failed. Got (%f XOR %f) == %f, want %f", input.AtVec(0), input.AtVec(1), output.AtVec(0), expect.AtVec(0))
		}
	}
	// OR
	orTest := [][]mat.Vector{
		{
			mat.NewVecDense(2, []float64{0, 0}),
			mat.NewVecDense(1, []float64{0}),
		},
		{
			mat.NewVecDense(2, []float64{1, 0}),
			mat.NewVecDense(1, []float64{1}),
		},
		{
			mat.NewVecDense(2, []float64{0, 1}),
			mat.NewVecDense(1, []float64{1}),
		},
		{
			mat.NewVecDense(2, []float64{1, 1}),
			mat.NewVecDense(1, []float64{1}),
		},
	}
	train = TrainingContext{
		Model: NewModel(rand.New(rand.NewPCG(30, 34)), ReLU, Sigmoid, 2, 4, 1),
	}
	train.Train(orTest, 3000, 0.4, func(epoch int, err float64) {
		if epoch%1000 == 999 {
			t.Logf("OR iteration %d: error %f", epoch, err)
		}
	})
	for _, test := range orTest {
		input, expect := test[0], test[1]
		output := train.Predict(input)
		if math.Round(output.AtVec(0)) != expect.AtVec(0) {
			t.Errorf("OR test failed. Got (%f OR %f) == %f, want %f", input.AtVec(0), input.AtVec(1), output.AtVec(0), expect.AtVec(0))
		}
	}

	// Regression test
	regTest := [][]mat.Vector{
		{
			mat.NewVecDense(1, []float64{3}),
			mat.NewVecDense(1, []float64{6}),
		},
		{
			mat.NewVecDense(1, []float64{4}),
			mat.NewVecDense(1, []float64{8}),
		},
		{
			mat.NewVecDense(1, []float64{5}),
			mat.NewVecDense(1, []float64{10}),
		},
		{
			mat.NewVecDense(1, []float64{6}),
			mat.NewVecDense(1, []float64{12}),
		},
	}
	train = TrainingContext{
		Model: NewModel(rand.New(rand.NewPCG(3453, 9988)), Sigmoid, Linear(1), 1, 3, 3, 1),
	}
	checkNaN := true
	train.Train(regTest, 30000, 0.1, func(epoch int, err float64) {
		if epoch%1000 == 999 {
			t.Logf("Reg iteration %d: error %f", epoch, err)
		}
		if checkNaN && train.Model.hasNaN() {
			t.Logf("Regession Model, epoch %d\n%s", epoch, train.Model)
			checkNaN = false
		}
	})
	for _, test := range regTest {
		input, expect := test[0], test[1]
		output := train.Predict(input)
		if math.Round(output.AtVec(0)) != expect.AtVec(0) {
			t.Errorf("Reg test failed. Got (%f) => %f, want %f", input.AtVec(0), output.AtVec(0), expect.AtVec(0))
		}
	}
	if t.Failed() {
		t.Log("Regession Model\n", train.Model)
	}
}

func TestChunkTraining(t *testing.T) {
	// Regression test
	regTest := [][]mat.Vector{
		{
			mat.NewVecDense(1, []float64{3}),
			mat.NewVecDense(1, []float64{6}),
		},
		{
			mat.NewVecDense(1, []float64{4}),
			mat.NewVecDense(1, []float64{8}),
		},
		{
			mat.NewVecDense(1, []float64{5}),
			mat.NewVecDense(1, []float64{10}),
		},
		{
			mat.NewVecDense(1, []float64{6}),
			mat.NewVecDense(1, []float64{12}),
		},
	}
	train := TrainingContext{
		Model: NewModel(rand.New(rand.NewPCG(3453, 9988)), Sigmoid, Linear(1), 1, 3, 3, 1),
	}
	train.TrainChunked(regTest, 30000, 2, 2, 0.1, nil)
	for _, test := range regTest {
		input, expect := test[0], test[1]
		output := train.Predict(input)
		if math.Round(output.AtVec(0)) != expect.AtVec(0) {
			t.Errorf("Reg test failed. Got (%f) => %f, want %f", input.AtVec(0), output.AtVec(0), expect.AtVec(0))
		}
	}
	if t.Failed() {
		t.Log("Regession Model\n", train.Model)
	}
}


func BenchmarkChunkTraining(b *testing.B) {
	gen := rand.New(rand.NewPCG(5987, 2908))
	randTest := make([][]mat.Vector, 50)
	inputsize := 10
	outputsize := 5
	for i := range randTest {
		randTest[i] = []mat.Vector{
			mat.NewVecDense(inputsize, genN(gen, inputsize)),
			mat.NewVecDense(outputsize, genN(gen, outputsize)),
		}
	}
	for workers := 6; workers <= 9; workers++ {
		for chunk := 8; chunk <= 15; chunk++ {
			// testing over different ranges of workers and chunks, best result on my machine was workers 7, chunk 13
			// to allow for more often updates to Model I am thinking the optimum might be workers 7, chunk 10
			b.Run(fmt.Sprintf("workers %d chunks %d", workers, chunk), func(b *testing.B) {
				train := TrainingContext{
					Model: NewModel(rand.New(rand.NewPCG(3453, 9988)), Tanh, Sigmoid, inputsize, 20, 20, outputsize),
				}
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					train.TrainChunked(randTest, 50, workers, chunk, 0.1, nil)
				}
			})
		}
	}
}

func BenchmarkTraining(b *testing.B) {
	gen := rand.New(rand.NewPCG(5987, 2908))
	randTest := make([][]mat.Vector, 50)
	inputsize := 10
	outputsize := 5
	for i := range randTest {
		randTest[i] = []mat.Vector{
			mat.NewVecDense(inputsize, genN(gen, inputsize)),
			mat.NewVecDense(outputsize, genN(gen, outputsize)),
		}
	}

	train := TrainingContext{
		Model: NewModel(rand.New(rand.NewPCG(3453, 9988)), Tanh, Sigmoid, inputsize, 20, 20, outputsize),
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		train.Train(randTest, 50, 0.1, nil)
	}
}


func genN(source *rand.Rand, n int) []float64 {
	result := make([]float64, n)
	for i := range result {
		result[i] = source.Float64()
	}
	return result
}

func TestTrainingVSChunk(t *testing.T) {
	regTest := [][]mat.Vector{
		{
			mat.NewVecDense(1, []float64{3}),
			mat.NewVecDense(1, []float64{6}),
		},
		{
			mat.NewVecDense(1, []float64{4}),
			mat.NewVecDense(1, []float64{8}),
		},
		{
			mat.NewVecDense(1, []float64{5}),
			mat.NewVecDense(1, []float64{10}),
		},
		{
			mat.NewVecDense(1, []float64{6}),
			mat.NewVecDense(1, []float64{12}),
		},
	}
	train1 := TrainingContext{
		Model: NewModel(rand.New(rand.NewPCG(3453, 9988)), Sigmoid, Linear(1), 1, 3, 3, 1),
	}
	train2 := TrainingContext{
		Model: train1.Clone(),
	}
	train1.Train(regTest, 10, 0.5, nil)
	train2.TrainChunked(regTest, 10, 1, 1, 0.5, nil) // when workers = 1 and chunk size = 1, should reproduce the same result as Train

	for layer, weights := range train1.Weights {
		R, C := weights.Dims()
		for r := 0; r < R; r++ {
			for c := 0; c < C; c++ {
				if weights.At(r, c) != train2.Weights[layer].At(r, c) {
					t.Fail()
					t.Logf("mismatch [%d][%d][%d]: %f != %f", layer, r, c, weights.At(r, c), train2.Weights[layer].At(r, c))
				}
			}
		}
	}
}
