package goregression

import (
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
		Weights: []*mat.Dense{
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
	train.Train(andTest, 3000, 0.4, func(epoch int, err float64) {
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
	if t.Failed() {
		t.Logf("%+v", train.Model)
		t.Logf("%+v", train)
	}
	// XOR
	// OR
}
