package goregression

import "math"

var Tanh = Activation{
	Activate: math.Tanh,
	Derivative: func(f float64) float64 {
		t := math.Tanh(f)
		return 1 - t*t
	},
}

func Linear(slope float64) Activation {
	return Activation{
		Activate: func(f float64) float64 {return slope*f},
		Derivative: func(f float64) float64 {return slope},
	}
}

func Strech(General Activation, factor float64) Activation {
	return Activation{
		Activate: func(f float64) float64 { return General.Activate(factor * f)},
		Derivative: func(f float64) float64 {return factor * General.Derivative(factor * f)},
	}
}

func Scale(General Activation, factor float64) Activation {
	return Activation{
		Activate: func(f float64) float64 { return factor * General.Activate(f)},
		Derivative: func(f float64) float64 {return factor * General.Derivative(f)},
	}
}

func sigmoid(f float64) float64 {
	return 1 / (1 + math.Exp(-f))
}

var Sigmoid = Activation{
	Activate: sigmoid,
	Derivative: func(f float64) float64 {
		s := sigmoid(f)
		return s*(1-s)
	},
}

var ReLU = Activation {
	Activate: func(f float64) float64 {
		return max(0, f)
	},
	Derivative: func(f float64) float64 {
		if f > 0 {
			return 1
		}
		return 0
	},
}

var BiLn = Activation{
	Activate: func(f float64) float64 {
		if f == 0 {
			return 0
		}
		if f > 0 {
			return math.Log(1 + f)
		}
		return -math.Log(1 - f)
	},
	Derivative: func(f float64) float64 {
		if f == 0 {
			return 1
		}
		if f > 0 {
			return 1/(1+f)
		}
		return 1/(1-f)
	},
}
