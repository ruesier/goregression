package goregression

import (
	"encoding/json"
	"fmt"
	"math"
	"regexp"
	"strconv"
)

type Activation struct {
	Activate   func(float64) float64
	Derivative func(float64) float64
	Show       func() string
}

var Tanh = Activation{
	Activate: math.Tanh,
	Derivative: func(f float64) float64 {
		t := math.Tanh(f)
		return 1 - t*t
	},
	Show: func() string { return "Tanh" },
}

func Linear(slope float64) Activation {
	return Activation{
		Activate:   func(f float64) float64 { return slope * f },
		Derivative: func(f float64) float64 { return slope },
		Show:       func() string { return fmt.Sprintf("Linear(%f)", slope) },
	}
}

var linearPattern = regexp.MustCompile(`Linear\(([.0-9eE+-]*)\)`)

func Strech(factor float64, General Activation) Activation {
	return Activation{
		Activate:   func(f float64) float64 { return General.Activate(factor * f) },
		Derivative: func(f float64) float64 { return factor * General.Derivative(factor*f) },
		Show:       func() string { return fmt.Sprintf("Strech(%f, %s)", factor, General.Show()) },
	}
}

var strechPattern = regexp.MustCompile(`Strech\(([.0-9eE+-]*), (.*)\)`)

func Scale(factor float64, General Activation) Activation {
	return Activation{
		Activate:   func(f float64) float64 { return factor * General.Activate(f) },
		Derivative: func(f float64) float64 { return factor * General.Derivative(f) },
		Show:       func() string { return fmt.Sprintf("Scale(%f, %s)", factor, General.Show()) },
	}
}

var scalePattern = regexp.MustCompile(`Scale\(([.0-9eE+-]*), (.*)\)`)

func sigmoid(f float64) float64 {
	return 1 / (1 + math.Exp(-f))
}

var Sigmoid = Activation{
	Activate: sigmoid,
	Derivative: func(f float64) float64 {
		s := sigmoid(f)
		return s * (1 - s)
	},
	Show: func() string { return "Sigmoid" },
}

var ReLU = Activation{
	Activate: func(f float64) float64 {
		return max(0, f)
	},
	Derivative: func(f float64) float64 {
		if f > 0 {
			return 1
		}
		return 0
	},
	Show: func() string { return "ReLU" },
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
			return 1 / (1 + f)
		}
		return 1 / (1 - f)
	},
	Show: func() string { return "BiLn" },
}

func (a Activation) String() string {
	return a.Show()
}

func (a Activation) MarshalJSON() ([]byte, error) {
	return json.Marshal(a.String())
}

func (a *Activation) UnmarshalJSON(text []byte) error {
	var name string
	if err := json.Unmarshal(text, &name); err != nil {
		return err
	}
	var err error
	*a, err = matchActivation(name)
	return err
}

func matchActivation(name string) (Activation, error) {
	switch name {
	case "Tanh":
		return Tanh, nil
	case "Sigmoid":
		return Sigmoid, nil
	case "BiLn":
		return BiLn, nil
	case "ReLU":
		return ReLU, nil
	}
	if match := linearPattern.FindStringSubmatch(name); len(match) == 2 {
		slope, err := strconv.ParseFloat(match[1], 64)
		if err != nil {
			return Activation{}, err
		}
		return Linear(slope), nil
	}
	if match := scalePattern.FindStringSubmatch(name); len(match) == 3 {
		factor, err := strconv.ParseFloat(match[1], 64)
		if err != nil {
			return Activation{}, err
		}
		subAc, err := matchActivation(match[2])
		if err != nil {
			return Activation{}, err
		}
		return Scale(factor, subAc), nil
	}
	if match := strechPattern.FindStringSubmatch(name); len(match) == 3 {
		factor, err := strconv.ParseFloat(match[1], 64)
		if err != nil {
			return Activation{}, err
		}
		subAc, err := matchActivation(match[2])
		if err != nil {
			return Activation{}, err
		}
		return Strech(factor, subAc), nil
	}
	return Activation{}, fmt.Errorf("unrecognized Activation: %s", name)
}
