package server_test

import (
	"testing"

	"github.com/mlflow/mlflow/mlflow/go/pkg/server"
)

var validator = server.NewValidator()

type Input struct {
	PositiveInteger string `validate:"stringAsPositiveInteger"`
}

func TestStringAsPositiveInteger(t *testing.T) {
	scenarios := []struct {
		name          string
		input         string
		shouldTrigger bool
	}{
		{name: "positive integer", input: "1", shouldTrigger: false},
		{name: "zero", input: "0", shouldTrigger: true},
		{name: "negative integer", input: "-1", shouldTrigger: true},
		{name: "alphabet", input: "a", shouldTrigger: true},
	}

	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			input := Input{PositiveInteger: scenario.input}
			errs := validator.Struct(input)

			if scenario.shouldTrigger && errs == nil {
				t.Errorf("Expected validation error, got nil")
			}

			if !scenario.shouldTrigger && errs != nil {
				t.Errorf("Expected no validation error, got %v", errs)
			}
		})
	}
}
