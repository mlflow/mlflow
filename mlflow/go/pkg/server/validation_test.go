package server_test

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/mlflow/mlflow/mlflow/go/pkg/server"
)

type PositiveInteger struct {
	Value string `validate:"stringAsPositiveInteger"`
}

type validationScenario struct {
	name          string
	input         any
	shouldTrigger bool
}

func runscenarios(t *testing.T, scenarios []validationScenario) {
	t.Helper()

	validator, err := server.NewValidator()
	require.NoError(t, err)

	for _, scenario := range scenarios {
		currentScenario := scenario
		t.Run(currentScenario.name, func(t *testing.T) {
			t.Parallel()

			errs := validator.Struct(currentScenario.input)

			if currentScenario.shouldTrigger && errs == nil {
				t.Errorf("Expected validation error, got nil")
			}

			if !currentScenario.shouldTrigger && errs != nil {
				t.Errorf("Expected no validation error, got %v", errs)
			}
		})
	}
}

func TestStringAsPositiveInteger(t *testing.T) {
	t.Parallel()

	scenarios := []validationScenario{
		{
			name:          "positive integer",
			input:         PositiveInteger{Value: "1"},
			shouldTrigger: false,
		},
		{
			name:          "zero",
			input:         PositiveInteger{Value: "0"},
			shouldTrigger: false,
		},
		{
			name:          "negative integer",
			input:         PositiveInteger{Value: "-1"},
			shouldTrigger: true,
		},
		{
			name:          "alphabet",
			input:         PositiveInteger{Value: "a"},
			shouldTrigger: true,
		},
	}

	runscenarios(t, scenarios)
}

type uriWithoutFragmentsOrParams struct {
	Value string `validate:"uriWithoutFragmentsOrParamsOrDotDotInQuery"`
}

func TestUriWithoutFragmentsOrParams(t *testing.T) {
	t.Parallel()

	scenarios := []validationScenario{
		{
			name:          "valid url",
			input:         uriWithoutFragmentsOrParams{Value: "http://example.com"},
			shouldTrigger: false,
		},
		{
			name:          "only trigger when url is not empty",
			input:         uriWithoutFragmentsOrParams{Value: ""},
			shouldTrigger: false,
		},
		{
			name:          "url with fragment",
			input:         uriWithoutFragmentsOrParams{Value: "http://example.com#fragment"},
			shouldTrigger: true,
		},
		{
			name:          "url with query parameters",
			input:         uriWithoutFragmentsOrParams{Value: "http://example.com?query=param"},
			shouldTrigger: true,
		},
		{
			name:          "unparsable url",
			input:         uriWithoutFragmentsOrParams{Value: ":invalid-url"},
			shouldTrigger: true,
		},
		{
			name:          ".. in query",
			input:         uriWithoutFragmentsOrParams{Value: "http://example.com?query=./.."},
			shouldTrigger: true,
		},
	}

	runscenarios(t, scenarios)
}
