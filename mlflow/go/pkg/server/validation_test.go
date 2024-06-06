package server_test

import (
	"errors"
	"testing"

	"github.com/go-playground/validator/v10"
	"github.com/stretchr/testify/require"

	"github.com/mlflow/mlflow/mlflow/go/pkg/protos"
	"github.com/mlflow/mlflow/mlflow/go/pkg/server"
	"github.com/mlflow/mlflow/mlflow/go/pkg/utils"
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

func TestUniqueParamsInLogBatch(t *testing.T) {
	t.Parallel()

	logBatchRequest := &protos.LogBatch{
		Params: []*protos.Param{
			{Key: utils.PtrTo("key1"), Value: utils.PtrTo("value1")},
			{Key: utils.PtrTo("key1"), Value: utils.PtrTo("value2")},
		},
	}

	validator, err := server.NewValidator()
	require.NoError(t, err)

	err = validator.Struct(logBatchRequest)
	if err == nil {
		t.Error("Expected uniqueParams validation error, got none")
	}
}

func TestEmptyParamsInLogBatch(t *testing.T) {
	t.Parallel()

	logBatchRequest := &protos.LogBatch{
		RunId:  utils.PtrTo("odcppTsGTMkHeDcqfZOYDMZSf"),
		Params: make([]*protos.Param, 0),
	}

	validator, err := server.NewValidator()
	require.NoError(t, err)

	err = validator.Struct(logBatchRequest)
	if err != nil {
		t.Errorf("Unexpected uniqueParams validation error, got %v", err)
	}
}

func TestMissingTimestampInNestedMetric(t *testing.T) {
	t.Parallel()

	serverValidator, err := server.NewValidator()
	require.NoError(t, err)

	logBatch := protos.LogBatch{
		RunId: utils.PtrTo("odcppTsGTMkHeDcqfZOYDMZSf"),
		Metrics: []*protos.Metric{
			{
				Key:   utils.PtrTo("mae"),
				Value: utils.PtrTo(2.5),
			},
		},
	}

	err = serverValidator.Struct(&logBatch)
	if err == nil {
		t.Error("Expected dip validation error, got none")
	}

	var validationErrors validator.ValidationErrors
	if errors.As(err, &validationErrors) {
		if len(validationErrors) != 1 {
			t.Errorf("Expected 1 validation error, got %v", len(validationErrors))
		}

		validationError := validationErrors[0]
		if validationError.Tag() != "dip" {
			t.Errorf("Expected dip validation error, got %v", validationError.Tag())
		}
	} else {
		t.Error("Expected validation error, got none")
	}
}
