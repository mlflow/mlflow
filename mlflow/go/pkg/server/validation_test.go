package server_test

import (
	"testing"

	"github.com/mlflow/mlflow/mlflow/go/pkg/server"
)

var validator = server.NewValidator()

type PositiveInteger struct {
	Value string `validate:"stringAsPositiveInteger"`
}

type Scenario struct {
	name          string
	input         any
	shouldTrigger bool
}

func runScenarios(t *testing.T, scenarios []Scenario) {
	t.Helper()

	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			errs := validator.Struct(scenario.input)

			if scenario.shouldTrigger && errs == nil {
				t.Errorf("Expected validation error, got nil")
			}

			if !scenario.shouldTrigger && errs != nil {
				t.Errorf("Expected no validation error, got %v", errs)
			}
		})
	}
}

func TestStringAsPositiveInteger(t *testing.T) {
	scenarios := []Scenario{
		{name: "positive integer", input: PositiveInteger{Value: "1"}, shouldTrigger: false},
		{name: "zero", input: PositiveInteger{Value: "0"}, shouldTrigger: true},
		{name: "negative integer", input: PositiveInteger{Value: "-1"}, shouldTrigger: true},
		{name: "alphabet", input: PositiveInteger{Value: "a"}, shouldTrigger: true},
	}

	runScenarios(t, scenarios)
}

type UrlWithoutFragmentsOrParams struct {
	Value string `validate:"uriWithoutFragmentsOrParamsOrDotDotInQuery"`
}

func TestUriWithoutFragmentsOrParams(t *testing.T) {
	scenarios := []Scenario{
		{name: "valid url", input: UrlWithoutFragmentsOrParams{Value: "http://example.com"}, shouldTrigger: false},
		{name: "only trigger when url is not empty", input: UrlWithoutFragmentsOrParams{Value: ""}, shouldTrigger: false},
		{name: "url with fragment", input: UrlWithoutFragmentsOrParams{Value: "http://example.com#fragment"}, shouldTrigger: true},
		{name: "url with query parameters", input: UrlWithoutFragmentsOrParams{Value: "http://example.com?query=param"}, shouldTrigger: true},
		{name: "unparsable url", input: UrlWithoutFragmentsOrParams{Value: ":invalid-url"}, shouldTrigger: true},
		{name: ".. in query", input: UrlWithoutFragmentsOrParams{Value: "http://example.com?query=./.."}, shouldTrigger: true},
	}

	runScenarios(t, scenarios)
}
