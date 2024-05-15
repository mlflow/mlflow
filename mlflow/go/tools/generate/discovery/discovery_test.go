package discovery

import (
	"testing"
)

func TestPatternt(t *testing.T) {
	scenarios := []struct {
		name     string
		endpoint Endpoint
		expected string
	}{
		{name: "simple GET", endpoint: Endpoint{Method: "GET", Path: "/mlflow/experiments/get-by-name"}, expected: "/mlflow/experiments/get-by-name"},
		{name: "simple POST", endpoint: Endpoint{Method: "POST", Path: "/mlflow/experiments/create"}, expected: "/mlflow/experiments/create"},
		{name: "PUT with route parameter", endpoint: Endpoint{Method: "PUT", Path: "/mlflow-artifacts/artifacts/<path:artifact_path>"}, expected: "/mlflow-artifacts/artifacts/:path"},
	}

	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			actual := scenario.endpoint.GetFiberPath()
			if actual != scenario.expected {
				t.Errorf("Expected %s, got %s", scenario.expected, actual)
			}
		})
	}
}
