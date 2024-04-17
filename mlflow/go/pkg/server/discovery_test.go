package server_test

import (
	"testing"

	"github.com/mlflow/mlflow/mlflow/go/pkg/server"
)

func TestPatternt(t *testing.T) {
	scenarios := []struct {
		name     string
		endpoint server.Endpoint
		expected string
	}{
		{name: "simple GET", endpoint: server.Endpoint{Method: "GET", Path: "/mlflow/experiments/get-by-name"}, expected: "GET /mlflow/experiments/get-by-name"},
		{name: "simple POST", endpoint: server.Endpoint{Method: "POST", Path: "/mlflow/experiments/create"}, expected: "POST /mlflow/experiments/create"},
		{name: "PUT with route parameter", endpoint: server.Endpoint{Method: "PUT", Path: "/mlflow-artifacts/artifacts/<path:artifact_path>"}, expected: "PUT /mlflow-artifacts/artifacts/{path}"},
	}

	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			actual := scenario.endpoint.GetPattern()
			if actual != scenario.expected {
				t.Errorf("Expected %s, got %s", scenario.expected, actual)
			}
		})
	}
}
