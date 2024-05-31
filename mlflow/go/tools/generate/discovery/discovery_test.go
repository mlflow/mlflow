package discovery_test

import (
	"testing"

	"github.com/mlflow/mlflow/mlflow/go/tools/generate/discovery"
)

func TestPattern(t *testing.T) {
	t.Parallel()

	scenarios := []struct {
		name     string
		endpoint discovery.Endpoint
		expected string
	}{
		{
			name: "simple GET",
			endpoint: discovery.Endpoint{
				Method: "GET",
				Path:   "/mlflow/experiments/get-by-name",
			},
			expected: "/mlflow/experiments/get-by-name",
		},
		{
			name: "simple POST",
			endpoint: discovery.Endpoint{
				Method: "POST",
				Path:   "/mlflow/experiments/create",
			},
			expected: "/mlflow/experiments/create",
		},
		{
			name: "PUT with route parameter",
			endpoint: discovery.Endpoint{
				Method: "PUT",
				Path:   "/mlflow-artifacts/artifacts/<path:artifact_path>",
			},
			expected: "/mlflow-artifacts/artifacts/:path",
		},
	}

	for _, scenario := range scenarios {
		currentScenario := scenario
		t.Run(currentScenario.name, func(t *testing.T) {
			t.Parallel()

			actual := currentScenario.endpoint.GetFiberPath()

			if actual != currentScenario.expected {
				t.Errorf("Expected %s, got %s", currentScenario.expected, actual)
			}
		})
	}
}
