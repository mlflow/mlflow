package server_test

import (
	"testing"

	"github.com/mlflow/mlflow/mlflow/go/pkg/server"
)

// This is sanity check test to ensure we have as much handler entries as we have service endpoints.
func TestHandlerSync(t *testing.T) {
	endpoints := server.GetServiceEndpoints()

	for endpointName := range endpoints {
		_, ok := server.Handlers[endpointName]
		if !ok {
			t.Errorf("No handler found for %s", endpointName)
		}
	}
}
