package parser_test

import (
	"testing"

	"github.com/mlflow/mlflow/mlflow/go/pkg/query"
)

func TestValidQueries(t *testing.T) {
	samples := []string{
		"metrics.foobar = 40",
	}

	for _, sample := range samples {
		t.Run(sample, func(t *testing.T) {
			_, err := query.ParseFilter(&sample)
			if err != nil {
				t.Errorf("unexpected parse error: %v", err)
			}
		})
	}
}
