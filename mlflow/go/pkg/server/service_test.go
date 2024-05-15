package server

import (
	"testing"

	"github.com/mlflow/mlflow/mlflow/go/pkg/contract"
	"github.com/mlflow/mlflow/mlflow/go/pkg/protos"
	"github.com/mlflow/mlflow/mlflow/go/pkg/store"
)

type FakeStore struct{}

func (f FakeStore) GetExperiment(id string) (*protos.Experiment, *contract.Error) {
	return nil, nil
}

func (f FakeStore) CreateExperiment(input *protos.CreateExperiment) (string, *contract.Error) {
	return "", nil
}

func toPtr(s string) *string {
	return &s
}

type testRelativeArtifactLocationScenario struct {
	name  string
	input string
}

func TestRelativeArtifactLocation(t *testing.T) {
	scenarios := []testRelativeArtifactLocationScenario{
		{name: "without scheme", input: "../yow"},
		{name: "with file scheme", input: "file:///../yow"},
	}

	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			var store store.MlflowStore = FakeStore{}
			service := MlflowService{
				store: store,
			}
			var input protos.CreateExperiment = protos.CreateExperiment{
				ArtifactLocation: toPtr(scenario.input),
			}
			response, err := service.CreateExperiment(&input)
			if err != nil {
				t.Error("expected create experiment to succeed")
			}
			if response == nil {
				t.Error("expected response to be non-nil")
			}
			if *input.ArtifactLocation == scenario.input {
				t.Errorf("expected artifact location to be absolute, got %s", *input.ArtifactLocation)
			}
		})
	}
}
