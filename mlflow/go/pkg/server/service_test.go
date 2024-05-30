package server

import (
	"testing"

	"github.com/mlflow/mlflow/mlflow/go/pkg/contract"
	"github.com/mlflow/mlflow/mlflow/go/pkg/protos"
	"github.com/mlflow/mlflow/mlflow/go/pkg/store"
)

type FakeStore struct{}

func (f FakeStore) GetExperiment(_ string) (*protos.Experiment, *contract.Error) {
	return nil, nil
}

func (f FakeStore) CreateExperiment(_ *protos.CreateExperiment) (string, *contract.Error) {
	return "", nil
}

func (f FakeStore) SearchRuns(
	_ []string,
	_ string,
	_ protos.ViewType,
	_ int,
	_ []string,
	_ string,
) (pagedList *store.PagedList[*protos.Run], err *contract.Error) {
	return nil, nil
}

func (f FakeStore) DeleteExperiment(_ string) *contract.Error {
	return nil
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
			if input.GetArtifactLocation() == scenario.input {
				t.Errorf("expected artifact location to be absolute, got %s", input.GetArtifactLocation())
			}
		})
	}
}
