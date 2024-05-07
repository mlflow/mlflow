package store

import (
	"github.com/mlflow/mlflow/mlflow/go/pkg/protos"
)

type ExperimentId = int32

type MlflowStore interface {
	// Get an experiment by the experiment ID.
	// The experiment should contain the linked tags.
	GetExperiment(id int32) (*protos.Experiment, error)

	CreateExperiment(input *protos.CreateExperiment) (ExperimentId, error)
}
