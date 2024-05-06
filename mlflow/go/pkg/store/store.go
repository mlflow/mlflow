package store

import (
	"github.com/mlflow/mlflow/mlflow/go/pkg/protos"
)

type MlflowStore interface {
	// Get an experiment by the experiment ID.
	// The experiment should contain the linked tags.
	GetExperiment(id int32) (error, *protos.Experiment)
}
