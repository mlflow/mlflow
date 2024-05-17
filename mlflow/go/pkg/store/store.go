package store

import (
	"github.com/mlflow/mlflow/mlflow/go/pkg/contract"
	"github.com/mlflow/mlflow/mlflow/go/pkg/protos"
)

type MlflowStore interface {
	// Get an experiment by the experiment ID.
	// The experiment should contain the linked tags.
	GetExperiment(id string) (*protos.Experiment, *contract.Error)

	CreateExperiment(input *protos.CreateExperiment) (string, *contract.Error)

	SearchRuns(
		experimentIDs []string,
		filter *string,
		runViewType protos.ViewType,
		maxResults int,
		orderBy []string,
		pageToken *string,
	) (runs []*protos.Run, nextPageToken *string, err *contract.Error) // TODO: not sure if this should be something more straightforward.
}
