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
		filter string,
		runViewType protos.ViewType,
		maxResults int,
		orderBy []string,
		pageToken string,
	) (pagedList *PagedList[*protos.Run], err *contract.Error)

	DeleteExperiment(id string) *contract.Error

	// LogParams(runID string, params []*protos.Param) *contract.Error

	// LogMetrics(runID string, metrics []*protos.Metric) *contract.Error

	// SetTags(runID string, tags []*protos.RunTag) *contract.Error

	LogBatch(
		runID string,
		metrics []*protos.Metric,
		params []*protos.Param,
		tags []*protos.RunTag) *contract.Error
}

type PagedList[T any] struct {
	Items         []T
	NextPageToken *string
}
