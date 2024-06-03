package server

import (
	"fmt"
	"net/url"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/mlflow/mlflow/mlflow/go/pkg/config"
	"github.com/mlflow/mlflow/mlflow/go/pkg/contract"
	"github.com/mlflow/mlflow/mlflow/go/pkg/protos"
	"github.com/mlflow/mlflow/mlflow/go/pkg/store"
	"github.com/mlflow/mlflow/mlflow/go/pkg/store/sql"
)

type MlflowService struct {
	config *config.Config
	store  store.MlflowStore
}

// CreateExperiment implements MlflowService.
func (m MlflowService) CreateExperiment(input *protos.CreateExperiment) (
	*protos.CreateExperiment_Response, *contract.Error,
) {
	if input.GetArtifactLocation() != "" {
		artifactLocation := strings.TrimRight(input.GetArtifactLocation(), "/")

		// We don't check the validation here as this was already covered in the validator.
		url, _ := url.Parse(artifactLocation)
		switch url.Scheme {
		case "file", "":
			path, err := filepath.Abs(url.Path)
			if err != nil {
				return nil, contract.NewError(
					protos.ErrorCode_INVALID_PARAMETER_VALUE,
					fmt.Sprintf("error getting absolute path: %v", err),
				)
			}

			if runtime.GOOS == "windows" {
				url.Scheme = "file"
				path = "/" + strings.ReplaceAll(path, "\\", "/")
			}

			url.Path = path
			artifactLocation = url.String()
		}

		input.ArtifactLocation = &artifactLocation
	}

	experimentID, err := m.store.CreateExperiment(input)
	if err != nil {
		return nil, err
	}

	response := protos.CreateExperiment_Response{
		ExperimentId: &experimentID,
	}

	return &response, nil
}

// GetExperiment implements MlflowService.
func (m MlflowService) GetExperiment(input *protos.GetExperiment) (*protos.GetExperiment_Response, *contract.Error) {
	experiment, cErr := m.store.GetExperiment(input.GetExperimentId())
	if cErr != nil {
		return nil, cErr
	}

	response := protos.GetExperiment_Response{
		Experiment: experiment,
	}

	return &response, nil
}

func (m MlflowService) SearchRuns(input *protos.SearchRuns) (*protos.SearchRuns_Response, *contract.Error) {
	var runViewType protos.ViewType
	if input.RunViewType == nil {
		runViewType = protos.ViewType_ALL
	} else {
		runViewType = input.GetRunViewType()
	}

	maxResults := int(input.GetMaxResults())

	page, err := m.store.SearchRuns(
		input.GetExperimentIds(),
		input.GetFilter(),
		runViewType,
		maxResults,
		input.GetOrderBy(),
		input.GetPageToken(),
	)
	if err != nil {
		return nil, contract.NewError(protos.ErrorCode_INTERNAL_ERROR, fmt.Sprintf("error getting runs: %v", err))
	}

	response := protos.SearchRuns_Response{
		Runs:          page.Items,
		NextPageToken: page.NextPageToken,
	}

	return &response, nil
}

func (m MlflowService) DeleteExperiment(
	input *protos.DeleteExperiment,
) (*protos.DeleteExperiment_Response, *contract.Error) {
	err := m.store.DeleteExperiment(input.GetExperimentId())
	if err != nil {
		return nil, err
	}

	return &protos.DeleteExperiment_Response{}, nil
}

var (
	modelRegistryService   contract.ModelRegistryService
	mlflowArtifactsService contract.MlflowArtifactsService
)

func NewMlflowService(config *config.Config) (*MlflowService, error) {
	store, err := sql.NewSQLStore(config)
	if err != nil {
		return nil, fmt.Errorf("could not create new sql store: %w", err)
	}

	return &MlflowService{
		config: config,
		store:  store,
	}, nil
}
