package server

import (
	"fmt"
	"net/url"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/mlflow/mlflow/mlflow/go/pkg/config"
	"github.com/mlflow/mlflow/mlflow/go/pkg/contract"
	"github.com/mlflow/mlflow/mlflow/go/pkg/protos"
	"github.com/mlflow/mlflow/mlflow/go/pkg/store"
	"github.com/mlflow/mlflow/mlflow/go/pkg/store/sql"
	"github.com/mlflow/mlflow/mlflow/go/pkg/utils"
)

type MlflowService struct {
	config *config.Config
	store  store.MlflowStore
}

// CreateExperiment implements MlflowService.
func (m MlflowService) CreateExperiment(input *protos.CreateExperiment) (*protos.CreateExperiment_Response, *contract.Error) {
	if utils.IsNotNilOrEmptyString(input.ArtifactLocation) {
		artifactLocation := strings.TrimRight(*input.ArtifactLocation, "/")

		// We don't check the validation here as this was already covered in the validator.
		u, _ := url.Parse(artifactLocation)
		switch u.Scheme {
		case "file", "":
			p, err := filepath.Abs(u.Path)
			if err != nil {
				return nil, contract.NewError(protos.ErrorCode_INVALID_PARAMETER_VALUE, fmt.Sprintf("error getting absolute path: %v", err))
			}
			u.Path = p
			artifactLocation = u.String()
		}

		input.ArtifactLocation = &artifactLocation
	}

	experimentId, err := m.store.CreateExperiment(input)
	if err != nil {
		return nil, contract.NewError(protos.ErrorCode_INTERNAL_ERROR, fmt.Sprintf("error creating experiment: %v", err))
	}

	id := strconv.Itoa(int(experimentId))

	response := protos.CreateExperiment_Response{
		ExperimentId: &id,
	}

	return &response, nil
}

// GetExperiment implements MlflowService.
func (m MlflowService) GetExperiment(input *protos.GetExperiment) (*protos.GetExperiment_Response, *contract.Error) {
	id, err := strconv.Atoi(*input.ExperimentId)
	if err != nil {
		return nil, contract.NewError(protos.ErrorCode_INVALID_PARAMETER_VALUE, fmt.Sprintf("error parsing experiment id: %v", err))
	}

	experiment, err := m.store.GetExperiment(int32(id))
	if err != nil {
		return nil, contract.NewError(protos.ErrorCode_INTERNAL_ERROR, fmt.Sprintf("error getting experiment: %v", err))
	}

	response := protos.GetExperiment_Response{
		Experiment: experiment,
	}

	return &response, nil
}

var (
	modelRegistryService   contract.ModelRegistryService
	mlflowArtifactsService contract.MlflowArtifactsService
)

func NewMlflowService(config *config.Config) (contract.MlflowService, error) {
	store, err := sql.NewSqlStore(config)
	if err != nil {
		return nil, err
	}

	return MlflowService{
		config: config,
		store:  store,
	}, nil
}
