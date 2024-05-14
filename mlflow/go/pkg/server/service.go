package server

import (
	"fmt"
	"net/url"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/go-playground/validator/v10"

	"github.com/mlflow/mlflow/mlflow/go/pkg/config"
	"github.com/mlflow/mlflow/mlflow/go/pkg/contract"
	"github.com/mlflow/mlflow/mlflow/go/pkg/protos"
	"github.com/mlflow/mlflow/mlflow/go/pkg/store"
	"github.com/mlflow/mlflow/mlflow/go/pkg/store/sql"
)

type MlflowService struct {
	config    *config.Config
	Store     store.MlflowStore
	validator *validator.Validate
}

func (m MlflowService) Validate(input interface{}) []string {
	validationErrors := make([]string, 0)
	errs := m.validator.Struct(input)

	if errs != nil {
		for _, err := range errs.(validator.ValidationErrors) {
			field := err.Field()
			tag := err.Tag()
			value := err.Value()
			validationErrors = append(validationErrors, fmt.Sprintf("%s should be %s, got %v", field, tag, value))
		}
	}

	return validationErrors
}

// CreateExperiment implements MlflowService.
func (m MlflowService) CreateExperiment(input *protos.CreateExperiment) (*protos.CreateExperiment_Response, *contract.MlflowError) {
	var artifactLocation string
	if input.ArtifactLocation != nil {
		artifactLocation = strings.TrimRight(*input.ArtifactLocation, "/")
	}

	if artifactLocation != "" {
		// We don't check the validation here as this was already covered in the validator.
		u, _ := url.Parse(artifactLocation)
		switch u.Scheme {
		case "file", "":
			p, err := filepath.Abs(u.Path)
			if err != nil {
				return nil, &contract.MlflowError{
					ErrorCode: protos.ErrorCode_INVALID_PARAMETER_VALUE,
					Message:   "error getting absolute path",
				}
			}
			u.Path = p
			artifactLocation = u.String()
		}
	}
	input.ArtifactLocation = &artifactLocation

	experimentId, err := m.Store.CreateExperiment(input)
	if err != nil {
		return nil, &contract.MlflowError{
			ErrorCode: protos.ErrorCode_INTERNAL_ERROR,
		}
	}

	id := strconv.Itoa(int(experimentId))

	response := protos.CreateExperiment_Response{
		ExperimentId: &id,
	}

	return &response, nil
}

// GetExperiment implements MlflowService.
func (m MlflowService) GetExperiment(input *protos.GetExperiment) (*protos.GetExperiment_Response, *contract.MlflowError) {
	id, err := strconv.Atoi(*input.ExperimentId)
	if err != nil {
		return nil, &contract.MlflowError{
			ErrorCode: protos.ErrorCode_INVALID_PARAMETER_VALUE,
		}
	}

	experiment, err := m.Store.GetExperiment(int32(id))
	if err != nil {
		return nil, &contract.MlflowError{
			ErrorCode: protos.ErrorCode_INTERNAL_ERROR,
		}
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

	validator := NewValidator()
	return MlflowService{
		config:    config,
		validator: validator,
		Store:     store,
	}, nil
}
