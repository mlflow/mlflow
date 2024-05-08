package server

import (
	"fmt"
	"strconv"

	"github.com/go-playground/validator/v10"

	"github.com/mlflow/mlflow/mlflow/go/pkg/contract"
	"github.com/mlflow/mlflow/mlflow/go/pkg/protos"
	"github.com/mlflow/mlflow/mlflow/go/pkg/store"
	"github.com/mlflow/mlflow/mlflow/go/pkg/store/sql"
)

type MlflowService struct {
	store     store.MlflowStore
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
	experimentId, err := m.store.CreateExperiment(input)
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

// CreateRun implements MlflowService.
func (m MlflowService) CreateRun(input *protos.CreateRun) (*protos.CreateRun_Response, *contract.MlflowError) {
	return &protos.CreateRun_Response{}, &contract.MlflowError{ErrorCode: protos.ErrorCode_NOT_IMPLEMENTED}
}

// DeleteExperiment implements MlflowService.
func (m MlflowService) DeleteExperiment(input *protos.DeleteExperiment) (*protos.DeleteExperiment_Response, *contract.MlflowError) {
	return &protos.DeleteExperiment_Response{}, &contract.MlflowError{ErrorCode: protos.ErrorCode_NOT_IMPLEMENTED}
}

// DeleteRun implements MlflowService.
func (m MlflowService) DeleteRun(input *protos.DeleteRun) (*protos.DeleteRun_Response, *contract.MlflowError) {
	return &protos.DeleteRun_Response{}, &contract.MlflowError{ErrorCode: protos.ErrorCode_NOT_IMPLEMENTED}
}

// DeleteTag implements MlflowService.
func (m MlflowService) DeleteTag(input *protos.DeleteTag) (*protos.DeleteTag_Response, *contract.MlflowError) {
	return &protos.DeleteTag_Response{}, &contract.MlflowError{ErrorCode: protos.ErrorCode_NOT_IMPLEMENTED}
}

// GetExperiment implements MlflowService.
func (m MlflowService) GetExperiment(input *protos.GetExperiment) (*protos.GetExperiment_Response, *contract.MlflowError) {
	id, err := strconv.Atoi(*input.ExperimentId)
	if err != nil {
		return nil, &contract.MlflowError{
			ErrorCode: protos.ErrorCode_INVALID_PARAMETER_VALUE,
		}
	}

	experiment, err := m.store.GetExperiment(int32(id))
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

// GetExperimentByName implements MlflowService.
func (m MlflowService) GetExperimentByName(input *protos.GetExperimentByName) (*protos.GetExperimentByName_Response, *contract.MlflowError) {
	return &protos.GetExperimentByName_Response{}, &contract.MlflowError{ErrorCode: protos.ErrorCode_NOT_IMPLEMENTED}
}

// GetMetricHistory implements MlflowService.
func (m MlflowService) GetMetricHistory(input *protos.GetMetricHistory) (*protos.GetMetricHistory_Response, *contract.MlflowError) {
	return &protos.GetMetricHistory_Response{}, &contract.MlflowError{ErrorCode: protos.ErrorCode_NOT_IMPLEMENTED}
}

// GetMetricHistoryBulkInterval implements MlflowService.
func (m MlflowService) GetMetricHistoryBulkInterval(input *protos.GetMetricHistoryBulkInterval) (*protos.GetMetricHistoryBulkInterval_Response, *contract.MlflowError) {
	return &protos.GetMetricHistoryBulkInterval_Response{}, &contract.MlflowError{ErrorCode: protos.ErrorCode_NOT_IMPLEMENTED}
}

// GetRun implements MlflowService.
func (m MlflowService) GetRun(input *protos.GetRun) (*protos.GetRun_Response, *contract.MlflowError) {
	return &protos.GetRun_Response{}, &contract.MlflowError{ErrorCode: protos.ErrorCode_NOT_IMPLEMENTED}
}

// ListArtifacts implements MlflowService.
func (m MlflowService) ListArtifacts(input *protos.ListArtifacts) (*protos.ListArtifacts_Response, *contract.MlflowError) {
	return &protos.ListArtifacts_Response{}, &contract.MlflowError{ErrorCode: protos.ErrorCode_NOT_IMPLEMENTED}
}

// LogBatch implements MlflowService.
func (m MlflowService) LogBatch(input *protos.LogBatch) (*protos.LogBatch_Response, *contract.MlflowError) {
	return &protos.LogBatch_Response{}, &contract.MlflowError{ErrorCode: protos.ErrorCode_NOT_IMPLEMENTED}
}

// LogInputs implements MlflowService.
func (m MlflowService) LogInputs(input *protos.LogInputs) (*protos.LogInputs_Response, *contract.MlflowError) {
	return &protos.LogInputs_Response{}, &contract.MlflowError{ErrorCode: protos.ErrorCode_NOT_IMPLEMENTED}
}

// LogMetric implements MlflowService.
func (m MlflowService) LogMetric(input *protos.LogMetric) (*protos.LogMetric_Response, *contract.MlflowError) {
	return &protos.LogMetric_Response{}, &contract.MlflowError{ErrorCode: protos.ErrorCode_NOT_IMPLEMENTED}
}

// LogModel implements MlflowService.
func (m MlflowService) LogModel(input *protos.LogModel) (*protos.LogModel_Response, *contract.MlflowError) {
	return &protos.LogModel_Response{}, &contract.MlflowError{ErrorCode: protos.ErrorCode_NOT_IMPLEMENTED}
}

// LogParam implements MlflowService.
func (m MlflowService) LogParam(input *protos.LogParam) (*protos.LogParam_Response, *contract.MlflowError) {
	return &protos.LogParam_Response{}, &contract.MlflowError{ErrorCode: protos.ErrorCode_NOT_IMPLEMENTED}
}

// RestoreExperiment implements MlflowService.
func (m MlflowService) RestoreExperiment(input *protos.RestoreExperiment) (*protos.RestoreExperiment_Response, *contract.MlflowError) {
	return &protos.RestoreExperiment_Response{}, &contract.MlflowError{ErrorCode: protos.ErrorCode_NOT_IMPLEMENTED}
}

// RestoreRun implements MlflowService.
func (m MlflowService) RestoreRun(input *protos.RestoreRun) (*protos.RestoreRun_Response, *contract.MlflowError) {
	return &protos.RestoreRun_Response{}, &contract.MlflowError{ErrorCode: protos.ErrorCode_NOT_IMPLEMENTED}
}

// SearchExperiments implements MlflowService.
func (m MlflowService) SearchExperiments(input *protos.SearchExperiments) (*protos.SearchExperiments_Response, *contract.MlflowError) {
	return &protos.SearchExperiments_Response{}, &contract.MlflowError{ErrorCode: protos.ErrorCode_NOT_IMPLEMENTED}
}

// SearchRuns implements MlflowService.
func (m MlflowService) SearchRuns(input *protos.SearchRuns) (*protos.SearchRuns_Response, *contract.MlflowError) {
	return &protos.SearchRuns_Response{}, &contract.MlflowError{ErrorCode: protos.ErrorCode_NOT_IMPLEMENTED}
}

// SetExperimentTag implements MlflowService.
func (m MlflowService) SetExperimentTag(input *protos.SetExperimentTag) (*protos.SetExperimentTag_Response, *contract.MlflowError) {
	return &protos.SetExperimentTag_Response{}, &contract.MlflowError{ErrorCode: protos.ErrorCode_NOT_IMPLEMENTED}
}

// SetTag implements MlflowService.
func (m MlflowService) SetTag(input *protos.SetTag) (*protos.SetTag_Response, *contract.MlflowError) {
	return &protos.SetTag_Response{}, &contract.MlflowError{ErrorCode: protos.ErrorCode_NOT_IMPLEMENTED}
}

// UpdateExperiment implements MlflowService.
func (m MlflowService) UpdateExperiment(input *protos.UpdateExperiment) (*protos.UpdateExperiment_Response, *contract.MlflowError) {
	return &protos.UpdateExperiment_Response{}, &contract.MlflowError{ErrorCode: protos.ErrorCode_NOT_IMPLEMENTED}
}

// UpdateRun implements MlflowService.
func (m MlflowService) UpdateRun(input *protos.UpdateRun) (*protos.UpdateRun_Response, *contract.MlflowError) {
	return &protos.UpdateRun_Response{}, &contract.MlflowError{ErrorCode: protos.ErrorCode_NOT_IMPLEMENTED}
}

var (
	modelRegistryService   contract.ModelRegistryService
	mlflowArtifactsService contract.MlflowArtifactsService
)

func NewMlflowService(storeUrl string) (contract.MlflowService, error) {
	store, err := sql.NewSqlStore(storeUrl)
	if err != nil {
		return nil, err
	}

	validator := NewValidator()
	return MlflowService{
		validator: validator,
		store:     store,
	}, nil
}
