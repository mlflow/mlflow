package server

import (
	"fmt"

	"github.com/mlflow/mlflow/mlflow/go/pkg/contract"
	"github.com/mlflow/mlflow/mlflow/go/pkg/protos"
)

type MlflowService struct{}

// CreateExperiment implements MlflowService.
func (g MlflowService) CreateExperiment(input *protos.CreateExperiment) (*protos.CreateExperiment_Response, *contract.MlflowError) {
	return &protos.CreateExperiment_Response{}, &contract.MlflowError{ErrorCode: protos.ErrorCode_NOT_IMPLEMENTED}
}

// CreateRun implements MlflowService.
func (g MlflowService) CreateRun(input *protos.CreateRun) (*protos.CreateRun_Response, *contract.MlflowError) {
	return &protos.CreateRun_Response{}, &contract.MlflowError{ErrorCode: protos.ErrorCode_NOT_IMPLEMENTED}
}

// DeleteExperiment implements MlflowService.
func (g MlflowService) DeleteExperiment(input *protos.DeleteExperiment) (*protos.DeleteExperiment_Response, *contract.MlflowError) {
	return &protos.DeleteExperiment_Response{}, &contract.MlflowError{ErrorCode: protos.ErrorCode_NOT_IMPLEMENTED}
}

// DeleteRun implements MlflowService.
func (g MlflowService) DeleteRun(input *protos.DeleteRun) (*protos.DeleteRun_Response, *contract.MlflowError) {
	return &protos.DeleteRun_Response{}, &contract.MlflowError{ErrorCode: protos.ErrorCode_NOT_IMPLEMENTED}
}

// DeleteTag implements MlflowService.
func (g MlflowService) DeleteTag(input *protos.DeleteTag) (*protos.DeleteTag_Response, *contract.MlflowError) {
	return &protos.DeleteTag_Response{}, &contract.MlflowError{ErrorCode: protos.ErrorCode_NOT_IMPLEMENTED}
}

func strPtr(s string) *string {
	return &s
}

func int64Ptr(i int64) *int64 {
	return &i
}

// GetExperiment implements MlflowService.
func (g MlflowService) GetExperiment(input *protos.GetExperiment) (*protos.GetExperiment_Response, *contract.MlflowError) {
	fmt.Printf("GetExperiment for %s\n", *input.ExperimentId)

	experiment := &protos.Experiment{
		ExperimentId:     strPtr("1"),
		Name:             strPtr("Default"),
		ArtifactLocation: strPtr("/tmp"),
		LifecycleStage:   strPtr("active"),
		LastUpdateTime:   int64Ptr(2),
		CreationTime:     int64Ptr(1),
		Tags:             make([]*protos.ExperimentTag, 0),
	}

	response := protos.GetExperiment_Response{
		Experiment: experiment,
	}

	return &response, nil
}

// GetExperimentByName implements MlflowService.
func (g MlflowService) GetExperimentByName(input *protos.GetExperimentByName) (*protos.GetExperimentByName_Response, *contract.MlflowError) {
	return &protos.GetExperimentByName_Response{}, &contract.MlflowError{ErrorCode: protos.ErrorCode_NOT_IMPLEMENTED}
}

// GetMetricHistory implements MlflowService.
func (g MlflowService) GetMetricHistory(input *protos.GetMetricHistory) (*protos.GetMetricHistory_Response, *contract.MlflowError) {
	return &protos.GetMetricHistory_Response{}, &contract.MlflowError{ErrorCode: protos.ErrorCode_NOT_IMPLEMENTED}
}

// GetMetricHistoryBulkInterval implements MlflowService.
func (g MlflowService) GetMetricHistoryBulkInterval(input *protos.GetMetricHistoryBulkInterval) (*protos.GetMetricHistoryBulkInterval_Response, *contract.MlflowError) {
	return &protos.GetMetricHistoryBulkInterval_Response{}, &contract.MlflowError{ErrorCode: protos.ErrorCode_NOT_IMPLEMENTED}
}

// GetRun implements MlflowService.
func (g MlflowService) GetRun(input *protos.GetRun) (*protos.GetRun_Response, *contract.MlflowError) {
	return &protos.GetRun_Response{}, &contract.MlflowError{ErrorCode: protos.ErrorCode_NOT_IMPLEMENTED}
}

// ListArtifacts implements MlflowService.
func (g MlflowService) ListArtifacts(input *protos.ListArtifacts) (*protos.ListArtifacts_Response, *contract.MlflowError) {
	return &protos.ListArtifacts_Response{}, &contract.MlflowError{ErrorCode: protos.ErrorCode_NOT_IMPLEMENTED}
}

// LogBatch implements MlflowService.
func (g MlflowService) LogBatch(input *protos.LogBatch) (*protos.LogBatch_Response, *contract.MlflowError) {
	return &protos.LogBatch_Response{}, &contract.MlflowError{ErrorCode: protos.ErrorCode_NOT_IMPLEMENTED}
}

// LogInputs implements MlflowService.
func (g MlflowService) LogInputs(input *protos.LogInputs) (*protos.LogInputs_Response, *contract.MlflowError) {
	return &protos.LogInputs_Response{}, &contract.MlflowError{ErrorCode: protos.ErrorCode_NOT_IMPLEMENTED}
}

// LogMetric implements MlflowService.
func (g MlflowService) LogMetric(input *protos.LogMetric) (*protos.LogMetric_Response, *contract.MlflowError) {
	return &protos.LogMetric_Response{}, &contract.MlflowError{ErrorCode: protos.ErrorCode_NOT_IMPLEMENTED}
}

// LogModel implements MlflowService.
func (g MlflowService) LogModel(input *protos.LogModel) (*protos.LogModel_Response, *contract.MlflowError) {
	return &protos.LogModel_Response{}, &contract.MlflowError{ErrorCode: protos.ErrorCode_NOT_IMPLEMENTED}
}

// LogParam implements MlflowService.
func (g MlflowService) LogParam(input *protos.LogParam) (*protos.LogParam_Response, *contract.MlflowError) {
	return &protos.LogParam_Response{}, &contract.MlflowError{ErrorCode: protos.ErrorCode_NOT_IMPLEMENTED}
}

// RestoreExperiment implements MlflowService.
func (g MlflowService) RestoreExperiment(input *protos.RestoreExperiment) (*protos.RestoreExperiment_Response, *contract.MlflowError) {
	return &protos.RestoreExperiment_Response{}, &contract.MlflowError{ErrorCode: protos.ErrorCode_NOT_IMPLEMENTED}
}

// RestoreRun implements MlflowService.
func (g MlflowService) RestoreRun(input *protos.RestoreRun) (*protos.RestoreRun_Response, *contract.MlflowError) {
	return &protos.RestoreRun_Response{}, &contract.MlflowError{ErrorCode: protos.ErrorCode_NOT_IMPLEMENTED}
}

// SearchExperiments implements MlflowService.
func (g MlflowService) SearchExperiments(input *protos.SearchExperiments) (*protos.SearchExperiments_Response, *contract.MlflowError) {
	return &protos.SearchExperiments_Response{}, &contract.MlflowError{ErrorCode: protos.ErrorCode_NOT_IMPLEMENTED}
}

// SearchRuns implements MlflowService.
func (g MlflowService) SearchRuns(input *protos.SearchRuns) (*protos.SearchRuns_Response, *contract.MlflowError) {
	return &protos.SearchRuns_Response{}, &contract.MlflowError{ErrorCode: protos.ErrorCode_NOT_IMPLEMENTED}
}

// SetExperimentTag implements MlflowService.
func (g MlflowService) SetExperimentTag(input *protos.SetExperimentTag) (*protos.SetExperimentTag_Response, *contract.MlflowError) {
	return &protos.SetExperimentTag_Response{}, &contract.MlflowError{ErrorCode: protos.ErrorCode_NOT_IMPLEMENTED}
}

// SetTag implements MlflowService.
func (g MlflowService) SetTag(input *protos.SetTag) (*protos.SetTag_Response, *contract.MlflowError) {
	return &protos.SetTag_Response{}, &contract.MlflowError{ErrorCode: protos.ErrorCode_NOT_IMPLEMENTED}
}

// UpdateExperiment implements MlflowService.
func (g MlflowService) UpdateExperiment(input *protos.UpdateExperiment) (*protos.UpdateExperiment_Response, *contract.MlflowError) {
	return &protos.UpdateExperiment_Response{}, &contract.MlflowError{ErrorCode: protos.ErrorCode_NOT_IMPLEMENTED}
}

// UpdateRun implements MlflowService.
func (g MlflowService) UpdateRun(input *protos.UpdateRun) (*protos.UpdateRun_Response, *contract.MlflowError) {
	return &protos.UpdateRun_Response{}, &contract.MlflowError{ErrorCode: protos.ErrorCode_NOT_IMPLEMENTED}
}

var (
	mlflowService          contract.MlflowService = MlflowService{}
	modelRegistryService   contract.ModelRegistryService
	mlflowArtifactsService contract.MlflowArtifactsService
)
