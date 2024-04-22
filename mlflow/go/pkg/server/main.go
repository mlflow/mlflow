package server

import (
	"fmt"
	"path/filepath"

	"github.com/gofiber/fiber/v2"
	"github.com/gofiber/fiber/v2/middleware/proxy"
	"github.com/mlflow/mlflow/mlflow/go/pkg/protos"
)

type GlowMlflowService struct{}

// CreateExperiment implements MlflowService.
func (g GlowMlflowService) CreateExperiment(input *protos.CreateExperiment) (protos.CreateExperiment_Response, *MlflowError) {
	return protos.CreateExperiment_Response{}, &MlflowError{ErrorCode: protos.ErrorCode_NOT_IMPLEMENTED}
}

// CreateRun implements MlflowService.
func (g GlowMlflowService) CreateRun(input *protos.CreateRun) (protos.CreateRun_Response, *MlflowError) {
	return protos.CreateRun_Response{}, &MlflowError{ErrorCode: protos.ErrorCode_NOT_IMPLEMENTED}
}

// DeleteExperiment implements MlflowService.
func (g GlowMlflowService) DeleteExperiment(input *protos.DeleteExperiment) (protos.DeleteExperiment_Response, *MlflowError) {
	return protos.DeleteExperiment_Response{}, &MlflowError{ErrorCode: protos.ErrorCode_NOT_IMPLEMENTED}
}

// DeleteRun implements MlflowService.
func (g GlowMlflowService) DeleteRun(input *protos.DeleteRun) (protos.DeleteRun_Response, *MlflowError) {
	return protos.DeleteRun_Response{}, &MlflowError{ErrorCode: protos.ErrorCode_NOT_IMPLEMENTED}
}

// DeleteTag implements MlflowService.
func (g GlowMlflowService) DeleteTag(input *protos.DeleteTag) (protos.DeleteTag_Response, *MlflowError) {
	return protos.DeleteTag_Response{}, &MlflowError{ErrorCode: protos.ErrorCode_NOT_IMPLEMENTED}
}

// GetExperiment implements MlflowService.
func (g GlowMlflowService) GetExperiment(input *protos.GetExperiment) (protos.GetExperiment_Response, *MlflowError) {
	return protos.GetExperiment_Response{}, &MlflowError{ErrorCode: protos.ErrorCode_NOT_IMPLEMENTED}
}

// GetExperimentByName implements MlflowService.
func (g GlowMlflowService) GetExperimentByName(input *protos.GetExperimentByName) (protos.GetExperimentByName_Response, *MlflowError) {
	return protos.GetExperimentByName_Response{}, &MlflowError{ErrorCode: protos.ErrorCode_NOT_IMPLEMENTED}
}

// GetMetricHistory implements MlflowService.
func (g GlowMlflowService) GetMetricHistory(input *protos.GetMetricHistory) (protos.GetMetricHistory_Response, *MlflowError) {
	return protos.GetMetricHistory_Response{}, &MlflowError{ErrorCode: protos.ErrorCode_NOT_IMPLEMENTED}
}

// GetMetricHistoryBulkInterval implements MlflowService.
func (g GlowMlflowService) GetMetricHistoryBulkInterval(input *protos.GetMetricHistoryBulkInterval) (protos.GetMetricHistoryBulkInterval_Response, *MlflowError) {
	return protos.GetMetricHistoryBulkInterval_Response{}, &MlflowError{ErrorCode: protos.ErrorCode_NOT_IMPLEMENTED}
}

// GetRun implements MlflowService.
func (g GlowMlflowService) GetRun(input *protos.GetRun) (protos.GetRun_Response, *MlflowError) {
	return protos.GetRun_Response{}, &MlflowError{ErrorCode: protos.ErrorCode_NOT_IMPLEMENTED}
}

// ListArtifacts implements MlflowService.
func (g GlowMlflowService) ListArtifacts(input *protos.ListArtifacts) (protos.ListArtifacts_Response, *MlflowError) {
	return protos.ListArtifacts_Response{}, &MlflowError{ErrorCode: protos.ErrorCode_NOT_IMPLEMENTED}
}

// LogBatch implements MlflowService.
func (g GlowMlflowService) LogBatch(input *protos.LogBatch) (protos.LogBatch_Response, *MlflowError) {
	return protos.LogBatch_Response{}, &MlflowError{ErrorCode: protos.ErrorCode_NOT_IMPLEMENTED}
}

// LogInputs implements MlflowService.
func (g GlowMlflowService) LogInputs(input *protos.LogInputs) (protos.LogInputs_Response, *MlflowError) {
	return protos.LogInputs_Response{}, &MlflowError{ErrorCode: protos.ErrorCode_NOT_IMPLEMENTED}
}

// LogMetric implements MlflowService.
func (g GlowMlflowService) LogMetric(input *protos.LogMetric) (protos.LogMetric_Response, *MlflowError) {
	return protos.LogMetric_Response{}, &MlflowError{ErrorCode: protos.ErrorCode_NOT_IMPLEMENTED}
}

// LogModel implements MlflowService.
func (g GlowMlflowService) LogModel(input *protos.LogModel) (protos.LogModel_Response, *MlflowError) {
	return protos.LogModel_Response{}, &MlflowError{ErrorCode: protos.ErrorCode_NOT_IMPLEMENTED}
}

// LogParam implements MlflowService.
func (g GlowMlflowService) LogParam(input *protos.LogParam) (protos.LogParam_Response, *MlflowError) {
	return protos.LogParam_Response{}, &MlflowError{ErrorCode: protos.ErrorCode_NOT_IMPLEMENTED}
}

// RestoreExperiment implements MlflowService.
func (g GlowMlflowService) RestoreExperiment(input *protos.RestoreExperiment) (protos.RestoreExperiment_Response, *MlflowError) {
	return protos.RestoreExperiment_Response{}, &MlflowError{ErrorCode: protos.ErrorCode_NOT_IMPLEMENTED}
}

// RestoreRun implements MlflowService.
func (g GlowMlflowService) RestoreRun(input *protos.RestoreRun) (protos.RestoreRun_Response, *MlflowError) {
	return protos.RestoreRun_Response{}, &MlflowError{ErrorCode: protos.ErrorCode_NOT_IMPLEMENTED}
}

// SearchExperiments implements MlflowService.
func (g GlowMlflowService) SearchExperiments(input *protos.SearchExperiments) (protos.SearchExperiments_Response, *MlflowError) {
	return protos.SearchExperiments_Response{}, &MlflowError{ErrorCode: protos.ErrorCode_NOT_IMPLEMENTED}
}

// SearchRuns implements MlflowService.
func (g GlowMlflowService) SearchRuns(input *protos.SearchRuns) (protos.SearchRuns_Response, *MlflowError) {
	return protos.SearchRuns_Response{}, &MlflowError{ErrorCode: protos.ErrorCode_NOT_IMPLEMENTED}
}

// SetExperimentTag implements MlflowService.
func (g GlowMlflowService) SetExperimentTag(input *protos.SetExperimentTag) (protos.SetExperimentTag_Response, *MlflowError) {
	return protos.SetExperimentTag_Response{}, &MlflowError{ErrorCode: protos.ErrorCode_NOT_IMPLEMENTED}
}

// SetTag implements MlflowService.
func (g GlowMlflowService) SetTag(input *protos.SetTag) (protos.SetTag_Response, *MlflowError) {
	return protos.SetTag_Response{}, &MlflowError{ErrorCode: protos.ErrorCode_NOT_IMPLEMENTED}
}

// UpdateExperiment implements MlflowService.
func (g GlowMlflowService) UpdateExperiment(input *protos.UpdateExperiment) (protos.UpdateExperiment_Response, *MlflowError) {
	return protos.UpdateExperiment_Response{}, &MlflowError{ErrorCode: protos.ErrorCode_NOT_IMPLEMENTED}
}

// UpdateRun implements MlflowService.
func (g GlowMlflowService) UpdateRun(input *protos.UpdateRun) (protos.UpdateRun_Response, *MlflowError) {
	return protos.UpdateRun_Response{}, &MlflowError{ErrorCode: protos.ErrorCode_NOT_IMPLEMENTED}
}

var (
	mlflowService          MlflowService = GlowMlflowService{}
	modelRegistryService   ModelRegistryService
	mlflowArtifactsService MlflowArtifactsService
)

type LaunchConfiguration struct {
	Port         int
	PythonPort   int
	StaticFolder string
}

func Launch(configuration LaunchConfiguration) {
	app := fiber.New()

	registerMlflowServiceRoutes(mlflowService, app)
	registerModelRegistryServiceRoutes(modelRegistryService, app)
	registerMlflowArtifactsServiceRoutes(mlflowArtifactsService, app)

	app.Static("/static-files", configuration.StaticFolder)
	app.Get("/", func(c *fiber.Ctx) error {
		return c.SendFile(filepath.Join(configuration.StaticFolder, "index.html"))
	})

	app.Use(proxy.BalancerForward([]string{fmt.Sprintf("http://127.0.0.1:%d", configuration.PythonPort)}))

	app.Listen(fmt.Sprintf(":%d", configuration.Port))
}
