package sql

import (
	"errors"
	"fmt"
	"math"

	"gorm.io/gorm"
	"gorm.io/gorm/clause"

	"github.com/mlflow/mlflow/mlflow/go/pkg/contract"
	"github.com/mlflow/mlflow/mlflow/go/pkg/protos"
	"github.com/mlflow/mlflow/mlflow/go/pkg/store/sql/model"
)

func checkRunIsActive(transaction *gorm.DB, runID string) *contract.Error {
	var lifecycleStage model.LifecycleStage

	err := transaction.
		Model(&model.Run{}).
		Where("run_uuid = ?", runID).
		Select("lifecycle_stage").
		Scan(&lifecycleStage).
		Error
	if err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return contract.NewError(
				protos.ErrorCode_RESOURCE_DOES_NOT_EXIST,
				fmt.Sprintf("Run with id=%s not found", runID),
			)
		}

		return contract.NewErrorWith(
			protos.ErrorCode_INTERNAL_ERROR,
			fmt.Sprintf(
				"failed to get lifecycle stage for run %q",
				runID,
			),
			err,
		)
	}

	if lifecycleStage != model.LifecycleStageActive {
		return contract.NewError(
			protos.ErrorCode_INVALID_PARAMETER_VALUE,
			fmt.Sprintf(
				"The run %s must be in the 'active' state.\n"+
					"Current state is %v.",
				runID,
				lifecycleStage,
			),
		)
	}

	return nil
}

func (s Store) setTagsWithTransaction(
	transaction *gorm.DB, runID string, tags []*protos.RunTag,
) error {
	runColumns := make(map[string]interface{})

	for _, tag := range tags {
		switch tag.GetKey() {
		case "mlflow.user":
			runColumns["user_id"] = tag.GetValue()
		case "mlflow.runName":
			runColumns["name"] = tag.GetValue()
		}
	}

	if len(runColumns) != 0 {
		err := transaction.
			Model(&model.Run{}).
			Where("run_uuid = ?", runID).
			UpdateColumns(runColumns).Error
		if err != nil {
			return fmt.Errorf("failed to update run columns: %w", err)
		}
	}

	runTags := make([]model.Tag, 0, len(tags))

	for _, tag := range tags {
		runTags = append(runTags, model.NewTagFromProto(runID, tag))
	}

	if err := transaction.Clauses(clause.OnConflict{
		UpdateAll: true,
	}).CreateInBatches(runTags, batchSize).Error; err != nil {
		return fmt.Errorf("failed to create tags for run %q: %w", runID, err)
	}

	return nil
}

const batchSize = 100

func verifyBatchParamsInserts(
	transaction *gorm.DB, runID string, deduplicatedParamsMap map[string]string,
) *contract.Error {
	keys := make([]string, 0, len(deduplicatedParamsMap))
	for key := range deduplicatedParamsMap {
		keys = append(keys, key)
	}

	var existingParams []model.Param

	err := transaction.
		Model(&model.Param{}).
		Select("key, value").
		Where("run_uuid = ?", runID).
		Where("key IN ?", keys).
		Find(&existingParams).Error
	if err != nil {
		return contract.NewErrorWith(
			protos.ErrorCode_INTERNAL_ERROR,
			fmt.Sprintf(
				"failed to get existing params to check if duplicates for run_id %q",
				runID,
			),
			err)
	}

	for _, existingParam := range existingParams {
		if currentValue, ok := deduplicatedParamsMap[*existingParam.Key]; ok && currentValue != *existingParam.Value {
			return contract.NewError(
				protos.ErrorCode_INVALID_PARAMETER_VALUE,
				fmt.Sprintf(
					"Changing param values is not allowed. "+
						"Params with key=%q was already logged "+
						"with value=%q for run ID=%q. "+
						"Attempted logging new value %q",
					*existingParam.Key,
					*existingParam.Value,
					runID,
					currentValue,
				),
			)
		}
	}

	return nil
}

func (s Store) logParamsWithTransaction(
	transaction *gorm.DB, runID string, params []*protos.Param,
) *contract.Error {
	deduplicatedParamsMap := make(map[string]string, len(params))
	deduplicatedParams := make([]model.Param, 0, len(deduplicatedParamsMap))

	for _, param := range params {
		oldValue, paramIsPresent := deduplicatedParamsMap[param.GetKey()]
		if paramIsPresent && param.GetValue() != oldValue {
			return contract.NewError(
				protos.ErrorCode_INVALID_PARAMETER_VALUE,
				fmt.Sprintf(
					"Changing param values is not allowed. "+
						"Params with key=%q was already logged "+
						"with value=%q for run ID=%q. "+
						"Attempted logging new value %q",
					param.GetKey(),
					oldValue,
					runID,
					param.GetValue(),
				),
			)
		}

		if !paramIsPresent {
			deduplicatedParamsMap[param.GetKey()] = param.GetValue()
			deduplicatedParams = append(deduplicatedParams, model.NewParamFromProto(runID, param))
		}
	}

	// Try and create all params.
	// Params are unique by (run_uuid, key) so any potentially conflicts will not be inserted.
	err := transaction.
		Clauses(clause.OnConflict{
			Columns:   []clause.Column{{Name: "run_uuid"}, {Name: "key"}},
			DoNothing: true,
		}).
		CreateInBatches(deduplicatedParams, batchSize).Error
	if err != nil {
		return contract.NewErrorWith(
			protos.ErrorCode_INTERNAL_ERROR,
			fmt.Sprintf("error creating params in batch for run_uuid %q", runID),
			err,
		)
	}

	// if there were ignored conflicts, we assert that the values are the same.
	if transaction.RowsAffected != int64(len(params)) {
		contractError := verifyBatchParamsInserts(transaction, runID, deduplicatedParamsMap)
		if contractError != nil {
			return contractError
		}
	}

	return nil
}

func getDistinctMetricKeys(metrics []model.Metric) []string {
	metricKeysMap := make(map[string]any)
	for _, m := range metrics {
		metricKeysMap[*m.Key] = nil
	}

	metricKeys := make([]string, 0, len(metricKeysMap))
	for key := range metricKeysMap {
		metricKeys = append(metricKeys, key)
	}

	return metricKeys
}

func getLatestMetrics(transaction *gorm.DB, runID string, metricKeys []string) ([]model.LatestMetric, error) {
	batchSize := 500
	latestMetrics := make([]model.LatestMetric, 0, len(metricKeys))

	for skip := 0; skip < len(metricKeys); skip += batchSize {
		take := int(math.Max(float64(skip+batchSize), float64(len(metricKeys))))
		if take > len(metricKeys) {
			take = len(metricKeys)
		}

		currentBatch := make([]model.LatestMetric, 0, take-skip)
		keys := metricKeys[skip:take]

		err := transaction.
			Model(&model.LatestMetric{}).
			Where("run_uuid = ?", runID).Where("key IN ?", keys).
			Clauses(clause.Locking{Strength: "UPDATE"}). // https://gorm.io/docs/advanced_query.html#Locking
			Order("run_uuid").
			Order("key").
			Find(&currentBatch).Error
		if err != nil {
			return latestMetrics, fmt.Errorf(
				"failed to get latest metrics for run_uuid %q, skip %d, take %d : %w",
				runID, skip, take, err,
			)
		}

		latestMetrics = append(latestMetrics, currentBatch...)
	}

	return latestMetrics, nil
}

func isNewerMetric(a model.Metric, b model.LatestMetric) bool {
	return *a.Step > *b.Step ||
		(*a.Step == *b.Step && *a.Timestamp > *b.Timestamp) ||
		(*a.Step == *b.Step && *a.Timestamp == *b.Timestamp && *a.Value > *b.Value)
}

//nolint:cyclop
func updateLatestMetricsIfNecessary(transaction *gorm.DB, runID string, metrics []model.Metric) error {
	if len(metrics) == 0 {
		return nil
	}

	metricKeys := getDistinctMetricKeys(metrics)

	latestMetrics, err := getLatestMetrics(transaction, runID, metricKeys)
	if err != nil {
		return fmt.Errorf("failed to get latest metrics for run_uuid %q: %w", runID, err)
	}

	latestMetricsMap := make(map[string]model.LatestMetric, len(latestMetrics))
	for _, m := range latestMetrics {
		latestMetricsMap[*m.Key] = m
	}

	nextLatestMetricsMap := make(map[string]model.LatestMetric, len(metrics))

	for _, metric := range metrics {
		latestMetric, found := latestMetricsMap[*metric.Key]
		nextLatestMetric, alreadyPresent := nextLatestMetricsMap[*metric.Key]

		switch {
		case !found && !alreadyPresent:
			// brand new latest metric
			nextLatestMetricsMap[*metric.Key] = metric.NewLatestMetricFromProto()
		case !found && alreadyPresent && isNewerMetric(metric, nextLatestMetric):
			// there is no row in the database but the metric is present twice
			// and we need to take the latest one from the batch.
			nextLatestMetricsMap[*metric.Key] = metric.NewLatestMetricFromProto()
		case found && isNewerMetric(metric, latestMetric):
			// compare with the row in the database
			nextLatestMetricsMap[*metric.Key] = metric.NewLatestMetricFromProto()
		}
	}

	nextLatestMetrics := make([]model.LatestMetric, 0, len(nextLatestMetricsMap))
	for _, nextLatestMetric := range nextLatestMetricsMap {
		nextLatestMetrics = append(nextLatestMetrics, nextLatestMetric)
	}

	if len(nextLatestMetrics) != 0 {
		if err := transaction.Clauses(clause.OnConflict{
			UpdateAll: true,
		}).Create(nextLatestMetrics).Error; err != nil {
			return fmt.Errorf("failed to upsert latest metrics for run_uuid %q: %w", runID, err)
		}
	}

	return nil
}

func (s Store) logMetricsWithTransaction(
	transaction *gorm.DB, runID string, metrics []*protos.Metric,
) *contract.Error {
	// Duplicate metric values are eliminated
	seenMetrics := make(map[model.Metric]struct{})
	modelMetrics := make([]model.Metric, 0, len(metrics))

	for _, metric := range metrics {
		currentMetric := model.NewMetricFromProto(runID, metric)
		if _, ok := seenMetrics[*currentMetric]; !ok {
			seenMetrics[*currentMetric] = struct{}{}

			modelMetrics = append(modelMetrics, *currentMetric)
		}
	}

	if err := transaction.Clauses(clause.OnConflict{DoNothing: true}).
		CreateInBatches(modelMetrics, batchSize).Error; err != nil {
		return contract.NewErrorWith(
			protos.ErrorCode_INTERNAL_ERROR,
			fmt.Sprintf("error creating metrics in batch for run_uuid %q", runID),
			err,
		)
	}

	if err := updateLatestMetricsIfNecessary(transaction, runID, modelMetrics); err != nil {
		return contract.NewErrorWith(
			protos.ErrorCode_INTERNAL_ERROR,
			fmt.Sprintf("error updating latest metrics for run_uuid %q", runID),
			err,
		)
	}

	return nil
}

func (s Store) LogBatch(
	runID string, metrics []*protos.Metric, params []*protos.Param, tags []*protos.RunTag,
) *contract.Error {
	err := s.db.Transaction(func(transaction *gorm.DB) error {
		contractError := checkRunIsActive(transaction, runID)
		if contractError != nil {
			return contractError
		}

		err := s.setTagsWithTransaction(transaction, runID, tags)
		if err != nil {
			return fmt.Errorf("error setting tags for run_id %q: %w", runID, err)
		}

		contractError = s.logParamsWithTransaction(transaction, runID, params)
		if contractError != nil {
			return contractError
		}

		contractError = s.logMetricsWithTransaction(transaction, runID, metrics)
		if contractError != nil {
			return contractError
		}

		return nil
	})
	if err != nil {
		var contractError *contract.Error
		if errors.As(err, &contractError) {
			return contractError
		}

		return contract.NewErrorWith(
			protos.ErrorCode_INTERNAL_ERROR,
			fmt.Sprintf("log batch transaction failed for %q", runID),
			err,
		)
	}

	return nil
}
