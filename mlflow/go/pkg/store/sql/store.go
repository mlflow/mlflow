package sql

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net/url"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/gofiber/fiber/v2/log"
	"github.com/ncruces/go-sqlite3/gormlite"
	"gorm.io/driver/mysql"
	"gorm.io/driver/postgres"
	"gorm.io/driver/sqlserver"
	"gorm.io/gorm"
	"gorm.io/gorm/clause"
	"gorm.io/gorm/logger"

	"github.com/mlflow/mlflow/mlflow/go/pkg/config"
	"github.com/mlflow/mlflow/mlflow/go/pkg/contract"
	"github.com/mlflow/mlflow/mlflow/go/pkg/protos"
	"github.com/mlflow/mlflow/mlflow/go/pkg/query"
	"github.com/mlflow/mlflow/mlflow/go/pkg/query/parser"
	"github.com/mlflow/mlflow/mlflow/go/pkg/store"
	"github.com/mlflow/mlflow/mlflow/go/pkg/store/sql/model"
	"github.com/mlflow/mlflow/mlflow/go/pkg/utils"

	_ "github.com/ncruces/go-sqlite3/embed" // embed sqlite3 driver
)

type Store struct {
	config *config.Config
	db     *gorm.DB
}

func (s Store) GetExperiment(id string) (*protos.Experiment, *contract.Error) {
	idInt, err := strconv.ParseInt(id, 10, 32)
	if err != nil {
		return nil, contract.NewErrorWith(
			protos.ErrorCode_INVALID_PARAMETER_VALUE,
			fmt.Sprintf("failed to convert experiment id %q to int", id),
			err,
		)
	}

	//nolint:exhaustruct
	experiment := model.Experiment{ID: utils.PtrTo(int32(idInt))}
	if err := s.db.Preload("Tags").First(&experiment).Error; err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, contract.NewError(
				protos.ErrorCode_RESOURCE_DOES_NOT_EXIST,
				fmt.Sprintf("No Experiment with id=%d exists", idInt),
			)
		}

		return nil, contract.NewErrorWith(
			protos.ErrorCode_INTERNAL_ERROR,
			"failed to get experiment",
			err,
		)
	}

	return experiment.ToProto(), nil
}

func (s Store) CreateExperiment(input *protos.CreateExperiment) (string, *contract.Error) {
	experiment := model.NewExperimentFromProto(input)

	if err := s.db.Transaction(func(transaction *gorm.DB) error {
		if err := transaction.Create(&experiment).Error; err != nil {
			return fmt.Errorf("failed to insert experiment: %w", err)
		}

		if utils.IsNilOrEmptyString(experiment.ArtifactLocation) {
			artifactLocation, err := url.JoinPath(s.config.DefaultArtifactRoot, strconv.Itoa(int(*experiment.ID)))
			if err != nil {
				return fmt.Errorf("failed to join artifact location: %w", err)
			}
			experiment.ArtifactLocation = &artifactLocation
			if err := transaction.Model(&experiment).UpdateColumn("artifact_location", artifactLocation).Error; err != nil {
				return fmt.Errorf("failed to update experiment artifact location: %w", err)
			}
		}

		return nil
	}); err != nil {
		if errors.Is(err, gorm.ErrDuplicatedKey) {
			return "", contract.NewError(
				protos.ErrorCode_RESOURCE_ALREADY_EXISTS,
				fmt.Sprintf("Experiment(name=%s) already exists.", *experiment.Name),
			)
		}

		return "", contract.NewErrorWith(protos.ErrorCode_INTERNAL_ERROR, "failed to create experiment", err)
	}

	return strconv.Itoa(int(*experiment.ID)), nil
}

type PageToken struct {
	Offset int32 `json:"offset"`
}

var runOrder = regexp.MustCompile(
	`^(attribute|metric|param|tag)s?\.("[^"]+"|` + "`[^`]+`" + `|[\w\.]+)(?i:\s+(ASC|DESC))?$`,
)

func getLifecyleStages(runViewType protos.ViewType) []model.LifecycleStage {
	switch runViewType {
	case protos.ViewType_ACTIVE_ONLY:
		return []model.LifecycleStage{
			model.LifecycleStageActive,
		}
	case protos.ViewType_DELETED_ONLY:
		return []model.LifecycleStage{
			model.LifecycleStageDeleted,
		}
	case protos.ViewType_ALL:
		return []model.LifecycleStage{
			model.LifecycleStageActive,
			model.LifecycleStageDeleted,
		}
	}

	return []model.LifecycleStage{
		model.LifecycleStageActive,
		model.LifecycleStageDeleted,
	}
}

func getOffset(pageToken string) (int, *contract.Error) {
	if pageToken != "" {
		var token PageToken
		if err := json.NewDecoder(
			base64.NewDecoder(
				base64.StdEncoding,
				strings.NewReader(pageToken),
			),
		).Decode(&token); err != nil {
			return 0, contract.NewErrorWith(
				protos.ErrorCode_INVALID_PARAMETER_VALUE,
				fmt.Sprintf("invalid page_token: %q", pageToken),
				err,
			)
		}

		return int(token.Offset), nil
	}

	return 0, nil
}

//nolint:exhaustruct,funlen,cyclop,gocognit
func applyFilters(store *Store, transaction *gorm.DB, filter string) *contract.Error {
	filterConditions, err := query.ParseFilter(filter)
	if err != nil {
		return contract.NewErrorWith(
			protos.ErrorCode_INVALID_PARAMETER_VALUE,
			"error parsing search filter",
			err,
		)
	}

	log.Debugf("Filter conditions: %#v", filterConditions)

	for index, clause := range filterConditions {
		var kind any

		key := clause.Key
		comparison := strings.ToUpper(clause.Operator.String())
		value := clause.Value

		switch clause.Identifier {
		case parser.Metric:
			kind = &model.LatestMetric{}
		case parser.Parameter:
			kind = &model.Param{}
		case parser.Tag:
			kind = &model.Tag{}
		case parser.Dataset:
			kind = &model.Dataset{}
		case parser.Attribute:
			kind = nil
		}

		// Treat "attributes.run_name == <value>" as "tags.`mlflow.runName` == <value>".
		// The name column in the runs table is empty for runs logged in MLflow <= 1.29.0.
		if key == "run_name" {
			kind = &model.Tag{}
			key = "mlflow.runName"
		}

		isSqliteAndILike := store.db.Dialector.Name() == "sqlite" && comparison == "ILIKE"
		table := fmt.Sprintf("filter_%d", index)

		switch {
		case kind == nil:
			if isSqliteAndILike {
				key = fmt.Sprintf("LOWER(runs.%s)", key)
				comparison = "LIKE"

				if str, ok := value.(string); ok {
					value = strings.ToLower(str)
				}

				transaction.Where(fmt.Sprintf("%s %s ?", key, comparison), value)
			} else {
				transaction.Where(fmt.Sprintf("runs.%s %s ?", key, comparison), value)
			}
		case clause.Identifier == parser.Dataset && key == "context":
			// SELECT *
			// FROM runs
			// JOIN (
			//   SELECT inputs.destination_id AS run_uuid
			//   FROM inputs
			//   JOIN input_tags
			//   ON inputs.input_uuid = input_tags.input_uuid
			//   AND input_tags.name = 'mlflow.data.context'
			//   AND input_tags.value %s ?
			//   WHERE inputs.destination_type = 'RUN'
			// ) AS filter_0
			// ON runs.run_uuid = filter_0.run_uuid
			valueColumn := "input_tags.value "
			if isSqliteAndILike {
				valueColumn = "LOWER(input_tags.value) "

				if str, ok := value.(string); ok {
					value = strings.ToLower(str)
				}
			}

			transaction.Joins(
				fmt.Sprintf("JOIN (?) AS %s ON runs.run_uuid = %s.run_uuid", table, table),
				store.db.Select("inputs.destination_id AS run_uuid").
					Joins(
						"JOIN input_tags ON inputs.input_uuid = input_tags.input_uuid"+
							" AND input_tags.name = 'mlflow.data.context'"+
							" AND "+valueColumn+comparison+" ?",
						value,
					).
					Where("inputs.destination_type = 'RUN'").
					Model(&model.Input{}),
			)
		case clause.Identifier == parser.Dataset:
			// add join with datasets
			// JOIN (
			// 		SELECT "experiment_id", key
			//		FROM datasests
			//		WHERE key comparison value
			// ) AS filter_0 ON runs.experiment_id = dataset.experiment_id
			//
			// columns: name, digest, context
			where := key + " " + comparison + " ?"
			if isSqliteAndILike {
				where = "LOWER(" + key + ") LIKE ?"

				if str, ok := value.(string); ok {
					value = strings.ToLower(str)
				}
			}

			transaction.Joins(
				fmt.Sprintf("JOIN (?) AS %s ON runs.experiment_id = %s.experiment_id", table, table),
				store.db.Select("experiment_id", key).Where(where, value).Model(kind),
			)
		default:
			where := fmt.Sprintf("value %s ?", comparison)
			if isSqliteAndILike {
				where = "LOWER(value) LIKE ?"

				if str, ok := value.(string); ok {
					value = strings.ToLower(str)
				}
			}

			transaction.Joins(
				fmt.Sprintf("JOIN (?) AS %s ON runs.run_uuid = %s.run_uuid", table, table),
				store.db.Select("run_uuid", "value").Where("key = ?", key).Where(where, value).Model(kind),
			)
		}
	}

	return nil
}

//nolint:exhaustruct, funlen, cyclop
func applyOrderBy(store *Store, transaction *gorm.DB, orderBy []string) *contract.Error {
	startTimeOrder := false

	for index, orderByClause := range orderBy {
		components := runOrder.FindStringSubmatch(orderByClause)
		log.Debugf("Components: %#v", components)
		//nolint:mnd
		if len(components) < 3 {
			return contract.NewError(
				protos.ErrorCode_INVALID_PARAMETER_VALUE,
				"invalid order by clause: "+orderByClause,
			)
		}

		column := strings.Trim(components[2], "`\"")

		var kind any

		switch components[1] {
		case "attribute":
			if column == "start_time" {
				startTimeOrder = true
			}
		case "metric":
			kind = &model.LatestMetric{}
		case "param":
			kind = &model.Param{}
		case "tag":
			kind = &model.Tag{}
		default:
			return contract.NewError(
				protos.ErrorCode_INVALID_PARAMETER_VALUE,
				fmt.Sprintf(
					"invalid entity type '%s'. Valid values are ['metric', 'parameter', 'tag', 'attribute']",
					components[1],
				),
			)
		}

		if kind != nil {
			table := fmt.Sprintf("order_%d", index)
			transaction.Joins(
				fmt.Sprintf("LEFT OUTER JOIN (?) AS %s ON runs.run_uuid = %s.run_uuid", table, table),
				store.db.Select("run_uuid", "value").Where("key = ?", column).Model(kind),
			)

			column = table + ".value"
		}

		transaction.Order(clause.OrderByColumn{
			Column: clause.Column{
				Name: column,
			},
			Desc: len(components) == 4 && strings.ToUpper(components[3]) == "DESC",
		})
	}

	if !startTimeOrder {
		transaction.Order("runs.start_time DESC")
	}

	transaction.Order("runs.run_uuid")

	return nil
}

func mkNextPageToken(runLength, maxResults, offset int) (*string, *contract.Error) {
	var nextPageToken *string

	if runLength == maxResults {
		var token strings.Builder
		if err := json.NewEncoder(
			base64.NewEncoder(base64.StdEncoding, &token),
		).Encode(PageToken{
			Offset: int32(offset + maxResults),
		}); err != nil {
			return nil, contract.NewErrorWith(
				protos.ErrorCode_INTERNAL_ERROR,
				"error encoding 'nextPageToken' value",
				err,
			)
		}

		nextPageToken = utils.PtrTo(token.String())
	}

	return nextPageToken, nil
}

func (s Store) SearchRuns(
	experimentIDs []string, filter string,
	runViewType protos.ViewType, maxResults int, orderBy []string, pageToken string,
) (*store.PagedList[*protos.Run], *contract.Error) {
	// ViewType
	lifecyleStages := getLifecyleStages(runViewType)
	transaction := s.db.Where("runs.experiment_id IN ?", experimentIDs).Where("runs.lifecycle_stage IN ?", lifecyleStages)

	// MaxResults
	transaction.Limit(maxResults)

	// PageToken
	offset, contractError := getOffset(pageToken)
	if contractError != nil {
		return nil, contractError
	}

	transaction.Offset(offset)

	// Filter
	contractError = applyFilters(&s, transaction, filter)
	if contractError != nil {
		return nil, contractError
	}

	// OrderBy
	contractError = applyOrderBy(&s, transaction, orderBy)
	if contractError != nil {
		return nil, contractError
	}

	// Actual query
	var runs []model.Run

	transaction.Preload("LatestMetrics").Preload("Params").Preload("Tags").
		Preload("Inputs", "inputs.destination_type = 'RUN'").
		Preload("Inputs.Dataset").Preload("Inputs.Tags").Find(&runs)

	if transaction.Error != nil {
		return nil, contract.NewErrorWith(
			protos.ErrorCode_INTERNAL_ERROR,
			"Failed to query search runs",
			transaction.Error,
		)
	}

	contractRuns := make([]*protos.Run, 0, len(runs))
	for _, run := range runs {
		contractRuns = append(contractRuns, run.ToProto())
	}

	nextPageToken, contractError := mkNextPageToken(len(runs), maxResults, offset)
	if contractError != nil {
		return nil, contractError
	}

	return &store.PagedList[*protos.Run]{
		Items:         contractRuns,
		NextPageToken: nextPageToken,
	}, nil
}

//nolint:exhaustruct
func (s Store) DeleteExperiment(id string) *contract.Error {
	idInt, err := strconv.ParseInt(id, 10, 32)
	if err != nil {
		return contract.NewErrorWith(
			protos.ErrorCode_INVALID_PARAMETER_VALUE,
			fmt.Sprintf("failed to convert experiment id (%s) to int", id),
			err,
		)
	}

	if err := s.db.Transaction(func(transaction *gorm.DB) error {
		// Update experiment
		uex := transaction.Model(&model.Experiment{}).
			Where("experiment_id = ?", idInt).
			Updates(&model.Experiment{
				LifecycleStage: utils.PtrTo(string(model.LifecycleStageDeleted)),
				LastUpdateTime: utils.PtrTo(time.Now().UnixMilli()),
			})

		if uex.Error != nil {
			return fmt.Errorf("failed to update experiment (%d) during delete: %w", idInt, err)
		}

		if uex.RowsAffected != 1 {
			return gorm.ErrRecordNotFound
		}

		// Update runs
		if err := transaction.Model(&model.Run{}).
			Where("experiment_id = ?", idInt).
			Updates(&model.Run{
				LifecycleStage: utils.PtrTo(string(model.LifecycleStageDeleted)),
				DeletedTime:    utils.PtrTo(time.Now().UnixMilli()),
			}).Error; err != nil {
			return fmt.Errorf("failed to update runs during delete: %w", err)
		}

		return nil
	}); err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return contract.NewError(
				protos.ErrorCode_RESOURCE_DOES_NOT_EXIST,
				fmt.Sprintf("No Experiment with id=%d exists", idInt),
			)
		}

		return contract.NewErrorWith(
			protos.ErrorCode_INTERNAL_ERROR,
			"failed to delete experiment",
			err,
		)
	}

	return nil
}

const batchSize = 100

type conflictedParam struct {
	RunID    string
	Key      string
	OldValue string
	NewValue string
}

//nolint:exhaustruct
func (s Store) logParamsWithTransaction(
	transaction *gorm.DB, runID string, params []*protos.Param,
) *contract.Error {
	existingParams := make([]model.Param, 0)

	err := transaction.Where("run_uuid = ?", runID).Find(&existingParams).Error
	if err != nil {
		return contract.NewErrorWith(
			protos.ErrorCode_INTERNAL_ERROR,
			fmt.Sprintf("failed to get existing params for %q", runID),
			err,
		)
	}

	conflictedParameters := make([]conflictedParam, 0)
	runParams := make([]model.Param, 0, len(params))

	for _, param := range params {
		isConflicting := false

		for _, existingParam := range existingParams {
			if param.GetKey() == *existingParam.Key && param.GetValue() != *existingParam.Value {
				conflictedParameters = append(conflictedParameters, conflictedParam{
					RunID:    runID,
					Key:      param.GetKey(),
					OldValue: *existingParam.Value,
					NewValue: param.GetValue(),
				})
				isConflicting = true

				break
			}
		}

		if !isConflicting {
			runParams = append(runParams, model.NewParamFromProto(runID, param))
		}
	}

	if len(conflictedParameters) > 0 {
		return contract.NewError(
			protos.ErrorCode_INVALID_PARAMETER_VALUE,
			fmt.Sprintf(
				"changing param values is not allowed. Params were already\n logged='%v' for run ID=%q",
				conflictedParameters, runID,
			),
		)
	}

	if err := transaction.Clauses(clause.OnConflict{
		Columns:   []clause.Column{{Name: "run_uuid"}, {Name: "key"}},
		DoNothing: true,
	}).CreateInBatches(runParams, batchSize).Error; err != nil {
		return contract.NewErrorWith(
			protos.ErrorCode_INTERNAL_ERROR,
			fmt.Sprintf("error creating params in batch for run_uuid %q", runID),
			err,
		)
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

//nolint:exhaustruct
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

//nolint:exhaustruct,cyclop
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
			nextLatestMetricsMap[*metric.Key] = metric.AsLatestMetric()
		case !found && alreadyPresent && isNewerMetric(metric, nextLatestMetric):
			// there is no row in the database but the metric is present twice
			// and we need to take the latest one from the batch.
			nextLatestMetricsMap[*metric.Key] = metric.AsLatestMetric()
		case found && isNewerMetric(metric, latestMetric):
			// compare with the row in the database
			nextLatestMetricsMap[*metric.Key] = metric.AsLatestMetric()
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

//nolint:exhaustruct
func (s Store) logMetricsWithTransaction(
	transaction *gorm.DB, runID string, metrics []*protos.Metric,
) *contract.Error {
	var lifecycleStage model.LifecycleStage

	err := transaction.
		Model(&model.Run{}).
		Where("run_uuid = ?", runID).
		Select("lifecycle_stage").
		Scan(&lifecycleStage).
		Error
	if err != nil {
		return contract.NewErrorWith(
			protos.ErrorCode_INTERNAL_ERROR,
			fmt.Sprintf(
				"the run %q must be in the 'active' state.\nCurrent state is %s",
				runID,
				lifecycleStage,
			),
			err,
		)
	}

	if lifecycleStage != model.LifecycleStageActive {
		return contract.NewError(
			protos.ErrorCode_INVALID_PARAMETER_VALUE,
			fmt.Sprintf("run %q does not exist", runID),
		)
	}

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

//nolint:exhaustruct
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

func (s Store) LogBatch(
	runID string, metrics []*protos.Metric, params []*protos.Param, tags []*protos.RunTag,
) *contract.Error {
	err := s.db.Transaction(func(transaction *gorm.DB) error {
		err := s.setTagsWithTransaction(transaction, runID, tags)
		if err != nil {
			return contract.NewErrorWith(
				protos.ErrorCode_INTERNAL_ERROR,
				"set tags inside log batch failed",
				err,
			)
		}

		contractError := s.logParamsWithTransaction(transaction, runID, params)
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
			"log batch transaction failed",
			err,
		)
	}

	return nil
}

func NewSQLStore(config *config.Config) (*Store, error) {
	uri, err := url.Parse(config.StoreURL)
	if err != nil {
		return nil, fmt.Errorf("failed to parse store URL %q: %w", config.StoreURL, err)
	}

	var dialector gorm.Dialector

	uri.Scheme, _, _ = strings.Cut(uri.Scheme, "+")

	switch uri.Scheme {
	case "mssql":
		uri.Scheme = "sqlserver"
		dialector = sqlserver.Open(uri.String())
	case "mysql":
		dialector = mysql.Open(fmt.Sprintf("%s@tcp(%s)%s?%s", uri.User, uri.Host, uri.Path, uri.RawQuery))
	case "postgres", "postgresql":
		dialector = postgres.Open(uri.String())
	case "sqlite":
		uri.Scheme = ""
		uri.Path = uri.Path[1:]
		dialector = gormlite.Open(uri.String())
	default:
		return nil, fmt.Errorf("unsupported store URL scheme %q", uri.Scheme) //nolint:err113
	}
	//nolint:exhaustruct
	database, err := gorm.Open(dialector, &gorm.Config{
		TranslateError: true,
		Logger:         logger.Default.LogMode(logger.Info),
	})
	if err != nil {
		return nil, fmt.Errorf("failed to connect to database %q: %w", uri.String(), err)
	}

	return &Store{config: config, db: database}, nil
}
