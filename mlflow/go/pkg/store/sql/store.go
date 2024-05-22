package sql

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"net/url"
	"regexp"
	"strconv"
	"strings"

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
		return nil, contract.NewError(
			protos.ErrorCode_INVALID_PARAMETER_VALUE,
			fmt.Sprintf("failed to convert experiment id to int: %v", err),
		)
	}

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

	if err := s.db.Transaction(func(tx *gorm.DB) error {
		if err := tx.Create(&experiment).Error; err != nil {
			return fmt.Errorf("failed to insert experiment: %w", err)
		}

		if utils.IsNilOrEmptyString(experiment.ArtifactLocation) {
			artifactLocation, err := url.JoinPath(s.config.DefaultArtifactRoot, strconv.Itoa(int(*experiment.ID)))
			if err != nil {
				return fmt.Errorf("failed to join artifact location: %w", err)
			}
			experiment.ArtifactLocation = &artifactLocation
			if err := tx.Model(&experiment).UpdateColumn("artifact_location", artifactLocation).Error; err != nil {
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

func (s Store) SearchRuns(
	experimentIDs []string,
	filter *string,
	runViewType protos.ViewType,
	maxResults int,
	orderBy []string,
	pageToken *string,
) (*store.PagedList[*protos.Run], *contract.Error) {
	// ViewType
	var lifecyleStages []model.LifecycleStage
	switch runViewType {
	case protos.ViewType_ACTIVE_ONLY:
		lifecyleStages = []model.LifecycleStage{
			model.LifecycleStageActive,
		}
	case protos.ViewType_DELETED_ONLY:
		lifecyleStages = []model.LifecycleStage{
			model.LifecycleStageDeleted,
		}
	case protos.ViewType_ALL:
		lifecyleStages = []model.LifecycleStage{
			model.LifecycleStageActive,
			model.LifecycleStageDeleted,
		}
	}

	tx := s.db.Where("experiment_id IN ?", experimentIDs).Where("lifecycle_stage IN ?", lifecyleStages)

	// MaxResults
	tx.Limit(maxResults)

	// PageToken
	var offset int
	if utils.IsNotNilOrEmptyString(pageToken) {
		var token PageToken
		if err := json.NewDecoder(
			base64.NewDecoder(
				base64.StdEncoding,
				strings.NewReader(*pageToken),
			),
		).Decode(&token); err != nil {
			return nil, contract.NewErrorWith(
				protos.ErrorCode_INVALID_PARAMETER_VALUE,
				fmt.Sprintf("invalid page_token: \"%s\"", *pageToken),
				err,
			)
		}
		offset = int(token.Offset)
	}
	tx.Offset(offset)

	// Filter
	filterConditions, err := query.ParseFilter(filter)
	if err != nil {
		return nil, contract.NewErrorWith(
			protos.ErrorCode_INVALID_PARAMETER_VALUE,
			"error parsing search filter",
			err,
		)
	}
	log.Debugf("Filter conditions: %#v", filterConditions)

	for n, clause := range filterConditions {
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
		}

		if key == "run_name" {
			fmt.Println("yow!")
			kind = &model.Tag{}
			key = "mlflow.runName"
		}

		if kind == nil {
			if s.db.Dialector.Name() == "sqlite" && comparison == "ILIKE" {
				key = fmt.Sprintf("LOWER(runs.%s)", key)
				comparison = "LIKE"
				value = strings.ToLower(value.(string))
				tx.Where(fmt.Sprintf("%s %s ?", key, comparison), value)
			} else {
				tx.Where(fmt.Sprintf("runs.%s %s ?", key, comparison), value)
			}
		} else {
			table := fmt.Sprintf("filter_%d", n)
			where := fmt.Sprintf("value %s ?", comparison)
			if s.db.Dialector.Name() == "sqlite" && comparison == "ILIKE" {
				where = "LOWER(value) LIKE ?"
				value = strings.ToLower(value.(string))
			}
			tx.Joins(
				fmt.Sprintf("JOIN (?) AS %s ON runs.run_uuid = %s.run_uuid", table, table),
				s.db.Select("run_uuid", "value").Where("key = ?", key).Where(where, value).Model(kind),
			)
		}
	}

	// OrderBy
	startTimeOrder := false
	for n, o := range orderBy {
		components := runOrder.FindStringSubmatch(o)
		log.Debugf("Components: %#v", components)
		if len(components) < 3 {
			return nil, contract.NewError(
				protos.ErrorCode_INVALID_PARAMETER_VALUE,
				"invalid order by clause: "+o,
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
			return nil, contract.NewError(
				protos.ErrorCode_INVALID_PARAMETER_VALUE,
				fmt.Sprintf(
					"invalid entity type '%s'. Valid values are ['metric', 'parameter', 'tag', 'attribute']",
					components[1],
				),
			)
		}
		if kind != nil {
			table := fmt.Sprintf("order_%d", n)
			tx.Joins(
				fmt.Sprintf("LEFT OUTER JOIN (?) AS %s ON runs.run_uuid = %s.run_uuid", table, table),
				s.db.Select("run_uuid", "value").Where("key = ?", column).Model(kind),
			)
			column = table + ".value"
		}
		tx.Order(clause.OrderByColumn{
			Column: clause.Column{
				Name: column,
			},
			Desc: len(components) == 4 && strings.ToUpper(components[3]) == "DESC",
		})
	}
	if !startTimeOrder {
		tx.Order("runs.start_time DESC")
	}
	tx.Order("runs.run_uuid")

	// Actual query
	var runs []model.Run
	tx.Preload("LatestMetrics").
		Preload("Params").
		Preload("Tags").
		Preload("Inputs").
		Preload("Inputs.Dataset").
		Preload("Inputs.Tags").
		Find(&runs)

	if tx.Error != nil {
		return nil, contract.NewErrorWith(
			protos.ErrorCode_INTERNAL_ERROR,
			"Failed to query search runs",
			tx.Error,
		)
	}

	contractRuns := make([]*protos.Run, 0, len(runs))
	for _, run := range runs {
		contractRuns = append(contractRuns, run.ToProto())
	}

	var nextPageToken *string
	if len(runs) == maxResults {
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

	return &store.PagedList[*protos.Run]{
		Items:         contractRuns,
		NextPageToken: nextPageToken,
	}, nil
}

func NewSQLStore(config *config.Config) (store.MlflowStore, error) {
	uri, err := url.Parse(config.StoreURL)
	if err != nil {
		return nil, fmt.Errorf("failed to parse store URL %q: %w", config.StoreURL, err)
	}

	uri.Scheme, _, _ = strings.Cut(uri.Scheme, "+")

	var dialector gorm.Dialector
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
		return nil, fmt.Errorf("unsupported store URL scheme %q", uri.Scheme)
	}
	db, err := gorm.Open(dialector, &gorm.Config{
		TranslateError: true,
		Logger:         logger.Default.LogMode(logger.Info),
	})
	if err != nil {
		return nil, fmt.Errorf("failed to connect to database %q: %w", uri.String(), err)
	}

	return &Store{config: config, db: db}, nil
}
