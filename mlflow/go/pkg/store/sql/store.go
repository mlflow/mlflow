package sql

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"net/url"
	"strconv"
	"strings"

	"github.com/ncruces/go-sqlite3/gormlite"
	"gorm.io/driver/postgres"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"

	"github.com/mlflow/mlflow/mlflow/go/pkg/config"
	"github.com/mlflow/mlflow/mlflow/go/pkg/contract"
	"github.com/mlflow/mlflow/mlflow/go/pkg/protos"
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

	experiment := model.Experiment{ExperimentID: utils.PtrTo(int32(idInt))}
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
			artifactLocation, err := url.JoinPath(s.config.DefaultArtifactRoot, strconv.Itoa(int(*experiment.ExperimentID)))
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
	return strconv.Itoa(int(*experiment.ExperimentID)), nil
}

// TODO: is this the right place?
type LifecycleStage string

const (
	LifecycleStageActive  LifecycleStage = "active"
	LifecycleStageDeleted LifecycleStage = "deleted"
)

// TODO: is this the right place? Keep here for now?
type PageToken struct {
	Offset int32 `json:"offset"`
}

func (s Store) SearchRuns(
	experimentIDs []string,
	filter *string,
	runViewType protos.ViewType,
	maxResults int,
	orderBy []string,
	pageToken *string,
) ([]*protos.Run, *string, *contract.Error) {
	// ViewType
	var lifecyleStages []LifecycleStage
	switch runViewType {
	case protos.ViewType_ACTIVE_ONLY:
		lifecyleStages = []LifecycleStage{
			LifecycleStageActive,
		}
	case protos.ViewType_DELETED_ONLY:
		lifecyleStages = []LifecycleStage{
			LifecycleStageDeleted,
		}
	case protos.ViewType_ALL:
		lifecyleStages = []LifecycleStage{
			LifecycleStageActive,
			LifecycleStageDeleted,
		}
	}

	// TODO: does we use constants for the column names here?
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
			return nil, nil, contract.NewErrorWith(
				protos.ErrorCode_INVALID_PARAMETER_VALUE,
				fmt.Sprintf("invalid page_token: \"%s\"", *pageToken),
				err,
			)
		}
		offset = int(token.Offset)
	}
	tx.Offset(offset)

	// Actual query
	var runs []model.Run
	tx.Preload("LatestMetrics").
		Preload("Params").
		Preload("Tags").
		Find(&runs)

	if tx.Error != nil {
		return nil, nil, contract.NewErrorWith(
			protos.ErrorCode_INTERNAL_ERROR,
			"Failed to query search runs",
			tx.Error,
		)
	}

	runIDs := make([]*string, 0, len(runs))
	for _, run := range runs {
		runIDs = append(runIDs, run.ID)
	}

	type runDataset struct {
		model.Dataset
		RunUUID *string
	}

	var datasets []runDataset
	if err := s.db.
		Joins(
			"JOIN inputs ON source_id = dataset_uuid "+
				"WHERE destination_type = 'RUN' "+
				"AND source_type = 'DATASET' "+
				"AND destination_id IN ?",
			runIDs,
		).
		Select("*", "inputs.destination_id AS run_uuid").
		Model(&model.Dataset{}).
		Find(&datasets).
		Error; err != nil {
		return nil, nil, contract.NewErrorWith(
			protos.ErrorCode_INTERNAL_ERROR,
			"Failed to query datasets for runs",
			err,
		)
	}

	runToDatasets := make(map[string][]*protos.DatasetInput, len(datasets))
	for _, dataset := range datasets {
		if entry, ok := runToDatasets[*dataset.RunUUID]; ok {
			runToDatasets[*dataset.RunUUID] = append(entry, dataset.Dataset.ToProto())
		} else {
			runToDatasets[*dataset.RunUUID] = []*protos.DatasetInput{dataset.ToProto()}
		}
	}

	contractRuns := make([]*protos.Run, 0, len(runs))
	for _, run := range runs {
		runDatasets, ok := runToDatasets[*run.ID]
		if ok {
			contractRuns = append(contractRuns, run.ToProto(runDatasets))
		} else {
			contractRuns = append(contractRuns, run.ToProto(make([]*protos.DatasetInput, 0)))
		}
	}

	return contractRuns, nil, nil
}

func NewSQLStore(config *config.Config) (store.MlflowStore, error) {
	uri, err := url.Parse(config.StoreURL)
	if err != nil {
		return nil, fmt.Errorf("failed to parse store URL %q: %w", config.StoreURL, err)
	}

	var dialector gorm.Dialector
	switch uri.Scheme {
	case "postgres", "postgresql":
		dialector = postgres.Open(config.StoreURL)
	case "sqlite":
		dialector = gormlite.Open(strings.TrimPrefix(config.StoreURL, "sqlite:///"))
	default:
		return nil, fmt.Errorf("unsupported store URL scheme %q", uri.Scheme)
	}
	db, err := gorm.Open(dialector, &gorm.Config{
		TranslateError: true,
		Logger:         logger.Default.LogMode(logger.Info),
	})
	if err != nil {
		return nil, fmt.Errorf("failed to connect to database %q: %w", config.StoreURL, err)
	}

	return &Store{config: config, db: db}, nil
}
