package sql

import (
	"fmt"
	"net/url"
	"strconv"

	"gorm.io/driver/postgres"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"

	"github.com/mlflow/mlflow/mlflow/go/pkg/config"
	"github.com/mlflow/mlflow/mlflow/go/pkg/protos"
	"github.com/mlflow/mlflow/mlflow/go/pkg/store"
	"github.com/mlflow/mlflow/mlflow/go/pkg/store/sql/model"
	"github.com/mlflow/mlflow/mlflow/go/pkg/utils"
)

type Store struct {
	config *config.Config
	db     *gorm.DB
}

func (s Store) GetExperiment(id int32) (*protos.Experiment, error) {
	experiment := model.Experiment{ExperimentID: utils.PtrTo(id)}
	if err := s.db.Preload("ExperimentTags").First(&experiment).Error; err != nil {
		return nil, fmt.Errorf("failed to get experiment with id %d: %w", id, err)
	}

	return experiment.ToProto(), nil
}

func (s Store) CreateExperiment(input *protos.CreateExperiment) (store.ExperimentId, error) {
	experiment := model.NewExperimentFromProto(input)

	if err := s.db.Transaction(func(tx *gorm.DB) error {
		err := tx.Create(&experiment).Error
		if err != nil {
			return err
		}

		if utils.IsNilOrEmptyString(experiment.ArtifactLocation) {
			artifactLocation, err := url.JoinPath(s.config.DefaultArtifactRoot, strconv.Itoa(int(*experiment.ExperimentID)))
			if err != nil {
				return fmt.Errorf("failed to join artifact location: %w", err)
			}
			experiment.ArtifactLocation = &artifactLocation
			return tx.Model(&experiment).UpdateColumn("artifact_location", artifactLocation).Error
		}

		return nil
	}); err != nil {
		return -1, fmt.Errorf("failed to create experiment: %w", err)
	}

	return *experiment.ExperimentID, nil
}

func NewSqlStore(config *config.Config) (store.MlflowStore, error) {
	db, err := gorm.Open(postgres.Open(config.StoreUrl), &gorm.Config{
		Logger: logger.Default.LogMode(logger.Info),
	})
	if err != nil {
		return nil, fmt.Errorf("failed to connect to database %q: %w", config.StoreUrl, err)
	}

	return &Store{config: config, db: db}, nil
}
