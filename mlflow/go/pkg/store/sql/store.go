package sql

import (
	"gorm.io/driver/postgres"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"

	"github.com/mlflow/mlflow/mlflow/go/pkg/protos"
	"github.com/mlflow/mlflow/mlflow/go/pkg/store"
	"github.com/mlflow/mlflow/mlflow/go/pkg/store/sql/model"
)

type Store struct {
	db *gorm.DB
}

func (s Store) GetExperiment(id int32) (*protos.Experiment, error) {
	experiment := model.Experiment{ExperimentID: id}
	if err := s.db.Preload("ExperimentTags").First(&experiment).Error; err != nil {
		return nil, err
	}

	return experiment.ToProto(), nil
}

func (s Store) CreateExperiment(input *protos.CreateExperiment) (store.ExperimentId, error) {
	experiment := model.NewExperimentFromProto(input)
	err := s.db.Create(&experiment).Error
	return experiment.ExperimentID, err
}

func NewSqlStore(url string) (store.MlflowStore, error) {
	db, err := gorm.Open(postgres.Open(url), &gorm.Config{
		Logger: logger.Default.LogMode(logger.Info),
	})
	if err != nil {
		return nil, err
	}

	return &Store{db: db}, nil
}
