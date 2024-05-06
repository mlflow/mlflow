package sql

import (
	"fmt"
	"os"

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

func (s Store) GetExperiment(id int32) (error, *protos.Experiment) {
	experiment := model.Experiment{ExperimentID: id}
	if err := s.db.Preload("ExperimentTags").First(&experiment).Error; err != nil {
		return err, nil
	}

	return nil, experiment.ToProto()
}

func NewSqlStore() (store.MlflowStore, error) {
	databaseUrl := "postgresql://postgres:postgres@localhost/postgres"
	db, err := gorm.Open(postgres.Open(databaseUrl), &gorm.Config{
		Logger: logger.Default.LogMode(logger.Info),
	})
	if err != nil {
		return nil, err
	}

	return &Store{db: db}, nil
}

func writeToFile(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("err opening file: %w", err)
	}
	defer f.Close()

	if _, err := f.Write([]byte("Foobar")); err != nil {
		return fmt.Errorf("err writing to file: %w", err)
	}

	return nil
}
