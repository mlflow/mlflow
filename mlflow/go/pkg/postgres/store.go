package postgres

import (
	"strconv"

	"github.com/mlflow/mlflow/mlflow/go/pkg/protos"
	"gorm.io/driver/postgres"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"
)

type Store struct {
	db *gorm.DB
}

func (s Store) GetExperiment(id int32) (error, *protos.Experiment) {
	experiment, err := Q.Experiment.Preload(Q.Experiment.ExperimentTags.RelationField).Where(Q.Experiment.ExperimentID.Eq(id)).First()

	if err != nil {
		return err, nil
	} else {
		// Map the experiment to the protos.Experiment
		id := strconv.FormatInt(int64(experiment.ExperimentID), 10)

		tags := make([]*protos.ExperimentTag, len(experiment.ExperimentTags))
		for i, tag := range experiment.ExperimentTags {
			tags[i] = &protos.ExperimentTag{
				Key:   &tag.Key,
				Value: &tag.Value,
			}
		}

		protoExperiment := &protos.Experiment{
			ExperimentId:     &id,
			Name:             &experiment.Name,
			ArtifactLocation: &experiment.ArtifactLocation,
			LifecycleStage:   &experiment.LifecycleStage,
			LastUpdateTime:   &experiment.LastUpdateTime,
			CreationTime:     &experiment.CreationTime,
			Tags:             tags,
		}
		return nil, protoExperiment
	}
}

func NewStore() *Store {
	databaseUrl := "postgresql://postgres:postgres@localhost/postgres"
	db, err := gorm.Open(postgres.Open(databaseUrl), &gorm.Config{
		Logger: logger.Default.LogMode(logger.Info),
	})
	SetDefault(db)
	if err != nil {
		panic(err)
	}
	return &Store{db: db}
}
