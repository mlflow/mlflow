package model

import (
	"strconv"
	"time"

	"github.com/mlflow/mlflow/mlflow/go/pkg/protos"
	"github.com/mlflow/mlflow/mlflow/go/pkg/utils"
)

// Experiment mapped from table <experiments>.
type Experiment struct {
	ExperimentID     *int32          `gorm:"column:experiment_id;primaryKey;autoIncrement:true" json:"experiment_id"`
	Name             *string         `gorm:"column:name;not null"                               json:"name"`
	ArtifactLocation *string         `gorm:"column:artifact_location"                           json:"artifact_location"`
	LifecycleStage   *string         `gorm:"column:lifecycle_stage"                             json:"lifecycle_stage"`
	CreationTime     *int64          `gorm:"column:creation_time"                               json:"creation_time"`
	LastUpdateTime   *int64          `gorm:"column:last_update_time"                            json:"last_update_time"`
	ExperimentTags   []ExperimentTag `gorm:"foreignKey:experiment_id;references:experiment_id"  json:"experiment_tags"`
}

func (e Experiment) ToProto() *protos.Experiment {
	id := strconv.FormatInt(int64(*e.ExperimentID), 10)
	tags := make([]*protos.ExperimentTag, len(e.ExperimentTags))
	for i, tag := range e.ExperimentTags {
		tags[i] = &protos.ExperimentTag{
			Key:   tag.Key,
			Value: tag.Value,
		}
	}

	return &protos.Experiment{
		ExperimentId:     &id,
		Name:             e.Name,
		ArtifactLocation: e.ArtifactLocation,
		LifecycleStage:   e.LifecycleStage,
		CreationTime:     e.CreationTime,
		LastUpdateTime:   e.LastUpdateTime,
		Tags:             tags,
	}
}

func NewExperimentFromProto(proto *protos.CreateExperiment) Experiment {
	tags := make([]ExperimentTag, len(proto.Tags))
	for i, tag := range proto.Tags {
		tags[i] = ExperimentTag{
			Key:   tag.Key,
			Value: tag.Value,
		}
	}

	return Experiment{
		Name:             proto.Name,
		ArtifactLocation: proto.ArtifactLocation,
		LifecycleStage:   utils.PtrTo("active"),
		CreationTime:     utils.PtrTo(time.Now().UnixMilli()),
		LastUpdateTime:   utils.PtrTo(time.Now().UnixMilli()),
		ExperimentTags:   tags,
	}
}
