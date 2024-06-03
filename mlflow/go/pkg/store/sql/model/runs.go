package model

import (
	"strings"

	"github.com/mlflow/mlflow/mlflow/go/pkg/protos"
	"github.com/mlflow/mlflow/mlflow/go/pkg/utils"
)

// Run mapped from table <runs>.
type Run struct {
	ID             *string `db:"run_uuid"         gorm:"column:run_uuid;primaryKey"`
	Name           *string `db:"name"             gorm:"column:name"`
	SourceType     *string `db:"source_type"      gorm:"column:source_type"`
	SourceName     *string `db:"source_name"      gorm:"column:source_name"`
	EntryPointName *string `db:"entry_point_name" gorm:"column:entry_point_name"`
	UserID         *string `db:"user_id"          gorm:"column:user_id"`
	Status         *string `db:"status"           gorm:"column:status"`
	StartTime      *int64  `db:"start_time"       gorm:"column:start_time"`
	EndTime        *int64  `db:"end_time"         gorm:"column:end_time"`
	SourceVersion  *string `db:"source_version"   gorm:"column:source_version"`
	LifecycleStage *string `db:"lifecycle_stage"  gorm:"column:lifecycle_stage"`
	ArtifactURI    *string `db:"artifact_uri"     gorm:"column:artifact_uri"`
	ExperimentID   *int32  `db:"experiment_id"    gorm:"column:experiment_id"`
	DeletedTime    *int64  `db:"deleted_time"     gorm:"column:deleted_time"`
	Params         []Param
	Tags           []Tag
	Metrics        []Metric
	LatestMetrics  []LatestMetric
	Inputs         []Input `gorm:"foreignKey:DestinationID"`
}

func RunStatusToProto(status *string) *protos.RunStatus {
	if status == nil {
		return nil
	}

	if protoStatus, ok := protos.RunStatus_value[strings.ToUpper(*status)]; ok {
		return (*protos.RunStatus)(&protoStatus)
	}

	return nil
}

func (r Run) ToProto() *protos.Run {
	info := &protos.RunInfo{
		RunId:          r.ID,
		RunUuid:        r.ID,
		RunName:        r.Name,
		ExperimentId:   utils.ConvertInt32PointerToStringPointer(r.ExperimentID),
		UserId:         r.UserID,
		Status:         RunStatusToProto(r.Status),
		StartTime:      r.StartTime,
		EndTime:        r.EndTime,
		ArtifactUri:    r.ArtifactURI,
		LifecycleStage: r.LifecycleStage,
	}

	metrics := make([]*protos.Metric, 0, len(r.Metrics))
	for _, metric := range r.LatestMetrics {
		metrics = append(metrics, metric.ToProto())
	}

	params := make([]*protos.Param, 0, len(r.Params))
	for _, param := range r.Params {
		params = append(params, param.ToProto())
	}

	tags := make([]*protos.RunTag, 0, len(r.Tags))
	for _, tag := range r.Tags {
		tags = append(tags, tag.ToProto())
	}

	data := &protos.RunData{
		Metrics: metrics,
		Params:  params,
		Tags:    tags,
	}

	datasetInputs := make([]*protos.DatasetInput, 0, len(r.Inputs))
	for _, input := range r.Inputs {
		datasetInputs = append(datasetInputs, input.ToProto())
	}

	inputs := &protos.RunInputs{
		DatasetInputs: datasetInputs,
	}

	return &protos.Run{
		Info:   info,
		Data:   data,
		Inputs: inputs,
	}
}
