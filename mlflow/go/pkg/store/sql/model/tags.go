package model

import (
	"github.com/mlflow/mlflow/mlflow/go/pkg/protos"
	"github.com/mlflow/mlflow/mlflow/go/pkg/utils"
)

// Tag mapped from table <tags>.
type Tag struct {
	Key   *string `db:"key"      gorm:"column:key;primaryKey"`
	Value *string `db:"value"    gorm:"column:value"`
	RunID *string `db:"run_uuid" gorm:"column:run_uuid;primaryKey"`
}

func (t Tag) ToProto() *protos.RunTag {
	return &protos.RunTag{
		Key:   t.Key,
		Value: t.Value,
	}
}

func NewTagFromProto(runID string, proto *protos.RunTag) Tag {
	return Tag{
		Key:   utils.PtrTo(proto.GetKey()),
		Value: utils.PtrTo(proto.GetValue()),
		RunID: &runID,
	}
}
