package model

import (
	"github.com/mlflow/mlflow/mlflow/go/pkg/protos"
	"github.com/mlflow/mlflow/mlflow/go/pkg/utils"
)

// Param mapped from table <params>.
type Param struct {
	Key   *string `db:"key"      gorm:"column:key;primaryKey"`
	Value *string `db:"value"    gorm:"column:value;not null"`
	RunID *string `db:"run_uuid" gorm:"column:run_uuid;primaryKey"`
}

func (p Param) ToProto() *protos.Param {
	return &protos.Param{
		Key:   p.Key,
		Value: p.Value,
	}
}

func NewParamFromProto(runID string, p *protos.Param) Param {
	return Param{
		Key:   utils.PtrTo(p.GetKey()),
		Value: utils.PtrTo(p.GetValue()),
		RunID: &runID,
	}
}
