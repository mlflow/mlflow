package model

import "github.com/mlflow/mlflow/mlflow/go/pkg/protos"

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
