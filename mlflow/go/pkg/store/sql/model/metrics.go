package model

import "github.com/mlflow/mlflow/mlflow/go/pkg/protos"

// Metric mapped from table <metrics>.
type Metric struct {
	Key       *string  `db:"key"       gorm:"column:key;primaryKey"`
	Value     *float64 `db:"value"     gorm:"column:value;primaryKey"`
	Timestamp *int64   `db:"timestamp" gorm:"column:timestamp;primaryKey"`
	RunID     *string  `db:"run_uuid"  gorm:"column:run_uuid;primaryKey"`
	Step      *int64   `db:"step"      gorm:"column:step;primaryKey"`
	IsNan     *bool    `db:"is_nan"    gorm:"column:is_nan;primaryKey"`
}

func (m Metric) ToProto() *protos.Metric {
	return &protos.Metric{
		Key:       m.Key,
		Value:     m.Value,
		Timestamp: m.Timestamp,
		Step:      m.Step,
	}
}
