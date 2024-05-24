package model

import "github.com/mlflow/mlflow/mlflow/go/pkg/protos"

// LatestMetric mapped from table <latest_metrics>.
type LatestMetric struct {
	Key       *string  `db:"key"       gorm:"column:key;primaryKey"`
	Value     *float64 `db:"value"     gorm:"column:value;not null"`
	Timestamp *int64   `db:"timestamp" gorm:"column:timestamp"`
	Step      *int64   `db:"step"      gorm:"column:step;not null"`
	IsNan     *bool    `db:"is_nan"    gorm:"column:is_nan;not null"`
	RunID     *string  `db:"run_uuid"  gorm:"column:run_uuid;primaryKey"`
}

func (lm LatestMetric) ToProto() *protos.Metric {
	return &protos.Metric{
		Key:       lm.Key,
		Value:     lm.Value,
		Timestamp: lm.Timestamp,
		Step:      lm.Step,
	}
}
