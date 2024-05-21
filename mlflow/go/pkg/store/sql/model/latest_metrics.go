package model

// LatestMetric mapped from table <latest_metrics>.
type LatestMetric struct {
	Key       *string  `db:"key"       gorm:"column:key;primaryKey"`
	Value     *float64 `db:"value"     gorm:"column:value;not null"`
	Timestamp *int64   `db:"timestamp" gorm:"column:timestamp"`
	Step      *int64   `db:"step"      gorm:"column:step;not null"`
	IsNan     *bool    `db:"is_nan"    gorm:"column:is_nan;not null"`
	RunID     *string  `db:"run_uuid"  gorm:"column:run_uuid;primaryKey"`
}
