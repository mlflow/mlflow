package model

// Metric mapped from table <metrics>
type Metric struct {
	Key       string  `db:"key"       gorm:"column:key;primaryKey"`
	Value     float64 `db:"value"     gorm:"column:value;primaryKey"`
	Timestamp int64   `db:"timestamp" gorm:"column:timestamp;primaryKey"`
	RunUUID   string  `db:"run_uuid"  gorm:"column:run_uuid;primaryKey"`
	Step      int64   `db:"step"      gorm:"column:step;primaryKey"`
	IsNan     bool    `db:"is_nan"    gorm:"column:is_nan;primaryKey"`
}
