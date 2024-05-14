package model

const TableNameExperimentTag = "experiment_tags"

// ExperimentTag mapped from table <experiment_tags>
type ExperimentTag struct {
	Key          *string `db:"key"           gorm:"column:key;primaryKey"`
	Value        *string `db:"value"         gorm:"column:value"`
	ExperimentID *int32  `db:"experiment_id" gorm:"column:experiment_id;primaryKey"`
}
