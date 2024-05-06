package model

// Dataset mapped from table <datasets>
type Dataset struct {
	DatasetUUID       string `db:"dataset_uuid"        gorm:"column:dataset_uuid;not null"`
	ExperimentID      int32  `db:"experiment_id"       gorm:"column:experiment_id;primaryKey"`
	Name              string `db:"name"                gorm:"column:name;primaryKey"`
	Digest            string `db:"digest"              gorm:"column:digest;primaryKey"`
	DatasetSourceType string `db:"dataset_source_type" gorm:"column:dataset_source_type;not null"`
	DatasetSource     string `db:"dataset_source"      gorm:"column:dataset_source;not null"`
	DatasetSchema     string `db:"dataset_schema"      gorm:"column:dataset_schema"`
	DatasetProfile    string `db:"dataset_profile"     gorm:"column:dataset_profile"`
}
