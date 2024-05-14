package model

// ModelVersionTag mapped from table <model_version_tags>
type ModelVersionTag struct {
	Key     *string `db:"key"     gorm:"column:key;primaryKey"`
	Value   *string `db:"value"   gorm:"column:value"`
	Name    *string `db:"name"    gorm:"column:name;primaryKey"`
	Version *int32  `db:"version" gorm:"column:version;primaryKey"`
}
