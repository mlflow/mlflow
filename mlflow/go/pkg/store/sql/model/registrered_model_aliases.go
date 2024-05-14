package model

// RegisteredModelAlias mapped from table <registered_model_aliases>
type RegisteredModelAlias struct {
	Alias_  *string `db:"alias"   gorm:"column:alias;primaryKey"`
	Version *int32  `db:"version" gorm:"column:version;not null"`
	Name    *string `db:"name"    gorm:"column:name;primaryKey"`
}
