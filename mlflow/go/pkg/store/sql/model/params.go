package model

// Param mapped from table <params>.
type Param struct {
	Key     *string `db:"key"      gorm:"column:key;primaryKey"`
	Value   *string `db:"value"    gorm:"column:value;not null"`
	RunUUID *string `db:"run_uuid" gorm:"column:run_uuid;primaryKey"`
}
