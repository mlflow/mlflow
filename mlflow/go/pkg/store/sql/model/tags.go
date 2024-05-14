package model

// Tag mapped from table <tags>
type Tag struct {
	Key     *string `db:"key"      gorm:"column:key;primaryKey"`
	Value   *string `db:"value"    gorm:"column:value"`
	RunUUID *string `db:"run_uuid" gorm:"column:run_uuid;primaryKey"`
}
