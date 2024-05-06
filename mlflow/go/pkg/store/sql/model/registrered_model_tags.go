package model

// RegisteredModelTag mapped from table <registered_model_tags>
type RegisteredModelTag struct {
	Key   string `db:"key"   gorm:"column:key;primaryKey"`
	Value string `db:"value" gorm:"column:value"`
	Name  string `db:"name"  gorm:"column:name;primaryKey"`
}
