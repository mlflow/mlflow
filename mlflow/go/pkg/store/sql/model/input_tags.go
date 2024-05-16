package model

// InputTag mapped from table <input_tags>.
type InputTag struct {
	InputUUID *string `gorm:"column:input_uuid;primaryKey"`
	Name      *string `gorm:"column:name;primaryKey"`
	Value     *string `gorm:"column:value;not null"`
}
