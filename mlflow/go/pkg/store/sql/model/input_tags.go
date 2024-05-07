package model

// InputTag mapped from table <input_tags>
type InputTag struct {
	InputUUID string `gorm:"column:input_uuid;primaryKey" json:"input_uuid"`
	Name      string `gorm:"column:name;primaryKey"       json:"name"`
	Value     string `gorm:"column:value;not null"        json:"value"`
}
