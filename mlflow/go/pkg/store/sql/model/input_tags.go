package model

import "github.com/mlflow/mlflow/mlflow/go/pkg/protos"

// InputTag mapped from table <input_tags>.
type InputTag struct {
	InputID *string `gorm:"column:input_uuid;primaryKey"`
	Key     *string `gorm:"column:name;primaryKey"`
	Value   *string `gorm:"column:value;not null"`
}

func (i *InputTag) ToProto() *protos.InputTag {
	return &protos.InputTag{
		Key:   i.Key,
		Value: i.Value,
	}
}
