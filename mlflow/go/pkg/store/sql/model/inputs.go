package model

// Input mapped from table <inputs>
type Input struct {
	InputUUID       *string `db:"input_uuid"       gorm:"column:input_uuid;not null"`
	SourceType      *string `db:"source_type"      gorm:"column:source_type;primaryKey"`
	SourceID        *string `db:"source_id"        gorm:"column:source_id;primaryKey"`
	DestinationType *string `db:"destination_type" gorm:"column:destination_type;primaryKey"`
	DestinationID   *string `db:"destination_id"   gorm:"column:destination_id;primaryKey"`
}
