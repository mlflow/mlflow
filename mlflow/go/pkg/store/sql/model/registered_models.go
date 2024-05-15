package model

// RegisteredModel mapped from table <registered_models>
type RegisteredModel struct {
	Name            *string `db:"name"              gorm:"column:name;primaryKey"`
	CreationTime    *int64  `db:"creation_time"     gorm:"column:creation_time"`
	LastUpdatedTime *int64  `db:"last_updated_time" gorm:"column:last_updated_time"`
	Description     *string `db:"description"       gorm:"column:description"`
}
