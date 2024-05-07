package model

// ModelVersion mapped from table <model_versions>
type ModelVersion struct {
	Name            string `db:"name"              gorm:"column:name;primaryKey"`
	Version         int32  `db:"version"           gorm:"column:version;primaryKey"`
	CreationTime    int64  `db:"creation_time"     gorm:"column:creation_time"`
	LastUpdatedTime int64  `db:"last_updated_time" gorm:"column:last_updated_time"`
	Description     string `db:"description"       gorm:"column:description"`
	UserID          string `db:"user_id"           gorm:"column:user_id"`
	CurrentStage    string `db:"current_stage"     gorm:"column:current_stage"`
	Source          string `db:"source"            gorm:"column:source"`
	RunID           string `db:"run_id"            gorm:"column:run_id"`
	Status          string `db:"status"            gorm:"column:status"`
	StatusMessage   string `db:"status_message"    gorm:"column:status_message"`
	RunLink         string `db:"run_link"          gorm:"column:run_link"`
	StorageLocation string `db:"storage_location"  gorm:"column:storage_location"`
}
