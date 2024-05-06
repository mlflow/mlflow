package model

// Run mapped from table <runs>
type Run struct {
	RunUUID        string `db:"run_uuid"         gorm:"column:run_uuid;primaryKey"`
	Name           string `db:"name"             gorm:"column:name"`
	SourceType     string `db:"source_type"      gorm:"column:source_type"`
	SourceName     string `db:"source_name"      gorm:"column:source_name"`
	EntryPointName string `db:"entry_point_name" gorm:"column:entry_point_name"`
	UserID         string `db:"user_id"          gorm:"column:user_id"`
	Status         string `db:"status"           gorm:"column:status"`
	StartTime      int64  `db:"start_time"       gorm:"column:start_time"`
	EndTime        int64  `db:"end_time"         gorm:"column:end_time"`
	SourceVersion  string `db:"source_version"   gorm:"column:source_version"`
	LifecycleStage string `db:"lifecycle_stage"  gorm:"column:lifecycle_stage"`
	ArtifactURI    string `db:"artifact_uri"     gorm:"column:artifact_uri"`
	ExperimentID   int32  `db:"experiment_id"    gorm:"column:experiment_id"`
	DeletedTime    int64  `db:"deleted_time"     gorm:"column:deleted_time"`
}
