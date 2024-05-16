package model

// AlembicVersion mapped from table <alembic_version>.
type AlembicVersion struct {
	VersionNum *string `db:"version_num" gorm:"column:version_num;primaryKey"`
}

// TableName AlembicVersion's table name.
func (*AlembicVersion) TableName() string {
	return "alembic_version"
}
