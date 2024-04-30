package main

import (
	"gorm.io/driver/postgres"
	"gorm.io/gen"
	"gorm.io/gen/field"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"
)

func main() {
	g := gen.NewGenerator(gen.Config{
		OutPath:      "../../pkg/postgres",
		ModelPkgPath: "../../pkg/postgres/model",
		Mode:         gen.WithoutContext | gen.WithDefaultQuery | gen.WithQueryInterface, // generate mode
	})

	databaseUrl := "postgresql://postgres:postgres@localhost/postgres"
	db, err := gorm.Open(postgres.Open(databaseUrl), &gorm.Config{
		Logger: logger.Default.LogMode(logger.Info),
	})
	if err != nil {
		panic(err)
	}

	g.UseDB(db)

	experiment_tags := g.GenerateModel("experiment_tags")

	experiments := g.GenerateModel("experiments", gen.FieldRelate(field.HasMany, "ExperimentTags", experiment_tags, &field.RelateConfig{GORMTag: field.GormTag{
		"foreignKey": {"experiment_id"},
		"references": {"experiment_id"},
	}}))

	g.ApplyBasic(
		experiment_tags,
		experiments,
	)
	g.Execute()
}
