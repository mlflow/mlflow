package main

var validations map[string]string = map[string]string{
	"GetExperiment_ExperimentId":        "required,stringAsPositiveInteger",
	"CreateExperiment_Name":             "required",
	"CreateExperiment_ArtifactLocation": "uriWithoutFragmentsOrParams",
}
