package main

import "fmt"

const (
	MaxResultsPerPage = 50000 // TODO: fasttrack ml has 1000000 here.
)

var validations map[string]string = map[string]string{
	"GetExperiment_ExperimentId":        "required,stringAsPositiveInteger",
	"CreateExperiment_Name":             "required",
	"CreateExperiment_ArtifactLocation": "omitempty,uriWithoutFragmentsOrParamsOrDotDotInQuery",
	"SearchRuns_RunViewType":            "required,oneof=ACTIVE ACTIVE_ONLY DELETED_ONLY",
	"SearchRuns_MaxResults":             fmt.Sprintf("lte=%d", MaxResultsPerPage),
}
