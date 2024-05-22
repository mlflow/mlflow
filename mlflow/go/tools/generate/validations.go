package main

import (
	"fmt"

	"github.com/mlflow/mlflow/mlflow/go/pkg/protos"
)

const (
	MaxResultsPerPage = 1000
)

var validations = map[string]string{
	"GetExperiment_ExperimentId":        "required,stringAsPositiveInteger",
	"CreateExperiment_Name":             "required",
	"CreateExperiment_ArtifactLocation": "omitempty,uriWithoutFragmentsOrParamsOrDotDotInQuery",
	"SearchRuns_RunViewType": fmt.Sprintf(
		"omitempty,oneof=%d %d %d",
		protos.ViewType_ALL,
		protos.ViewType_ACTIVE_ONLY,
		protos.ViewType_DELETED_ONLY,
	),
	"SearchRuns_MaxResults": fmt.Sprintf("lte=%d", MaxResultsPerPage),
}
