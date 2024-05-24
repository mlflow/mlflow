package main

import (
	"fmt"

	"github.com/mlflow/mlflow/mlflow/go/pkg/contract"
)

var validations = map[string]string{
	"GetExperiment_ExperimentId":        "required,stringAsPositiveInteger",
	"CreateExperiment_Name":             "required",
	"CreateExperiment_ArtifactLocation": "omitempty,uriWithoutFragmentsOrParamsOrDotDotInQuery",
	"SearchRuns_RunViewType":            "omitempty",
	"SearchRuns_MaxResults":             fmt.Sprintf("lte=%d", contract.MaxResultsPerPage),
}
