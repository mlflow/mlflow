package main

import (
	"fmt"

	"github.com/mlflow/mlflow/mlflow/go/pkg/protos"
)

var validations = map[string]string{
	"GetExperiment_ExperimentId":        "required,stringAsPositiveInteger",
	"CreateExperiment_Name":             "required",
	"CreateExperiment_ArtifactLocation": "omitempty,uriWithoutFragmentsOrParamsOrDotDotInQuery",
	"SearchRuns_RunViewType":            "omitempty",
	"SearchRuns_MaxResults":             fmt.Sprintf("lte=%d", protos.Default_SearchRuns_MaxResults),
	"DeleteExperiment_ExperimentId":     "required,stringAsPositiveInteger",
	"RunTag_Key":                        "required,max=250,validMetricParamOrTagName,pathIsUnique",
	"RunTag_Value":                      "omitempty,max=5000",
	"Param_Key":                         "required,max=250,validMetricParamOrTagName,pathIsUnique",
	"Param_Value":                       "omitempty,max=6000",
}
