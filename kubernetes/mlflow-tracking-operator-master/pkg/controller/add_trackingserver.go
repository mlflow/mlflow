package controller

import (
	"github.com/zmhassan/mlflow-tracking-operator/pkg/controller/trackingserver"
)

func init() {
	// AddToManagerFuncs is a list of functions to create controllers and add them to a manager.
	AddToManagerFuncs = append(AddToManagerFuncs, trackingserver.Add)
}
