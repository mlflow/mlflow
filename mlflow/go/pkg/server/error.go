package server

import "github.com/mlflow/mlflow/mlflow/go/pkg/protos"

type MlflowError struct {
	ErrorCode protos.ErrorCode
}

func (e *MlflowError) Error() string {
	return "MlflowError"
}
