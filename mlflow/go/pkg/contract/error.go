package contract

import "github.com/mlflow/mlflow/mlflow/go/pkg/protos"

type MlflowError struct {
	ErrorCode protos.ErrorCode
	Message   string
}

func (e *MlflowError) Error() string {
	return "MlflowError"
}
