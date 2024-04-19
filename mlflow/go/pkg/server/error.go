package server

type MlflowError struct {
}

func (e *MlflowError) Error() string {
	return "MlflowError"
}
