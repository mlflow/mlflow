package contract

import (
	"encoding/json"
	"fmt"

	"github.com/mlflow/mlflow/mlflow/go/pkg/protos"
)

type ErrorCode protos.ErrorCode

func (e ErrorCode) String() string {
	return protos.ErrorCode(e).String()
}

// Custom json marshalling for ErrorCode.
func (e ErrorCode) MarshalJSON() ([]byte, error) {
	//nolint:wrapcheck
	return json.Marshal(e.String())
}

type Error struct {
	Code    ErrorCode `json:"error_code"` //nolint:tagliatelle
	Message string    `json:"message"`
	Inner   error     `json:"-"`
}

func NewError(code protos.ErrorCode, message string) *Error {
	return NewErrorWith(code, message, nil)
}

func NewErrorWith(code protos.ErrorCode, message string, err error) *Error {
	return &Error{
		Code:    ErrorCode(code),
		Message: message,
		Inner:   err,
	}
}

func (e *Error) Error() string {
	msg := fmt.Sprintf("[%s] %s", e.Code.String(), e.Message)
	if e.Inner != nil {
		return fmt.Sprintf("%s: %s", msg, e.Inner)
	}

	return msg
}

func (e *Error) Unwrap() error {
	return e.Inner
}

//nolint:cyclop
func (e *Error) StatusCode() int {
	//nolint:exhaustive,mnd
	switch protos.ErrorCode(e.Code) {
	case protos.ErrorCode_BAD_REQUEST, protos.ErrorCode_INVALID_PARAMETER_VALUE, protos.ErrorCode_RESOURCE_ALREADY_EXISTS:
		return 400
	case protos.ErrorCode_CUSTOMER_UNAUTHORIZED, protos.ErrorCode_UNAUTHENTICATED:
		return 401
	case protos.ErrorCode_PERMISSION_DENIED:
		return 403
	case protos.ErrorCode_ENDPOINT_NOT_FOUND, protos.ErrorCode_NOT_FOUND, protos.ErrorCode_RESOURCE_DOES_NOT_EXIST:
		return 404
	case protos.ErrorCode_ABORTED, protos.ErrorCode_ALREADY_EXISTS, protos.ErrorCode_RESOURCE_CONFLICT:
		return 409
	case protos.ErrorCode_RESOURCE_EXHAUSTED, protos.ErrorCode_RESOURCE_LIMIT_EXCEEDED:
		return 429
	case protos.ErrorCode_CANCELLED:
		return 499
	case protos.ErrorCode_DATA_LOSS, protos.ErrorCode_INTERNAL_ERROR, protos.ErrorCode_INVALID_STATE:
		return 500
	case protos.ErrorCode_NOT_IMPLEMENTED:
		return 501
	case protos.ErrorCode_TEMPORARILY_UNAVAILABLE:
		return 503
	case protos.ErrorCode_DEADLINE_EXCEEDED:
		return 504
	default:
		return 500
	}
}
