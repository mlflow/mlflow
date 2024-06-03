package server

import (
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strings"

	"github.com/go-playground/validator/v10"
	"github.com/gofiber/fiber/v2"
	"github.com/tidwall/gjson"

	"github.com/mlflow/mlflow/mlflow/go/pkg/contract"
	"github.com/mlflow/mlflow/mlflow/go/pkg/protos"
)

type HTTPRequestParser struct {
	validator *validator.Validate
}

func NewHTTPRequestParser() (*HTTPRequestParser, error) {
	validator, err := NewValidator()
	if err != nil {
		return nil, err
	}

	return &HTTPRequestParser{
		validator: validator,
	}, nil
}

func (p *HTTPRequestParser) ParseBody(ctx *fiber.Ctx, input interface{}) *contract.Error {
	if err := ctx.BodyParser(input); err != nil {
		var unmarshalTypeError *json.UnmarshalTypeError
		if errors.As(err, &unmarshalTypeError) {
			result := gjson.GetBytes(ctx.Body(), unmarshalTypeError.Field)

			value := result.Str
			if value == "" {
				value = result.Raw
			}

			return contract.NewError(
				protos.ErrorCode_INVALID_PARAMETER_VALUE,
				fmt.Sprintf("Invalid value %s for parameter '%s'", value, unmarshalTypeError.Field),
			)
		}

		return contract.NewError(protos.ErrorCode_BAD_REQUEST, err.Error())
	}

	if err := p.validator.Struct(input); err != nil {
		return newErrorFromValidationError(err)
	}

	return nil
}

func (p *HTTPRequestParser) ParseQuery(ctx *fiber.Ctx, input interface{}) *contract.Error {
	if err := ctx.QueryParser(input); err != nil {
		return contract.NewError(protos.ErrorCode_BAD_REQUEST, err.Error())
	}

	if err := p.validator.Struct(input); err != nil {
		return newErrorFromValidationError(err)
	}

	return nil
}

func dereference(value interface{}) interface{} {
	valueOf := reflect.ValueOf(value)
	if valueOf.Kind() == reflect.Ptr {
		if valueOf.IsNil() {
			return ""
		}

		return valueOf.Elem().Interface()
	}

	return value
}

func newErrorFromValidationError(err error) *contract.Error {
	var ve validator.ValidationErrors
	if errors.As(err, &ve) {
		validationErrors := make([]string, 0)

		for _, err := range ve {
			field := err.Field()
			tag := err.Tag()
			value := dereference(err.Value())

			switch tag {
			case "required":
				validationErrors = append(
					validationErrors,
					fmt.Sprintf("Missing value for required parameter '%s'", field),
				)
			case "lte":
				validationErrors = append(
					validationErrors,
					fmt.Sprintf("Invalid value %v for parameter '%s' supplied", value, field),
				)
			default:
				validationErrors = append(
					validationErrors,
					fmt.Sprintf("%s should be %s, got %v", field, tag, value),
				)
			}
		}

		return contract.NewError(protos.ErrorCode_INVALID_PARAMETER_VALUE, strings.Join(validationErrors, ", "))
	}

	return contract.NewError(protos.ErrorCode_INTERNAL_ERROR, err.Error())
}
