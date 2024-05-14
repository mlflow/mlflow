package server

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/go-playground/validator/v10"
	"github.com/gofiber/fiber/v2"
	"github.com/tidwall/gjson"

	"github.com/mlflow/mlflow/mlflow/go/pkg/contract"
	"github.com/mlflow/mlflow/mlflow/go/pkg/protos"
)

type Parser struct {
	validator *validator.Validate
}

func NewParser() *Parser {
	return &Parser{
		validator: NewValidator(),
	}
}

func (p *Parser) ParseBody(ctx *fiber.Ctx, input interface{}) *contract.Error {
	if err := ctx.BodyParser(input); err != nil {
		switch err := err.(type) {
		case *json.UnmarshalTypeError:
			result := gjson.GetBytes(ctx.Body(), err.Field)
			value := result.Str
			if value == "" {
				value = result.Raw
			}
			return contract.NewError(protos.ErrorCode_INVALID_PARAMETER_VALUE, fmt.Sprintf("Invalid value %s for parameter '%s'", value, err.Field))
		default:
			return contract.NewError(protos.ErrorCode_BAD_REQUEST, err.Error())
		}
	}

	if err := p.validator.Struct(input); err != nil {
		return newErrorFromValidationError(err)
	}

	return nil
}

func (p *Parser) ParseQuery(ctx *fiber.Ctx, input interface{}) *contract.Error {
	if err := ctx.QueryParser(input); err != nil {
		return contract.NewError(protos.ErrorCode_BAD_REQUEST, err.Error())
	}

	if err := p.validator.Struct(input); err != nil {
		return newErrorFromValidationError(err)
	}

	return nil
}

func newErrorFromValidationError(err error) *contract.Error {
	errs, ok := err.(validator.ValidationErrors)
	if !ok {
		return contract.NewError(protos.ErrorCode_INTERNAL_ERROR, err.Error())
	}

	validationErrors := make([]string, 0)
	for _, err := range errs {
		field := err.Field()
		tag := err.Tag()
		value := err.Value()
		var vErr string
		switch tag {
		case "required":
			vErr = fmt.Sprintf("Missing value for required parameter '%s'", field)
		default:
			vErr = fmt.Sprintf("%s should be %s, got %v", field, tag, value)
		}
		validationErrors = append(validationErrors, vErr)
	}

	return contract.NewError(protos.ErrorCode_INVALID_PARAMETER_VALUE, strings.Join(validationErrors, ", "))
}
