package server

import (
	"net/url"
	"reflect"
	"strconv"
	"strings"

	"github.com/go-playground/validator/v10"
)

func NewValidator() *validator.Validate {
	validate := validator.New()

	validate.RegisterTagNameFunc(func(fld reflect.StructField) string {
		name := strings.SplitN(fld.Tag.Get("json"), ",", 2)[0]
		// skip if tag key says it should be ignored
		if name == "-" {
			return ""
		}
		return name
	})

	// Verify that the input string is a positive integer.
	validate.RegisterValidation("stringAsPositiveInteger", func(fl validator.FieldLevel) bool {
		valueStr := fl.Field().String()
		value, err := strconv.Atoi(valueStr)
		if err != nil {
			return false
		}
		return value > 0
	})

	// Verify that the input string, if present, is a Url without fragment or query parameters
	validate.RegisterValidation("uriWithoutFragmentsOrParamsOrDotDotInQuery", func(fl validator.FieldLevel) bool {
		valueStr := fl.Field().String()
		if valueStr == "" {
			return true
		}

		u, err := url.Parse(valueStr)
		if err != nil {
			return false
		}

		return u.Fragment == "" && u.RawQuery == "" && !strings.Contains(u.RawQuery, "..")
	})

	return validate
}
