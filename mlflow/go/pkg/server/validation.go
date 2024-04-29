package server

import (
	"strconv"

	"github.com/go-playground/validator/v10"
)

func NewValidator() *validator.Validate {
	validate := validator.New()

	// Verify that the input string is a positive integer.
	validate.RegisterValidation("positiveInteger", func(fl validator.FieldLevel) bool {
		valueStr := fl.Field().String()
		value, err := strconv.Atoi(valueStr)
		if err != nil {
			return false
		}
		return value > 0
	})

	return validate
}
