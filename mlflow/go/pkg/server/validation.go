package server

import (
	"fmt"
	"net/url"
	"path/filepath"
	"reflect"
	"regexp"
	"strconv"
	"strings"

	"github.com/go-playground/validator/v10"
	"github.com/mlflow/mlflow/mlflow/go/pkg/protos"
)

const QuoteLenght = 2
const MaxEntitiesPerBatch = 1000

// regex for valid param and metric names: may only contain slashes, alphanumerics,
// underscores, periods, dashes, and spaces.
var paramAndMetricNameRegex = regexp.MustCompile(`^[/\w.\- ]*$`)

// regex for valid run IDs: must be an alphanumeric string of length 1 to 256.
var runIDRegex = regexp.MustCompile(`^[a-zA-Z0-9][\w\-]{0,255}$`)

func NewValidator() (*validator.Validate, error) {
	validate := validator.New()

	validate.RegisterTagNameFunc(func(fld reflect.StructField) string {
		name := strings.SplitN(fld.Tag.Get("json"), ",", QuoteLenght)[0]
		// skip if tag key says it should be ignored
		if name == "-" {
			return ""
		}

		return name
	})

	// Verify that the input string is a positive integer.
	if err := validate.RegisterValidation(
		"stringAsPositiveInteger",
		func(fl validator.FieldLevel) bool {
			valueStr := fl.Field().String()
			value, err := strconv.Atoi(valueStr)
			if err != nil {
				return false
			}

			return value > -1
		},
	); err != nil {
		return nil, fmt.Errorf("validation registration for 'stringAsPositiveInteger' failed: %w", err)
	}

	// Verify that the input string, if present, is a Url without fragment or query parameters
	if err := validate.RegisterValidation(
		"uriWithoutFragmentsOrParamsOrDotDotInQuery",
		func(fl validator.FieldLevel) bool {
			valueStr := fl.Field().String()
			if valueStr == "" {
				return true
			}

			u, err := url.Parse(valueStr)
			if err != nil {
				return false
			}

			return u.Fragment == "" && u.RawQuery == "" && !strings.Contains(u.RawQuery, "..")
		},
	); err != nil {
		return nil, fmt.Errorf("validation registration for 'uriWithoutFragmentsOrParamsOrDotDotInQuery' failed: %w", err)
	}

	if err := validate.RegisterValidation(
		"validMetricParamOrTagName",
		func(fl validator.FieldLevel) bool {
			valueStr := fl.Field().String()

			return paramAndMetricNameRegex.MatchString(valueStr)
		},
	); err != nil {
		return nil, fmt.Errorf("validation registration for 'validMetricParamOrTagName' failed: %w", err)
	}

	if err := validate.RegisterValidation(
		"pathIsUnique",
		func(fl validator.FieldLevel) bool {
			valueStr := fl.Field().String()
			norm := filepath.Clean(valueStr)

			return norm != valueStr || norm == "." || strings.HasPrefix(norm, "..") || strings.HasPrefix(norm, "/")
		},
	); err != nil {
		return nil, fmt.Errorf("validation registration for 'validMetricParamOrTagValue' failed: %w", err)
	}

	// unique params in LogBatch
	if err := validate.RegisterValidation(
		"uniqueParams",
		func(fl validator.FieldLevel) bool {
			value := fl.Field()
			params, areParams := value.Interface().([]*protos.Param)
			if !areParams {
				return true
			}

			hasDuplicates := false
			keys := make(map[string]bool, len(params))

			for _, param := range params {
				if _, ok := keys[param.GetKey()]; ok {
					hasDuplicates = true

					break
				}

				keys[param.GetKey()] = true
			}

			return !hasDuplicates
		},
	); err != nil {
		return nil, fmt.Errorf("validation registration for 'uniqueParams' failed: %w", err)
	}

	if err := validate.RegisterValidation(
		"validRunId",
		func(fl validator.FieldLevel) bool {
			valueStr := fl.Field().String()

			return runIDRegex.MatchString(valueStr)
		},
	); err != nil {
		return nil, fmt.Errorf("validation registration for 'validRunId' failed: %w", err)
	}

	// see _validate_batch_log_limits in validation.py
	validate.RegisterStructValidation(func(structLevel validator.StructLevel) {
		logBatch, isLogBatch := structLevel.Current().Interface().(*protos.LogBatch)

		if isLogBatch {
			total := len(logBatch.GetParams()) + len(logBatch.GetMetrics()) + len(logBatch.GetTags())
			if total > MaxEntitiesPerBatch {
				structLevel.ReportError(&logBatch, "metrics, params, and tags", "", "", "")
			}
		}
	},
		//nolint:exhaustruct
		&protos.LogBatch{})

	return validate, nil
}
