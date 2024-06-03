package parser

import (
	"fmt"
	"strings"
)

/*

This is the equivalent of type-checking the untyped tree.
Not every parsed tree is a valid one.

Grammar rule: identifier.key operator value

The rules are:

For identifiers:

identifier.key

Or if only key is passed, the identifier is "attribute"

Identifiers can have aliases.

if the identifier is dataset, the allowed keys are: name, digest and context.

*/

type ValidIdentifier int

const (
	Metric ValidIdentifier = iota
	Parameter
	Tag
	Attribute
	Dataset
)

func (v ValidIdentifier) String() string {
	switch v {
	case Metric:
		return "metric"
	case Parameter:
		return "parameter"
	case Tag:
		return "tag"
	case Attribute:
		return "attribute"
	case Dataset:
		return "dataset"
	default:
		return "unknown"
	}
}

type ValidCompareExpr struct {
	Identifier ValidIdentifier
	Key        string
	Operator   OperatorKind
	Value      interface{}
}

type ValidationError struct {
	message string
}

func (e *ValidationError) Error() string {
	return e.message
}

func NewValidationError(format string, a ...interface{}) *ValidationError {
	return &ValidationError{message: fmt.Sprintf(format, a...)}
}

const (
	metricIdentifier    = "metric"
	parameterIdentifier = "parameter"
	tagIdentifier       = "tag"
	attributeIdentifier = "attribute"
	datasetIdentifier   = "dataset"
)

var identifiers = []string{
	metricIdentifier,
	parameterIdentifier,
	tagIdentifier,
	attributeIdentifier,
	datasetIdentifier,
}

func parseValidIdentifier(identifier string) (ValidIdentifier, error) {
	switch identifier {
	case metricIdentifier, "metrics":
		return Metric, nil
	case parameterIdentifier, "parameters", "param", "params":
		return Parameter, nil
	case tagIdentifier, "tags":
		return Tag, nil
	case "", attributeIdentifier, "attr", "attributes", "run":
		return Attribute, nil
	case datasetIdentifier, "datasets":
		return Dataset, nil
	default:
		return -1, NewValidationError("invalid identifier %q", identifier)
	}
}

const (
	RunName = "run_name"
	Created = "created"
)

// This should be configurable and only applies to the runs table.
var searchableRunAttributes = []string{
	"run_id",
	"experiment_id",
	RunName,
	"user_id",
	"status",
	"start_time",
	"end_time",
	"artifact_uri",
	"lifecycle_stage",
}

var datasetAttributes = []string{"name", "digest", "context"}

func parseAttributeKey(key string) (string, error) {
	switch key {
	case "run_id":
		return "run_uuid", nil
	case "experiment_id",
		"user_id",
		"status",
		"start_time",
		"end_time",
		"artifact_uri",
		"lifecycle_stage":
		return key, nil
	case Created, "Created":
		return Created, nil
	case RunName, "run name", "Run name", "Run Name":
		return RunName, nil
	default:
		return "", NewValidationError(
			"invalid attribute key valid: %s. Allowed values are %v",
			key,
			searchableRunAttributes,
		)
	}
}

func parseKey(identifier ValidIdentifier, key string) (string, error) {
	if key == "" {
		return attributeIdentifier, nil
	}

	//nolint:exhaustive
	switch identifier {
	case Attribute:
		return parseAttributeKey(key)
	case Dataset:
		switch key {
		case "name", "digest", "context":
			return key, nil
		default:
			return "", NewValidationError(
				"invalid dataset attribute key: %s. Allowed values are %v",
				key,
				datasetAttributes,
			)
		}
	default:
		return key, nil
	}
}

// Returns a standardized LongIdentifierExpr.
func validatedIdentifier(identifier *Identifier) (ValidIdentifier, string, error) {
	validIdentifier, err := parseValidIdentifier(identifier.Identifier)
	if err != nil {
		return -1, "", err
	}

	validKey, err := parseKey(validIdentifier, identifier.Key)
	if err != nil {
		return -1, "", err
	}

	identifier.Key = validKey

	return validIdentifier, validKey, nil
}

/*

The value part is determined by the identifier

"metric" takes numbers
"parameter" and "tag" takes strings

"attribute" could be either string or number,
number when "start_time", "end_time" or "created", "Created"
otherwise string

"dataset" takes strings for "name", "digest" and "context"

*/

func validateDatasetValue(key string, value Value) (interface{}, error) {
	switch key {
	case "name", "digest", "context":
		if _, ok := value.(NumberExpr); ok {
			return nil, NewValidationError(
				"expected dataset.%s to be either a string or list of strings. Found %s",
				key,
				value,
			)
		}

		return value.value(), nil
	default:
		return nil, NewValidationError(
			"expected dataset attribute key to be one of %s. Found %s",
			strings.Join(datasetAttributes, ", "),
			key,
		)
	}
}

// Port of _get_value in search_utils.py.
func validateValue(identifier ValidIdentifier, key string, value Value) (interface{}, error) {
	switch identifier {
	case Metric:
		if _, ok := value.(NumberExpr); !ok {
			return nil, NewValidationError(
				"expected numeric value type for metric. Found %s",
				value,
			)
		}

		return value.value(), nil
	case Parameter, Tag:
		if _, ok := value.(StringExpr); !ok {
			return nil, NewValidationError(
				"expected a quoted string value for %s. Found %s",
				identifier, value,
			)
		}

		return value.value(), nil
	case Attribute:
		value, err := validateAttributeValue(key, value)

		return value, err
	case Dataset:
		return validateDatasetValue(key, value)
	default:
		return nil, NewValidationError(
			"Invalid identifier type %s. Expected one of %s",
			identifier,
			strings.Join(identifiers, ", "),
		)
	}
}

func validateAttributeValue(key string, value Value) (interface{}, error) {
	switch key {
	case "start_time", "end_time", Created:
		if _, ok := value.(NumberExpr); !ok {
			return nil, NewValidationError(
				"expected numeric value type for numeric attribute: %s. Found %s",
				key,
				value,
			)
		}

		return value.value(), nil
	default:
		if _, ok := value.(StringListExpr); key != RunName && ok {
			return nil, NewValidationError(
				"only the 'run_id' attribute supports comparison with a list of quoted string values",
			)
		}

		return value.value(), nil
	}
}

// Validate an expression according to the mlflow domain.
// This represent is a simple type-checker for the expression.
// Not every identifier is valid according to the mlflow domain.
// The same for the value part.
func ValidateExpression(expression *CompareExpr) (*ValidCompareExpr, error) {
	validIdentifier, validKey, err := validatedIdentifier(&expression.Left)
	if err != nil {
		return nil, fmt.Errorf("Error on parsing filter expression: %w", err)
	}

	value, err := validateValue(validIdentifier, validKey, expression.Right)
	if err != nil {
		return nil, fmt.Errorf("Error on parsing filter expression: %w", err)
	}

	return &ValidCompareExpr{
		Identifier: validIdentifier,
		Key:        validKey,
		Operator:   expression.Operator,
		Value:      value,
	}, nil
}
