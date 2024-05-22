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

var (
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

var (
	alternateMetricIdentifiers    = []string{"metrics"}
	alternateParamIdentifiers     = []string{"parameters", "param", "params"}
	alternateTagIdentifiers       = []string{"tags"}
	alternateAttributeIdentifiers = []string{"attr", "attributes", "run"}
	alternateDatassetIdentifiers  = []string{"datasets"}
)

func mkValidIdentifiers() []string {
	len := len(identifiers) + len(alternateMetricIdentifiers) + len(alternateParamIdentifiers) + len(alternateTagIdentifiers) + len(alternateAttributeIdentifiers) + len(alternateDatassetIdentifiers)
	validIdentifiers := make([]string, 0, len)
	validIdentifiers = append(validIdentifiers, identifiers...)
	validIdentifiers = append(validIdentifiers, alternateMetricIdentifiers...)
	validIdentifiers = append(validIdentifiers, alternateParamIdentifiers...)
	validIdentifiers = append(validIdentifiers, alternateTagIdentifiers...)
	validIdentifiers = append(validIdentifiers, alternateAttributeIdentifiers...)
	validIdentifiers = append(validIdentifiers, alternateDatassetIdentifiers...)
	return validIdentifiers
}

var validIdentifiers = mkValidIdentifiers()

func contains[T comparable](s []T, elem T) bool {
	for _, e := range s {
		if e == elem {
			return true
		}
	}
	return false
}

func parseValidIdentifier(identifier string) (string, error) {
	switch identifier {
	case metricIdentifier, "metrics":
		return metricIdentifier, nil
	case parameterIdentifier:
		return parameterIdentifier, nil
	case tagIdentifier:
		return tagIdentifier, nil
	case attributeIdentifier:
		return attributeIdentifier, nil
	case datasetIdentifier:
		return datasetIdentifier, nil
	default:
		if contains(alternateMetricIdentifiers, identifier) {
			return metricIdentifier, nil
		} else if contains(alternateParamIdentifiers, identifier) {
			return parameterIdentifier, nil
		} else if contains(alternateTagIdentifiers, identifier) {
			return tagIdentifier, nil
		} else if contains(alternateAttributeIdentifiers, identifier) {
			return attributeIdentifier, nil
		} else if contains(alternateDatassetIdentifiers, identifier) {
			return datasetIdentifier, nil
		} else {
			return "", fmt.Errorf("invalid identifier: %s", identifier)
		}
	}
}

// This should be configurable and only applies to the runs table.
var searchableRunAttributes = []string{
	"run_id",
	"experiment_id",
	"run_name",
	"user_id",
	"status",
	"start_time",
	"end_time",
	"artifact_uri",
	"lifecycle_stage",
}

var (
	builtinNumericAttributes   = []string{"start_time", "end_time"}
	alternateNumericAttributes = []string{"created", "Created"}
	alternateStringAttributes  = []string{"run name", "Run name", "Run Name"} // Accurate?
)

var numericAttributes = append(builtinNumericAttributes, alternateNumericAttributes...)

var datasetAttributes = []string{"name", "digest", "context"}

func parseKey(identifier, key string) (result string, error error) {
	switch identifier {
	case attributeIdentifier:
		if !(contains(searchableRunAttributes, key)) && !(contains(alternateNumericAttributes, key)) && !(contains(alternateStringAttributes, key)) {
			error = fmt.Errorf("Invalid attribute key valid: %s. Allowed values are %v", key, searchableRunAttributes)
		}
	case datasetIdentifier:
		if !(contains(datasetAttributes, key)) {
			error = fmt.Errorf("Invalid dataset attribute key: %s. Allowed values are %v", key, datasetAttributes)
		}
	default:
		result = key
	}

	return
}

// Returns a standardized LongIdentifierExpr.
func validatedIdentifier(identifier *Identifier) error {
	if identifier.Key == "" {
		identifier.Key = attributeIdentifier
	}

	validIdentifier, err := parseValidIdentifier(identifier.Identifier)
	if err != nil {
		return err
	}
	identifier.Identifier = validIdentifier

	validKey, err := parseKey(validIdentifier, identifier.Key)
	if err != nil {
		return err
	}
	identifier.Key = validKey

	return nil
}

/*

The value part is determined by the identifier

"metric" takes numbers
"parameter" and "tag" takes strings

"attribute" could be either string or number
number when "start_time", "end_time" or "created", "Created"
otherwise string

"dataset" takes strings for "name", "digest" and "context"

*/

// Port of _get_value in search_utils.py.
func validateValue(expression *CompareExpr) error {
	switch expression.Left.Identifier {
	case metricIdentifier:
		if _, ok := expression.Right.(NumberExpr); !ok {
			return fmt.Errorf("Expected numeric value type for metric. Found %s", expression.Right)
		}
	case parameterIdentifier, tagIdentifier:
		if _, ok := expression.Right.(StringExpr); !ok {
			return fmt.Errorf(
				"Expected a quoted string value for %s. Found %s",
				expression.Left.Identifier, expression.Right,
			)
		}
	case attributeIdentifier:
		if contains(numericAttributes, expression.Left.Key) {
			if _, ok := expression.Right.(NumberExpr); !ok {
				return fmt.Errorf(
					"Expected numeric value type for numeric attribute: %s. Found %s",
					expression.Left.Key,
					expression.Right,
				)
			}
		}

		if _, ok := expression.Right.(StringListExpr); expression.Left.Key != "run_name" && ok {
			return fmt.Errorf(
				"Only the 'run_id' attribute supports comparison with a list of quoted string values.",
			)
		}
	case datasetIdentifier:
		if !contains(datasetAttributes, expression.Left.Key) {
			return fmt.Errorf(
				"Expected dataset attribute key to be one of %s. Found %s",
				strings.Join(datasetAttributes, ", "),
				expression.Left.Key,
			)
		}

		if _, ok := expression.Right.(NumberExpr); ok {
			return fmt.Errorf(
				"Expected dataset.%s to be either a string or list of strings. Found %s",
				expression.Left.Key,
				expression.Right,
			)
		}
	default:
		return fmt.Errorf(
			"Invalid identifier type %s. Expected one of %s",
			expression.Left.Identifier,
			strings.Join(identifiers, ", "),
		)
	}

	return nil
}

// Validate an expression according to the mlflow domain.
// This represent is a simple type-checker for the expression.
// Not every identifier is valid according to the mlflow domain.
// The same for the value part.
// The identifier is sanitized and will be mutated to use the standard identifier.
func ValidateExpression(expression *CompareExpr) error {
	err := validatedIdentifier(&expression.Left)
	if err != nil {
		return fmt.Errorf("Error on parsing filter expression: %s", err)
	}

	err = validateValue(expression)
	if err != nil {
		return fmt.Errorf("Error on parsing filter expression: %s", err)
	}

	return nil
}
