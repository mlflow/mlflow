package lexer_test

import (
	"fmt"
	"strings"
	"testing"

	"github.com/mlflow/mlflow/mlflow/go/pkg/query/lexer"
)

type Sample struct {
	input    string
	expected string
}

func TestQueries(t *testing.T) {
	samples := []Sample{
		{
			input:    "metrics.accuracy > 0.72",
			expected: "identifier(metrics) dot identifier(accuracy) greater number(0.72) eof",
		},
		{
			input:    "metrics.\"accuracy\" > 0.72",
			expected: "identifier(metrics) dot string(\"accuracy\") greater number(0.72) eof",
		},
		{
			input:    "metrics.accuracy > 0.72 AND metrics.loss <= 0.15",
			expected: "identifier(metrics) dot identifier(accuracy) greater number(0.72) and identifier(metrics) dot identifier(loss) less_equals number(0.15) eof",
		},
		{
			input:    "params.batch_size = \"2\"",
			expected: "identifier(params) dot identifier(batch_size) equals string(\"2\") eof",
		},
		{
			input:    "tags.task ILIKE \"classif%\"",
			expected: "identifier(tags) dot identifier(task) ilike string(\"classif%\") eof",
		},
		{
			input:    "datasets.digest IN ('s8ds293b', 'jks834s2')",
			expected: "identifier(datasets) dot identifier(digest) in open_paren string('s8ds293b') comma string('jks834s2') close_paren eof",
		},
		{
			input:    "attributes.created > 1664067852747",
			expected: "identifier(attributes) dot identifier(created) greater number(1664067852747) eof",
		},
		{
			input:    "params.batch_size != \"None\"",
			expected: "identifier(params) dot identifier(batch_size) not_equals string(\"None\") eof",
		},
	}

	for _, sample := range samples {
		t.Run(sample.input, func(t *testing.T) {
			tokens := lexer.Tokenize(sample.input)
			output := ""
			for _, token := range tokens {
				output += fmt.Sprintf(" %s", token.Debug())
			}

			output = strings.TrimLeft(output, " ")

			if output != sample.expected {
				t.Errorf("expected %s, got %s", sample.expected, output)
			}
		})
	}
}
