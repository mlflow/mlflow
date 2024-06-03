package lexer_test

import (
	"strings"
	"testing"

	"github.com/mlflow/mlflow/mlflow/go/pkg/query/lexer"
)

type Sample struct {
	input    string
	expected string
}

//nolint:lll,funlen
func TestQueries(t *testing.T) {
	t.Parallel()

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
		{
			input:    "datasets.digest NOT IN ('s8ds293b', 'jks834s2')",
			expected: "identifier(datasets) dot identifier(digest) not in open_paren string('s8ds293b') comma string('jks834s2') close_paren eof",
		},
		{
			input:    "params.`random_state` = \"8888\"",
			expected: "identifier(params) dot string(`random_state`) equals string(\"8888\") eof",
		},
	}

	for _, sample := range samples {
		currentSample := sample

		t.Run(currentSample.input, func(t *testing.T) {
			t.Parallel()

			tokens, err := lexer.Tokenize(&currentSample.input)
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}

			output := ""

			for _, token := range tokens {
				output += " " + token.Debug()
			}

			output = strings.TrimLeft(output, " ")

			if output != currentSample.expected {
				t.Errorf("expected %s, got %s", currentSample.expected, output)
			}
		})
	}
}

func TestInvalidInput(t *testing.T) {
	t.Parallel()

	samples := []string{
		"params.'acc = LR",
		"params.acc = 'LR",
		"params.acc = LR'",
		"params.acc = \"LR'",
		"tags.acc = \"LR'",
	}

	for _, sample := range samples {
		currentSample := sample
		t.Run(currentSample, func(t *testing.T) {
			t.Parallel()

			_, err := lexer.Tokenize(&currentSample)
			if err == nil {
				t.Errorf("expected error, got nil")
			}
		})
	}
}
