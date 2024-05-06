package parser_test

import (
	"reflect"
	"testing"

	"github.com/mlflow/mlflow/mlflow/go/pkg/query/lexer"
	"github.com/mlflow/mlflow/mlflow/go/pkg/query/parser"
)

type Sample struct {
	input    string
	expected parser.AndExpr
}

func TestQueries(t *testing.T) {
	samples := []Sample{
		{
			input: "metrics.accuracy > 0.72",
			expected: parser.AndExpr{
				Exprs: []parser.Expr{
					parser.BinaryExpr{
						Left:     parser.LongIdentifierExpr{"metrics", "accuracy"},
						Operator: parser.GREATER,
						Right:    parser.NumberExpr{Value: 0.72},
					},
				},
			},
		},
		{
			input: "metrics.\"accuracy\" > 0.72",
			expected: parser.AndExpr{
				Exprs: []parser.Expr{
					parser.BinaryExpr{
						Left:     parser.LongIdentifierExpr{"metrics", "accuracy"},
						Operator: parser.GREATER,
						Right:    parser.NumberExpr{Value: 0.72},
					},
				},
			},
		},
		{
			input: "metrics.accuracy > 0.72 AND metrics.loss <= 0.15",
			expected: parser.AndExpr{
				Exprs: []parser.Expr{
					parser.BinaryExpr{
						Left:     parser.LongIdentifierExpr{"metrics", "accuracy"},
						Operator: parser.GREATER,
						Right:    parser.NumberExpr{Value: 0.72},
					},
					parser.BinaryExpr{
						Left:     parser.LongIdentifierExpr{"metrics", "loss"},
						Operator: parser.LESS_EQUALS,
						Right:    parser.NumberExpr{Value: 0.15},
					},
				},
			},
		},
		{
			input: "params.batch_size = \"2\"",
			expected: parser.AndExpr{
				Exprs: []parser.Expr{
					parser.BinaryExpr{
						Left:     parser.LongIdentifierExpr{"params", "batch_size"},
						Operator: parser.EQUALS,
						Right:    parser.StringExpr{Value: "2"},
					},
				},
			},
		},
		{
			input: "tags.task ILIKE \"classif%\"",
			expected: parser.AndExpr{
				Exprs: []parser.Expr{
					parser.BinaryExpr{
						Left:     parser.LongIdentifierExpr{"tags", "task"},
						Operator: parser.ILIKE,
						Right:    parser.StringExpr{Value: "classif%"},
					},
				},
			},
		},
		{
			input: "datasets.digest IN ('s8ds293b', 'jks834s2')",
			expected: parser.AndExpr{
				Exprs: []parser.Expr{
					parser.InSetExpr{
						Identifier: parser.LongIdentifierExpr{"datasets", "digest"},
						Set:        []string{"s8ds293b", "jks834s2"},
					},
				},
			},
		},
		{
			input: "attributes.created > 1664067852747",
			expected: parser.AndExpr{
				[]parser.Expr{
					parser.BinaryExpr{
						Left:     parser.LongIdentifierExpr{"attributes", "created"},
						Operator: parser.GREATER,
						Right:    parser.NumberExpr{Value: 1664067852747},
					},
				},
			},
		},
		{
			input: "params.batch_size != \"None\"",
			expected: parser.AndExpr{
				Exprs: []parser.Expr{
					parser.BinaryExpr{
						Left:     parser.LongIdentifierExpr{"params", "batch_size"},
						Operator: parser.NOT_EQUALS,
						Right:    parser.StringExpr{Value: "None"},
					},
				},
			},
		},
		{
			input: "datasets.digest NOT IN ('s8ds293b', 'jks834s2')",
			expected: parser.AndExpr{
				Exprs: []parser.Expr{
					parser.NotInSetExpr{
						Identifier: parser.LongIdentifierExpr{"datasets", "digest"},
						Set:        []string{"s8ds293b", "jks834s2"},
					},
				},
			},
		},
	}

	for _, sample := range samples {
		t.Run(sample.input, func(t *testing.T) {
			tokens, err := lexer.Tokenize(sample.input)
			if err != nil {
				t.Errorf("unexpected lex error: %v", err)
			}

			ast, err := parser.Parse(tokens)
			if err != nil {
				t.Errorf("error parsing: %s", err)
			}

			if !reflect.DeepEqual(ast, sample.expected) {
				t.Errorf("expected %#v, got %#v", sample.expected, ast)
			}
		})
	}
}

func TestInvalidSyntax(t *testing.T) {
	samples := []string{
		"attribute.status IS 'RUNNING'",
	}

	for _, sample := range samples {
		t.Run(sample, func(t *testing.T) {
			tokens, err := lexer.Tokenize(sample)
			if err != nil {
				t.Errorf("unexpected lex error: %v", err)
			}

			_, err = parser.Parse(tokens)
			if err == nil {
				t.Errorf("expected parse error, got nil")
			}
		})
	}
}
