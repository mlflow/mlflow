package query

import (
	"github.com/mlflow/mlflow/mlflow/go/pkg/query/lexer"
	"github.com/mlflow/mlflow/mlflow/go/pkg/query/parser"
)

func ParseFilter(input *string) (*parser.AndExpr, error) {
	tokens, err := lexer.Tokenize(input)
	if err != nil {
		return nil, err
	}

	ast, err := parser.Parse(tokens)
	if err != nil {
		return nil, err
	}

	for _, expr := range ast.Exprs {
		if err := parser.ValidateExpression(expr); err != nil {
			return nil, err
		}
	}

	return ast, nil
}
