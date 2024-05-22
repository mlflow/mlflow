package parser

import (
	"fmt"
	"strconv"

	"github.com/mlflow/mlflow/mlflow/go/pkg/query/lexer"
)

type parser struct {
	tokens []lexer.Token
	pos    int
}

func newParser(tokens []lexer.Token) *parser {
	return &parser{
		tokens: tokens,
		pos:    0,
	}
}

func (p *parser) currentTokenKind() lexer.TokenKind {
	return p.tokens[p.pos].Kind
}

func (p *parser) hasTokens() bool {
	return p.pos < len(p.tokens) && p.currentTokenKind() != lexer.EOF
}

func (p *parser) printCurrentToken() string {
	return p.tokens[p.pos].Debug()
}

func (p *parser) currentToken() lexer.Token {
	return p.tokens[p.pos]
}

func (p *parser) advance() lexer.Token {
	tk := p.currentToken()
	p.pos++
	return tk
}

func (p *parser) parseIdentifier() (Identifier, error) {
	if p.hasTokens() && p.currentTokenKind() != lexer.IDENTIFIER {
		return Identifier{}, fmt.Errorf("Expected identifier, got %s", p.printCurrentToken())
	}

	identToken := p.advance()

	if p.currentTokenKind() == lexer.DOT {
		p.advance() // Consume the DOT
		switch p.currentTokenKind() {
		case lexer.IDENTIFIER:
			column := p.advance().Value
			return Identifier{Identifier: identToken.Value, Key: column}, nil
		case lexer.STRING:
			column := p.advance().Value
			column = column[1 : len(column)-1] // Remove quotes
			return Identifier{Identifier: identToken.Value, Key: column}, nil
		default:
			return Identifier{}, fmt.Errorf("Expected IDENTIFIER or STRING, got %s", p.printCurrentToken())
		}
	} else {
		return Identifier{Key: identToken.Value}, nil
	}
}

func (p *parser) parseOperator() (OperatorKind, error) {
	switch p.advance().Kind {
	case lexer.EQUALS:
		return EQUALS, nil
	case lexer.NOT_EQUALS:
		return NOT_EQUALS, nil
	case lexer.LESS:
		return LESS, nil
	case lexer.LESS_EQUALS:
		return LESS_EQUALS, nil
	case lexer.GREATER:
		return GREATER, nil
	case lexer.GREATER_EQUALS:
		return GREATER_EQUALS, nil
	case lexer.LIKE:
		return LIKE, nil
	case lexer.ILIKE:
		return ILIKE, nil
	default:
		return -1, fmt.Errorf("Expected operator, got %s", p.printCurrentToken())
	}
}

func (p *parser) parseValue() (Value, error) {
	if p.currentTokenKind() == lexer.NUMBER {
		n, err := strconv.ParseFloat(p.advance().Value, 64)
		if err != nil {
			return nil, err
		} else {
			return NumberExpr{Value: n}, nil
		}
	} else if p.currentTokenKind() == lexer.STRING {
		value := p.advance().Value
		value = value[1 : len(value)-1] // Remove quotes
		return StringExpr{Value: value}, nil
	} else {
		return nil, fmt.Errorf("Expected NUMBER or STRING, got %s", p.printCurrentToken())
	}
}

func (p *parser) parseInSetExpr(ident Identifier) (*CompareExpr, error) {
	if p.currentTokenKind() != lexer.OPEN_PAREN {
		return nil, fmt.Errorf("Expected '(', got %s", p.printCurrentToken())
	}

	p.advance() // Consume the OPEN_PAREN

	set := make([]string, 0)
	for p.hasTokens() && p.currentTokenKind() != lexer.CLOSE_PAREN {
		if p.currentTokenKind() != lexer.STRING {
			return nil, fmt.Errorf("Expected STRING, got %s", p.printCurrentToken())
		}

		value := p.advance().Value
		value = value[1 : len(value)-1] // Remove quotes

		set = append(set, value)

		if p.currentTokenKind() == lexer.COMMA {
			p.advance() // Consume the COMMA
		}
	}

	if p.currentTokenKind() != lexer.CLOSE_PAREN {
		return nil, fmt.Errorf("Expected ')', got %s", p.printCurrentToken())
	}

	p.advance() // Consume the CLOSE_PAREN

	return &CompareExpr{Left: ident, Operator: IN, Right: StringListExpr{Values: set}}, nil
}

func (p *parser) parseExpression() (*CompareExpr, error) {
	ident, err := p.parseIdentifier()
	if err != nil {
		return nil, err
	}

	if p.currentTokenKind() == lexer.IN {
		p.advance() // Consume the IN
		return p.parseInSetExpr(ident)
	} else if p.currentTokenKind() == lexer.NOT {
		p.advance() // Consume the NOT
		if p.currentTokenKind() != lexer.IN {
			return nil, fmt.Errorf("Expected IN after NOT, got %s", p.printCurrentToken())
		}

		p.advance() // Consume the IN
		expr, err := p.parseInSetExpr(ident)
		if err != nil {
			return nil, err
		}

		expr.Operator = NOT_IN
		return expr, nil

	} else {
		operator, err := p.parseOperator()
		if err != nil {
			return nil, err
		}

		value, err := p.parseValue()
		if err != nil {
			return nil, err
		}

		return &CompareExpr{Left: ident, Operator: operator, Right: value}, nil
	}
}

func (p *parser) parse() (*AndExpr, error) {
	exprs := make([]*CompareExpr, 0)
	leftExpr, err := p.parseExpression()
	if err != nil {
		return nil, err
	}

	exprs = append(exprs, leftExpr)

	// While there are tokens and the next token is AND
	for p.currentTokenKind() == lexer.AND {
		p.advance() // Consume the AND
		rightExpr, err := p.parseExpression()
		if err != nil {
			return nil, err
		}
		exprs = append(exprs, rightExpr)
	}

	if p.hasTokens() {
		return nil, fmt.Errorf("Unexpected leftover token(s) after parsing: %s", p.printCurrentToken())
	}

	return &AndExpr{Exprs: exprs}, nil
}

func Parse(tokens []lexer.Token) (*AndExpr, error) {
	parser := newParser(tokens)
	return parser.parse()
}
