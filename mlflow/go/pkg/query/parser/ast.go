package parser

import (
	"fmt"
	"strings"
)

// --------------------
// Literal Expressions
// --------------------

type Value interface {
	value()
	fmt.Stringer
}

type NumberExpr struct {
	Value float64
}

func (n NumberExpr) value() {}

func (n NumberExpr) String() string {
	return fmt.Sprintf("%f", n.Value)
}

type StringExpr struct {
	Value string
}

func (n StringExpr) value() {}

func (n StringExpr) String() string {
	return fmt.Sprint("\"%s\"", n.Value)
}

type StringListExpr struct {
	Values []string
}

func (n StringListExpr) value() {}

func (n StringListExpr) String() string {
	items := make([]string, 0, len(n.Values))
	for _, v := range n.Values {
		items = append(items, fmt.Sprintf("\"%s\"", v))
	}
	return strings.Join(items, ", ")
}

//-----------------------
// Identifier Expressions
// ----------------------

// indentifier.key expression, like metric.foo
type Identifier struct {
	Identifier string
	Key        string
}

func (i Identifier) String() string {
	if i.Key == "" {
		return i.Identifier
	}
	return fmt.Sprintf("%s.%s", i.Identifier, i.Key)
}

// --------------------
// Comparison Expression
// --------------------

type OperatorKind int

const (
	EQUALS OperatorKind = iota
	NOT_EQUALS
	LESS
	LESS_EQUALS
	GREATER
	GREATER_EQUALS
	LIKE
	ILIKE
	IN
	NOT_IN
)

func (op OperatorKind) String() string {
	switch op {
	case EQUALS:
		return "="
	case NOT_EQUALS:
		return "!="
	case LESS:
		return "<"
	case LESS_EQUALS:
		return "<="
	case GREATER:
		return ">"
	case GREATER_EQUALS:
		return ">="
	case LIKE:
		return "LIKE"
	case ILIKE:
		return "ILIKE"
	case IN:
		return "IN"
	case NOT_IN:
		return "NOT IN"
	default:
		return "UNKNOWN"
	}
}

// a operator b
type CompareExpr struct {
	Left     Identifier
	Operator OperatorKind
	Right    Value
}

func (expr *CompareExpr) String() string {
	return fmt.Sprintf("%s %s %s", expr.Left, expr.Operator, expr.Right)
}

// AND
type AndExpr struct {
	Exprs []*CompareExpr
}
