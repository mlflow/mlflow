package parser

import (
	"fmt"
	"strings"
)

// --------------------
// Literal Expressions
// --------------------

type Value interface {
	value() interface{}
	fmt.Stringer
}

type NumberExpr struct {
	Value float64
}

func (n NumberExpr) value() interface{} {
	return n.Value
}

func (n NumberExpr) String() string {
	return fmt.Sprintf("%f", n.Value)
}

type StringExpr struct {
	Value string
}

func (n StringExpr) value() interface{} {
	return n.Value
}

func (n StringExpr) String() string {
	return fmt.Sprintf("\"%s\"", n.Value)
}

type StringListExpr struct {
	Values []string
}

func (n StringListExpr) value() interface{} {
	return n.Values
}

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

// identifier.key expression, like metric.foo.
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
	Equals OperatorKind = iota
	NotEquals
	Less
	LessEquals
	Greater
	GreaterEquals
	Like
	ILike
	In //nolint:varnamelen
	NotIn
)

//nolint:cyclop
func (op OperatorKind) String() string {
	switch op {
	case Equals:
		return "="
	case NotEquals:
		return "!="
	case Less:
		return "<"
	case LessEquals:
		return "<="
	case Greater:
		return ">"
	case GreaterEquals:
		return ">="
	case Like:
		return "LIKE"
	case ILike:
		return "ILIKE"
	case In:
		return "IN"
	case NotIn:
		return "NOT IN"
	default:
		return "UNKNOWN"
	}
}

// a operator b.
type CompareExpr struct {
	Left     Identifier
	Operator OperatorKind
	Right    Value
}

func (expr *CompareExpr) String() string {
	return fmt.Sprintf("%s %s %s", expr.Left, expr.Operator, expr.Right)
}

// AND.
type AndExpr struct {
	Exprs []*CompareExpr
}
