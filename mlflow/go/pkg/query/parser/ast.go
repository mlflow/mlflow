package parser

// --------------------
// Literal Expressions
// --------------------

type Value interface {
	value()
}

type NumberExpr struct {
	Value float64
}

func (n NumberExpr) value() {}

type StringExpr struct {
	Value string
}

func (n StringExpr) value() {}

type StringListExpr struct {
	Values []string
}

func (n StringListExpr) value() {}

//-----------------------
// Identifier Expressions
// ----------------------

// indentifier.key expression, like metric.foo
type Identifier struct {
	Identifier string
	Key        string
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

// a operator b
type CompareExpr struct {
	Left     Identifier
	Operator OperatorKind
	Right    Value
}

// AND
type AndExpr struct {
	Exprs []*CompareExpr
}
