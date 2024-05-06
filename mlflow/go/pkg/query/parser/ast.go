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

//-----------------------
// Identifier Expressions
// ----------------------

type Identifier interface {
	identifier()
}

type IdentifierExpr struct {
	Value string
}

func (n IdentifierExpr) identifier() {}

// foo.bar expression
type LongIdentifierExpr struct {
	Table  string
	Column string
}

func (n LongIdentifierExpr) identifier() {}

// --------------------
// Operator Expressions
// --------------------

type Expr interface {
	expr()
}

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
)

// a operator b
type BinaryExpr struct {
	Left     Identifier
	Operator OperatorKind
	Right    Value
}

func (n BinaryExpr) expr() {}

// a IN (b, c, d)
type InSetExpr struct {
	Identifier Identifier
	Set        []string
}

// a NOT IN (b, c, d)
type NotInSetExpr struct {
	Identifier Identifier
	Set        []string
}

func (n NotInSetExpr) expr() {}

func (n InSetExpr) expr() {}

// AND
type AndExpr struct {
	Exprs []Expr
}
