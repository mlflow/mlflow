package lexer

import "fmt"

type TokenKind int

const (
	EOF TokenKind = iota
	Number
	String
	Identifier

	// Grouping & Braces.
	OpenParen
	CloseParen

	// Equivilance.
	Equals
	NotEquals

	// Conditional.
	Less
	LessEquals
	Greater
	GreaterEquals

	// Symbols.
	Dot
	Comma

	// Reserved Keywords.
	In //nolint:varnamelen
	Not
	Like
	ILike
	And
)

//nolint:gochecknoglobals
var reservedLu = map[string]TokenKind{
	"AND":   And,
	"NOT":   Not,
	"IN":    In,
	"LIKE":  Like,
	"ILIKE": ILike,
}

type Token struct {
	Kind  TokenKind
	Value string
}

func (token Token) Debug() string {
	if token.Kind == Identifier || token.Kind == Number || token.Kind == String {
		return fmt.Sprintf("%s(%s)", TokenKindString(token.Kind), token.Value)
	}

	return TokenKindString(token.Kind)
}

//nolint:funlen,cyclop
func TokenKindString(kind TokenKind) string {
	switch kind {
	case EOF:
		return "eof"
	case Number:
		return "number"
	case String:
		return "string"
	case Identifier:
		return "identifier"
	case OpenParen:
		return "open_paren"
	case CloseParen:
		return "close_paren"
	case Equals:
		return "equals"
	case NotEquals:
		return "not_equals"
	case Less:
		return "less"
	case LessEquals:
		return "less_equals"
	case Greater:
		return "greater"
	case GreaterEquals:
		return "greater_equals"
	case And:
		return "and"
	case Dot:
		return "dot"
	case Comma:
		return "comma"
	case In:
		return "in"
	case Not:
		return "not"
	case Like:
		return "like"
	case ILike:
		return "ilike"
	default:
		return fmt.Sprintf("unknown(%d)", kind)
	}
}

func newUniqueToken(kind TokenKind, value string) Token {
	return Token{
		kind, value,
	}
}
