package lexer

import "fmt"

type TokenKind int

const (
	EOF TokenKind = iota
	NUMBER
	STRING
	IDENTIFIER

	// Grouping & Braces
	OPEN_PAREN
	CLOSE_PAREN

	// Equivilance
	EQUALS
	NOT_EQUALS
	NOT

	// Conditional
	LESS
	LESS_EQUALS
	GREATER
	GREATER_EQUALS

	// Symbols
	DOT
	COMMA

	// Reserved Keywords
	IN
	LIKE
	ILIKE
	AND
)

var reserved_lu map[string]TokenKind = map[string]TokenKind{
	"AND":   AND,
	"IN":    IN,
	"LIKE":  LIKE,
	"ILIKE": ILIKE,
}

type Token struct {
	Kind  TokenKind
	Value string
}

func (token Token) Debug() string {
	if token.Kind == IDENTIFIER || token.Kind == NUMBER || token.Kind == STRING {
		return fmt.Sprintf("%s(%s)", TokenKindString(token.Kind), token.Value)
	} else {
		return TokenKindString(token.Kind)
	}
}

func TokenKindString(kind TokenKind) string {
	switch kind {
	case EOF:
		return "eof"
	case NUMBER:
		return "number"
	case STRING:
		return "string"
	case IDENTIFIER:
		return "identifier"
	case OPEN_PAREN:
		return "open_paren"
	case CLOSE_PAREN:
		return "close_paren"
	case EQUALS:
		return "equals"
	case NOT_EQUALS:
		return "not_equals"
	case NOT:
		return "not"
	case LESS:
		return "less"
	case LESS_EQUALS:
		return "less_equals"
	case GREATER:
		return "greater"
	case GREATER_EQUALS:
		return "greater_equals"
	case AND:
		return "and"
	case DOT:
		return "dot"
	case COMMA:
		return "comma"
	case IN:
		return "in"
	case LIKE:
		return "like"
	case ILIKE:
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
