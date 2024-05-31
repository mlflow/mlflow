package lexer

import (
	"fmt"
	"regexp"
	"strings"
)

type regexPattern struct {
	regex   *regexp.Regexp
	handler regexHandler
}

type lexer struct {
	patterns []regexPattern
	Tokens   []Token
	source   *string
	pos      int
	line     int
}

type Error struct {
	message string
}

func NewLexerError(format string, a ...any) *Error {
	return &Error{message: fmt.Sprintf(format, a...)}
}

func (e *Error) Error() string {
	return e.message
}

func Tokenize(source *string) ([]Token, error) {
	lex := createLexer(source)

	for !lex.atEOF() {
		matched := false

		for _, pattern := range lex.patterns {
			loc := pattern.regex.FindStringIndex(lex.remainder())
			if loc != nil && loc[0] == 0 {
				pattern.handler(lex, pattern.regex)

				matched = true

				break // Exit the loop after the first match
			}
		}

		if !matched {
			return lex.Tokens, NewLexerError("unrecognized token near '%v'", lex.remainder())
		}
	}

	lex.push(newUniqueToken(EOF, "EOF"))

	return lex.Tokens, nil
}

func (lex *lexer) advanceN(n int) {
	lex.pos += n
}

func (lex *lexer) remainder() string {
	return (*lex.source)[lex.pos:]
}

func (lex *lexer) push(token Token) {
	lex.Tokens = append(lex.Tokens, token)
}

func (lex *lexer) atEOF() bool {
	return lex.pos >= len(*lex.source)
}

func createLexer(source *string) *lexer {
	return &lexer{
		pos:    0,
		line:   1,
		source: source,
		Tokens: make([]Token, 0),
		patterns: []regexPattern{
			{regexp.MustCompile(`\s+`), skipHandler},
			{regexp.MustCompile(`"[^"]*"`), stringHandler},
			{regexp.MustCompile(`'[^\']*\'`), stringHandler},
			{regexp.MustCompile("`[^`]*`"), stringHandler},
			{regexp.MustCompile(`[0-9]+(\.[0-9]+)?`), numberHandler},
			{regexp.MustCompile(`[a-zA-Z_][a-zA-Z0-9_]*`), symbolHandler},
			{regexp.MustCompile(`\(`), defaultHandler(OpenParen, "(")},
			{regexp.MustCompile(`\)`), defaultHandler(CloseParen, ")")},
			{regexp.MustCompile(`!=`), defaultHandler(NotEquals, "!=")},
			{regexp.MustCompile(`=`), defaultHandler(Equals, "=")},
			{regexp.MustCompile(`<=`), defaultHandler(LessEquals, "<=")},
			{regexp.MustCompile(`<`), defaultHandler(Less, "<")},
			{regexp.MustCompile(`>=`), defaultHandler(GreaterEquals, ">=")},
			{regexp.MustCompile(`>`), defaultHandler(Greater, ">")},
			{regexp.MustCompile(`\.`), defaultHandler(Dot, ".")},
			{regexp.MustCompile(`,`), defaultHandler(Comma, ",")},
		},
	}
}

type regexHandler func(lex *lexer, regex *regexp.Regexp)

// Created a default handler which will simply create a token with the matched contents.
// This handler is used with most simple tokens.
func defaultHandler(kind TokenKind, value string) regexHandler {
	return func(lex *lexer, _ *regexp.Regexp) {
		lex.advanceN(len(value))
		lex.push(newUniqueToken(kind, value))
	}
}

func stringHandler(lex *lexer, regex *regexp.Regexp) {
	match := regex.FindStringIndex(lex.remainder())
	stringLiteral := lex.remainder()[match[0]:match[1]]

	lex.push(newUniqueToken(String, stringLiteral))
	lex.advanceN(len(stringLiteral))
}

func numberHandler(lex *lexer, regex *regexp.Regexp) {
	match := regex.FindString(lex.remainder())
	lex.push(newUniqueToken(Number, match))
	lex.advanceN(len(match))
}

func symbolHandler(lex *lexer, regex *regexp.Regexp) {
	match := regex.FindString(lex.remainder())
	keyword := strings.ToUpper(match)

	if kind, found := reservedLu[keyword]; found {
		lex.push(newUniqueToken(kind, match))
	} else {
		lex.push(newUniqueToken(Identifier, match))
	}

	lex.advanceN(len(match))
}

func skipHandler(lex *lexer, regex *regexp.Regexp) {
	match := regex.FindStringIndex(lex.remainder())
	lex.advanceN(match[1])
}
