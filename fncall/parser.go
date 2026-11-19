package fncall

import (
	"fmt"
	"strconv"
	"strings"
	"unicode"
)

type TokenType int

const (
	TokEOF TokenType = iota
	TokIdent
	TokString
	TokNumber
	TokTrue
	TokFalse
	TokNone
	TokLParen
	TokRParen
	TokEqual
	TokComma
	TokLBracket
	TokRBracket
	TokLBrace
	TokRBrace
	TokColon
)

type Token struct {
	Type        TokenType
	Val         string
	IsMultiLine bool
	Pos         int
}

type Lexer struct {
	src []rune
	pos int
}

func NewLexer(input string) *Lexer {
	return &Lexer{src: []rune(input), pos: 0}
}

func (l *Lexer) peek(offset int) rune {
	idx := l.pos + offset
	if idx >= len(l.src) {
		return 0
	}
	return l.src[idx]
}

func (l *Lexer) next() rune {
	if l.pos >= len(l.src) {
		return 0
	}
	r := l.src[l.pos]
	l.pos++
	return r
}

func (l *Lexer) skipWhitespace() {
	for l.pos < len(l.src) {
		r := l.src[l.pos]
		if unicode.IsSpace(r) {
			l.pos++
		} else {
			break
		}
	}
}

func (l *Lexer) NextToken() (Token, error) {
	l.skipWhitespace()
	if l.pos >= len(l.src) {
		return Token{Type: TokEOF, Pos: l.pos}, nil
	}

	start := l.pos
	r := l.peek(0)

	// Check multiline string """
	if r == '"' && l.peek(1) == '"' && l.peek(2) == '"' {
		l.pos += 3
		var sb strings.Builder
		for {
			if l.pos >= len(l.src) {
				return Token{}, fmt.Errorf("ErrSyntax: unclosed multiline string starting at pos %d", start)
			}
			if l.peek(0) == '"' && l.peek(1) == '"' && l.peek(2) == '"' {
				l.pos += 3
				return Token{Type: TokString, Val: sb.String(), IsMultiLine: true, Pos: start}, nil
			}
			sb.WriteRune(l.next())
		}
	}

	// Single-line string "..."
	if r == '"' {
		l.next()
		var sb strings.Builder
		for {
			if l.pos >= len(l.src) {
				return Token{}, fmt.Errorf("ErrSyntax: unclosed string starting at pos %d", start)
			}
			ch := l.next()
			if ch == '"' {
				return Token{Type: TokString, Val: sb.String(), IsMultiLine: false, Pos: start}, nil
			}
			if ch == '\\' {
				if l.pos >= len(l.src) {
					return Token{}, fmt.Errorf("ErrSyntax: unclosed escape at pos %d", start)
				}
				esc := l.next()
				switch esc {
				case '"':
					sb.WriteByte('"')
				case '\\':
					sb.WriteByte('\\')
				case '/':
					sb.WriteByte('/')
				case 'b':
					sb.WriteByte('\b')
				case 'f':
					sb.WriteByte('\f')
				case 'n':
					sb.WriteByte('\n')
				case 'r':
					sb.WriteByte('\r')
				case 't':
					sb.WriteByte('\t')
				case 'u':
					var hexRunes []rune
					for i := 0; i < 4; i++ {
						if l.pos >= len(l.src) {
							return Token{}, fmt.Errorf("ErrSyntax: incomplete unicode escape")
						}
						hexRunes = append(hexRunes, l.next())
					}
					uVal, err := strconv.ParseUint(string(hexRunes), 16, 32)
					if err != nil {
						return Token{}, fmt.Errorf("ErrSyntax: invalid unicode escape \\u%s", string(hexRunes))
					}
					sb.WriteRune(rune(uVal))
				default:
					return Token{}, fmt.Errorf("ErrSyntax: invalid escape \\%c", esc)
				}
			} else {
				sb.WriteRune(ch)
			}
		}
	}

	// Delimiters
	switch r {
	case '(':
		l.next()
		return Token{Type: TokLParen, Val: "(", Pos: start}, nil
	case ')':
		l.next()
		return Token{Type: TokRParen, Val: ")", Pos: start}, nil
	case '=':
		l.next()
		return Token{Type: TokEqual, Val: "=", Pos: start}, nil
	case ',':
		l.next()
		return Token{Type: TokComma, Val: ",", Pos: start}, nil
	case '[':
		l.next()
		return Token{Type: TokLBracket, Val: "[", Pos: start}, nil
	case ']':
		l.next()
		return Token{Type: TokRBracket, Val: "]", Pos: start}, nil
	case '{':
		l.next()
		return Token{Type: TokLBrace, Val: "{", Pos: start}, nil
	case '}':
		l.next()
		return Token{Type: TokRBrace, Val: "}", Pos: start}, nil
	case ':':
		l.next()
		return Token{Type: TokColon, Val: ":", Pos: start}, nil
	}

	// Numbers (negative sign or digit)
	if (r == '-' && unicode.IsDigit(l.peek(1))) || unicode.IsDigit(r) {
		var sb strings.Builder
		if r == '-' {
			sb.WriteRune(l.next())
		}
		for unicode.IsDigit(l.peek(0)) {
			sb.WriteRune(l.next())
		}
		if l.peek(0) == '.' && unicode.IsDigit(l.peek(1)) {
			sb.WriteRune(l.next()) // '.'
			for unicode.IsDigit(l.peek(0)) {
				sb.WriteRune(l.next())
			}
		}
		if l.peek(0) == 'e' || l.peek(0) == 'E' {
			sb.WriteRune(l.next())
			if l.peek(0) == '+' || l.peek(0) == '-' {
				sb.WriteRune(l.next())
			}
			for unicode.IsDigit(l.peek(0)) {
				sb.WriteRune(l.next())
			}
		}
		return Token{Type: TokNumber, Val: sb.String(), Pos: start}, nil
	}

	// Identifiers and keywords
	if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || r == '_' {
		var sb strings.Builder
		for {
			p := l.peek(0)
			if (p >= 'a' && p <= 'z') || (p >= 'A' && p <= 'Z') || (p >= '0' && p <= '9') || p == '_' {
				sb.WriteRune(l.next())
			} else {
				break
			}
		}
		word := sb.String()
		switch word {
		case "True":
			return Token{Type: TokTrue, Val: "True", Pos: start}, nil
		case "False":
			return Token{Type: TokFalse, Val: "False", Pos: start}, nil
		case "None":
			return Token{Type: TokNone, Val: "None", Pos: start}, nil
		default:
			return Token{Type: TokIdent, Val: word, Pos: start}, nil
		}
	}

	return Token{}, fmt.Errorf("ErrSyntax: unexpected character '%c' at pos %d", r, start)
}

type Parser struct {
	tokens []Token
	pos    int
}

func CleanCallString(s string) string {
	s = strings.TrimSpace(s)
	// Strip <|call|> and <|/call|> tags if present
	if strings.HasPrefix(s, "<|call|>") {
		s = strings.TrimPrefix(s, "<|call|>")
		s = strings.TrimSpace(s)
	}
	if strings.HasSuffix(s, "<|/call|>") {
		s = strings.TrimSuffix(s, "<|/call|>")
		s = strings.TrimSpace(s)
	}
	return s
}

func ParseCall(input string) (*Call, error) {
	input = CleanCallString(input)
	if input == "" {
		return nil, fmt.Errorf("ErrSyntax: empty tool call")
	}

	lexer := NewLexer(input)
	var tokens []Token
	for {
		tok, err := lexer.NextToken()
		if err != nil {
			return nil, err
		}
		tokens = append(tokens, tok)
		if tok.Type == TokEOF {
			break
		}
	}

	p := &Parser{tokens: tokens, pos: 0}
	return p.parseCall()
}

func (p *Parser) current() Token {
	if p.pos >= len(p.tokens) {
		return Token{Type: TokEOF}
	}
	return p.tokens[p.pos]
}

func (p *Parser) advance() Token {
	tok := p.current()
	p.pos++
	return tok
}

func (p *Parser) parseCall() (*Call, error) {
	// Expect tool name IDENT
	if p.current().Type != TokIdent {
		return nil, fmt.Errorf("ErrSyntax: expected tool name identifier, got %v", p.current().Val)
	}
	name := p.advance().Val

	// Expect '('
	if p.current().Type != TokLParen {
		return nil, fmt.Errorf("ErrSyntax: expected '(' after tool name, got %v", p.current().Val)
	}
	p.advance()

	var kwargs []Kwarg
	seenKeys := make(map[string]bool)

	for p.current().Type != TokRParen && p.current().Type != TokEOF {
		// Positional argument check: if it is not an IDENT followed by '=', flag positional argument error
		if p.current().Type != TokIdent {
			return nil, fmt.Errorf("ErrArgs: positional arguments are not allowed; use key=value")
		}
		keyTok := p.advance()
		if p.current().Type != TokEqual {
			return nil, fmt.Errorf("ErrArgs: positional arguments are not allowed; expected '=' after parameter '%s'", keyTok.Val)
		}
		p.advance() // consume '='

		val, err := p.parseValue()
		if err != nil {
			return nil, err
		}

		if seenKeys[keyTok.Val] {
			return nil, fmt.Errorf("ErrArgs: duplicate parameter '%s'", keyTok.Val)
		}
		seenKeys[keyTok.Val] = true
		kwargs = append(kwargs, Kwarg{Key: keyTok.Val, Val: val})

		if p.current().Type == TokComma {
			p.advance()
			if p.current().Type == TokRParen {
				// Trailing comma before ')'
				break
			}
		} else if p.current().Type != TokRParen {
			return nil, fmt.Errorf("ErrSyntax: expected ',' or ')' after parameter value, got %v", p.current().Val)
		}
	}

	if p.current().Type != TokRParen {
		return nil, fmt.Errorf("ErrSyntax: expected ')' at end of call")
	}
	p.advance()

	if p.current().Type != TokEOF {
		return nil, fmt.Errorf("ErrSyntax: unexpected token '%s' after call closed", p.current().Val)
	}

	return &Call{Name: name, Kwargs: kwargs}, nil
}

func (p *Parser) parseValue() (Value, error) {
	tok := p.current()
	switch tok.Type {
	case TokString:
		p.advance()
		return Value{Type: ValueTypeString, StrVal: tok.Val, IsMultiLine: tok.IsMultiLine}, nil
	case TokNumber:
		p.advance()
		return Value{Type: ValueTypeNumber, NumStr: tok.Val}, nil
	case TokTrue:
		p.advance()
		return Value{Type: ValueTypeBool, BoolVal: true}, nil
	case TokFalse:
		p.advance()
		return Value{Type: ValueTypeBool, BoolVal: false}, nil
	case TokNone:
		p.advance()
		return Value{Type: ValueTypeNone}, nil
	case TokLBracket:
		return p.parseList()
	case TokLBrace:
		return p.parseDict()
	default:
		return Value{}, fmt.Errorf("ErrSyntax: expected value literal, got %v", tok.Val)
	}
}

func (p *Parser) parseList() (Value, error) {
	p.advance() // consume '['
	var items []Value
	for p.current().Type != TokRBracket && p.current().Type != TokEOF {
		item, err := p.parseValue()
		if err != nil {
			return Value{}, err
		}
		items = append(items, item)

		if p.current().Type == TokComma {
			p.advance()
			if p.current().Type == TokRBracket {
				break
			}
		} else if p.current().Type != TokRBracket {
			return Value{}, fmt.Errorf("ErrSyntax: expected ',' or ']' in list")
		}
	}

	if p.current().Type != TokRBracket {
		return Value{}, fmt.Errorf("ErrSyntax: unclosed list, expected ']'")
	}
	p.advance()
	return Value{Type: ValueTypeList, ListVal: items}, nil
}

func (p *Parser) parseDict() (Value, error) {
	p.advance() // consume '{'
	var entries []DictEntry
	for p.current().Type != TokRBrace && p.current().Type != TokEOF {
		if p.current().Type != TokString {
			return Value{}, fmt.Errorf("ErrSyntax: dict keys must be strings, got %v", p.current().Val)
		}
		key := p.advance().Val

		if p.current().Type != TokColon {
			return Value{}, fmt.Errorf("ErrSyntax: expected ':' after dict key '%s'", key)
		}
		p.advance() // consume ':'

		val, err := p.parseValue()
		if err != nil {
			return Value{}, err
		}
		entries = append(entries, DictEntry{Key: key, Val: val})

		if p.current().Type == TokComma {
			p.advance()
			if p.current().Type == TokRBrace {
				break
			}
		} else if p.current().Type != TokRBrace {
			return Value{}, fmt.Errorf("ErrSyntax: expected ',' or '}' in dict")
		}
	}

	if p.current().Type != TokRBrace {
		return Value{}, fmt.Errorf("ErrSyntax: unclosed dict, expected '}'")
	}
	p.advance()
	return Value{Type: ValueTypeDict, DictVal: entries}, nil
}
