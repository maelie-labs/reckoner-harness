package main

import (
	"fmt"
	"math"
	"math/big"
	"strconv"
	"strings"
	"unicode"
)

// -----------------------------------------------------------------------------
// Mathematical Helpers compatible with Lua & integer functions
// -----------------------------------------------------------------------------

func isPrime(n int64) bool {
	if n <= 1 {
		return false
	}
	return big.NewInt(n).ProbablyPrime(20)
}

func gcdInt(a, b int64) int64 {
	x := new(big.Int).Abs(big.NewInt(a))
	y := new(big.Int).Abs(big.NewInt(b))
	return new(big.Int).GCD(nil, nil, x, y).Int64()
}

func lcmInt(a, b int64) int64 {
	if a == 0 || b == 0 {
		return 0
	}
	g := gcdInt(a, b)
	prod := new(big.Int).Mul(big.NewInt(a), big.NewInt(b))
	prod.Abs(prod)
	return prod.Div(prod, big.NewInt(g)).Int64()
}

func primeFactorsWithBudget(n int64, maxSteps int) ([]int64, error) {
	var factors []int64
	steps := 0
	d := int64(2)
	for d*d <= n {
		steps++
		if steps > maxSteps {
			return nil, fmt.Errorf("ErrTimeout: step budget exceeded")
		}
		for n%d == 0 {
			factors = append(factors, d)
			n /= d
		}
		if d == 2 {
			d = 3
		} else {
			d += 2
		}
	}
	if n > 1 {
		factors = append(factors, n)
	}
	return factors, nil
}

func primeFactors(n int64) []int64 {
	facs, _ := primeFactorsWithBudget(n, 1000000)
	return facs
}

func combInt(n, k int64) (int64, error) {
	if k < 0 || n < 0 {
		return 0, fmt.Errorf("ErrDomain: comb expects non-negative integers")
	}
	if k > n {
		return 0, nil
	}
	if k > n/2 {
		k = n - k
	}
	res := big.NewInt(1)
	for i := int64(1); i <= k; i++ {
		res.Mul(res, big.NewInt(n-i+1))
		res.Div(res, big.NewInt(i))
	}
	if !res.IsInt64() {
		return 0, fmt.Errorf("ErrOverflow: comb result exceeds 64-bit integer limit")
	}
	return res.Int64(), nil
}

func permInt(n, k int64) (int64, error) {
	if k < 0 || n < 0 {
		return 0, fmt.Errorf("ErrDomain: perm expects non-negative integers")
	}
	if k > n {
		return 0, nil
	}
	res := big.NewInt(1)
	for i := int64(0); i < k; i++ {
		res.Mul(res, big.NewInt(n-i))
	}
	if !res.IsInt64() {
		return 0, fmt.Errorf("ErrOverflow: perm result exceeds 64-bit integer limit")
	}
	return res.Int64(), nil
}

func factorialInt(n int64) (int64, error) {
	if n < 0 {
		return 0, fmt.Errorf("ErrDomain: factorial expects a non-negative integer")
	}
	if n > 20000 {
		return 0, fmt.Errorf("ErrOverflow: factorial exceeds maximum n=20,000")
	}
	if n > 20 {
		return 0, fmt.Errorf("ErrOverflow: factorial result exceeds 64-bit integer limit")
	}
	res := int64(1)
	for i := int64(2); i <= n; i++ {
		res *= i
	}
	return res, nil
}

func modInverse(a, m int64) (int64, error) {
	if m == 0 {
		return 0, fmt.Errorf("ErrDivZero: modinv with modulus zero")
	}
	bigA := big.NewInt(a)
	bigM := big.NewInt(m)
	res := new(big.Int).ModInverse(bigA, bigM)
	if res == nil {
		return 0, fmt.Errorf("ErrDomain: %d has no modular inverse modulo %d", a, m)
	}
	return res.Int64(), nil
}

// -----------------------------------------------------------------------------
// Lexer & Tokens
// -----------------------------------------------------------------------------

type TokenType int

const (
	TokEOF TokenType = iota
	TokNum
	TokString
	TokIdent
	TokRegister
	TokPlus
	TokMinus
	TokStar
	TokSlash
	TokFloorDiv
	TokMod
	TokPow
	TokLParen
	TokRParen
	TokLBracket
	TokRBracket
	TokComma
	TokEq
	TokNeq
	TokLt
	TokLte
	TokGt
	TokGte
	TokAssign
	TokAt
)

type Token struct {
	Type   TokenType
	Text   string
	Col    int
	NumRat *big.Rat
	RegIdx int
}

type Lexer struct {
	src   []rune
	pos   int
	ch    rune
	isCAS bool
}

func NewLexer(src string) *Lexer {
	l := &Lexer{src: []rune(src), pos: 0}
	if len(l.src) > 0 {
		l.ch = l.src[0]
	}
	return l
}

func NewCASLexer(src string) *Lexer {
	l := NewLexer(src)
	l.isCAS = true
	return l
}

func (l *Lexer) next() {
	l.pos++
	if l.pos < len(l.src) {
		l.ch = l.src[l.pos]
	} else {
		l.ch = 0
	}
}

func (l *Lexer) peek() rune {
	if l.pos+1 < len(l.src) {
		return l.src[l.pos+1]
	}
	return 0
}

func (l *Lexer) skipWhitespace() {
	for l.ch != 0 && unicode.IsSpace(l.ch) {
		l.next()
	}
}

func (l *Lexer) NextToken() (Token, error) {
	l.skipWhitespace()
	if l.ch == 0 {
		return Token{Type: TokEOF, Col: l.pos}, nil
	}

	col := l.pos

	// Check for register tag: <|r0|> .. <|r15|>
	if l.ch == '<' && l.pos+5 < len(l.src) && l.src[l.pos+1] == '|' && l.src[l.pos+2] == 'r' {
		// Look for closing "|>"
		endIdx := -1
		for k := l.pos + 3; k < len(l.src) && k <= l.pos+6; k++ {
			if l.src[k] == '|' && k+1 < len(l.src) && l.src[k+1] == '>' {
				endIdx = k
				break
			}
		}
		if endIdx != -1 {
			numStr := string(l.src[l.pos+3 : endIdx])
			idx, err := strconv.Atoi(numStr)
			if err == nil && idx >= 0 && idx < NumRegisters {
				tagLen := (endIdx + 2) - l.pos
				tag := string(l.src[l.pos : endIdx+2])
				for k := 0; k < tagLen; k++ {
					l.next()
				}
				return Token{Type: TokRegister, Text: tag, Col: col, RegIdx: idx}, nil
			}
		}
	}

	// Two-character operators
	if l.ch == '/' && l.peek() == '/' {
		l.next()
		l.next()
		return Token{Type: TokFloorDiv, Text: "//", Col: col}, nil
	}
	if l.ch == '*' && l.peek() == '*' {
		l.next()
		l.next()
		return Token{Type: TokPow, Text: "**", Col: col}, nil
	}
	if l.ch == '=' && l.peek() == '=' {
		l.next()
		l.next()
		return Token{Type: TokEq, Text: "==", Col: col}, nil
	}
	if l.ch == '!' && l.peek() == '=' {
		l.next()
		l.next()
		return Token{Type: TokNeq, Text: "!=", Col: col}, nil
	}
	if l.ch == '<' && l.peek() == '=' {
		l.next()
		l.next()
		return Token{Type: TokLte, Text: "<=", Col: col}, nil
	}
	if l.ch == '>' && l.peek() == '=' {
		l.next()
		l.next()
		return Token{Type: TokGte, Text: ">=", Col: col}, nil
	}

	// Check for compound units: m/s^2, m/s, km/h
	remStr := string(l.src[l.pos:])
	if strings.HasPrefix(remStr, "m/s^2") && (len(remStr) == 5 || !unicode.IsLetter(rune(remStr[5]))) {
		for k := 0; k < 5; k++ {
			l.next()
		}
		return Token{Type: TokIdent, Text: "m/s^2", Col: col}, nil
	}
	if strings.HasPrefix(remStr, "m/s2") && (len(remStr) == 4 || !unicode.IsLetter(rune(remStr[4]))) {
		for k := 0; k < 4; k++ {
			l.next()
		}
		return Token{Type: TokIdent, Text: "m/s^2", Col: col}, nil
	}
	if strings.HasPrefix(remStr, "m/s") && (len(remStr) == 3 || !unicode.IsLetter(rune(remStr[3]))) {
		for k := 0; k < 3; k++ {
			l.next()
		}
		return Token{Type: TokIdent, Text: "m/s", Col: col}, nil
	}
	if strings.HasPrefix(remStr, "km/h") && (len(remStr) == 4 || !unicode.IsLetter(rune(remStr[4]))) {
		for k := 0; k < 4; k++ {
			l.next()
		}
		return Token{Type: TokIdent, Text: "km/h", Col: col}, nil
	}

	// Single-character operators
	switch l.ch {
	case '+':
		l.next()
		return Token{Type: TokPlus, Text: "+", Col: col}, nil
	case '-':
		l.next()
		return Token{Type: TokMinus, Text: "-", Col: col}, nil
	case '*':
		l.next()
		return Token{Type: TokStar, Text: "*", Col: col}, nil
	case '/':
		l.next()
		return Token{Type: TokSlash, Text: "/", Col: col}, nil
	case '%':
		l.next()
		return Token{Type: TokMod, Text: "%", Col: col}, nil
	case '^':
		l.next()
		return Token{Type: TokPow, Text: "^", Col: col}, nil
	case '(':
		l.next()
		return Token{Type: TokLParen, Text: "(", Col: col}, nil
	case ')':
		l.next()
		return Token{Type: TokRParen, Text: ")", Col: col}, nil
	case '[':
		l.next()
		return Token{Type: TokLBracket, Text: "[", Col: col}, nil
	case ']':
		l.next()
		return Token{Type: TokRBracket, Text: "]", Col: col}, nil
	case ',':
		l.next()
		return Token{Type: TokComma, Text: ",", Col: col}, nil
	case '<':
		l.next()
		return Token{Type: TokLt, Text: "<", Col: col}, nil
	case '>':
		l.next()
		return Token{Type: TokGt, Text: ">", Col: col}, nil
	case '=':
		l.next()
		return Token{Type: TokAssign, Text: "=", Col: col}, nil
	case '@':
		l.next()
		return Token{Type: TokAt, Text: "@", Col: col}, nil
	}

	// String literal
	if l.ch == '"' || l.ch == '\'' {
		quote := l.ch
		l.next()
		start := l.pos
		for l.ch != 0 && l.ch != quote {
			if l.ch == '\\' {
				l.next()
			}
			l.next()
		}
		if l.ch != quote {
			return Token{}, fmt.Errorf("ErrSyntax: unclosed string at column %d", col)
		}
		strVal := string(l.src[start:l.pos])
		l.next()
		return Token{Type: TokString, Text: strVal, Col: col}, nil
	}

	// Number literal
	if unicode.IsDigit(l.ch) || (l.ch == '.' && unicode.IsDigit(l.peek())) {
		start := l.pos
		hasDot := false
		for l.ch != 0 && (unicode.IsDigit(l.ch) || (!hasDot && l.ch == '.')) {
			if l.ch == '.' {
				hasDot = true
			}
			l.next()
		}
		// Scientific notation: 1.5e3, 1e-4
		if (l.ch == 'e' || l.ch == 'E') && (unicode.IsDigit(l.peek()) || ((l.peek() == '+' || l.peek() == '-') && l.pos+2 < len(l.src) && unicode.IsDigit(l.src[l.pos+2]))) {
			l.next()
			if l.ch == '+' || l.ch == '-' {
				l.next()
			}
			for l.ch != 0 && unicode.IsDigit(l.ch) {
				l.next()
			}
		}

		numStr := string(l.src[start:l.pos])
		rat := new(big.Rat)
		if _, ok := rat.SetString(numStr); !ok {
			f, err := strconv.ParseFloat(numStr, 64)
			if err != nil {
				return Token{}, fmt.Errorf("ErrSyntax: invalid number %q at column %d", numStr, col)
			}
			rat.SetFloat64(f)
		}

		// Check for implicit multiplication vs attached unit:
		// If directly followed by a letter:
		if unicode.IsLetter(l.ch) {
			if l.isCAS {
				return Token{}, fmt.Errorf("ErrSyntax: implicit multiplication not supported; use explicit '*'")
			}
			idStart := l.pos
			for l.ch != 0 && (unicode.IsLetter(l.ch) || unicode.IsDigit(l.ch) || l.ch == '_') {
				l.next()
			}
			trailingId := string(l.src[idStart:l.pos])
			if !IsKnownUnit(trailingId) {
				return Token{}, fmt.Errorf("ErrSyntax: implicit multiplication not supported; use explicit '*'")
			}
			// Backtrack so next token reads the attached unit
			l.pos = idStart
			l.ch = l.src[l.pos]
		}

		return Token{Type: TokNum, Text: numStr, NumRat: rat, Col: col}, nil
	}

	// Identifier or unit (never includes '/')
	if unicode.IsLetter(l.ch) || l.ch == '_' {
		start := l.pos
		for l.ch != 0 && (unicode.IsLetter(l.ch) || unicode.IsDigit(l.ch) || l.ch == '_') {
			l.next()
		}
		id := string(l.src[start:l.pos])
		return Token{Type: TokIdent, Text: id, Col: col}, nil
	}

	return Token{}, fmt.Errorf("ErrSyntax: unexpected character %q at column %d", string(l.ch), col)
}

// -----------------------------------------------------------------------------
// Parser
// -----------------------------------------------------------------------------

// Unit Disambiguation Rule:
// A unit identifier represents a physical unit only in quantity position:
// 1. Directly following a numeric literal (e.g. "3 m", "120 V", "101325 Pa", "7c")
// 2. Directly following a parenthesized numeric expression (e.g. "(-5) m", "(120) V")
// 3. Directly following a register containing a numeric quantity (e.g. "<|r0|> s")
// 4. Inside a compound unit that follows a quantity or unit register (e.g. "9.8 m/s^2", "N * m^2 / C^2")
// 5. As a string argument to convert() or lookup()
// Everywhere else, bare identifiers are symbolic polynomial variables (e.g. "5*m + 20", "k/2 - k/3", "(x - m) + (10 - x)").
// In solve, equations are pure algebraic CAS polynomials and units are never active.
func hasUnitContext(tokens []Token, store *RegisterStore) bool {
	for i, tok := range tokens {
		// Register holding a unit
		if tok.Type == TokRegister && store != nil {
			if val, err := store.Get(tok.RegIdx); err == nil && val.Kind == ValKindUnit {
				return true
			}
		}
		// Register followed by a known unit: e.g. <|r0|> s
		if tok.Type == TokRegister && i+1 < len(tokens) && tokens[i+1].Type == TokIdent && IsKnownUnit(tokens[i+1].Text) {
			return true
		}
		// Quantity position: number followed by a known unit (e.g. "3 m", "10 kg", "2 T")
		if tok.Type == TokNum && i+1 < len(tokens) && tokens[i+1].Type == TokIdent && IsKnownUnit(tokens[i+1].Text) {
			return true
		}
		// Quantity position: paren followed by a known unit (e.g. "(-5) m")
		if tok.Type == TokRParen && i+1 < len(tokens) && tokens[i+1].Type == TokIdent && IsKnownUnit(tokens[i+1].Text) {
			return true
		}
		// Explicit unit conversion or lookup call
		if tok.Type == TokIdent && (tok.Text == "convert" || tok.Text == "lookup") {
			return true
		}
	}
	return false
}

type Parser struct {
	tokens        []Token
	pos           int
	store         *RegisterStore
	hasUnits      bool
	inDenominator bool
}

func (p *Parser) isUnitChain(tok Token) bool {
	// 1. Followed by power operator: e.g. m^2, s^2, A^2, kg^2, cm^3
	if p.cur().Type == TokPow {
		return true
	}
	// 2. Unspaced compound unit directly attached to an operator: e.g. "N*m" in "3 N*m" or "g/mol"
	if p.pos >= 3 {
		prevOp := p.tokens[p.pos-2]
		prevTok := p.tokens[p.pos-3]
		if (prevOp.Type == TokStar || prevOp.Type == TokSlash) &&
			prevTok.Col+len(prevTok.Text) == prevOp.Col &&
			prevOp.Col+len(prevOp.Text) == tok.Col {
			return true
		}
	}
	// 3. In denominator of division (e.g. J / mol, J / (mol * K), g / mol, m / s)
	// 't' is excluded because 't' is never a denominator unit and collides with time
	if p.inDenominator && tok.Text != "t" {
		return true
	}
	// 4. Recognized compound unit product pair (e.g. J * s, m * K, N * m, mol * K, kg * K)
	if p.pos >= 3 && tok.Text != "t" {
		prevOp := p.tokens[p.pos-2]
		prevTok := p.tokens[p.pos-3]
		if prevOp.Type == TokStar && IsKnownUnit(prevTok.Text) {
			if isValidCompoundProduct(prevTok.Text, tok.Text) {
				return true
			}
		}
	}
	return false
}

func isValidCompoundProduct(u1, u2 string) bool {
	info1, ok1 := NormalizeUnit(u1)
	info2, ok2 := NormalizeUnit(u2)
	if !ok1 || !ok2 {
		return false
	}
	n1, n2 := info1.Name, info2.Name
	switch {
	case (n1 == "J" && n2 == "s") || (n1 == "s" && n2 == "J"):
		return true
	case (n1 == "m" && n2 == "K") || (n1 == "K" && n2 == "m"):
		return true
	case (n1 == "N" && n2 == "m") || (n1 == "m" && n2 == "N"):
		return true
	case (n1 == "W" && n2 == "s") || (n1 == "s" && n2 == "W"):
		return true
	case (n1 == "V" && n2 == "A") || (n1 == "A" && n2 == "V"):
		return true
	case (n1 == "A" && n2 == "s") || (n1 == "s" && n2 == "A"):
		return true
	case (n1 == "mol" && n2 == "K") || (n1 == "K" && n2 == "mol"):
		return true
	case (n1 == "kg" && n2 == "K") || (n1 == "K" && n2 == "kg"):
		return true
	}
	return false
}

func NewParser(tokens []Token, store *RegisterStore) *Parser {
	return &Parser{
		tokens:   tokens,
		pos:      0,
		store:    store,
		hasUnits: hasUnitContext(tokens, store),
	}
}

func (p *Parser) cur() Token {
	if p.pos < len(p.tokens) {
		return p.tokens[p.pos]
	}
	return Token{Type: TokEOF}
}

func (p *Parser) next() Token {
	t := p.cur()
	if p.pos < len(p.tokens) {
		p.pos++
	}
	return t
}

func (p *Parser) parseExpression() (CalcVal, error) {
	return p.parseComparison()
}

// parseComparison handles chained comparisons :
// "3 < 4 < 2" -> 3 < 4 and 4 < 2 -> false.
// Type strict: strings only with strings, units only with units.
func (p *Parser) parseComparison() (CalcVal, error) {
	left, err := p.parseAdditive()
	if err != nil {
		return CalcVal{}, err
	}

	isCompOp := func(t TokenType) bool {
		return t == TokEq || t == TokNeq || t == TokLt || t == TokLte || t == TokGt || t == TokGte || t == TokAssign
	}

	if !isCompOp(p.cur().Type) {
		return left, nil
	}

	// We have a comparison expression (potentially chained)
	var operands []CalcVal
	var operators []TokenType

	operands = append(operands, left)

	for isCompOp(p.cur().Type) {
		op := p.next()
		opType := op.Type
		if opType == TokAssign {
			// Single '=' in calc comparison is treated as ErrSyntax (L3)
			return CalcVal{}, fmt.Errorf("ErrSyntax: unexpected '=' in expression; use '==' for comparison")
		}
		operators = append(operators, opType)

		right, err := p.parseAdditive()
		if err != nil {
			return CalcVal{}, err
		}
		operands = append(operands, right)
	}

	// Evaluate all pairwise comparisons
	for i := 0; i < len(operators); i++ {
		v1 := operands[i]
		v2 := operands[i+1]
		op := operators[i]

		isExactKind := func(k CalcValKind) bool {
			return k == ValKindRat || k == ValKindSurd || k == ValKindExactNum
		}
		var cmp int
		var err error

		if (op == TokEq || op == TokNeq) && isExactKind(v1.Kind) && isExactKind(v2.Kind) {
			e1 := calcValToExactNum(v1)
			e2 := calcValToExactNum(v2)
			if e1 != nil && e2 != nil && (!e1.IsReal() || !e2.IsReal()) {
				diff := e1.Sub(e2)
				if diff.IsZero() {
					cmp = 0
				} else {
					cmp = 1
				}
			} else {
				cmp, err = compareCalcVals(v1, v2)
			}
		} else {
			cmp, err = compareCalcVals(v1, v2)
		}

		if err != nil {
			return CalcVal{}, err
		}

		var res bool
		switch op {
		case TokEq:
			res = (cmp == 0)
		case TokNeq:
			res = (cmp != 0)
		case TokLt:
			res = (cmp < 0)
		case TokLte:
			res = (cmp <= 0)
		case TokGt:
			res = (cmp > 0)
		case TokGte:
			res = (cmp >= 0)
		}

		if !res {
			// Any false in chain produces false
			return CalcVal{Kind: ValKindBool, Bool: false}, nil
		}
	}

	return CalcVal{Kind: ValKindBool, Bool: true}, nil
}

func compareCalcVals(v1, v2 CalcVal) (int, error) {
	// String comparisons: type-strict
	if v1.Kind == ValKindString || v2.Kind == ValKindString {
		if v1.Kind != ValKindString || v2.Kind != ValKindString {
			return 0, fmt.Errorf("ErrTypeMismatch: cannot compare string with non-string")
		}
		return strings.Compare(v1.Str, v2.Str), nil
	}

	// Unit comparisons: type-strict
	if v1.Kind == ValKindUnit || v2.Kind == ValKindUnit {
		if v1.Kind != ValKindUnit || v2.Kind != ValKindUnit {
			return 0, fmt.Errorf("ErrUnitMismatch: cannot compare unit value with scalar")
		}
		if v1.UnitName != v2.UnitName {
			conv, err := ConvertUnits(v2.UnitVal, v2.UnitName, v1.UnitName)
			if err != nil {
				return 0, err
			}
			return v1.UnitVal.Cmp(conv), nil
		}
		return v1.UnitVal.Cmp(v2.UnitVal), nil
	}

	// Boolean comparisons: only == and != allowed
	if v1.Kind == ValKindBool || v2.Kind == ValKindBool {
		if v1.Kind != ValKindBool || v2.Kind != ValKindBool {
			return 0, fmt.Errorf("ErrTypeMismatch: cannot compare boolean with non-boolean")
		}
		if v1.Bool == v2.Bool {
			return 0, nil
		}
		return -1, nil
	}

	// Exact rational comparison
	if v1.Kind == ValKindRat && v2.Kind == ValKindRat {
		return v1.Rat.Cmp(v2.Rat), nil
	}

	// Prohibit list comparisons (C15)
	if v1.Kind == ValKindList || v2.Kind == ValKindList {
		return 0, fmt.Errorf("ErrTypeMismatch: cannot compare list value")
	}

	isNumeric := func(k CalcValKind) bool {
		return k == ValKindRat || k == ValKindSurd || k == ValKindFloat || k == ValKindExactNum
	}
	if !isNumeric(v1.Kind) || !isNumeric(v2.Kind) {
		return 0, fmt.Errorf("ErrTypeMismatch: cannot compare incompatible types")
	}

	isExactKind := func(k CalcValKind) bool {
		return k == ValKindRat || k == ValKindSurd || k == ValKindExactNum
	}

	if isExactKind(v1.Kind) && isExactKind(v2.Kind) {
		e1 := calcValToExactNum(v1)
		e2 := calcValToExactNum(v2)
		if e1 != nil && e2 != nil {
			if !e1.IsReal() || !e2.IsReal() {
				diff := e1.Sub(e2)
				if diff.IsZero() {
					return 0, nil
				}
				return 0, fmt.Errorf("ErrDomain: cannot compare complex numbers with ordering operators")
			}
			diff := e1.Sub(e2)
			if diff.IsZero() {
				return 0, nil
			}
			if rat, ok := diff.AsRational(); ok {
				return rat.Sign(), nil
			}
			if len(diff.Terms) == 1 {
				for _, coeff := range diff.Terms {
					return coeff.Sign(), nil
				}
			}
		}
	}

	// General float comparison
	f1 := toFloat(v1)
	f2 := toFloat(v2)

	diffF := math.Abs(f1 - f2)
	scale := math.Max(math.Abs(f1), math.Abs(f2))

	if scale > 0 && diffF <= 4*floatUlp(scale) {
		return 0, fmt.Errorf("ErrUnsupported: cannot decide equality of inexact values")
	}

	if f1 < f2 {
		return -1, nil
	} else if f1 > f2 {
		return 1, nil
	}
	return 0, nil
}

func floatUlp(x float64) float64 {
	x = math.Abs(x)
	if math.IsNaN(x) || math.IsInf(x, 0) {
		return x
	}
	next := math.Nextafter(x, math.Inf(1))
	if math.IsInf(next, 0) {
		return x - math.Nextafter(x, 0)
	}
	return next - x
}

func toFloat(v CalcVal) float64 {
	switch v.Kind {
	case ValKindRat:
		if v.Rat == nil {
			return 0
		}
		f, _ := v.Rat.Float64()
		return f
	case ValKindSurd:
		if v.Surd == nil {
			return 0
		}
		return v.Surd.ToFloat64()
	case ValKindFloat:
		return v.Float
	case ValKindUnit:
		if v.UnitVal == nil {
			return 0
		}
		f, _ := v.UnitVal.Float64()
		return f
	case ValKindExactNum:
		if v.Exact == nil {
			return 0
		}
		return v.Exact.ToFloat64()
	case ValKindPoly:
		if v.Poly != nil {
			if c, ok := v.Poly.AsConstant(); ok {
				return c.ToFloat64()
			}
		}
		return math.NaN()
	default:
		return 0
	}
}

func isZeroCalcVal(v CalcVal) bool {
	switch v.Kind {
	case ValKindRat:
		return v.Rat != nil && v.Rat.Sign() == 0
	case ValKindFloat:
		return v.Float == 0
	case ValKindExactNum:
		return v.Exact == nil || v.Exact.IsZero()
	case ValKindSurd:
		return v.Surd == nil || v.Surd.a.Sign() == 0
	default:
		return false
	}
}

func sameTokens(t1, t2 []Token) bool {
	stripParens := func(ts []Token) []Token {
		for len(ts) >= 2 && ts[0].Type == TokLParen && ts[len(ts)-1].Type == TokRParen {
			depth := 0
			matchedAtEnd := false
			for i, t := range ts {
				if t.Type == TokLParen {
					depth++
				} else if t.Type == TokRParen {
					depth--
					if depth == 0 && i == len(ts)-1 {
						matchedAtEnd = true
					}
				}
			}
			if matchedAtEnd {
				ts = ts[1 : len(ts)-1]
			} else {
				break
			}
		}
		return ts
	}
	t1 = stripParens(t1)
	t2 = stripParens(t2)
	if len(t1) == 0 || len(t1) != len(t2) {
		return false
	}
	for i := range t1 {
		if t1[i].Type != t2[i].Type || t1[i].Text != t2[i].Text {
			return false
		}
	}
	return true
}

func (p *Parser) parseAdditive() (CalcVal, error) {
	startPos := p.pos
	left, err := p.parseMultiplicative()
	if err != nil {
		return CalcVal{}, err
	}

	for p.cur().Type == TokPlus || p.cur().Type == TokMinus {
		op := p.next()
		rightStart := p.pos
		right, err := p.parseMultiplicative()
		if err != nil {
			return CalcVal{}, err
		}

		if left.Kind == ValKindBool || right.Kind == ValKindBool {
			return CalcVal{}, fmt.Errorf("ErrSyntax: boolean arithmetic is not supported")
		}

		if op.Type == TokPlus {
			left, err = addCalcVals(left, right)
		} else {
			if startPos >= 0 && sameTokens(p.tokens[startPos:rightStart-1], p.tokens[rightStart:p.pos]) {
				left = CalcVal{Kind: ValKindRat, Rat: big.NewRat(0, 1)}
			} else {
				left, err = subCalcVals(left, right)
			}
		}
		if err != nil {
			return CalcVal{}, err
		}
		startPos = -1
	}

	return left, nil
}

func tryPromoteList(v CalcVal) CalcVal {
	if v.Kind == ValKindList {
		if m, err := valToMatrix(v); err == nil {
			return CalcVal{Kind: ValKindMatrix, Mat: m}
		}
		if vec, err := valToVector(v); err == nil {
			return CalcVal{Kind: ValKindVector, Vec: vec}
		}
	}
	return v
}

func addCalcVals(v1, v2 CalcVal) (CalcVal, error) {
	v1 = tryPromoteList(v1)
	v2 = tryPromoteList(v2)
	// Unit handling
	if v1.Kind == ValKindUnit || v2.Kind == ValKindUnit {
		if v1.Kind != ValKindUnit || v2.Kind != ValKindUnit {
			return CalcVal{}, fmt.Errorf("ErrUnitMismatch: cannot perform operation on unit and dimensionless quantity")
		}
		if v1.UnitName != v2.UnitName {
			return CalcVal{}, fmt.Errorf("ErrUnitMismatch: cannot add different units %s and %s directly; use convert()", v1.UnitName, v2.UnitName)
		}
		res := new(big.Rat).Add(v1.UnitVal, v2.UnitVal)
		return CalcVal{Kind: ValKindUnit, UnitVal: res, UnitName: v1.UnitName}, nil
	}

	// Polynomial addition
	if v1.Kind == ValKindPoly || v2.Kind == ValKindPoly {
		p1 := calcValToPoly(v1)
		p2 := calcValToPoly(v2)
		if p1 != nil && p2 != nil {
			res := p1.Add(p2)
			if c, ok := res.AsConstant(); ok {
				if rat, isRat := c.AsRational(); isRat {
					return CalcVal{Kind: ValKindRat, Rat: rat}, nil
				}
				return CalcVal{Kind: ValKindExactNum, Exact: c}, nil
			}
			return CalcVal{Kind: ValKindPoly, Poly: res}, nil
		}
	}

	// Matrix addition
	if v1.Kind == ValKindMatrix && v2.Kind == ValKindMatrix {
		m, err := v1.Mat.Add(v2.Mat)
		if err != nil {
			return CalcVal{}, err
		}
		return CalcVal{Kind: ValKindMatrix, Mat: m}, nil
	}

	// Vector addition
	if v1.Kind == ValKindVector && v2.Kind == ValKindVector {
		v, err := v1.Vec.Add(v2.Vec)
		if err != nil {
			return CalcVal{}, err
		}
		return CalcVal{Kind: ValKindVector, Vec: v}, nil
	}

	// ExactNum addition
	if v1.Kind == ValKindExactNum || v2.Kind == ValKindExactNum ||
		(v1.Kind == ValKindSurd && v2.Kind == ValKindRat) ||
		(v1.Kind == ValKindRat && v2.Kind == ValKindSurd) ||
		(v1.Kind == ValKindSurd && v2.Kind == ValKindSurd && v1.Surd.b.Cmp(v2.Surd.b) != 0) {
		e1 := calcValToExactNum(v1)
		e2 := calcValToExactNum(v2)
		sum := e1.Add(e2)
		if rat, ok := sum.AsRational(); ok {
			return CalcVal{Kind: ValKindRat, Rat: rat}, nil
		}
		return CalcVal{Kind: ValKindExactNum, Exact: sum}, nil
	}

	// Surd addition
	if v1.Kind == ValKindSurd && v2.Kind == ValKindSurd {
		if v1.Surd.b.Cmp(v2.Surd.b) == 0 {
			num1 := new(big.Int).Mul(big.NewInt(int64(v1.Surd.sign)), v1.Surd.a)
			r1 := new(big.Rat).SetFrac(num1, v1.Surd.c)
			num2 := new(big.Int).Mul(big.NewInt(int64(v2.Surd.sign)), v2.Surd.a)
			r2 := new(big.Rat).SetFrac(num2, v2.Surd.c)
			r := new(big.Rat).Add(r1, r2)
			if r.Sign() == 0 {
				return CalcVal{Kind: ValKindRat, Rat: big.NewRat(0, 1)}, nil
			}
			return CalcVal{
				Kind: ValKindSurd,
				Surd: &SurdVal{
					sign: r.Sign(),
					a:    new(big.Int).Abs(r.Num()),
					b:    new(big.Int).Set(v1.Surd.b),
					c:    new(big.Int).Set(r.Denom()),
				},
			}, nil
		}
		return CalcVal{Kind: ValKindFloat, Float: v1.Surd.ToFloat64() + v2.Surd.ToFloat64()}, nil
	}

	if v1.Kind == ValKindSurd || v2.Kind == ValKindSurd || v1.Kind == ValKindFloat || v2.Kind == ValKindFloat {
		f1 := toFloat(v1)
		f2 := toFloat(v2)
		sum := f1 + f2
		if (f1 > 0 && f2 < 0) || (f1 < 0 && f2 > 0) {
			scale := math.Max(math.Abs(f1), math.Abs(f2))
			if scale > 0 && math.Abs(sum) <= 4*floatUlp(scale) {
				return CalcVal{}, fmt.Errorf("ErrUnsupported: result is below floating-point precision")
			}
		}
		return CalcVal{Kind: ValKindFloat, Float: sum}, nil
	}

	if v1.Kind == ValKindRat && v2.Kind == ValKindRat {
		res := new(big.Rat).Add(v1.Rat, v2.Rat)
		return CalcVal{Kind: ValKindRat, Rat: res, ForcedFrac: v1.ForcedFrac || v2.ForcedFrac, ForcedDec: v1.ForcedDec || v2.ForcedDec}, nil
	}

	return CalcVal{}, fmt.Errorf("ErrTypeMismatch: invalid addition operands")
}

func subCalcVals(v1, v2 CalcVal) (CalcVal, error) {
	v1 = tryPromoteList(v1)
	v2 = tryPromoteList(v2)
	// Unit handling
	if v1.Kind == ValKindUnit || v2.Kind == ValKindUnit {
		if v1.Kind != ValKindUnit || v2.Kind != ValKindUnit {
			return CalcVal{}, fmt.Errorf("ErrUnitMismatch: cannot perform operation on unit and dimensionless quantity")
		}
		if v1.UnitName != v2.UnitName {
			return CalcVal{}, fmt.Errorf("ErrUnitMismatch: cannot subtract different units %s and %s directly; use convert()", v1.UnitName, v2.UnitName)
		}
		res := new(big.Rat).Sub(v1.UnitVal, v2.UnitVal)
		return CalcVal{Kind: ValKindUnit, UnitVal: res, UnitName: v1.UnitName}, nil
	}

	// Polynomial subtraction
	if v1.Kind == ValKindPoly || v2.Kind == ValKindPoly {
		p1 := calcValToPoly(v1)
		p2 := calcValToPoly(v2)
		if p1 != nil && p2 != nil {
			res := p1.Sub(p2)
			if c, ok := res.AsConstant(); ok {
				if rat, isRat := c.AsRational(); isRat {
					return CalcVal{Kind: ValKindRat, Rat: rat}, nil
				}
				return CalcVal{Kind: ValKindExactNum, Exact: c}, nil
			}
			return CalcVal{Kind: ValKindPoly, Poly: res}, nil
		}
	}

	// Matrix subtraction
	if v1.Kind == ValKindMatrix && v2.Kind == ValKindMatrix {
		m, err := v1.Mat.Sub(v2.Mat)
		if err != nil {
			return CalcVal{}, err
		}
		return CalcVal{Kind: ValKindMatrix, Mat: m}, nil
	}

	// Vector subtraction
	if v1.Kind == ValKindVector && v2.Kind == ValKindVector {
		v, err := v1.Vec.Sub(v2.Vec)
		if err != nil {
			return CalcVal{}, err
		}
		return CalcVal{Kind: ValKindVector, Vec: v}, nil
	}

	// ExactNum subtraction
	if v1.Kind == ValKindExactNum || v2.Kind == ValKindExactNum ||
		(v1.Kind == ValKindSurd && v2.Kind == ValKindRat) ||
		(v1.Kind == ValKindRat && v2.Kind == ValKindSurd) ||
		(v1.Kind == ValKindSurd && v2.Kind == ValKindSurd && v1.Surd.b.Cmp(v2.Surd.b) != 0) {
		e1 := calcValToExactNum(v1)
		e2 := calcValToExactNum(v2)
		diff := e1.Sub(e2)
		if rat, ok := diff.AsRational(); ok {
			return CalcVal{Kind: ValKindRat, Rat: rat}, nil
		}
		return CalcVal{Kind: ValKindExactNum, Exact: diff}, nil
	}

	if v1.Kind == ValKindSurd && v2.Kind == ValKindSurd {
		negV2 := v2
		negSurd := v2.Surd.Clone()
		negSurd.sign = -negSurd.sign
		negV2.Surd = negSurd
		return addCalcVals(v1, negV2)
	}

	if v1.Kind == ValKindSurd || v2.Kind == ValKindSurd || v1.Kind == ValKindFloat || v2.Kind == ValKindFloat {
		f1 := toFloat(v1)
		f2 := toFloat(v2)
		diff := f1 - f2
		scale := math.Max(math.Abs(f1), math.Abs(f2))
		if (f1 > 0 && f2 > 0) || (f1 < 0 && f2 < 0) {
			if scale > 0 && math.Abs(diff) <= 4*floatUlp(scale) {
				return CalcVal{}, fmt.Errorf("ErrUnsupported: result is below floating-point precision")
			}
		}
		return CalcVal{Kind: ValKindFloat, Float: diff}, nil
	}

	if v1.Kind == ValKindRat && v2.Kind == ValKindRat {
		res := new(big.Rat).Sub(v1.Rat, v2.Rat)
		return CalcVal{Kind: ValKindRat, Rat: res, ForcedFrac: v1.ForcedFrac || v2.ForcedFrac, ForcedDec: v1.ForcedDec || v2.ForcedDec}, nil
	}

	return CalcVal{}, fmt.Errorf("ErrTypeMismatch: invalid subtraction operands")
}

func (p *Parser) parseMultiplicative() (CalcVal, error) {
	startPos := p.pos
	left, err := p.parseUnary()
	if err != nil {
		return CalcVal{}, err
	}

	for p.cur().Type == TokStar || p.cur().Type == TokSlash || p.cur().Type == TokFloorDiv || p.cur().Type == TokMod || p.cur().Type == TokAt {
		op := p.next()
		rightStart := p.pos
		oldDenom := p.inDenominator
		if op.Type == TokSlash {
			p.inDenominator = true
		}
		right, err := p.parseUnary()
		p.inDenominator = oldDenom
		if err != nil {
			return CalcVal{}, err
		}

		switch op.Type {
		case TokStar:
			left, err = mulCalcVals(left, right)
		case TokSlash:
			if startPos >= 0 && sameTokens(p.tokens[startPos:rightStart-1], p.tokens[rightStart:p.pos]) && !isZeroCalcVal(right) {
				left = CalcVal{Kind: ValKindRat, Rat: big.NewRat(1, 1)}
			} else {
				left, err = divCalcVals(left, right)
			}
		case TokFloorDiv:
			left, err = floorDivCalcVals(left, right)
		case TokMod:
			left, err = modCalcVals(left, right)
		case TokAt:
			m1, err1 := valToMatrix(left)
			if err1 != nil {
				return CalcVal{}, err1
			}
			m2, err2 := valToMatrix(right)
			if err2 != nil {
				return CalcVal{}, err2
			}
			prod, errM := m1.Mul(m2)
			if errM != nil {
				return CalcVal{}, errM
			}
			left = CalcVal{Kind: ValKindMatrix, Mat: prod}
		}
		if err != nil {
			return CalcVal{}, err
		}
		startPos = -1
	}

	return left, nil
}

func mulCalcVals(v1, v2 CalcVal) (CalcVal, error) {
	v1 = tryPromoteList(v1)
	v2 = tryPromoteList(v2)
	// Unit x scalar -> unit
	if v1.Kind == ValKindUnit && v2.Kind == ValKindRat {
		res := new(big.Rat).Mul(v1.UnitVal, v2.Rat)
		return CalcVal{Kind: ValKindUnit, UnitVal: res, UnitName: v1.UnitName}, nil
	}
	if v1.Kind == ValKindRat && v2.Kind == ValKindUnit {
		res := new(big.Rat).Mul(v1.Rat, v2.UnitVal)
		return CalcVal{Kind: ValKindUnit, UnitVal: res, UnitName: v2.UnitName}, nil
	}
	if v1.Kind == ValKindUnit && v2.Kind == ValKindFloat {
		r := new(big.Rat).SetFloat64(v2.Float)
		res := new(big.Rat).Mul(v1.UnitVal, r)
		return CalcVal{Kind: ValKindUnit, UnitVal: res, UnitName: v1.UnitName}, nil
	}
	if v1.Kind == ValKindFloat && v2.Kind == ValKindUnit {
		r := new(big.Rat).SetFloat64(v1.Float)
		res := new(big.Rat).Mul(r, v2.UnitVal)
		return CalcVal{Kind: ValKindUnit, UnitVal: res, UnitName: v2.UnitName}, nil
	}
	if v1.Kind == ValKindUnit && v2.Kind == ValKindUnit {
		resVal, comb, isDimless, err := MultiplyUnits(v1.UnitVal, v1.UnitName, v2.UnitVal, v2.UnitName)
		if err != nil {
			return CalcVal{}, err
		}
		if isDimless {
			return CalcVal{Kind: ValKindRat, Rat: resVal}, nil
		}
		return CalcVal{Kind: ValKindUnit, UnitVal: resVal, UnitName: comb}, nil
	}
	if v1.Kind == ValKindUnit || v2.Kind == ValKindUnit {
		return CalcVal{}, fmt.Errorf("ErrUnitMismatch: unit multiplication is not supported")
	}

	// Polynomial multiplication
	if v1.Kind == ValKindPoly || v2.Kind == ValKindPoly {
		p1 := calcValToPoly(v1)
		p2 := calcValToPoly(v2)
		if p1 != nil && p2 != nil {
			res, err := p1.Mul(p2)
			if err != nil {
				return CalcVal{}, err
			}
			if c, ok := res.AsConstant(); ok {
				if rat, isRat := c.AsRational(); isRat {
					return CalcVal{Kind: ValKindRat, Rat: rat}, nil
				}
				return CalcVal{Kind: ValKindExactNum, Exact: c}, nil
			}
			return CalcVal{Kind: ValKindPoly, Poly: res}, nil
		}
	}

	// Matrix & Vector multiplication
	if v1.Kind == ValKindMatrix && (v2.Kind == ValKindRat || v2.Kind == ValKindExactNum) {
		s := calcValToExactNum(v2)
		return CalcVal{Kind: ValKindMatrix, Mat: v1.Mat.MulScalar(s)}, nil
	}
	if (v1.Kind == ValKindRat || v1.Kind == ValKindExactNum) && v2.Kind == ValKindMatrix {
		s := calcValToExactNum(v1)
		return CalcVal{Kind: ValKindMatrix, Mat: v2.Mat.MulScalar(s)}, nil
	}
	if v1.Kind == ValKindVector && (v2.Kind == ValKindRat || v2.Kind == ValKindExactNum) {
		s := calcValToExactNum(v2)
		return CalcVal{Kind: ValKindVector, Vec: v1.Vec.MulScalar(s)}, nil
	}
	if (v1.Kind == ValKindRat || v1.Kind == ValKindExactNum) && v2.Kind == ValKindVector {
		s := calcValToExactNum(v1)
		return CalcVal{Kind: ValKindVector, Vec: v2.Vec.MulScalar(s)}, nil
	}
	if v1.Kind == ValKindMatrix && v2.Kind == ValKindMatrix {
		m, err := v1.Mat.Mul(v2.Mat)
		if err != nil {
			return CalcVal{}, err
		}
		return CalcVal{Kind: ValKindMatrix, Mat: m}, nil
	}

	// ExactNum multiplication
	if v1.Kind == ValKindExactNum || v2.Kind == ValKindExactNum {
		e1 := calcValToExactNum(v1)
		e2 := calcValToExactNum(v2)
		prod, ok := e1.Mul(e2)
		if ok {
			if rat, isR := prod.AsRational(); isR {
				return CalcVal{Kind: ValKindRat, Rat: rat}, nil
			}
			return CalcVal{Kind: ValKindExactNum, Exact: prod}, nil
		}
	}

	// Surd multiplication
	if v1.Kind == ValKindSurd && v2.Kind == ValKindSurd {
		sign := v1.Surd.sign * v2.Surd.sign
		numA := new(big.Int).Mul(v1.Surd.a, v2.Surd.a)
		denC := new(big.Int).Mul(v1.Surd.c, v2.Surd.c)
		radB := new(big.Int).Mul(v1.Surd.b, v2.Surd.b)

		// Simplify radB
		extracted, squareFree, err := simplifyRadicalBig(radB, 50000)
		if err != nil {
			return CalcVal{}, err
		}
		totalA := new(big.Int).Mul(numA, extracted)
		g := new(big.Int).GCD(nil, nil, totalA, denC)
		totalA.Quo(totalA, g)
		denC.Quo(denC, g)

		if squareFree.Cmp(big.NewInt(1)) == 0 {
			// Radical simplified completely to rational!
			r := new(big.Rat).SetFrac(totalA, denC)
			if sign < 0 {
				r.Neg(r)
			}
			return CalcVal{Kind: ValKindRat, Rat: r}, nil
		}

		return CalcVal{
			Kind: ValKindSurd,
			Surd: &SurdVal{
				sign: sign,
				a:    totalA,
				b:    squareFree,
				c:    denC,
			},
		}, nil
	}

	// Surd x Rational
	if v1.Kind == ValKindSurd && v2.Kind == ValKindRat {
		if v2.Rat.Sign() == 0 {
			return CalcVal{Kind: ValKindRat, Rat: big.NewRat(0, 1)}, nil
		}
		sign := v1.Surd.sign * v2.Rat.Sign()
		absNum := new(big.Int).Abs(v2.Rat.Num())
		absDen := v2.Rat.Denom()

		totalA := new(big.Int).Mul(v1.Surd.a, absNum)
		totalC := new(big.Int).Mul(v1.Surd.c, absDen)
		g := new(big.Int).GCD(nil, nil, totalA, totalC)
		totalA.Quo(totalA, g)
		totalC.Quo(totalC, g)

		return CalcVal{
			Kind: ValKindSurd,
			Surd: &SurdVal{
				sign: sign,
				a:    totalA,
				b:    new(big.Int).Set(v1.Surd.b),
				c:    totalC,
			},
		}, nil
	}
	if v1.Kind == ValKindRat && v2.Kind == ValKindSurd {
		return mulCalcVals(v2, v1)
	}

	if v1.Kind == ValKindExactNum || v2.Kind == ValKindExactNum || v1.Kind == ValKindSurd || v2.Kind == ValKindSurd || v1.Kind == ValKindFloat || v2.Kind == ValKindFloat {
		return CalcVal{Kind: ValKindFloat, Float: toFloat(v1) * toFloat(v2)}, nil
	}

	if v1.Kind == ValKindRat && v2.Kind == ValKindRat {
		res := new(big.Rat).Mul(v1.Rat, v2.Rat)
		return CalcVal{Kind: ValKindRat, Rat: res, ForcedFrac: v1.ForcedFrac || v2.ForcedFrac, ForcedDec: v1.ForcedDec || v2.ForcedDec}, nil
	}

	return CalcVal{}, fmt.Errorf("ErrTypeMismatch: invalid multiplication operands")
}

func divCalcVals(v1, v2 CalcVal) (CalcVal, error) {
	v1 = tryPromoteList(v1)
	v2 = tryPromoteList(v2)
	// Division by zero check
	if (v2.Kind == ValKindRat && v2.Rat.Sign() == 0) ||
		(v2.Kind == ValKindFloat && v2.Float == 0) ||
		(v2.Kind == ValKindSurd && v2.Surd.a.Sign() == 0) {
		return CalcVal{}, fmt.Errorf("ErrDivZero: division by zero")
	}

	// Unit / scalar -> unit
	if v1.Kind == ValKindUnit && v2.Kind == ValKindRat {
		res := new(big.Rat).Quo(v1.UnitVal, v2.Rat)
		return CalcVal{Kind: ValKindUnit, UnitVal: res, UnitName: v1.UnitName}, nil
	}
	if v1.Kind == ValKindUnit && v2.Kind == ValKindFloat {
		r := new(big.Rat).SetFloat64(v2.Float)
		res := new(big.Rat).Quo(v1.UnitVal, r)
		return CalcVal{Kind: ValKindUnit, UnitVal: res, UnitName: v1.UnitName}, nil
	}
	// Unit / Unit -> general dimensional analysis
	if v1.Kind == ValKindUnit && v2.Kind == ValKindUnit {
		resVal, divUnit, isDimless, err := DivideUnits(v1.UnitVal, v1.UnitName, v2.UnitVal, v2.UnitName)
		if err != nil {
			return CalcVal{}, err
		}
		if isDimless {
			return CalcVal{Kind: ValKindRat, Rat: resVal}, nil
		}
		return CalcVal{Kind: ValKindUnit, UnitVal: resVal, UnitName: divUnit}, nil
	}
	// Scalar / Unit -> general dimensional analysis
	if v1.Kind == ValKindRat && v2.Kind == ValKindUnit {
		resVal, divUnit, isDimless, err := DivideUnits(v1.Rat, "", v2.UnitVal, v2.UnitName)
		if err != nil {
			return CalcVal{}, err
		}
		if isDimless {
			return CalcVal{Kind: ValKindRat, Rat: resVal}, nil
		}
		return CalcVal{Kind: ValKindUnit, UnitVal: resVal, UnitName: divUnit}, nil
	}
	if v1.Kind == ValKindFloat && v2.Kind == ValKindUnit {
		r := new(big.Rat).SetFloat64(v1.Float)
		resVal, divUnit, isDimless, err := DivideUnits(r, "", v2.UnitVal, v2.UnitName)
		if err != nil {
			return CalcVal{}, err
		}
		if isDimless {
			f, _ := resVal.Float64()
			return CalcVal{Kind: ValKindFloat, Float: f}, nil
		}
		return CalcVal{Kind: ValKindUnit, UnitVal: resVal, UnitName: divUnit}, nil
	}
	if v1.Kind == ValKindUnit || v2.Kind == ValKindUnit {
		return CalcVal{}, fmt.Errorf("ErrUnitMismatch: invalid unit division")
	}

	// Polynomial division
	if v1.Kind == ValKindPoly || v2.Kind == ValKindPoly {
		p1 := calcValToPoly(v1)
		p2 := calcValToPoly(v2)
		if p1 != nil && p2 != nil {
			if c, ok := p2.AsConstant(); ok {
				if c.IsZero() {
					return CalcVal{}, fmt.Errorf("ErrDivZero: division by zero in polynomial")
				}
				invC, okInv, errInv := NewExactInt(1).Div(c)
				if errInv != nil {
					return CalcVal{}, errInv
				}
				if !okInv {
					return CalcVal{}, fmt.Errorf("ErrUnsupported: exact division by constant %s not possible", c.String())
				}
				res, err := p1.Mul(NewMultiPolyConst(invC))
				if err != nil {
					return CalcVal{}, err
				}
				if rc, rok := res.AsConstant(); rok {
					if rat, isRat := rc.AsRational(); isRat {
						return CalcVal{Kind: ValKindRat, Rat: rat}, nil
					}
					return CalcVal{Kind: ValKindExactNum, Exact: rc}, nil
				}
				return CalcVal{Kind: ValKindPoly, Poly: res}, nil
			}
			pRat, v1, err1 := MultiPolyToRatPoly(p1)
			qRat, v2, err2 := MultiPolyToRatPoly(p2)
			if err1 == nil && err2 == nil && (v1 == v2 || qRat.Degree() == 0 || pRat.Degree() == 0) {
				if qRat.Degree() == 0 && qRat[0].Sign() == 0 {
					return CalcVal{}, fmt.Errorf("ErrDivZero: division by zero in polynomial")
				}
				quo, rem := PolyDivRemRat(pRat, qRat)
				if isPolyZero(rem) {
					res := RatPolyToMultiPoly(quo, v1)
					if c, ok := res.AsConstant(); ok {
						if rat, isRat := c.AsRational(); isRat {
							return CalcVal{Kind: ValKindRat, Rat: rat}, nil
						}
						return CalcVal{Kind: ValKindExactNum, Exact: c}, nil
					}
					return CalcVal{Kind: ValKindPoly, Poly: res}, nil
				}
			}
			return CalcVal{}, fmt.Errorf("ErrUnsupported: polynomial division with remainder; use divide(P, Q)")
		}
	}

	// Matrix & Vector division by scalar
	if v1.Kind == ValKindMatrix && (v2.Kind == ValKindRat || v2.Kind == ValKindExactNum) {
		s := calcValToExactNum(v2)
		invS, ok, err := NewExactInt(1).Div(s)
		if err != nil {
			return CalcVal{}, err
		}
		if ok {
			return CalcVal{Kind: ValKindMatrix, Mat: v1.Mat.MulScalar(invS)}, nil
		}
	}
	if v1.Kind == ValKindVector && (v2.Kind == ValKindRat || v2.Kind == ValKindExactNum) {
		s := calcValToExactNum(v2)
		invS, ok, err := NewExactInt(1).Div(s)
		if err != nil {
			return CalcVal{}, err
		}
		if ok {
			return CalcVal{Kind: ValKindVector, Vec: v1.Vec.MulScalar(invS)}, nil
		}
	}

	// ExactNum division
	if v1.Kind == ValKindExactNum || v2.Kind == ValKindExactNum {
		e1 := calcValToExactNum(v1)
		e2 := calcValToExactNum(v2)
		quot, ok, err := e1.Div(e2)
		if err != nil {
			return CalcVal{}, err
		}
		if ok {
			if rat, isR := quot.AsRational(); isR {
				return CalcVal{Kind: ValKindRat, Rat: rat}, nil
			}
			return CalcVal{Kind: ValKindExactNum, Exact: quot}, nil
		}
	}

	// Surd / Rational: e.g. sqrt(8)/2 -> 2*sqrt(2)/2 -> sqrt(2)
	if v1.Kind == ValKindSurd && v2.Kind == ValKindRat {
		absNum := new(big.Int).Abs(v2.Rat.Num())
		absDen := v2.Rat.Denom()
		sign := v1.Surd.sign * v2.Rat.Sign()

		totalA := new(big.Int).Mul(v1.Surd.a, absDen)
		totalC := new(big.Int).Mul(v1.Surd.c, absNum)
		g := new(big.Int).GCD(nil, nil, totalA, totalC)
		totalA.Quo(totalA, g)
		totalC.Quo(totalC, g)

		return CalcVal{
			Kind: ValKindSurd,
			Surd: &SurdVal{
				sign: sign,
				a:    totalA,
				b:    new(big.Int).Set(v1.Surd.b),
				c:    totalC,
			},
		}, nil
	}

	// Rational / Surd: e.g. 2/sqrt(2) -> 2*sqrt(2)/2 = sqrt(2) (M3)
	if v1.Kind == ValKindRat && v2.Kind == ValKindSurd {
		// r / (s * a * sqrt(b) / c) = (r * c) / (s * a * sqrt(b))
		// Rationalize denominator: multiply by sqrt(b) / sqrt(b)
		// = (r * c * sqrt(b)) / (s * a * b)
		sign := v1.Rat.Sign() * v2.Surd.sign
		absNum := new(big.Int).Abs(v1.Rat.Num())
		absDen := v1.Rat.Denom()

		totalA := new(big.Int).Mul(absNum, v2.Surd.c)
		prod := new(big.Int).Mul(v2.Surd.a, v2.Surd.b)
		totalC := new(big.Int).Mul(absDen, prod)
		g := new(big.Int).GCD(nil, nil, totalA, totalC)
		totalA.Quo(totalA, g)
		totalC.Quo(totalC, g)

		return CalcVal{
			Kind: ValKindSurd,
			Surd: &SurdVal{
				sign: sign,
				a:    totalA,
				b:    new(big.Int).Set(v2.Surd.b),
				c:    totalC,
			},
		}, nil
	}

	// Surd / Surd:
	if v1.Kind == ValKindSurd && v2.Kind == ValKindSurd {
		// (s1*a1*sqrt(b1)/c1) / (s2*a2*sqrt(b2)/c2)
		// = (s1*s2 * a1*c2 * sqrt(b1*b2)) / (a2*c1 * b2)
		sign := v1.Surd.sign * v2.Surd.sign
		rad := new(big.Int).Mul(v1.Surd.b, v2.Surd.b)
		extracted, squareFree, err := simplifyRadicalBig(rad, 50000)
		if err != nil {
			return CalcVal{}, err
		}

		numA := new(big.Int).Mul(v1.Surd.a, v2.Surd.c)
		totalA := new(big.Int).Mul(numA, extracted)
		denC := new(big.Int).Mul(v2.Surd.a, v1.Surd.c)
		totalC := new(big.Int).Mul(denC, v2.Surd.b)
		g := new(big.Int).GCD(nil, nil, totalA, totalC)
		totalA.Quo(totalA, g)
		totalC.Quo(totalC, g)

		if squareFree.Cmp(big.NewInt(1)) == 0 {
			resRat := new(big.Rat).SetFrac(totalA, totalC)
			if sign < 0 {
				resRat.Neg(resRat)
			}
			return CalcVal{
				Kind: ValKindRat,
				Rat:  resRat,
			}, nil
		}
		return CalcVal{
			Kind: ValKindSurd,
			Surd: &SurdVal{
				sign: sign,
				a:    totalA,
				b:    squareFree,
				c:    totalC,
			},
		}, nil
	}

	if v1.Kind == ValKindExactNum || v2.Kind == ValKindExactNum || v1.Kind == ValKindSurd || v2.Kind == ValKindSurd || v1.Kind == ValKindFloat || v2.Kind == ValKindFloat {
		f2 := toFloat(v2)
		if f2 == 0 {
			return CalcVal{}, fmt.Errorf("ErrDivZero: division by zero")
		}
		return CalcVal{Kind: ValKindFloat, Float: toFloat(v1) / f2}, nil
	}

	if v1.Kind == ValKindRat && v2.Kind == ValKindRat {
		res := new(big.Rat).Quo(v1.Rat, v2.Rat)
		return CalcVal{Kind: ValKindRat, Rat: res, ForcedFrac: v1.ForcedFrac || v2.ForcedFrac, ForcedDec: v1.ForcedDec || v2.ForcedDec}, nil
	}

	return CalcVal{}, fmt.Errorf("ErrTypeMismatch: invalid division operands")
}

// floorDivCalcVals computes exact floor division // .
func floorDivCalcVals(v1, v2 CalcVal) (CalcVal, error) {
	if (v2.Kind == ValKindRat && v2.Rat.Sign() == 0) ||
		(v2.Kind == ValKindFloat && v2.Float == 0) {
		return CalcVal{}, fmt.Errorf("ErrDivZero: floor division by zero")
	}

	if v1.Kind == ValKindRat && v2.Kind == ValKindRat {
		// quotient = v1 / v2
		quo := new(big.Rat).Quo(v1.Rat, v2.Rat)
		// floor(quo)
		q := new(big.Int)
		rem := new(big.Int)
		q.QuoRem(quo.Num(), quo.Denom(), rem)
		if quo.Sign() < 0 && rem.Sign() != 0 {
			q.Sub(q, big.NewInt(1))
		}
		return CalcVal{Kind: ValKindRat, Rat: new(big.Rat).SetInt(q)}, nil
	}

	f1 := toFloat(v1)
	f2 := toFloat(v2)
	res := math.Floor(f1 / f2)
	return CalcVal{Kind: ValKindFloat, Float: res}, nil
}

// modCalcVals computes exact floor modulo % : a - b * floor(a/b).
func modCalcVals(v1, v2 CalcVal) (CalcVal, error) {
	if (v2.Kind == ValKindRat && v2.Rat.Sign() == 0) ||
		(v2.Kind == ValKindFloat && v2.Float == 0) {
		return CalcVal{}, fmt.Errorf("ErrDivZero: modulo by zero")
	}

	if v1.Kind == ValKindRat && v2.Kind == ValKindRat {
		// q = floor(v1 / v2)
		quo := new(big.Rat).Quo(v1.Rat, v2.Rat)
		q := new(big.Int)
		rem := new(big.Int)
		q.QuoRem(quo.Num(), quo.Denom(), rem)
		if quo.Sign() < 0 && rem.Sign() != 0 {
			q.Sub(q, big.NewInt(1))
		}
		// r = v1 - v2 * q
		qRat := new(big.Rat).SetInt(q)
		term := new(big.Rat).Mul(v2.Rat, qRat)
		r := new(big.Rat).Sub(v1.Rat, term)
		return CalcVal{Kind: ValKindRat, Rat: r}, nil
	}

	f1 := toFloat(v1)
	f2 := toFloat(v2)
	// Python floor modulo on floats: f1 - f2 * floor(f1 / f2)
	res := f1 - f2*math.Floor(f1/f2)
	return CalcVal{Kind: ValKindFloat, Float: res}, nil
}

func (p *Parser) parseUnary() (CalcVal, error) {
	tok := p.cur()
	if tok.Type == TokMinus {
		p.next()
		// Unary minus binds looser than exponentiation: -x^2 = -(x^2)
		val, err := p.parsePower()
		if err != nil {
			return CalcVal{}, err
		}
		switch val.Kind {
		case ValKindRat:
			return CalcVal{Kind: ValKindRat, Rat: new(big.Rat).Neg(val.Rat), ForcedFrac: val.ForcedFrac, ForcedDec: val.ForcedDec}, nil
		case ValKindSurd:
			cp := val.Surd.Clone()
			cp.sign = -cp.sign
			return CalcVal{Kind: ValKindSurd, Surd: cp}, nil
		case ValKindExactNum:
			return CalcVal{Kind: ValKindExactNum, Exact: val.Exact.Neg()}, nil
		case ValKindPoly:
			return CalcVal{Kind: ValKindPoly, Poly: val.Poly.Neg()}, nil
		case ValKindMatrix:
			return CalcVal{Kind: ValKindMatrix, Mat: val.Mat.Neg()}, nil
		case ValKindVector:
			return CalcVal{Kind: ValKindVector, Vec: val.Vec.Neg()}, nil
		case ValKindFloat:
			return CalcVal{Kind: ValKindFloat, Float: -val.Float}, nil
		case ValKindUnit:
			return CalcVal{Kind: ValKindUnit, UnitVal: new(big.Rat).Neg(val.UnitVal), UnitName: val.UnitName}, nil
		default:
			return CalcVal{}, fmt.Errorf("ErrSyntax: cannot negate %v", val.Kind)
		}
	}
	if tok.Type == TokPlus {
		p.next()
		return p.parsePower()
	}
	return p.parsePower()
}

func (p *Parser) parsePower() (CalcVal, error) {
	left, err := p.parsePrimary()
	if err != nil {
		return CalcVal{}, err
	}

	// Exponentiation is right-associative: a ^ b ^ c = a ^ (b ^ c)
	if p.cur().Type == TokPow {
		p.next()
		// After '^', unary minus is allowed: 2^-2
		right, err := p.parseUnary()
		if err != nil {
			return CalcVal{}, err
		}

		// Zero power: x^0 = 1 for any x (exact rational 1)
		if isZeroCalcVal(right) {
			return CalcVal{Kind: ValKindRat, Rat: big.NewRat(1, 1)}, nil
		}

		// Exact integer powers on Unit
		if left.Kind == ValKindUnit && right.Kind == ValKindRat && right.Rat.IsInt() {
			if !right.Rat.Num().IsInt64() {
				return CalcVal{}, fmt.Errorf("ErrOverflow: exponent exceeds 64-bit integer limit")
			}
			exp := right.Rat.Num().Int64()
			if exp > 100 || exp < -100 {
				return CalcVal{}, fmt.Errorf("ErrOverflow: unit power exponent exceeds limit of 100")
			}
			resVal, unitRes, isDimless, err := PowerUnits(left.UnitVal, left.UnitName, exp)
			if err != nil {
				return CalcVal{}, err
			}
			if isDimless {
				return CalcVal{Kind: ValKindRat, Rat: resVal}, nil
			}
			return CalcVal{Kind: ValKindUnit, UnitVal: resVal, UnitName: unitRes}, nil
		}

		// Exact integer powers on ExactNum and Surd
		if (left.Kind == ValKindExactNum || left.Kind == ValKindSurd) && right.Kind == ValKindRat && right.Rat.IsInt() {
			if !right.Rat.Num().IsInt64() {
				return CalcVal{}, fmt.Errorf("ErrOverflow: exponent exceeds 64-bit integer limit")
			}
			exp := right.Rat.Num().Int64()
			if exp > 1000 || exp < -1000 {
				return CalcVal{}, fmt.Errorf("ErrOverflow: exponent exceeds maximum limit of 1000")
			}
			ex := calcValToExactNum(left)
			powRes, ok, err := ex.Pow(exp)
			if err != nil {
				return CalcVal{}, err
			}
			if ok {
				if rat, isR := powRes.AsRational(); isR {
					return CalcVal{Kind: ValKindRat, Rat: rat}, nil
				}
				return CalcVal{Kind: ValKindExactNum, Exact: powRes}, nil
			}
		}

		// Exact integer powers on Polynomial
		if left.Kind == ValKindPoly && right.Kind == ValKindRat && right.Rat.IsInt() {
			if !right.Rat.Num().IsInt64() {
				return CalcVal{}, fmt.Errorf("ErrOverflow: polynomial exponent exceeds limit")
			}
			exp := right.Rat.Num().Int64()
			if exp < 0 || exp > 100 {
				return CalcVal{}, fmt.Errorf("ErrDomain: polynomial exponent must be between 0 and 100")
			}
			res, err := left.Poly.Pow(int(exp))
			if err != nil {
				return CalcVal{}, err
			}
			return CalcVal{Kind: ValKindPoly, Poly: res}, nil
		}

		// Exact integer powers on Matrix
		left = tryPromoteList(left)
		if left.Kind == ValKindMatrix && right.Kind == ValKindRat && right.Rat.IsInt() {
			exp := int(right.Rat.Num().Int64())
			if exp < 0 || exp > 100 {
				return CalcVal{}, fmt.Errorf("ErrDomain: matrix power exponent must be between 0 and 100")
			}
			res, err := left.Mat.Pow(int64(exp))
			if err != nil {
				return CalcVal{}, err
			}
			return CalcVal{Kind: ValKindMatrix, Mat: res}, nil
		}

		// Exact integer powers up to 2^20 bits cap
		if left.Kind == ValKindRat && right.Kind == ValKindRat && right.Rat.IsInt() {
			baseNum := left.Rat.Num()
			baseDen := left.Rat.Denom()
			exp := right.Rat.Num()

			if baseNum.Sign() == 0 {
				if exp.Sign() < 0 {
					return CalcVal{}, fmt.Errorf("ErrDivZero: zero to a negative power")
				}
				if exp.Sign() == 0 {
					return CalcVal{Kind: ValKindRat, Rat: big.NewRat(1, 1)}, nil
				}
				return CalcVal{Kind: ValKindRat, Rat: big.NewRat(0, 1)}, nil
			}

			if exp.Sign() == 0 {
				return CalcVal{Kind: ValKindRat, Rat: big.NewRat(1, 1)}, nil
			}

			// Pre-check result bit size ceiling: 2^20 bits = 1,048,576 bits
			absExp := new(big.Int).Abs(exp)
			maxBaseBits := int64(baseNum.BitLen())
			if int64(baseDen.BitLen()) > maxBaseBits {
				maxBaseBits = int64(baseDen.BitLen())
			}

			if absExp.BitLen() > 30 || absExp.Int64()*maxBaseBits > 1048576 {
				return CalcVal{}, fmt.Errorf("ErrOverflow: exponentiation result exceeds 2^20 bits ceiling")
			}

			posExp := absExp.Int64()
			resNum := new(big.Int).Exp(baseNum, big.NewInt(posExp), nil)
			resDen := new(big.Int).Exp(baseDen, big.NewInt(posExp), nil)

			if exp.Sign() < 0 {
				// Negative exponent: exact reciprocal
				if resNum.Sign() < 0 {
					resNum.Neg(resNum)
					resDen.Neg(resDen)
				}
				return CalcVal{Kind: ValKindRat, Rat: new(big.Rat).SetFrac(resDen, resNum), ForcedFrac: left.ForcedFrac}, nil
			}

			return CalcVal{Kind: ValKindRat, Rat: new(big.Rat).SetFrac(resNum, resDen), ForcedFrac: left.ForcedFrac}, nil
		}

		// Fractional powers on exact rationals: e.g. 8^(1/3) -> 2, 16^(3/4) -> 8, (27/8)^(1/3) -> 3/2
		if left.Kind == ValKindRat && right.Kind == ValKindRat && !right.Rat.IsInt() {
			pExp := right.Rat.Num()
			qExp := right.Rat.Denom()

			if qExp.IsInt64() {
				q := qExp.Int64()
				sign := left.Rat.Sign()
				if sign == 0 {
					if pExp.Sign() < 0 {
						return CalcVal{}, fmt.Errorf("ErrDivZero: division by zero in rational power")
					}
					return CalcVal{Kind: ValKindRat, Rat: big.NewRat(0, 1)}, nil
				}

				if sign < 0 && q%2 == 0 {
					if q == 2 {
						return CalcVal{}, fmt.Errorf("ErrDomain: square root of negative number")
					}
					return CalcVal{}, fmt.Errorf("ErrDomain: negative base with fractional exponent")
				}

				absNum := new(big.Int).Abs(left.Rat.Num())
				absDen := left.Rat.Denom()

				numRoot, numExact := integerNthRoot(absNum, q)
				denRoot, denExact := integerNthRoot(absDen, q)

				if numExact && denExact {
					// Base root is exact rational
					absP := new(big.Int).Abs(pExp)
					rootBits := int64(numRoot.BitLen())
					if int64(denRoot.BitLen()) > rootBits {
						rootBits = int64(denRoot.BitLen())
					}

					if absP.BitLen() > 30 || absP.Int64()*rootBits > 1048576 {
						return CalcVal{}, fmt.Errorf("ErrOverflow: exponentiation result exceeds 2^20 bits ceiling")
					}

					posExp := absP.Int64()
					resNum := new(big.Int).Exp(numRoot, big.NewInt(posExp), nil)
					resDen := new(big.Int).Exp(denRoot, big.NewInt(posExp), nil)

					if sign < 0 && posExp%2 != 0 {
						resNum.Neg(resNum)
					}

					if pExp.Sign() < 0 {
						// Invert
						if resNum.Sign() < 0 {
							resNum.Neg(resNum)
							resDen.Neg(resDen)
						}
						return CalcVal{Kind: ValKindRat, Rat: new(big.Rat).SetFrac(resDen, resNum), ForcedFrac: left.ForcedFrac}, nil
					}
					return CalcVal{Kind: ValKindRat, Rat: new(big.Rat).SetFrac(resNum, resDen), ForcedFrac: left.ForcedFrac}, nil
				}
			}

			// If not exact rational, but exponent is 1/2:
			if right.Rat.Cmp(big.NewRat(1, 2)) == 0 {
				if left.Rat.Sign() < 0 {
					return CalcVal{}, fmt.Errorf("ErrDomain: square root of negative number")
				}
				return evalSqrt(left)
			}
		}

		f1 := toFloat(left)
		f2 := toFloat(right)
		res := math.Pow(f1, f2)
		if math.IsNaN(res) {
			return CalcVal{}, fmt.Errorf("ErrDomain: power resulted in NaN")
		}
		if math.IsInf(res, 0) {
			return CalcVal{}, fmt.Errorf("ErrOverflow: power calculation overflowed float precision")
		}
		return CalcVal{Kind: ValKindFloat, Float: res}, nil
	}

	return left, nil
}

func (p *Parser) parsePrimary() (CalcVal, error) {
	tok := p.cur()

	// Number literal
	if tok.Type == TokNum {
		p.next()
		val := CalcVal{Kind: ValKindRat, Rat: tok.NumRat}

		// Check if immediately followed by a unit (e.g. 18480 ft)
		if p.cur().Type == TokIdent && IsKnownUnit(p.cur().Text) {
			unitTok := p.next()
			info, _ := NormalizeUnit(unitTok.Text)
			return CalcVal{Kind: ValKindUnit, UnitVal: val.Rat, UnitName: info.Name}, nil
		}
		return val, nil
	}

	// Register token: <|r0|> .. <|r7|>
	if tok.Type == TokRegister {
		p.next()
		if p.store == nil {
			return CalcVal{}, fmt.Errorf("ErrUnsetRegister: <|r%d|> is empty", tok.RegIdx)
		}
		val, err := p.store.Get(tok.RegIdx)
		if err != nil {
			return CalcVal{}, err
		}
		if p.cur().Type == TokIdent && IsKnownUnit(p.cur().Text) {
			if val.Kind == ValKindRat {
				unitTok := p.next()
				info, _ := NormalizeUnit(unitTok.Text)
				return CalcVal{Kind: ValKindUnit, UnitVal: val.Rat, UnitName: info.Name}, nil
			}
			if val.Kind == ValKindFloat {
				unitTok := p.next()
				info, _ := NormalizeUnit(unitTok.Text)
				r := new(big.Rat).SetFloat64(val.Float)
				return CalcVal{Kind: ValKindUnit, UnitVal: r, UnitName: info.Name}, nil
			}
		}
		return *val, nil
	}

	// String literal
	if tok.Type == TokString {
		p.next()
		return CalcVal{Kind: ValKindString, Str: tok.Text}, nil
	}

	// Parenthesized expression
	if tok.Type == TokLParen {
		p.next()
		val, err := p.parseExpression()
		if err != nil {
			return CalcVal{}, err
		}
		if p.cur().Type != TokRParen {
			return CalcVal{}, fmt.Errorf("ErrSyntax: expected ')' at column %d", p.cur().Col)
		}
		p.next()

		// Check if followed by a unit: e.g. (-5) m
		if p.cur().Type == TokIdent && IsKnownUnit(p.cur().Text) && val.Kind == ValKindRat {
			unitTok := p.next()
			info, _ := NormalizeUnit(unitTok.Text)
			return CalcVal{Kind: ValKindUnit, UnitVal: val.Rat, UnitName: info.Name}, nil
		}
		return val, nil
	}

	// List literal: [1, 2, 3]
	if tok.Type == TokLBracket {
		p.next()
		var items []CalcVal
		for p.cur().Type != TokRBracket && p.cur().Type != TokEOF {
			item, err := p.parseExpression()
			if err != nil {
				return CalcVal{}, err
			}
			items = append(items, item)
			if p.cur().Type == TokComma {
				p.next()
			} else if p.cur().Type != TokRBracket {
				return CalcVal{}, fmt.Errorf("ErrSyntax: expected ',' or ']' at column %d", p.cur().Col)
			}
		}
		if p.cur().Type != TokRBracket {
			return CalcVal{}, fmt.Errorf("ErrSyntax: expected ']' at column %d", p.cur().Col)
		}
		p.next()
		return CalcVal{Kind: ValKindList, List: items}, nil
	}

	// Identifiers (constants, units, function calls)
	if tok.Type == TokIdent {
		p.next()
		name := tok.Text

		if (name == "sum" || name == "product") && p.isBoundedSumOrProduct() {
			p.next() // consume '('
			var exprTokens []Token
			depth := 0
			for p.pos < len(p.tokens) {
				t := p.cur()
				if t.Type == TokLParen || t.Type == TokLBracket {
					depth++
				} else if t.Type == TokRParen || t.Type == TokRBracket {
					if depth == 0 {
						break
					}
					depth--
				} else if t.Type == TokComma && depth == 0 {
					break
				}
				exprTokens = append(exprTokens, t)
				p.next()
			}
			if p.cur().Type != TokComma {
				return CalcVal{}, fmt.Errorf("ErrSyntax: expected ',' in %s at column %d", name, p.cur().Col)
			}
			p.next() // consume ','

			varTok := p.cur()
			if varTok.Type != TokIdent {
				return CalcVal{}, fmt.Errorf("ErrSyntax: expected variable name in %s at column %d", name, varTok.Col)
			}
			p.next()
			varName := varTok.Text

			if p.cur().Type != TokComma {
				return CalcVal{}, fmt.Errorf("ErrSyntax: expected ',' after variable in %s at column %d", name, p.cur().Col)
			}
			p.next() // consume ','

			aVal, errA := p.parseExpression()
			if errA != nil {
				return CalcVal{}, errA
			}
			if aVal.Kind != ValKindRat || !aVal.Rat.IsInt() {
				return CalcVal{}, fmt.Errorf("ErrDomain: %s lower bound must be an integer", name)
			}
			aInt := aVal.Rat.Num().Int64()

			if p.cur().Type != TokComma {
				return CalcVal{}, fmt.Errorf("ErrSyntax: expected ',' after lower bound in %s at column %d", name, p.cur().Col)
			}
			p.next() // consume ','

			bVal, errB := p.parseExpression()
			if errB != nil {
				return CalcVal{}, errB
			}
			if bVal.Kind != ValKindRat || !bVal.Rat.IsInt() {
				return CalcVal{}, fmt.Errorf("ErrDomain: %s upper bound must be an integer", name)
			}
			bInt := bVal.Rat.Num().Int64()

			if p.cur().Type != TokRParen {
				return CalcVal{}, fmt.Errorf("ErrSyntax: expected ')' in %s at column %d", name, p.cur().Col)
			}
			p.next() // consume ')'

			if name == "sum" {
				return EvalSumTokens(exprTokens, varName, aInt, bInt, p.store)
			}
			return EvalProductTokens(exprTokens, varName, aInt, bInt, p.store)
		}

		// Function call: func(...)
		if p.cur().Type == TokLParen {
			p.next()
			var args []CalcVal
			for p.cur().Type != TokRParen && p.cur().Type != TokEOF {
				arg, err := p.parseExpression()
				if err != nil {
					return CalcVal{}, err
				}
				args = append(args, arg)
				if p.cur().Type == TokComma {
					p.next()
				} else if p.cur().Type != TokRParen {
					return CalcVal{}, fmt.Errorf("ErrSyntax: expected ',' or ')' at column %d", p.cur().Col)
				}
			}
			if p.cur().Type != TokRParen {
				return CalcVal{}, fmt.Errorf("ErrSyntax: expected ')' at column %d", p.cur().Col)
			}
			p.next()
			return p.evalFunction(name, args)
		}

		// Constants
		switch name {
		case "pi":
			return CalcVal{Kind: ValKindExactNum, Exact: NewExactPi()}, nil
		case "i":
			return CalcVal{Kind: ValKindExactNum, Exact: NewExactI()}, nil
		case "e":
			return CalcVal{Kind: ValKindExactNum, Exact: NewExactE()}, nil
		case "true":
			return CalcVal{Kind: ValKindBool, Bool: true}, nil
		case "false":
			return CalcVal{Kind: ValKindBool, Bool: false}, nil
		}

		// Standalone unit identifier (only active when expression has unit context)
		if p.hasUnits && IsKnownUnit(name) {
			if p.isUnitChain(tok) {
				info, _ := NormalizeUnit(name)
				return CalcVal{Kind: ValKindUnit, UnitVal: big.NewRat(1, 1), UnitName: info.Name}, nil
			}
			return CalcVal{}, fmt.Errorf("ErrAmbiguous: '%s' is both a unit and a variable here; write 1 %s for the unit or rename the variable", name, name)
		}

		// Unknown identifier -> Polynomial variable
		return CalcVal{Kind: ValKindPoly, Poly: NewMultiPolyVar(name)}, nil
	}

	if tok.Type == TokEOF {
		return CalcVal{}, fmt.Errorf("ErrSyntax: unexpected end of expression")
	}

	return CalcVal{}, fmt.Errorf("ErrSyntax: unexpected token %q at column %d", tok.Text, tok.Col)
}

func (p *Parser) isBoundedSumOrProduct() bool {
	if p.cur().Type != TokLParen {
		return false
	}
	depth := 0
	commaCount := 0
	var secondArgTokens []Token
	for i := p.pos + 1; i < len(p.tokens); i++ {
		t := p.tokens[i]
		if t.Type == TokLParen || t.Type == TokLBracket {
			depth++
		} else if t.Type == TokRParen || t.Type == TokRBracket {
			if depth == 0 {
				break
			}
			depth--
		} else if depth == 0 && t.Type == TokComma {
			commaCount++
		} else if depth == 0 && commaCount == 1 {
			secondArgTokens = append(secondArgTokens, t)
		}
	}
	return commaCount == 3 && len(secondArgTokens) == 1 && secondArgTokens[0].Type == TokIdent
}

func (p *Parser) evalFunction(fn string, args []CalcVal) (CalcVal, error) {
	if res, handled, err := evalExtendedFunction(fn, args, p.store); handled {
		return res, err
	}
	return evalFunction(fn, args)
}

// -----------------------------------------------------------------------------
// Functions Evaluator
// -----------------------------------------------------------------------------

func evalFunction(fn string, args []CalcVal) (CalcVal, error) {
	for _, arg := range args {
		if isSymbolic(arg) {
			return CalcVal{}, fmt.Errorf("ErrUnsupported: %s of a symbolic expression is not supported", fn)
		}
	}

	// Unwrap single list argument for aggregation functions: mean([1, 2, 3]) -> mean(1, 2, 3)
	if len(args) == 1 && args[0].Kind == ValKindList {
		args = args[0].List
	}

	switch fn {
	case "sqrt":
		if len(args) != 1 {
			return CalcVal{}, fmt.Errorf("ErrSyntax: sqrt expects 1 argument")
		}
		return evalSqrt(args[0])

	case "cbrt":
		if len(args) != 1 {
			return CalcVal{}, fmt.Errorf("ErrSyntax: cbrt expects 1 argument")
		}
		return evalCbrt(args[0])

	case "abs":
		if len(args) != 1 {
			return CalcVal{}, fmt.Errorf("ErrSyntax: abs expects 1 argument")
		}
		v := args[0]
		if v.Kind == ValKindUnit {
			res := new(big.Rat).Abs(v.UnitVal)
			return CalcVal{Kind: ValKindUnit, UnitVal: res, UnitName: v.UnitName}, nil
		}
		if v.Kind == ValKindRat {
			return CalcVal{Kind: ValKindRat, Rat: new(big.Rat).Abs(v.Rat)}, nil
		}
		if v.Kind == ValKindSurd {
			cp := v.Surd.Clone()
			cp.sign = 1
			return CalcVal{Kind: ValKindSurd, Surd: cp}, nil
		}
		if v.Kind == ValKindExactNum {
			res, ok := v.Exact.Abs()
			if ok {
				if rat, isR := res.AsRational(); isR {
					return CalcVal{Kind: ValKindRat, Rat: rat}, nil
				}
				return CalcVal{Kind: ValKindExactNum, Exact: res}, nil
			}
			return CalcVal{Kind: ValKindFloat, Float: math.Abs(toFloat(v))}, nil
		}
		return CalcVal{Kind: ValKindFloat, Float: math.Abs(toFloat(v))}, nil

	case "round":
		if len(args) < 1 || len(args) > 2 {
			return CalcVal{}, fmt.Errorf("ErrSyntax: round expects 1 or 2 arguments")
		}
		v := args[0]
		places := 0
		if len(args) == 2 {
			if args[1].Kind != ValKindRat || !args[1].Rat.IsInt() {
				return CalcVal{}, fmt.Errorf("ErrDomain: round places must be an integer")
			}
			places = int(args[1].Rat.Num().Int64())
		}
		// Exact round on surds: convert to float and round
		if v.Kind == ValKindSurd {
			f := v.Surd.ToFloat64()
			return CalcVal{Kind: ValKindFloat, Float: f, HasRound: true, RoundPlaces: places}, nil
		}
		if v.Kind == ValKindFloat {
			return CalcVal{Kind: ValKindFloat, Float: v.Float, HasRound: true, RoundPlaces: places}, nil
		}
		if v.Kind == ValKindUnit {
			roundedRat := roundRational(v.UnitVal, places)
			return CalcVal{Kind: ValKindUnit, UnitVal: roundedRat, UnitName: v.UnitName, HasRound: true, RoundPlaces: places}, nil
		}
		if v.Kind == ValKindRat {
			roundedRat := roundRational(v.Rat, places)
			return CalcVal{Kind: ValKindRat, Rat: roundedRat, HasRound: true, RoundPlaces: places}, nil
		}
		return CalcVal{}, fmt.Errorf("ErrTypeMismatch: round expects numeric argument")

	case "floor":
		if len(args) != 1 {
			return CalcVal{}, fmt.Errorf("ErrSyntax: floor expects 1 argument")
		}
		v := args[0]
		if v.Kind == ValKindUnit {
			q := new(big.Int)
			rem := new(big.Int)
			q.QuoRem(v.UnitVal.Num(), v.UnitVal.Denom(), rem)
			if v.UnitVal.Sign() < 0 && rem.Sign() != 0 {
				q.Sub(q, big.NewInt(1))
			}
			return CalcVal{Kind: ValKindUnit, UnitVal: new(big.Rat).SetInt(q), UnitName: v.UnitName}, nil
		}
		if v.Kind == ValKindRat {
			q := new(big.Int)
			rem := new(big.Int)
			q.QuoRem(v.Rat.Num(), v.Rat.Denom(), rem)
			if v.Rat.Sign() < 0 && rem.Sign() != 0 {
				q.Sub(q, big.NewInt(1))
			}
			return CalcVal{Kind: ValKindRat, Rat: new(big.Rat).SetInt(q)}, nil
		}
		return CalcVal{Kind: ValKindFloat, Float: math.Floor(toFloat(v))}, nil

	case "ceil":
		if len(args) != 1 {
			return CalcVal{}, fmt.Errorf("ErrSyntax: ceil expects 1 argument")
		}
		v := args[0]
		if v.Kind == ValKindUnit {
			q := new(big.Int)
			rem := new(big.Int)
			q.QuoRem(v.UnitVal.Num(), v.UnitVal.Denom(), rem)
			if v.UnitVal.Sign() > 0 && rem.Sign() != 0 {
				q.Add(q, big.NewInt(1))
			}
			return CalcVal{Kind: ValKindUnit, UnitVal: new(big.Rat).SetInt(q), UnitName: v.UnitName}, nil
		}
		if v.Kind == ValKindRat {
			q := new(big.Int)
			rem := new(big.Int)
			q.QuoRem(v.Rat.Num(), v.Rat.Denom(), rem)
			if v.Rat.Sign() > 0 && rem.Sign() != 0 {
				q.Add(q, big.NewInt(1))
			}
			return CalcVal{Kind: ValKindRat, Rat: new(big.Rat).SetInt(q)}, nil
		}
		return CalcVal{Kind: ValKindFloat, Float: math.Ceil(toFloat(v))}, nil

	case "frac":
		if len(args) != 1 {
			return CalcVal{}, fmt.Errorf("ErrSyntax: frac expects 1 argument")
		}
		v := args[0]
		v.ForcedFrac = true
		return v, nil

	case "float":
		if len(args) != 1 {
			return CalcVal{}, fmt.Errorf("ErrSyntax: float expects 1 argument")
		}
		if args[0].Kind == ValKindUnit || args[0].UnitName != "" {
			v := args[0]
			v.ForcedDec = true
			return v, nil
		}
		f := toFloat(args[0])
		return CalcVal{Kind: ValKindFloat, Float: f}, nil

	case "approx":
		if len(args) != 1 {
			return CalcVal{}, fmt.Errorf("ErrSyntax: approx expects 1 argument")
		}
		v := args[0]
		v.ForcedDec = true
		return v, nil

	case "min", "max":
		if len(args) == 0 {
			return CalcVal{}, fmt.Errorf("ErrSyntax: %s expects at least 1 argument", fn)
		}
		res := args[0]
		for i := 1; i < len(args); i++ {
			cmp, err := compareCalcVals(res, args[i])
			if err != nil {
				return CalcVal{}, err
			}
			if fn == "min" && cmp > 0 {
				res = args[i]
			}
			if fn == "max" && cmp < 0 {
				res = args[i]
			}
		}
		return res, nil

	case "sum":
		if len(args) == 0 {
			return CalcVal{Kind: ValKindRat, Rat: big.NewRat(0, 1)}, nil
		}
		res := args[0]
		for i := 1; i < len(args); i++ {
			var err error
			res, err = addCalcVals(res, args[i])
			if err != nil {
				return CalcVal{}, err
			}
		}
		return res, nil

	case "prod":
		if len(args) == 0 {
			return CalcVal{Kind: ValKindRat, Rat: big.NewRat(1, 1)}, nil
		}
		res := args[0]
		for i := 1; i < len(args); i++ {
			var err error
			res, err = mulCalcVals(res, args[i])
			if err != nil {
				return CalcVal{}, err
			}
		}
		return res, nil

	case "mean":
		if len(args) == 0 {
			return CalcVal{}, fmt.Errorf("ErrDomain: mean requires at least 1 argument")
		}
		// Exact if all inputs are rationals/units
		allRat := true
		for _, a := range args {
			if a.Kind != ValKindRat && a.Kind != ValKindUnit {
				allRat = false
				break
			}
		}
		if allRat {
			sumVal := args[0]
			for i := 1; i < len(args); i++ {
				var err error
				sumVal, err = addCalcVals(sumVal, args[i])
				if err != nil {
					return CalcVal{}, err
				}
			}
			n := big.NewRat(int64(len(args)), 1)
			return divCalcVals(sumVal, CalcVal{Kind: ValKindRat, Rat: n})
		}
		var sum float64
		for _, a := range args {
			sum += toFloat(a)
		}
		return CalcVal{Kind: ValKindFloat, Float: sum / float64(len(args))}, nil

	case "median":
		if len(args) == 0 {
			return CalcVal{}, fmt.Errorf("ErrDomain: median requires at least 1 argument")
		}
		// Check if all rationals
		allRat := true
		for _, a := range args {
			if a.Kind != ValKindRat {
				allRat = false
				break
			}
		}
		if allRat {
			// Sort rationals
			sorted := make([]CalcVal, len(args))
			copy(sorted, args)
			for i := 0; i < len(sorted); i++ {
				for j := i + 1; j < len(sorted); j++ {
					if sorted[i].Rat.Cmp(sorted[j].Rat) > 0 {
						sorted[i], sorted[j] = sorted[j], sorted[i]
					}
				}
			}
			n := len(sorted)
			if n%2 == 1 {
				return sorted[n/2], nil
			}
			m1 := sorted[n/2-1]
			m2 := sorted[n/2]
			sumR := new(big.Rat).Add(m1.Rat, m2.Rat)
			res := new(big.Rat).Quo(sumR, big.NewRat(2, 1))
			return CalcVal{Kind: ValKindRat, Rat: res}, nil
		}
		// Float median
		var vals []float64
		for _, a := range args {
			vals = append(vals, toFloat(a))
		}
		for i := 0; i < len(vals); i++ {
			for j := i + 1; j < len(vals); j++ {
				if vals[i] > vals[j] {
					vals[i], vals[j] = vals[j], vals[i]
				}
			}
		}
		n := len(vals)
		if n%2 == 1 {
			return CalcVal{Kind: ValKindFloat, Float: vals[n/2]}, nil
		}
		return CalcVal{Kind: ValKindFloat, Float: (vals[n/2-1] + vals[n/2]) / 2}, nil

	case "std", "std_pop":
		if len(args) < 2 && fn == "std" {
			return CalcVal{}, fmt.Errorf("ErrDomain: sample std requires at least 2 values")
		}
		if len(args) == 0 {
			return CalcVal{}, fmt.Errorf("ErrDomain: std requires at least 1 value")
		}
		var vals []float64
		for _, a := range args {
			vals = append(vals, toFloat(a))
		}
		var mean float64
		for _, v := range vals {
			mean += v
		}
		mean /= float64(len(vals))
		var sumSq float64
		for _, v := range vals {
			d := v - mean
			sumSq += d * d
		}
		denom := float64(len(vals) - 1)
		if fn == "std_pop" {
			denom = float64(len(vals))
		}
		return CalcVal{Kind: ValKindFloat, Float: math.Sqrt(sumSq / denom)}, nil

	case "exp":
		if len(args) != 1 {
			return CalcVal{}, fmt.Errorf("ErrSyntax: exp expects 1 argument")
		}
		arg := args[0]
		// exp(0) -> 1
		if isZeroCalcVal(arg) {
			return CalcVal{Kind: ValKindRat, Rat: big.NewRat(1, 1)}, nil
		}
		if arg.Kind == ValKindExactNum && arg.Exact != nil {
			if r, ok := arg.Exact.AsLogRational(); ok {
				return CalcVal{Kind: ValKindRat, Rat: r}, nil
			}
			if rat, ok := arg.Exact.AsRational(); ok {
				if rat.Sign() == 0 {
					return CalcVal{Kind: ValKindRat, Rat: big.NewRat(1, 1)}, nil
				}
				if rat.Cmp(big.NewRat(1, 1)) == 0 {
					return CalcVal{Kind: ValKindExactNum, Exact: NewExactE()}, nil
				}
				if rat.IsInt() {
					k := rat.Num().Int64()
					if k >= -10 && k <= 10 {
						return CalcVal{Kind: ValKindExactNum, Exact: NewExactEPow(int(k))}, nil
					}
				}
			}
		}
		if arg.Kind == ValKindRat && arg.Rat != nil {
			if arg.Rat.Sign() == 0 {
				return CalcVal{Kind: ValKindRat, Rat: big.NewRat(1, 1)}, nil
			}
			if arg.Rat.Cmp(big.NewRat(1, 1)) == 0 {
				return CalcVal{Kind: ValKindExactNum, Exact: NewExactE()}, nil
			}
			if arg.Rat.IsInt() {
				k := arg.Rat.Num().Int64()
				if k >= -10 && k <= 10 {
					return CalcVal{Kind: ValKindExactNum, Exact: NewExactEPow(int(k))}, nil
				}
			}
		}
		f := toFloat(arg)
		if f > 709.78 {
			return CalcVal{}, fmt.Errorf("ErrOverflow: exp calculation overflowed float precision")
		}
		return CalcVal{Kind: ValKindFloat, Float: math.Exp(f)}, nil

	case "ln", "log":
		// log(x) is natural log. log(x, b) is base-b log.
		if len(args) == 1 {
			arg := args[0]
			// ln(1) -> 0
			if arg.Kind == ValKindRat && arg.Rat != nil && arg.Rat.Cmp(big.NewRat(1, 1)) == 0 {
				return CalcVal{Kind: ValKindRat, Rat: big.NewRat(0, 1)}, nil
			}
			if arg.Kind == ValKindFloat && arg.Float == 1.0 {
				return CalcVal{Kind: ValKindRat, Rat: big.NewRat(0, 1)}, nil
			}
			if arg.Kind == ValKindExactNum && arg.Exact != nil {
				if rat, ok := arg.Exact.AsRational(); ok && rat.Cmp(big.NewRat(1, 1)) == 0 {
					return CalcVal{Kind: ValKindRat, Rat: big.NewRat(0, 1)}, nil
				}
				if p, ok := arg.Exact.AsEPow(); ok {
					return CalcVal{Kind: ValKindRat, Rat: big.NewRat(int64(p), 1)}, nil
				}
			}
			if arg.Kind == ValKindRat && arg.Rat != nil {
				if arg.Rat.Sign() <= 0 {
					return CalcVal{}, fmt.Errorf("ErrDomain: natural log of non-positive number")
				}
				if ex, ok := NewExactLn(arg.Rat); ok {
					if rat, isR := ex.AsRational(); isR {
						return CalcVal{Kind: ValKindRat, Rat: rat}, nil
					}
					return CalcVal{Kind: ValKindExactNum, Exact: ex}, nil
				}
			}
			f := toFloat(arg)
			if f <= 0 {
				return CalcVal{}, fmt.Errorf("ErrDomain: natural log of non-positive number")
			}
			return CalcVal{Kind: ValKindFloat, Float: math.Log(f)}, nil
		}
		if len(args) == 2 {
			xVal := args[0]
			bVal := args[1]
			// log(b, b) -> 1
			cmp, err := compareCalcVals(xVal, bVal)
			if err == nil && cmp == 0 {
				f := toFloat(xVal)
				if f <= 0 || f == 1.0 {
					return CalcVal{}, fmt.Errorf("ErrDomain: invalid base or argument for logarithm")
				}
				return CalcVal{Kind: ValKindRat, Rat: big.NewRat(1, 1)}, nil
			}
			if (xVal.Kind == ValKindRat && xVal.Rat != nil && xVal.Rat.Cmp(big.NewRat(1, 1)) == 0) ||
				(xVal.Kind == ValKindFloat && xVal.Float == 1.0) {
				bF := toFloat(bVal)
				if bF <= 0 || bF == 1.0 {
					return CalcVal{}, fmt.Errorf("ErrDomain: invalid base or argument for logarithm")
				}
				return CalcVal{Kind: ValKindRat, Rat: big.NewRat(0, 1)}, nil
			}
			x := toFloat(xVal)
			b := toFloat(bVal)
			if x <= 0 || b <= 0 || b == 1.0 {
				return CalcVal{}, fmt.Errorf("ErrDomain: invalid base or argument for logarithm")
			}
			return CalcVal{Kind: ValKindFloat, Float: math.Log(x) / math.Log(b)}, nil
		}
		return CalcVal{}, fmt.Errorf("ErrSyntax: log expects 1 or 2 arguments")

	case "log10":
		if len(args) != 1 {
			return CalcVal{}, fmt.Errorf("ErrSyntax: log10 expects 1 argument")
		}
		arg := args[0]
		if (arg.Kind == ValKindRat && arg.Rat != nil && arg.Rat.Cmp(big.NewRat(1, 1)) == 0) ||
			(arg.Kind == ValKindFloat && arg.Float == 1.0) {
			return CalcVal{Kind: ValKindRat, Rat: big.NewRat(0, 1)}, nil
		}
		if arg.Kind == ValKindExactNum && arg.Exact != nil {
			if rat, ok := arg.Exact.AsRational(); ok {
				if rat.Sign() <= 0 {
					return CalcVal{}, fmt.Errorf("ErrDomain: log10 of non-positive number")
				}
				if k, ok := isRatPowerOf10(rat); ok {
					return CalcVal{Kind: ValKindRat, Rat: big.NewRat(k, 1)}, nil
				}
				if ex, ok := NewExactLog10(rat); ok {
					if r, isR := ex.AsRational(); isR {
						return CalcVal{Kind: ValKindRat, Rat: r}, nil
					}
					return CalcVal{Kind: ValKindExactNum, Exact: ex}, nil
				}
			}
		}
		if arg.Kind == ValKindRat && arg.Rat != nil {
			if arg.Rat.Sign() <= 0 {
				return CalcVal{}, fmt.Errorf("ErrDomain: log10 of non-positive number")
			}
			if k, ok := isRatPowerOf10(arg.Rat); ok {
				return CalcVal{Kind: ValKindRat, Rat: big.NewRat(k, 1)}, nil
			}
			if ex, ok := NewExactLog10(arg.Rat); ok {
				if r, isR := ex.AsRational(); isR {
					return CalcVal{Kind: ValKindRat, Rat: r}, nil
				}
				return CalcVal{Kind: ValKindExactNum, Exact: ex}, nil
			}
		}
		f := toFloat(arg)
		if f <= 0 {
			return CalcVal{}, fmt.Errorf("ErrDomain: log10 of non-positive number")
		}
		return CalcVal{Kind: ValKindFloat, Float: math.Log10(f)}, nil

	case "log2":
		if len(args) != 1 {
			return CalcVal{}, fmt.Errorf("ErrSyntax: log2 expects 1 argument")
		}
		arg := args[0]
		if (arg.Kind == ValKindRat && arg.Rat != nil && arg.Rat.Cmp(big.NewRat(1, 1)) == 0) ||
			(arg.Kind == ValKindFloat && arg.Float == 1.0) {
			return CalcVal{Kind: ValKindRat, Rat: big.NewRat(0, 1)}, nil
		}
		if arg.Kind == ValKindExactNum && arg.Exact != nil {
			if rat, ok := arg.Exact.AsRational(); ok {
				if rat.Sign() <= 0 {
					return CalcVal{}, fmt.Errorf("ErrDomain: log2 of non-positive number")
				}
				if k, ok := isRatPowerOf2(rat); ok {
					return CalcVal{Kind: ValKindRat, Rat: big.NewRat(k, 1)}, nil
				}
			}
		}
		if arg.Kind == ValKindRat && arg.Rat != nil {
			if arg.Rat.Sign() <= 0 {
				return CalcVal{}, fmt.Errorf("ErrDomain: log2 of non-positive number")
			}
			if k, ok := isRatPowerOf2(arg.Rat); ok {
				return CalcVal{Kind: ValKindRat, Rat: big.NewRat(k, 1)}, nil
			}
		}
		f := toFloat(arg)
		if f <= 0 {
			return CalcVal{}, fmt.Errorf("ErrDomain: log2 of non-positive number")
		}
		return CalcVal{Kind: ValKindFloat, Float: math.Log2(f)}, nil

	case "sin":
		if len(args) != 1 {
			return CalcVal{}, fmt.Errorf("ErrSyntax: sin expects 1 argument")
		}
		return CalcVal{Kind: ValKindFloat, Float: math.Sin(toFloat(args[0]))}, nil

	case "cos":
		if len(args) != 1 {
			return CalcVal{}, fmt.Errorf("ErrSyntax: cos expects 1 argument")
		}
		return CalcVal{Kind: ValKindFloat, Float: math.Cos(toFloat(args[0]))}, nil

	case "tan":
		if len(args) != 1 {
			return CalcVal{}, fmt.Errorf("ErrSyntax: tan expects 1 argument")
		}
		return CalcVal{Kind: ValKindFloat, Float: math.Tan(toFloat(args[0]))}, nil

	case "factorial":
		if len(args) != 1 {
			return CalcVal{}, fmt.Errorf("ErrSyntax: factorial expects 1 argument")
		}
		v := args[0]
		if v.Kind != ValKindRat || !v.Rat.IsInt() {
			return CalcVal{}, fmt.Errorf("ErrDomain: factorial expects a non-negative integer")
		}
		n := v.Rat.Num()
		if n.Sign() < 0 {
			return CalcVal{}, fmt.Errorf("ErrDomain: factorial expects a non-negative integer")
		}
		if n.Cmp(big.NewInt(20000)) > 0 {
			return CalcVal{}, fmt.Errorf("ErrOverflow: factorial exceeds maximum n=20,000")
		}
		res := big.NewInt(1)
		intN := n.Int64()
		for i := int64(2); i <= intN; i++ {
			res.Mul(res, big.NewInt(i))
		}
		return CalcVal{Kind: ValKindRat, Rat: new(big.Rat).SetInt(res)}, nil

	case "comb":
		if len(args) != 2 {
			return CalcVal{}, fmt.Errorf("ErrSyntax: comb expects 2 arguments")
		}
		v1, v2 := args[0], args[1]
		if v1.Kind != ValKindRat || !v1.Rat.IsInt() || v2.Kind != ValKindRat || !v2.Rat.IsInt() {
			return CalcVal{}, fmt.Errorf("ErrDomain: comb expects integer arguments")
		}
		n := v1.Rat.Num()
		k := v2.Rat.Num()
		if n.Sign() < 0 || k.Sign() < 0 {
			return CalcVal{}, fmt.Errorf("ErrDomain: comb expects non-negative integers")
		}
		if k.Cmp(n) > 0 {
			return CalcVal{Kind: ValKindRat, Rat: big.NewRat(0, 1)}, nil
		}
		diff := new(big.Int).Sub(n, k)
		if k.Cmp(diff) > 0 {
			k = diff
		}
		if !k.IsInt64() {
			return CalcVal{}, fmt.Errorf("ErrOverflow: comb result exceeds 2^20 bits ceiling")
		}
		kInt := k.Int64()
		if kInt > 100000 || int64(n.BitLen())*kInt > 1048576 {
			return CalcVal{}, fmt.Errorf("ErrOverflow: comb result exceeds 2^20 bits ceiling")
		}
		res := big.NewInt(1)
		for i := int64(1); i <= kInt; i++ {
			term := new(big.Int).Sub(n, big.NewInt(i-1))
			res.Mul(res, term)
			res.Div(res, big.NewInt(i))
			if res.BitLen() > 1048576 {
				return CalcVal{}, fmt.Errorf("ErrOverflow: comb result exceeds 2^20 bits ceiling")
			}
		}
		return CalcVal{Kind: ValKindRat, Rat: new(big.Rat).SetInt(res)}, nil

	case "perm":
		if len(args) != 2 {
			return CalcVal{}, fmt.Errorf("ErrSyntax: perm expects 2 arguments")
		}
		v1, v2 := args[0], args[1]
		if v1.Kind != ValKindRat || !v1.Rat.IsInt() || v2.Kind != ValKindRat || !v2.Rat.IsInt() {
			return CalcVal{}, fmt.Errorf("ErrDomain: perm expects integer arguments")
		}
		n := v1.Rat.Num()
		k := v2.Rat.Num()
		if n.Sign() < 0 || k.Sign() < 0 {
			return CalcVal{}, fmt.Errorf("ErrDomain: perm expects non-negative integers")
		}
		if k.Cmp(n) > 0 {
			return CalcVal{Kind: ValKindRat, Rat: big.NewRat(0, 1)}, nil
		}
		if !k.IsInt64() {
			return CalcVal{}, fmt.Errorf("ErrOverflow: perm result exceeds 2^20 bits ceiling")
		}
		kInt := k.Int64()
		if kInt > 100000 || int64(n.BitLen())*kInt > 1048576 {
			return CalcVal{}, fmt.Errorf("ErrOverflow: perm result exceeds 2^20 bits ceiling")
		}
		res := big.NewInt(1)
		for i := int64(0); i < kInt; i++ {
			term := new(big.Int).Sub(n, big.NewInt(i))
			res.Mul(res, term)
			if res.BitLen() > 1048576 {
				return CalcVal{}, fmt.Errorf("ErrOverflow: perm result exceeds 2^20 bits ceiling")
			}
		}
		return CalcVal{Kind: ValKindRat, Rat: new(big.Rat).SetInt(res)}, nil

	case "gcd":
		if len(args) != 2 {
			return CalcVal{}, fmt.Errorf("ErrSyntax: gcd expects 2 arguments")
		}
		v1, v2 := args[0], args[1]
		if v1.Kind != ValKindRat || !v1.Rat.IsInt() || v2.Kind != ValKindRat || !v2.Rat.IsInt() {
			return CalcVal{}, fmt.Errorf("ErrDomain: gcd expects integer arguments")
		}
		a := new(big.Int).Abs(v1.Rat.Num())
		b := new(big.Int).Abs(v2.Rat.Num())
		res := new(big.Int).GCD(nil, nil, a, b)
		return CalcVal{Kind: ValKindRat, Rat: new(big.Rat).SetInt(res)}, nil

	case "lcm":
		if len(args) != 2 {
			return CalcVal{}, fmt.Errorf("ErrSyntax: lcm expects 2 arguments")
		}
		v1, v2 := args[0], args[1]
		if v1.Kind != ValKindRat || !v1.Rat.IsInt() || v2.Kind != ValKindRat || !v2.Rat.IsInt() {
			return CalcVal{}, fmt.Errorf("ErrDomain: lcm expects integer arguments")
		}
		a := new(big.Int).Abs(v1.Rat.Num())
		b := new(big.Int).Abs(v2.Rat.Num())
		if a.Sign() == 0 || b.Sign() == 0 {
			return CalcVal{Kind: ValKindRat, Rat: big.NewRat(0, 1)}, nil
		}
		g := new(big.Int).GCD(nil, nil, a, b)
		prod := new(big.Int).Mul(a, b)
		res := prod.Div(prod, g)
		return CalcVal{Kind: ValKindRat, Rat: new(big.Rat).SetInt(res)}, nil

	case "isprime":
		if len(args) != 1 {
			return CalcVal{}, fmt.Errorf("ErrSyntax: isprime expects 1 argument")
		}
		v := args[0]
		if v.Kind != ValKindRat || !v.Rat.IsInt() {
			return CalcVal{}, fmt.Errorf("ErrDomain: isprime expects an integer argument")
		}
		n := v.Rat.Num()
		if n.Cmp(big.NewInt(1)) <= 0 {
			return CalcVal{Kind: ValKindBool, Bool: false}, nil
		}
		return CalcVal{Kind: ValKindBool, Bool: n.ProbablyPrime(20)}, nil

	case "factors":
		if len(args) != 1 {
			return CalcVal{}, fmt.Errorf("ErrSyntax: factors expects 1 argument")
		}
		v := args[0]
		if v.Kind != ValKindRat || !v.Rat.IsInt() {
			return CalcVal{}, fmt.Errorf("ErrDomain: factors expects an integer argument")
		}
		n := v.Rat.Num()
		if n.Sign() <= 0 {
			return CalcVal{}, fmt.Errorf("ErrDomain: factors expects positive integer")
		}
		if n.Cmp(big.NewInt(100000000000000)) > 0 {
			return CalcVal{}, fmt.Errorf("ErrOverflow: factors input exceeds 10^14 limit")
		}
		facs, err := primeFactorsWithBudget(n.Int64(), 1000000)
		if err != nil {
			return CalcVal{}, err
		}
		var items []CalcVal
		for _, f := range facs {
			items = append(items, CalcVal{Kind: ValKindRat, Rat: big.NewRat(f, 1)})
		}
		return CalcVal{Kind: ValKindList, List: items}, nil

	case "factor":
		if len(args) != 1 {
			return CalcVal{}, fmt.Errorf("ErrSyntax: factor expects 1 argument")
		}
		v := args[0]
		if v.Kind != ValKindRat || !v.Rat.IsInt() {
			return CalcVal{}, fmt.Errorf("ErrDomain: factor expects an integer argument")
		}
		n := v.Rat.Num()
		if n.Sign() <= 0 {
			return CalcVal{}, fmt.Errorf("ErrDomain: factor expects positive integer")
		}
		if n.Cmp(big.NewInt(100000000000000)) > 0 {
			return CalcVal{}, fmt.Errorf("ErrOverflow: factor input exceeds 10^14 limit")
		}
		facs, err := primeFactorsWithBudget(n.Int64(), 1000000)
		if err != nil {
			return CalcVal{}, err
		}
		counts := make(map[int64]int)
		var primes []int64
		for _, f := range facs {
			if counts[f] == 0 {
				primes = append(primes, f)
			}
			counts[f]++
		}
		var terms []string
		for _, p := range primes {
			c := counts[p]
			if c == 1 {
				terms = append(terms, fmt.Sprintf("%d", p))
			} else {
				terms = append(terms, fmt.Sprintf("%d^%d", p, c))
			}
		}
		return CalcVal{Kind: ValKindString, Str: strings.Join(terms, " * ")}, nil

	case "totient":
		if len(args) != 1 {
			return CalcVal{}, fmt.Errorf("ErrSyntax: totient expects 1 argument")
		}
		v := args[0]
		if v.Kind != ValKindRat || !v.Rat.IsInt() {
			return CalcVal{}, fmt.Errorf("ErrDomain: totient expects an integer argument")
		}
		n := v.Rat.Num()
		if n.Sign() <= 0 {
			return CalcVal{}, fmt.Errorf("ErrDomain: totient expects positive integer")
		}
		if n.Cmp(big.NewInt(100000000000000)) > 0 {
			return CalcVal{}, fmt.Errorf("ErrOverflow: totient input exceeds 10^14 limit")
		}
		facs, err := primeFactorsWithBudget(n.Int64(), 1000000)
		if err != nil {
			return CalcVal{}, err
		}
		seen := make(map[int64]bool)
		res := n.Int64()
		for _, p := range facs {
			if !seen[p] {
				seen[p] = true
				res = res / p * (p - 1)
			}
		}
		return CalcVal{Kind: ValKindRat, Rat: big.NewRat(res, 1)}, nil

	case "modexp":
		if len(args) != 3 {
			return CalcVal{}, fmt.Errorf("ErrSyntax: modexp expects 3 arguments: (base, exp, mod)")
		}
		v1, v2, v3 := args[0], args[1], args[2]
		if v1.Kind != ValKindRat || !v1.Rat.IsInt() || v2.Kind != ValKindRat || !v2.Rat.IsInt() || v3.Kind != ValKindRat || !v3.Rat.IsInt() {
			return CalcVal{}, fmt.Errorf("ErrDomain: modexp expects integer arguments")
		}
		base := v1.Rat.Num()
		exp := v2.Rat.Num()
		mod := v3.Rat.Num()
		if mod.Sign() == 0 {
			return CalcVal{}, fmt.Errorf("ErrDivZero: modexp with modulus zero")
		}
		// Pre-check exponent bit size ceiling (H1)
		if exp.BitLen() > 2000000 {
			return CalcVal{}, fmt.Errorf("ErrOverflow: modexp exponent exceeds bit ceiling")
		}
		res := new(big.Int).Exp(base, exp, mod)
		return CalcVal{Kind: ValKindRat, Rat: new(big.Rat).SetInt(res)}, nil

	case "modinv":
		if len(args) != 2 {
			return CalcVal{}, fmt.Errorf("ErrSyntax: modinv expects 2 arguments: (a, m)")
		}
		v1, v2 := args[0], args[1]
		if v1.Kind != ValKindRat || !v1.Rat.IsInt() || v2.Kind != ValKindRat || !v2.Rat.IsInt() {
			return CalcVal{}, fmt.Errorf("ErrDomain: modinv expects integer arguments")
		}
		a := v1.Rat.Num()
		m := v2.Rat.Num()
		if m.Sign() == 0 {
			return CalcVal{}, fmt.Errorf("ErrDivZero: modinv with modulus zero")
		}
		res := new(big.Int).ModInverse(a, m)
		if res == nil {
			return CalcVal{}, fmt.Errorf("ErrDomain: %v has no modular inverse modulo %v", a, m)
		}
		return CalcVal{Kind: ValKindRat, Rat: new(big.Rat).SetInt(res)}, nil

	case "sum_range":
		if len(args) != 2 {
			return CalcVal{}, fmt.Errorf("ErrSyntax: sum_range expects 2 arguments: (start, end)")
		}
		v1, v2 := args[0], args[1]
		if v1.Kind != ValKindRat || !v1.Rat.IsInt() || v2.Kind != ValKindRat || !v2.Rat.IsInt() {
			return CalcVal{}, fmt.Errorf("ErrDomain: sum_range expects integer arguments")
		}
		start := v1.Rat.Num()
		end := v2.Rat.Num()
		// sum = (start + end) * (end - start + 1) / 2
		sumTerm := new(big.Int).Add(start, end)
		countTerm := new(big.Int).Sub(end, start)
		countTerm.Add(countTerm, big.NewInt(1))
		prod := new(big.Int).Mul(sumTerm, countTerm)
		res := prod.Div(prod, big.NewInt(2))
		return CalcVal{Kind: ValKindRat, Rat: new(big.Rat).SetInt(res)}, nil

	case "base":
		if len(args) != 2 {
			return CalcVal{}, fmt.Errorf("ErrSyntax: base expects 2 arguments: (n, radix)")
		}
		v1, v2 := args[0], args[1]
		if v1.Kind != ValKindRat || !v1.Rat.IsInt() || v2.Kind != ValKindRat || !v2.Rat.IsInt() {
			return CalcVal{}, fmt.Errorf("ErrDomain: base expects integer arguments")
		}
		n := v1.Rat.Num()
		radix := int(v2.Rat.Num().Int64())
		if radix < 2 || radix > 36 {
			return CalcVal{}, fmt.Errorf("ErrDomain: base radix must be between 2 and 36")
		}
		if radix == 16 {
			// Uppercase digits, as a quoted string
			return CalcVal{Kind: ValKindString, Str: strings.ToUpper(n.Text(16))}, nil
		}
		return CalcVal{Kind: ValKindRawString, Str: n.Text(radix)}, nil

	case "det":
		if len(args) != 4 {
			return CalcVal{}, fmt.Errorf("ErrSyntax: det (2x2) expects 4 arguments (a, b, c, d)")
		}
		ad, err := mulCalcVals(args[0], args[3])
		if err != nil {
			return CalcVal{}, err
		}
		bc, err := mulCalcVals(args[1], args[2])
		if err != nil {
			return CalcVal{}, err
		}
		return subCalcVals(ad, bc)

	case "det3":
		if len(args) != 9 {
			return CalcVal{}, fmt.Errorf("ErrSyntax: det3 (3x3) expects 9 arguments")
		}
		a, b, c := args[0], args[1], args[2]
		d, e, f := args[3], args[4], args[5]
		g, h, i := args[6], args[7], args[8]

		ei, err := mulCalcVals(e, i)
		if err != nil {
			return CalcVal{}, err
		}
		fh, err := mulCalcVals(f, h)
		if err != nil {
			return CalcVal{}, err
		}
		t1, err := subCalcVals(ei, fh)
		if err != nil {
			return CalcVal{}, err
		}
		at1, err := mulCalcVals(a, t1)
		if err != nil {
			return CalcVal{}, err
		}

		di, err := mulCalcVals(d, i)
		if err != nil {
			return CalcVal{}, err
		}
		fg, err := mulCalcVals(f, g)
		if err != nil {
			return CalcVal{}, err
		}
		t2, err := subCalcVals(di, fg)
		if err != nil {
			return CalcVal{}, err
		}
		bt2, err := mulCalcVals(b, t2)
		if err != nil {
			return CalcVal{}, err
		}

		dh, err := mulCalcVals(d, h)
		if err != nil {
			return CalcVal{}, err
		}
		eg, err := mulCalcVals(e, g)
		if err != nil {
			return CalcVal{}, err
		}
		t3, err := subCalcVals(dh, eg)
		if err != nil {
			return CalcVal{}, err
		}
		ct3, err := mulCalcVals(c, t3)
		if err != nil {
			return CalcVal{}, err
		}

		diff, err := subCalcVals(at1, bt2)
		if err != nil {
			return CalcVal{}, err
		}
		return addCalcVals(diff, ct3)

	case "dot":
		if len(args) == 4 {
			p1, err := mulCalcVals(args[0], args[2])
			if err != nil {
				return CalcVal{}, err
			}
			p2, err := mulCalcVals(args[1], args[3])
			if err != nil {
				return CalcVal{}, err
			}
			return addCalcVals(p1, p2)
		}
		if len(args) == 6 {
			p1, err := mulCalcVals(args[0], args[3])
			if err != nil {
				return CalcVal{}, err
			}
			p2, err := mulCalcVals(args[1], args[4])
			if err != nil {
				return CalcVal{}, err
			}
			p3, err := mulCalcVals(args[2], args[5])
			if err != nil {
				return CalcVal{}, err
			}
			sum, err := addCalcVals(p1, p2)
			if err != nil {
				return CalcVal{}, err
			}
			return addCalcVals(sum, p3)
		}
		return CalcVal{}, fmt.Errorf("ErrSyntax: dot expects 4 or 6 arguments")

	case "convert":
		if len(args) != 3 {
			return CalcVal{}, fmt.Errorf("ErrSyntax: convert expects 3 arguments: (val, from_unit, to_unit)")
		}
		valArg := args[0]
		fromArg := args[1]
		toArg := args[2]

		var r *big.Rat
		var fromStr string

		if valArg.Kind == ValKindUnit {
			r = valArg.UnitVal
			fromStr = strings.Trim(fromArg.String(), `"'`)
			if fromStr == "" {
				fromStr = valArg.UnitName
			}
		} else if valArg.Kind == ValKindRat {
			r = valArg.Rat
			fromStr = strings.Trim(fromArg.String(), `"'`)
		} else {
			return CalcVal{}, fmt.Errorf("ErrTypeMismatch: convert value must be a number or unit value")
		}

		toStr := strings.Trim(toArg.String(), `"'`)
		resRat, err := ConvertUnits(r, fromStr, toStr)
		if err != nil {
			return CalcVal{}, err
		}
		return CalcVal{
			Kind:       ValKindUnit,
			UnitVal:    resRat,
			UnitName:   toStr,
			ForcedFrac: valArg.ForcedFrac,
			ForcedDec:  valArg.ForcedDec,
		}, nil

	default:
		return CalcVal{}, fmt.Errorf("ErrSyntax: unknown function %q", fn)
	}
}

func evalSqrt(arg CalcVal) (CalcVal, error) {
	if arg.Kind == ValKindExactNum {
		res, ok := arg.Exact.Sqrt()
		if ok {
			if rat, isR := res.AsRational(); isR {
				return CalcVal{Kind: ValKindRat, Rat: rat}, nil
			}
			return CalcVal{Kind: ValKindExactNum, Exact: res}, nil
		}
		if !arg.Exact.IsReal() {
			return CalcVal{}, fmt.Errorf("ErrDomain: sqrt cannot be simplified exactly: %s", arg.Exact.String())
		}
	}
	if arg.Kind == ValKindSurd {
		ex := calcValToExactNum(arg)
		res, ok := ex.Sqrt()
		if ok {
			if rat, isR := res.AsRational(); isR {
				return CalcVal{Kind: ValKindRat, Rat: rat}, nil
			}
			return CalcVal{Kind: ValKindExactNum, Exact: res}, nil
		}
	}

	if arg.Kind == ValKindRat {
		r := arg.Rat
		if r.Sign() < 0 {
			ex := calcValToExactNum(arg)
			res, ok := ex.Sqrt()
			if ok {
				return CalcVal{Kind: ValKindExactNum, Exact: res}, nil
			}
			return CalcVal{}, fmt.Errorf("ErrDomain: square root of negative number")
		}
		if r.Sign() == 0 {
			return CalcVal{Kind: ValKindRat, Rat: big.NewRat(0, 1)}, nil
		}

		num := r.Num()
		den := r.Denom()

		numExt, numSqFree, err := simplifyRadicalBig(num, 50000)
		if err != nil {
			return CalcVal{}, err
		}
		denExt, denSqFree, err := simplifyRadicalBig(den, 50000)
		if err != nil {
			return CalcVal{}, err
		}

		// Rationalize denominator: multiply by sqrt(denSqFree) / sqrt(denSqFree)
		rad := new(big.Int).Mul(numSqFree, denSqFree)
		ext, sqFree, err := simplifyRadicalBig(rad, 50000)
		if err != nil {
			return CalcVal{}, err
		}

		totalA := new(big.Int).Mul(numExt, ext)
		totalC := new(big.Int).Mul(denExt, denSqFree)
		g := new(big.Int).GCD(nil, nil, totalA, totalC)
		totalA.Quo(totalA, g)
		totalC.Quo(totalC, g)

		if sqFree.Cmp(big.NewInt(1)) == 0 {
			// Perfect rational square
			return CalcVal{Kind: ValKindRat, Rat: new(big.Rat).SetFrac(totalA, totalC)}, nil
		}

		return CalcVal{
			Kind: ValKindSurd,
			Surd: &SurdVal{
				sign: 1,
				a:    totalA,
				b:    sqFree,
				c:    totalC,
			},
		}, nil
	}

	f := toFloat(arg)
	if f < 0 {
		return CalcVal{}, fmt.Errorf("ErrDomain: square root of negative number")
	}
	return CalcVal{Kind: ValKindFloat, Float: math.Sqrt(f)}, nil
}

func evalCbrt(arg CalcVal) (CalcVal, error) {
	if arg.Kind == ValKindRat {
		r := arg.Rat
		sign := r.Sign()
		absNum := new(big.Int).Abs(r.Num())
		absDen := r.Denom()

		numRoot, numExact := integerCubeRoot(absNum)
		denRoot, denExact := integerCubeRoot(absDen)

		if numExact && denExact {
			res := new(big.Rat).SetFrac(numRoot, denRoot)
			if sign < 0 {
				res.Neg(res)
			}
			return CalcVal{Kind: ValKindRat, Rat: res}, nil
		}
	}
	f := toFloat(arg)
	return CalcVal{Kind: ValKindFloat, Float: math.Cbrt(f)}, nil
}

func integerCubeRoot(n *big.Int) (*big.Int, bool) {
	if n == nil || n.Sign() < 0 {
		return nil, false
	}
	if n.Sign() == 0 {
		return big.NewInt(0), true
	}
	if n.Cmp(big.NewInt(1)) == 0 {
		return big.NewInt(1), true
	}
	// Binary search for exact cube root
	low := big.NewInt(1)
	high := new(big.Int).Lsh(big.NewInt(1), uint(n.BitLen()/3+2))
	mid := new(big.Int)
	cube := new(big.Int)

	for low.Cmp(high) <= 0 {
		mid.Add(low, high)
		mid.Rsh(mid, 1)

		cube.Mul(mid, mid)
		cube.Mul(cube, mid)

		cmp := cube.Cmp(n)
		if cmp == 0 {
			return new(big.Int).Set(mid), true
		}
		if cmp < 0 {
			low.Add(mid, big.NewInt(1))
		} else {
			high.Sub(mid, big.NewInt(1))
		}
	}
	return nil, false
}

func integerNthRoot(n *big.Int, q int64) (*big.Int, bool) {
	if n == nil || n.Sign() < 0 || q <= 0 {
		return nil, false
	}
	if n.Sign() == 0 {
		return big.NewInt(0), true
	}
	if n.Cmp(big.NewInt(1)) == 0 {
		return big.NewInt(1), true
	}
	if q == 1 {
		return new(big.Int).Set(n), true
	}
	if q == 2 {
		return integerSquareRoot(n)
	}
	if q == 3 {
		return integerCubeRoot(n)
	}
	if q > int64(n.BitLen()) {
		return nil, false
	}

	low := big.NewInt(1)
	high := new(big.Int).Lsh(big.NewInt(1), uint(int64(n.BitLen())/q+2))
	mid := new(big.Int)
	pow := new(big.Int)
	bigQ := big.NewInt(q)

	for low.Cmp(high) <= 0 {
		mid.Add(low, high)
		mid.Rsh(mid, 1)

		pow.Exp(mid, bigQ, nil)

		cmp := pow.Cmp(n)
		if cmp == 0 {
			return new(big.Int).Set(mid), true
		}
		if cmp < 0 {
			low.Add(mid, big.NewInt(1))
		} else {
			high.Sub(mid, big.NewInt(1))
		}
	}
	return nil, false
}

// simplifyRadicalBig extracts square factors from n: n = a^2 * b.
// Returns (a, b) such that a^2 * b == n.
// Bounded by maxSteps of trial division.
func simplifyRadicalBig(n *big.Int, maxSteps int) (*big.Int, *big.Int, error) {
	if n == nil || n.Sign() < 0 {
		return nil, nil, fmt.Errorf("ErrDomain: square root of negative number")
	}
	if n.Sign() == 0 {
		return big.NewInt(0), big.NewInt(0), nil
	}
	if n.Cmp(big.NewInt(1)) == 0 {
		return big.NewInt(1), big.NewInt(1), nil
	}

	// Operand size pre-check: 4096 bits
	if n.BitLen() > 4096 {
		return nil, nil, fmt.Errorf("ErrOverflow: integer exceeds square root precision limit")
	}

	// 1. Direct perfect square test
	sq, exact := integerSquareRoot(n)
	if exact {
		return sq, big.NewInt(1), nil
	}

	extracted := big.NewInt(1)
	rem := new(big.Int).Set(n)

	// 2. Factor out powers of 4 via trailing zero bits
	tz := rem.TrailingZeroBits()
	if tz >= 2 {
		shift := (tz / 2) * 2
		rem.Rsh(rem, shift)
		extracted.Lsh(extracted, tz/2)

		if rem.Cmp(big.NewInt(1)) == 0 {
			return extracted, big.NewInt(1), nil
		}
		if s, ok := integerSquareRoot(rem); ok {
			extracted.Mul(extracted, s)
			return extracted, big.NewInt(1), nil
		}
	}

	// 3. Trial division for odd square factors: d = 3, 5, 7, ...
	steps := 0
	d := big.NewInt(3)
	two := big.NewInt(2)
	d2 := new(big.Int)
	remMod := new(big.Int)

	for {
		d2.Mul(d, d)
		if d2.Cmp(rem) > 0 {
			// rem is proved square-free
			break
		}

		steps++
		if steps > maxSteps {
			// Work limit reached: stop factoring, leave remainder under radical.
			// Invariant holds: extracted^2 * rem == n.
			break
		}

		// Check if d2 divides rem
		remMod.Mod(rem, d2)
		for remMod.Sign() == 0 {
			extracted.Mul(extracted, d)
			rem.Quo(rem, d2)

			if rem.Cmp(big.NewInt(1)) == 0 {
				return extracted, big.NewInt(1), nil
			}
			if s, ok := integerSquareRoot(rem); ok {
				extracted.Mul(extracted, s)
				return extracted, big.NewInt(1), nil
			}
			remMod.Mod(rem, d2)
		}

		d.Add(d, two)
	}

	return extracted, rem, nil
}

// -----------------------------------------------------------------------------
// Public Evaluation Entrypoints
// -----------------------------------------------------------------------------

// EvaluateInfix evaluates an infix mathematical expression against globalRegisters.
func EvaluateInfix(expr string) (string, error) {
	val, err := EvaluateInfixToVal(expr, globalRegisters)
	if err != nil {
		return "", err
	}
	return val.String(), nil
}

// EvaluateInfixWithStore evaluates an infix mathematical expression against the given RegisterStore.
func EvaluateInfixWithStore(expr string, store *RegisterStore) (string, error) {
	val, err := EvaluateInfixToVal(expr, store)
	if err != nil {
		return "", err
	}
	return val.String(), nil
}

// EvaluateInfixToVal evaluates an infix mathematical expression against the given RegisterStore.
func EvaluateInfixToVal(expr string, store *RegisterStore) (CalcVal, error) {
	lexer := NewLexer(expr)
	var tokens []Token
	for {
		tok, err := lexer.NextToken()
		if err != nil {
			return CalcVal{}, err
		}
		tokens = append(tokens, tok)
		if tok.Type == TokEOF {
			break
		}
	}

	if len(tokens) == 0 || tokens[0].Type == TokEOF {
		return CalcVal{}, fmt.Errorf("ErrSyntax: empty expression")
	}

	parser := NewParser(tokens, store)
	res, err := parser.parseExpression()
	if err != nil {
		return CalcVal{}, err
	}

	if parser.cur().Type != TokEOF {
		return CalcVal{}, fmt.Errorf("ErrSyntax: unexpected token %q at column %d", parser.cur().Text, parser.cur().Col)
	}

	if res.Kind == ValKindFloat {
		if math.IsNaN(res.Float) {
			return CalcVal{}, fmt.Errorf("ErrDomain: NaN value cannot be bound")
		}
		if math.IsInf(res.Float, 0) {
			return CalcVal{}, fmt.Errorf("ErrOverflow: infinite value cannot be bound")
		}
	}

	return res, nil
}
