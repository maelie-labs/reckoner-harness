package main

import (
	"fmt"
	"math"
	"math/big"
	"sort"
	"strings"
)

// -----------------------------------------------------------------------------
// Multivariate Polynomial Representation (for expand and multivariate factor)
// -----------------------------------------------------------------------------

type VarPower struct {
	Var string
	Exp int
}

// MonoKey represents a product of variables with integer powers: e.g. x^2 * y
type MonoKey struct {
	Powers []VarPower // sorted by Var
}

func NewMonoKey(powers map[string]int) MonoKey {
	var list []VarPower
	for v, exp := range powers {
		if exp > 0 {
			list = append(list, VarPower{Var: v, Exp: exp})
		}
	}
	sort.Slice(list, func(i, j int) bool {
		return list[i].Var < list[j].Var
	})
	return MonoKey{Powers: list}
}

func (mk MonoKey) TotalDegree() int {
	deg := 0
	for _, vp := range mk.Powers {
		deg += vp.Exp
	}
	return deg
}

func (mk MonoKey) CmpDegLex(other MonoKey) int {
	d1 := mk.TotalDegree()
	d2 := other.TotalDegree()
	if d1 != d2 {
		if d1 > d2 {
			return 1
		}
		return -1
	}
	i, j := 0, 0
	for i < len(mk.Powers) && j < len(other.Powers) {
		v1 := mk.Powers[i].Var
		v2 := other.Powers[j].Var
		if v1 < v2 {
			return 1
		} else if v1 > v2 {
			return -1
		} else {
			e1 := mk.Powers[i].Exp
			e2 := other.Powers[j].Exp
			if e1 != e2 {
				if e1 > e2 {
					return 1
				}
				return -1
			}
			i++
			j++
		}
	}
	if i < len(mk.Powers) {
		return 1
	}
	if j < len(other.Powers) {
		return -1
	}
	return 0
}

func (mk MonoKey) String() string {
	if len(mk.Powers) == 0 {
		return ""
	}
	var parts []string
	for _, vp := range mk.Powers {
		if vp.Exp == 1 {
			parts = append(parts, vp.Var)
		} else {
			parts = append(parts, fmt.Sprintf("%s^%d", vp.Var, vp.Exp))
		}
	}
	return strings.Join(parts, "*")
}

func (mk MonoKey) Mul(other MonoKey) MonoKey {
	powers := make(map[string]int)
	for _, vp := range mk.Powers {
		powers[vp.Var] += vp.Exp
	}
	for _, vp := range other.Powers {
		powers[vp.Var] += vp.Exp
	}
	return NewMonoKey(powers)
}

func (mk MonoKey) CanonicalString() string {
	var parts []string
	for _, vp := range mk.Powers {
		parts = append(parts, fmt.Sprintf("%s:%d", vp.Var, vp.Exp))
	}
	return strings.Join(parts, ",")
}

// MultiPoly represents a multivariate polynomial with exact rational coefficients.
type MultiPoly struct {
	Terms map[string]*monoTerm // canonical key -> term
}

type monoTerm struct {
	key   MonoKey
	coeff *ExactNum
}

func NewMultiPoly() *MultiPoly {
	return &MultiPoly{Terms: make(map[string]*monoTerm)}
}

func NewMultiPolyConst(en *ExactNum) *MultiPoly {
	p := NewMultiPoly()
	if en != nil && !en.IsZero() {
		mk := NewMonoKey(nil)
		p.Terms[mk.CanonicalString()] = &monoTerm{key: mk, coeff: en.Clone()}
	}
	return p
}

func NewMultiPolyConstRat(r *big.Rat) *MultiPoly {
	return NewMultiPolyConst(NewExactRational(r))
}

func NewMultiPolyVar(varName string) *MultiPoly {
	p := NewMultiPoly()
	mk := NewMonoKey(map[string]int{varName: 1})
	p.Terms[mk.CanonicalString()] = &monoTerm{key: mk, coeff: NewExactInt(1)}
	return p
}

func (mp *MultiPoly) Degree() int {
	maxDeg := 0
	for _, t := range mp.Terms {
		deg := t.key.TotalDegree()
		if deg > maxDeg {
			maxDeg = deg
		}
	}
	return maxDeg
}

func (mp *MultiPoly) Clone() *MultiPoly {
	res := NewMultiPoly()
	for k, t := range mp.Terms {
		res.Terms[k] = &monoTerm{key: t.key, coeff: t.coeff.Clone()}
	}
	return res
}

func (mp *MultiPoly) AsConstant() (*ExactNum, bool) {
	if mp == nil || mp.IsZero() {
		return NewExactZero(), true
	}
	if len(mp.Terms) == 1 {
		for _, t := range mp.Terms {
			if len(t.key.Powers) == 0 {
				return t.coeff.Clone(), true
			}
		}
	}
	return nil, false
}

func (mp *MultiPoly) Add(other *MultiPoly) *MultiPoly {
	res := mp.Clone()
	for k, t := range other.Terms {
		if cur, ok := res.Terms[k]; ok {
			sum := cur.coeff.Add(t.coeff)
			if sum.IsZero() {
				delete(res.Terms, k)
			} else {
				cur.coeff = sum
			}
		} else {
			res.Terms[k] = &monoTerm{key: t.key, coeff: t.coeff.Clone()}
		}
	}
	return res
}

func (mp *MultiPoly) Sub(other *MultiPoly) *MultiPoly {
	res := mp.Clone()
	for k, t := range other.Terms {
		if cur, ok := res.Terms[k]; ok {
			diff := cur.coeff.Sub(t.coeff)
			if diff.IsZero() {
				delete(res.Terms, k)
			} else {
				cur.coeff = diff
			}
		} else {
			res.Terms[k] = &monoTerm{key: t.key, coeff: t.coeff.Neg()}
		}
	}
	return res
}

func (mp *MultiPoly) Neg() *MultiPoly {
	zero := NewMultiPoly()
	return zero.Sub(mp)
}

func (mp *MultiPoly) Mul(other *MultiPoly) (*MultiPoly, error) {
	// Deterministic limit check: degree <= 100
	if mp.Degree()+other.Degree() > 100 {
		return nil, fmt.Errorf("ErrOverflow: polynomial degree exceeds maximum limit of 100")
	}
	// Deterministic limit check: term count product <= 1000
	if len(mp.Terms)*len(other.Terms) > 1000 {
		return nil, fmt.Errorf("ErrOverflow: polynomial expansion exceeds limit of 1000 terms")
	}

	res := NewMultiPoly()
	for _, t1 := range mp.Terms {
		for _, t2 := range other.Terms {
			c, ok := t1.coeff.Mul(t2.coeff)
			if !ok {
				return nil, fmt.Errorf("ErrOverflow: polynomial multiplication exceeded exact number bounds")
			}
			mk := t1.key.Mul(t2.key)
			k := mk.CanonicalString()

			if cur, ok := res.Terms[k]; ok {
				sum := cur.coeff.Add(c)
				if sum.IsZero() {
					delete(res.Terms, k)
				} else {
					cur.coeff = sum
				}
			} else {
				if len(res.Terms) >= 1000 {
					return nil, fmt.Errorf("ErrOverflow: polynomial term count exceeds limit of 1000")
				}
				res.Terms[k] = &monoTerm{key: mk, coeff: c}
			}
		}
	}
	return res, nil
}

func (mp *MultiPoly) Pow(exp int) (*MultiPoly, error) {
	if exp < 0 || exp > 100 {
		return nil, fmt.Errorf("ErrOverflow: polynomial exponent exceeds maximum 100")
	}
	if exp == 0 {
		return NewMultiPolyConst(NewExactInt(1)), nil
	}
	if mp.Degree()*exp > 100 {
		return nil, fmt.Errorf("ErrOverflow: polynomial degree exceeds maximum limit of 100")
	}

	res := NewMultiPolyConst(NewExactInt(1))
	base := mp.Clone()
	curExp := exp

	for curExp > 0 {
		if curExp%2 == 1 {
			var err error
			res, err = res.Mul(base)
			if err != nil {
				return nil, err
			}
		}
		if curExp > 1 {
			var err error
			base, err = base.Mul(base)
			if err != nil {
				return nil, err
			}
		}
		curExp /= 2
	}
	return res, nil
}

func (mp *MultiPoly) IsZero() bool {
	return len(mp.Terms) == 0
}

func (mp *MultiPoly) String() string {
	if mp.IsZero() {
		return "0"
	}

	var terms []*monoTerm
	for _, t := range mp.Terms {
		terms = append(terms, t)
	}

	// Sort terms: higher total degree first, then graded lexicographical order
	sort.Slice(terms, func(i, j int) bool {
		return terms[i].key.CmpDegLex(terms[j].key) > 0
	})

	var parts []string
	for i, t := range terms {
		c := t.coeff
		varStr := t.key.String()

		if varStr == "" {
			// Constant term
			cStr := c.String()
			if i == 0 {
				parts = append(parts, cStr)
			} else {
				if strings.HasPrefix(cStr, "-") {
					parts = append(parts, " - "+strings.TrimPrefix(cStr, "-"))
				} else {
					parts = append(parts, " + "+cStr)
				}
			}
			continue
		}

		// Non-constant term: c * varStr
		if rat, ok := c.AsRational(); ok {
			isNeg := rat.Sign() < 0
			absRat := new(big.Rat).Abs(rat)
			signStr := ""
			if i == 0 {
				if isNeg {
					signStr = "-"
				}
			} else {
				if isNeg {
					signStr = " - "
				} else {
					signStr = " + "
				}
			}

			coeffStr := ""
			if absRat.Cmp(big.NewRat(1, 1)) != 0 {
				coeffStr = formatRational(absRat, false, false, false, 0) + "*"
			}
			parts = append(parts, fmt.Sprintf("%s%s%s", signStr, coeffStr, varStr))
		} else {
			// Non-rational coefficient (surds, complex, pi)
			cStr := c.String()
			if len(c.Terms) > 1 {
				// Multiple basis terms: parenthesize, e.g. (1+sqrt(2))*x
				if i == 0 {
					parts = append(parts, fmt.Sprintf("(%s)*%s", cStr, varStr))
				} else {
					parts = append(parts, fmt.Sprintf(" + (%s)*%s", cStr, varStr))
				}
			} else {
				// Single basis term: e.g. 2*sqrt(2), sqrt(2), -2*sqrt(2), pi, 3*i
				isNeg := strings.HasPrefix(cStr, "-")
				absCStr := strings.TrimPrefix(cStr, "-")
				signStr := ""
				if i == 0 {
					if isNeg {
						signStr = "-"
					}
				} else {
					if isNeg {
						signStr = " - "
					} else {
						signStr = " + "
					}
				}
				coeffStr := ""
				if absCStr == "1" {
					coeffStr = ""
				} else {
					coeffStr = absCStr + "*"
				}
				parts = append(parts, fmt.Sprintf("%s%s%s", signStr, coeffStr, varStr))
			}
		}
	}

	return strings.Join(parts, "")
}

// -----------------------------------------------------------------------------
// Polynomial Parsing & Expand
// -----------------------------------------------------------------------------

type polyParser struct {
	tokens []Token
	pos    int
	store  *RegisterStore
}

func newPolyParser(tokens []Token, store *RegisterStore) *polyParser {
	return &polyParser{tokens: tokens, pos: 0, store: store}
}

func (pp *polyParser) cur() Token {
	if pp.pos >= len(pp.tokens) {
		return Token{Type: TokEOF}
	}
	return pp.tokens[pp.pos]
}

func (pp *polyParser) next() Token {
	tok := pp.cur()
	if pp.pos < len(pp.tokens) {
		pp.pos++
	}
	return tok
}

func (pp *polyParser) parseExpr() (*MultiPoly, error) {
	return pp.parseAdditive()
}

func (pp *polyParser) parseAdditive() (*MultiPoly, error) {
	left, err := pp.parseMultiplicative()
	if err != nil {
		return nil, err
	}

	for pp.cur().Type == TokPlus || pp.cur().Type == TokMinus {
		op := pp.next().Type
		right, err := pp.parseMultiplicative()
		if err != nil {
			return nil, err
		}
		if op == TokPlus {
			left = left.Add(right)
		} else {
			left = left.Sub(right)
		}
	}
	return left, nil
}

func (pp *polyParser) parseMultiplicative() (*MultiPoly, error) {
	left, err := pp.parseUnary()
	if err != nil {
		return nil, err
	}

	for pp.cur().Type == TokStar || pp.cur().Type == TokSlash {
		op := pp.next().Type
		right, err := pp.parseUnary()
		if err != nil {
			return nil, err
		}
		if op == TokStar {
			var err error
			left, err = left.Mul(right)
			if err != nil {
				return nil, err
			}
		} else {
			// Division by constant
			c, ok := right.AsConstant()
			if !ok || c.IsZero() {
				return nil, fmt.Errorf("ErrUnsupported: polynomial division in expand only supports constant divisor")
			}
			invC, okInv, errInv := NewExactInt(1).Div(c)
			if errInv != nil {
				return nil, errInv
			}
			if !okInv {
				return nil, fmt.Errorf("ErrUnsupported: exact division by constant %s not possible", c.String())
			}
			leftConst := NewMultiPolyConst(invC)
			var err error
			left, err = left.Mul(leftConst)
			if err != nil {
				return nil, err
			}
		}
	}
	if pp.cur().Type == TokLParen || pp.cur().Type == TokIdent || pp.cur().Type == TokNum || pp.cur().Type == TokRegister {
		return nil, fmt.Errorf("ErrSyntax: implicit multiplication not supported; use explicit '*'")
	}
	return left, nil
}

func (pp *polyParser) parseUnary() (*MultiPoly, error) {
	if pp.cur().Type == TokMinus {
		pp.next()
		operand, err := pp.parsePower()
		if err != nil {
			return nil, err
		}
		zero := NewMultiPoly()
		return zero.Sub(operand), nil
	}
	if pp.cur().Type == TokPlus {
		pp.next()
	}
	return pp.parsePower()
}

func (pp *polyParser) parsePower() (*MultiPoly, error) {
	base, err := pp.parsePrimary()
	if err != nil {
		return nil, err
	}

	if pp.cur().Type == TokPow {
		pp.next()
		// Exponent must be integer
		expTok := pp.next()
		if expTok.Type != TokNum || !expTok.NumRat.IsInt() {
			return nil, fmt.Errorf("ErrSyntax: polynomial exponent must be a non-negative integer")
		}
		if !expTok.NumRat.Num().IsInt64() {
			return nil, fmt.Errorf("ErrOverflow: polynomial exponent exceeds limit")
		}
		exp := expTok.NumRat.Num().Int64()
		if exp < 0 || exp > 100 {
			return nil, fmt.Errorf("ErrOverflow: polynomial exponent exceeds maximum 100")
		}
		return base.Pow(int(exp))
	}
	return base, nil
}

func (pp *polyParser) parsePrimary() (*MultiPoly, error) {
	tok := pp.next()
	switch tok.Type {
	case TokNum:
		return NewMultiPolyConstRat(tok.NumRat), nil

	case TokIdent:
		name := tok.Text
		if name == "pi" {
			return NewMultiPolyConst(NewExactPi()), nil
		}
		if name == "i" {
			return NewMultiPolyConst(NewExactI()), nil
		}
		if pp.cur().Type == TokLParen {
			if name == "sqrt" {
				pp.next() // consume TokLParen
				inner, err := pp.parseExpr()
				if err != nil {
					return nil, err
				}
				if pp.cur().Type != TokRParen {
					return nil, fmt.Errorf("ErrSyntax: expected ')' after sqrt argument")
				}
				pp.next() // consume TokRParen
				c, ok := inner.AsConstant()
				if !ok {
					return nil, fmt.Errorf("ErrUnsupported: sqrt of non-constant polynomial")
				}
				sqrtC, ok := c.Sqrt()
				if !ok {
					return nil, fmt.Errorf("ErrUnsupported: sqrt could not be evaluated exactly")
				}
				return NewMultiPolyConst(sqrtC), nil
			}
			if isKnownMathFunction(name) {
				return nil, fmt.Errorf("ErrUnsupported: functions like %s(...) are not supported in solve", name)
			}
		}
		if name == "e" {
			return nil, fmt.Errorf("ErrUnsupported: constant e in polynomials")
		}
		return NewMultiPolyVar(name), nil

	case TokRegister:
		if pp.store != nil {
			idx := tok.RegIdx
			val, err := pp.store.Get(idx)
			if err != nil {
				return nil, err
			}
			if val.Kind == ValKindPoly && val.Poly != nil {
				return val.Poly.Clone(), nil
			}
			if val.Kind == ValKindRat && val.Rat != nil {
				return NewMultiPolyConstRat(val.Rat), nil
			}
			if val.Kind == ValKindExactNum && val.Exact != nil {
				return NewMultiPolyConst(val.Exact), nil
			}
			if val.Kind == ValKindSurd && val.Surd != nil {
				return NewMultiPolyConst(NewExactSurd(val.Surd)), nil
			}
		}
		return nil, fmt.Errorf("ErrSyntax: register cannot be parsed as polynomial")

	case TokLParen:
		inner, err := pp.parseExpr()
		if err != nil {
			return nil, err
		}
		if pp.cur().Type != TokRParen {
			return nil, fmt.Errorf("ErrSyntax: expected ')'")
		}
		pp.next()
		return inner, nil
	}

	return nil, fmt.Errorf("ErrSyntax: unexpected token %q in polynomial expression", tok.Text)
}

// ExpandPoly evaluates expand(expr) into canonical expanded polynomial string.
func ExpandPoly(expr string, store *RegisterStore) (string, *MultiPoly, error) {
	tokens, err := tokenizeExpr(expr)
	if err != nil {
		return "", nil, err
	}
	parser := newPolyParser(tokens, store)
	poly, err := parser.parseExpr()
	if err != nil {
		return "", nil, err
	}
	if parser.cur().Type != TokEOF {
		return "", nil, fmt.Errorf("ErrSyntax: unexpected token %q at column %d", parser.cur().Text, parser.cur().Col)
	}
	return poly.String(), poly, nil
}

// -----------------------------------------------------------------------------
// Univariate Polynomial Operations: divide, pgcd, roots, factor, evaluate
// -----------------------------------------------------------------------------

// RatPolyToMultiPoly converts a RatPoly in varName to MultiPoly.
func RatPolyToMultiPoly(p RatPoly, varName string) *MultiPoly {
	mp := NewMultiPoly()
	for deg, c := range p {
		if c != nil && c.Sign() != 0 {
			var mk MonoKey
			if deg == 0 {
				mk = NewMonoKey(nil)
			} else {
				mk = NewMonoKey(map[string]int{varName: deg})
			}
			mp.Terms[mk.CanonicalString()] = &monoTerm{key: mk, coeff: NewExactRational(c)}
		}
	}
	return mp
}

// MultiPolyToRatPoly converts a MultiPoly in a single variable to RatPoly.
func MultiPolyToRatPoly(mp *MultiPoly) (RatPoly, string, error) {
	varName := ""
	for _, t := range mp.Terms {
		for _, vp := range t.key.Powers {
			if varName == "" {
				varName = vp.Var
			} else if varName != vp.Var {
				return nil, "", fmt.Errorf("ErrUnsupported: expression is multivariate (%s, %s)", varName, vp.Var)
			}
		}
	}
	if varName == "" {
		varName = "x"
	}

	deg := mp.Degree()
	res := make(RatPoly, deg+1)
	for i := 0; i <= deg; i++ {
		res[i] = big.NewRat(0, 1)
	}

	for _, t := range mp.Terms {
		d := t.key.TotalDegree()
		rat, ok := t.coeff.AsRational()
		if !ok {
			return nil, "", fmt.Errorf("ErrUnsupported: polynomial has non-rational coefficient %s", t.coeff.String())
		}
		res[d] = new(big.Rat).Set(rat)
	}
	return res.Trim(), varName, nil
}

// DividePoly performs polynomial long division: p = q * Q + R with deg(R) < deg(q).
// Returns [Q, R] as a string and as polynomials.
func DividePoly(pStr, qStr string, store *RegisterStore) (string, error) {
	_, pMulti, err := ExpandPoly(pStr, store)
	if err != nil {
		return "", err
	}
	_, qMulti, err := ExpandPoly(qStr, store)
	if err != nil {
		return "", err
	}

	pRat, v1, err := MultiPolyToRatPoly(pMulti)
	if err != nil {
		return "", err
	}
	qRat, v2, err := MultiPolyToRatPoly(qMulti)
	if err != nil {
		return "", err
	}
	if v1 != v2 && qMulti.Degree() > 0 && pMulti.Degree() > 0 {
		return "", fmt.Errorf("ErrUnsupported: polynomials have different variables (%s, %s)", v1, v2)
	}

	pRat = pRat.Trim()
	qRat = qRat.Trim()
	if qRat.Degree() == 0 && qRat[0].Sign() == 0 {
		return "", fmt.Errorf("ErrDivZero: polynomial division by zero")
	}

	quo, rem := PolyDivRemRat(pRat, qRat)
	qMultiRes := RatPolyToMultiPoly(quo, v1)
	rMultiRes := RatPolyToMultiPoly(rem, v1)

	return fmt.Sprintf("[%s, %s]", qMultiRes.String(), rMultiRes.String()), nil
}

// PolyDivRemRat computes polynomial quotient and remainder over Q.
func PolyDivRemRat(p, q RatPoly) (RatPoly, RatPoly) {
	p = p.Trim()
	q = q.Trim()

	degP := p.Degree()
	degQ := q.Degree()

	if degP < degQ {
		return RatPoly{big.NewRat(0, 1)}, p
	}

	quo := make(RatPoly, degP-degQ+1)
	for i := range quo {
		quo[i] = big.NewRat(0, 1)
	}

	rem := make(RatPoly, len(p))
	for i, c := range p {
		rem[i] = new(big.Rat).Set(c)
	}

	leadQ := q.LeadingCoeff()

	for rem.Degree() >= degQ && !isPolyZero(rem) {
		curDeg := rem.Degree()
		coeff := new(big.Rat).Quo(rem.LeadingCoeff(), leadQ)
		quoDeg := curDeg - degQ
		quo[quoDeg] = coeff

		// rem = rem - coeff * x^(quoDeg) * q
		for i := 0; i <= degQ; i++ {
			term := new(big.Rat).Mul(coeff, q[i])
			rem[quoDeg+i] = new(big.Rat).Sub(rem[quoDeg+i], term)
		}
		rem = rem.Trim()
	}

	return quo.Trim(), rem.Trim()
}

func isPolyZero(p RatPoly) bool {
	p = p.Trim()
	return len(p) <= 1 && (len(p) == 0 || p[0].Sign() == 0)
}

// PolyGCD computes monic polynomial GCD over Q using Euclidean algorithm.
func PolyGCD(pStr, qStr string, store *RegisterStore) (string, error) {
	_, pMulti, err := ExpandPoly(pStr, store)
	if err != nil {
		return "", err
	}
	_, qMulti, err := ExpandPoly(qStr, store)
	if err != nil {
		return "", err
	}

	pRat, v1, err := MultiPolyToRatPoly(pMulti)
	if err != nil {
		return "", err
	}
	qRat, _, err := MultiPolyToRatPoly(qMulti)
	if err != nil {
		return "", err
	}

	a := pRat.Trim()
	b := qRat.Trim()

	for !isPolyZero(b) {
		_, rem := PolyDivRemRat(a, b)
		a = b
		b = rem
	}

	// Make monic
	lead := a.LeadingCoeff()
	if lead.Sign() != 0 && lead.Cmp(big.NewRat(1, 1)) != 0 {
		invLead := new(big.Rat).Inv(lead)
		for i := range a {
			a[i] = new(big.Rat).Mul(a[i], invLead)
		}
	}

	resMulti := RatPolyToMultiPoly(a.Trim(), v1)
	return resMulti.String(), nil
}

func sortRoots(exactRoots []ExactNum) {
	sort.Slice(exactRoots, func(i, j int) bool {
		r1 := exactRoots[i]
		r2 := exactRoots[j]
		if r1.IsReal() && r2.IsReal() {
			return r1.ToFloat64() < r2.ToFloat64()
		}
		if r1.IsReal() && !r2.IsReal() {
			return true
		}
		if !r1.IsReal() && r2.IsReal() {
			return false
		}
		re1 := r1.Re().ToFloat64()
		re2 := r2.Re().ToFloat64()
		if math.Abs(re1-re2) > 1e-9 {
			return re1 < re2
		}
		return r1.Im().ToFloat64() < r2.Im().ToFloat64()
	})
}

// RootsPoly returns exact roots with multiplicity for polynomial p up to degree 4.
func RootsPoly(pStr string, store *RegisterStore) (string, []ExactNum, error) {
	_, pMulti, err := ExpandPoly(pStr, store)
	if err != nil {
		return "", nil, err
	}
	pRat, _, err := MultiPolyToRatPoly(pMulti)
	if err != nil {
		// Non-rational coefficients: support degree <= 2 via quadratic formula
		deg := pMulti.Degree()
		if deg > 2 {
			return "", nil, fmt.Errorf("ErrUnsupported: roots over non-rational coefficients only supported for degree <= 2")
		}
		if deg == 0 {
			return "[]", nil, nil
		}
		var a, b, c *ExactNum
		b = NewExactZero()
		c = NewExactZero()
		for _, t := range pMulti.Terms {
			d := t.key.TotalDegree()
			if d == 2 {
				a = t.coeff
			} else if d == 1 {
				b = t.coeff
			} else if d == 0 {
				c = t.coeff
			}
		}
		if deg == 1 {
			if a == nil || a.IsZero() {
				return "", nil, fmt.Errorf("ErrSyntax: degenerate linear polynomial")
			}
			r, ok, err := b.Neg().Div(a)
			if err != nil || !ok {
				return "", nil, fmt.Errorf("ErrUnsupported: exact root division failed")
			}
			exactRoots := []ExactNum{*r}
			return fmt.Sprintf("[%s]", r.String()), exactRoots, nil
		}
		// deg == 2
		if a == nil || a.IsZero() {
			return "", nil, fmt.Errorf("ErrSyntax: degenerate quadratic polynomial")
		}
		bSq, ok := b.Mul(b)
		if !ok {
			return "", nil, fmt.Errorf("ErrOverflow: discriminant square out of bounds")
		}
		fourAC, ok := NewExactInt(4).Mul(a)
		if ok {
			fourAC, ok = fourAC.Mul(c)
		}
		if !ok {
			return "", nil, fmt.Errorf("ErrOverflow: discriminant 4ac out of bounds")
		}
		disc := bSq.Sub(fourAC)
		twoA, ok := NewExactInt(2).Mul(a)
		if !ok {
			return "", nil, fmt.Errorf("ErrOverflow: 2a out of bounds")
		}
		negB := b.Neg()

		sqrtD, ok := disc.Sqrt()
		if !ok {
			return "", nil, fmt.Errorf("ErrUnsupported: quadratic discriminant sqrt could not be evaluated exactly")
		}
		r1Num := negB.Sub(sqrtD)
		r2Num := negB.Add(sqrtD)
		r1, ok1, err1 := r1Num.Div(twoA)
		r2, ok2, err2 := r2Num.Div(twoA)
		if err1 != nil || err2 != nil || !ok1 || !ok2 {
			return "", nil, fmt.Errorf("ErrUnsupported: quadratic root division could not be evaluated exactly")
		}
		var exactRoots []ExactNum
		if disc.IsZero() {
			exactRoots = []ExactNum{*r1, *r1}
		} else {
			exactRoots = []ExactNum{*r1, *r2}
		}
		sortRoots(exactRoots)
		var parts []string
		for _, r := range exactRoots {
			parts = append(parts, r.String())
		}
		return fmt.Sprintf("[%s]", strings.Join(parts, ", ")), exactRoots, nil
	}

	pRat = pRat.Trim()
	deg := pRat.Degree()
	if deg > 4 {
		return "", nil, fmt.Errorf("ErrUnsupported: roots only supported for degree <= 4 (got %d)", deg)
	}
	if deg == 0 {
		return "[]", nil, nil
	}

	var exactRoots []ExactNum
	currP := pRat

	// Factor out x = 0 roots
	for currP.Degree() > 0 && currP[0].Sign() == 0 {
		exactRoots = append(exactRoots, *NewExactZero())
		nextP := make(RatPoly, currP.Degree())
		for i := 1; i <= currP.Degree(); i++ {
			nextP[i-1] = currP[i]
		}
		currP = nextP.Trim()
	}

	// Find rational roots
	for currP.Degree() > 2 {
		r, err := findSingleRationalRoot(currP)
		if err != nil || r == nil {
			break
		}
		exactRoots = append(exactRoots, *NewExactRational(r))
		currP = currP.DivSynthetic(r).Trim()
	}

	if currP.Degree() == 1 {
		// Linear root: c1*x + c0 = 0 -> x = -c0/c1
		r := new(big.Rat).Quo(new(big.Rat).Neg(currP[0]), currP[1])
		exactRoots = append(exactRoots, *NewExactRational(r))
	} else if currP.Degree() == 2 {
		// Quadratic formula: a*x^2 + b*x + c = 0
		a := currP[2]
		b := currP[1]
		c := currP[0]

		bSq := new(big.Rat).Mul(b, b)
		fourAC := new(big.Rat).Mul(big.NewRat(4, 1), new(big.Rat).Mul(a, c))
		disc := new(big.Rat).Sub(bSq, fourAC)
		twoA := new(big.Rat).Mul(big.NewRat(2, 1), a)
		negB := new(big.Rat).Neg(b)

		// sqrt(D)
		sqrtDisc, ok := NewExactRational(disc).Sqrt()
		if ok {
			// r1 = (-b - sqrt(D)) / 2a
			r1Num := NewExactRational(negB).Sub(sqrtDisc)
			r2Num := NewExactRational(negB).Add(sqrtDisc)

			twoAExact := NewExactRational(twoA)
			r1, _, _ := r1Num.Div(twoAExact)
			r2, _, _ := r2Num.Div(twoAExact)

			if disc.Sign() == 0 {
				exactRoots = append(exactRoots, *r1, *r1)
			} else {
				exactRoots = append(exactRoots, *r1, *r2)
			}
		} else {
			return "", nil, fmt.Errorf("ErrUnsupported: quadratic discriminant sqrt could not be evaluated exactly")
		}
	} else if currP.Degree() > 2 {
		return "", nil, fmt.Errorf("ErrUnsupported: irreducible polynomial factor of degree %d", currP.Degree())
	}

	sortRoots(exactRoots)

	var parts []string
	for _, r := range exactRoots {
		parts = append(parts, r.String())
	}
	return fmt.Sprintf("[%s]", strings.Join(parts, ", ")), exactRoots, nil
}

// FactorPoly factors a polynomial over Q up to degree 4.
// Supports univariate and multivariate difference of squares/cubes, common factors.
func FactorPoly(expr string, store *RegisterStore) (string, error) {
	_, pMulti, err := ExpandPoly(expr, store)
	if err != nil {
		return "", err
	}

	// 1. Multivariate checks:
	// Check common variable factors
	commonVar := ""
	minExp := 1000
	varNames := make(map[string]bool)
	for _, t := range pMulti.Terms {
		for _, vp := range t.key.Powers {
			varNames[vp.Var] = true
		}
	}

	if len(varNames) > 1 {
		// Check difference of squares: x^2 - y^2 -> (x - y)*(x + y)
		if len(pMulti.Terms) == 2 {
			var terms []*monoTerm
			for _, t := range pMulti.Terms {
				terms = append(terms, t)
			}
			sort.Slice(terms, func(i, j int) bool {
				return terms[i].key.CmpDegLex(terms[j].key) > 0
			})
			t1, t2 := terms[0], terms[1]

			// One coeff positive 1, one coeff -1
			one := NewExactInt(1)
			negOne := NewExactInt(-1)
			if (t1.coeff.Equal(one) && t2.coeff.Equal(negOne)) ||
				(t2.coeff.Equal(one) && t1.coeff.Equal(negOne)) {
				posT := t1
				negT := t2
				if posT.coeff.Equal(negOne) {
					posT, negT = negT, posT
				}
				// Difference of squares: a^2 - b^2
				if len(posT.key.Powers) == 1 && len(negT.key.Powers) == 1 {
					v1 := posT.key.Powers[0]
					v2 := negT.key.Powers[0]
					if v1.Exp == 2 && v2.Exp == 2 {
						return fmt.Sprintf("(%s - %s)*(%s + %s)", v1.Var, v2.Var, v1.Var, v2.Var), nil
					}
					if v1.Exp == 3 && v2.Exp == 3 {
						return fmt.Sprintf("(%s - %s)*(%s^2 + %s*%s + %s^2)", v1.Var, v2.Var, v1.Var, v1.Var, v2.Var, v2.Var), nil
					}
				}
			}
			// Sum of cubes: a^3 + b^3 -> (a + b)*(a^2 - a*b + b^2)
			if t1.coeff.Equal(one) && t2.coeff.Equal(one) {
				if len(t1.key.Powers) == 1 && len(t2.key.Powers) == 1 {
					v1 := t1.key.Powers[0]
					v2 := t2.key.Powers[0]
					if v1.Var > v2.Var {
						v1, v2 = v2, v1
					}
					if v1.Exp == 3 && v2.Exp == 3 {
						return fmt.Sprintf("(%s + %s)*(%s^2 - %s*%s + %s^2)", v1.Var, v2.Var, v1.Var, v1.Var, v2.Var, v2.Var), nil
					}
				}
			}
		}

		// Multivariate common variable factor: e.g. 2*x*y + 4*x -> 2*x*(y + 2)
		var sortedVarNames []string
		for v := range varNames {
			sortedVarNames = append(sortedVarNames, v)
		}
		sort.Strings(sortedVarNames)
		for _, v := range sortedVarNames {
			allHave := true
			minE := 1000
			for _, t := range pMulti.Terms {
				e := 0
				for _, vp := range t.key.Powers {
					if vp.Var == v {
						e = vp.Exp
					}
				}
				if e == 0 {
					allHave = false
					break
				}
				if e < minE {
					minE = e
				}
			}
			if allHave && minE > 0 {
				commonVar = v
				minExp = minE
				break
			}
		}

		if commonVar != "" {
			// Factor out commonVar^minExp
			// Divide each term
			quotient := NewMultiPoly()
			for _, t := range pMulti.Terms {
				newPowers := make(map[string]int)
				for _, vp := range t.key.Powers {
					if vp.Var == commonVar {
						if vp.Exp > minExp {
							newPowers[vp.Var] = vp.Exp - minExp
						}
					} else {
						newPowers[vp.Var] = vp.Exp
					}
				}
				mk := NewMonoKey(newPowers)
				quotient.Terms[mk.CanonicalString()] = &monoTerm{key: mk, coeff: t.coeff.Clone()}
			}
			varFactor := commonVar
			if minExp > 1 {
				varFactor = fmt.Sprintf("%s^%d", commonVar, minExp)
			}
			return fmt.Sprintf("%s*(%s)", varFactor, quotient.String()), nil
		}

		return "", fmt.Errorf("ErrUnsupported: multivariate polynomial factoring not supported for this form")
	}

	// 2. Univariate factoring
	pRat, varName, err := MultiPolyToRatPoly(pMulti)
	if err != nil {
		// Non-rational coefficients: support quadratics via formula
		if pMulti.Degree() == 2 {
			_, roots, rErr := RootsPoly(expr, store)
			if rErr == nil && len(roots) == 2 {
				var a *ExactNum
				for _, t := range pMulti.Terms {
					if t.key.TotalDegree() == 2 {
						a = t.coeff
					}
				}
				// Detect variable name
				vName := "x"
				for _, t := range pMulti.Terms {
					for _, vp := range t.key.Powers {
						if vp.Var != "" {
							vName = vp.Var
							break
						}
					}
				}
				r1Str := roots[0].String()
				r2Str := roots[1].String()
				f1 := fmt.Sprintf("(%s - %s)", vName, r1Str)
				if strings.HasPrefix(r1Str, "-") {
					f1 = fmt.Sprintf("(%s + %s)", vName, strings.TrimPrefix(r1Str, "-"))
				}
				f2 := fmt.Sprintf("(%s - %s)", vName, r2Str)
				if strings.HasPrefix(r2Str, "-") {
					f2 = fmt.Sprintf("(%s + %s)", vName, strings.TrimPrefix(r2Str, "-"))
				}
				if a != nil && !a.Equal(NewExactInt(1)) {
					return fmt.Sprintf("%s*%s*%s", a.String(), f1, f2), nil
				}
				return fmt.Sprintf("%s*%s", f1, f2), nil
			}
		}
		return "", fmt.Errorf("ErrUnsupported: factoring polynomial with non-rational coefficients is not supported")
	}

	deg := pRat.Degree()
	if deg > 4 {
		return "", fmt.Errorf("ErrUnsupported: factoring degree exceeds limit 4 (got %d)", deg)
	}
	if deg <= 1 {
		return pMulti.String(), nil
	}

	// Content (leading coefficient or gcd)
	lead := pRat.LeadingCoeff()
	currLead := new(big.Rat).Set(lead)
	var factors []string
	if lead.Cmp(big.NewRat(1, 1)) != 0 && lead.Sign() != 0 {
		invLead := new(big.Rat).Inv(lead)
		for i := range pRat {
			pRat[i] = new(big.Rat).Mul(pRat[i], invLead)
		}
	}

	currP := pRat.Trim()

	// Factor out x = 0
	xCount := 0
	for currP.Degree() > 0 && currP[0].Sign() == 0 {
		xCount++
		nextP := make(RatPoly, currP.Degree())
		for i := 1; i <= currP.Degree(); i++ {
			nextP[i-1] = currP[i]
		}
		currP = nextP.Trim()
	}
	if xCount == 1 {
		factors = append(factors, varName)
	} else if xCount > 1 {
		factors = append(factors, fmt.Sprintf("%s^%d", varName, xCount))
	}

	appendLinearFactor := func(r *big.Rat) {
		den := r.Denom()
		num := r.Num()
		if currLead.IsInt() && new(big.Int).Rem(currLead.Num(), den).Sign() == 0 {
			currLead.Quo(currLead, new(big.Rat).SetInt(den))
			if num.Sign() < 0 {
				posNum := new(big.Int).Neg(num)
				if den.Cmp(big.NewInt(1)) == 0 {
					factors = append(factors, fmt.Sprintf("(%s + %s)", varName, posNum.String()))
				} else {
					factors = append(factors, fmt.Sprintf("(%s*%s + %s)", den.String(), varName, posNum.String()))
				}
			} else {
				if den.Cmp(big.NewInt(1)) == 0 {
					factors = append(factors, fmt.Sprintf("(%s - %s)", varName, num.String()))
				} else {
					factors = append(factors, fmt.Sprintf("(%s*%s - %s)", den.String(), varName, num.String()))
				}
			}
		} else {
			negR := new(big.Rat).Neg(r)
			rStr := formatRational(new(big.Rat).Abs(r), false, false, false, 0)
			if negR.Sign() > 0 {
				factors = append(factors, fmt.Sprintf("(%s + %s)", varName, rStr))
			} else {
				factors = append(factors, fmt.Sprintf("(%s - %s)", varName, rStr))
			}
		}
	}

	// Find rational roots
	for currP.Degree() > 0 {
		r, err := findSingleRationalRoot(currP)
		if err != nil || r == nil {
			break
		}
		appendLinearFactor(r)
		currP = currP.DivSynthetic(r).Trim()
	}

	if currP.Degree() == 2 {
		// Remaining quadratic: check if reducible over Q
		a := currP[2]
		b := currP[1]
		c := currP[0]
		bSq := new(big.Rat).Mul(b, b)
		fourAC := new(big.Rat).Mul(big.NewRat(4, 1), new(big.Rat).Mul(a, c))
		disc := new(big.Rat).Sub(bSq, fourAC)

		numSq, numExact := integerSquareRoot(disc.Num())
		denSq, denExact := integerSquareRoot(disc.Denom())
		if numExact && denExact {
			// Reducible
			sqrtD := new(big.Rat).SetFrac(numSq, denSq)
			twoA := new(big.Rat).Mul(big.NewRat(2, 1), a)
			r1 := new(big.Rat).Quo(new(big.Rat).Sub(new(big.Rat).Neg(b), sqrtD), twoA)
			r2 := new(big.Rat).Quo(new(big.Rat).Add(new(big.Rat).Neg(b), sqrtD), twoA)

			appendLinearFactor(r1)
			appendLinearFactor(r2)
		} else {
			// Irreducible quadratic
			factors = append(factors, fmt.Sprintf("(%s)", RatPolyToMultiPoly(currP, varName).String()))
		}
	} else if currP.Degree() > 2 {
		return "", fmt.Errorf("ErrUnsupported: irreducible polynomial factor of degree %d", currP.Degree())
	}

	if currLead.Cmp(big.NewRat(1, 1)) != 0 {
		if currLead.Cmp(big.NewRat(-1, 1)) == 0 {
			if len(factors) > 0 {
				factors[0] = "-" + factors[0]
			} else {
				factors = append(factors, "-1")
			}
		} else {
			factors = append([]string{formatRational(currLead, false, false, false, 0)}, factors...)
		}
	}

	return strings.Join(factors, "*"), nil
}

// EvaluatePoly evaluates polynomial p at x = val.
func EvaluatePoly(pStr, varName, valStr string, store *RegisterStore) (string, error) {
	_, pMulti, err := ExpandPoly(pStr, store)
	if err != nil {
		return "", err
	}

	// Parse val as ExactNum or rational
	valEx, err := EvaluateInfixToExactNum(valStr, store)
	if err != nil {
		return "", err
	}

	// Direct evaluation of MultiPoly at varName = valEx
	res := NewExactZero()
	for _, t := range pMulti.Terms {
		termVal := t.coeff.Clone()
		for _, vp := range t.key.Powers {
			if vp.Var == varName {
				powVal, ok, err := valEx.Pow(int64(vp.Exp))
				if err != nil {
					return "", err
				}
				if !ok {
					return "", fmt.Errorf("ErrOverflow: polynomial evaluation exponent overflowed")
				}
				var mulOk bool
				termVal, mulOk = termVal.Mul(powVal)
				if !mulOk {
					return "", fmt.Errorf("ErrOverflow: polynomial evaluation overflowed")
				}
			}
		}
		res = res.Add(termVal)
	}

	return res.String(), nil
}

func calcValToPoly(v CalcVal) *MultiPoly {
	switch v.Kind {
	case ValKindPoly:
		return v.Poly
	case ValKindRat:
		if v.Rat != nil {
			return NewMultiPolyConstRat(v.Rat)
		}
	case ValKindExactNum:
		if v.Exact != nil {
			return NewMultiPolyConst(v.Exact)
		}
	case ValKindSurd:
		if v.Surd != nil {
			return NewMultiPolyConst(NewExactSurd(v.Surd))
		}
	}
	return nil
}

func isKnownMathFunction(name string) bool {
	switch strings.ToLower(name) {
	case "log", "ln", "log10", "log2", "exp",
		"sin", "cos", "tan", "sec", "csc", "cot",
		"asin", "acos", "atan", "arcsin", "arccos", "arctan",
		"sinh", "cosh", "tanh",
		"abs", "cbrt", "ceil", "floor", "round", "sign":
		return true
	default:
		return false
	}
}
