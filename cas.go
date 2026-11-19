package main

import (
	"fmt"
	"math"
	"math/big"
	"sort"
	"strings"
	"unicode"
)

type SolveResult struct {
	VarName string
	Val     CalcVal
}

// RatPoly represents a polynomial with exact rational coefficients:
// P(x) = c[0] + c[1]*x + c[2]*x^2 + ... + c[d]*x^d
type RatPoly []*big.Rat

func (p RatPoly) Degree() int {
	for i := len(p) - 1; i >= 0; i-- {
		if p[i] != nil && p[i].Sign() != 0 {
			return i
		}
	}
	return 0
}

func (p RatPoly) LeadingCoeff() *big.Rat {
	deg := p.Degree()
	if deg < len(p) && p[deg] != nil {
		return new(big.Rat).Set(p[deg])
	}
	return big.NewRat(0, 1)
}

func (p RatPoly) Trim() RatPoly {
	deg := p.Degree()
	res := make(RatPoly, deg+1)
	for i := 0; i <= deg; i++ {
		if i < len(p) && p[i] != nil {
			res[i] = new(big.Rat).Set(p[i])
		} else {
			res[i] = big.NewRat(0, 1)
		}
	}
	return res
}

func PolyConstRat(c *big.Rat) RatPoly {
	return RatPoly{new(big.Rat).Set(c)}
}

func PolyVarRat() RatPoly {
	return RatPoly{big.NewRat(0, 1), big.NewRat(1, 1)}
}

func PolyAddRat(p1, p2 RatPoly) RatPoly {
	maxL := len(p1)
	if len(p2) > maxL {
		maxL = len(p2)
	}
	res := make(RatPoly, maxL)
	for i := 0; i < maxL; i++ {
		r := big.NewRat(0, 1)
		if i < len(p1) && p1[i] != nil {
			r.Add(r, p1[i])
		}
		if i < len(p2) && p2[i] != nil {
			r.Add(r, p2[i])
		}
		res[i] = r
	}
	return res.Trim()
}

func PolySubRat(p1, p2 RatPoly) RatPoly {
	maxL := len(p1)
	if len(p2) > maxL {
		maxL = len(p2)
	}
	res := make(RatPoly, maxL)
	for i := 0; i < maxL; i++ {
		r := big.NewRat(0, 1)
		if i < len(p1) && p1[i] != nil {
			r.Add(r, p1[i])
		}
		if i < len(p2) && p2[i] != nil {
			r.Sub(r, p2[i])
		}
		res[i] = r
	}
	return res.Trim()
}

func PolyMulRat(p1, p2 RatPoly) RatPoly {
	p1 = p1.Trim()
	p2 = p2.Trim()
	deg1 := p1.Degree()
	deg2 := p2.Degree()
	if (deg1 == 0 && p1[0].Sign() == 0) || (deg2 == 0 && p2[0].Sign() == 0) {
		return RatPoly{big.NewRat(0, 1)}
	}
	res := make(RatPoly, deg1+deg2+1)
	for i := 0; i < len(res); i++ {
		res[i] = big.NewRat(0, 1)
	}
	for i := 0; i <= deg1; i++ {
		if p1[i].Sign() == 0 {
			continue
		}
		for j := 0; j <= deg2; j++ {
			if p2[j].Sign() == 0 {
				continue
			}
			term := new(big.Rat).Mul(p1[i], p2[j])
			res[i+j].Add(res[i+j], term)
		}
	}
	return res.Trim()
}

func PolyDivConstRat(p RatPoly, c *big.Rat) (RatPoly, error) {
	if c.Sign() == 0 {
		return nil, fmt.Errorf("ErrDivZero: division by zero in polynomial")
	}
	res := make(RatPoly, len(p))
	for i := 0; i < len(p); i++ {
		res[i] = new(big.Rat).Quo(p[i], c)
	}
	return res.Trim(), nil
}

func PolyPowRat(p RatPoly, k int) (RatPoly, error) {
	if k < 0 {
		return nil, fmt.Errorf("ErrUnsupported: negative polynomial power not supported")
	}
	if k == 0 {
		return PolyConstRat(big.NewRat(1, 1)), nil
	}
	res := PolyConstRat(big.NewRat(1, 1))
	base := p
	for k > 0 {
		if k%2 == 1 {
			res = PolyMulRat(res, base)
		}
		base = PolyMulRat(base, base)
		k /= 2
	}
	return res.Trim(), nil
}

// Eval evaluates P(x) at rational x.
func (p RatPoly) Eval(x *big.Rat) *big.Rat {
	res := big.NewRat(0, 1)
	xPow := big.NewRat(1, 1)
	for i := 0; i < len(p); i++ {
		if p[i] != nil && p[i].Sign() != 0 {
			term := new(big.Rat).Mul(p[i], xPow)
			res.Add(res, term)
		}
		xPow = new(big.Rat).Mul(xPow, x)
	}
	return res
}

// DivSynthetic divides p by (x - root) returning quotient.
func (p RatPoly) DivSynthetic(root *big.Rat) RatPoly {
	p = p.Trim()
	deg := p.Degree()
	if deg <= 0 {
		return RatPoly{big.NewRat(0, 1)}
	}
	q := make(RatPoly, deg)
	current := new(big.Rat).Set(p[deg])
	q[deg-1] = current
	for i := deg - 1; i >= 1; i-- {
		current = new(big.Rat).Add(p[i], new(big.Rat).Mul(current, root))
		q[i-1] = current
	}
	return q.Trim()
}

// -----------------------------------------------------------------------------
// Polynomial Parser for Solve
// -----------------------------------------------------------------------------

type CASParser struct {
	tokens  []Token
	pos     int
	varName string
	store   *RegisterStore
}

func NewCASParser(tokens []Token, varName string, store *RegisterStore) *CASParser {
	return &CASParser{
		tokens:  tokens,
		pos:     0,
		varName: strings.ToLower(strings.TrimSpace(varName)),
		store:   store,
	}
}

func (cp *CASParser) cur() Token {
	if cp.pos < len(cp.tokens) {
		return cp.tokens[cp.pos]
	}
	return Token{Type: TokEOF}
}

func (cp *CASParser) next() Token {
	t := cp.cur()
	if cp.pos < len(cp.tokens) {
		cp.pos++
	}
	return t
}

func (cp *CASParser) parseAdditive() (RatPoly, error) {
	left, err := cp.parseMultiplicative()
	if err != nil {
		return nil, err
	}
	for cp.cur().Type == TokPlus || cp.cur().Type == TokMinus {
		op := cp.next()
		right, err := cp.parseMultiplicative()
		if err != nil {
			return nil, err
		}
		if op.Type == TokPlus {
			left = PolyAddRat(left, right)
		} else {
			left = PolySubRat(left, right)
		}
	}
	return left, nil
}

func (cp *CASParser) parseMultiplicative() (RatPoly, error) {
	left, err := cp.parseUnary()
	if err != nil {
		return nil, err
	}
	for cp.cur().Type == TokStar || cp.cur().Type == TokSlash {
		op := cp.next()
		right, err := cp.parseUnary()
		if err != nil {
			return nil, err
		}
		if op.Type == TokStar {
			left = PolyMulRat(left, right)
		} else {
			// Division: right must be a constant
			if right.Degree() != 0 || right[0].Sign() == 0 {
				return nil, fmt.Errorf("ErrUnsupported: division by non-constant polynomial in solve")
			}
			var err error
			left, err = PolyDivConstRat(left, right[0])
			if err != nil {
				return nil, err
			}
		}
	}
	if cp.cur().Type == TokLParen || cp.cur().Type == TokIdent || cp.cur().Type == TokNum || cp.cur().Type == TokRegister {
		return nil, fmt.Errorf("ErrSyntax: implicit multiplication not supported; use explicit '*'")
	}
	return left, nil
}

func (cp *CASParser) parseUnary() (RatPoly, error) {
	if cp.cur().Type == TokMinus {
		cp.next()
		// Unary minus binds looser than exponentiation: -x^2 = -(x^2) 
		val, err := cp.parsePower()
		if err != nil {
			return nil, err
		}
		zero := PolyConstRat(big.NewRat(0, 1))
		return PolySubRat(zero, val), nil
	}
	if cp.cur().Type == TokPlus {
		cp.next()
		return cp.parsePower()
	}
	return cp.parsePower()
}

func (cp *CASParser) parsePower() (RatPoly, error) {
	left, err := cp.parsePrimary()
	if err != nil {
		return nil, err
	}
	if cp.cur().Type == TokPow {
		cp.next()
		// Exponent must be a non-negative integer constant
		expTok := cp.cur()
		if expTok.Type != TokNum || !expTok.NumRat.IsInt() {
			return nil, fmt.Errorf("ErrSyntax: polynomial exponent must be a non-negative integer")
		}
		cp.next()
		k := int(expTok.NumRat.Num().Int64())
		if k < 0 {
			return nil, fmt.Errorf("ErrUnsupported: negative polynomial exponent")
		}
		if k > 100 {
			return nil, fmt.Errorf("ErrOverflow: polynomial exponent exceeds maximum 100")
		}
		return PolyPowRat(left, k)
	}
	return left, nil
}

func (cp *CASParser) parsePrimary() (RatPoly, error) {
	tok := cp.cur()

	// Number literal
	if tok.Type == TokNum {
		cp.next()
		return PolyConstRat(tok.NumRat), nil
	}

	// Register token: <|r0|> .. <|r7|> 
	if tok.Type == TokRegister {
		cp.next()
		if cp.store == nil {
			return nil, fmt.Errorf("ErrUnsetRegister: <|r%d|> is empty", tok.RegIdx)
		}
		val, err := cp.store.Get(tok.RegIdx)
		if err != nil {
			return nil, err
		}
		if val.Kind == ValKindRat {
			return PolyConstRat(val.Rat), nil
		}
		return nil, fmt.Errorf("ErrTypeMismatch: register <|r%d|> does not contain rational number", tok.RegIdx)
	}

	// Variable identifier
	if tok.Type == TokIdent {
		cp.next()
		if cp.cur().Type == TokLParen {
			return nil, fmt.Errorf("ErrUnsupported: functions like %s(...) are not supported in solve", tok.Text)
		}
		id := strings.ToLower(tok.Text)
		if id == cp.varName {
			return PolyVarRat(), nil
		}
		// Constants
		if id == "pi" {
			f := math.Pi
			return PolyConstRat(new(big.Rat).SetFloat64(f)), nil
		}
		if id == "e" {
			f := math.E
			return PolyConstRat(new(big.Rat).SetFloat64(f)), nil
		}
		return nil, fmt.Errorf("ErrSyntax: unexpected variable %q (expected %s)", tok.Text, cp.varName)
	}

	// Parenthesized expression
	if tok.Type == TokLParen {
		cp.next()
		val, err := cp.parseAdditive()
		if err != nil {
			return nil, err
		}
		if cp.cur().Type != TokRParen {
			return nil, fmt.Errorf("ErrSyntax: expected ')' at column %d", cp.cur().Col)
		}
		cp.next()
		return val, nil
	}

	return nil, fmt.Errorf("ErrSyntax: unexpected token %q at column %d", tok.Text, tok.Col)
}

// -----------------------------------------------------------------------------
// Polynomial Solving Algorithms
// -----------------------------------------------------------------------------

func solvePolynomial(p RatPoly, varName string, isComplex bool) ([]SolveResult, error) {
	p = p.Trim()
	deg := p.Degree()

	if deg == 0 {
		if p[0].Sign() == 0 {
			return nil, fmt.Errorf("ErrInfiniteSolutions: all real numbers")
		}
		return nil, fmt.Errorf("ErrNoSolution: equation has no solution")
	}

	// Degree 1: c1*x + c0 = 0 -> x = -c0 / c1
	if deg == 1 {
		c0 := p[0]
		c1 := p[1]
		negC0 := new(big.Rat).Neg(c0)
		root := new(big.Rat).Quo(negC0, c1)
		return []SolveResult{
			{VarName: varName, Val: CalcVal{Kind: ValKindRat, Rat: root}},
		}, nil
	}

	// Degree 2: a*x^2 + b*x + c = 0
	if deg == 2 {
		c := p[0]
		b := p[1]
		a := p[2]

		// D = b^2 - 4ac
		bSq := new(big.Rat).Mul(b, b)
		fourAC := new(big.Rat).Mul(big.NewRat(4, 1), new(big.Rat).Mul(a, c))
		disc := new(big.Rat).Sub(bSq, fourAC)
		twoA := new(big.Rat).Mul(big.NewRat(2, 1), a)
		negB := new(big.Rat).Neg(b)

		if disc.Sign() < 0 {
			if !isComplex {
				// No real roots (binds nothing)
				return []SolveResult{}, nil
			}
			sqrtD, ok := NewExactRational(disc).Sqrt()
			if !ok {
				return nil, fmt.Errorf("ErrUnsupported: discriminant sqrt could not be evaluated exactly")
			}
			twoAEx := NewExactRational(twoA)
			r1Num := NewExactRational(negB).Sub(sqrtD)
			r2Num := NewExactRational(negB).Add(sqrtD)
			r1, _, _ := r1Num.Div(twoAEx)
			r2, _, _ := r2Num.Div(twoAEx)
			return []SolveResult{
				{VarName: varName, Val: CalcVal{Kind: ValKindExactNum, Exact: r1}},
				{VarName: varName, Val: CalcVal{Kind: ValKindExactNum, Exact: r2}},
			}, nil
		}

		if disc.Sign() == 0 {
			// Double root listed ONCE 
			root := new(big.Rat).Quo(negB, twoA)
			return []SolveResult{
				{VarName: varName, Val: CalcVal{Kind: ValKindRat, Rat: root}},
			}, nil
		}

		// Exact quadratic roots (rational or p + q*sqrt(r))
		sqrtD, ok := NewExactRational(disc).Sqrt()
		if !ok {
			return nil, fmt.Errorf("ErrUnsupported: discriminant sqrt could not be evaluated exactly")
		}

		twoAEx := NewExactRational(twoA)
		r1Num := NewExactRational(negB).Sub(sqrtD)
		r2Num := NewExactRational(negB).Add(sqrtD)
		r1, _, _ := r1Num.Div(twoAEx)
		r2, _, _ := r2Num.Div(twoAEx)

		val1 := CalcVal{Kind: ValKindExactNum, Exact: r1}
		if rat, isR := r1.AsRational(); isR {
			val1 = CalcVal{Kind: ValKindRat, Rat: rat}
		}
		val2 := CalcVal{Kind: ValKindExactNum, Exact: r2}
		if rat, isR := r2.AsRational(); isR {
			val2 = CalcVal{Kind: ValKindRat, Rat: rat}
		}

		if r1.ToFloat64() > r2.ToFloat64() {
			val1, val2 = val2, val1
		}

		return []SolveResult{
			{VarName: varName, Val: val1},
			{VarName: varName, Val: val2},
		}, nil
	}

	// Degree >= 3: Rational Root Theorem 
	if deg > 100 {
		return nil, fmt.Errorf("ErrOverflow: polynomial degree exceeds maximum degree 100")
	}

	var allRoots []*big.Rat
	currP := p

	// Factor out x = 0 roots
	for currP.Degree() > 0 && currP[0].Sign() == 0 {
		allRoots = append(allRoots, big.NewRat(0, 1))
		// Shift coefficients down
		nextP := make(RatPoly, currP.Degree())
		for i := 1; i <= currP.Degree(); i++ {
			nextP[i-1] = currP[i]
		}
		currP = nextP.Trim()
	}

	// Find rational roots for currP
	for currP.Degree() >= 3 {
		root, err := findSingleRationalRoot(currP)
		if err != nil {
			return nil, err
		}
		if root == nil {
			// Cannot reduce cubic+ factor with rational roots
			return nil, fmt.Errorf("ErrUnsupported: cubic factor not solvable exactly")
		}
		allRoots = append(allRoots, root)
		currP = currP.DivSynthetic(root)
	}

	// Now currP has degree <= 2
	if currP.Degree() == 1 {
		c0 := currP[0]
		c1 := currP[1]
		negC0 := new(big.Rat).Neg(c0)
		root := new(big.Rat).Quo(negC0, c1)
		allRoots = append(allRoots, root)
	} else if currP.Degree() == 2 {
		qSolns, err := solvePolynomial(currP, varName, isComplex)
		if err != nil {
			return nil, err
		}
		for _, qs := range qSolns {
			if qs.Val.Kind == ValKindRat {
				allRoots = append(allRoots, qs.Val.Rat)
			} else {
				// Exact/surd root from quadratic factor
				return combineRoots(allRoots, qSolns, varName), nil
			}
		}
	}

	// Deduplicate rational roots
	unique := deduplicateRationals(allRoots)
	sort.Slice(unique, func(i, j int) bool {
		return unique[i].Cmp(unique[j]) < 0
	})

	var results []SolveResult
	for _, r := range unique {
		results = append(results, SolveResult{
			VarName: varName,
			Val:     CalcVal{Kind: ValKindRat, Rat: r},
		})
	}
	return results, nil
}

func combineRoots(ratRoots []*big.Rat, nonRatSolns []SolveResult, varName string) []SolveResult {
	uniqueRats := deduplicateRationals(ratRoots)
	sort.Slice(uniqueRats, func(i, j int) bool {
		return uniqueRats[i].Cmp(uniqueRats[j]) < 0
	})
	var res []SolveResult
	for _, r := range uniqueRats {
		res = append(res, SolveResult{VarName: varName, Val: CalcVal{Kind: ValKindRat, Rat: r}})
	}
	res = append(res, nonRatSolns...)
	return res
}

func deduplicateRationals(rats []*big.Rat) []*big.Rat {
	var unique []*big.Rat
	for _, r := range rats {
		exists := false
		for _, u := range unique {
			if r.Cmp(u) == 0 {
				exists = true
				break
			}
		}
		if !exists {
			unique = append(unique, r)
		}
	}
	return unique
}

func findSingleRationalRoot(p RatPoly) (*big.Rat, error) {
	// Scale polynomial to integers
	deg := p.Degree()
	if deg < 1 {
		return nil, nil
	}

	// Common denominator of all coefficients
	commonDen := big.NewInt(1)
	for i := 0; i <= deg; i++ {
		if p[i] != nil && p[i].Sign() != 0 {
			g := new(big.Int).GCD(nil, nil, commonDen, p[i].Denom())
			prod := new(big.Int).Mul(commonDen, p[i].Denom())
			commonDen.Div(prod, g)
		}
	}

	intCoeffs := make([]*big.Int, deg+1)
	for i := 0; i <= deg; i++ {
		if p[i] != nil {
			factor := new(big.Int).Quo(commonDen, p[i].Denom())
			intCoeffs[i] = new(big.Int).Mul(p[i].Num(), factor)
		} else {
			intCoeffs[i] = big.NewInt(0)
		}
	}

	a0 := new(big.Int).Abs(intCoeffs[0])
	ad := new(big.Int).Abs(intCoeffs[deg])

	if a0.Sign() == 0 {
		return big.NewRat(0, 1), nil
	}

	// Divisors of a0 and ad with deterministic budget
	pDivs, err := getDivisorsWithBudget(a0, 50000)
	if err != nil {
		return nil, err
	}
	qDivs, err := getDivisorsWithBudget(ad, 50000)
	if err != nil {
		return nil, err
	}

	if len(pDivs)*len(qDivs) > 100000 {
		return nil, fmt.Errorf("ErrOverflow: candidate root space exceeds limit")
	}

	evalSteps := 0
	for _, pVal := range pDivs {
		for _, qVal := range qDivs {
			evalSteps++
			if evalSteps > 20000 {
				return nil, fmt.Errorf("ErrTimeout: step budget exceeded")
			}
			candPos := big.NewRat(pVal, qVal)
			if p.Eval(candPos).Sign() == 0 {
				return candPos, nil
			}
			candNeg := big.NewRat(-pVal, qVal)
			if p.Eval(candNeg).Sign() == 0 {
				return candNeg, nil
			}
		}
	}

	return nil, nil
}

func getDivisorsWithBudget(n *big.Int, maxSteps int) ([]int64, error) {
	if !n.IsInt64() || n.Int64() <= 0 {
		if n.Sign() <= 0 {
			return []int64{1}, nil
		}
		return nil, fmt.Errorf("ErrOverflow: polynomial coefficient exceeds 64-bit integer limit")
	}
	val := n.Int64()
	if val > 1000000000000 { // 10^12
		return nil, fmt.Errorf("ErrOverflow: polynomial coefficient exceeds rational root limit")
	}
	var divs []int64
	steps := 0
	for d := int64(1); d*d <= val; d++ {
		steps++
		if steps > maxSteps {
			return nil, fmt.Errorf("ErrTimeout: step budget exceeded")
		}
		if val%d == 0 {
			divs = append(divs, d)
			if d*d != val {
				divs = append(divs, val/d)
			}
		}
		if len(divs) > 500 {
			break
		}
	}
	return divs, nil
}

func getDivisors(n *big.Int) []int64 {
	divs, _ := getDivisorsWithBudget(n, 50000)
	return divs
}

func integerSquareRoot(n *big.Int) (*big.Int, bool) {
	if n.Sign() < 0 {
		return nil, false
	}
	if n.Sign() == 0 {
		return big.NewInt(0), true
	}
	sq := new(big.Int).Sqrt(n)
	sq2 := new(big.Int).Mul(sq, sq)
	if sq2.Cmp(n) == 0 {
		return sq, true
	}
	return nil, false
}

// -----------------------------------------------------------------------------
// Linear Systems Engine (N x N Gaussian Elimination on big.Rat)
// -----------------------------------------------------------------------------

func solveLinearSystem(equations []string, varList []string, store *RegisterStore) ([]SolveResult, error) {
	numEqs := len(equations)
	numVars := len(varList)

	if numEqs != numVars {
		if numEqs < numVars {
			return nil, fmt.Errorf("ErrInfiniteSolutions: system has infinitely many solutions")
		}
		return nil, fmt.Errorf("ErrNoSolution: system requires %d equations, got %d", numVars, numEqs)
	}

	// Parse each equation: LHS = RHS -> sum(a_j * v_j) = b
	A := make([][]*big.Rat, numEqs)
	B := make([]*big.Rat, numEqs)

	for i := 0; i < numEqs; i++ {
		eq := equations[i]
		eqParts := strings.Split(eq, "=")
		if len(eqParts) != 2 {
			return nil, fmt.Errorf("ErrSyntax: equation %d missing '=': %s", i+1, eq)
		}

		lhsTokens, err := tokenizeExpr(eqParts[0])
		if err != nil {
			if strings.HasPrefix(err.Error(), "Err") {
				return nil, err
			}
			return nil, fmt.Errorf("ErrSyntax: eq%d error: %v", i+1, err)
		}
		rhsTokens, err := tokenizeExpr(eqParts[1])
		if err != nil {
			if strings.HasPrefix(err.Error(), "Err") {
				return nil, err
			}
			return nil, fmt.Errorf("ErrSyntax: eq%d error: %v", i+1, err)
		}

		rowCoeffs, constTerm, err := parseLinearEquation(lhsTokens, rhsTokens, varList, store)
		if err != nil {
			if strings.HasPrefix(err.Error(), "Err") {
				return nil, err
			}
			return nil, fmt.Errorf("ErrSyntax: eq%d error: %v", i+1, err)
		}

		A[i] = rowCoeffs
		B[i] = constTerm
	}

	// Gaussian elimination with partial pivoting on big.Rat
	n := numVars
	for col := 0; col < n; col++ {
		// Pivot
		pivotRow := -1
		for row := col; row < n; row++ {
			if A[row][col].Sign() != 0 {
				pivotRow = row
				break
			}
		}
		if pivotRow == -1 {
			// Singular matrix: check for consistency
			// If all coefficients in row are 0 and B != 0 -> ErrNoSolution
			// Else -> ErrInfiniteSolutions
			for r := col; r < n; r++ {
				allZero := true
				for c := 0; c < n; c++ {
					if A[r][c].Sign() != 0 {
						allZero = false
						break
					}
				}
				if allZero && B[r].Sign() != 0 {
					return nil, fmt.Errorf("ErrNoSolution: singular system: no solution")
				}
			}
			return nil, fmt.Errorf("ErrInfiniteSolutions: singular system: infinitely many solutions")
		}

		// Swap rows
		A[col], A[pivotRow] = A[pivotRow], A[col]
		B[col], B[pivotRow] = B[pivotRow], B[col]

		// Eliminate below
		pivotVal := A[col][col]
		for row := col + 1; row < n; row++ {
			if A[row][col].Sign() != 0 {
				factor := new(big.Rat).Quo(A[row][col], pivotVal)
				for c := col; c < n; c++ {
					term := new(big.Rat).Mul(factor, A[col][c])
					A[row][c].Sub(A[row][c], term)
				}
				bTerm := new(big.Rat).Mul(factor, B[col])
				B[row].Sub(B[row], bTerm)
			}
		}
	}

	// Back-substitution
	sol := make([]*big.Rat, n)
	for row := n - 1; row >= 0; row-- {
		sum := new(big.Rat).Set(B[row])
		for c := row + 1; c < n; c++ {
			term := new(big.Rat).Mul(A[row][c], sol[c])
			sum.Sub(sum, term)
		}
		if A[row][row].Sign() == 0 {
			if sum.Sign() != 0 {
				return nil, fmt.Errorf("ErrNoSolution: system has no solution")
			}
			return nil, fmt.Errorf("ErrInfiniteSolutions: system has infinitely many solutions")
		}
		sol[row] = new(big.Rat).Quo(sum, A[row][row])
	}

	var results []SolveResult
	for i := 0; i < n; i++ {
		results = append(results, SolveResult{
			VarName: varList[i],
			Val:     CalcVal{Kind: ValKindRat, Rat: sol[i]},
		})
	}
	return results, nil
}

func parseLinearEquation(lhsTokens, rhsTokens []Token, varList []string, store *RegisterStore) ([]*big.Rat, *big.Rat, error) {
	// Parse LHS and RHS into linear combination of varList + constant
	lhsCoeffs, lhsConst, err := parseLinearExpr(lhsTokens, varList, store)
	if err != nil {
		return nil, nil, err
	}
	rhsCoeffs, rhsConst, err := parseLinearExpr(rhsTokens, varList, store)
	if err != nil {
		return nil, nil, err
	}

	// sum(lhs_c - rhs_c) * v = rhsConst - lhsConst
	n := len(varList)
	row := make([]*big.Rat, n)
	for i := 0; i < n; i++ {
		row[i] = new(big.Rat).Sub(lhsCoeffs[i], rhsCoeffs[i])
	}
	b := new(big.Rat).Sub(rhsConst, lhsConst)
	return row, b, nil
}

type LinearLinearTerm struct {
	Coeffs []*big.Rat // for each var in varList
	Const  *big.Rat
}

func parseLinearExpr(tokens []Token, varList []string, store *RegisterStore) ([]*big.Rat, *big.Rat, error) {
	// Simple linear expression evaluator supporting: (+, -, *, /)
	// We can use a recursive descent parser for linear expressions:
	lp := &linearExprParser{
		tokens:  tokens,
		pos:     0,
		varList: varList,
		store:   store,
	}
	term, err := lp.parseAdditive()
	if err != nil {
		return nil, nil, err
	}
	if lp.pos < len(lp.tokens) && lp.tokens[lp.pos].Type != TokEOF {
		return nil, nil, fmt.Errorf("unexpected token %q", lp.tokens[lp.pos].Text)
	}
	return term.Coeffs, term.Const, nil
}

type linearExprParser struct {
	tokens  []Token
	pos     int
	varList []string
	store   *RegisterStore
}

func (lp *linearExprParser) cur() Token {
	if lp.pos < len(lp.tokens) {
		return lp.tokens[lp.pos]
	}
	return Token{Type: TokEOF}
}

func (lp *linearExprParser) next() Token {
	t := lp.cur()
	if lp.pos < len(lp.tokens) {
		lp.pos++
	}
	return t
}

func (lp *linearExprParser) newZeroTerm() LinearLinearTerm {
	n := len(lp.varList)
	coeffs := make([]*big.Rat, n)
	for i := 0; i < n; i++ {
		coeffs[i] = big.NewRat(0, 1)
	}
	return LinearLinearTerm{
		Coeffs: coeffs,
		Const:  big.NewRat(0, 1),
	}
}

func (lp *linearExprParser) parseAdditive() (LinearLinearTerm, error) {
	left, err := lp.parseMultiplicative()
	if err != nil {
		return LinearLinearTerm{}, err
	}
	for lp.cur().Type == TokPlus || lp.cur().Type == TokMinus {
		op := lp.next()
		right, err := lp.parseMultiplicative()
		if err != nil {
			return LinearLinearTerm{}, err
		}
		if op.Type == TokPlus {
			for i := 0; i < len(left.Coeffs); i++ {
				left.Coeffs[i].Add(left.Coeffs[i], right.Coeffs[i])
			}
			left.Const.Add(left.Const, right.Const)
		} else {
			for i := 0; i < len(left.Coeffs); i++ {
				left.Coeffs[i].Sub(left.Coeffs[i], right.Coeffs[i])
			}
			left.Const.Sub(left.Const, right.Const)
		}
	}
	return left, nil
}

func (lp *linearExprParser) parseMultiplicative() (LinearLinearTerm, error) {
	left, err := lp.parseUnary()
	if err != nil {
		return LinearLinearTerm{}, err
	}
	for lp.cur().Type == TokStar || lp.cur().Type == TokSlash {
		op := lp.next()
		right, err := lp.parseUnary()
		if err != nil {
			return LinearLinearTerm{}, err
		}
		if op.Type == TokStar {
			// At least one must be a pure constant
			leftIsConst := true
			for _, c := range left.Coeffs {
				if c.Sign() != 0 {
					leftIsConst = false
					break
				}
			}
			rightIsConst := true
			for _, c := range right.Coeffs {
				if c.Sign() != 0 {
					rightIsConst = false
					break
				}
			}
			if !leftIsConst && !rightIsConst {
				return LinearLinearTerm{}, fmt.Errorf("non-linear term in linear system")
			}
			if leftIsConst {
				k := left.Const
				for i := 0; i < len(right.Coeffs); i++ {
					right.Coeffs[i].Mul(right.Coeffs[i], k)
				}
				right.Const.Mul(right.Const, k)
				left = right
			} else {
				k := right.Const
				for i := 0; i < len(left.Coeffs); i++ {
					left.Coeffs[i].Mul(left.Coeffs[i], k)
				}
				left.Const.Mul(left.Const, k)
			}
		} else {
			// Division: right must be a non-zero constant
			for _, c := range right.Coeffs {
				if c.Sign() != 0 {
					return LinearLinearTerm{}, fmt.Errorf("division by variable in linear system")
				}
			}
			if right.Const.Sign() == 0 {
				return LinearLinearTerm{}, fmt.Errorf("division by zero in linear system")
			}
			k := right.Const
			for i := 0; i < len(left.Coeffs); i++ {
				left.Coeffs[i].Quo(left.Coeffs[i], k)
			}
			left.Const.Quo(left.Const, k)
		}
	}
	if lp.cur().Type == TokLParen || lp.cur().Type == TokIdent || lp.cur().Type == TokNum || lp.cur().Type == TokRegister {
		return LinearLinearTerm{}, fmt.Errorf("ErrSyntax: implicit multiplication not supported; use explicit '*'")
	}
	return left, nil
}

func (lp *linearExprParser) parseUnary() (LinearLinearTerm, error) {
	if lp.cur().Type == TokMinus {
		lp.next()
		val, err := lp.parsePrimary()
		if err != nil {
			return LinearLinearTerm{}, err
		}
		for i := 0; i < len(val.Coeffs); i++ {
			val.Coeffs[i].Neg(val.Coeffs[i])
		}
		val.Const.Neg(val.Const)
		return val, nil
	}
	if lp.cur().Type == TokPlus {
		lp.next()
		return lp.parsePrimary()
	}
	return lp.parsePrimary()
}

func (lp *linearExprParser) parsePrimary() (LinearLinearTerm, error) {
	tok := lp.cur()
	res := lp.newZeroTerm()

	if tok.Type == TokNum {
		lp.next()
		res.Const.Set(tok.NumRat)
		return res, nil
	}

	if tok.Type == TokRegister {
		lp.next()
		if lp.store == nil {
			return LinearLinearTerm{}, fmt.Errorf("ErrUnsetRegister: <|r%d|> is empty", tok.RegIdx)
		}
		val, err := lp.store.Get(tok.RegIdx)
		if err != nil {
			return LinearLinearTerm{}, err
		}
		if val.Kind == ValKindRat {
			res.Const.Set(val.Rat)
			return res, nil
		}
		return LinearLinearTerm{}, fmt.Errorf("ErrTypeMismatch: register does not contain rational number")
	}

	if tok.Type == TokIdent {
		lp.next()
		id := strings.ToLower(tok.Text)
		for idx, v := range lp.varList {
			if id == strings.ToLower(v) {
				res.Coeffs[idx].Set(big.NewRat(1, 1))
				return res, nil
			}
		}
		return LinearLinearTerm{}, fmt.Errorf("unexpected identifier %q in linear equation", tok.Text)
	}

	if tok.Type == TokLParen {
		lp.next()
		term, err := lp.parseAdditive()
		if err != nil {
			return LinearLinearTerm{}, err
		}
		if lp.cur().Type != TokRParen {
			return LinearLinearTerm{}, fmt.Errorf("expected ')'")
		}
		lp.next()
		return term, nil
	}

	return LinearLinearTerm{}, fmt.Errorf("unexpected token %q", tok.Text)
}

func tokenizeExpr(expr string) ([]Token, error) {
	lexer := NewCASLexer(expr)
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
	return tokens, nil
}

// -----------------------------------------------------------------------------
// Solve Entrypoints
// -----------------------------------------------------------------------------

// SolveEquation evaluates a solve command against globalRegisters without auto-binding.
// Tool dispatch handles register binding via DispatchTool/DispatchToolWithStore.
func SolveEquation(payload string) (string, error) {
	solns, err := SolveEquationWithStore(payload, globalRegisters)
	if err != nil {
		return "", err
	}
	if len(solns) == 0 {
		return "no real solutions", nil
	}
	if len(solns) == 1 && solns[0].Val.Kind == ValKindInterval {
		return solns[0].Val.Intervals.String(), nil
	}
	var parts []string
	for _, soln := range solns {
		parts = append(parts, fmt.Sprintf("%s = %s", soln.VarName, soln.Val.String()))
	}
	return strings.Join(parts, ", "), nil
}

// SolveEquationWithStore parses and solves an equation or linear system against the given RegisterStore.
func SolveEquationWithStore(payload string, store *RegisterStore) ([]SolveResult, error) {
	payload = strings.TrimSpace(payload)
	if payload == "" {
		return nil, fmt.Errorf("ErrSyntax: empty equation payload")
	}

	// Remove commas within numbers like 1,000 (M15)
	// A comma followed by 3 digits and then end/operator/space is part of a number
	cleanedPayload := cleanNumericCommas(payload)

	// Check for inequality operators: <=, >=, <, > (excluding ==)
	for _, op := range []string{"<=", ">=", "<", ">"} {
		if strings.Contains(cleanedPayload, op) && !strings.Contains(cleanedPayload, "==") {
			varName := "x"
			detected := detectVariableInString(cleanedPayload)
			if detected != "" {
				varName = detected
			}
			ivs, err := SolveInequality(cleanedPayload, varName, store)
			if err != nil {
				return nil, err
			}
			return []SolveResult{
				{VarName: varName, Val: CalcVal{Kind: ValKindInterval, Intervals: ivs}},
			}, nil
		}
	}

	// Check if this is a linear system or single equation
	// Split top-level commas that are equation separators
	parts := splitTopLevelCommas(cleanedPayload)

	var equations []string
	var varList []string
	isComplex := false

	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		if p == "complex" {
			isComplex = true
			continue
		}
		if strings.Contains(p, "=") {
			equations = append(equations, p)
		} else if strings.HasPrefix(p, "[") && strings.HasSuffix(p, "]") {
			// Variable list: [x, y]
			inner := strings.Trim(p, "[]")
			for _, v := range strings.Split(inner, ",") {
				v = strings.TrimSpace(v)
				if v == "complex" {
					isComplex = true
				} else if v != "" {
					varList = append(varList, v)
				}
			}
		} else {
			// Single variable specification: e.g. "x"
			v := strings.TrimSpace(p)
			if v != "" && !strings.Contains(v, " ") {
				varList = append(varList, v)
			}
		}
	}

	if len(equations) == 0 {
		return nil, fmt.Errorf("ErrSyntax: invalid equation (missing '='): %s", payload)
	}

	// Multi-equation linear system
	if len(equations) > 1 {
		if len(varList) == 0 {
			// Extract variables in order of appearance
			varList = extractVariablesFromEquations(equations)
		}
		return solveLinearSystem(equations, varList, store)
	}

	// Single equation
	eq := equations[0]
	eqSides := strings.Split(eq, "=")
	if len(eqSides) != 2 {
		return nil, fmt.Errorf("ErrSyntax: invalid equation (missing '='): %s", eq)
	}

	varName := "x"
	if len(varList) > 0 {
		varName = varList[0]
	} else {
		// Detect variable in equation
		detected := detectVariableInString(eq)
		if detected != "" {
			varName = detected
		}
	}

	return SolveEquationWithSides(eqSides[0], eqSides[1], varName, isComplex, store)
}

func solveMultiPoly(p *MultiPoly, varName string, isComplex bool) ([]SolveResult, error) {
	if p.IsZero() {
		return nil, fmt.Errorf("ErrInfiniteSolutions: all real numbers")
	}

	// Verify univariate in varName (or constant)
	deg := 0
	for _, t := range p.Terms {
		for _, vp := range t.key.Powers {
			if vp.Var != varName {
				return nil, fmt.Errorf("ErrSyntax: unexpected variable %q (expected %s)", vp.Var, varName)
			}
			if vp.Exp > deg {
				deg = vp.Exp
			}
		}
	}

	if deg == 0 {
		return nil, fmt.Errorf("ErrNoSolution: equation has no solution")
	}

	// Degree 1: a*x + b = 0 -> x = -b/a
	if deg == 1 {
		var a, b *ExactNum
		b = NewExactZero()
		for _, t := range p.Terms {
			d := 0
			if len(t.key.Powers) == 1 {
				d = t.key.Powers[0].Exp
			}
			if d == 1 {
				a = t.coeff
			} else if d == 0 {
				b = t.coeff
			}
		}
		if a == nil || a.IsZero() {
			return nil, fmt.Errorf("ErrSyntax: degenerate linear equation")
		}
		r, ok, err := b.Neg().Div(a)
		if err != nil {
			return nil, err
		}
		if !ok {
			return nil, fmt.Errorf("ErrUnsupported: root division could not be evaluated exactly")
		}
		val := CalcVal{Kind: ValKindExactNum, Exact: r}
		if rat, isRat := r.AsRational(); isRat {
			val = CalcVal{Kind: ValKindRat, Rat: rat}
		}
		return []SolveResult{
			{VarName: varName, Val: val},
		}, nil
	}

	// Degree 2: a*x^2 + b*x + c = 0
	if deg == 2 {
		var a, b, c *ExactNum
		b = NewExactZero()
		c = NewExactZero()
		for _, t := range p.Terms {
			d := 0
			if len(t.key.Powers) == 1 {
				d = t.key.Powers[0].Exp
			}
			if d == 2 {
				a = t.coeff
			} else if d == 1 {
				b = t.coeff
			} else if d == 0 {
				c = t.coeff
			}
		}
		if a == nil || a.IsZero() {
			return nil, fmt.Errorf("ErrSyntax: degenerate quadratic equation")
		}
		bSq, ok := b.Mul(b)
		if !ok {
			return nil, fmt.Errorf("ErrOverflow: discriminant square out of bounds")
		}
		fourAC, ok := NewExactInt(4).Mul(a)
		if ok {
			fourAC, ok = fourAC.Mul(c)
		}
		if !ok {
			return nil, fmt.Errorf("ErrOverflow: discriminant 4ac out of bounds")
		}
		disc := bSq.Sub(fourAC)
		twoA, ok := NewExactInt(2).Mul(a)
		if !ok {
			return nil, fmt.Errorf("ErrOverflow: 2a out of bounds")
		}
		negB := b.Neg()

		if disc.IsReal() {
			isNeg := false
			if rat, ok := disc.AsRational(); ok {
				isNeg = (rat.Sign() < 0)
			} else {
				isNeg = (disc.ToFloat64() < 0)
			}
			if isNeg {
				if !isComplex {
					// No real solutions
					return []SolveResult{}, nil
				}
			}
			if disc.IsZero() {
				r, ok, err := negB.Div(twoA)
				if err != nil {
					return nil, err
				}
				if !ok {
					return nil, fmt.Errorf("ErrUnsupported: root division could not be evaluated exactly")
				}
				val := CalcVal{Kind: ValKindExactNum, Exact: r}
				if rat, isRat := r.AsRational(); isRat {
					val = CalcVal{Kind: ValKindRat, Rat: rat}
				}
				return []SolveResult{
					{VarName: varName, Val: val},
				}, nil
			}
		}

		sqrtD, ok := disc.Sqrt()
		if !ok {
			return nil, fmt.Errorf("ErrUnsupported: discriminant sqrt could not be evaluated exactly")
		}
		r1Num := negB.Sub(sqrtD)
		r2Num := negB.Add(sqrtD)
		r1, ok1, err1 := r1Num.Div(twoA)
		r2, ok2, err2 := r2Num.Div(twoA)
		if err1 != nil || err2 != nil || !ok1 || !ok2 {
			return nil, fmt.Errorf("ErrUnsupported: root division could not be evaluated exactly")
		}

		val1 := CalcVal{Kind: ValKindExactNum, Exact: r1}
		if rat, isR := r1.AsRational(); isR {
			val1 = CalcVal{Kind: ValKindRat, Rat: rat}
		}
		val2 := CalcVal{Kind: ValKindExactNum, Exact: r2}
		if rat, isR := r2.AsRational(); isR {
			val2 = CalcVal{Kind: ValKindRat, Rat: rat}
		}

		if r1.IsReal() && r2.IsReal() {
			if r1.ToFloat64() > r2.ToFloat64() {
				val1, val2 = val2, val1
			}
		}

		return []SolveResult{
			{VarName: varName, Val: val1},
			{VarName: varName, Val: val2},
		}, nil
	}

	// Degree > 2: if all rational, use solvePolynomial
	pRat, _, err := MultiPolyToRatPoly(p)
	if err == nil {
		return solvePolynomial(pRat, varName, isComplex)
	}

	return nil, fmt.Errorf("ErrUnsupported: solving polynomial of degree > 2 with non-rational coefficients is not supported")
}

func SolveEquationWithSides(lhsStr, rhsStr, varName string, isComplex bool, store *RegisterStore) ([]SolveResult, error) {
	_, lhsPoly, err := ExpandPoly(lhsStr, store)
	if err != nil {
		if strings.HasPrefix(err.Error(), "Err") {
			return nil, err
		}
		return nil, fmt.Errorf("ErrSyntax: error parsing LHS: %v", err)
	}
	_, rhsPoly, err := ExpandPoly(rhsStr, store)
	if err != nil {
		if strings.HasPrefix(err.Error(), "Err") {
			return nil, err
		}
		return nil, fmt.Errorf("ErrSyntax: error parsing RHS: %v", err)
	}

	// P(x) = LHS - RHS = 0
	p := lhsPoly.Sub(rhsPoly)
	return solveMultiPoly(p, varName, isComplex)
}

func cleanNumericCommas(s string) string {
	// Replaces commas between digits: e.g. 1,000 -> 1000
	runes := []rune(s)
	var out []rune
	for i := 0; i < len(runes); i++ {
		if runes[i] == ',' && i > 0 && i+1 < len(runes) && unicode.IsDigit(runes[i-1]) && unicode.IsDigit(runes[i+1]) {
			// Skip comma
			continue
		}
		out = append(out, runes[i])
	}
	return string(out)
}

func splitTopLevelCommas(s string) []string {
	var parts []string
	var cur strings.Builder
	depthParen := 0
	depthBracket := 0

	for _, r := range s {
		switch r {
		case '(':
			depthParen++
			cur.WriteRune(r)
		case ')':
			depthParen--
			cur.WriteRune(r)
		case '[':
			depthBracket++
			cur.WriteRune(r)
		case ']':
			depthBracket--
			cur.WriteRune(r)
		case ',':
			if depthParen == 0 && depthBracket == 0 {
				parts = append(parts, cur.String())
				cur.Reset()
			} else {
				cur.WriteRune(r)
			}
		default:
			cur.WriteRune(r)
		}
	}
	if cur.Len() > 0 {
		parts = append(parts, cur.String())
	}
	return parts
}

func detectVariableInString(s string) string {
	tokens, err := tokenizeExpr(s)
	if err != nil {
		return ""
	}
	for _, t := range tokens {
		if t.Type == TokIdent {
			id := strings.ToLower(t.Text)
			if id != "pi" && id != "e" && id != "sqrt" && id != "sin" && id != "cos" {
				return id
			}
		}
	}
	return ""
}

func extractVariablesFromEquations(eqs []string) []string {
	var vars []string
	seen := make(map[string]bool)
	for _, eq := range eqs {
		tokens, err := tokenizeExpr(eq)
		if err != nil {
			continue
		}
		for _, t := range tokens {
			if t.Type == TokIdent {
				id := strings.ToLower(t.Text)
				if id != "pi" && id != "e" && id != "sqrt" && !seen[id] {
					seen[id] = true
					vars = append(vars, id)
				}
			}
		}
	}
	return vars
}
