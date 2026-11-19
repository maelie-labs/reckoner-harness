package main

import (
	"fmt"
	"math"
	"math/big"
)

// EvalSum evaluates sum(expr, k, a, b): exact sum over finite range a..b with limit <= 10^6 terms.
func EvalSum(expr, varName string, a, b int64, store *RegisterStore) (CalcVal, error) {
	tokens, err := tokenizeExpr(expr)
	if err != nil {
		return CalcVal{}, err
	}
	return EvalSumTokens(tokens, varName, a, b, store)
}

// EvalSumTokens evaluates sum from a slice of tokens.
func EvalSumTokens(tokens []Token, varName string, a, b int64, store *RegisterStore) (CalcVal, error) {
	if b < a {
		return CalcVal{Kind: ValKindExactNum, Exact: NewExactZero()}, nil
	}
	count := b - a + 1
	if count > 1000000 {
		return CalcVal{}, fmt.Errorf("ErrWorkLimitExceeded: sum range %d exceeds maximum limit of 10^6 terms", count)
	}

	const maxSumWorkBudget = 10000000 // 10^7 work units (cumulative operand bit-length)
	var cumulativeWork int64 = 0

	acc := NewExactZero()
	for i := a; i <= b; i++ {
		subTokens := substituteVarInTokens(tokens, varName, i)
		parser := NewParser(subTokens, store)
		val, err := parser.parseExpression()
		if err != nil {
			return CalcVal{}, err
		}

		exVal := calcValToExactNum(val)
		acc = acc.Add(exVal)
		if len(acc.Terms) > 100 {
			return CalcVal{}, fmt.Errorf("ErrOverflow: sum term count exceeds limit of 100")
		}

		stepCost := 64
		for _, c := range acc.Terms {
			nBits := c.Num().BitLen()
			dBits := c.Denom().BitLen()
			if nBits > 100000 || dBits > 100000 {
				return CalcVal{}, fmt.Errorf("ErrOverflow: sum coefficient exceeds limit of 10^5 bits")
			}
			stepCost += nBits + dBits
		}
		for _, c := range exVal.Terms {
			stepCost += c.Num().BitLen() + c.Denom().BitLen()
		}
		cumulativeWork += int64(stepCost)
		if cumulativeWork > maxSumWorkBudget {
			return CalcVal{}, fmt.Errorf("ErrOverflow: sum work budget exceeded")
		}
	}

	if rat, ok := acc.AsRational(); ok {
		return CalcVal{Kind: ValKindRat, Rat: rat}, nil
	}
	return CalcVal{Kind: ValKindExactNum, Exact: acc}, nil
}

// EvalProduct evaluates product(expr, k, a, b): exact product over finite range a..b with limit <= 10^6 terms.
func EvalProduct(expr, varName string, a, b int64, store *RegisterStore) (CalcVal, error) {
	tokens, err := tokenizeExpr(expr)
	if err != nil {
		return CalcVal{}, err
	}
	return EvalProductTokens(tokens, varName, a, b, store)
}

// EvalProductTokens evaluates product from a slice of tokens.
func EvalProductTokens(tokens []Token, varName string, a, b int64, store *RegisterStore) (CalcVal, error) {
	if b < a {
		return CalcVal{Kind: ValKindExactNum, Exact: NewExactInt(1)}, nil
	}
	count := b - a + 1
	if count > 1000000 {
		return CalcVal{}, fmt.Errorf("ErrWorkLimitExceeded: product range %d exceeds maximum limit of 10^6 terms", count)
	}

	const maxProductWorkBudget = 10000000 // 10^7 work units (cumulative operand bit-length)
	var cumulativeWork int64 = 0

	acc := NewExactInt(1)
	for i := a; i <= b; i++ {
		subTokens := substituteVarInTokens(tokens, varName, i)
		parser := NewParser(subTokens, store)
		val, err := parser.parseExpression()
		if err != nil {
			return CalcVal{}, err
		}

		exVal := calcValToExactNum(val)
		var ok bool
		acc, ok = acc.Mul(exVal)
		if !ok {
			return CalcVal{}, fmt.Errorf("ErrOverflow: product multiplication overflowed")
		}
		if len(acc.Terms) > 100 {
			return CalcVal{}, fmt.Errorf("ErrOverflow: product term count exceeds limit of 100")
		}

		stepCost := 64
		for _, c := range acc.Terms {
			nBits := c.Num().BitLen()
			dBits := c.Denom().BitLen()
			if nBits > 100000 || dBits > 100000 {
				return CalcVal{}, fmt.Errorf("ErrOverflow: product coefficient exceeds limit of 10^5 bits")
			}
			stepCost += nBits + dBits
		}
		for _, c := range exVal.Terms {
			stepCost += c.Num().BitLen() + c.Denom().BitLen()
		}
		cumulativeWork += int64(stepCost)
		if cumulativeWork > maxProductWorkBudget {
			return CalcVal{}, fmt.Errorf("ErrOverflow: product work budget exceeded")
		}
	}

	if rat, ok := acc.AsRational(); ok {
		return CalcVal{Kind: ValKindRat, Rat: rat}, nil
	}
	return CalcVal{Kind: ValKindExactNum, Exact: acc}, nil
}

func substituteVarInTokens(tokens []Token, varName string, val int64) []Token {
	var res []Token
	valStr := fmt.Sprintf("%d", val)
	for _, tok := range tokens {
		if tok.Type == TokIdent && tok.Text == varName {
			if val < 0 {
				res = append(res, Token{Type: TokLParen, Text: "(", Col: tok.Col})
				res = append(res, Token{Type: TokNum, Text: valStr, NumRat: big.NewRat(val, 1), Col: tok.Col})
				res = append(res, Token{Type: TokRParen, Text: ")", Col: tok.Col})
			} else {
				res = append(res, Token{Type: TokNum, Text: valStr, NumRat: big.NewRat(val, 1), Col: tok.Col})
			}
		} else {
			res = append(res, tok)
		}
	}
	return res
}

func calcValToExactNum(v CalcVal) *ExactNum {
	switch v.Kind {
	case ValKindExactNum:
		if v.Exact != nil {
			return v.Exact
		}
		return NewExactZero()
	case ValKindRat:
		if v.Rat != nil {
			return NewExactRational(v.Rat)
		}
		return NewExactZero()
	case ValKindSurd:
		if v.Surd != nil {
			return NewExactSurd(v.Surd)
		}
		return NewExactZero()
	case ValKindFloat:
		if math.IsNaN(v.Float) || math.IsInf(v.Float, 0) {
			return nil
		}
		r := new(big.Rat).SetFloat64(v.Float)
		return NewExactRational(r)
	case ValKindUnit:
		if v.UnitVal != nil {
			return NewExactRational(v.UnitVal)
		}
		return NewExactZero()
	case ValKindPoly:
		if v.Poly != nil {
			if c, ok := v.Poly.AsConstant(); ok {
				return c
			}
		}
		return nil
	default:
		return nil
	}
}

// BinomPMF computes binom_pmf(k, n, p) = C(n, k) * p^k * (1-p)^(n-k).
func BinomPMF(k, n int64, pRat *big.Rat) (*big.Rat, error) {
	if k < 0 || n < 0 || k > n {
		return big.NewRat(0, 1), nil
	}
	if n > 10000 {
		return nil, fmt.Errorf("ErrOverflow: binomial n exceeds limit of 10,000")
	}
	if pRat.Sign() < 0 || pRat.Cmp(big.NewRat(1, 1)) > 0 {
		return nil, fmt.Errorf("ErrDomain: probability p must be in [0, 1]")
	}

	c, err := combBigInt(n, k)
	if err != nil {
		return nil, err
	}

	pK := ratPowInt(pRat, k)
	qRat := new(big.Rat).Sub(big.NewRat(1, 1), pRat)
	qNK := ratPowInt(qRat, n-k)
	if pK == nil || qNK == nil {
		return nil, fmt.Errorf("ErrOverflow: binomial power computation overflowed")
	}

	res := new(big.Rat).Mul(pK, qNK)
	res.Mul(res, new(big.Rat).SetInt(c))
	return res, nil
}

// BinomCDF computes binom_cdf(k, n, p) = sum_{j=0}^k C(n, j) * p^j * (1-p)^(n-j).
func BinomCDF(k, n int64, pRat *big.Rat) (*big.Rat, error) {
	if k < 0 {
		return big.NewRat(0, 1), nil
	}
	if k >= n {
		return big.NewRat(1, 1), nil
	}
	if n > 10000 {
		return nil, fmt.Errorf("ErrOverflow: binomial n exceeds limit of 10,000")
	}
	if k > 1000 {
		return nil, fmt.Errorf("ErrWorkLimitExceeded: binom_cdf k exceeds limit of 1000")
	}
	if pRat.Sign() < 0 || pRat.Cmp(big.NewRat(1, 1)) > 0 {
		return nil, fmt.Errorf("ErrDomain: probability p must be in [0, 1]")
	}

	total := new(big.Rat)
	for j := int64(0); j <= k; j++ {
		term, err := BinomPMF(j, n, pRat)
		if err != nil {
			return nil, err
		}
		total.Add(total, term)
	}
	return total, nil
}

func combBigInt(n, k int64) (*big.Int, error) {
	if k < 0 || n < 0 || k > n {
		return big.NewInt(0), nil
	}
	if n > 10000 {
		return nil, fmt.Errorf("ErrOverflow: binomial n exceeds limit of 10,000")
	}
	if k > n/2 {
		k = n - k
	}
	res := big.NewInt(1)
	for i := int64(1); i <= k; i++ {
		res.Mul(res, big.NewInt(n-i+1))
		res.Div(res, big.NewInt(i))
		if res.BitLen() > 100000 {
			return nil, fmt.Errorf("ErrOverflow: binomial coefficient exceeds limit of 10^5 bits")
		}
	}
	return res, nil
}
