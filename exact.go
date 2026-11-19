package main

import (
	"fmt"
	"math"
	"math/big"
	"sort"
	"strings"
)

// BasisKey represents the basis monomial b = i^E * pi^K * sqrt(R) * e^P * log_LogBase(LogArg)
// where:
// - E in {0, 1}
// - K in {-2..3}
// - R is a square-free positive integer >= 1
// - P in {-10..10} (power of e)
// - LogBase in {0 (for ln), 10 (for log10)}
// - LogArg is prime p >= 2 for log(p); 0 if no log
type BasisKey struct {
	E       int
	K       int
	R       int64
	P       int
	LogBase int64
	LogArg  int64
}

// ExactNum represents a finite linear combination of basis monomials with rational coefficients:
// sum_j c_j * b_j
type ExactNum struct {
	Terms map[BasisKey]*big.Rat
}

func NewExactZero() *ExactNum {
	return &ExactNum{Terms: make(map[BasisKey]*big.Rat)}
}

func NewExactRational(r *big.Rat) *ExactNum {
	en := NewExactZero()
	if r != nil && r.Sign() != 0 {
		en.Terms[BasisKey{E: 0, K: 0, R: 1, P: 0, LogBase: 0, LogArg: 0}] = new(big.Rat).Set(r)
	}
	return en
}

func NewExactInt(n int64) *ExactNum {
	return NewExactRational(big.NewRat(n, 1))
}

func NewExactPi() *ExactNum {
	en := NewExactZero()
	en.Terms[BasisKey{E: 0, K: 1, R: 1, P: 0, LogBase: 0, LogArg: 0}] = big.NewRat(1, 1)
	return en
}

func NewExactI() *ExactNum {
	en := NewExactZero()
	en.Terms[BasisKey{E: 1, K: 0, R: 1, P: 0, LogBase: 0, LogArg: 0}] = big.NewRat(1, 1)
	return en
}

func NewExactE() *ExactNum {
	en := NewExactZero()
	en.Terms[BasisKey{E: 0, K: 0, R: 1, P: 1, LogBase: 0, LogArg: 0}] = big.NewRat(1, 1)
	return en
}

func NewExactEPow(p int) *ExactNum {
	if p == 0 {
		return NewExactInt(1)
	}
	en := NewExactZero()
	en.Terms[BasisKey{E: 0, K: 0, R: 1, P: p, LogBase: 0, LogArg: 0}] = big.NewRat(1, 1)
	return en
}

func isPowerOf10(n *big.Int) (int64, bool) {
	if n == nil || n.Sign() <= 0 {
		return 0, false
	}
	s := n.String()
	if s[0] != '1' {
		return 0, false
	}
	for i := 1; i < len(s); i++ {
		if s[i] != '0' {
			return 0, false
		}
	}
	return int64(len(s) - 1), true
}

func isRatPowerOf10(r *big.Rat) (int64, bool) {
	if r == nil || r.Sign() <= 0 {
		return 0, false
	}
	num := r.Num()
	den := r.Denom()
	if den.Cmp(big.NewInt(1)) == 0 {
		return isPowerOf10(num)
	}
	if num.Cmp(big.NewInt(1)) == 0 {
		k, ok := isPowerOf10(den)
		if ok {
			return -k, true
		}
	}
	return 0, false
}

func isPowerOf2(n *big.Int) (int64, bool) {
	if n == nil || n.Sign() <= 0 {
		return 0, false
	}
	sub1 := new(big.Int).Sub(n, big.NewInt(1))
	and := new(big.Int).And(n, sub1)
	if and.Sign() == 0 {
		return int64(n.BitLen() - 1), true
	}
	return 0, false
}

func isRatPowerOf2(r *big.Rat) (int64, bool) {
	if r == nil || r.Sign() <= 0 {
		return 0, false
	}
	num := r.Num()
	den := r.Denom()
	if den.Cmp(big.NewInt(1)) == 0 {
		return isPowerOf2(num)
	}
	if num.Cmp(big.NewInt(1)) == 0 {
		k, ok := isPowerOf2(den)
		if ok {
			return -k, true
		}
	}
	return 0, false
}

func factorInt64(n *big.Int) ([]int64, bool) {
	if n == nil || n.Sign() <= 0 {
		return nil, false
	}
	if !n.IsInt64() {
		return nil, false
	}
	val := n.Int64()
	if val <= 0 {
		return nil, false
	}
	if val == 1 {
		return nil, true
	}
	facs, err := primeFactorsWithBudget(val, 100000)
	if err != nil {
		return nil, false
	}
	return facs, true
}

func NewExactLn(r *big.Rat) (*ExactNum, bool) {
	if r == nil || r.Sign() <= 0 {
		return nil, false
	}
	if r.Cmp(big.NewRat(1, 1)) == 0 {
		return NewExactZero(), true
	}
	numFactors, ok1 := factorInt64(r.Num())
	denFactors, ok2 := factorInt64(r.Denom())
	if !ok1 || !ok2 {
		return nil, false
	}

	en := NewExactZero()
	powers := make(map[int64]int64)
	for _, p := range numFactors {
		powers[p]++
	}
	for _, p := range denFactors {
		powers[p]--
	}

	for p, count := range powers {
		if count != 0 {
			key := BasisKey{E: 0, K: 0, R: 1, P: 0, LogBase: 0, LogArg: p}
			en.Terms[key] = big.NewRat(count, 1)
		}
	}
	return en, true
}

func NewExactLog10(r *big.Rat) (*ExactNum, bool) {
	if r == nil || r.Sign() <= 0 {
		return nil, false
	}
	if r.Cmp(big.NewRat(1, 1)) == 0 {
		return NewExactZero(), true
	}
	if k, ok := isRatPowerOf10(r); ok {
		return NewExactInt(k), true
	}

	numFactors, ok1 := factorInt64(r.Num())
	denFactors, ok2 := factorInt64(r.Denom())
	if !ok1 || !ok2 {
		return nil, false
	}

	en := NewExactZero()
	powers := make(map[int64]int64)
	for _, p := range numFactors {
		powers[p]++
	}
	for _, p := range denFactors {
		powers[p]--
	}

	var rationalConst int64 = 0
	for p, count := range powers {
		if count == 0 {
			continue
		}
		if p == 5 {
			// log10(5) = 1 - log10(2)
			rationalConst += count
			powers[2] -= count
		} else if p == 2 {
			// handled after loop
		} else {
			key := BasisKey{E: 0, K: 0, R: 1, P: 0, LogBase: 10, LogArg: p}
			en.Terms[key] = big.NewRat(count, 1)
		}
	}
	if powers[2] != 0 {
		key := BasisKey{E: 0, K: 0, R: 1, P: 0, LogBase: 10, LogArg: 2}
		en.Terms[key] = big.NewRat(powers[2], 1)
	}
	if rationalConst != 0 {
		ratKey := BasisKey{E: 0, K: 0, R: 1, P: 0, LogBase: 0, LogArg: 0}
		if cur, ok := en.Terms[ratKey]; ok {
			sum := new(big.Rat).Add(cur, big.NewRat(rationalConst, 1))
			if sum.Sign() == 0 {
				delete(en.Terms, ratKey)
			} else {
				en.Terms[ratKey] = sum
			}
		} else {
			en.Terms[ratKey] = big.NewRat(rationalConst, 1)
		}
	}
	return en, true
}

func (en *ExactNum) AsLogRational() (*big.Rat, bool) {
	if en == nil || len(en.Terms) == 0 {
		return nil, false
	}
	res := big.NewRat(1, 1)
	for k, v := range en.Terms {
		if k.E != 0 || k.K != 0 || k.R > 1 || k.P != 0 || k.LogBase != 0 || k.LogArg <= 0 {
			return nil, false
		}
		if !v.IsInt() {
			return nil, false
		}
		p := k.LogArg
		exp := v.Num().Int64()
		if exp > 50 || exp < -50 {
			return nil, false
		}
		pRat := big.NewRat(p, 1)
		if exp > 0 {
			for i := int64(0); i < exp; i++ {
				res.Mul(res, pRat)
			}
		} else if exp < 0 {
			for i := int64(0); i < -exp; i++ {
				res.Quo(res, pRat)
			}
		}
	}
	return res, true
}

func (en *ExactNum) AsEPow() (int, bool) {
	if en == nil || len(en.Terms) != 1 {
		return 0, false
	}
	for k, v := range en.Terms {
		if k.E == 0 && k.K == 0 && k.R == 1 && k.LogArg == 0 && k.P != 0 {
			if v.Cmp(big.NewRat(1, 1)) == 0 {
				return k.P, true
			}
		}
	}
	return 0, false
}

func (en *ExactNum) HasFloatFormat() bool {
	for k := range en.Terms {
		if k.P != 0 || k.LogArg != 0 {
			return true
		}
	}
	return false
}

func NewExactSurd(s *SurdVal) *ExactNum {
	en := NewExactZero()
	if s == nil || s.a == nil || s.b == nil || s.c == nil {
		return en
	}
	if s.a.Sign() == 0 || s.b.Sign() == 0 {
		return en
	}

	coeffNum := new(big.Int).Set(s.a)
	if s.sign < 0 {
		coeffNum.Neg(coeffNum)
	}
	coeff := new(big.Rat).SetFrac(coeffNum, s.c)

	r := s.b.Int64()
	if r <= 1 {
		en.Terms[BasisKey{E: 0, K: 0, R: 1}] = coeff
	} else {
		en.Terms[BasisKey{E: 0, K: 0, R: r}] = coeff
	}
	return en
}

func (en *ExactNum) Clone() *ExactNum {
	res := NewExactZero()
	for k, v := range en.Terms {
		res.Terms[k] = new(big.Rat).Set(v)
	}
	return res
}

func (en *ExactNum) IsZero() bool {
	return len(en.Terms) == 0
}

func (en *ExactNum) Equal(other *ExactNum) bool {
	if en == nil && other == nil {
		return true
	}
	if en == nil || other == nil {
		return false
	}
	if len(en.Terms) != len(other.Terms) {
		return false
	}
	for k, v1 := range en.Terms {
		v2, ok := other.Terms[k]
		if !ok || v1.Cmp(v2) != 0 {
			return false
		}
	}
	return true
}

func (en *ExactNum) IsRational() bool {
	if len(en.Terms) == 0 {
		return true
	}
	if len(en.Terms) == 1 {
		_, ok := en.Terms[BasisKey{E: 0, K: 0, R: 1}]
		return ok
	}
	return false
}

func (en *ExactNum) AsRational() (*big.Rat, bool) {
	if len(en.Terms) == 0 {
		return big.NewRat(0, 1), true
	}
	if len(en.Terms) == 1 {
		if r, ok := en.Terms[BasisKey{E: 0, K: 0, R: 1}]; ok {
			return new(big.Rat).Set(r), true
		}
	}
	return nil, false
}

func (en *ExactNum) IsReal() bool {
	for k := range en.Terms {
		if k.E != 0 {
			return false
		}
	}
	return true
}

func (en *ExactNum) Add(other *ExactNum) *ExactNum {
	res := en.Clone()
	for k, v := range other.Terms {
		if cur, ok := res.Terms[k]; ok {
			sum := new(big.Rat).Add(cur, v)
			if sum.Sign() == 0 {
				delete(res.Terms, k)
			} else {
				res.Terms[k] = sum
			}
		} else {
			res.Terms[k] = new(big.Rat).Set(v)
		}
	}
	return res
}

func (en *ExactNum) Sub(other *ExactNum) *ExactNum {
	res := en.Clone()
	for k, v := range other.Terms {
		if cur, ok := res.Terms[k]; ok {
			diff := new(big.Rat).Sub(cur, v)
			if diff.Sign() == 0 {
				delete(res.Terms, k)
			} else {
				res.Terms[k] = diff
			}
		} else {
			res.Terms[k] = new(big.Rat).Neg(v)
		}
	}
	return res
}

func (en *ExactNum) Neg() *ExactNum {
	res := NewExactZero()
	for k, v := range en.Terms {
		res.Terms[k] = new(big.Rat).Neg(v)
	}
	return res
}

// Mul multiplies two exact numbers. If any resulting pi power is outside {-2..3}, returns nil, false.
func (en *ExactNum) Mul(other *ExactNum) (*ExactNum, bool) {
	res := NewExactZero()
	for k1, v1 := range en.Terms {
		for k2, v2 := range other.Terms {
			c := new(big.Rat).Mul(v1, v2)

			// i^(e1+e2)
			e := k1.E + k2.E
			if e == 2 {
				c.Neg(c)
				e = 0
			}

			// pi^(k1+k2)
			k := k1.K + k2.K
			if k < -2 || k > 3 {
				return nil, false // pi power out of basis bounds -> fallback to float
			}

			// e^(p1+p2)
			p := k1.P + k2.P
			if p < -10 || p > 10 {
				return nil, false
			}

			// Log terms: cannot multiply two log terms or log with surd/pi/i/e
			if k1.LogArg != 0 && k2.LogArg != 0 {
				return nil, false
			}
			logBase := int64(0)
			logArg := int64(0)
			if k1.LogArg != 0 {
				if k2.E != 0 || k2.K != 0 || k2.R > 1 || k2.P != 0 {
					return nil, false
				}
				logBase = k1.LogBase
				logArg = k1.LogArg
			} else if k2.LogArg != 0 {
				if k1.E != 0 || k1.K != 0 || k1.R > 1 || k1.P != 0 {
					return nil, false
				}
				logBase = k2.LogBase
				logArg = k2.LogArg
			}

			// e^p cannot multiply with surds, pi, or i in exact basis
			if p != 0 && (e != 0 || k != 0) {
				return nil, false
			}

			// sqrt(r1 * r2)
			prodR := big.NewInt(k1.R * k2.R)
			sq, rem, err := simplifyRadicalBig(prodR, 100000)
			if err != nil {
				return nil, false
			}
			c.Mul(c, new(big.Rat).SetInt(sq))
			r := rem.Int64()
			if p != 0 && r > 1 {
				return nil, false
			}
			if c.Num().BitLen() > 100000 || c.Denom().BitLen() > 100000 {
				return nil, false
			}

			key := BasisKey{E: e, K: k, R: r, P: p, LogBase: logBase, LogArg: logArg}
			if cur, ok := res.Terms[key]; ok {
				sum := new(big.Rat).Add(cur, c)
				if sum.Sign() == 0 {
					delete(res.Terms, key)
				} else {
					if sum.Num().BitLen() > 100000 || sum.Denom().BitLen() > 100000 {
						return nil, false
					}
					res.Terms[key] = sum
				}
			} else {
				if len(res.Terms) >= 100 {
					return nil, false
				}
				res.Terms[key] = c
			}
		}
	}
	return res, true
}

func (en *ExactNum) MulRational(q *big.Rat) *ExactNum {
	if q == nil || q.Sign() == 0 {
		return NewExactZero()
	}
	res := NewExactZero()
	for k, v := range en.Terms {
		res.Terms[k] = new(big.Rat).Mul(v, q)
	}
	return res
}

// Div divides en by other. If exact division is possible, returns (*ExactNum, true).
// If divisor is zero, returns error. If inexact/unsupported denominator, returns nil, false.
func (en *ExactNum) Div(other *ExactNum) (*ExactNum, bool, error) {
	if other.IsZero() {
		return nil, false, fmt.Errorf("ErrDivZero: division by zero")
	}

	if en.Equal(other) {
		return NewExactInt(1), true, nil
	}

	// 1. Division by monomial
	if len(other.Terms) == 1 {
		var dk BasisKey
		var dv *big.Rat
		for k, v := range other.Terms {
			dk = k
			dv = v
		}

		res := NewExactZero()
		for nk, nv := range en.Terms {
			c := new(big.Rat).Quo(nv, dv)

			// i^(ne - de) = i^ne * (-1)^de * i^de
			e := nk.E
			if dk.E == 1 {
				if e == 1 {
					// i / i = 1
					e = 0
				} else {
					// 1 / i = -i
					e = 1
					c.Neg(c)
				}
			}

			// pi^(nk - dk)
			k := nk.K - dk.K
			if k < -2 || k > 3 {
				return nil, false, nil // pi power out of bounds
			}

			// e^(np - dp)
			p := nk.P - dk.P
			if p < -10 || p > 10 {
				return nil, false, nil
			}

			// Log terms
			logBase := int64(0)
			logArg := int64(0)
			if dk.LogArg != 0 {
				if nk.LogArg == dk.LogArg && nk.LogBase == dk.LogBase {
					logBase = 0
					logArg = 0
				} else {
					return nil, false, nil
				}
			} else if nk.LogArg != 0 {
				logBase = nk.LogBase
				logArg = nk.LogArg
			}

			if p != 0 && (e != 0 || k != 0) {
				return nil, false, nil
			}

			// sqrt(nr) / sqrt(dr) = sqrt(nr * dr) / dr
			prodR := big.NewInt(nk.R * dk.R)
			sq, rem, err := simplifyRadicalBig(prodR, 100000)
			if err != nil {
				return nil, false, nil
			}
			c.Mul(c, new(big.Rat).SetFrac(sq, big.NewInt(dk.R)))
			r := rem.Int64()
			if p != 0 && r > 1 {
				return nil, false, nil
			}

			key := BasisKey{E: e, K: k, R: r, P: p, LogBase: logBase, LogArg: logArg}
			if cur, ok := res.Terms[key]; ok {
				sum := new(big.Rat).Add(cur, c)
				if sum.Sign() == 0 {
					delete(res.Terms, key)
				} else {
					res.Terms[key] = sum
				}
			} else {
				res.Terms[key] = c
			}
		}
		return res, true, nil
	}

	// 2. Division by binomial/elements in Q(sqrt(r), i) via conjugates
	// Check if other is in Q(sqrt(r), i) with K == 0 for all terms
	var surdR int64 = 1
	for k := range other.Terms {
		if k.K != 0 || k.P != 0 || k.LogArg != 0 {
			return nil, false, nil // Pi/e/log terms in multi-term denominator -> float fallback
		}
		if k.R > 1 {
			if surdR == 1 {
				surdR = k.R
			} else if surdR != k.R {
				return nil, false, nil // Multiple distinct surds in denominator -> float fallback
			}
		}
	}

	// other = (a + b*sqrt(r)) + (c + d*sqrt(r))*i
	// Complex conjugate: other^* = (a + b*sqrt(r)) - (c + d*sqrt(r))*i
	conj1 := NewExactZero()
	for k, v := range other.Terms {
		if k.E == 1 {
			conj1.Terms[k] = new(big.Rat).Neg(v)
		} else {
			conj1.Terms[k] = new(big.Rat).Set(v)
		}
	}

	// prod1 = other * conj1 (purely real, in Q(sqrt(r)))
	prod1, ok := other.Mul(conj1)
	if !ok {
		return nil, false, nil
	}

	// Now prod1 = X + Y*sqrt(r) with X, Y in Q
	// Radical conjugate: prod2 = X - Y*sqrt(r)
	conj2 := NewExactZero()
	for k, v := range prod1.Terms {
		if k.R > 1 {
			conj2.Terms[k] = new(big.Rat).Neg(v)
		} else {
			conj2.Terms[k] = new(big.Rat).Set(v)
		}
	}

	// denomRational = prod1 * conj2
	denomProd, ok := prod1.Mul(conj2)
	if !ok {
		return nil, false, nil
	}
	denomRat, isRat := denomProd.AsRational()
	if !isRat || denomRat.Sign() == 0 {
		return nil, false, nil
	}

	// Numerator = en * conj1 * conj2
	num1, ok := en.Mul(conj1)
	if !ok {
		return nil, false, nil
	}
	num2, ok := num1.Mul(conj2)
	if !ok {
		return nil, false, nil
	}

	invDenom := new(big.Rat).Inv(denomRat)
	res := num2.MulRational(invDenom)
	return res, true, nil
}

// Sqrt computes square root of en.
// If exact (rational, surd, imaginary, or denested p + q*sqrt(r)), returns (*ExactNum, true).
// Otherwise returns nil, false (indicating float fallback).
func (en *ExactNum) Sqrt() (*ExactNum, bool) {
	if en.IsZero() {
		return NewExactZero(), true
	}

	// Case 1: rational
	if rat, ok := en.AsRational(); ok {
		if rat.Sign() > 0 {
			num := rat.Num()
			den := rat.Denom()
			prod := new(big.Int).Mul(num, den)
			sq, rem, err := simplifyRadicalBig(prod, 100000)
			if err != nil {
				return nil, false
			}
			coeff := new(big.Rat).SetFrac(sq, den)
			res := NewExactZero()
			r := rem.Int64()
			res.Terms[BasisKey{E: 0, K: 0, R: r}] = coeff
			return res, true
		} else {
			// Negative rational: i * sqrt(-rat)
			posRat := new(big.Rat).Neg(rat)
			num := posRat.Num()
			den := posRat.Denom()
			prod := new(big.Int).Mul(num, den)
			sq, rem, err := simplifyRadicalBig(prod, 100000)
			if err != nil {
				return nil, false
			}
			coeff := new(big.Rat).SetFrac(sq, den)
			res := NewExactZero()
			r := rem.Int64()
			res.Terms[BasisKey{E: 1, K: 0, R: r}] = coeff
			return res, true
		}
	}

	// Case 2: denesting of p + q*sqrt(r)
	// Requires all terms to have E == 0, K == 0, and at most two terms (one R=1, one R=r>1)
	if en.IsReal() {
		hasPi := false
		for k := range en.Terms {
			if k.K != 0 {
				hasPi = true
				break
			}
		}
		if !hasPi {
			var pRat *big.Rat = big.NewRat(0, 1)
			var qRat *big.Rat = big.NewRat(0, 1)
			var rInt int64 = 1

			for k, v := range en.Terms {
				if k.R == 1 {
					pRat = new(big.Rat).Set(v)
				} else {
					if rInt == 1 {
						rInt = k.R
						qRat = new(big.Rat).Set(v)
					} else {
						// More than one distinct radical
						return nil, false
					}
				}
			}

			if rInt > 1 && qRat.Sign() != 0 && pRat.Sign() > 0 {
				// Delta = p^2 - q^2 * r
				pSq := new(big.Rat).Mul(pRat, pRat)
				qSq := new(big.Rat).Mul(qRat, qRat)
				rRat := new(big.Rat).SetInt64(rInt)
				qSqR := new(big.Rat).Mul(qSq, rRat)
				delta := new(big.Rat).Sub(pSq, qSqR)

				if delta.Sign() >= 0 {
					// Check if delta is a rational square
					numSq, numExact := integerSquareRoot(delta.Num())
					denSq, denExact := integerSquareRoot(delta.Denom())
					if numExact && denExact {
						d := new(big.Rat).SetFrac(numSq, denSq)
						// u = (p + d) / 2
						u := new(big.Rat).Quo(new(big.Rat).Add(pRat, d), big.NewRat(2, 1))
						// v = (p - d) / 2
						v := new(big.Rat).Quo(new(big.Rat).Sub(pRat, d), big.NewRat(2, 1))

						if u.Sign() >= 0 && v.Sign() >= 0 {
							uSqrt, okU := NewExactRational(u).Sqrt()
							vSqrt, okV := NewExactRational(v).Sqrt()
							if okU && okV {
								if qRat.Sign() < 0 {
									return uSqrt.Sub(vSqrt), true
								}
								return uSqrt.Add(vSqrt), true
							}
						}
					}
				}
			}
		}
	}

	return nil, false
}

func (en *ExactNum) Pow(exp int64) (*ExactNum, bool, error) {
	if exp > 1000 || exp < -1000 {
		return nil, false, fmt.Errorf("ErrOverflow: exponent exceeds maximum limit of 1000")
	}
	if exp == 0 {
		return NewExactInt(1), true, nil
	}
	if exp < 0 {
		posPow, ok, err := en.Pow(-exp)
		if err != nil {
			return nil, false, err
		}
		if !ok {
			return nil, false, nil
		}
		return NewExactInt(1).Div(posPow)
	}

	res := NewExactInt(1)
	base := en.Clone()
	curExp := exp

	for curExp > 0 {
		if curExp%2 == 1 {
			var ok bool
			res, ok = res.Mul(base)
			if !ok {
				return nil, false, nil
			}
		}
		if curExp > 1 {
			var ok bool
			base, ok = base.Mul(base)
			if !ok {
				return nil, false, nil
			}
		}
		curExp /= 2
	}
	return res, true, nil
}

func (en *ExactNum) Re() *ExactNum {
	res := NewExactZero()
	for k, v := range en.Terms {
		if k.E == 0 {
			res.Terms[k] = new(big.Rat).Set(v)
		}
	}
	return res
}

func (en *ExactNum) Im() *ExactNum {
	res := NewExactZero()
	for k, v := range en.Terms {
		if k.E == 1 {
			realKey := BasisKey{E: 0, K: k.K, R: k.R, P: k.P, LogBase: k.LogBase, LogArg: k.LogArg}
			res.Terms[realKey] = new(big.Rat).Set(v)
		}
	}
	return res
}

func (en *ExactNum) Conj() *ExactNum {
	res := NewExactZero()
	for k, v := range en.Terms {
		if k.E == 1 {
			res.Terms[k] = new(big.Rat).Neg(v)
		} else {
			res.Terms[k] = new(big.Rat).Set(v)
		}
	}
	return res
}

func (en *ExactNum) Abs() (*ExactNum, bool) {
	if en.IsReal() {
		// abs(real)
		if rat, ok := en.AsRational(); ok {
			if rat.Sign() < 0 {
				return NewExactRational(new(big.Rat).Neg(rat)), true
			}
			return NewExactRational(rat), true
		}
		f := en.ToFloat64()
		if f < 0 {
			return en.Neg(), true
		}
		return en.Clone(), true
	}

	// abs(z) = sqrt(z * conj(z))
	conj := en.Conj()
	normSq, ok := en.Mul(conj)
	if !ok {
		return nil, false
	}
	return normSq.Sqrt()
}

func (en *ExactNum) Arg() (*ExactNum, bool) {
	// Exact standard angles: 0, pi/6, pi/4, pi/3, pi/2, etc.
	re := en.Re()
	im := en.Im()

	reRat, reIsRat := re.AsRational()
	imRat, imIsRat := im.AsRational()

	if reIsRat && imIsRat {
		if reRat.Sign() == 0 && imRat.Sign() == 0 {
			return NewExactZero(), true
		}
		// Purely real
		if imRat.Sign() == 0 {
			if reRat.Sign() > 0 {
				return NewExactZero(), true
			}
			return NewExactPi(), true // pi
		}
		// Purely imaginary
		if reRat.Sign() == 0 {
			piOver2, _, _ := NewExactPi().Div(NewExactInt(2))
			if imRat.Sign() > 0 {
				return piOver2, true // pi/2
			}
			return piOver2.Neg(), true // -pi/2
		}
		// Equal magnitude: 1+i, 1-i, -1+i, -1-i -> pi/4
		if new(big.Rat).Abs(reRat).Cmp(new(big.Rat).Abs(imRat)) == 0 {
			piOver4, _, _ := NewExactPi().Div(NewExactInt(4))
			if reRat.Sign() > 0 && imRat.Sign() > 0 {
				return piOver4, true // pi/4
			}
			if reRat.Sign() < 0 && imRat.Sign() > 0 {
				// 3*pi/4
				threePiOver4, _ := piOver4.Mul(NewExactInt(3))
				return threePiOver4, true
			}
			if reRat.Sign() < 0 && imRat.Sign() < 0 {
				// -3*pi/4
				threePiOver4, _ := piOver4.Mul(NewExactInt(3))
				return threePiOver4.Neg(), true
			}
			if reRat.Sign() > 0 && imRat.Sign() < 0 {
				return piOver4.Neg(), true // -pi/4
			}
		}
	}

	// Check if ratio im/re or re/im matches sqrt(3)
	// z = 1 + sqrt(3)*i -> pi/3
	// z = sqrt(3) + i -> pi/6
	reF := re.ToFloat64()
	imF := im.ToFloat64()
	thetaF := math.Atan2(imF, reF)

	// Check standard multiples of pi/12
	kFloat := thetaF * 12.0 / math.Pi
	kRound := math.Round(kFloat)
	if math.Abs(kFloat-kRound) < 1e-9 {
		k := int64(kRound)
		// k * pi / 12
		rat := big.NewRat(k, 12)
		res := NewExactPi().MulRational(rat)
		return res, true
	}

	return nil, false
}

func (en *ExactNum) ToFloat64() float64 {
	sum := 0.0
	for k, v := range en.Terms {
		coeffF, _ := v.Float64()
		termF := coeffF
		if k.K != 0 {
			termF *= math.Pow(math.Pi, float64(k.K))
		}
		if k.R > 1 {
			termF *= math.Sqrt(float64(k.R))
		}
		if k.P != 0 {
			termF *= math.Pow(math.E, float64(k.P))
		}
		if k.LogArg > 0 {
			if k.LogBase == 10 {
				termF *= math.Log10(float64(k.LogArg))
			} else {
				termF *= math.Log(float64(k.LogArg))
			}
		}
		if k.E == 1 {
			// Real projection or float approximation
			// For complex numbers, ToFloat64 gives real part if purely real, or magnitude if needed
		}
		sum += termF
	}
	return sum
}

// -----------------------------------------------------------------------------
// Canonical String Display
// -----------------------------------------------------------------------------

type termDisplay struct {
	key   BasisKey
	num   *big.Int // numerator with common denominator cleared
	coeff *big.Rat
}

// String produces canonical, deterministic display of the exact number.
// - terms ordered: rational, then surds by r, then pi terms, then imaginary;
// - a common denominator is factored out when every term shares it.
// Examples:
//   25*pi
//   (1+sqrt(5))/2
//   3-4*i
//   2*sqrt(3)+pi
//   8*sqrt(2)*pi
//   (3+sqrt(7)*i)/2
func (en *ExactNum) String() string {
	if en.IsZero() {
		return "0"
	}

	// If purely rational, use formatRational
	if rat, ok := en.AsRational(); ok {
		return formatRational(rat, false, false, false, 0)
	}

	// If it contains uncanceled e or log terms, display as float (e.g. bare e -> 2.718282)
	if en.HasFloatFormat() {
		return formatFloat(en.ToFloat64(), false, 0)
	}

	// Collect terms
	var terms []termDisplay
	for k, v := range en.Terms {
		terms = append(terms, termDisplay{key: k, coeff: v})
	}

	// Sort terms per specification:
	// rational (E=0, K=0, R=1),
	// surds by r (E=0, K=0, R>1),
	// pi terms (E=0, K!=0),
	// imaginary (E=1)
	sort.Slice(terms, func(i, j int) bool {
		ki := terms[i].key
		kj := terms[j].key

		rank := func(k BasisKey) (int, int64, int) {
			if k.E == 0 {
				if k.K == 0 && k.R == 1 {
					return 1, 0, 0 // rational
				}
				if k.K == 0 && k.R > 1 {
					return 2, k.R, 0 // surds by r
				}
				return 3, int64(k.K), int(k.R) // pi terms
			}
			// imaginary
			if k.K == 0 && k.R == 1 {
				return 4, 0, 0 // rational imaginary
			}
			if k.K == 0 && k.R > 1 {
				return 5, k.R, 0 // surd imaginary
			}
			return 6, int64(k.K), int(k.R) // pi imaginary
		}

		r1, a1, b1 := rank(ki)
		r2, a2, b2 := rank(kj)
		if r1 != r2 {
			return r1 < r2
		}
		if a1 != a2 {
			return a1 < a2
		}
		return b1 < b2
	})

	// Find common denominator: D = lcm(denoms)
	D := big.NewInt(1)
	for _, t := range terms {
		den := t.coeff.Denom()
		g := new(big.Int).GCD(nil, nil, D, den)
		D.Mul(D, den)
		D.Quo(D, g)
	}

	// Compute integer numerators: num = coeff.Num * (D / coeff.Denom)
	for i := range terms {
		mul := new(big.Int).Quo(D, terms[i].coeff.Denom())
		terms[i].num = new(big.Int).Mul(terms[i].coeff.Num(), mul)
	}

	// Format basis monomial with integer coefficient N
	formatMonomial := func(n *big.Int, k BasisKey) string {
		signStr := ""
		absN := new(big.Int).Abs(n)
		if n.Sign() < 0 {
			signStr = "-"
		}

		// Basis string parts
		var parts []string
		if k.R > 1 {
			parts = append(parts, fmt.Sprintf("sqrt(%d)", k.R))
		}
		if k.K == 1 {
			parts = append(parts, "pi")
		} else if k.K > 1 {
			parts = append(parts, fmt.Sprintf("pi^%d", k.K))
		} else if k.K == -1 {
			parts = append(parts, "1/pi")
		} else if k.K < -1 {
			parts = append(parts, fmt.Sprintf("1/pi^%d", -k.K))
		}
		if k.E == 1 {
			parts = append(parts, "i")
		}

		basisStr := strings.Join(parts, "*")

		if basisStr == "" {
			return signStr + absN.String()
		}
		if absN.Cmp(big.NewInt(1)) == 0 {
			return signStr + basisStr
		}
		return fmt.Sprintf("%s%s*%s", signStr, absN.String(), basisStr)
	}

	// If single term
	if len(terms) == 1 {
		t := terms[0]
		if D.Cmp(big.NewInt(1)) == 0 {
			return formatMonomial(t.num, t.key)
		}
		// Single term with denominator: e.g. sqrt(5)/2, pi/2, 3*sqrt(2)/4
		numStr := formatMonomial(t.num, t.key)
		return fmt.Sprintf("%s/%s", numStr, D.String())
	}

	// Multiple terms:
	// Check if common denominator > 1 is factored out: e.g. (1+sqrt(5))/2, (3+sqrt(7)*i)/2
	var numParts []string
	for i, t := range terms {
		mStr := formatMonomial(t.num, t.key)
		if i == 0 {
			numParts = append(numParts, mStr)
		} else {
			if t.num.Sign() < 0 {
				// starts with '-'
				numParts = append(numParts, mStr)
			} else {
				numParts = append(numParts, "+"+mStr)
			}
		}
	}

	numeratorStr := strings.Join(numParts, "")

	if D.Cmp(big.NewInt(1)) == 0 {
		return numeratorStr
	}

	// Factored out common denominator: (num)/D
	return fmt.Sprintf("(%s)/%s", numeratorStr, D.String())
}

// -----------------------------------------------------------------------------
// Exact Trigonometry Tables (Multiples of pi/12)
// -----------------------------------------------------------------------------

// SinExact computes exact sin(theta) if theta is a rational multiple of pi with denominator dividing 12.
func SinExact(theta *ExactNum) (*ExactNum, bool) {
	k, ok := getPi12Multiple(theta)
	if !ok {
		return nil, false
	}
	return sinKPi12(k), true
}

func CosExact(theta *ExactNum) (*ExactNum, bool) {
	k, ok := getPi12Multiple(theta)
	if !ok {
		return nil, false
	}
	return cosKPi12(k), true
}

func TanExact(theta *ExactNum) (*ExactNum, bool, error) {
	k, ok := getPi12Multiple(theta)
	if !ok {
		return nil, false, nil
	}
	if k%12 == 6 {
		return nil, true, fmt.Errorf("ErrDivZero: tan(pi/2) is undefined")
	}
	res := tanKPi12(k)
	if res != nil {
		return res, true, nil
	}
	return nil, false, nil
}

func SecExact(theta *ExactNum) (*ExactNum, bool, error) {
	k, ok := getPi12Multiple(theta)
	if !ok {
		return nil, false, nil
	}
	if k%12 == 6 {
		return nil, true, fmt.Errorf("ErrDivZero: sec is undefined when cos is 0")
	}
	m := ((k % 24) + 24) % 24
	switch m {
	case 1, 23: // sqrt(6) - sqrt(2)
		res := NewExactZero()
		res.Terms[BasisKey{E: 0, K: 0, R: 6}] = big.NewRat(1, 1)
		res.Terms[BasisKey{E: 0, K: 0, R: 2}] = big.NewRat(-1, 1)
		return res, true, nil
	case 5, 19: // sqrt(6) + sqrt(2)
		res := NewExactZero()
		res.Terms[BasisKey{E: 0, K: 0, R: 6}] = big.NewRat(1, 1)
		res.Terms[BasisKey{E: 0, K: 0, R: 2}] = big.NewRat(1, 1)
		return res, true, nil
	case 7, 17: // -sqrt(6) - sqrt(2)
		res := NewExactZero()
		res.Terms[BasisKey{E: 0, K: 0, R: 6}] = big.NewRat(-1, 1)
		res.Terms[BasisKey{E: 0, K: 0, R: 2}] = big.NewRat(-1, 1)
		return res, true, nil
	case 11, 13: // -sqrt(6) + sqrt(2)
		res := NewExactZero()
		res.Terms[BasisKey{E: 0, K: 0, R: 6}] = big.NewRat(-1, 1)
		res.Terms[BasisKey{E: 0, K: 0, R: 2}] = big.NewRat(1, 1)
		return res, true, nil
	default:
		c := cosKPi12(k)
		res, exact, err := NewExactInt(1).Div(c)
		return res, exact, err
	}
}

func CscExact(theta *ExactNum) (*ExactNum, bool, error) {
	k, ok := getPi12Multiple(theta)
	if !ok {
		return nil, false, nil
	}
	if k%12 == 0 {
		return nil, true, fmt.Errorf("ErrDivZero: csc is undefined when sin is 0")
	}
	m := ((k % 24) + 24) % 24
	switch m {
	case 1, 11: // sqrt(6) + sqrt(2)
		res := NewExactZero()
		res.Terms[BasisKey{E: 0, K: 0, R: 6}] = big.NewRat(1, 1)
		res.Terms[BasisKey{E: 0, K: 0, R: 2}] = big.NewRat(1, 1)
		return res, true, nil
	case 5, 7: // sqrt(6) - sqrt(2)
		res := NewExactZero()
		res.Terms[BasisKey{E: 0, K: 0, R: 6}] = big.NewRat(1, 1)
		res.Terms[BasisKey{E: 0, K: 0, R: 2}] = big.NewRat(-1, 1)
		return res, true, nil
	case 13, 23: // -sqrt(6) - sqrt(2)
		res := NewExactZero()
		res.Terms[BasisKey{E: 0, K: 0, R: 6}] = big.NewRat(-1, 1)
		res.Terms[BasisKey{E: 0, K: 0, R: 2}] = big.NewRat(-1, 1)
		return res, true, nil
	case 17, 19: // -sqrt(6) + sqrt(2)
		res := NewExactZero()
		res.Terms[BasisKey{E: 0, K: 0, R: 6}] = big.NewRat(-1, 1)
		res.Terms[BasisKey{E: 0, K: 0, R: 2}] = big.NewRat(1, 1)
		return res, true, nil
	default:
		s := sinKPi12(k)
		res, exact, err := NewExactInt(1).Div(s)
		return res, exact, err
	}
}

func CotExact(theta *ExactNum) (*ExactNum, bool, error) {
	k, ok := getPi12Multiple(theta)
	if !ok {
		return nil, false, nil
	}
	if k%12 == 0 {
		return nil, true, fmt.Errorf("ErrDivZero: cot is undefined when sin is 0")
	}
	res := cotKPi12(k)
	if res != nil {
		return res, true, nil
	}
	return nil, false, nil
}

func tanKPi12(k int) *ExactNum {
	m := ((k % 12) + 12) % 12
	switch m {
	case 0:
		return NewExactZero()
	case 1: // 2 - sqrt(3)
		res := NewExactInt(2)
		res.Terms[BasisKey{E: 0, K: 0, R: 3}] = big.NewRat(-1, 1)
		return res
	case 2: // sqrt(3)/3
		res := NewExactZero()
		res.Terms[BasisKey{E: 0, K: 0, R: 3}] = big.NewRat(1, 3)
		return res
	case 3: // 1
		return NewExactInt(1)
	case 4: // sqrt(3)
		res := NewExactZero()
		res.Terms[BasisKey{E: 0, K: 0, R: 3}] = big.NewRat(1, 1)
		return res
	case 5: // 2 + sqrt(3)
		res := NewExactInt(2)
		res.Terms[BasisKey{E: 0, K: 0, R: 3}] = big.NewRat(1, 1)
		return res
	case 7: // -2 - sqrt(3)
		res := NewExactInt(-2)
		res.Terms[BasisKey{E: 0, K: 0, R: 3}] = big.NewRat(-1, 1)
		return res
	case 8: // -sqrt(3)
		res := NewExactZero()
		res.Terms[BasisKey{E: 0, K: 0, R: 3}] = big.NewRat(-1, 1)
		return res
	case 9: // -1
		return NewExactInt(-1)
	case 10: // -sqrt(3)/3
		res := NewExactZero()
		res.Terms[BasisKey{E: 0, K: 0, R: 3}] = big.NewRat(-1, 3)
		return res
	case 11: // -2 + sqrt(3)
		res := NewExactInt(-2)
		res.Terms[BasisKey{E: 0, K: 0, R: 3}] = big.NewRat(1, 1)
		return res
	}
	return nil
}

func cotKPi12(k int) *ExactNum {
	m := ((k % 12) + 12) % 12
	switch m {
	case 1: // 2 + sqrt(3)
		res := NewExactInt(2)
		res.Terms[BasisKey{E: 0, K: 0, R: 3}] = big.NewRat(1, 1)
		return res
	case 2: // sqrt(3)
		res := NewExactZero()
		res.Terms[BasisKey{E: 0, K: 0, R: 3}] = big.NewRat(1, 1)
		return res
	case 3: // 1
		return NewExactInt(1)
	case 4: // sqrt(3)/3
		res := NewExactZero()
		res.Terms[BasisKey{E: 0, K: 0, R: 3}] = big.NewRat(1, 3)
		return res
	case 5: // 2 - sqrt(3)
		res := NewExactInt(2)
		res.Terms[BasisKey{E: 0, K: 0, R: 3}] = big.NewRat(-1, 1)
		return res
	case 6: // 0
		return NewExactZero()
	case 7: // -2 + sqrt(3)
		res := NewExactInt(-2)
		res.Terms[BasisKey{E: 0, K: 0, R: 3}] = big.NewRat(1, 1)
		return res
	case 8: // -sqrt(3)/3
		res := NewExactZero()
		res.Terms[BasisKey{E: 0, K: 0, R: 3}] = big.NewRat(-1, 3)
		return res
	case 9: // -1
		return NewExactInt(-1)
	case 10: // -sqrt(3)
		res := NewExactZero()
		res.Terms[BasisKey{E: 0, K: 0, R: 3}] = big.NewRat(-1, 1)
		return res
	case 11: // -2 - sqrt(3)
		res := NewExactInt(-2)
		res.Terms[BasisKey{E: 0, K: 0, R: 3}] = big.NewRat(-1, 1)
		return res
	}
	return nil
}

func getPi12Multiple(theta *ExactNum) (int, bool) {
	if theta.IsZero() {
		return 0, true
	}
	if len(theta.Terms) != 1 {
		return 0, false
	}
	for k, v := range theta.Terms {
		if k.E == 0 && k.K == 1 && k.R == 1 {
			// theta = v * pi
			// Check if v * 12 is an integer
			twelveV := new(big.Rat).Mul(v, big.NewRat(12, 1))
			if twelveV.IsInt() {
				m := twelveV.Num().Int64()
				m = ((m % 24) + 24) % 24
				return int(m), true
			}
		}
	}
	return 0, false
}

// sin(k * pi / 12) for k in 0..23
func sinKPi12(k int) *ExactNum {
	k = ((k % 24) + 24) % 24
	if k >= 12 {
		return sinKPi12(k - 12).Neg()
	}
	if k > 6 {
		return sinKPi12(12 - k)
	}

	switch k {
	case 0:
		return NewExactZero()
	case 1: // 15 deg: (sqrt(6) - sqrt(2)) / 4
		res := NewExactZero()
		res.Terms[BasisKey{E: 0, K: 0, R: 6}] = big.NewRat(1, 4)
		res.Terms[BasisKey{E: 0, K: 0, R: 2}] = big.NewRat(-1, 4)
		return res
	case 2: // 30 deg: 1/2
		return NewExactRational(big.NewRat(1, 2))
	case 3: // 45 deg: sqrt(2)/2
		res := NewExactZero()
		res.Terms[BasisKey{E: 0, K: 0, R: 2}] = big.NewRat(1, 2)
		return res
	case 4: // 60 deg: sqrt(3)/2
		res := NewExactZero()
		res.Terms[BasisKey{E: 0, K: 0, R: 3}] = big.NewRat(1, 2)
		return res
	case 5: // 75 deg: (sqrt(6) + sqrt(2)) / 4
		res := NewExactZero()
		res.Terms[BasisKey{E: 0, K: 0, R: 6}] = big.NewRat(1, 4)
		res.Terms[BasisKey{E: 0, K: 0, R: 2}] = big.NewRat(1, 4)
		return res
	case 6: // 90 deg: 1
		return NewExactInt(1)
	}
	return NewExactZero()
}

// cos(k * pi / 12) for k in 0..23
func cosKPi12(k int) *ExactNum {
	k = ((k % 24) + 24) % 24
	if k >= 12 {
		return cosKPi12(k - 12).Neg()
	}
	if k > 6 {
		return cosKPi12(12 - k).Neg()
	}

	switch k {
	case 0:
		return NewExactInt(1)
	case 1: // 15 deg: (sqrt(6) + sqrt(2)) / 4
		res := NewExactZero()
		res.Terms[BasisKey{E: 0, K: 0, R: 6}] = big.NewRat(1, 4)
		res.Terms[BasisKey{E: 0, K: 0, R: 2}] = big.NewRat(1, 4)
		return res
	case 2: // 30 deg: sqrt(3)/2
		res := NewExactZero()
		res.Terms[BasisKey{E: 0, K: 0, R: 3}] = big.NewRat(1, 2)
		return res
	case 3: // 45 deg: sqrt(2)/2
		res := NewExactZero()
		res.Terms[BasisKey{E: 0, K: 0, R: 2}] = big.NewRat(1, 2)
		return res
	case 4: // 60 deg: 1/2
		return NewExactRational(big.NewRat(1, 2))
	case 5: // 75 deg: (sqrt(6) - sqrt(2)) / 4
		res := NewExactZero()
		res.Terms[BasisKey{E: 0, K: 0, R: 6}] = big.NewRat(1, 4)
		res.Terms[BasisKey{E: 0, K: 0, R: 2}] = big.NewRat(-1, 4)
		return res
	case 6: // 90 deg: 0
		return NewExactZero()
	}
	return NewExactZero()
}

// Inverse trig helpers matching exact values
func AsinExact(val *ExactNum) (*ExactNum, bool) {
	for k := -6; k <= 6; k++ {
		target := sinKPi12(k)
		diff := val.Sub(target)
		if diff.IsZero() {
			rat := big.NewRat(int64(k), 12)
			return NewExactPi().MulRational(rat), true
		}
	}
	return nil, false
}

func AcosExact(val *ExactNum) (*ExactNum, bool) {
	for k := 0; k <= 12; k++ {
		target := cosKPi12(k)
		diff := val.Sub(target)
		if diff.IsZero() {
			rat := big.NewRat(int64(k), 12)
			return NewExactPi().MulRational(rat), true
		}
	}
	return nil, false
}

func AtanExact(val *ExactNum) (*ExactNum, bool) {
	for k := -5; k <= 5; k++ {
		target, ok, _ := TanExact(NewExactPi().MulRational(big.NewRat(int64(k), 12)))
		if ok && target != nil {
			diff := val.Sub(target)
			if diff.IsZero() {
				rat := big.NewRat(int64(k), 12)
				return NewExactPi().MulRational(rat), true
			}
		}
	}
	return nil, false
}

// Exact log(x, b): exact when x = b^(p/q)
func LogExact(xRat, bRat *big.Rat) (*big.Rat, bool) {
	if xRat.Sign() <= 0 || bRat.Sign() <= 0 {
		return nil, false
	}
	if bRat.Cmp(big.NewRat(1, 1)) == 0 {
		return nil, false
	}
	if xRat.Cmp(big.NewRat(1, 1)) == 0 {
		return big.NewRat(0, 1), true
	}
	if xRat.Cmp(bRat) == 0 {
		return big.NewRat(1, 1), true
	}

	// Try integer powers p in -20..20, q in 1..10
	for q := int64(1); q <= 10; q++ {
		for p := int64(-20); p <= 20; p++ {
			if p == 0 {
				continue
			}
			// Test if b^(p/q) == x <=> b^p == x^q
			var bp, xq *big.Rat
			if p > 0 {
				bp = ratPowInt(bRat, p)
			} else {
				bp = new(big.Rat).Inv(ratPowInt(bRat, -p))
			}
			xq = ratPowInt(xRat, q)
			if bp != nil && xq != nil && bp.Cmp(xq) == 0 {
				return big.NewRat(p, q), true
			}
		}
	}

	return nil, false
}

func ratPowInt(r *big.Rat, exp int64) *big.Rat {
	if r == nil {
		return nil
	}
	if exp < 0 {
		if exp < -100000 {
			return nil
		}
		inv := new(big.Rat).Inv(r)
		return ratPowInt(inv, -exp)
	}
	if exp > 100000 {
		return nil
	}
	numBits := int64(r.Num().BitLen())
	denBits := int64(r.Denom().BitLen())
	maxBits := numBits
	if denBits > maxBits {
		maxBits = denBits
	}
	if maxBits > 1 && maxBits*exp > 1048576 {
		return nil
	}
	num := new(big.Int).Exp(r.Num(), big.NewInt(exp), nil)
	den := new(big.Int).Exp(r.Denom(), big.NewInt(exp), nil)
	return new(big.Rat).SetFrac(num, den)
}
