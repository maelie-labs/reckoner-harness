package main

import (
	"math/big"
	"testing"
)

func TestExactNumRequiredExamples(t *testing.T) {
	// 1. 25*pi
	pi25 := NewExactPi().MulRational(big.NewRat(25, 1))
	if got := pi25.String(); got != "25*pi" {
		t.Errorf("25*pi display = %q, want %q", got, "25*pi")
	}

	// 2. (1+sqrt(5))/2
	one := NewExactInt(1)
	sqrt5 := NewExactSurd(&SurdVal{sign: 1, a: big.NewInt(1), b: big.NewInt(5), c: big.NewInt(1)})
	sum1 := one.Add(sqrt5)
	golden, ok, err := sum1.Div(NewExactInt(2))
	if !ok || err != nil {
		t.Fatalf("Div failed: %v", err)
	}
	if got := golden.String(); got != "(1+sqrt(5))/2" {
		t.Errorf("(1+sqrt(5))/2 display = %q, want %q", got, "(1+sqrt(5))/2")
	}

	// 3. 3-4*i
	three := NewExactInt(3)
	fourI := NewExactI().MulRational(big.NewRat(4, 1))
	c34 := three.Sub(fourI)
	if got := c34.String(); got != "3-4*i" {
		t.Errorf("3-4*i display = %q, want %q", got, "3-4*i")
	}

	// 4. 2*sqrt(3)+pi
	surd3 := NewExactSurd(&SurdVal{sign: 1, a: big.NewInt(2), b: big.NewInt(3), c: big.NewInt(1)})
	piTerm := NewExactPi()
	s23pi := surd3.Add(piTerm)
	if got := s23pi.String(); got != "2*sqrt(3)+pi" {
		t.Errorf("2*sqrt(3)+pi display = %q, want %q", got, "2*sqrt(3)+pi")
	}

	// 5. 8*sqrt(2)*pi
	sqrt2 := NewExactSurd(&SurdVal{sign: 1, a: big.NewInt(1), b: big.NewInt(2), c: big.NewInt(1)})
	p8sqrt2, ok := sqrt2.Mul(piTerm)
	if !ok {
		t.Fatalf("Mul failed")
	}
	p8sqrt2 = p8sqrt2.MulRational(big.NewRat(8, 1))
	if got := p8sqrt2.String(); got != "8*sqrt(2)*pi" {
		t.Errorf("8*sqrt(2)*pi display = %q, want %q", got, "8*sqrt(2)*pi")
	}

	// 6. (3+sqrt(7)*i)/2
	threeHalf := NewExactInt(3)
	sqrt7 := NewExactSurd(&SurdVal{sign: 1, a: big.NewInt(1), b: big.NewInt(7), c: big.NewInt(1)})
	sqrt7I, ok := sqrt7.Mul(NewExactI())
	if !ok {
		t.Fatalf("Mul failed")
	}
	c37 := threeHalf.Add(sqrt7I)
	c37Half, ok, err := c37.Div(NewExactInt(2))
	if !ok || err != nil {
		t.Fatalf("Div failed: %v", err)
	}
	if got := c37Half.String(); got != "(3+sqrt(7)*i)/2" {
		t.Errorf("(3+sqrt(7)*i)/2 display = %q, want %q", got, "(3+sqrt(7)*i)/2")
	}
}

func TestExactNumDenesting(t *testing.T) {
	// sqrt(3 + 2*sqrt(2)) = 1 + sqrt(2)
	three := NewExactInt(3)
	twoSqrt2 := NewExactSurd(&SurdVal{sign: 1, a: big.NewInt(2), b: big.NewInt(2), c: big.NewInt(1)})
	inner := three.Add(twoSqrt2)
	res, ok := inner.Sqrt()
	if !ok {
		t.Fatalf("Sqrt denesting failed")
	}
	if got := res.String(); got != "1+sqrt(2)" {
		t.Errorf("sqrt(3+2*sqrt(2)) = %q, want %q", got, "1+sqrt(2)")
	}
}

func TestExactNumDivisionByConjugate(t *testing.T) {
	// 1 / (1 + sqrt(2)) = -1 + sqrt(2)
	one := NewExactInt(1)
	sqrt2 := NewExactSurd(&SurdVal{sign: 1, a: big.NewInt(1), b: big.NewInt(2), c: big.NewInt(1)})
	denom := one.Add(sqrt2)
	res, ok, err := one.Div(denom)
	if !ok || err != nil {
		t.Fatalf("Div failed: %v", err)
	}
	if got := res.String(); got != "-1+sqrt(2)" {
		t.Errorf("1/(1+sqrt(2)) = %q, want %q", got, "-1+sqrt(2)")
	}

	// 1 / (3 + 4*i) = (3 - 4*i) / 25
	c := NewExactInt(3).Add(NewExactI().MulRational(big.NewRat(4, 1)))
	res2, ok, err := one.Div(c)
	if !ok || err != nil {
		t.Fatalf("Div failed: %v", err)
	}
	if got := res2.String(); got != "(3-4*i)/25" {
		t.Errorf("1/(3+4*i) = %q, want %q", got, "(3-4*i)/25")
	}
}

func TestExactTrig(t *testing.T) {
	// sin(pi/6) = 1/2
	piOver6 := NewExactPi().MulRational(big.NewRat(1, 6))
	s, ok := SinExact(piOver6)
	if !ok || s.String() != "0.5" && s.String() != "1/2" {
		t.Errorf("sin(pi/6) = %v, want 1/2 or 0.5", s)
	}

	// cos(pi/3) = 1/2
	piOver3 := NewExactPi().MulRational(big.NewRat(1, 3))
	c, ok := CosExact(piOver3)
	if !ok || c.String() != "0.5" && c.String() != "1/2" {
		t.Errorf("cos(pi/3) = %v, want 1/2 or 0.5", c)
	}

	// tan(pi/4) = 1
	piOver4 := NewExactPi().MulRational(big.NewRat(1, 4))
	tan, ok, err := TanExact(piOver4)
	if !ok || err != nil || tan.String() != "1" {
		t.Errorf("tan(pi/4) = %v (err: %v), want 1", tan, err)
	}

	// arcsin(1/2) = pi/6
	half := NewExactRational(big.NewRat(1, 2))
	asinVal, ok := AsinExact(half)
	if !ok || asinVal.String() != "pi/6" {
		t.Errorf("arcsin(1/2) = %v, want pi/6", asinVal)
	}
}

func TestExactLog(t *testing.T) {
	// log(8, 2) = 3
	res, ok := LogExact(big.NewRat(8, 1), big.NewRat(2, 1))
	if !ok || res.Cmp(big.NewRat(3, 1)) != 0 {
		t.Errorf("log(8, 2) = %v, want 3", res)
	}

	// log(1/9, 3) = -2
	res2, ok := LogExact(big.NewRat(1, 9), big.NewRat(3, 1))
	if !ok || res2.Cmp(big.NewRat(-2, 1)) != 0 {
		t.Errorf("log(1/9, 3) = %v, want -2", res2)
	}
}
