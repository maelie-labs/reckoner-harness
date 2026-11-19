package main

import (
	"fmt"
	"math"
	"math/big"
	"strconv"
	"strings"
	"testing"
)

func TestEvaluateInfix(t *testing.T) {
	tests := []struct {
		expr     string
		expected string
	}{
		{"2 + 3", "5"},
		{"482 * 19 - 14", "9144"},
		{"100 / 4 + 2 * 3", "31"},
		{"(1/4) * 4 * 7", "7"},
		{"2250 % 360", "90"},
		{"sqrt(144)", "12"},
		{"abs(-42)", "42"},
		{"2^10", "1024"},
		{"2**10", "1024"},
		{"1.5e3 + 500", "2000"},
		{"gcd(12, 18)", "6"},
		{"lcm(4, 6)", "12"},
		{"comb(5, 2)", "10"},
		{"perm(5, 2)", "20"},
		{"factorial(5)", "120"},
		{"min(10, 5, 20)", "5"},
		{"max(10, 5, 20)", "20"},
		{"modinv(3, 7)", "5"},
		{"modinv(10, 17)", "12"},
		{"isprime(97)", "true"},
		{"isprime(100)", "false"},
		{"factors(60)", "[2, 2, 3, 5]"},
		{"factors(13)", "[13]"},
		{"base(25, 2)", "11001"},
		{"base(45, 7)", "63"},
		{"det(1, 2, 3, 4)", "-2"},
		{"det3(1, 2, 3, 0, 1, 4, 5, 6, 0)", "1"},
		{"round(3.14159, 2)", "3.14"},
		{"round(3.14159)", "3"},
		{"sum(1, 2, 3, 4, 5)", "15"},
		{"prod(2, 3, 4)", "24"},
		{"sum_range(1, 100)", "5050"},
		{"2/3", "2/3"},
		{"1/4 + 1/6", "5/12"},
		{"(5/6) - (1/4)", "7/12"},
		{"1 - 5/8", "0.375"},
		{"frac(1 - 5/8)", "3/8"},
		{"sqrt(12)", "2*sqrt(3)"},
		{"sqrt(18)", "3*sqrt(2)"},
		{"sqrt(3) / 2", "sqrt(3)/2"},
		{"-sqrt(8)", "-2*sqrt(2)"},
		{"float(2/3)", "0.666667"},
		{"2.0 / 3", "2/3"},
		{"approx(85/3)", "28.3333"},
		{"round(85/3, 2)", "28.33"},
		{"float(500 * 6.6446573450e-27 kg)", "3.32233e-24 kg"},
		{"approx(2.5e-7 kg)", "2.5e-07 kg"},
		{"approx(500 * 6.6446573450e-27)", "3.32233e-24"},
		{"float(4.4995394185e-10 J * 2500 / (299792458 m/s)^2)", "1.2516e-23 kg"},
		{"convert(50 g / (4.5 g/cm^3), \"m^3\", \"cm^3\")", "100/9 cm^3"},
		{"5*m + 20", "5*m + 20"},
		{"k/2 - k/3", "1/6*k"},
		{"4*A - 5*A", "-A"},
		{"(x - m) + (10 - x)", "-m + 10"},
		{"9.8 m/s^2", "9.8 m/s^2"},
		{"convert(3 km, \"km\", \"m\") + 200 m", "3200 m"},
	}

	for _, tc := range tests {
		t.Run(tc.expr, func(t *testing.T) {
			res, err := EvaluateInfix(tc.expr)
			if err != nil {
				t.Fatalf("EvaluateInfix(%q) returned error: %v", tc.expr, err)
			}
			if res != tc.expected {
				t.Errorf("EvaluateInfix(%q) = %q, expected %q", tc.expr, res, tc.expected)
			}
		})
	}
}

func TestEvaluateInfixConstants(t *testing.T) {
	res, err := EvaluateInfix("pi * 4")
	if err != nil {
		t.Fatalf("pi error: %v", err)
	}
	if res == "4*pi" {
		return // Exact symbolic representation
	}
	val, _ := strconv.ParseFloat(res, 64)
	expected := math.Pi * 4
	if math.Abs(val-expected) > 1e-4 {
		t.Errorf("pi * 4 = %f, expected %f", val, expected)
	}
}

func TestSolveLinear(t *testing.T) {
	tests := []struct {
		query    string
		expected string
	}{
		{"2*x = 10", "x = 5"},
		{"5*x - 1 = x + 3", "x = 1"},
		{"5*x - 1 = x + 3, x", "x = 1"},
		{"(1/2)*x + 3 = 7", "x = 8"},
		{"x/2 + 3 = 7", "x = 8"},
		{"2*(x + 3) = 4*(x - 1)", "x = 5"},
		{"-3*x + 12 = 0", "x = 4"},
		{"x + 4 = 10", "x = 6"},
		{"3*x = 2", "x = 2/3"},
		{"4*x + 1 = 6", "x = 1.25"},
		{"c + 7*c = 560, c", "c = 70"},
		{"3*t = 12, t", "t = 4"},
		{"2*s + s = 9, s", "s = 3"},
		{"24 + S + 3*S = 40, S", "S = 4"},
		{"-2*m = m - 3, m", "m = 1"},
	}

	for _, tc := range tests {
		t.Run(tc.query, func(t *testing.T) {
			res, err := SolveEquation(tc.query)
			if err != nil {
				t.Fatalf("SolveEquation(%q) error: %v", tc.query, err)
			}
			if res != tc.expected {
				t.Errorf("SolveEquation(%q) = %q, expected %q", tc.query, res, tc.expected)
			}
		})
	}
}

func TestSolveQuadratic(t *testing.T) {
	tests := []struct {
		query    string
		expected string
	}{
		{"x^2 = 9", "x = -3, x = 3"},
		{"x^2 - 5*x + 6 = 0", "x = 2, x = 3"},
		{"x^2 - 4 = 0", "x = -2, x = 2"},
		{"(x - 1)*(x - 2) = 0", "x = 1, x = 2"},
		{"x*(x - 4) = -3", "x = 1, x = 3"},
		{"x^2 + 2*x + 1 = 0", "x = -1"},
		{"x^2 + 1 = 0", "no real solutions"},
		{"x^2 = 12", "x = -2*sqrt(3), x = 2*sqrt(3)"},
		{"m^2 - 3*m + 3 = 1, m", "m = 1, m = 2"},
	}

	for _, tc := range tests {
		t.Run(tc.query, func(t *testing.T) {
			res, err := SolveEquation(tc.query)
			if err != nil {
				t.Fatalf("SolveEquation(%q) error: %v", tc.query, err)
			}
			if res != tc.expected {
				t.Errorf("SolveEquation(%q) = %q, expected %q", tc.query, res, tc.expected)
			}
		})
	}
}

func TestSolveStrictImplicitMultiplication(t *testing.T) {
	strictCases := []string{
		"c + 7c = 560, c",
		"3t = 12, t",
		"2s + s = 9, s",
		"24 + S + 3S = 40, S",
		"-2m = m - 3, m",
		"m^2 - 3m + 3 = 1, m",
		"n(n+1) = 12, n",
		"x(x-1) = 0, x",
	}

	for _, query := range strictCases {
		t.Run(query, func(t *testing.T) {
			_, err := SolveEquation(query)
			if err == nil {
				t.Fatalf("SolveEquation(%q) expected error, got nil", query)
			}
			if !strings.Contains(err.Error(), "ErrSyntax: implicit multiplication not supported; use explicit '*'") {
				t.Errorf("SolveEquation(%q) expected implicit multiplication error, got %v", query, err)
			}
		})
	}
}

func TestSolve2x2System(t *testing.T) {
	tests := []struct {
		query    string
		expected string
	}{
		{"2*x + 3*y = 13, x - y = -1", "x = 2, y = 3"},
		{"2*x + 3*y = 13, x - y = -1, [x, y]", "x = 2, y = 3"},
		{"x + y = 10, x - y = 2", "x = 6, y = 4"},
		{"2*x + 3*y = 1, x - y = 2", "x = 1.4, y = -0.6"},
	}

	for _, tc := range tests {
		t.Run(tc.query, func(t *testing.T) {
			res, err := SolveEquation(tc.query)
			if err != nil {
				t.Fatalf("SolveEquation(%q) error: %v", tc.query, err)
			}
			if res != tc.expected {
				t.Errorf("SolveEquation(%q) = %q, expected %q", tc.query, res, tc.expected)
			}
		})
	}
}

func TestRegistersAndRingBuffer(t *testing.T) {
	rs := NewRegisterStore()
	cmd1 := "r0 = 40 * 2"
	res1, err := rs.ExecuteRegisterCmd(cmd1)
	if err != nil || res1 != "<|r0|> = 80" {
		t.Fatalf("cmd1 failed: %v, %s", err, res1)
	}

	cmd2 := "r1 = <|r0|> + 15"
	res2, err := rs.ExecuteRegisterCmd(cmd2)
	if err != nil || res2 != "<|r1|> = 95" {
		t.Fatalf("cmd2 failed: %v, %s", err, res2)
	}

	val, err := rs.ExecuteRegisterCmd("r1")
	if err != nil || val != "95" {
		t.Fatalf("read failed: %v, %s", err, val)
	}

	// Test ring buffer overflow: 16 calls fill r0..r15, call 17 overwrites r0 
	rs.Reset()
	for i := 0; i < NumRegisters; i++ {
		tag, bound, _ := rs.AutoBind(CalcVal{Kind: ValKindRat, Rat: big.NewRat(int64(100+i), 1)})
		expectedTag := fmt.Sprintf("<|r%d|>", i)
		if tag != expectedTag {
			t.Fatalf("AutoBind %d: expected %s, got %s", i, expectedTag, tag)
		}
		expectedBound := fmt.Sprintf("%s = %d", expectedTag, 100+i)
		if bound != expectedBound {
			t.Fatalf("AutoBind %d bound mismatch: got %q, expected %q", i, bound, expectedBound)
		}
	}
	// Call 17: overwrites r0, announces replacement: "<|r0|> = 999 (replaced 100)"
	tag17, bound17, _ := rs.AutoBind(CalcVal{Kind: ValKindRat, Rat: big.NewRat(999, 1)})
	if tag17 != "<|r0|>" || bound17 != "<|r0|> = 999 (replaced 100)" {
		t.Fatalf("Ring buffer overflow failed: expected <|r0|> = 999 (replaced 100), got %s", bound17)
	}
	// Verify evaluating with store uses the new r0 value
	cval, err := EvaluateInfixToVal("<|r0|> + 1", rs)
	if err != nil || cval.String() != "1000" {
		t.Fatalf("Evaluating with store failed after overflow: expected 1000, got %v (err: %v)", cval, err)
	}
}

func BenchmarkEvaluateInfix(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_, _ = EvaluateInfix("482 * 19 - 14 + sqrt(144) * (2^3)")
	}
}

func BenchmarkSolveEquation(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_, _ = SolveEquation("5*x - 1 = x + 3, x")
	}
}

func BenchmarkStr(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_, _ = EvaluateStr(`reverse("hello world")`)
	}
}

func BenchmarkDate(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_, _ = EvaluateDate(`add("2024-03-14", 7, "day")`)
	}
}

func BenchmarkRegisterOps(b *testing.B) {
	store := NewRegisterStore()
	v := CalcVal{Kind: ValKindRat, Rat: big.NewRat(42, 1)}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		store.AutoBind(v)
	}
}

func TestUnitAmbiguityAndDisambiguation(t *testing.T) {
	rs := NewRegisterStore()

	// 5 m * t must fail with ErrAmbiguous
	_, err := EvaluateInfixWithStore("5 m * t", rs)
	if err == nil || !strings.Contains(err.Error(), "ErrAmbiguous") {
		t.Errorf("expected ErrAmbiguous for '5 m * t', got err: %v", err)
	}

	// Binding speed of light to <|r0|> then <|r0|> * t must fail with ErrAmbiguous
	_, err = EvaluateLookupWithStore(`lookup("c")`, rs)
	if err != nil {
		t.Fatalf("lookup c failed: %v", err)
	}
	_, err = EvaluateInfixWithStore("<|r0|> * t", rs)
	if err == nil || !strings.Contains(err.Error(), "ErrAmbiguous") {
		t.Errorf("expected ErrAmbiguous for '<|r0|> * t', got err: %v", err)
	}

	// (2 T) * (3 * m^2) must evaluate to 6 Wb
	resFlux, err := EvaluateInfixWithStore("(2 T) * (3 * m^2)", rs)
	if err != nil {
		t.Errorf("expected 6 Wb for '(2 T) * (3 * m^2)', got err: %v", err)
	} else if !strings.Contains(resFlux, "6 Wb") {
		t.Errorf("expected 6 Wb, got %q", resFlux)
	}

	// Pure algebra must remain polynomials
	resAlg, err := EvaluateInfixWithStore("5*m + 20", rs)
	if err != nil || !strings.Contains(resAlg, "5*m + 20") {
		t.Errorf("expected 5*m + 20, got res=%q err=%v", resAlg, err)
	}

	resAlg2, err := EvaluateInfixWithStore("k/2 - k/3", rs)
	if err != nil || !strings.Contains(resAlg2, "1/6*k") {
		t.Errorf("expected 1/6*k, got res=%q err=%v", resAlg2, err)
	}

	resAlg3, err := EvaluateInfixWithStore("4*A - 5*A", rs)
	if err != nil || !strings.Contains(resAlg3, "-A") {
		t.Errorf("expected -A, got res=%q err=%v", resAlg3, err)
	}
}

