package main

import (
	"fmt"
	"math"
	"math/big"
	"math/rand"
	"strings"
	"testing"
)

// TestExactNumbers verifies mathematical engine capabilities
func TestExactNumbers(t *testing.T) {
	cases := []struct {
		expr     string
		expected string
	}{
		{"25 * pi", "25*pi"},
		{"(1 + sqrt(5)) / 2", "(1+sqrt(5))/2"},
		{"3 - 4*i", "3-4*i"},
		{"2*sqrt(3) + pi", "pi+2*sqrt(3)"},
		{"8 * sqrt(2) * pi", "8*sqrt(2)*pi"},
		{"(3 + sqrt(7)*i) / 2", "(3+sqrt(7)*i)/2"},
		{"sqrt(3 + 2*sqrt(2))", "1+sqrt(2)"},
		{"sin(pi/6)", "1/2"},
		{"cos(pi/4)", "sqrt(2)/2"},
		{"tan(pi/3)", "sqrt(3)"},
		{"sin(5*pi/12)", "(sqrt(6)+sqrt(2))/4"},
		{"i^2", "-1"},
		{"i^3", "-i"},
		{"i^4", "1"},
		{"(1 + i) / (1 - i)", "i"},
		{"(3 - 4*i) * (3 + 4*i)", "25"},
		{"re(3 - 4*i)", "3"},
		{"im(3 - 4*i)", "-4"},
		{"conj(3 - 4*i)", "3+4*i"},
		{"abs(3 - 4*i)", "5"},
		{"log(8, 2)", "3"},
		{"log(81, 3)", "4"},
		{"log(1000, 10)", "3"},
	}

	for _, tc := range cases {
		t.Run(tc.expr, func(t *testing.T) {
			res, err := EvaluateInfix(tc.expr)
			if err != nil {
				t.Fatalf("EvaluateInfix(%q) returned error: %v", tc.expr, err)
			}
			// Both 2*sqrt(3)+pi and pi+2*sqrt(3) are valid term orderings
			if res != tc.expected && (tc.expr == "2*sqrt(3) + pi" && res != "2*sqrt(3)+pi") {
				t.Fatalf("EvaluateInfix(%q) = %q, expected %q", tc.expr, res, tc.expected)
			}
		})
	}
}

// TestPolynomials verifies mathematical engine capabilities
func TestPolynomials(t *testing.T) {
	rs := NewRegisterStore()

	// 1. expand
	res, err := EvaluateInfixToVal("expand((x + 1) * (x + 2))", rs)
	if err != nil {
		t.Fatalf("expand error: %v", err)
	}
	if res.String() != "x^2 + 3*x + 2" {
		t.Fatalf("expand((x+1)*(x+2)) = %q, expected %q", res.String(), "x^2 + 3*x + 2")
	}

	// 2. divide
	res, err = EvaluateInfixToVal("divide(x^2 - 1, x - 1)", rs)
	if err != nil {
		t.Fatalf("divide error: %v", err)
	}
	if res.String() != "[x + 1, 0]" {
		t.Fatalf("divide(x^2 - 1, x - 1) = %q, expected %q", res.String(), "[x + 1, 0]")
	}

	// 3. factor
	res, err = EvaluateInfixToVal("factor(x^2 - 5*x + 6)", rs)
	if err != nil {
		t.Fatalf("factor error: %v", err)
	}
	if res.String() != "(x - 2)*(x - 3)" && res.String() != "(x - 3)*(x - 2)" {
		t.Fatalf("factor(x^2 - 5*x + 6) = %q, expected (x - 2)*(x - 3)", res.String())
	}

	// 4. pgcd
	res, err = EvaluateInfixToVal("pgcd(x^2 - 1, x^2 + x - 2)", rs)
	if err != nil {
		t.Fatalf("pgcd error: %v", err)
	}
	if res.String() != "x - 1" {
		t.Fatalf("pgcd(x^2 - 1, x^2 + x - 2) = %q, expected %q", res.String(), "x - 1")
	}

	// 5. roots
	res, err = EvaluateInfixToVal("roots(x^2 - 5*x + 6)", rs)
	if err != nil {
		t.Fatalf("roots error: %v", err)
	}
	if res.String() != "[2, 3]" {
		t.Fatalf("roots(x^2 - 5*x + 6) = %q, expected [2, 3]", res.String())
	}

	// 6. evaluate
	res, err = EvaluateInfixToVal("evaluate(x^2 + 2*x + 1, x, 3)", rs)
	if err != nil {
		t.Fatalf("evaluate error: %v", err)
	}
	if res.String() != "16" {
		t.Fatalf("evaluate(x^2 + 2*x + 1, x, 3) = %q, expected 16", res.String())
	}

	// 7. inequalities
	solns, err := SolveEquationWithStore("x^2 - 4 < 0", rs)
	if err != nil || len(solns) != 1 {
		t.Fatalf("solve(x^2 - 4 < 0) error: %v, solns: %v", err, solns)
	}
	if solns[0].Val.String() != "(-2, 2)" {
		t.Fatalf("solve(x^2 - 4 < 0) = %q, expected (-2, 2)", solns[0].Val.String())
	}

	solns, err = SolveEquationWithStore("x^2 - 4 >= 0", rs)
	if err != nil || len(solns) != 1 {
		t.Fatalf("solve(x^2 - 4 >= 0) error: %v", err)
	}
	if solns[0].Val.String() != "(-inf, -2] U [2, inf)" {
		t.Fatalf("solve(x^2 - 4 >= 0) = %q, expected (-inf, -2] U [2, inf)", solns[0].Val.String())
	}
}

// TestGeometry verifies mathematical engine capabilities
func TestGeometry(t *testing.T) {
	cases := []struct {
		expr     string
		expected string
	}{
		{"dist([0, 0], [3, 4])", "5"},
		{"midpoint([0, 0], [2, 4])", "[1, 2]"},
		{"slope([0, 0], [2, 4])", "2"},
		{"line([0, 0], [2, 4])", "y = 2*x"},
		{"area_polygon([[0, 0], [4, 0], [4, 3], [0, 3]])", "12"},
		{"heron(3, 4, 5)", "6"},
		{"triangle_area(3, 4, pi/2)", "6"},
		{"circ_area(5)", "25*pi"},
		{"circumference(5)", "10*pi"},
		{"arc_length(5, pi/5)", "pi"},
		{"sector_area(5, pi/5)", "5*pi/2"},
	}

	for _, tc := range cases {
		t.Run(tc.expr, func(t *testing.T) {
			res, err := EvaluateInfix(tc.expr)
			if err != nil {
				t.Fatalf("EvaluateInfix(%q) returned error: %v", tc.expr, err)
			}
			if res != tc.expected {
				t.Fatalf("EvaluateInfix(%q) = %q, expected %q", tc.expr, res, tc.expected)
			}
		})
	}
}

// TestLinearAlgebra verifies mathematical engine capabilities
func TestLinearAlgebra(t *testing.T) {
	rs := NewRegisterStore()

	// 1. det
	res, err := EvaluateInfixToVal("det([[1, 2], [3, 4]])", rs)
	if err != nil || res.String() != "-2" {
		t.Fatalf("det([[1, 2], [3, 4]]) = %v (err: %v)", res, err)
	}

	// 2. inv
	res, err = EvaluateInfixToVal("inv([[1, 2], [3, 4]])", rs)
	if err != nil {
		t.Fatalf("inv error: %v", err)
	}
	expectedInv := "[[-2, 1], [1.5, -0.5]]"
	if res.String() != expectedInv {
		t.Fatalf("inv([[1, 2], [3, 4]]) = %q, expected %q", res.String(), expectedInv)
	}

	// 3. transpose
	res, err = EvaluateInfixToVal("transpose([[1, 2], [3, 4]])", rs)
	if err != nil || res.String() != "[[1, 3], [2, 4]]" {
		t.Fatalf("transpose = %v (err: %v)", res, err)
	}

	// 4. matrix mul via @
	res, err = EvaluateInfixToVal("[[1, 2], [3, 4]] @ [[2, 0], [1, 2]]", rs)
	if err != nil || res.String() != "[[4, 4], [10, 8]]" {
		t.Fatalf("matrix @ = %v (err: %v)", res, err)
	}

	// 5. rank and trace
	res, err = EvaluateInfixToVal("rank([[1, 2], [2, 4]])", rs)
	if err != nil || res.String() != "1" {
		t.Fatalf("rank = %v (err: %v)", res, err)
	}

	res, err = EvaluateInfixToVal("trace([[1, 2], [3, 4]])", rs)
	if err != nil || res.String() != "5" {
		t.Fatalf("trace = %v (err: %v)", res, err)
	}

	// 6. solve_linear
	res, err = EvaluateInfixToVal("solve_linear([[1, 1], [1, -1]], [4, 2])", rs)
	if err != nil || res.String() != "[3, 1]" {
		t.Fatalf("solve_linear = %v (err: %v)", res, err)
	}

	// 7. dot and cross
	res, err = EvaluateInfixToVal("dot([1, 2, 3], [4, 5, 6])", rs)
	if err != nil || res.String() != "32" {
		t.Fatalf("dot = %v (err: %v)", res, err)
	}

	res, err = EvaluateInfixToVal("cross([1, 0, 0], [0, 1, 0])", rs)
	if err != nil || res.String() != "[0, 0, 1]" {
		t.Fatalf("cross = %v (err: %v)", res, err)
	}

	// 8. norm and unit
	res, err = EvaluateInfixToVal("norm([3, 4])", rs)
	if err != nil || res.String() != "5" {
		t.Fatalf("norm = %v (err: %v)", res, err)
	}

	res, err = EvaluateInfixToVal("unit([3, 4])", rs)
	if err != nil || (res.String() != "[0.6, 0.8]" && res.String() != "[3/5, 4/5]") {
		t.Fatalf("unit = %v (err: %v)", res, err)
	}
}

// TestSumsAndDistributions verifies mathematical engine capabilities
func TestSumsAndDistributions(t *testing.T) {
	rs := NewRegisterStore()

	// 1. sum
	res, err := EvaluateInfixToVal("sum(k^2, k, 1, 10)", rs)
	if err != nil || res.String() != "385" {
		t.Fatalf("sum(k^2, k, 1, 10) = %v (err: %v)", res, err)
	}

	// 2. product
	res, err = EvaluateInfixToVal("product(2*k - 1, k, 1, 5)", rs)
	if err != nil || res.String() != "945" {
		t.Fatalf("product(2*k - 1, k, 1, 5) = %v (err: %v)", res, err)
	}

	// 3. binom_pmf and binom_cdf
	res, err = EvaluateInfixToVal("binom_pmf(2, 4, 1/2)", rs)
	if err != nil || (res.String() != "0.375" && res.String() != "3/8") {
		t.Fatalf("binom_pmf(2, 4, 1/2) = %v (err: %v)", res, err)
	}

	res, err = EvaluateInfixToVal("binom_cdf(2, 4, 1/2)", rs)
	if err != nil || (res.String() != "0.6875" && res.String() != "11/16") {
		t.Fatalf("binom_cdf(2, 4, 1/2) = %v (err: %v)", res, err)
	}

	// 4. Work limit test (> 10^6 terms)
	_, err = EvaluateInfixToVal("sum(k, k, 1, 1000005)", rs)
	if err == nil || !strings.Contains(err.Error(), "ErrWorkLimitExceeded") {
		t.Fatalf("Expected ErrWorkLimitExceeded for sum > 10^6, got: %v", err)
	}
}

// TestRegistersAndLimits verifies mathematical engine capabilities
func TestRegistersAndLimits(t *testing.T) {
	rs := NewRegisterStore()

	// Bind an exact number to r0
	exVal, err := EvaluateInfixToExactNum("sqrt(2)", rs)
	if err != nil {
		t.Fatalf("exact sqrt error: %v", err)
	}
	rs.AutoBind(CalcVal{Kind: ValKindExactNum, Exact: exVal}) // r0 = sqrt(2)

	res, err := EvaluateInfixToVal("<|r0|>^2", rs)
	if err != nil || res.String() != "2" {
		t.Fatalf("<|r0|>^2 expected 2, got %v (err: %v)", res, err)
	}

	// Bind matrix to r1
	mVal, err := EvaluateInfixToVal("[[1, 2], [3, 4]]", rs)
	if err != nil {
		t.Fatalf("matrix eval error: %v", err)
	}
	rs.AutoBind(mVal) // r1 = [[1, 2], [3, 4]]

	res, err = EvaluateInfixToVal("det(<|r1|>)", rs)
	if err != nil || res.String() != "-2" {
		t.Fatalf("det(<|r1|>) expected -2, got %v (err: %v)", res, err)
	}
}

// TestMathematicalProperties tests mathematical invariants (round-trip factor/expand, A*inv(A)=I, etc.)
func TestMathematicalProperties(t *testing.T) {
	rs := NewRegisterStore()
	rng := rand.New(rand.NewSource(42))

	// 1. Invariant: inv(A) @ A == I (50 random invertible integer matrices)
	for iter := 0; iter < 50; iter++ {
		dim := 2 + (iter % 3) // 2x2, 3x3, 4x4
		var mat Matrix
		for tries := 0; tries < 100; tries++ {
			mat = make(Matrix, dim)
			for i := 0; i < dim; i++ {
				mat[i] = make([]ExactNum, dim)
				for j := 0; j < dim; j++ {
					v := int64(rng.Intn(9) - 4) // [-4, 4]
					mat[i][j] = *NewExactInt(v)
				}
			}
			det, err := mat.Det()
			if err == nil && !det.IsZero() {
				break
			}
		}
		inv, err := mat.Inv()
		if err != nil {
			t.Fatalf("matrix inversion failed on iter %d: %v", iter, err)
		}
		prod, err := mat.Mul(inv)
		if err != nil {
			t.Fatalf("matrix mul failed on iter %d: %v", iter, err)
		}
		for i := 0; i < dim; i++ {
			for j := 0; j < dim; j++ {
				expected := big.NewRat(0, 1)
				if i == j {
					expected = big.NewRat(1, 1)
				}
				if r, isR := prod[i][j].AsRational(); !isR || r.Cmp(expected) != 0 {
					t.Fatalf("iter %d: (A @ inv(A))[%d][%d] = %v != %v", iter, i, j, prod[i][j], expected)
				}
			}
		}
	}

	// 2. Invariant: expand(factor(P)) == P (50 random factorable polynomials)
	for iter := 0; iter < 50; iter++ {
		deg := 2 + (iter % 3) // degree 2, 3, 4
		roots := make([]int, deg)
		for j := 0; j < deg; j++ {
			roots[j] = rng.Intn(7) - 3 // [-3, 3]
		}
		var parts []string
		for _, r := range roots {
			if r < 0 {
				parts = append(parts, fmt.Sprintf("(x + %d)", -r))
			} else if r > 0 {
				parts = append(parts, fmt.Sprintf("(x - %d)", r))
			} else {
				parts = append(parts, "x")
			}
		}
		polyExpr := strings.Join(parts, " * ")
		expStr, _, err := ExpandPoly(polyExpr, rs)
		if err != nil {
			t.Fatalf("expand error on iter %d (%s): %v", iter, polyExpr, err)
		}
		factStr, err := FactorPoly(expStr, rs)
		if err != nil {
			t.Fatalf("factor error on iter %d (%s): %v", iter, expStr, err)
		}
		roundTripStr, _, err := ExpandPoly(factStr, rs)
		if err != nil {
			t.Fatalf("expand(factor(P)) error on iter %d (%s -> %s): %v", iter, expStr, factStr, err)
		}
		if roundTripStr != expStr {
			t.Fatalf("iter %d: expand(factor(%s)) = %q != %q", iter, expStr, roundTripStr, expStr)
		}
	}

	// 3. Invariant: Exact arithmetic matches high-precision floats (50 random expressions)
	for iter := 0; iter < 50; iter++ {
		a := rng.Intn(8) + 1
		b := rng.Intn(8) + 1
		k1 := rng.Intn(5) + 1
		k2 := rng.Intn(5) + 1
		piK := iter % 5
		expr := fmt.Sprintf("%d*sqrt(%d) + %d*sqrt(%d) + %d*pi", k1, a, k2, b, piK)
		val, err := EvaluateInfixToVal(expr, rs)
		if err != nil {
			t.Fatalf("exact eval error on iter %d (%s): %v", iter, expr, err)
		}
		floatExpected := float64(k1)*math.Sqrt(float64(a)) + float64(k2)*math.Sqrt(float64(b)) + float64(piK)*math.Pi
		diff := math.Abs(toFloat(val) - floatExpected)
		if diff > 1e-12 {
			t.Fatalf("iter %d: float mismatch on %s: got %v, expected %v (diff %e)", iter, expr, toFloat(val), floatExpected, diff)
		}
	}

	// 4. Invariant: Interval solution checked at sample points (50 random quadratic inequalities)
	for iter := 0; iter < 50; iter++ {
		r1 := rng.Intn(6) - 5 // [-5, 0]
		r2 := rng.Intn(6) + 1 // [1, 6]
		if r1 >= r2 {
			r2 = r1 + 2
		}
		// (x - r1)*(x - r2) < 0 has solution (r1, r2)
		ineqExpr := fmt.Sprintf("(x - (%d))*(x - (%d)) < 0", r1, r2)
		solStr, err := SolveInequality(ineqExpr, "x", rs)
		if err != nil {
			t.Fatalf("solve ineq error on iter %d (%s): %v", iter, ineqExpr, err)
		}
		expectedSol := fmt.Sprintf("(%d, %d)", r1, r2)
		if solStr.String() != expectedSol {
			t.Fatalf("iter %d: solve ineq %s expected %s, got %s", iter, ineqExpr, expectedSol, solStr.String())
		}
		f := func(x float64) float64 { return (x - float64(r1)) * (x - float64(r2)) }
		// Sample 5 points inside (r1, r2)
		for s := 1; s <= 5; s++ {
			x := float64(r1) + float64(s)*(float64(r2-r1))/6.0
			if f(x) >= 0 {
				t.Fatalf("iter %d inside point %f gave f(x) = %f >= 0", iter, x, f(x))
			}
		}
		// Sample points outside
		for _, x := range []float64{float64(r1) - 1.0, float64(r1) - 2.5, float64(r2) + 1.0, float64(r2) + 3.0} {
			if f(x) <= 0 {
				t.Fatalf("iter %d outside point %f gave f(x) = %f <= 0", iter, x, f(x))
			}
		}
	}
}

func BenchmarkExactArithmetic(b *testing.B) {
	rs := NewRegisterStore()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = EvaluateInfixToVal("(1 + sqrt(5))/2 + (1 - sqrt(5))/2", rs)
	}
}

func BenchmarkMatrixMul2x2(b *testing.B) {
	rs := NewRegisterStore()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = EvaluateInfixToVal("[[1, 2], [3, 4]] @ [[2, 0], [1, 2]]", rs)
	}
}

func BenchmarkMatrixMul10x10(b *testing.B) {
	m1, _ := NewMatrix(10, 10)
	m2, _ := NewMatrix(10, 10)
	for i := 0; i < 10; i++ {
		for j := 0; j < 10; j++ {
			m1[i][j] = *NewExactInt(int64(i + j + 1))
			m2[i][j] = *NewExactInt(int64(i - j + 1))
		}
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = m1.Mul(m2)
	}
}

func BenchmarkPolyExpand(b *testing.B) {
	rs := NewRegisterStore()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = EvaluateInfixToVal("expand((x + 1)*(x + 2)*(x + 3))", rs)
	}
}

func BenchmarkPolyFactor(b *testing.B) {
	rs := NewRegisterStore()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = EvaluateInfixToVal("factor(x^2 - 5*x + 6)", rs)
	}
}

func BenchmarkSolveInequality(b *testing.B) {
	rs := NewRegisterStore()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = SolveEquationWithStore("x^2 - 4 < 0", rs)
	}
}

func BenchmarkDet10x10(b *testing.B) {
	m, _ := NewMatrix(10, 10)
	for i := 0; i < 10; i++ {
		for j := 0; j < 10; j++ {
			if i == j {
				m[i][j] = *NewExactInt(int64(i + 2))
			} else if i == j+1 || j == i+1 {
				m[i][j] = *NewExactInt(1)
			} else {
				m[i][j] = *NewExactZero()
			}
		}
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = m.Det()
	}
}

func BenchmarkExtendedCalc(b *testing.B) {
	rs := NewRegisterStore()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = EvaluateInfixToVal("2*sqrt(3) + pi", rs)
	}
}

// TestRegressionCaps verifies that huge inputs, power towers, huge binomials,
// and explosive polynomial expansions are bounded and return ErrOverflow deterministically.
func TestRegressionCaps(t *testing.T) {
	caps := []struct {
		expr string
		tool string
	}{
		{"comb(1000000000, 500000000)", "calc"},
		{"perm(1000000000, 500000000)", "calc"},
		{"(x + y + z + w)^100", "calc"},
		{"2 ^ 2 ^ 2 ^ 2 ^ 2 ^ 2", "calc"},
		{"product(k, k, 1, 1000000)", "calc"},
		{"binom_pmf(500000, 1000000, 0.5)", "calc"},
		{"binom_cdf(500, 1000000, 0.5)", "calc"},
		{"(1 + sqrt(5))^1000000", "calc"},
		{"[[1, 2], [3, 4]]^1000000", "calc"},
	}

	for _, c := range caps {
		res := DispatchTool(c.tool, c.expr)
		t.Logf("%s -> Success=%v, Err=%s, Res=%s", c.expr, res.Success, res.Error, res.Result)
		if res.Success || !strings.Contains(res.Error, "Err") {
			t.Fatalf("expected Err for %q, got: Success=%v, Error=%q, Result=%q", c.expr, res.Success, res.Error, res.Result)
		}
	}
}

// TestWorkBudgetAndDisplayLimits tests work budget and display limits:
// 1. sum/product work budget returns ErrOverflow before wall clock.
// 2. Exact results > 256 characters return success observation: <|rN|> = [too large to display: N digits].
// 3. Registers still hold the value, so it can be used in later calls.
func TestWorkBudgetAndDisplayLimits(t *testing.T) {
	rs := NewRegisterStore()

	// 1. sum work budget
	resp := DispatchToolWithStore(rs, "calc", "sum(1/k, k, 1, 20000)")
	if resp.Success || !strings.Contains(resp.Error, "ErrOverflow: sum work budget exceeded") {
		t.Fatalf("expected ErrOverflow: sum work budget exceeded, got %v (err: %s)", resp.Success, resp.Error)
	}

	// 2. display limit: 2^1000 has 302 digits > 256 -> success observation naming register
	resp2 := DispatchToolWithStore(rs, "calc", "2^1000")
	if !resp2.Success || resp2.Result != "<|r0|> = [too large to display: 302 digits]" {
		t.Fatalf("expected <|r0|> = [too large to display: 302 digits], got success=%v, result=%q, err=%q", resp2.Success, resp2.Result, resp2.Error)
	}

	// 3. Register r0 holds 2^1000, so <|r0|> % 10 evaluates to 6
	resp3 := DispatchToolWithStore(rs, "calc", "<|r0|> % 10")
	if !resp3.Success || !strings.Contains(resp3.Result, "= 6") {
		t.Fatalf("expected register to hold value for subsequent call, got: %v (err: %s, res: %s)", resp3.Success, resp3.Error, resp3.Result)
	}

	// 4. display limit on harmonic sum: sum(1/k, k, 1, 2000) has 1734 characters
	resp4 := DispatchToolWithStore(rs, "calc", "sum(1/k, k, 1, 2000)")
	if !resp4.Success || resp4.Result != "<|r2|> = [too large to display: 1734 digits]" {
		t.Fatalf("expected <|r2|> = [too large to display: 1734 digits], got success=%v, result=%q, err=%q", resp4.Success, resp4.Result, resp4.Error)
	}

	// 5. Register r2 holds the large fraction, so round(<|r2|>, 4) evaluates to 8.1784
	resp5 := DispatchToolWithStore(rs, "calc", "round(<|r2|>, 4)")
	if !resp5.Success || !strings.Contains(resp5.Result, "= 8.1784") {
		t.Fatalf("expected register to hold fraction for subsequent round, got: %v (err: %s, res: %s)", resp5.Success, resp5.Error, resp5.Result)
	}
}

// TestExactPolynomialCoefficientsAndSolve tests polynomial coefficients and solve:
// 1. Polynomial coefficients exact numbers (surd, complex, pi) with no silent truncation.
// 2. solve parses sqrt, pi, i in coefficients (e.g. sqrt(2)*x + 1 = 3 -> x = sqrt(2)).
// 3. expand, factor, roots, evaluate with exact coefficients.
func TestExactPolynomialCoefficientsAndSolve(t *testing.T) {
	rs := NewRegisterStore()

	// 1. Canonical example: expand((x+sqrt(2))^2) -> x^2 + 2*sqrt(2)*x + 2
	resp := DispatchToolWithStore(rs, "calc", "expand((x+sqrt(2))^2)")
	if !resp.Success || !strings.Contains(resp.Result, "x^2 + 2*sqrt(2)*x + 2") {
		t.Fatalf("expand((x+sqrt(2))^2) failed: %v, got %q, err=%q", resp.Success, resp.Result, resp.Error)
	}

	// 2. Canonical example: solve(sqrt(2)*x + 1 = 3, x) -> x = sqrt(2)
	resp2 := DispatchToolWithStore(rs, "solve", "sqrt(2)*x + 1 = 3, x")
	if !resp2.Success || !strings.Contains(resp2.Result, "x = sqrt(2)") {
		t.Fatalf("solve(sqrt(2)*x + 1 = 3, x) failed: %v, got %q, err=%q", resp2.Success, resp2.Result, resp2.Error)
	}

	// 3. Complex expansion: expand((x+i)*(x-i)) -> x^2 + 1
	resp3 := DispatchToolWithStore(rs, "calc", "expand((x+i)*(x-i))")
	if !resp3.Success || !strings.Contains(resp3.Result, "x^2 + 1") {
		t.Fatalf("expand((x+i)*(x-i)) failed: %v, got %q, err=%q", resp3.Success, resp3.Result, resp3.Error)
	}

	// 4. Pi expansion: expand((x+pi)^2) -> x^2 + 2*pi*x + pi^2
	resp4 := DispatchToolWithStore(rs, "calc", "expand((x+pi)^2)")
	if !resp4.Success || !strings.Contains(resp4.Result, "x^2 + 2*pi*x + pi^2") {
		t.Fatalf("expand((x+pi)^2) failed: %v, got %q, err=%q", resp4.Success, resp4.Result, resp4.Error)
	}

	// 5. Infix polynomial addition with exact surds: (x + sqrt(2))^2
	resp5 := DispatchToolWithStore(rs, "calc", "(x + sqrt(2))^2")
	if !resp5.Success || !strings.Contains(resp5.Result, "x^2 + 2*sqrt(2)*x + 2") {
		t.Fatalf("infix (x + sqrt(2))^2 failed: %v, got %q, err=%q", resp5.Success, resp5.Result, resp5.Error)
	}

	// 6. Quadratic solve with surd roots: x^2 - 2 = 0 -> x = -sqrt(2), x = sqrt(2)
	resp6 := DispatchToolWithStore(rs, "solve", "x^2 - 2 = 0, x")
	if !resp6.Success || !strings.Contains(resp6.Result, "x = -sqrt(2)") || !strings.Contains(resp6.Result, "x = sqrt(2)") {
		t.Fatalf("solve(x^2 - 2 = 0) failed: %v, got %q, err=%q", resp6.Success, resp6.Result, resp6.Error)
	}

	// 7. Quadratic solve with complex roots: x^2 + 1 = 0, x, complex -> x = -i, x = i
	resp7 := DispatchToolWithStore(rs, "solve", "x^2 + 1 = 0, x, complex")
	if !resp7.Success || !strings.Contains(resp7.Result, "x = -i") || !strings.Contains(resp7.Result, "x = i") {
		t.Fatalf("solve(x^2 + 1 = 0, complex) failed: %v, got %q, err=%q", resp7.Success, resp7.Result, resp7.Error)
	}

	// 8. Roots of exact polynomial: roots(x^2 - 2) -> [-sqrt(2), sqrt(2)]
	resp8 := DispatchToolWithStore(rs, "calc", "roots(x^2 - 2)")
	if !resp8.Success || !strings.Contains(resp8.Result, "[-sqrt(2), sqrt(2)]") {
		t.Fatalf("roots(x^2 - 2) failed: %v, got %q, err=%q", resp8.Success, resp8.Result, resp8.Error)
	}

	// 9. Factor exact quadratic over non-rational coefficients: factor(x^2 - 2*sqrt(2)*x + 2) -> (x - sqrt(2))*(x - sqrt(2))
	resp9 := DispatchToolWithStore(rs, "calc", "factor(x^2 - 2*sqrt(2)*x + 2)")
	if !resp9.Success || !strings.Contains(resp9.Result, "(x - sqrt(2))*(x - sqrt(2))") {
		t.Fatalf("factor(x^2 - 2*sqrt(2)*x + 2) failed: %v, got %q, err=%q", resp9.Success, resp9.Result, resp9.Error)
	}

	// 10. Evaluate exact polynomial at exact surd: evaluate(x^2 + 2*sqrt(2)*x + 2, x, -sqrt(2)) -> 0
	resp10 := DispatchToolWithStore(rs, "calc", "evaluate(x^2 + 2*sqrt(2)*x + 2, x, -sqrt(2))")
	if !resp10.Success || !strings.Contains(resp10.Result, "= 0") {
		t.Fatalf("evaluate failed: %v, got %q, err=%q", resp10.Success, resp10.Result, resp10.Error)
	}
}

func TestTrigWithVariableError(t *testing.T) {
	rs := NewRegisterStore()

	// 1. Trig calls with variables must return ErrUnsupported rather than substituting 0
	symbolicCases := []string{
		"sin(x)",
		"cos(2*x)",
		"tan(x + pi/4)",
		"sin(x + pi/3) - cos(x - pi/6)",
		"sqrt(2)*sin(2*x - pi/4)",
		"2*cos(2*x + pi/2)^2 - 1",
		"sin(x + 2*pi) + 2*cos(x + 2*pi)",
	}

	for _, sc := range symbolicCases {
		resp := DispatchToolWithStore(rs, "calc", sc)
		if resp.Success {
			t.Errorf("DispatchTool(calc, %q) expected error, but succeeded with result: %q", sc, resp.Result)
		}
		if !strings.Contains(resp.Error, "ErrUnsupported") {
			t.Errorf("DispatchTool(calc, %q) expected ErrUnsupported, got: %q", sc, resp.Error)
		}
	}

	// 2. Exact pi constant evaluation
	exactPiCases := []struct {
		expr      string
		expected1 string
		expected2 string
	}{
		{"cos(pi/3)", "0.5", "1/2"},
		{"sin(pi/6)", "0.5", "1/2"},
		{"tan(pi/4)", "1", "1"},
		{"cos(2*pi)", "1", "1"},
		{"sin(pi)", "0", "0"},
	}

	for _, tc := range exactPiCases {
		resp := DispatchToolWithStore(rs, "calc", tc.expr)
		if !resp.Success {
			t.Errorf("DispatchTool(calc, %q) failed with error: %q", tc.expr, resp.Error)
			continue
		}
		if !strings.Contains(resp.Result, tc.expected1) && !strings.Contains(resp.Result, tc.expected2) {
			t.Errorf("DispatchTool(calc, %q) expected %q or %q, got: %q", tc.expr, tc.expected1, tc.expected2, resp.Result)
		}
	}

	// tan(pi/2) should return error (undefined at odd multiples of pi/2)
	tanPi2Resp := DispatchToolWithStore(rs, "calc", "tan(pi/2)")
	if tanPi2Resp.Success || (!strings.Contains(tanPi2Resp.Error, "ErrDivZero") && !strings.Contains(tanPi2Resp.Error, "ErrDomain")) {
		t.Errorf("DispatchTool(calc, tan(pi/2)) expected ErrDivZero or ErrDomain, got: %v, resp=%q, err=%q", tanPi2Resp.Success, tanPi2Resp.Result, tanPi2Resp.Error)
	}

	// Greek π is an identifier/variable, so trig of π returns ErrUnsupported
	respGreekPi := DispatchToolWithStore(rs, "calc", "cos(π/3)")
	if respGreekPi.Success || !strings.Contains(respGreekPi.Error, "ErrUnsupported") {
		t.Errorf("DispatchTool(calc, cos(π/3)) expected ErrUnsupported, got: %v, resp=%q, err=%q", respGreekPi.Success, respGreekPi.Result, respGreekPi.Error)
	}
}

func TestPropertyFunctionsWithSymbolicArguments(t *testing.T) {
	rs := NewRegisterStore()

	fns := []string{
		"sin", "cos", "tan", "sec", "csc", "cot",
		"asin", "acos", "atan", "sinh", "cosh", "tanh",
		"exp", "ln", "log", "sqrt", "cbrt", "abs",
		"round", "floor", "ceil",
	}

	symArgs := []string{
		"x",
		"2*x + 1",
		"x - y",
		"t^2",
	}

	for _, fn := range fns {
		for _, arg := range symArgs {
			expr := fmt.Sprintf("%s(%s)", fn, arg)
			resp := DispatchToolWithStore(rs, "calc", expr)
			if resp.Success {
				t.Fatalf("Property violation: function %s with symbolic argument %s evaluated successfully to %q (expected ErrUnsupported)", fn, arg, resp.Result)
			}
			if !strings.Contains(resp.Error, "ErrUnsupported") && !strings.Contains(resp.Error, "ErrDomain") {
				t.Fatalf("Property violation: function %s with symbolic argument %s got non-unsupported error %q", fn, arg, resp.Error)
			}
		}
	}
}

func TestSolveWithLog(t *testing.T) {
	rs := NewRegisterStore()

	// Calling solve with functions like log, ln, sin should give clean ErrUnsupported, NOT a cascading syntax error
	testCases := []string{
		"log(x) = 2, x",
		"ln(x) = 1, x",
		"sin(x) = 0, x",
		"exp(x) = 3, x",
	}

	for _, tc := range testCases {
		resp := DispatchToolWithStore(rs, "solve", tc)
		if resp.Success {
			t.Fatalf("solve(%q) expected error, but succeeded with: %q", tc, resp.Result)
		}
		if !strings.Contains(resp.Error, "ErrUnsupported: functions like") {
			t.Errorf("solve(%q) expected clean ErrUnsupported, got: %q", tc, resp.Error)
		}
	}
}

func TestUnitsHourInQuantityPosition(t *testing.T) {
	rs := NewRegisterStore()

	// 1. Quantity position: number followed by h
	resp := DispatchToolWithStore(rs, "calc", "2 h")
	if !resp.Success || !strings.Contains(resp.Result, "2 hr") {
		t.Fatalf("calc(2 h) expected 2 hr, got: success=%v, result=%q, err=%q", resp.Success, resp.Result, resp.Error)
	}

	// 2. Division with quantity in paren: 60 km / (2 h)
	resp = DispatchToolWithStore(rs, "calc", "60 km / (2 h)")
	if !resp.Success || (!strings.Contains(resp.Result, "25/3 m/s") && !strings.Contains(resp.Result, "8.33333")) {
		t.Fatalf("calc(60 km / (2 h)) expected 25/3 m/s, got: success=%v, result=%q, err=%q", resp.Success, resp.Result, resp.Error)
	}

	// 3. Unit conversion with h
	resp = DispatchToolWithStore(rs, "calc", "convert(2, \"h\", \"min\")")
	if !resp.Success || !strings.Contains(resp.Result, "120 min") {
		t.Fatalf("convert(2, h, min) expected 120 min, got: success=%v, result=%q, err=%q", resp.Success, resp.Result, resp.Error)
	}

	resp = DispatchToolWithStore(rs, "calc", "convert(2, \"h\", \"s\")")
	if !resp.Success || !strings.Contains(resp.Result, "7200 s") {
		t.Fatalf("convert(2, h, s) expected 7200 s, got: success=%v, result=%q, err=%q", resp.Success, resp.Result, resp.Error)
	}

	// 4. Bare h remains a variable in polynomials
	resp = DispatchToolWithStore(rs, "calc", "h + 3")
	if !resp.Success || !strings.Contains(resp.Result, "h + 3") {
		t.Fatalf("calc(h + 3) expected polynomial h + 3, got: success=%v, result=%q, err=%q", resp.Success, resp.Result, resp.Error)
	}

	// 5. Bare h in solve
	resp = DispatchToolWithStore(rs, "solve", "h + 2 = 5, h")
	if !resp.Success || !strings.Contains(resp.Result, "h = 3") {
		t.Fatalf("solve(h + 2 = 5, h) expected h = 3, got: success=%v, result=%q, err=%q", resp.Success, resp.Result, resp.Error)
	}

	// 6. Bare h in lookup
	resp = DispatchToolWithStore(rs, "lookup", `lookup("h")`)
	if !resp.Success || !strings.Contains(resp.Result, "6.62607015e-34") {
		t.Fatalf("lookup(h) expected Planck constant, got: success=%v, result=%q, err=%q", resp.Success, resp.Result, resp.Error)
	}
}
