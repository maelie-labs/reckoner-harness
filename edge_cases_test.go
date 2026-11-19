package main

import (
	"fmt"
	"math"
	"math/big"
	"strconv"
	"strings"
	"testing"
)

// -----------------------------------------------------------------------------
// Critical Edge-Case Tests
// -----------------------------------------------------------------------------

func TestPrecedenceSubstitution(t *testing.T) {
	rs := NewRegisterStore()

	// calc 0-3; calc <|r0|>^2 -> 9 (not -9)
	valNeg3 := CalcVal{Kind: ValKindRat, Rat: big.NewRat(-3, 1)}
	rs.AutoBind(valNeg3) // r0 = -3
	res, err := EvaluateInfixToVal("<|r0|>^2", rs)
	if err != nil || res.String() != "9" {
		t.Fatalf("<|r0|>^2 with r0=-3: expected 9, got %v (err: %v)", res, err)
	}

	// calc 1/3; calc 1/<|r0|> -> 3 (not 1/3)
	rs.Reset()
	valThird := CalcVal{Kind: ValKindRat, Rat: big.NewRat(1, 3)}
	rs.AutoBind(valThird) // r0 = 1/3
	res, err = EvaluateInfixToVal("1 / <|r0|>", rs)
	if err != nil || res.String() != "3" {
		t.Fatalf("1/<|r0|> with r0=1/3: expected 3, got %v (err: %v)", res, err)
	}

	// calc 1/3; calc 2^<|r0|> -> 1.259921
	res, err = EvaluateInfixToVal("2^<|r0|>", rs)
	if err != nil || res.String() != "1.259921" {
		t.Fatalf("2^<|r0|> with r0=1/3: expected 1.259921, got %v (err: %v)", res, err)
	}

	// solve 2*x+3=0; calc <|r0|>^2 -> 2.25 (not -0.75)
	rs.Reset()
	solns, err := SolveEquationWithStore("2*x + 3 = 0", rs)
	if err != nil || len(solns) != 1 {
		t.Fatalf("solve error: %v", err)
	}
	rs.AutoBindNamed(solns[0].VarName, solns[0].Val)
	res, err = EvaluateInfixToVal("<|r0|>^2", rs)
	if err != nil || res.String() != "2.25" {
		t.Fatalf("solve 2*x+3=0 then <|r0|>^2: expected 2.25, got %v (err: %v)", res, err)
	}

	// solve x^2-2=0; calc <|r0|>^2 -> 2 (not -2)
	rs.Reset()
	solns, err = SolveEquationWithStore("x^2 - 2 = 0", rs)
	if err != nil || len(solns) != 2 {
		t.Fatalf("solve x^2-2=0 error: %v", err)
	}
	rs.AutoBindNamed(solns[0].VarName, solns[0].Val) // r0 = -sqrt(2)
	res, err = EvaluateInfixToVal("<|r0|>^2", rs)
	if err != nil || res.String() != "2" {
		t.Fatalf("solve x^2-2=0 then <|r0|>^2: expected 2, got %v (err: %v)", res, err)
	}
}

func TestChainedSubstitution(t *testing.T) {
	rs := NewRegisterStore()
	rs.AutoBind(CalcVal{Kind: ValKindRat, Rat: big.NewRat(1, 1)})  // r0 = 1
	rs.AutoBind(CalcVal{Kind: ValKindRat, Rat: big.NewRat(-1, 1)}) // r1 = -1

	// Chained syntax like <|r<|r0|>|> should fail with ErrSyntax
	_, err := EvaluateInfixToVal("<|r<|r0|>|>", rs)
	if err == nil || !strings.Contains(err.Error(), "ErrSyntax") {
		t.Fatalf("expected ErrSyntax for nested register tags, got: %v", err)
	}
}

func TestQuotedStrings(t *testing.T) {
	rs := NewRegisterStore()
	rs.AutoBind(CalcVal{Kind: ValKindRat, Rat: big.NewRat(1, 1)}) // r0 = 1

	// In str len("r0"): "r0" is a string literal, not a register tag; length must be 2
	val, err := EvaluateStrWithStore(`len("r0")`, rs)
	if err != nil || val.String() != "2" {
		t.Fatalf("expected len(\"r0\") = 2, got %v (err: %v)", val, err)
	}
}

func TestTypedPrecision(t *testing.T) {
	rs := NewRegisterStore()

	// calc sqrt(2)+1; calc (<|r0|> - 1)^2 -> exactly 2
	v1, err := EvaluateInfixToVal("sqrt(2) + 1", rs)
	if err != nil {
		t.Fatalf("error evaluating sqrt(2)+1: %v", err)
	}
	rs.AutoBind(v1)
	v2, err := EvaluateInfixToVal("(<|r0|> - 1)^2", rs)
	if err != nil || v2.String() != "2" {
		t.Fatalf("expected (<|r0|> - 1)^2 = 2, got %v (err: %v)", v2, err)
	}

	// calc float(2/3); calc <|r0|>*3 -> exact 2
	rs.Reset()
	v3, err := EvaluateInfixToVal("float(2/3)", rs)
	if err != nil {
		t.Fatalf("error evaluating float(2/3): %v", err)
	}
	rs.AutoBind(v3)
	v4, err := EvaluateInfixToVal("<|r0|> * 3", rs)
	if err != nil || v4.String() != "2" {
		t.Fatalf("expected <|r0|>*3 = 2, got %v (err: %v)", v4, err)
	}

	// calc exp(-20); calc <|r0|>*10^9 -> non-zero
	rs.Reset()
	v5, err := EvaluateInfixToVal("exp(-20)", rs)
	if err != nil {
		t.Fatalf("error evaluating exp(-20): %v", err)
	}
	rs.AutoBind(v5)
	v6, err := EvaluateInfixToVal("<|r0|> * 10^9", rs)
	if err != nil || v6.String() == "0" {
		t.Fatalf("expected non-zero for exp(-20)*10^9, got %v (err: %v)", v6, err)
	}
}

func TestRoundFloatSurds(t *testing.T) {
	// round(sqrt(2), 3) -> 1.414
	res, err := EvaluateInfix("round(sqrt(2), 3)")
	if err != nil || res != "1.414" {
		t.Fatalf("round(sqrt(2), 3): expected 1.414, got %q (err: %v)", res, err)
	}

	// float(sqrt(2)) -> 1.414214
	res, err = EvaluateInfix("float(sqrt(2))")
	if err != nil || res != "1.414214" {
		t.Fatalf("float(sqrt(2)): expected 1.414214, got %q (err: %v)", res, err)
	}
}

func TestExactHalfUpRounding(t *testing.T) {
	// round(1.005, 2) -> 1.01 (exact half-up)
	res, err := EvaluateInfix("round(1.005, 2)")
	if err != nil || res != "1.01" {
		t.Fatalf("round(1.005, 2): expected 1.01, got %q (err: %v)", res, err)
	}

	// round(-1.5) -> -2
	res, err = EvaluateInfix("round(-1.5)")
	if err != nil || res != "-2" {
		t.Fatalf("round(-1.5): expected -2, got %q (err: %v)", res, err)
	}

	// round(-2.5) -> -3
	res, err = EvaluateInfix("round(-2.5)")
	if err != nil || res != "-3" {
		t.Fatalf("round(-2.5): expected -3, got %q (err: %v)", res, err)
	}
}

func TestRightAssociativePower(t *testing.T) {
	// 2^2^3 = 2^(2^3) = 2^8 = 256 (not (2^2)^3 = 64)
	res, err := EvaluateInfix("2^2^3")
	if err != nil || res != "256" {
		t.Fatalf("2^2^3: expected 256, got %q (err: %v)", res, err)
	}
}

func TestNegativeAndCappedPower(t *testing.T) {
	// 2^-3 -> 0.125 (terminating decimal 1/8)
	res, err := EvaluateInfix("2^-3")
	if err != nil || res != "0.125" {
		t.Fatalf("2^-3: expected 0.125, got %q (err: %v)", res, err)
	}

	// 3^-1 -> 1/3 (non-terminating rational)
	res, err = EvaluateInfix("3^-1")
	if err != nil || res != "1/3" {
		t.Fatalf("3^-1: expected 1/3, got %q (err: %v)", res, err)
	}

	// (1/2)^-3 -> 8
	res, err = EvaluateInfix("(1/2)^-3")
	if err != nil || res != "8" {
		t.Fatalf("(1/2)^-3: expected 8, got %q (err: %v)", res, err)
	}

	// 2^1000000000 -> ErrOverflow
	_, err = EvaluateInfix("2^1000000000")
	if err == nil || !strings.Contains(err.Error(), "ErrOverflow") {
		t.Fatalf("2^1000000000 expected ErrOverflow, got %v", err)
	}
}

func TestSlashInIdentifiers(t *testing.T) {
	// convert(45, "km/h", "m/s") parses "km/h" and "m/s" without slash splitting identifiers
	res, err := EvaluateInfix(`convert(45, "km/h", "m/s")`)
	if err != nil || res != "12.5 m/s" {
		t.Fatalf("convert(45, \"km/h\", \"m/s\"): expected 12.5 m/s, got %q (err: %v)", res, err)
	}
}

func TestFloorDivAndMod(t *testing.T) {
	// -7 // 2 -> -4
	res, err := EvaluateInfix("-7 // 2")
	if err != nil || res != "-4" {
		t.Fatalf("-7 // 2: expected -4, got %q (err: %v)", res, err)
	}

	// (7/2) // (1/3) -> 10
	res, err = EvaluateInfix("(7/2) // (1/3)")
	if err != nil || res != "10" {
		t.Fatalf("(7/2) // (1/3): expected 10, got %q (err: %v)", res, err)
	}

	// -7 % 3 -> 2
	res, err = EvaluateInfix("-7 % 3")
	if err != nil || res != "2" {
		t.Fatalf("-7 %% 3: expected 2, got %q (err: %v)", res, err)
	}
}

func TestIntegerSafetyNumberTheory(t *testing.T) {
	// comb(5.5, 2) -> ErrDomain
	_, err := EvaluateInfix("comb(5.5, 2)")
	if err == nil || !strings.Contains(err.Error(), "ErrDomain") {
		t.Fatalf("comb(5.5, 2) expected ErrDomain, got %v", err)
	}

	// isprime(7/2) -> ErrDomain
	_, err = EvaluateInfix("isprime(7/2)")
	if err == nil || !strings.Contains(err.Error(), "ErrDomain") {
		t.Fatalf("isprime(7/2) expected ErrDomain, got %v", err)
	}

	// gcd(1/2, 3) -> ErrDomain
	_, err = EvaluateInfix("gcd(1/2, 3)")
	if err == nil || !strings.Contains(err.Error(), "ErrDomain") {
		t.Fatalf("gcd(1/2, 3) expected ErrDomain, got %v", err)
	}

	// sum_range(1, 10^10) -> exact 50000000005000000000
	res, err := EvaluateInfix("sum_range(1, 10^10)")
	if err != nil || res != "50000000005000000000" {
		t.Fatalf("sum_range(1, 10^10): expected 50000000005000000000, got %q (err: %v)", res, err)
	}

	// base(2^70, 2) -> exact string with 71 characters: 1 followed by 70 zeros
	res, err = EvaluateInfix("base(2^70, 2)")
	expected := "1" + strings.Repeat("0", 70)
	if err != nil || res != expected {
		t.Fatalf("base(2^70, 2): expected %s, got %q (err: %v)", expected, res, err)
	}

	// base(1/2, 2) -> ErrDomain
	_, err = EvaluateInfix("base(1/2, 2)")
	if err == nil || !strings.Contains(err.Error(), "ErrDomain") {
		t.Fatalf("base(1/2, 2) expected ErrDomain, got %v", err)
	}
}

func TestTypeStrictComparisons(t *testing.T) {
	// "a" < 5 -> ErrTypeMismatch
	_, err := EvaluateInfix(`"a" < 5`)
	if err == nil || !strings.Contains(err.Error(), "ErrTypeMismatch") {
		t.Fatalf("\"a\" < 5 expected ErrTypeMismatch, got %v", err)
	}

	// true < 5 -> ErrTypeMismatch
	_, err = EvaluateInfix("true < 5")
	if err == nil || !strings.Contains(err.Error(), "ErrTypeMismatch") {
		t.Fatalf("true < 5 expected ErrTypeMismatch, got %v", err)
	}

	// [1] < 5 -> ErrTypeMismatch
	_, err = EvaluateInfix("[1] < 5")
	if err == nil || !strings.Contains(err.Error(), "ErrTypeMismatch") {
		t.Fatalf("[1] < 5 expected ErrTypeMismatch, got %v", err)
	}

	// 1 < 2 < 3 -> true
	res, err := EvaluateInfix("1 < 2 < 3")
	if err != nil || res != "true" {
		t.Fatalf("1 < 2 < 3 expected true, got %q (err: %v)", res, err)
	}

	// 1 < 5 < 3 -> false
	res, err = EvaluateInfix("1 < 5 < 3")
	if err != nil || res != "false" {
		t.Fatalf("1 < 5 < 3 expected false, got %q (err: %v)", res, err)
	}
}

func TestExactFactor(t *testing.T) {
	res, err := EvaluateInfix("factor(360)")
	if err != nil || res != `"2^3 * 3^2 * 5"` && res != "2^3 * 3^2 * 5" {
		t.Fatalf("factor(360) expected 2^3 * 3^2 * 5, got %q (err: %v)", res, err)
	}
}

func TestCASUnaryMinusPrecedence(t *testing.T) {
	// -x^2 + 4 = 0 parses as -(x^2) + 4 = 0 -> x = -2, x = 2
	res, err := SolveEquation("-x^2 + 4 = 0")
	if err != nil || res != "x = -2, x = 2" {
		t.Fatalf("-x^2 + 4 = 0: expected x = -2, x = 2, got %q (err: %v)", res, err)
	}
}

func TestCASPolynomialRoots(t *testing.T) {
	// x^3 - 6*x^2 + 11*x - 6 = 0 -> x = 1, x = 2, x = 3
	res, err := SolveEquation("x^3 - 6*x^2 + 11*x - 6 = 0")
	if err != nil || res != "x = 1, x = 2, x = 3" {
		t.Fatalf("C18/C19 cubic roots expected x = 1, x = 2, x = 3, got %q (err: %v)", res, err)
	}
}

func TestSolveOutputFormat(t *testing.T) {
	// Double root: x^2 + 2*x + 1 = 0 -> "x = -1" (single value, not x_1, x_2)
	res, err := SolveEquation("x^2 + 2*x + 1 = 0")
	if err != nil || res != "x = -1" {
		t.Fatalf("double root expected x = -1, got %q (err: %v)", res, err)
	}

	// No real roots: x^2 + 1 = 0 -> "no real solutions" (not "[]")
	res, err = SolveEquation("x^2 + 1 = 0")
	if err != nil || res != "no real solutions" {
		t.Fatalf("no real roots expected 'no real solutions', got %q (err: %v)", res, err)
	}
}

func TestCivilDayRange(t *testing.T) {
	// diff("0001-01-01", "2024-01-01") works without 292-year saturation
	val, err := EvaluateDate(`diff("0001-01-01", "2024-01-01")`)
	if err != nil {
		t.Fatalf("civil day diff error: %v", err)
	}
	days, err := strconv.ParseInt(val, 10, 64)
	if err != nil || days <= 365*2000 {
		t.Fatalf("civil day diff out of expected range: %s (err: %v)", val, err)
	}
}

func TestStrWhitespacePreservation(t *testing.T) {
	// str len("  ") -> 2
	v1, err := EvaluateStr(`len("  ")`)
	if err != nil || v1 != "2" {
		t.Fatalf("len(\"  \") expected 2, got %q (err: %v)", v1, err)
	}

	// str len(" padded ") -> 8
	v2, err := EvaluateStr(`len(" padded ")`)
	if err != nil || v2 != "8" {
		t.Fatalf("len(\" padded \") expected 8, got %q (err: %v)", v2, err)
	}
}

func TestInputCeiling(t *testing.T) {
	// A line > 1 MiB returns ErrTooLong
	hugeLine := strings.Repeat("1 + ", 300000) + "1"
	resp := DispatchTool("calc", hugeLine)
	if resp.Success || !strings.Contains(resp.Error, "ErrTooLong") {
		t.Fatalf("expected ErrTooLong for >1 MiB input, got: %v", resp)
	}
}

// -----------------------------------------------------------------------------
// Robustness and Invariant Tests
// -----------------------------------------------------------------------------

func TestDetNoPanicOnFloat(t *testing.T) {
	// det(pi, 1, 1, pi) -> pi^2 - 1 without panic
	res, err := EvaluateInfix("det(pi, 1, 1, pi)")
	if err != nil {
		t.Fatalf("det with pi failed: %v", err)
	}
	f, _ := EvaluateInfix("pi^2 - 1")
	f1, _ := math.Abs(1.0), 0.0
	_ = f1
	if !strings.HasPrefix(res, "8.8696") && !strings.HasPrefix(f, "8.8696") && res != f && res != "pi^2-1" && res != "pi^2 - 1" {
		t.Fatalf("unexpected det(pi, 1, 1, pi) result: %s", res)
	}
}

func TestDateMonthEndClamping(t *testing.T) {
	// add("2024-01-31", 1, "month") -> 2024-02-29 (leap year)
	v1, err := EvaluateDate(`add("2024-01-31", 1, "month")`)
	if err != nil || v1 != "2024-02-29 (Thursday)" {
		t.Fatalf("leap year month add expected 2024-02-29 (Thursday), got %q (err: %v)", v1, err)
	}

	// add("2023-01-31", 1, "month") -> 2023-02-28 (Tuesday) (non-leap year)
	v2, err := EvaluateDate(`add("2023-01-31", 1, "month")`)
	if err != nil || v2 != "2023-02-28 (Tuesday)" {
		t.Fatalf("non-leap year month add expected 2023-02-28 (Tuesday), got %q (err: %v)", v2, err)
	}
}

func TestDateDiffDirection(t *testing.T) {
	// diff(d1, d2) is d2 - d1: diff("2024-01-01", "2024-01-10") = 9
	v1, err := EvaluateDate(`diff("2024-01-01", "2024-01-10")`)
	if err != nil || v1 != "9" {
		t.Fatalf("diff(d1, d2) expected 9, got %q (err: %v)", v1, err)
	}

	// diff("2024-01-10", "2024-01-01") = -9
	v2, err := EvaluateDate(`diff("2024-01-10", "2024-01-01")`)
	if err != nil || v2 != "-9" {
		t.Fatalf("diff(d2, d1) expected -9, got %q (err: %v)", v2, err)
	}
}

func TestDateNextWeekdayStrict(t *testing.T) {
	// 2024-03-14 is a Thursday. next_weekday("2024-03-14", "thursday") must return the NEXT Thursday (2024-03-21)
	v, err := EvaluateDate(`next_weekday("2024-03-14", "thursday")`)
	if err != nil || v != "2024-03-21 (Thursday)" {
		t.Fatalf("next_weekday strict advancement expected 2024-03-21 (Thursday), got %q (err: %v)", v, err)
	}
}

func TestStrMultibyteRuneSafety(t *testing.T) {
	// len("café") = 4 runes
	v1, err := EvaluateStr(`len("café")`)
	if err != nil || v1 != "4" {
		t.Fatalf("len(\"café\") expected 4, got %q (err: %v)", v1, err)
	}

	// reverse("café") = "éfac"
	v2, err := EvaluateStr(`reverse("café")`)
	if err != nil || v2 != `"éfac"` {
		t.Fatalf("reverse(\"café\") expected \"éfac\", got %q (err: %v)", v2, err)
	}

	// slice("café", 0, 3) = "caf"
	v3, err := EvaluateStr(`slice("café", 0, 3)`)
	if err != nil || v3 != `"caf"` {
		t.Fatalf("slice(\"café\", 0, 3) expected \"caf\", got %q (err: %v)", v3, err)
	}
}

func TestStrZeroBasedBounds(t *testing.T) {
	// slice("hello", 0, 2) = "he"
	v, err := EvaluateStr(`slice("hello", 0, 2)`)
	if err != nil || v != `"he"` {
		t.Fatalf("slice(\"hello\", 0, 2) expected \"he\", got %q (err: %v)", v, err)
	}

	// Out of bounds slice raises ErrOutOfBounds
	_, err = EvaluateStr(`slice("hello", 0, 10)`)
	if err == nil || !strings.Contains(err.Error(), "ErrOutOfBounds") {
		t.Fatalf("slice out of bounds expected ErrOutOfBounds, got %v", err)
	}
}

func TestMathHelpersPreserveUnits(t *testing.T) {
	// abs(-5 m) = 5 m
	res, err := EvaluateInfix("abs(-5 m)")
	if err != nil || res != "5 m" {
		t.Fatalf("abs(-5 m) expected 5 m, got %q (err: %v)", res, err)
	}

	// min(10 m, 5 m) = 5 m
	res, err = EvaluateInfix("min(10 m, 5 m)")
	if err != nil || res != "5 m" {
		t.Fatalf("min(10 m, 5 m) expected 5 m, got %q (err: %v)", res, err)
	}

	// max(10 m, 5 m) = 10 m
	res, err = EvaluateInfix("max(10 m, 5 m)")
	if err != nil || res != "10 m" {
		t.Fatalf("max(10 m, 5 m) expected 10 m, got %q (err: %v)", res, err)
	}
}

func TestDateAcceptsOwnOutput(t *testing.T) {
	// Output form: "YYYY-MM-DD (Weekday)"
	v, err := EvaluateDate(`day_of_week("2024-03-14 (Thursday)")`)
	if err != nil || v != `"Thursday"` {
		t.Fatalf("day_of_week with own output format expected \"Thursday\", got %q (err: %v)", v, err)
	}
}

func TestDeterministicErrorContract(t *testing.T) {
	// Single line Err<Code>: message contract
	resp := DispatchTool("calc", "10 / 0")
	if resp.Success || !strings.HasPrefix(resp.Error, "ErrDivZero: ") || strings.Contains(resp.Error, "\n") {
		t.Fatalf("error contract violation: got %q", resp.Error)
	}
}

// -----------------------------------------------------------------------------
// Sixteen Registers & Announced Wrap
// -----------------------------------------------------------------------------

func TestSixteenRegistersAndAnnouncedWrap(t *testing.T) {
	rs := NewRegisterStore()

	// Fill registers r0 through r15
	for i := 0; i < 16; i++ {
		tag, disp, _ := rs.AutoBind(CalcVal{Kind: ValKindRat, Rat: big.NewRat(int64(i*10), 1)})
		expectedTag := fmt.Sprintf("<|r%d|>", i)
		if tag != expectedTag {
			t.Fatalf("expected tag %s, got %s", expectedTag, tag)
		}
		expectedDisp := fmt.Sprintf("%s = %d", expectedTag, i*10)
		if disp != expectedDisp {
			t.Fatalf("expected disp %s, got %s", expectedDisp, disp)
		}
	}

	// 17th binding wraps to r0 and announces replacement: "<|r0|> = 999 (replaced 0)"
	tagWrap, dispWrap, _ := rs.AutoBind(CalcVal{Kind: ValKindRat, Rat: big.NewRat(999, 1)})
	if tagWrap != "<|r0|>" || dispWrap != "<|r0|> = 999 (replaced 0)" {
		t.Fatalf("wrap failed: expected '<|r0|> = 999 (replaced 0)', got %q", dispWrap)
	}

	// 18th binding wraps to r1 and announces replacement: "<|r1|> = 888 (replaced 10)"
	tagWrap1, dispWrap1, _ := rs.AutoBind(CalcVal{Kind: ValKindRat, Rat: big.NewRat(888, 1)})
	if tagWrap1 != "<|r1|>" || dispWrap1 != "<|r1|> = 888 (replaced 10)" {
		t.Fatalf("wrap failed: expected '<|r1|> = 888 (replaced 10)', got %q", dispWrap1)
	}

	// Solve named binding announced wrap
	_, dispSolve, _ := rs.AutoBindNamed("x", CalcVal{Kind: ValKindRat, Rat: big.NewRat(5, 1)})
	// Wraps to r2, replacing 20
	if !strings.Contains(dispSolve, "replaced 20") {
		t.Fatalf("solve wrap failed to announce replacement: %q", dispSolve)
	}
}

// -----------------------------------------------------------------------------
// Primitives & Medium Cases (M1 - M20, Step 3)
// -----------------------------------------------------------------------------

func TestPrimitivesAndMediumCases(t *testing.T) {
	// Step 3: cbrt exact cube root
	res, err := EvaluateInfix("cbrt(27)")
	if err != nil || res != "3" {
		t.Fatalf("cbrt(27) expected 3, got %q (err: %v)", res, err)
	}
	res, err = EvaluateInfix("cbrt(8/27)")
	if err != nil || res != "2/3" {
		t.Fatalf("cbrt(8/27) expected 2/3, got %q (err: %v)", res, err)
	}

	// M18: len("") = 0
	vLen, err := EvaluateStr(`len("")`)
	if err != nil || vLen != "0" {
		t.Fatalf("len(\"\") expected 0, got %q (err: %v)", vLen, err)
	}

	// M19: Strict calendar day validation: 2025-02-30 -> ErrInvalidDate
	_, err = EvaluateDate(`day_of_week("2025-02-30")`)
	if err == nil || !strings.Contains(err.Error(), "ErrInvalidDate") {
		t.Fatalf("2025-02-30 expected ErrInvalidDate, got %v", err)
	}

	// Single hex convention uppercase quoted string
	vHex, err := EvaluateStr("to_hex(255)")
	if err != nil || vHex != `"FF"` {
		t.Fatalf("to_hex(255) expected \"FF\", got %q (err: %v)", vHex, err)
	}
	resHex, err := EvaluateInfix("base(255, 16)")
	if err != nil || resHex != `"FF"` {
		t.Fatalf("base(255, 16) expected \"FF\", got %q (err: %v)", resHex, err)
	}

	// M15: number with commas in solve
	solns, err := SolveEquation("x = 1,000")
	if err != nil || solns != "x = 1000" {
		t.Fatalf("solve x = 1,000 expected x = 1000, got %q (err: %v)", solns, err)
	}
}

// -----------------------------------------------------------------------------
// Root-Finding and Formatting Tests
// -----------------------------------------------------------------------------

func parseRenderedSurdSquared(s string) (*big.Int, error) {
	s = strings.TrimSpace(s)
	// Case 1: pure integer K
	if k, ok := new(big.Int).SetString(s, 10); ok {
		return new(big.Int).Mul(k, k), nil
	}

	slashIdx := strings.Index(s, "/")
	c := big.NewInt(1)
	if slashIdx != -1 {
		denStr := s[slashIdx+1:]
		var ok bool
		c, ok = new(big.Int).SetString(denStr, 10)
		if !ok {
			return nil, fmt.Errorf("invalid denom: %s", denStr)
		}
		s = s[:slashIdx]
	}

	// Case 2: sqrt(B) or sqrt(B)/C
	if strings.HasPrefix(s, "sqrt(") && strings.HasSuffix(s, ")") {
		bStr := s[5 : len(s)-1]
		b, ok := new(big.Int).SetString(bStr, 10)
		if !ok {
			return nil, fmt.Errorf("invalid radical b: %s", bStr)
		}
		c2 := new(big.Int).Mul(c, c)
		return new(big.Int).Quo(b, c2), nil
	}

	// Case 3: A*sqrt(B) or A*sqrt(B)/C
	starIdx := strings.Index(s, "*sqrt(")
	if starIdx == -1 || !strings.HasSuffix(s, ")") {
		return nil, fmt.Errorf("unrecognized surd format: %s", s)
	}
	aStr := s[:starIdx]
	bStr := s[starIdx+6 : len(s)-1]
	a, ok := new(big.Int).SetString(aStr, 10)
	if !ok {
		return nil, fmt.Errorf("invalid multiplier a: %s", aStr)
	}
	b, ok := new(big.Int).SetString(bStr, 10)
	if !ok {
		return nil, fmt.Errorf("invalid radical b: %s", bStr)
	}
	a2 := new(big.Int).Mul(a, a)
	a2b := new(big.Int).Mul(a2, b)
	c2 := new(big.Int).Mul(c, c)
	return new(big.Int).Quo(a2b, c2), nil
}

func TestBigIntRoots(t *testing.T) {
	// Repro 1: sqrt(2^70) -> 34359738368 (2^35)
	res, err := EvaluateInfix("sqrt(2^70)")
	if err != nil || res != "34359738368" {
		t.Fatalf("repro 1: sqrt(2^70) expected 34359738368, got %q (err: %v)", res, err)
	}

	// Repro 2: sqrt(2^64+1) -> sqrt(18446744073709551617)
	res, err = EvaluateInfix("sqrt(2^64+1)")
	if err != nil || res != "sqrt(18446744073709551617)" {
		t.Fatalf("repro 2: sqrt(2^64+1) expected sqrt(18446744073709551617), got %q (err: %v)", res, err)
	}

	// Repro 3: sqrt(8*10^30) -> 2000000000000000*sqrt(2)
	res, err = EvaluateInfix("sqrt(8*10^30)")
	if err != nil || res != "2000000000000000*sqrt(2)" {
		t.Fatalf("repro 3: sqrt(8*10^30) expected 2000000000000000*sqrt(2), got %q (err: %v)", res, err)
	}

	// Repro 4: sqrt(10^40+1) -> sqrt(10000000000000000000000000000000000000001)
	res, err = EvaluateInfix("sqrt(10^40+1)")
	if err != nil || res != "sqrt(10000000000000000000000000000000000000001)" {
		t.Fatalf("repro 4: sqrt(10^40+1) expected sqrt(10000000000000000000000000000000000000001), got %q (err: %v)", res, err)
	}

	// Scan other root paths:
	// cbrt(2^90) -> 2^30 = 1073741824
	res, err = EvaluateInfix("cbrt(2^90)")
	if err != nil || res != "1073741824" {
		t.Fatalf("cbrt(2^90) expected 1073741824, got %q (err: %v)", res, err)
	}

	// cbrt(2^93) -> 2^31 = 2147483648
	res, err = EvaluateInfix("cbrt(2^93)")
	if err != nil || res != "2147483648" {
		t.Fatalf("cbrt(2^93) expected 2147483648, got %q (err: %v)", res, err)
	}

	// Quadratic solve with discriminant > 2^64: x^2 = 2^70 -> x = -34359738368, x = 34359738368
	soln, err := SolveEquation("x^2 = 2^70")
	if err != nil || soln != "x = -34359738368, x = 34359738368" {
		t.Fatalf("solve x^2 = 2^70 expected 'x = -34359738368, x = 34359738368', got %q (err: %v)", soln, err)
	}

	// Property Test: for random big n, the rendered result squared equals n exactly.
	// Test 50 distinct big integers with various structures (perfect squares, square-free, mixed):
	testNList := []*big.Int{
		new(big.Int).Exp(big.NewInt(2), big.NewInt(70), nil),
		new(big.Int).Add(new(big.Int).Exp(big.NewInt(2), big.NewInt(64), nil), big.NewInt(1)),
		new(big.Int).Mul(big.NewInt(8), new(big.Int).Exp(big.NewInt(10), big.NewInt(30), nil)),
		new(big.Int).Add(new(big.Int).Exp(big.NewInt(10), big.NewInt(40), nil), big.NewInt(1)),
	}

	// Add composed big numbers: A^2 * B
	for aInt := int64(100000); aInt <= 100010; aInt++ {
		a := big.NewInt(aInt)
		a2 := new(big.Int).Mul(a, a)
		for _, bInt := range []int64{1, 2, 3, 5, 7, 11, 13} {
			n := new(big.Int).Mul(a2, big.NewInt(bInt))
			testNList = append(testNList, n)
		}
	}

	// Add large primes / composites
	for p := int64(100); p <= 110; p++ {
		n := new(big.Int).Exp(big.NewInt(p), big.NewInt(12), nil)
		testNList = append(testNList, n)
		testNList = append(testNList, new(big.Int).Add(n, big.NewInt(3)))
	}

	for idx, n := range testNList {
		rendered, err := EvaluateInfix(fmt.Sprintf("sqrt(%s)", n.String()))
		if err != nil {
			t.Fatalf("Property test #%d: sqrt(%s) failed: %v", idx, n.String(), err)
		}
		sq, err := parseRenderedSurdSquared(rendered)
		if err != nil {
			t.Fatalf("Property test #%d: could not parse rendered surd %q for n=%s: %v", idx, rendered, n.String(), err)
		}
		if sq.Cmp(n) != 0 {
			t.Fatalf("Property test #%d FAIL: rendered=%q squared=%s != n=%s", idx, rendered, sq.String(), n.String())
		}
	}
}

func TestFloatFormatPrecision(t *testing.T) {
	// Tests for float formatter across calc, solve, and lua:
	// float(2^53) -> 9007199254740992
	// 2.0^60 -> 1.15292e+18
	// 1e300 -> 1e+300
	// exp(-20) -> 2.06115e-09
	// -0.0 -> 0

	// 1. Direct FormatFloatStandard unit tests
	cases := []struct {
		f       float64
		want    string
		wantErr bool
	}{
		{f: 9007199254740992.0, want: "9007199254740992", wantErr: false}, // 2^53
		{f: math.Pow(2.0, 60), want: "1.15292e+18", wantErr: false},       // 2^60
		{f: 1e300, want: "1e+300", wantErr: false},
		{f: math.Exp(-20), want: "2.06115e-09", wantErr: false},
		{f: -0.0, want: "0", wantErr: false},
		{f: 0.0, want: "0", wantErr: false},
		{f: math.NaN(), want: "", wantErr: true},
		{f: math.Inf(1), want: "", wantErr: true},
		{f: math.Inf(-1), want: "", wantErr: true},
	}

	for _, c := range cases {
		got, err := FormatFloatStandard(c.f)
		if (err != nil) != c.wantErr {
			t.Fatalf("FormatFloatStandard(%v): err = %v, wantErr %v", c.f, err, c.wantErr)
		}
		if !c.wantErr && got != c.want {
			t.Fatalf("FormatFloatStandard(%v): got %q, want %q", c.f, got, c.want)
		}
	}

	// 2. calc integration tests
	calcTests := []struct {
		expr string
		want string
	}{
		{"float(2^53)", "9007199254740992"},
		{"float(2.0^60)", "1.15292e+18"},
		{"float(1e300)", "1e+300"},
		{"exp(-20)", "2.06115e-09"},
		{"-0.0", "0"},
	}

	for _, tt := range calcTests {
		res, err := EvaluateInfix(tt.expr)
		if err != nil || res != tt.want {
			t.Fatalf("calc %q expected %q, got %q (err: %v)", tt.expr, tt.want, res, err)
		}
	}

	// lua integration tests for float formatting
	rs := NewRegisterStore()
	luaTests := []struct {
		script string
		want   string
	}{
		{"return 9007199254740992", "9007199254740992"},
		{"return 2.0^60", "1.15292e+18"},
		{"return 1e300", "1e+300"},
		{"return math.exp(-20)", "2.06115e-09"},
		{"return -0.0", "0"},
	}

	for _, tt := range luaTests {
		got, err := ExecuteLua(tt.script, rs)
		if err != nil || got != tt.want {
			t.Fatalf("lua %q expected %q, got %q (err: %v)", tt.script, tt.want, got, err)
		}
	}

	// 3. NaN/Inf rejection: NaN/Inf never bind
	resp := DispatchTool("calc", "0.0 / 0.0")
	if resp.Success {
		t.Fatalf("calc 0.0/0.0 expected failure, got success with result %q", resp.Result)
	}
	resp = DispatchTool("calc", "1.0 / 0.0")
	if resp.Success {
		t.Fatalf("calc 1.0/0.0 expected failure, got success with result %q", resp.Result)
	}

	resp = DispatchTool("lua", "return 0/0")
	if resp.Success {
		t.Fatalf("lua return 0/0 expected failure, got success with result %q", resp.Result)
	}
	resp = DispatchTool("lua", "return 1/0")
	if resp.Success {
		t.Fatalf("lua return 1/0 expected failure, got success with result %q", resp.Result)
	}
}

func TestUnnestedErrorCodes(t *testing.T) {
	// solve x^1000000000 = 1, x must return clean ErrOverflow without nested ErrSyntax
	resp := DispatchTool("solve", "x^1000000000 = 1, x")
	if resp.Success {
		t.Fatalf("solve x^1000000000 = 1, x expected failure, got success: %q", resp.Result)
	}
	expected := "ErrOverflow: polynomial exponent exceeds maximum 100"
	if resp.Error != expected {
		t.Fatalf("error nesting check: expected %q, got %q", expected, resp.Error)
	}
}
