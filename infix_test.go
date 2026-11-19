package main

import (
	"math"
	"strconv"
	"strings"
	"testing"
)

func TestSpec81Cases(t *testing.T) {
	// 1. Exact fractions: 3/7 + 5/11 == 68/77
	res, err := EvaluateInfix("3/7 + 5/11")
	if err != nil || res != "68/77" {
		t.Fatalf("3/7 + 5/11 failed: got %q, err %v", res, err)
	}

	// 2. Terminating decimals: 0.1 + 0.2 == 0.3
	res, err = EvaluateInfix("0.1 + 0.2")
	if err != nil || res != "0.3" {
		t.Fatalf("0.1 + 0.2 failed: got %q, err %v", res, err)
	}

	// 3. Clean integer display: 0.25 * 4 == 1
	res, err = EvaluateInfix("0.25 * 4")
	if err != nil || res != "1" {
		t.Fatalf("0.25 * 4 failed: got %q, err %v", res, err)
	}

	// 4. Surds: sqrt(72) == 6*sqrt(2)
	res, err = EvaluateInfix("sqrt(72)")
	if err != nil || res != "6*sqrt(2)" {
		t.Fatalf("sqrt(72) failed: got %q, err %v", res, err)
	}

	// 5. Modular arithmetic: modexp(17, 2023, 10) == 3
	res, err = EvaluateInfix("modexp(17, 2023, 10)")
	if err != nil || res != "3" {
		t.Fatalf("modexp(17, 2023, 10) failed: got %q, err %v", res, err)
	}

	// 6. Factorization timeout & cap: factor(3528) succeeds; inputs > 10^14 trigger ErrFactorOverflow
	res, err = EvaluateInfix("factor(3528)")
	if err != nil || res != "\"2^3 * 3^2 * 7^2\"" && res != "2^3 * 3^2 * 7^2" {
		t.Fatalf("factor(3528) failed: got %q, err %v", res, err)
	}
	_, err = EvaluateInfix("factor(100000000000001)")
	if err == nil || (!strings.Contains(err.Error(), "ErrFactorOverflow") && !strings.Contains(err.Error(), "ErrOverflow")) {
		t.Fatalf("factor(>10^14) expected ErrOverflow or ErrFactorOverflow, got %v", err)
	}

	// 7. Statistics: std([2, 4, 4, 4, 5, 5, 7, 9]) == 2.13809
	res, err = EvaluateInfix("std([2, 4, 4, 4, 5, 5, 7, 9])")
	if err != nil || res != "2.13809" {
		t.Fatalf("std([2, 4, 4, 4, 5, 5, 7, 9]) failed: got %q, err %v", res, err)
	}

	// 8. Units: convert(45, "mph", "m/s") == 20.1168 m/s
	res, err = EvaluateInfix("convert(45, \"mph\", \"m/s\")")
	if err != nil || res != "20.1168 m/s" {
		t.Fatalf("convert(45, \"mph\", \"m/s\") failed: got %q, err %v", res, err)
	}

	// 9. Unit Safety: 18480 ft + 10 m raises ErrUnitMismatch
	_, err = EvaluateInfix("18480 ft + 10 m")
	if err == nil || !strings.Contains(err.Error(), "ErrUnitMismatch") {
		t.Fatalf("18480 ft + 10 m expected ErrUnitMismatch, got %v", err)
	}
}

func TestDecimalFormattingCases(t *testing.T) {
	// Clean decimals
	cases := []struct {
		expr     string
		expected string
	}{
		{"18/12", "1.5"},
		{"44/5", "8.8"},
		{"0.25 * 4", "1"},
		{"1/3", "1/3"},
		{"3/7 + 5/11", "68/77"},
		{"sqrt(72)", "6*sqrt(2)"},
		{"frac(0.75)", "3/4"},
		{"frac(18/12)", "3/2"},
		{"float(2/3)", "0.666667"},
		{"round(85/3, 2)", "28.33"},
		{"24 // 5", "4"},
		{"85 // 12", "7"},
		{"24 // 5 == 4", "true"},
		{"5 < 3", "false"},
		{"10 >= 10", "true"},
		{"totient(9)", "6"},
		{"totient(10)", "4"},
		{"gcd(12, 18)", "6"},
		{"lcm(4, 6)", "12"},
		{"comb(5, 2)", "10"},
		{"perm(5, 2)", "20"},
		{"factorial(5)", "120"},
		{"mean([2, 4, 4, 4, 5, 5, 7, 9])", "5"},
		{"median([2, 4, 4, 4, 5, 5, 7, 9])", "4.5"},
		{"sum([1, 2, 3, 4, 5])", "15"},
		{"prod([2, 3, 4])", "24"},
		{"sum_range(1, 100)", "5050"},
	}

	for _, tc := range cases {
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

func TestSyntaxAndUnitErrors(t *testing.T) {
	// Explicit multiplication required: 2x -> ErrSyntax
	_, err := EvaluateInfix("2x")
	if err == nil || !strings.Contains(err.Error(), "ErrSyntax") {
		t.Errorf("2x expected ErrSyntax, got %v", err)
	}

	// Boolean arithmetic prohibition: true + 1 -> ErrSyntax
	_, err = EvaluateInfix("true + 1")
	if err == nil || !strings.Contains(err.Error(), "ErrSyntax") {
		t.Errorf("true + 1 expected ErrSyntax, got %v", err)
	}

	// Division by zero: 10 / 0 -> ErrDivZero
	_, err = EvaluateInfix("10 / 0")
	if err == nil || !strings.Contains(err.Error(), "ErrDivZero") {
		t.Errorf("10 / 0 expected ErrDivZero, got %v", err)
	}

	// Incompatible unit conversion: convert(10, "m", "kg") -> ErrUnitMismatch
	_, err = EvaluateInfix("convert(10, \"m\", \"kg\")")
	if err == nil || !strings.Contains(err.Error(), "ErrUnitMismatch") {
		t.Errorf("convert(10, m, kg) expected ErrUnitMismatch, got %v", err)
	}
}

func BenchmarkTypicalCalc(b *testing.B) {
	expr := "482 * 19 - 14 + sqrt(144) * (2^3)"
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = EvaluateInfix(expr)
	}
}

func TestHarnessFix3Cases(t *testing.T) {
	rs := NewRegisterStore()

	// 1. Issue 1, exact rational roots
	exactRootCases := []struct {
		expr     string
		expected string
	}{
		{"8^(1/3)", "2"},
		{"(27/8)^(1/3)", "1.5"}, // terminating decimal 3/2 -> 1.5
		{"16^(3/4)", "8"},
		{"1000^(1/3)", "10"},
		{"8^(-1/3)", "0.5"}, // terminating decimal 1/2 -> 0.5
		{"(-8)^(1/3)", "-2"},
		{"(-8)^(2/3)", "4"},
		{"8^(1/3) == 2", "true"},
		{"2 == 8^(1/3)", "true"},
		{"27^(1/3) == 3", "true"},
		{"1000^(1/3) == 10", "true"},
		{"8^(1/3) >= 2", "true"},
		{"8^(1/3) - 2", "0"},
		{"2 - 8^(1/3)", "0"},
	}

	for _, tc := range exactRootCases {
		res, err := EvaluateInfixWithStore(tc.expr, rs)
		if err != nil {
			t.Errorf("%s failed with error: %v", tc.expr, err)
		} else if res != tc.expected {
			t.Errorf("%s = %q, want %q", tc.expr, res, tc.expected)
		}
	}

	// Domain errors for negative bases with even denominators
	domainErrorCases := []string{
		"(-8)^(1/2)",
		"(-16)^(1/4)",
		"(-4)^(3/2)",
	}
	for _, expr := range domainErrorCases {
		_, err := EvaluateInfixWithStore(expr, rs)
		if err == nil || !strings.Contains(err.Error(), "ErrDomain") {
			t.Errorf("%s expected ErrDomain, got %v", expr, err)
		}
	}

	// 2. Issue 1, inexact comparison ties return ErrUnsupported (inside the tie)
	inexactComparisonTieCases := []string{
		"3^(3/4) == 27^(1/4)",
		"27^(1/4) == 3^(3/4)",
		"3^(3/4) != 27^(1/4)",
		"3^(3/4) < 27^(1/4)",
		"27^(1/4) < 3^(3/4)",
		"3^(3/4) <= 27^(1/4)",
		"27^(1/4) <= 3^(3/4)",
		"3^(3/4) > 27^(1/4)",
		"27^(1/4) > 3^(3/4)",
		"3^(3/4) >= 27^(1/4)",
		"27^(1/4) >= 3^(3/4)",
	}
	for _, expr := range inexactComparisonTieCases {
		_, err := EvaluateInfixWithStore(expr, rs)
		if err == nil || !strings.Contains(err.Error(), "ErrUnsupported") {
			t.Errorf("%s expected ErrUnsupported, got %v", expr, err)
		}
	}

	// Inexact comparison outside the tie (clear margin)
	inexactComparisonClearCases := []struct {
		expr     string
		expected string
	}{
		{"3^(3/4) > 2", "true"},
		{"3^(3/4) < 2.5", "true"},
		{"27^(1/4) > 2.2", "true"},
		{"27^(1/4) < 2.3", "true"},
		{"pi > 3", "true"},
		{"pi == 3.14159", "false"},
	}
	for _, tc := range inexactComparisonClearCases {
		res, err := EvaluateInfixWithStore(tc.expr, rs)
		if err != nil {
			t.Errorf("%s failed with error: %v", tc.expr, err)
		} else if res != tc.expected {
			t.Errorf("%s = %q, want %q", tc.expr, res, tc.expected)
		}
	}

	// Float cancellation below 4 ulps for floats of different origin returns ErrUnsupported
	cancellationTieCases := []string{
		"3^(3/4) - 27^(1/4)",
		"27^(1/4) - 3^(3/4)",
		"tan(20*pi/180) + 4*sin(20*pi/180) - sqrt(3)",
	}
	for _, expr := range cancellationTieCases {
		_, err := EvaluateInfixWithStore(expr, rs)
		if err == nil || !strings.Contains(err.Error(), "result is below floating-point precision") {
			t.Errorf("%s expected ErrUnsupported result below floating-point precision, got: %v", expr, err)
		}
	}

	// Exact symbolic atoms and structural identities evaluate to exact 0
	symbolicZeroCases := []string{
		"exp(ln(2)) - 2",
		"-e + e",
		"e - e",
		"9 + ln(3) - (ln(3) + 9)",
		"sin(1) - sin(1)",
		"ln(e) - (1/e)*e",
		"log(2)/log(2) + 2 - 3",
		"log10(2) + log10(5) - (pi/23)^0",
	}
	for _, expr := range symbolicZeroCases {
		res, err := EvaluateInfixWithStore(expr, rs)
		if err != nil {
			t.Errorf("%s expected exact 0, got error: %v", expr, err)
		} else if res != "0" {
			t.Errorf("%s = %q, want 0", expr, res)
		}
	}

	// Cancellation outside the tie (clear difference returns non-zero float)
	cancellationClearCases := []string{
		"1.00001 - 1.0",
		"3^(3/4) - 2",
		"27^(1/4) - 2.5",
	}
	for _, expr := range cancellationClearCases {
		res, err := EvaluateInfixWithStore(expr, rs)
		if err != nil {
			t.Errorf("%s failed with error: %v", expr, err)
		} else if res == "0" {
			t.Errorf("%s expected non-zero, got 0", expr)
		}
	}

	// 3. Issue 2, exact trig and constant comparisons
	trigCases := []struct {
		expr     string
		expected string
	}{
		{"sin(pi/6) == 1/2", "true"},
		{"1/2 == sin(pi/6)", "true"},
		{"sin(pi/6) == 0.5", "true"},
		{"cos(0) == 1", "true"},
		{"1 == cos(0)", "true"},
		{"cos(0) == 2", "false"},
		{"sin(pi/6) < 1", "true"},
		{"1 > sin(pi/6)", "true"},
		{"pi > 3", "true"},
		{"3 < pi", "true"},
		{"pi == 3.14159", "false"},
		{"3.14159 == pi", "false"},
	}
	for _, tc := range trigCases {
		res, err := EvaluateInfixWithStore(tc.expr, rs)
		if err != nil {
			t.Errorf("%s failed with error: %v", tc.expr, err)
		} else if res != tc.expected {
			t.Errorf("%s = %q, want %q", tc.expr, res, tc.expected)
		}
	}

	// Near float tie on pi
	_, err := EvaluateInfixWithStore("pi == 3.141592653589793", rs)
	if err == nil || !strings.Contains(err.Error(), "ErrUnsupported") {
		t.Errorf("pi == 3.141592653589793 expected ErrUnsupported, got %v", err)
	}

	// 4. Issue 3, lookup candidate formatting
	fCand, err := EvaluateLookupWithStore(`lookup("F")`, rs)
	if err != nil {
		t.Errorf("lookup(F) failed: %v", err)
	} else if fCand != `candidates: element "F" (Fluorine), "Faraday constant"` {
		t.Errorf("lookup(F) = %q, want %q", fCand, `candidates: element "F" (Fluorine), "Faraday constant"`)
	}

	mercCand, err := EvaluateLookupWithStore(`lookup("Mercury")`, rs)
	if err != nil {
		t.Errorf("lookup(Mercury) failed: %v", err)
	} else if mercCand != `candidates: element "Hg" (Mercury), planet "Mercury"` {
		t.Errorf("lookup(Mercury) = %q, want %q", mercCand, `candidates: element "Hg" (Mercury), planet "Mercury"`)
	}
}

func TestFloatCancellationCases(t *testing.T) {
	rs := NewRegisterStore()

	// 1. Symbolic cancellation test table
	// Row 1: 2^(1/3) - 1.25992104989487 -> approx 3.16e-15 (~28 float ulps)
	res1, err := EvaluateInfixWithStore("2^(1/3) - 1.25992104989487", rs)
	if err != nil {
		t.Errorf("2^(1/3) - 1.25992104989487 expected success, got error: %v", err)
	} else {
		f, parseErr := strconv.ParseFloat(res1, 64)
		if parseErr != nil || f == 0.0 || math.Abs(f-3.1e-15) > 1e-15 {
			t.Errorf("2^(1/3) - 1.25992104989487 = %q (%v), want approx 3.16e-15", res1, f)
		}
	}

	// Row 2: 2^(1/3) - 1.2599210498948732 -> undecidable at float precision (under 1 ulp): error
	_, err = EvaluateInfixWithStore("2^(1/3) - 1.2599210498948732", rs)
	if err == nil || !strings.Contains(err.Error(), "result is below floating-point precision") {
		t.Errorf("2^(1/3) - 1.2599210498948732 expected result is below floating-point precision, got: %v", err)
	}

	// Row 3: exp(1) - e -> exact symbolic 0
	res3, err := EvaluateInfixWithStore("exp(1) - e", rs)
	if err != nil {
		t.Errorf("exp(1) - e expected exact symbolic 0, got error: %v", err)
	} else if res3 != "0" {
		t.Errorf("exp(1) - e = %q, want \"0\"", res3)
	}

	res3Eq, err := EvaluateInfixWithStore("exp(1) == e", rs)
	if err != nil {
		t.Errorf("exp(1) == e expected true, got error: %v", err)
	} else if res3Eq != "true" {
		t.Errorf("exp(1) == e = %q, want \"true\"", res3Eq)
	}

	// Row 3: 3^(3/4) - 27^(1/4) -> error (float cancellation of different origin)
	_, err = EvaluateInfixWithStore("3^(3/4) - 27^(1/4)", rs)
	if err == nil || !strings.Contains(err.Error(), "result is below floating-point precision") {
		t.Errorf("3^(3/4) - 27^(1/4) expected result is below floating-point precision, got: %v", err)
	}

	// Row 4: 2^(1/3) > 1.25992104989487 -> true (decidable: gap is ~28 ulps)
	res4, err := EvaluateInfixWithStore("2^(1/3) > 1.25992104989487", rs)
	if err != nil {
		t.Errorf("2^(1/3) > 1.25992104989487 expected success, got error: %v", err)
	} else if res4 != "true" {
		t.Errorf("2^(1/3) > 1.25992104989487 = %q, want \"true\"", res4)
	}

	// Comparisons at or below 4 ulps for floats of different origin: cannot decide equality
	cmpTieCases := []string{
		"2^(1/3) == 1.2599210498948732",
		"3^(3/4) == 27^(1/4)",
		"3^(3/4) < 27^(1/4)",
		"3^(3/4) > 27^(1/4)",
	}
	for _, expr := range cmpTieCases {
		_, err := EvaluateInfixWithStore(expr, rs)
		if err == nil || !strings.Contains(err.Error(), "cannot decide equality of inexact values") {
			t.Errorf("%s expected cannot decide equality of inexact values, got: %v", expr, err)
		}
	}

	// 2. Exact operands keep exact arithmetic
	exactCases := []struct {
		expr     string
		expected string
	}{
		{"0.1 + 0.2 - 0.3", "0"},
		{"8^(1/3) - 2", "0"},
		{"2 - 8^(1/3)", "0"},
	}
	for _, tc := range exactCases {
		res, err := EvaluateInfixWithStore(tc.expr, rs)
		if err != nil {
			t.Errorf("%s failed: %v", tc.expr, err)
		} else if res != tc.expected {
			t.Errorf("%s = %q, want %q", tc.expr, res, tc.expected)
		}
	}

	// sqrt(2) - 1.41421356237309505 exact symbolic
	resSym, err := EvaluateInfixWithStore("sqrt(2) - 1.41421356237309505", rs)
	if err != nil {
		t.Errorf("sqrt(2) - 1.41421356237309505 expected exact symbolic, got error: %v", err)
	} else if resSym == "0" {
		t.Errorf("sqrt(2) - 1.41421356237309505 should not be 0, got %q", resSym)
	}

	// 3. Both sides of the 4-ulp boundary across magnitudes and signs
	magnitudes := []float64{1.0, 1.25992104989487, 1e-300, 1e300, -1.0, -1e-300, -1e300}
	ulpOffsets := []struct {
		ulps      int64
		expectErr bool
	}{
		{0, true},
		{1, true},
		{2, true},
		{3, true},
		{4, true},
		{5, false},
		{6, false},
		{8, false},
		{16, false},
		{32, false},
		{1000000, false},
	}

	for _, base := range magnitudes {
		baseBits := math.Float64bits(base)
		scale := math.Abs(base)
		u := floatUlp(scale)

		for _, offset := range ulpOffsets {
			// Construct candidate near base
			var near float64
			if base >= 0 {
				near = math.Float64frombits(baseBits + uint64(offset.ulps))
			} else {
				// For negative numbers in IEEE-754, increasing magnitude increases bit pattern
				near = math.Float64frombits(baseBits + uint64(offset.ulps))
			}
			diff := math.Abs(base - near)
			_ = diff

			vBase := CalcVal{Kind: ValKindFloat, Float: base}
			vNear := CalcVal{Kind: ValKindFloat, Float: near}

			// Test subtraction: vBase - vNear
			subRes, subErr := subCalcVals(vBase, vNear)
			if offset.expectErr {
				if subErr == nil || !strings.Contains(subErr.Error(), "result is below floating-point precision") {
					t.Errorf("sub(%g, %g) [offset %d ulps] expected ErrUnsupported precision, got res=%v, err=%v", base, near, offset.ulps, subRes, subErr)
				}
			} else {
				if subErr != nil {
					t.Errorf("sub(%g, %g) [offset %d ulps] expected success, got error: %v", base, near, offset.ulps, subErr)
				}
			}

			// Test addition with opposite sign: vBase + (-vNear)
			vNegNear := CalcVal{Kind: ValKindFloat, Float: -near}
			addRes, addErr := addCalcVals(vBase, vNegNear)
			if offset.expectErr {
				if addErr == nil || !strings.Contains(addErr.Error(), "result is below floating-point precision") {
					t.Errorf("add(%g, %g) [offset %d ulps] expected ErrUnsupported precision, got res=%v, err=%v", base, -near, offset.ulps, addRes, addErr)
				}
			} else {
				if addErr != nil {
					t.Errorf("add(%g, %g) [offset %d ulps] expected success, got error: %v", base, -near, offset.ulps, addErr)
				}
			}

			// Test comparison: compareCalcVals(vBase, vNear)
			cmpRes, cmpErr := compareCalcVals(vBase, vNear)
			if offset.expectErr {
				if cmpErr == nil || !strings.Contains(cmpErr.Error(), "cannot decide equality of inexact values") {
					t.Errorf("cmp(%g, %g) [offset %d ulps] expected cannot decide equality, got cmp=%d, err=%v", base, near, offset.ulps, cmpRes, cmpErr)
				}
			} else {
				if cmpErr != nil {
					t.Errorf("cmp(%g, %g) [offset %d ulps] expected success, got error: %v", base, near, offset.ulps, cmpErr)
				}
			}
		}
		_ = u
	}
}

func TestHarnessFix4Round2Cases(t *testing.T) {
	rs := NewRegisterStore()

	// 1. Exact special values
	specialValues := []struct {
		expr     string
		expected string
	}{
		{"e^0", "1"},
		{"exp(0)", "1"},
		{"ln(1)", "0"},
		{"log(1)", "0"},
		{"log10(1)", "0"},
		{"log2(1)", "0"},
		{"ln(e)", "1"},
		{"log(3, 3)", "1"},
		{"log(e, e)", "1"},
		{"log(2.5, 2.5)", "1"},
		{"log10(10)", "1"},
		{"log10(100)", "2"},
		{"log10(1000)", "3"},
		{"log10(0.1)", "-1"},
		{"log10(0.01)", "-2"},
		{"log2(2)", "1"},
		{"log2(8)", "3"},
		{"log2(0.25)", "-2"},
		{"exp(ln(2))", "2"},
		{"exp(ln(5))", "5"},
		{"ln(exp(1))", "1"},
		{"ln(exp(3))", "3"},
	}
	for _, tc := range specialValues {
		res, err := EvaluateInfixWithStore(tc.expr, rs)
		if err != nil {
			t.Errorf("special value %s failed with error: %v", tc.expr, err)
		} else if res != tc.expected {
			t.Errorf("special value %s = %q, want %q", tc.expr, res, tc.expected)
		}
	}

	// 2. Symbolic atoms and cancellation to exact rational 0 / rational results
	symbolicIdentities := []struct {
		expr     string
		expected string
	}{
		{"-e + e", "0"},
		{"e - e", "0"},
		{"9 + ln(3) - (ln(3) + 9)", "0"},
		{"ln(e) - (1/e)*e", "0"},
		{"log(2)/log(2)", "1"},
		{"log(2)/log(2) + 2 - 3", "0"},
		{"log10(2) + log10(5) - (pi/23)^0", "0"},
		{"0.5*ln(1) + 1 - 1/1 - 2", "-2"},
		{"e^1 - (e-1)*1 - 1", "0"},
		{"e^0 - (0+1)", "0"},
		{"e^0 - (e-1)*0 - 1", "0"},
		{"exp(ln(2)) - 2", "0"},
		{"1 - ln(e)", "0"},
		{"sin(50*pi/180)/sin(70*pi/180) - sin(50*pi/180)/sin(70*pi/180)", "0"},
		{"sin(1) - sin(1)", "0"},
	}
	for _, tc := range symbolicIdentities {
		res, err := EvaluateInfixWithStore(tc.expr, rs)
		if err != nil {
			t.Errorf("identity %s failed with error: %v", tc.expr, err)
		} else if res != tc.expected {
			t.Errorf("identity %s = %q, want %q", tc.expr, res, tc.expected)
		}
	}

	// 3. Show a float only where a float is needed
	floatCases := []struct {
		expr     string
		expected string
	}{
		{"e^0.5 - (e-1)*0.5 - 1", "-0.21042"},
		{"e", "2.718282"},
		{"16 - 4*e", "5.126873"},
		{"ln(5)/5", "0.321888"},
		{"0.5*ln(1/e) + 1/e - 1/(1/e) - 2", "-4.850402"},
		{"0.5*ln(2) + 2 - 1/2 - 2", "-0.153426"},
		{"0.5*ln(e) + e - 1/e - 2", "0.850402"},
	}
	for _, tc := range floatCases {
		res, err := EvaluateInfixWithStore(tc.expr, rs)
		if err != nil {
			t.Errorf("float case %s failed with error: %v", tc.expr, err)
		} else if res != tc.expected {
			t.Errorf("float case %s = %q, want %q", tc.expr, res, tc.expected)
		}
	}

	// 4. Floats of different origin: 4-ulp error is retained
	diffOriginCases := []string{
		"tan(20*pi/180) + 4*sin(20*pi/180) - sqrt(3)",
		"3^(3/4) - 27^(1/4)",
		"2^(1/3) - 1.2599210498948732",
	}
	for _, expr := range diffOriginCases {
		_, err := EvaluateInfixWithStore(expr, rs)
		if err == nil || !strings.Contains(err.Error(), "result is below floating-point precision") {
			t.Errorf("different origin case %s expected ErrUnsupported result below floating-point precision, got: %v", expr, err)
		}
	}
}


