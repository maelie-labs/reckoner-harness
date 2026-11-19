package main

import (
	"fmt"
	"math"
	"sort"
	"strings"
)

// Interval represents a single real interval [a, b], (a, b), [a, b), etc.
type Interval struct {
	LeftOpen  bool
	LeftVal   *ExactNum // nil represents -inf
	RightOpen bool
	RightVal  *ExactNum // nil represents +inf
}

func (iv Interval) String() string {
	lBracket := "["
	if iv.LeftOpen || iv.LeftVal == nil {
		lBracket = "("
	}
	rBracket := "]"
	if iv.RightOpen || iv.RightVal == nil {
		rBracket = ")"
	}

	lStr := "-inf"
	if iv.LeftVal != nil {
		lStr = iv.LeftVal.String()
	}

	rStr := "inf"
	if iv.RightVal != nil {
		rStr = iv.RightVal.String()
	}

	return fmt.Sprintf("%s%s, %s%s", lBracket, lStr, rStr, rBracket)
}

// IntervalSet represents a union of disjoint real intervals.
type IntervalSet []Interval

func (is IntervalSet) String() string {
	if len(is) == 0 {
		return "no solution"
	}
	if len(is) == 1 && is[0].LeftVal == nil && is[0].RightVal == nil {
		return "all real numbers"
	}
	var parts []string
	for _, iv := range is {
		parts = append(parts, iv.String())
	}
	return strings.Join(parts, " U ")
}

// SolveInequality parses and solves an inequality expression like "x^2 - 4 <= 0", "2*x - 3 > 5", "(x-1)/(x+2) >= 0".
func SolveInequality(expr, varName string, store *RegisterStore) (IntervalSet, error) {
	expr = strings.TrimSpace(expr)

	// Detect inequality operator: <=, >=, <, >
	op := ""
	idx := -1

	for _, cand := range []string{"<=", ">=", "<", ">"} {
		if p := strings.Index(expr, cand); p != -1 {
			op = cand
			idx = p
			break
		}
	}

	if op == "" {
		return nil, fmt.Errorf("ErrSyntax: no inequality operator found")
	}

	lhsStr := strings.TrimSpace(expr[:idx])
	rhsStr := strings.TrimSpace(expr[idx+len(op):])

	// Check if this is an absolute value inequality: abs(A) op B
	if strings.HasPrefix(lhsStr, "abs(") && strings.HasSuffix(lhsStr, ")") {
		inner := strings.TrimSuffix(strings.TrimPrefix(lhsStr, "abs("), ")")
		return solveAbsInequality(inner, rhsStr, op, varName, store)
	}

	// Try standard polynomial inequality first: LHS - RHS op 0
	diffExpr := fmt.Sprintf("(%s) - (%s)", lhsStr, rhsStr)
	_, pMulti, err := ExpandPoly(diffExpr, store)
	if err == nil {
		pRat, detectedVar, errPoly := MultiPolyToRatPoly(pMulti)
		if errPoly == nil {
			if varName == "" {
				varName = detectedVar
			}
			return solvePolyInequality(pRat, op, varName, store)
		}
	}

	// Fallback: Check if LHS has division (rational inequality): (P)/(Q) op RHS or P/Q op RHS
	if strings.Contains(lhsStr, "/") {
		return solveRationalInequality(lhsStr, rhsStr, op, varName, store)
	}

	return nil, err
}

func solvePolyInequality(p RatPoly, op, varName string, store *RegisterStore) (IntervalSet, error) {
	p = p.Trim()
	deg := p.Degree()

	if deg == 0 {
		c := p[0]
		sat := false
		switch op {
		case "<":
			sat = c.Sign() < 0
		case "<=":
			sat = c.Sign() <= 0
		case ">":
			sat = c.Sign() > 0
		case ">=":
			sat = c.Sign() >= 0
		}
		if sat {
			return IntervalSet{{LeftVal: nil, RightVal: nil, LeftOpen: true, RightOpen: true}}, nil
		}
		return IntervalSet{}, nil
	}

	// Find real roots of p
	var roots []ExactNum
	pMulti := RatPolyToMultiPoly(p, varName)
	_, rList, err := RootsPoly(pMulti.String(), store)
	if err != nil {
		return nil, err
	}

	for _, r := range rList {
		if r.IsReal() {
			roots = append(roots, r)
		}
	}

	// Sort roots ascending by float value
	sort.Slice(roots, func(i, j int) bool {
		return roots[i].ToFloat64() < roots[j].ToFloat64()
	})

	// Deduplicate roots
	var uniqueRoots []ExactNum
	for _, r := range roots {
		if len(uniqueRoots) == 0 || math.Abs(uniqueRoots[len(uniqueRoots)-1].ToFloat64()-r.ToFloat64()) > 1e-9 {
			uniqueRoots = append(uniqueRoots, r)
		}
	}

	evalPolyAt := func(x float64) float64 {
		sum := 0.0
		xPow := 1.0
		for _, c := range p {
			cF, _ := c.Float64()
			sum += cF * xPow
			xPow *= x
		}
		return sum
	}

	testSat := func(val float64) bool {
		switch op {
		case "<":
			return val < -1e-12
		case "<=":
			return val <= 1e-12
		case ">":
			return val > 1e-12
		case ">=":
			return val >= -1e-12
		}
		return false
	}

	// Generate candidate intervals
	n := len(uniqueRoots)
	var intervalSat []bool
	testPoints := make([]float64, n+1)

	if n == 0 {
		testPoints[0] = 0.0
	} else {
		testPoints[0] = uniqueRoots[0].ToFloat64() - 1.0
		for i := 0; i < n-1; i++ {
			testPoints[i+1] = (uniqueRoots[i].ToFloat64() + uniqueRoots[i+1].ToFloat64()) / 2.0
		}
		testPoints[n] = uniqueRoots[n-1].ToFloat64() + 1.0
	}

	for _, pt := range testPoints {
		val := evalPolyAt(pt)
		intervalSat = append(intervalSat, testSat(val))
	}

	rootSat := make([]bool, n)
	for i, r := range uniqueRoots {
		val := evalPolyAt(r.ToFloat64())
		rootSat[i] = testSat(val)
	}

	// Build intervals
	var rawIntervals []Interval
	for i := 0; i <= n; i++ {
		if intervalSat[i] {
			var left *ExactNum
			leftOpen := true
			if i > 0 {
				left = &uniqueRoots[i-1]
				leftOpen = !rootSat[i-1]
			}

			var right *ExactNum
			rightOpen := true
			if i < n {
				right = &uniqueRoots[i]
				rightOpen = !rootSat[i]
			}

			rawIntervals = append(rawIntervals, Interval{
				LeftVal:   left,
				LeftOpen:  leftOpen,
				RightVal:  right,
				RightOpen: rightOpen,
			})
		} else {
			// Check if single root is isolated solution: [r, r]
			if i < n && rootSat[i] && (i == 0 || !intervalSat[i]) && (i+1 > n || !intervalSat[i+1]) {
				rawIntervals = append(rawIntervals, Interval{
					LeftVal:   &uniqueRoots[i],
					LeftOpen:  false,
					RightVal:  &uniqueRoots[i],
					RightOpen: false,
				})
			}
		}
	}

	return mergeIntervals(rawIntervals), nil
}

func solveRationalInequality(lhsStr, rhsStr, op, varName string, store *RegisterStore) (IntervalSet, error) {
	// Find top-level /
	slashIdx := -1
	depth := 0
	for i, r := range lhsStr {
		if r == '(' {
			depth++
		} else if r == ')' {
			depth--
		} else if r == '/' && depth == 0 {
			slashIdx = i
			break
		}
	}
	var numStr, denStr string
	if slashIdx != -1 {
		numStr = stripOuterParens(lhsStr[:slashIdx])
		denStr = stripOuterParens(lhsStr[slashIdx+1:])
	} else {
		parts := strings.SplitN(lhsStr, "/", 2)
		numStr = stripOuterParens(parts[0])
		denStr = stripOuterParens(parts[1])
	}

	rhsExpr := fmt.Sprintf("(%s)", rhsStr)

	// Numerator expression: (num) - (rhs) * (den)
	combNumExpr := fmt.Sprintf("(%s) - (%s)*(%s)", numStr, rhsExpr, denStr)
	_, numMulti, err := ExpandPoly(combNumExpr, store)
	if err != nil {
		return nil, err
	}
	_, denMulti, err := ExpandPoly(denStr, store)
	if err != nil {
		return nil, err
	}

	numPoly, detectedVar, err := MultiPolyToRatPoly(numMulti)
	if err != nil {
		return nil, err
	}
	if varName == "" {
		varName = detectedVar
	}
	denPoly, _, err := MultiPolyToRatPoly(denMulti)
	if err != nil {
		return nil, err
	}

	// Critical points: roots of numPoly and roots of denPoly
	var numRoots, denRoots []ExactNum
	_, nRoots, _ := RootsPoly(numMulti.String(), store)
	for _, r := range nRoots {
		if r.IsReal() {
			numRoots = append(numRoots, r)
		}
	}
	_, dRoots, _ := RootsPoly(denMulti.String(), store)
	for _, r := range dRoots {
		if r.IsReal() {
			denRoots = append(denRoots, r)
		}
	}

	var allCrit []ExactNum
	allCrit = append(allCrit, numRoots...)
	allCrit = append(allCrit, denRoots...)

	sort.Slice(allCrit, func(i, j int) bool {
		return allCrit[i].ToFloat64() < allCrit[j].ToFloat64()
	})

	var crit []ExactNum
	for _, r := range allCrit {
		if len(crit) == 0 || math.Abs(crit[len(crit)-1].ToFloat64()-r.ToFloat64()) > 1e-9 {
			crit = append(crit, r)
		}
	}

	isPole := func(val float64) bool {
		for _, dr := range denRoots {
			if math.Abs(dr.ToFloat64()-val) < 1e-9 {
				return true
			}
		}
		return false
	}

	evalRational := func(x float64) (float64, bool) {
		sumD := 0.0
		xPow := 1.0
		for _, c := range denPoly {
			cF, _ := c.Float64()
			sumD += cF * xPow
			xPow *= x
		}
		if math.Abs(sumD) < 1e-12 {
			return 0, false
		}

		sumN := 0.0
		xPow = 1.0
		for _, c := range numPoly {
			cF, _ := c.Float64()
			sumN += cF * xPow
			xPow *= x
		}
		return sumN / sumD, true
	}

	testSat := func(val float64) bool {
		switch op {
		case "<":
			return val < -1e-12
		case "<=":
			return val <= 1e-12
		case ">":
			return val > 1e-12
		case ">=":
			return val >= -1e-12
		}
		return false
	}

	n := len(crit)
	var intervalSat []bool
	testPoints := make([]float64, n+1)

	if n == 0 {
		testPoints[0] = 0.0
	} else {
		testPoints[0] = crit[0].ToFloat64() - 1.0
		for i := 0; i < n-1; i++ {
			testPoints[i+1] = (crit[i].ToFloat64() + crit[i+1].ToFloat64()) / 2.0
		}
		testPoints[n] = crit[n-1].ToFloat64() + 1.0
	}

	for _, pt := range testPoints {
		val, ok := evalRational(pt)
		intervalSat = append(intervalSat, ok && testSat(val))
	}

	rootSat := make([]bool, n)
	for i, r := range crit {
		if isPole(r.ToFloat64()) {
			rootSat[i] = false
		} else {
			val, ok := evalRational(r.ToFloat64())
			rootSat[i] = ok && testSat(val)
		}
	}

	var rawIntervals []Interval
	for i := 0; i <= n; i++ {
		if intervalSat[i] {
			var left *ExactNum
			leftOpen := true
			if i > 0 {
				left = &crit[i-1]
				leftOpen = !rootSat[i-1] || isPole(crit[i-1].ToFloat64())
			}

			var right *ExactNum
			rightOpen := true
			if i < n {
				right = &crit[i]
				rightOpen = !rootSat[i] || isPole(crit[i].ToFloat64())
			}

			rawIntervals = append(rawIntervals, Interval{
				LeftVal:   left,
				LeftOpen:  leftOpen,
				RightVal:  right,
				RightOpen: rightOpen,
			})
		}
	}

	return mergeIntervals(rawIntervals), nil
}

func solveAbsInequality(inner, rhsStr, op, varName string, store *RegisterStore) (IntervalSet, error) {
	// abs(A) < B  <=> A < B and A > -B  <=> A - B < 0 and -A - B < 0
	// abs(A) > B  <=> A > B or A < -B   <=> A - B > 0 or -A - B > 0
	eq1 := fmt.Sprintf("(%s) - (%s)", inner, rhsStr)
	eq2 := fmt.Sprintf("-(%s) - (%s)", inner, rhsStr)

	set1, err := SolveInequality(fmt.Sprintf("%s %s 0", eq1, op), varName, store)
	if err != nil {
		return nil, err
	}
	set2, err := SolveInequality(fmt.Sprintf("%s %s 0", eq2, op), varName, store)
	if err != nil {
		return nil, err
	}

	if op == "<" || op == "<=" {
		// Intersection of set1 and set2
		return intersectIntervalSets(set1, set2), nil
	}
	// Union of set1 and set2
	var combined []Interval
	combined = append(combined, set1...)
	combined = append(combined, set2...)
	return mergeIntervals(combined), nil
}

func mergeIntervals(intervals []Interval) IntervalSet {
	if len(intervals) <= 1 {
		return intervals
	}

	// Sort intervals by left value
	sort.Slice(intervals, func(i, j int) bool {
		f1 := -math.MaxFloat64
		if intervals[i].LeftVal != nil {
			f1 = intervals[i].LeftVal.ToFloat64()
		}
		f2 := -math.MaxFloat64
		if intervals[j].LeftVal != nil {
			f2 = intervals[j].LeftVal.ToFloat64()
		}
		return f1 < f2
	})

	var res []Interval
	cur := intervals[0]

	for i := 1; i < len(intervals); i++ {
		next := intervals[i]

		curRightF := math.MaxFloat64
		if cur.RightVal != nil {
			curRightF = cur.RightVal.ToFloat64()
		}
		nextLeftF := -math.MaxFloat64
		if next.LeftVal != nil {
			nextLeftF = next.LeftVal.ToFloat64()
		}

		if curRightF > nextLeftF || (math.Abs(curRightF-nextLeftF) < 1e-9 && (!cur.RightOpen || !next.LeftOpen)) {
			// Overlap or touch -> merge
			nextRightF := math.MaxFloat64
			if next.RightVal != nil {
				nextRightF = next.RightVal.ToFloat64()
			}
			if nextRightF > curRightF {
				cur.RightVal = next.RightVal
				cur.RightOpen = next.RightOpen
			} else if math.Abs(nextRightF-curRightF) < 1e-9 {
				cur.RightOpen = cur.RightOpen && next.RightOpen
			}
		} else {
			res = append(res, cur)
			cur = next
		}
	}
	res = append(res, cur)
	return res
}

func intersectIntervalSets(s1, s2 IntervalSet) IntervalSet {
	var res []Interval
	for _, i1 := range s1 {
		for _, i2 := range s2 {
			f1L := -math.MaxFloat64
			if i1.LeftVal != nil {
				f1L = i1.LeftVal.ToFloat64()
			}
			f1R := math.MaxFloat64
			if i1.RightVal != nil {
				f1R = i1.RightVal.ToFloat64()
			}
			f2L := -math.MaxFloat64
			if i2.LeftVal != nil {
				f2L = i2.LeftVal.ToFloat64()
			}
			f2R := math.MaxFloat64
			if i2.RightVal != nil {
				f2R = i2.RightVal.ToFloat64()
			}

			maxL := math.Max(f1L, f2L)
			minR := math.Min(f1R, f2R)

			if maxL < minR {
				var left *ExactNum
				leftOpen := false
				if f1L > f2L {
					left = i1.LeftVal
					leftOpen = i1.LeftOpen
				} else if f2L > f1L {
					left = i2.LeftVal
					leftOpen = i2.LeftOpen
				} else {
					left = i1.LeftVal
					leftOpen = i1.LeftOpen || i2.LeftOpen
				}

				var right *ExactNum
				rightOpen := false
				if f1R < f2R {
					right = i1.RightVal
					rightOpen = i1.RightOpen
				} else if f2R < f1R {
					right = i2.RightVal
					rightOpen = i2.RightOpen
				} else {
					right = i1.RightVal
					rightOpen = i1.RightOpen || i2.RightOpen
				}

				res = append(res, Interval{
					LeftVal:   left,
					LeftOpen:  leftOpen,
					RightVal:  right,
					RightOpen: rightOpen,
				})
			}
		}
	}
	return mergeIntervals(res)
}

func stripOuterParens(s string) string {
	s = strings.TrimSpace(s)
	for strings.HasPrefix(s, "(") && strings.HasSuffix(s, ")") {
		depth := 0
		matches := true
		for i, r := range s {
			if r == '(' {
				depth++
			} else if r == ')' {
				depth--
				if depth == 0 && i < len(s)-1 {
					matches = false
					break
				}
			}
		}
		if matches && depth == 0 {
			s = strings.TrimSpace(s[1 : len(s)-1])
		} else {
			break
		}
	}
	return s
}
