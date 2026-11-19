package main

import (
	"fmt"
	"math"
	"strings"
)

// Point represents a 2D or 3D point with exact coordinates.
type Point []ExactNum

func (pt Point) String() string {
	var parts []string
	for _, c := range pt {
		parts = append(parts, c.String())
	}
	return fmt.Sprintf("[%s]", strings.Join(parts, ", "))
}

// Distance computes Euclidean distance between p1 and p2.
func GeomDist(p1, p2 Point) (*ExactNum, bool, error) {
	if len(p1) != len(p2) || (len(p1) != 2 && len(p1) != 3) {
		return nil, false, fmt.Errorf("ErrDimension: points must have matching dimensions (2 or 3)")
	}

	sumSq := NewExactZero()
	for i := range p1 {
		diff := p1[i].Sub(&p2[i])
		sq, ok := diff.Mul(diff)
		if !ok {
			return nil, false, fmt.Errorf("ErrOverflow: coordinate square overflowed")
		}
		sumSq = sumSq.Add(sq)
	}

	sqrtVal, ok := sumSq.Sqrt()
	if ok {
		return sqrtVal, true, nil
	}
	// Fallback to float
	_ = math.Sqrt(sumSq.ToFloat64())
	return nil, false, nil
}

// Midpoint computes midpoint between p1 and p2.
func GeomMidpoint(p1, p2 Point) (Point, error) {
	if len(p1) != len(p2) || (len(p1) != 2 && len(p1) != 3) {
		return nil, fmt.Errorf("ErrDimension: points must have matching dimensions (2 or 3)")
	}

	res := make(Point, len(p1))
	two := NewExactInt(2)
	for i := range p1 {
		sum := p1[i].Add(&p2[i])
		mid, ok, err := sum.Div(two)
		if !ok || err != nil {
			return nil, fmt.Errorf("ErrInternal: failed to divide coordinate by 2")
		}
		res[i] = *mid
	}
	return res, nil
}

// Slope computes slope between two 2D points.
func GeomSlope(p1, p2 Point) (*ExactNum, bool, error) {
	if len(p1) != 2 || len(p2) != 2 {
		return nil, false, fmt.Errorf("ErrDimension: slope requires 2D points")
	}

	dx := p2[0].Sub(&p1[0])
	dy := p2[1].Sub(&p1[1])

	if dx.IsZero() {
		return nil, false, fmt.Errorf("ErrDivZero: vertical line has undefined slope")
	}

	m, ok, err := dy.Div(dx)
	return m, ok, err
}

// Line computes the line equation through p1 and p2: y = m*x + b or x = c.
func GeomLine(p1, p2 Point) (string, error) {
	if len(p1) != 2 || len(p2) != 2 {
		return "", fmt.Errorf("ErrDimension: line requires 2D points")
	}

	dx := p2[0].Sub(&p1[0])
	dy := p2[1].Sub(&p1[1])

	if dx.IsZero() {
		// Vertical line: x = c
		return fmt.Sprintf("x = %s", p1[0].String()), nil
	}

	m, ok, err := dy.Div(dx)
	if !ok || err != nil {
		return "", fmt.Errorf("ErrUnsupported: slope could not be evaluated exactly")
	}

	// b = y1 - m * x1
	mx1, ok := m.Mul(&p1[0])
	if !ok {
		return "", fmt.Errorf("ErrUnsupported: y-intercept could not be evaluated exactly")
	}
	b := p1[1].Sub(mx1)

	if m.IsZero() {
		// Horizontal line: y = b
		return fmt.Sprintf("y = %s", b.String()), nil
	}

	mStr := m.String()
	mPart := ""
	if mStr == "1" {
		mPart = "x"
	} else if mStr == "-1" {
		mPart = "-x"
	} else {
		mPart = fmt.Sprintf("%s*x", mStr)
	}

	if b.IsZero() {
		return fmt.Sprintf("y = %s", mPart), nil
	}

	bF := b.ToFloat64()
	if bF < 0 {
		absB := b.Neg()
		return fmt.Sprintf("y = %s - %s", mPart, absB.String()), nil
	}
	return fmt.Sprintf("y = %s + %s", mPart, b.String()), nil
}

// AreaPolygon computes polygon area using shoelace formula for 2D points.
func GeomAreaPolygon(points []Point) (*ExactNum, error) {
	n := len(points)
	if n < 3 {
		return nil, fmt.Errorf("ErrDomain: polygon must have at least 3 vertices")
	}
	for _, p := range points {
		if len(p) != 2 {
			return nil, fmt.Errorf("ErrDimension: polygon vertices must be 2D points")
		}
	}

	sum := NewExactZero()
	for i := 0; i < n; i++ {
		j := (i + 1) % n
		// xi * yj - xj * yi
		p1, ok1 := points[i][0].Mul(&points[j][1])
		p2, ok2 := points[j][0].Mul(&points[i][1])
		if !ok1 || !ok2 {
			return nil, fmt.Errorf("ErrOverflow: shoelace multiplication overflowed")
		}
		sum = sum.Add(p1.Sub(p2))
	}

	absSum, ok := sum.Abs()
	if !ok {
		return nil, fmt.Errorf("ErrUnsupported: failed to evaluate absolute polygon area")
	}

	area, ok, err := absSum.Div(NewExactInt(2))
	if !ok || err != nil {
		return nil, fmt.Errorf("ErrInternal: failed to divide area by 2")
	}
	return area, nil
}

// Heron computes triangle area from 3 side lengths: sqrt(s * (s-a) * (s-b) * (s-c)).
func GeomHeron(a, b, c *ExactNum) (*ExactNum, bool, error) {
	aF := a.ToFloat64()
	bF := b.ToFloat64()
	cF := c.ToFloat64()

	if aF <= 0 || bF <= 0 || cF <= 0 || aF+bF <= cF || bF+cF <= aF || cF+aF <= bF {
		return nil, false, fmt.Errorf("ErrDomain: invalid triangle side lengths")
	}

	// s = (a + b + c) / 2
	semi := a.Add(b).Add(c)
	s, ok, err := semi.Div(NewExactInt(2))
	if !ok || err != nil {
		return nil, false, fmt.Errorf("ErrInternal: failed to compute semiperimeter")
	}

	sa := s.Sub(a)
	sb := s.Sub(b)
	sc := s.Sub(c)

	p1, ok1 := s.Mul(sa)
	p2, ok2 := sb.Mul(sc)
	if !ok1 || !ok2 {
		return nil, false, fmt.Errorf("ErrOverflow: heron multiplication overflowed")
	}
	prod, ok3 := p1.Mul(p2)
	if !ok3 {
		return nil, false, fmt.Errorf("ErrOverflow: heron multiplication overflowed")
	}

	res, ok := prod.Sqrt()
	if ok {
		return res, true, nil
	}
	return nil, false, nil
}

// TriangleArea computes area from two sides and included angle C: (1/2) * a * b * sin(C).
func GeomTriangleArea(a, b, C *ExactNum) (*ExactNum, bool, error) {
	sinC, ok := SinExact(C)
	if !ok {
		// Float fallback
		_ = 0.5 * a.ToFloat64() * b.ToFloat64() * math.Sin(C.ToFloat64())
		return nil, false, nil
	}

	ab, ok := a.Mul(b)
	if !ok {
		return nil, false, fmt.Errorf("ErrOverflow: side multiplication overflowed")
	}
	prod, ok := ab.Mul(sinC)
	if !ok {
		return nil, false, fmt.Errorf("ErrOverflow: area product overflowed")
	}

	res, ok, err := prod.Div(NewExactInt(2))
	return res, ok, err
}

// LawCosSide computes side c = sqrt(a^2 + b^2 - 2*a*b*cos(C)).
func GeomLawCosSide(a, b, C *ExactNum) (*ExactNum, bool, error) {
	cosC, ok := CosExact(C)
	if !ok {
		c2 := a.ToFloat64()*a.ToFloat64() + b.ToFloat64()*b.ToFloat64() - 2*a.ToFloat64()*b.ToFloat64()*math.Cos(C.ToFloat64())
		if c2 < 0 {
			return nil, false, fmt.Errorf("ErrDomain: negative squared side length")
		}
		return nil, false, nil
	}

	aSq, ok1 := a.Mul(a)
	bSq, ok2 := b.Mul(b)
	twoAB, ok3 := NewExactInt(2).Mul(a)
	if ok3 {
		twoAB, ok3 = twoAB.Mul(b)
	}
	if !ok1 || !ok2 || !ok3 {
		return nil, false, fmt.Errorf("ErrOverflow: law of cosines multiplication overflowed")
	}

	term3, ok4 := twoAB.Mul(cosC)
	if !ok4 {
		return nil, false, fmt.Errorf("ErrOverflow: law of cosines product overflowed")
	}

	cSq := aSq.Add(bSq).Sub(term3)
	res, ok := cSq.Sqrt()
	if ok {
		return res, true, nil
	}
	return nil, false, nil
}

// LawCosAngle computes angle C = arccos((a^2 + b^2 - c^2) / (2*a*b)).
func GeomLawCosAngle(a, b, c *ExactNum) (*ExactNum, bool, error) {
	aF := a.ToFloat64()
	bF := b.ToFloat64()
	cF := c.ToFloat64()

	if aF <= 0 || bF <= 0 || cF <= 0 || aF+bF <= cF || bF+cF <= aF || cF+aF <= bF {
		return nil, false, fmt.Errorf("ErrDomain: invalid triangle side lengths")
	}

	aSq, _ := a.Mul(a)
	bSq, _ := b.Mul(b)
	cSq, _ := c.Mul(c)

	num := aSq.Add(bSq).Sub(cSq)
	twoAB, _ := NewExactInt(2).Mul(a)
	twoAB, _ = twoAB.Mul(b)

	cosC, ok, err := num.Div(twoAB)
	if ok && err == nil {
		ang, ok := AcosExact(cosC)
		if ok {
			return ang, true, nil
		}
	}

	return nil, false, nil
}

// Circle Area: pi * r^2
func GeomCircArea(r *ExactNum) (*ExactNum, error) {
	rSq, ok := r.Mul(r)
	if !ok {
		return nil, fmt.Errorf("ErrOverflow: radius square overflowed")
	}
	area, ok := rSq.Mul(NewExactPi())
	if !ok {
		return nil, fmt.Errorf("ErrOverflow: circle area multiplication overflowed")
	}
	return area, nil
}

// Circle Circumference: 2 * pi * r
func GeomCircumference(r *ExactNum) (*ExactNum, error) {
	twoPi, ok := NewExactInt(2).Mul(NewExactPi())
	if !ok {
		return nil, fmt.Errorf("ErrInternal: failed to multiply 2*pi")
	}
	res, ok := twoPi.Mul(r)
	if !ok {
		return nil, fmt.Errorf("ErrOverflow: circumference multiplication overflowed")
	}
	return res, nil
}

// Arc Length: r * theta
func GeomArcLength(r, theta *ExactNum) (*ExactNum, error) {
	res, ok := r.Mul(theta)
	if !ok {
		return nil, fmt.Errorf("ErrOverflow: arc length multiplication overflowed")
	}
	return res, nil
}

// Sector Area: (1/2) * r^2 * theta
func GeomSectorArea(r, theta *ExactNum) (*ExactNum, error) {
	rSq, ok := r.Mul(r)
	if !ok {
		return nil, fmt.Errorf("ErrOverflow: radius square overflowed")
	}
	prod, ok := rSq.Mul(theta)
	if !ok {
		return nil, fmt.Errorf("ErrOverflow: sector area multiplication overflowed")
	}
	res, ok, err := prod.Div(NewExactInt(2))
	if !ok || err != nil {
		return nil, fmt.Errorf("ErrInternal: failed to divide sector area by 2")
	}
	return res, nil
}
