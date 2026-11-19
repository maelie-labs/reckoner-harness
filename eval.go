package main

import (
	"fmt"
	"math"
	"math/big"
)

// EvaluateInfixToExactNum evaluates an infix expression to an ExactNum.
func EvaluateInfixToExactNum(expr string, store *RegisterStore) (*ExactNum, error) {
	v, err := EvaluateInfixToVal(expr, store)
	if err != nil {
		return nil, err
	}
	return calcValToExactNum(v), nil
}

func valToVector(v CalcVal) (Vector, error) {
	if v.Kind == ValKindVector {
		return v.Vec, nil
	}
	if v.Kind == ValKindList {
		vec := make(Vector, len(v.List))
		for i, item := range v.List {
			vec[i] = *calcValToExactNum(item)
		}
		return vec, nil
	}
	return nil, fmt.Errorf("ErrTypeMismatch: expected vector or list, got %v", v.Kind)
}

func valToMatrix(v CalcVal) (Matrix, error) {
	if v.Kind == ValKindMatrix {
		return v.Mat, nil
	}
	if v.Kind == ValKindList {
		rows := len(v.List)
		if rows == 0 {
			return nil, fmt.Errorf("ErrDimension: empty matrix")
		}
		cols := -1
		m := make(Matrix, rows)
		for i, rowVal := range v.List {
			if rowVal.Kind != ValKindList {
				return nil, fmt.Errorf("ErrDimension: expected list of lists for matrix")
			}
			if cols == -1 {
				cols = len(rowVal.List)
				if cols == 0 {
					return nil, fmt.Errorf("ErrDimension: empty matrix row")
				}
			} else if len(rowVal.List) != cols {
				return nil, fmt.Errorf("ErrDimension: matrix rows have differing lengths (%d vs %d)", len(rowVal.List), cols)
			}
			m[i] = make([]ExactNum, cols)
			for j, cell := range rowVal.List {
				m[i][j] = *calcValToExactNum(cell)
			}
		}
		return m, nil
	}
	return nil, fmt.Errorf("ErrTypeMismatch: expected matrix or list of lists")
}

func valToPoint(v CalcVal) (Point, error) {
	vec, err := valToVector(v)
	if err != nil {
		return nil, err
	}
	if len(vec) != 2 && len(vec) != 3 {
		return nil, fmt.Errorf("ErrDimension: point must be 2D or 3D (got dimension %d)", len(vec))
	}
	return Point(vec), nil
}

func calcValToRational(v CalcVal) (*big.Rat, error) {
	if v.Kind == ValKindRat && v.Rat != nil {
		return new(big.Rat).Set(v.Rat), nil
	}
	if v.Kind == ValKindExactNum && v.Exact != nil {
		if r, ok := v.Exact.AsRational(); ok {
			return r, nil
		}
	}
	if v.Kind == ValKindFloat {
		return new(big.Rat).SetFloat64(v.Float), nil
	}
	return nil, fmt.Errorf("ErrDomain: expected rational value")
}

// evalExtendedFunction dispatches extended mathematical and CAS functions. Returns (result, handled, error).
func evalExtendedFunction(fn string, args []CalcVal, store *RegisterStore) (CalcVal, bool, error) {
	switch fn {
	case "expand", "factor", "divide", "pgcd", "roots", "evaluate":
		// Polynomial CAS functions handle symbolic expressions
	default:
		for _, arg := range args {
			if isSymbolic(arg) {
				return CalcVal{}, true, fmt.Errorf("ErrUnsupported: %s of a symbolic expression is not supported", fn)
			}
		}
	}

	switch fn {

	// -------------------------------------------------------------------------
	// 1. Complex & Trig & Logs
	// -------------------------------------------------------------------------

	case "re":
		if len(args) != 1 {
			return CalcVal{}, true, fmt.Errorf("ErrSyntax: re expects 1 argument")
		}
		ex := calcValToExactNum(args[0])
		return CalcVal{Kind: ValKindExactNum, Exact: ex.Re()}, true, nil

	case "im":
		if len(args) != 1 {
			return CalcVal{}, true, fmt.Errorf("ErrSyntax: im expects 1 argument")
		}
		ex := calcValToExactNum(args[0])
		return CalcVal{Kind: ValKindExactNum, Exact: ex.Im()}, true, nil

	case "conj":
		if len(args) != 1 {
			return CalcVal{}, true, fmt.Errorf("ErrSyntax: conj expects 1 argument")
		}
		ex := calcValToExactNum(args[0])
		return CalcVal{Kind: ValKindExactNum, Exact: ex.Conj()}, true, nil

	case "arg":
		if len(args) != 1 {
			return CalcVal{}, true, fmt.Errorf("ErrSyntax: arg expects 1 argument")
		}
		ex := calcValToExactNum(args[0])
		ang, ok := ex.Arg()
		if ok {
			return CalcVal{Kind: ValKindExactNum, Exact: ang}, true, nil
		}
		reF := ex.Re().ToFloat64()
		imF := ex.Im().ToFloat64()
		return CalcVal{Kind: ValKindFloat, Float: math.Atan2(imF, reF)}, true, nil

	case "cis":
		if len(args) != 1 {
			return CalcVal{}, true, fmt.Errorf("ErrSyntax: cis expects 1 argument")
		}
		ex := calcValToExactNum(args[0])
		cosT, okC := CosExact(ex)
		sinT, okS := SinExact(ex)
		if okC && okS {
			sinTI, _ := sinT.Mul(NewExactI())
			res := cosT.Add(sinTI)
			return CalcVal{Kind: ValKindExactNum, Exact: res}, true, nil
		}
		thF := ex.ToFloat64()
		// cis(theta) = cos(theta) + i*sin(theta)
		cosF := math.Cos(thF)
		sinF := math.Sin(thF)
		if math.Abs(sinF) < 1e-12 {
			return CalcVal{Kind: ValKindFloat, Float: cosF}, true, nil
		}
		return CalcVal{}, true, fmt.Errorf("ErrUnsupported: inexact cis float formatting")

	case "polar":
		if len(args) != 1 {
			return CalcVal{}, true, fmt.Errorf("ErrSyntax: polar expects 1 argument")
		}
		ex := calcValToExactNum(args[0])
		rVal, okR := ex.Abs()
		thetaVal, okT := ex.Arg()
		if okR && okT {
			return CalcVal{
				Kind: ValKindList,
				List: []CalcVal{
					{Kind: ValKindExactNum, Exact: rVal},
					{Kind: ValKindExactNum, Exact: thetaVal},
				},
			}, true, nil
		}
		rF := math.Hypot(ex.Re().ToFloat64(), ex.Im().ToFloat64())
		thF := math.Atan2(ex.Im().ToFloat64(), ex.Re().ToFloat64())
		return CalcVal{
			Kind: ValKindList,
			List: []CalcVal{
				{Kind: ValKindFloat, Float: rF},
				{Kind: ValKindFloat, Float: thF},
			},
		}, true, nil

	case "deg":
		if len(args) != 1 {
			return CalcVal{}, true, fmt.Errorf("ErrSyntax: deg expects 1 argument")
		}
		ex := calcValToExactNum(args[0])
		// deg(x) converts x degrees to radians: x * pi / 180
		scaled := ex.MulRational(big.NewRat(1, 180))
		rad, ok := scaled.Mul(NewExactPi())
		if !ok {
			return CalcVal{}, true, fmt.Errorf("ErrOverflow: deg conversion overflowed")
		}
		return CalcVal{Kind: ValKindExactNum, Exact: rad}, true, nil

	case "to_deg", "rad_to_deg":
		if len(args) != 1 {
			return CalcVal{}, true, fmt.Errorf("ErrSyntax: to_deg expects 1 argument")
		}
		ex := calcValToExactNum(args[0])
		// x * 180 / pi
		piTerm := NewExactPi()
		divPi, ok, err := ex.Div(piTerm)
		if ok && err == nil {
			degVal := divPi.MulRational(big.NewRat(180, 1))
			return CalcVal{Kind: ValKindExactNum, Exact: degVal}, true, nil
		}
		return CalcVal{Kind: ValKindFloat, Float: ex.ToFloat64() * 180.0 / math.Pi}, true, nil

	case "sin":
		if len(args) != 1 {
			return CalcVal{}, true, fmt.Errorf("ErrSyntax: sin expects 1 argument")
		}
		ex := calcValToExactNum(args[0])
		if s, ok := SinExact(ex); ok {
			return CalcVal{Kind: ValKindExactNum, Exact: s}, true, nil
		}
		return CalcVal{Kind: ValKindFloat, Float: math.Sin(toFloat(args[0]))}, true, nil

	case "cos":
		if len(args) != 1 {
			return CalcVal{}, true, fmt.Errorf("ErrSyntax: cos expects 1 argument")
		}
		ex := calcValToExactNum(args[0])
		if c, ok := CosExact(ex); ok {
			return CalcVal{Kind: ValKindExactNum, Exact: c}, true, nil
		}
		return CalcVal{Kind: ValKindFloat, Float: math.Cos(toFloat(args[0]))}, true, nil

	case "tan":
		if len(args) != 1 {
			return CalcVal{}, true, fmt.Errorf("ErrSyntax: tan expects 1 argument")
		}
		ex := calcValToExactNum(args[0])
		t, ok, err := TanExact(ex)
		if err != nil {
			return CalcVal{}, true, err
		}
		if ok {
			return CalcVal{Kind: ValKindExactNum, Exact: t}, true, nil
		}
		return CalcVal{Kind: ValKindFloat, Float: math.Tan(toFloat(args[0]))}, true, nil

	case "sec":
		if len(args) != 1 {
			return CalcVal{}, true, fmt.Errorf("ErrSyntax: sec expects 1 argument")
		}
		ex := calcValToExactNum(args[0])
		sec, ok, err := SecExact(ex)
		if err != nil {
			return CalcVal{}, true, err
		}
		if ok {
			return CalcVal{Kind: ValKindExactNum, Exact: sec}, true, nil
		}
		cosVal := math.Cos(toFloat(args[0]))
		if math.Abs(cosVal) < 1e-12 {
			return CalcVal{}, true, fmt.Errorf("ErrDivZero: sec is undefined when cos is 0")
		}
		return CalcVal{Kind: ValKindFloat, Float: 1.0 / cosVal}, true, nil

	case "csc":
		if len(args) != 1 {
			return CalcVal{}, true, fmt.Errorf("ErrSyntax: csc expects 1 argument")
		}
		ex := calcValToExactNum(args[0])
		csc, ok, err := CscExact(ex)
		if err != nil {
			return CalcVal{}, true, err
		}
		if ok {
			return CalcVal{Kind: ValKindExactNum, Exact: csc}, true, nil
		}
		sinVal := math.Sin(toFloat(args[0]))
		if math.Abs(sinVal) < 1e-12 {
			return CalcVal{}, true, fmt.Errorf("ErrDivZero: csc is undefined when sin is 0")
		}
		return CalcVal{Kind: ValKindFloat, Float: 1.0 / sinVal}, true, nil

	case "cot":
		if len(args) != 1 {
			return CalcVal{}, true, fmt.Errorf("ErrSyntax: cot expects 1 argument")
		}
		ex := calcValToExactNum(args[0])
		cot, ok, err := CotExact(ex)
		if err != nil {
			return CalcVal{}, true, err
		}
		if ok {
			return CalcVal{Kind: ValKindExactNum, Exact: cot}, true, nil
		}
		tanVal := math.Tan(toFloat(args[0]))
		if math.Abs(tanVal) < 1e-12 {
			return CalcVal{}, true, fmt.Errorf("ErrDivZero: cot is undefined when tan is 0")
		}
		return CalcVal{Kind: ValKindFloat, Float: 1.0 / tanVal}, true, nil

	case "arcsin", "asin":
		if len(args) != 1 {
			return CalcVal{}, true, fmt.Errorf("ErrSyntax: arcsin expects 1 argument")
		}
		ex := calcValToExactNum(args[0])
		if a, ok := AsinExact(ex); ok {
			return CalcVal{Kind: ValKindExactNum, Exact: a}, true, nil
		}
		f := toFloat(args[0])
		if f < -1.0 || f > 1.0 {
			return CalcVal{}, true, fmt.Errorf("ErrDomain: arcsin domain is [-1, 1]")
		}
		return CalcVal{Kind: ValKindFloat, Float: math.Asin(f)}, true, nil

	case "arccos", "acos":
		if len(args) != 1 {
			return CalcVal{}, true, fmt.Errorf("ErrSyntax: arccos expects 1 argument")
		}
		ex := calcValToExactNum(args[0])
		if a, ok := AcosExact(ex); ok {
			return CalcVal{Kind: ValKindExactNum, Exact: a}, true, nil
		}
		f := toFloat(args[0])
		if f < -1.0 || f > 1.0 {
			return CalcVal{}, true, fmt.Errorf("ErrDomain: arccos domain is [-1, 1]")
		}
		return CalcVal{Kind: ValKindFloat, Float: math.Acos(f)}, true, nil

	case "arctan", "atan":
		if len(args) != 1 {
			return CalcVal{}, true, fmt.Errorf("ErrSyntax: arctan expects 1 argument")
		}
		ex := calcValToExactNum(args[0])
		if a, ok := AtanExact(ex); ok {
			return CalcVal{Kind: ValKindExactNum, Exact: a}, true, nil
		}
		return CalcVal{Kind: ValKindFloat, Float: math.Atan(toFloat(args[0]))}, true, nil

	case "log":
		if len(args) == 2 {
			xRat, err1 := calcValToRational(args[0])
			bRat, err2 := calcValToRational(args[1])
			if err1 == nil && err2 == nil {
				resRat, ok := LogExact(xRat, bRat)
				if ok {
					return CalcVal{Kind: ValKindRat, Rat: resRat}, true, nil
				}
			}
			x := toFloat(args[0])
			b := toFloat(args[1])
			if x <= 0 || b <= 0 || b == 1.0 {
				return CalcVal{}, true, fmt.Errorf("ErrDomain: invalid base or argument for logarithm")
			}
			return CalcVal{Kind: ValKindFloat, Float: math.Log(x) / math.Log(b)}, true, nil
		}

	// -------------------------------------------------------------------------
	// 2. Polynomials
	// -------------------------------------------------------------------------

	case "expand":
		if len(args) != 1 {
			return CalcVal{}, true, fmt.Errorf("ErrSyntax: expand expects 1 argument")
		}
		if args[0].Kind == ValKindPoly && args[0].Poly != nil {
			return args[0], true, nil
		}
		exprStr := args[0].String()
		_, pMulti, err := ExpandPoly(exprStr, store)
		if err != nil {
			return CalcVal{}, true, err
		}
		return CalcVal{Kind: ValKindPoly, Poly: pMulti}, true, nil

	case "factor":
		if len(args) != 1 {
			return CalcVal{}, true, fmt.Errorf("ErrSyntax: factor expects 1 argument")
		}
		if args[0].Kind == ValKindRat && args[0].Rat.IsInt() {
			return CalcVal{}, false, nil
		}
		exprStr := args[0].String()
		resStr, err := FactorPoly(exprStr, store)
		if err != nil {
			return CalcVal{}, true, err
		}
		return CalcVal{Kind: ValKindRawString, Str: resStr}, true, nil

	case "divide":
		if len(args) != 2 {
			return CalcVal{}, true, fmt.Errorf("ErrSyntax: divide expects 2 arguments")
		}
		resStr, err := DividePoly(args[0].String(), args[1].String(), store)
		if err != nil {
			return CalcVal{}, true, err
		}
		return CalcVal{Kind: ValKindRawString, Str: resStr}, true, nil

	case "pgcd":
		if len(args) != 2 {
			return CalcVal{}, true, fmt.Errorf("ErrSyntax: pgcd expects 2 arguments")
		}
		resStr, err := PolyGCD(args[0].String(), args[1].String(), store)
		if err != nil {
			return CalcVal{}, true, err
		}
		return CalcVal{Kind: ValKindRawString, Str: resStr}, true, nil

	case "roots":
		if len(args) != 1 {
			return CalcVal{}, true, fmt.Errorf("ErrSyntax: roots expects 1 argument")
		}
		resStr, _, err := RootsPoly(args[0].String(), store)
		if err != nil {
			return CalcVal{}, true, err
		}
		return CalcVal{Kind: ValKindRawString, Str: resStr}, true, nil

	case "evaluate":
		if len(args) < 2 || len(args) > 3 {
			return CalcVal{}, true, fmt.Errorf("ErrSyntax: evaluate expects 2 or 3 arguments (poly, [var], val)")
		}
		if len(args) == 2 {
			// evaluate(p, val)
			resStr, err := EvaluatePoly(args[0].String(), "x", args[1].String(), store)
			if err != nil {
				return CalcVal{}, true, err
			}
			return CalcVal{Kind: ValKindRawString, Str: resStr}, true, nil
		}
		// evaluate(p, var, val)
		resStr, err := EvaluatePoly(args[0].String(), args[1].String(), args[2].String(), store)
		if err != nil {
			return CalcVal{}, true, err
		}
		return CalcVal{Kind: ValKindRawString, Str: resStr}, true, nil

	// -------------------------------------------------------------------------
	// 3. Geometry Helpers
	// -------------------------------------------------------------------------

	case "dist":
		if len(args) != 2 {
			return CalcVal{}, true, fmt.Errorf("ErrSyntax: dist expects 2 points")
		}
		p1, err1 := valToPoint(args[0])
		p2, err2 := valToPoint(args[1])
		if err1 != nil {
			return CalcVal{}, true, err1
		}
		if err2 != nil {
			return CalcVal{}, true, err2
		}
		d, ok, err := GeomDist(p1, p2)
		if err != nil {
			return CalcVal{}, true, err
		}
		if ok {
			return CalcVal{Kind: ValKindExactNum, Exact: d}, true, nil
		}
		// Float distance
		sumSq := 0.0
		for i := range p1 {
			diff := p1[i].ToFloat64() - p2[i].ToFloat64()
			sumSq += diff * diff
		}
		return CalcVal{Kind: ValKindFloat, Float: math.Sqrt(sumSq)}, true, nil

	case "midpoint":
		if len(args) != 2 {
			return CalcVal{}, true, fmt.Errorf("ErrSyntax: midpoint expects 2 points")
		}
		p1, err1 := valToPoint(args[0])
		p2, err2 := valToPoint(args[1])
		if err1 != nil {
			return CalcVal{}, true, err1
		}
		if err2 != nil {
			return CalcVal{}, true, err2
		}
		mid, err := GeomMidpoint(p1, p2)
		if err != nil {
			return CalcVal{}, true, err
		}
		var list []CalcVal
		for _, c := range mid {
			list = append(list, CalcVal{Kind: ValKindExactNum, Exact: &c})
		}
		return CalcVal{Kind: ValKindList, List: list}, true, nil

	case "slope":
		if len(args) != 2 {
			return CalcVal{}, true, fmt.Errorf("ErrSyntax: slope expects 2 points")
		}
		p1, err1 := valToPoint(args[0])
		p2, err2 := valToPoint(args[1])
		if err1 != nil {
			return CalcVal{}, true, err1
		}
		if err2 != nil {
			return CalcVal{}, true, err2
		}
		m, ok, err := GeomSlope(p1, p2)
		if err != nil {
			return CalcVal{}, true, err
		}
		if ok {
			return CalcVal{Kind: ValKindExactNum, Exact: m}, true, nil
		}
		dy := p2[1].ToFloat64() - p1[1].ToFloat64()
		dx := p2[0].ToFloat64() - p1[0].ToFloat64()
		return CalcVal{Kind: ValKindFloat, Float: dy / dx}, true, nil

	case "line":
		if len(args) != 2 {
			return CalcVal{}, true, fmt.Errorf("ErrSyntax: line expects 2 points")
		}
		p1, err1 := valToPoint(args[0])
		p2, err2 := valToPoint(args[1])
		if err1 != nil {
			return CalcVal{}, true, err1
		}
		if err2 != nil {
			return CalcVal{}, true, err2
		}
		lineStr, err := GeomLine(p1, p2)
		if err != nil {
			return CalcVal{}, true, err
		}
		return CalcVal{Kind: ValKindRawString, Str: lineStr}, true, nil

	case "area_polygon":
		var pts []Point
		// Accepts either area_polygon([p1, p2, p3]) or area_polygon(p1, p2, p3)
		argList := args
		if len(args) == 1 && args[0].Kind == ValKindList {
			argList = args[0].List
		}
		for _, a := range argList {
			p, err := valToPoint(a)
			if err != nil {
				return CalcVal{}, true, err
			}
			pts = append(pts, p)
		}
		area, err := GeomAreaPolygon(pts)
		if err != nil {
			return CalcVal{}, true, err
		}
		return CalcVal{Kind: ValKindExactNum, Exact: area}, true, nil

	case "heron":
		if len(args) != 3 {
			return CalcVal{}, true, fmt.Errorf("ErrSyntax: heron expects 3 side lengths")
		}
		a := calcValToExactNum(args[0])
		b := calcValToExactNum(args[1])
		c := calcValToExactNum(args[2])
		area, ok, err := GeomHeron(a, b, c)
		if err != nil {
			return CalcVal{}, true, err
		}
		if ok {
			return CalcVal{Kind: ValKindExactNum, Exact: area}, true, nil
		}
		sF := (a.ToFloat64() + b.ToFloat64() + c.ToFloat64()) / 2.0
		prodF := sF * (sF - a.ToFloat64()) * (sF - b.ToFloat64()) * (sF - c.ToFloat64())
		return CalcVal{Kind: ValKindFloat, Float: math.Sqrt(prodF)}, true, nil

	case "triangle_area":
		if len(args) != 3 {
			return CalcVal{}, true, fmt.Errorf("ErrSyntax: triangle_area expects 3 arguments (a, b, C)")
		}
		a := calcValToExactNum(args[0])
		b := calcValToExactNum(args[1])
		C := calcValToExactNum(args[2])
		area, ok, err := GeomTriangleArea(a, b, C)
		if err != nil {
			return CalcVal{}, true, err
		}
		if ok {
			return CalcVal{Kind: ValKindExactNum, Exact: area}, true, nil
		}
		return CalcVal{Kind: ValKindFloat, Float: 0.5 * a.ToFloat64() * b.ToFloat64() * math.Sin(C.ToFloat64())}, true, nil

	case "law_cos_side":
		if len(args) != 3 {
			return CalcVal{}, true, fmt.Errorf("ErrSyntax: law_cos_side expects 3 arguments (a, b, C)")
		}
		a := calcValToExactNum(args[0])
		b := calcValToExactNum(args[1])
		C := calcValToExactNum(args[2])
		side, ok, err := GeomLawCosSide(a, b, C)
		if err != nil {
			return CalcVal{}, true, err
		}
		if ok {
			return CalcVal{Kind: ValKindExactNum, Exact: side}, true, nil
		}
		c2 := a.ToFloat64()*a.ToFloat64() + b.ToFloat64()*b.ToFloat64() - 2*a.ToFloat64()*b.ToFloat64()*math.Cos(C.ToFloat64())
		return CalcVal{Kind: ValKindFloat, Float: math.Sqrt(c2)}, true, nil

	case "law_cos_angle":
		if len(args) != 3 {
			return CalcVal{}, true, fmt.Errorf("ErrSyntax: law_cos_angle expects 3 side lengths (a, b, c)")
		}
		a := calcValToExactNum(args[0])
		b := calcValToExactNum(args[1])
		c := calcValToExactNum(args[2])
		ang, ok, err := GeomLawCosAngle(a, b, c)
		if err != nil {
			return CalcVal{}, true, err
		}
		if ok {
			return CalcVal{Kind: ValKindExactNum, Exact: ang}, true, nil
		}
		aF, bF, cF := a.ToFloat64(), b.ToFloat64(), c.ToFloat64()
		cosVal := (aF*aF + bF*bF - cF*cF) / (2 * aF * bF)
		return CalcVal{Kind: ValKindFloat, Float: math.Acos(cosVal)}, true, nil

	case "circ_area":
		if len(args) != 1 {
			return CalcVal{}, true, fmt.Errorf("ErrSyntax: circ_area expects radius")
		}
		r := calcValToExactNum(args[0])
		area, err := GeomCircArea(r)
		if err != nil {
			return CalcVal{}, true, err
		}
		return CalcVal{Kind: ValKindExactNum, Exact: area}, true, nil

	case "circumference":
		if len(args) != 1 {
			return CalcVal{}, true, fmt.Errorf("ErrSyntax: circumference expects radius")
		}
		r := calcValToExactNum(args[0])
		circ, err := GeomCircumference(r)
		if err != nil {
			return CalcVal{}, true, err
		}
		return CalcVal{Kind: ValKindExactNum, Exact: circ}, true, nil

	case "arc_length":
		if len(args) != 2 {
			return CalcVal{}, true, fmt.Errorf("ErrSyntax: arc_length expects r, theta")
		}
		r := calcValToExactNum(args[0])
		th := calcValToExactNum(args[1])
		al, err := GeomArcLength(r, th)
		if err != nil {
			return CalcVal{}, true, err
		}
		return CalcVal{Kind: ValKindExactNum, Exact: al}, true, nil

	case "sector_area":
		if len(args) != 2 {
			return CalcVal{}, true, fmt.Errorf("ErrSyntax: sector_area expects r, theta")
		}
		r := calcValToExactNum(args[0])
		th := calcValToExactNum(args[1])
		sa, err := GeomSectorArea(r, th)
		if err != nil {
			return CalcVal{}, true, err
		}
		return CalcVal{Kind: ValKindExactNum, Exact: sa}, true, nil

	// -------------------------------------------------------------------------
	// 4. Linear Algebra
	// -------------------------------------------------------------------------

	case "det":
		if len(args) == 4 {
			ad, err := mulCalcVals(args[0], args[3])
			if err != nil {
				return CalcVal{}, true, err
			}
			bc, err := mulCalcVals(args[1], args[2])
			if err != nil {
				return CalcVal{}, true, err
			}
			res, err := subCalcVals(ad, bc)
			return res, true, err
		}
		if len(args) != 1 {
			return CalcVal{}, true, fmt.Errorf("ErrSyntax: det expects 1 matrix or 4 elements for 2x2")
		}
		m, err := valToMatrix(args[0])
		if err != nil {
			return CalcVal{}, true, err
		}
		d, err := m.Det()
		if err != nil {
			return CalcVal{}, true, err
		}
		if rat, isR := d.AsRational(); isR {
			return CalcVal{Kind: ValKindRat, Rat: rat}, true, nil
		}
		return CalcVal{Kind: ValKindExactNum, Exact: d}, true, nil

	case "inv":
		if len(args) != 1 {
			return CalcVal{}, true, fmt.Errorf("ErrSyntax: inv expects 1 matrix")
		}
		m, err := valToMatrix(args[0])
		if err != nil {
			return CalcVal{}, true, err
		}
		invM, err := m.Inv()
		if err != nil {
			return CalcVal{}, true, err
		}
		return CalcVal{Kind: ValKindMatrix, Mat: invM}, true, nil

	case "transpose":
		if len(args) != 1 {
			return CalcVal{}, true, fmt.Errorf("ErrSyntax: transpose expects 1 matrix")
		}
		m, err := valToMatrix(args[0])
		if err != nil {
			return CalcVal{}, true, err
		}
		return CalcVal{Kind: ValKindMatrix, Mat: m.Transpose()}, true, nil

	case "mul":
		if len(args) != 2 {
			return CalcVal{}, true, fmt.Errorf("ErrSyntax: mul expects 2 matrices")
		}
		m1, err1 := valToMatrix(args[0])
		m2, err2 := valToMatrix(args[1])
		if err1 != nil {
			return CalcVal{}, true, err1
		}
		if err2 != nil {
			return CalcVal{}, true, err2
		}
		prod, err := m1.Mul(m2)
		if err != nil {
			return CalcVal{}, true, err
		}
		return CalcVal{Kind: ValKindMatrix, Mat: prod}, true, nil

	case "rank":
		if len(args) != 1 {
			return CalcVal{}, true, fmt.Errorf("ErrSyntax: rank expects 1 matrix")
		}
		m, err := valToMatrix(args[0])
		if err != nil {
			return CalcVal{}, true, err
		}
		rk, err := m.Rank()
		if err != nil {
			return CalcVal{}, true, err
		}
		return CalcVal{Kind: ValKindRat, Rat: big.NewRat(int64(rk), 1)}, true, nil

	case "trace":
		if len(args) != 1 {
			return CalcVal{}, true, fmt.Errorf("ErrSyntax: trace expects 1 matrix")
		}
		m, err := valToMatrix(args[0])
		if err != nil {
			return CalcVal{}, true, err
		}
		tr, err := m.Trace()
		if err != nil {
			return CalcVal{}, true, err
		}
		return CalcVal{Kind: ValKindExactNum, Exact: tr}, true, nil

	case "solve_linear":
		if len(args) != 2 {
			return CalcVal{}, true, fmt.Errorf("ErrSyntax: solve_linear expects matrix A and vector b")
		}
		m, err1 := valToMatrix(args[0])
		b, err2 := valToVector(args[1])
		if err1 != nil {
			return CalcVal{}, true, err1
		}
		if err2 != nil {
			return CalcVal{}, true, err2
		}
		sol, err := SolveLinearMatrix(m, b)
		if err != nil {
			return CalcVal{}, true, err
		}
		return CalcVal{Kind: ValKindVector, Vec: sol}, true, nil

	case "rotation":
		if len(args) != 1 {
			return CalcVal{}, true, fmt.Errorf("ErrSyntax: rotation expects angle theta")
		}
		th := calcValToExactNum(args[0])
		rot, err := RotationMatrix(th)
		if err != nil {
			return CalcVal{}, true, err
		}
		return CalcVal{Kind: ValKindMatrix, Mat: rot}, true, nil

	case "eigen":
		if len(args) != 1 {
			return CalcVal{}, true, fmt.Errorf("ErrSyntax: eigen expects 2x2 matrix")
		}
		m, err := valToMatrix(args[0])
		if err != nil {
			return CalcVal{}, true, err
		}
		vals, err := Eigen2x2(m)
		if err != nil {
			return CalcVal{}, true, err
		}
		var list []CalcVal
		for _, v := range vals {
			list = append(list, CalcVal{Kind: ValKindExactNum, Exact: &v})
		}
		return CalcVal{Kind: ValKindList, List: list}, true, nil

	case "dot":
		if len(args) == 4 {
			u := Vector{*calcValToExactNum(args[0]), *calcValToExactNum(args[1])}
			v := Vector{*calcValToExactNum(args[2]), *calcValToExactNum(args[3])}
			d, err := VecDot(u, v)
			if err != nil {
				return CalcVal{}, true, err
			}
			if rat, isR := d.AsRational(); isR {
				return CalcVal{Kind: ValKindRat, Rat: rat}, true, nil
			}
			return CalcVal{Kind: ValKindExactNum, Exact: d}, true, nil
		}
		if len(args) == 6 {
			u := Vector{*calcValToExactNum(args[0]), *calcValToExactNum(args[1]), *calcValToExactNum(args[2])}
			v := Vector{*calcValToExactNum(args[3]), *calcValToExactNum(args[4]), *calcValToExactNum(args[5])}
			d, err := VecDot(u, v)
			if err != nil {
				return CalcVal{}, true, err
			}
			if rat, isR := d.AsRational(); isR {
				return CalcVal{Kind: ValKindRat, Rat: rat}, true, nil
			}
			return CalcVal{Kind: ValKindExactNum, Exact: d}, true, nil
		}
		if len(args) != 2 {
			return CalcVal{}, true, fmt.Errorf("ErrSyntax: dot expects 2 vectors or 4/6 coordinates")
		}
		u, err1 := valToVector(args[0])
		v, err2 := valToVector(args[1])
		if err1 != nil {
			return CalcVal{}, true, err1
		}
		if err2 != nil {
			return CalcVal{}, true, err2
		}
		d, err := VecDot(u, v)
		if err != nil {
			return CalcVal{}, true, err
		}
		if rat, isR := d.AsRational(); isR {
			return CalcVal{Kind: ValKindRat, Rat: rat}, true, nil
		}
		return CalcVal{Kind: ValKindExactNum, Exact: d}, true, nil

	case "cross":
		if len(args) == 6 {
			u := Vector{*calcValToExactNum(args[0]), *calcValToExactNum(args[1]), *calcValToExactNum(args[2])}
			v := Vector{*calcValToExactNum(args[3]), *calcValToExactNum(args[4]), *calcValToExactNum(args[5])}
			c, err := VecCross(u, v)
			if err != nil {
				return CalcVal{}, true, err
			}
			return CalcVal{Kind: ValKindVector, Vec: c}, true, nil
		}
		if len(args) != 2 {
			return CalcVal{}, true, fmt.Errorf("ErrSyntax: cross expects two 3D vectors or 6 coordinates")
		}
		u, err1 := valToVector(args[0])
		v, err2 := valToVector(args[1])
		if err1 != nil {
			return CalcVal{}, true, err1
		}
		if err2 != nil {
			return CalcVal{}, true, err2
		}
		c, err := VecCross(u, v)
		if err != nil {
			return CalcVal{}, true, err
		}
		return CalcVal{Kind: ValKindVector, Vec: c}, true, nil

	case "norm":
		if len(args) != 1 {
			return CalcVal{}, true, fmt.Errorf("ErrSyntax: norm expects 1 vector")
		}
		u, err := valToVector(args[0])
		if err != nil {
			return CalcVal{}, true, err
		}
		n, ok, err := VecNorm(u)
		if err != nil {
			return CalcVal{}, true, err
		}
		if ok {
			return CalcVal{Kind: ValKindExactNum, Exact: n}, true, nil
		}
		sumSq := 0.0
		for _, coord := range u {
			f := coord.ToFloat64()
			sumSq += f * f
		}
		return CalcVal{Kind: ValKindFloat, Float: math.Sqrt(sumSq)}, true, nil

	case "unit":
		if len(args) != 1 {
			return CalcVal{}, true, fmt.Errorf("ErrSyntax: unit expects 1 vector")
		}
		u, err := valToVector(args[0])
		if err != nil {
			return CalcVal{}, true, err
		}
		un, err := VecUnit(u)
		if err != nil {
			return CalcVal{}, true, err
		}
		return CalcVal{Kind: ValKindVector, Vec: un}, true, nil

	case "proj":
		if len(args) != 2 {
			return CalcVal{}, true, fmt.Errorf("ErrSyntax: proj expects 2 vectors (u, v)")
		}
		u, err1 := valToVector(args[0])
		v, err2 := valToVector(args[1])
		if err1 != nil {
			return CalcVal{}, true, err1
		}
		if err2 != nil {
			return CalcVal{}, true, err2
		}
		pr, err := VecProj(u, v)
		if err != nil {
			return CalcVal{}, true, err
		}
		return CalcVal{Kind: ValKindVector, Vec: pr}, true, nil

	case "angle":
		if len(args) != 2 {
			return CalcVal{}, true, fmt.Errorf("ErrSyntax: angle expects 2 vectors")
		}
		u, err1 := valToVector(args[0])
		v, err2 := valToVector(args[1])
		if err1 != nil {
			return CalcVal{}, true, err1
		}
		if err2 != nil {
			return CalcVal{}, true, err2
		}
		ang, ok, err := VecAngle(u, v)
		if err != nil {
			return CalcVal{}, true, err
		}
		if ok {
			return CalcVal{Kind: ValKindExactNum, Exact: ang}, true, nil
		}
		dotF, _ := VecDot(u, v)
		normU, _, _ := VecNorm(u)
		normV, _, _ := VecNorm(v)
		cosF := dotF.ToFloat64() / (normU.ToFloat64() * normV.ToFloat64())
		return CalcVal{Kind: ValKindFloat, Float: math.Acos(cosF)}, true, nil

	// -------------------------------------------------------------------------
	// 5. Distributions
	// -------------------------------------------------------------------------

	case "binom_pmf":
		if len(args) != 3 {
			return CalcVal{}, true, fmt.Errorf("ErrSyntax: binom_pmf expects (k, n, p)")
		}
		k := int64(toFloat(args[0]))
		n := int64(toFloat(args[1]))
		pRat, err := calcValToRational(args[2])
		if err != nil {
			return CalcVal{}, true, err
		}
		res, err := BinomPMF(k, n, pRat)
		if err != nil {
			return CalcVal{}, true, err
		}
		return CalcVal{Kind: ValKindRat, Rat: res}, true, nil

	case "binom_cdf":
		if len(args) != 3 {
			return CalcVal{}, true, fmt.Errorf("ErrSyntax: binom_cdf expects (k, n, p)")
		}
		k := int64(toFloat(args[0]))
		n := int64(toFloat(args[1]))
		pRat, err := calcValToRational(args[2])
		if err != nil {
			return CalcVal{}, true, err
		}
		res, err := BinomCDF(k, n, pRat)
		if err != nil {
			return CalcVal{}, true, err
		}
		return CalcVal{Kind: ValKindRat, Rat: res}, true, nil
	}

	return CalcVal{}, false, nil
}
