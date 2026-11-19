package main

import (
	"fmt"
	"math"
	"math/big"
	"strings"
)

// Matrix represents an m x n matrix with exact entries.
type Matrix [][]ExactNum

// Vector represents a vector of exact entries.
type Vector []ExactNum

func (m Matrix) Rows() int {
	return len(m)
}

func (m Matrix) Cols() int {
	if len(m) == 0 {
		return 0
	}
	return len(m[0])
}

func (m Matrix) String() string {
	var rowStrs []string
	for _, row := range m {
		var colStrs []string
		for _, cell := range row {
			colStrs = append(colStrs, cell.String())
		}
		rowStrs = append(rowStrs, fmt.Sprintf("[%s]", strings.Join(colStrs, ", ")))
	}
	return fmt.Sprintf("[%s]", strings.Join(rowStrs, ", "))
}

func (v Vector) String() string {
	var parts []string
	for _, c := range v {
		parts = append(parts, c.String())
	}
	return fmt.Sprintf("[%s]", strings.Join(parts, ", "))
}

func NewMatrix(rows, cols int) (Matrix, error) {
	if rows <= 0 || cols <= 0 {
		return nil, fmt.Errorf("ErrDimension: matrix dimensions must be positive")
	}
	if rows > 10 || cols > 10 {
		return nil, fmt.Errorf("ErrDimension: matrix dimensions exceed limit of 10x10")
	}
	m := make(Matrix, rows)
	for i := range m {
		m[i] = make([]ExactNum, cols)
		for j := range m[i] {
			m[i][j] = *NewExactZero()
		}
	}
	return m, nil
}

func (m Matrix) Clone() Matrix {
	res := make(Matrix, len(m))
	for i := range m {
		res[i] = make([]ExactNum, len(m[i]))
		for j := range m[i] {
			res[i][j] = *m[i][j].Clone()
		}
	}
	return res
}

// Add computes matrix sum m + other.
func (m Matrix) Add(other Matrix) (Matrix, error) {
	if m.Rows() != other.Rows() || m.Cols() != other.Cols() {
		return nil, fmt.Errorf("ErrDimension: matrix addition dimension mismatch (%dx%d vs %dx%d)", m.Rows(), m.Cols(), other.Rows(), other.Cols())
	}
	res, _ := NewMatrix(m.Rows(), m.Cols())
	for i := 0; i < m.Rows(); i++ {
		for j := 0; j < m.Cols(); j++ {
			res[i][j] = *m[i][j].Add(&other[i][j])
		}
	}
	return res, nil
}

// Sub computes matrix difference m - other.
func (m Matrix) Sub(other Matrix) (Matrix, error) {
	if m.Rows() != other.Rows() || m.Cols() != other.Cols() {
		return nil, fmt.Errorf("ErrDimension: matrix subtraction dimension mismatch (%dx%d vs %dx%d)", m.Rows(), m.Cols(), other.Rows(), other.Cols())
	}
	res, _ := NewMatrix(m.Rows(), m.Cols())
	for i := 0; i < m.Rows(); i++ {
		for j := 0; j < m.Cols(); j++ {
			res[i][j] = *m[i][j].Sub(&other[i][j])
		}
	}
	return res, nil
}

// MulScalar multiplies matrix by a scalar.
func (m Matrix) MulScalar(s *ExactNum) Matrix {
	res, _ := NewMatrix(m.Rows(), m.Cols())
	for i := 0; i < m.Rows(); i++ {
		for j := 0; j < m.Cols(); j++ {
			p, _ := m[i][j].Mul(s)
			if p != nil {
				res[i][j] = *p
			}
		}
	}
	return res
}

// Neg negates matrix -m.
func (m Matrix) Neg() Matrix {
	return m.MulScalar(NewExactInt(-1))
}

// Det computes determinant of square matrix (dim <= 10).
func (m Matrix) Det() (*ExactNum, error) {
	n := m.Rows()
	if n != m.Cols() {
		return nil, fmt.Errorf("ErrDimension: determinant requires a square matrix (%dx%d)", n, m.Cols())
	}
	if n > 10 {
		return nil, fmt.Errorf("ErrDimension: matrix size exceeds limit of 10x10")
	}
	if n == 1 {
		return m[0][0].Clone(), nil
	}
	if n == 2 {
		// a*d - b*c
		ad, ok1 := m[0][0].Mul(&m[1][1])
		bc, ok2 := m[0][1].Mul(&m[1][0])
		if !ok1 || !ok2 {
			return nil, fmt.Errorf("ErrOverflow: matrix determinant multiplication overflowed")
		}
		return ad.Sub(bc), nil
	}

	// Gaussian elimination over ExactNum
	a := m.Clone()
	detSign := 1
	detMultiplier := NewExactInt(1)

	for col := 0; col < n; col++ {
		// Find non-zero pivot
		pivotRow := -1
		for row := col; row < n; row++ {
			if !a[row][col].IsZero() {
				pivotRow = row
				break
			}
		}

		if pivotRow == -1 {
			// Singular matrix -> det = 0
			return NewExactZero(), nil
		}

		if pivotRow != col {
			a[col], a[pivotRow] = a[pivotRow], a[col]
			detSign = -detSign
		}

		pivot := &a[col][col]
		var ok bool
		detMultiplier, ok = detMultiplier.Mul(pivot)
		if !ok {
			return nil, fmt.Errorf("ErrOverflow: determinant product overflowed")
		}

		for row := col + 1; row < n; row++ {
			if a[row][col].IsZero() {
				continue
			}
			factor, ok, err := a[row][col].Div(pivot)
			if !ok || err != nil {
				return nil, fmt.Errorf("ErrUnsupported: determinant pivot division failed")
			}
			for j := col; j < n; j++ {
				term, ok := factor.Mul(&a[col][j])
				if !ok {
					return nil, fmt.Errorf("ErrOverflow: elimination multiplication overflowed")
				}
				a[row][j] = *a[row][j].Sub(term)
			}
		}
	}

	if detSign < 0 {
		detMultiplier = detMultiplier.Neg()
	}
	return detMultiplier, nil
}

// Transpose computes transpose of m.
func (m Matrix) Transpose() Matrix {
	r := m.Rows()
	c := m.Cols()
	res, _ := NewMatrix(c, r)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			res[j][i] = *m[i][j].Clone()
		}
	}
	return res
}

// Trace computes trace of a square matrix.
func (m Matrix) Trace() (*ExactNum, error) {
	if m.Rows() != m.Cols() {
		return nil, fmt.Errorf("ErrDimension: trace requires a square matrix")
	}
	res := NewExactZero()
	for i := 0; i < m.Rows(); i++ {
		res = res.Add(&m[i][i])
	}
	return res, nil
}

// Mul multiplies two matrices (or @ operator).
func (m Matrix) Mul(other Matrix) (Matrix, error) {
	if m.Cols() != other.Rows() {
		return nil, fmt.Errorf("ErrDimension: incompatible dimensions for matrix multiplication (%dx%d vs %dx%d)", m.Rows(), m.Cols(), other.Rows(), other.Cols())
	}
	r := m.Rows()
	c := other.Cols()
	kLen := m.Cols()

	if r > 10 || c > 10 {
		return nil, fmt.Errorf("ErrDimension: result matrix dimension exceeds limit of 10x10")
	}

	res, err := NewMatrix(r, c)
	if err != nil {
		return nil, err
	}

	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			sum := NewExactZero()
			for k := 0; k < kLen; k++ {
				prod, ok := m[i][k].Mul(&other[k][j])
				if !ok {
					return nil, fmt.Errorf("ErrOverflow: matrix multiplication overflowed")
				}
				sum = sum.Add(prod)
			}
			res[i][j] = *sum
		}
	}
	return res, nil
}

// Inv computes matrix inverse via Gauss-Jordan elimination.
func (m Matrix) Inv() (Matrix, error) {
	n := m.Rows()
	if n != m.Cols() {
		return nil, fmt.Errorf("ErrDimension: inverse requires a square matrix")
	}
	if n > 10 {
		return nil, fmt.Errorf("ErrDimension: matrix size exceeds limit of 10x10")
	}

	det, err := m.Det()
	if err != nil {
		return nil, err
	}
	if det.IsZero() {
		return nil, fmt.Errorf("ErrNoSolution: singular matrix (det is 0)")
	}

	// Augmented matrix [A | I]
	aug := make([][]ExactNum, n)
	for i := range aug {
		aug[i] = make([]ExactNum, 2*n)
		for j := 0; j < n; j++ {
			aug[i][j] = *m[i][j].Clone()
		}
		for j := 0; j < n; j++ {
			if i == j {
				aug[i][n+j] = *NewExactInt(1)
			} else {
				aug[i][n+j] = *NewExactZero()
			}
		}
	}

	for col := 0; col < n; col++ {
		pivotRow := -1
		for row := col; row < n; row++ {
			if !aug[row][col].IsZero() {
				pivotRow = row
				break
			}
		}
		if pivotRow == -1 {
			return nil, fmt.Errorf("ErrNoSolution: singular matrix")
		}

		if pivotRow != col {
			aug[col], aug[pivotRow] = aug[pivotRow], aug[col]
		}

		pivot := &aug[col][col]
		invPivot, ok, err := NewExactInt(1).Div(pivot)
		if !ok || err != nil {
			return nil, fmt.Errorf("ErrUnsupported: division by pivot failed in inverse")
		}

		// Normalize pivot row
		for j := col; j < 2*n; j++ {
			val, ok := aug[col][j].Mul(invPivot)
			if !ok {
				return nil, fmt.Errorf("ErrOverflow: matrix inverse normalization overflowed")
			}
			aug[col][j] = *val
		}

		// Eliminate other rows
		for row := 0; row < n; row++ {
			if row == col || aug[row][col].IsZero() {
				continue
			}
			factor := aug[row][col].Clone()
			for j := col; j < 2*n; j++ {
				term, ok := factor.Mul(&aug[col][j])
				if !ok {
					return nil, fmt.Errorf("ErrOverflow: elimination multiplication overflowed")
				}
				aug[row][j] = *aug[row][j].Sub(term)
			}
		}
	}

	invM, _ := NewMatrix(n, n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			invM[i][j] = *aug[i][n+j].Clone()
		}
	}
	return invM, nil
}

// Rank computes matrix rank.
func (m Matrix) Rank() (int, error) {
	r := m.Rows()
	c := m.Cols()
	a := m.Clone()

	lead := 0
	for row := 0; row < r; row++ {
		if lead >= c {
			break
		}
		i := row
		for a[i][lead].IsZero() {
			i++
			if i == r {
				i = row
				lead++
				if lead == c {
					break
				}
			}
		}
		if lead == c {
			break
		}
		a[i], a[row] = a[row], a[i]

		pivot := &a[row][lead]
		for k := 0; k < r; k++ {
			if k != row && !a[k][lead].IsZero() {
				factor, ok, err := a[k][lead].Div(pivot)
				if !ok || err != nil {
					return 0, fmt.Errorf("ErrUnsupported: pivot division failed in rank")
				}
				for j := 0; j < c; j++ {
					term, _ := factor.Mul(&a[row][j])
					a[k][j] = *a[k][j].Sub(term)
				}
			}
		}
		lead++
	}

	rank := 0
	for row := 0; row < r; row++ {
		allZero := true
		for col := 0; col < c; col++ {
			if !a[row][col].IsZero() {
				allZero = false
				break
			}
		}
		if !allZero {
			rank++
		}
	}
	return rank, nil
}

// Pow computes matrix integer powers A^k.
func (m Matrix) Pow(exp int64) (Matrix, error) {
	if m.Rows() != m.Cols() {
		return nil, fmt.Errorf("ErrDimension: matrix power requires a square matrix")
	}
	if exp < -100 || exp > 100 {
		return nil, fmt.Errorf("ErrOverflow: matrix power exponent must be between -100 and 100")
	}
	n := m.Rows()
	if exp == 0 {
		id, _ := NewMatrix(n, n)
		for i := 0; i < n; i++ {
			id[i][i] = *NewExactInt(1)
		}
		return id, nil
	}
	if exp < 0 {
		inv, err := m.Inv()
		if err != nil {
			return nil, err
		}
		return inv.Pow(-exp)
	}

	res, _ := NewMatrix(n, n)
	for i := 0; i < n; i++ {
		res[i][i] = *NewExactInt(1)
	}

	base := m.Clone()
	curExp := exp

	for curExp > 0 {
		if curExp%2 == 1 {
			var err error
			res, err = res.Mul(base)
			if err != nil {
				return nil, err
			}
		}
		if curExp > 1 {
			var err error
			base, err = base.Mul(base)
			if err != nil {
				return nil, err
			}
		}
		curExp /= 2
	}
	return res, nil
}

// SolveLinear solves A * x = b.
func SolveLinearMatrix(A Matrix, b Vector) (Vector, error) {
	n := A.Rows()
	if n != A.Cols() {
		return nil, fmt.Errorf("ErrDimension: solve_linear requires square coefficient matrix (%dx%d)", n, A.Cols())
	}
	if len(b) != n {
		return nil, fmt.Errorf("ErrDimension: vector b length %d does not match matrix rows %d", len(b), n)
	}

	invA, err := A.Inv()
	if err != nil {
		return nil, err
	}

	// x = invA * b
	x := make(Vector, n)
	for i := 0; i < n; i++ {
		sum := NewExactZero()
		for j := 0; j < n; j++ {
			prod, ok := invA[i][j].Mul(&b[j])
			if !ok {
				return nil, fmt.Errorf("ErrOverflow: matrix vector product overflowed")
			}
			sum = sum.Add(prod)
		}
		x[i] = *sum
	}
	return x, nil
}

// Rotation computes 2x2 rotation matrix [[cos(theta), -sin(theta)], [sin(theta), cos(theta)]].
func RotationMatrix(theta *ExactNum) (Matrix, error) {
	cosT, ok1 := CosExact(theta)
	sinT, ok2 := SinExact(theta)
	if !ok1 || !ok2 {
		return nil, fmt.Errorf("ErrUnsupported: exact rotation matrix requires standard angle")
	}

	m, _ := NewMatrix(2, 2)
	m[0][0] = *cosT
	m[0][1] = *sinT.Neg()
	m[1][0] = *sinT
	m[1][1] = *cosT
	return m, nil
}

// Eigen2x2 computes exact eigenvalues for 2x2 matrix: lambda = (tr +- sqrt(tr^2 - 4*det)) / 2.
func Eigen2x2(m Matrix) ([]ExactNum, error) {
	if m.Rows() != 2 || m.Cols() != 2 {
		return nil, fmt.Errorf("ErrDimension: eigen only supported for 2x2 matrices")
	}

	tr, err := m.Trace()
	if err != nil {
		return nil, err
	}
	det, err := m.Det()
	if err != nil {
		return nil, err
	}

	trSq, ok1 := tr.Mul(tr)
	fourDet, ok2 := NewExactInt(4).Mul(det)
	if !ok1 || !ok2 {
		return nil, fmt.Errorf("ErrOverflow: eigenvalue discriminant overflowed")
	}

	disc := trSq.Sub(fourDet)
	sqrtDisc, ok := disc.Sqrt()
	if !ok {
		return nil, fmt.Errorf("ErrUnsupported: eigenvalue discriminant cannot be evaluated exactly")
	}

	r1Num := tr.Sub(sqrtDisc)
	r2Num := tr.Add(sqrtDisc)

	two := NewExactInt(2)
	r1, ok1, _ := r1Num.Div(two)
	r2, ok2, _ := r2Num.Div(two)
	if !ok1 || !ok2 {
		return nil, fmt.Errorf("ErrInternal: failed to divide eigenvalues by 2")
	}

	return []ExactNum{*r1, *r2}, nil
}

// -----------------------------------------------------------------------------
// Vector Operations
// -----------------------------------------------------------------------------

func VecDot(u, v Vector) (*ExactNum, error) {
	if len(u) != len(v) {
		return nil, fmt.Errorf("ErrDimension: vector lengths do not match (%d vs %d)", len(u), len(v))
	}
	sum := NewExactZero()
	for i := range u {
		prod, ok := u[i].Mul(&v[i])
		if !ok {
			return nil, fmt.Errorf("ErrOverflow: dot product overflowed")
		}
		sum = sum.Add(prod)
	}
	return sum, nil
}

func VecCross(u, v Vector) (Vector, error) {
	if len(u) != 3 || len(v) != 3 {
		return nil, fmt.Errorf("ErrDimension: cross product requires 3D vectors")
	}
	// [u2*v3 - u3*v2, u3*v1 - u1*v3, u1*v2 - u2*v1]
	p1, ok1 := u[1].Mul(&v[2])
	p2, ok2 := u[2].Mul(&v[1])
	c0 := p1.Sub(p2)

	p3, ok3 := u[2].Mul(&v[0])
	p4, ok4 := u[0].Mul(&v[2])
	c1 := p3.Sub(p4)

	p5, ok5 := u[0].Mul(&v[1])
	p6, ok6 := u[1].Mul(&v[0])
	c2 := p5.Sub(p6)

	if !ok1 || !ok2 || !ok3 || !ok4 || !ok5 || !ok6 {
		return nil, fmt.Errorf("ErrOverflow: cross product multiplication overflowed")
	}
	return Vector{*c0, *c1, *c2}, nil
}

func VecNorm(u Vector) (*ExactNum, bool, error) {
	normSq, err := VecDot(u, u)
	if err != nil {
		return nil, false, err
	}
	res, ok := normSq.Sqrt()
	return res, ok, nil
}

func VecUnit(u Vector) (Vector, error) {
	norm, ok, err := VecNorm(u)
	if err != nil {
		return nil, err
	}
	if !ok {
		return nil, fmt.Errorf("ErrUnsupported: vector norm could not be evaluated exactly")
	}
	if norm.IsZero() {
		return nil, fmt.Errorf("ErrDivZero: zero vector has no unit vector")
	}

	res := make(Vector, len(u))
	for i := range u {
		coord, ok, err := u[i].Div(norm)
		if !ok || err != nil {
			return nil, fmt.Errorf("ErrUnsupported: failed to normalize vector coordinate")
		}
		res[i] = *coord
	}
	return res, nil
}

func VecProj(u, v Vector) (Vector, error) {
	dotUV, err := VecDot(u, v)
	if err != nil {
		return nil, err
	}
	dotVV, err := VecDot(v, v)
	if err != nil {
		return nil, err
	}
	if dotVV.IsZero() {
		return nil, fmt.Errorf("ErrDivZero: cannot project onto zero vector")
	}

	scalar, ok, err := dotUV.Div(dotVV)
	if !ok || err != nil {
		return nil, fmt.Errorf("ErrUnsupported: projection scalar division failed")
	}

	res := make(Vector, len(v))
	for i := range v {
		coord, ok := v[i].Mul(scalar)
		if !ok {
			return nil, fmt.Errorf("ErrOverflow: projection scalar multiplication overflowed")
		}
		res[i] = *coord
	}
	return res, nil
}

func VecAngle(u, v Vector) (*ExactNum, bool, error) {
	dotUV, err := VecDot(u, v)
	if err != nil {
		return nil, false, err
	}
	normU, ok1, err1 := VecNorm(u)
	normV, ok2, err2 := VecNorm(v)
	if err1 != nil || err2 != nil {
		return nil, false, fmt.Errorf("ErrDimension: error calculating norms")
	}
	if (ok1 && normU.IsZero()) || (ok2 && normV.IsZero()) {
		return nil, false, fmt.Errorf("ErrDivZero: zero vector has undefined angle")
	}

	if ok1 && ok2 {
		denom, ok := normU.Mul(normV)
		if ok {
			cosTheta, ok, err := dotUV.Div(denom)
			if ok && err == nil {
				ang, ok := AcosExact(cosTheta)
				if ok {
					return ang, true, nil
				}
			}
		}
	}

	// Float fallback
	uF := make([]float64, len(u))
	vF := make([]float64, len(v))
	dotF := 0.0
	normSqUF := 0.0
	normSqVF := 0.0
	for i := range u {
		uF[i] = u[i].ToFloat64()
		vF[i] = v[i].ToFloat64()
		dotF += uF[i] * vF[i]
		normSqUF += uF[i] * uF[i]
		normSqVF += vF[i] * vF[i]
	}
	cosF := dotF / (math.Sqrt(normSqUF) * math.Sqrt(normSqVF))
	if cosF > 1.0 {
		cosF = 1.0
	} else if cosF < -1.0 {
		cosF = -1.0
	}
	thetaF := math.Acos(cosF)

	// Check if thetaF is a standard multiple of pi/12
	kFloat := thetaF * 12.0 / math.Pi
	kRound := math.Round(kFloat)
	if math.Abs(kFloat-kRound) < 1e-9 {
		k := int64(kRound)
		rat := big.NewRat(k, 12)
		return NewExactPi().MulRational(rat), true, nil
	}

	return nil, false, nil
}

// Add computes vector sum v + other.
func (v Vector) Add(other Vector) (Vector, error) {
	if len(v) != len(other) {
		return nil, fmt.Errorf("ErrDimension: vector addition dimension mismatch (%d vs %d)", len(v), len(other))
	}
	res := make(Vector, len(v))
	for i := range v {
		res[i] = *v[i].Add(&other[i])
	}
	return res, nil
}

// Sub computes vector difference v - other.
func (v Vector) Sub(other Vector) (Vector, error) {
	if len(v) != len(other) {
		return nil, fmt.Errorf("ErrDimension: vector subtraction dimension mismatch (%d vs %d)", len(v), len(other))
	}
	res := make(Vector, len(v))
	for i := range v {
		res[i] = *v[i].Sub(&other[i])
	}
	return res, nil
}

// MulScalar multiplies vector by a scalar.
func (v Vector) MulScalar(s *ExactNum) Vector {
	res := make(Vector, len(v))
	for i := range v {
		p, _ := v[i].Mul(s)
		if p != nil {
			res[i] = *p
		}
	}
	return res
}

// Neg negates vector -v.
func (v Vector) Neg() Vector {
	return v.MulScalar(NewExactInt(-1))
}
