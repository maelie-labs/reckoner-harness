package main

import (
	"fmt"
	"math"
	"math/big"
	"strconv"
	"strings"
	"sync"
	"time"
	"unicode/utf8"
)

// CalcValKind represents the runtime type of a value in the tool subsystem.
type CalcValKind int

const (
	ValKindRat CalcValKind = iota
	ValKindSurd
	ValKindBool
	ValKindUnit
	ValKindString
	ValKindList
	ValKindFloat
	ValKindDate
	ValKindRawString
	ValKindExactNum
	ValKindPoly
	ValKindInterval
	ValKindMatrix
	ValKindVector
)

// SurdVal represents an exact algebraic radical: sign * (a * sqrt(b)) / c.
// b > 1 is square-free, a, c >= 1, gcd(a, c) = 1.
type SurdVal struct {
	sign int      // +1 or -1
	a    *big.Int // multiplier >= 1
	b    *big.Int // square-free part > 1
	c    *big.Int // denominator >= 1
}

func (s *SurdVal) String() string {
	signStr := ""
	if s.sign < 0 {
		signStr = "-"
	}

	radStr := fmt.Sprintf("sqrt(%s)", s.b.String())
	if s.a.Cmp(big.NewInt(1)) != 0 {
		radStr = fmt.Sprintf("%s*sqrt(%s)", s.a.String(), s.b.String())
	}

	if s.c.Cmp(big.NewInt(1)) == 0 {
		return fmt.Sprintf("%s%s", signStr, radStr)
	}
	return fmt.Sprintf("%s%s/%s", signStr, radStr, s.c.String())
}

func (s *SurdVal) ToFloat64() float64 {
	aF, _ := s.a.Float64()
	bF, _ := s.b.Float64()
	cF, _ := s.c.Float64()
	val := aF * math.Sqrt(bF) / cF
	if s.sign < 0 {
		return -val
	}
	return val
}

func (s *SurdVal) Clone() *SurdVal {
	if s == nil {
		return nil
	}
	return &SurdVal{
		sign: s.sign,
		a:    new(big.Int).Set(s.a),
		b:    new(big.Int).Set(s.b),
		c:    new(big.Int).Set(s.c),
	}
}

// CalcVal represents any typed value stored in registers or produced by tools.
type CalcVal struct {
	Kind        CalcValKind
	Rat         *big.Rat
	Surd        *SurdVal
	Bool        bool
	UnitVal     *big.Rat
	UnitName    string
	Str         string
	DateStr     string
	List        []CalcVal
	Float       float64
	ForcedFrac  bool
	ForcedDec   bool
	HasRound    bool
	RoundPlaces int
	Exact       *ExactNum
	Poly        *MultiPoly
	Intervals   IntervalSet
	Mat         Matrix
	Vec         Vector
}

// isSymbolic returns true if the value represents a symbolic expression containing variables.
func isSymbolic(v CalcVal) bool {
	switch v.Kind {
	case ValKindPoly:
		if v.Poly != nil {
			_, isConst := v.Poly.AsConstant()
			return !isConst
		}
		return false
	case ValKindList:
		for _, item := range v.List {
			if isSymbolic(item) {
				return true
			}
		}
		return false
	default:
		return false
	}
}

// escapeControlTokens escapes "<|" to "<\|"  to prevent forging control tokens.
func escapeControlTokens(s string) string {
	return strings.ReplaceAll(s, "<|", "<\\|")
}

// MaxSafeFloatInt is 2^53 = 9,007,199,254,740,992, the maximum integer exactly representable in float64.
const MaxSafeFloatInt = 9007199254740992

// FormatFloatStandard formats a float64 according to the float specification:
// - Integral floats with |x| <= 2^53 print as exact integers without scientific notation;
// - All other floats with |x| >= 1e15 or |x| < 1e-4 use "%.6g";
// - Floats in between print with at most 6 decimal places (trailing zeros stripped);
// - -0.0 -> "0";
// - NaN and Inf are rejected: ErrDomain for NaN, ErrOverflow for Inf (NaN/Inf never bind).
func FormatFloatStandard(f float64) (string, error) {
	if math.IsNaN(f) {
		return "", fmt.Errorf("ErrDomain: NaN value cannot be bound")
	}
	if math.IsInf(f, 0) {
		return "", fmt.Errorf("ErrOverflow: infinite value cannot be bound")
	}
	if f == 0.0 || math.Abs(f) == 0.0 {
		return "0", nil
	}
	absF := math.Abs(f)
	if math.Floor(f) == f && absF <= MaxSafeFloatInt {
		return fmt.Sprintf("%.0f", f), nil
	}
	if absF >= 1e15 || absF < 1e-4 {
		return fmt.Sprintf("%.6g", f), nil
	}
	s := fmt.Sprintf("%.6f", f)
	res := strings.TrimRight(strings.TrimRight(s, "0"), ".")
	if res == "-0" || res == "" {
		res = "0"
	}
	return res, nil
}

// formatFloat formats floats according to float specification:
// - with rounding if hasRound is set;
// - otherwise delegates to FormatFloatStandard.
func formatFloat(f float64, hasRound bool, roundPlaces int) string {
	if hasRound {
		factor := math.Pow(10, float64(roundPlaces))
		rounded := math.Round(f*factor) / factor
		if roundPlaces <= 0 {
			return fmt.Sprintf("%.0f", rounded)
		}
		return fmt.Sprintf("%.*f", roundPlaces, rounded)
	}

	str, err := FormatFloatStandard(f)
	if err != nil {
		if math.IsNaN(f) {
			return "NaN"
		}
		if math.IsInf(f, 1) {
			return "Inf"
		}
		if math.IsInf(f, -1) {
			return "-Inf"
		}
		return "0"
	}
	return str
}

// roundRational rounds an exact big.Rat to places decimal places using school rounding (away from zero).
func roundRational(r *big.Rat, places int) *big.Rat {
	if r == nil {
		return new(big.Rat)
	}
	if r.Sign() == 0 {
		return new(big.Rat)
	}

	if places >= 0 {
		// scale = 10^places
		scale := new(big.Int).Exp(big.NewInt(10), big.NewInt(int64(places)), nil)
		// y = r * scale = (num * scale) / denom
		num := new(big.Int).Mul(r.Num(), scale)
		denom := new(big.Int).Set(r.Denom())

		// We need integer division of num by denom with halves rounded away from zero:
		// q = floor(num / denom), rem = num % denom
		// 2 * rem >= denom means >= 0.5
		q := new(big.Int)
		rem := new(big.Int)
		q.QuoRem(num, denom, rem)

		// Check remainder for rounding
		twoRem := new(big.Int).Mul(rem, big.NewInt(2))
		twoRem.Abs(twoRem)
		denomAbs := new(big.Int).Abs(denom)

		if twoRem.Cmp(denomAbs) >= 0 {
			if r.Sign() > 0 {
				q.Add(q, big.NewInt(1))
			} else {
				q.Sub(q, big.NewInt(1))
			}
		}

		res := new(big.Rat).SetFrac(q, scale)
		return res
	}

	// places < 0: round to tens, hundreds, etc.
	k := -places
	scale := new(big.Int).Exp(big.NewInt(10), big.NewInt(int64(k)), nil)
	// y = r / scale = num / (denom * scale)
	denom := new(big.Int).Mul(r.Denom(), scale)
	num := new(big.Int).Set(r.Num())

	q := new(big.Int)
	rem := new(big.Int)
	q.QuoRem(num, denom, rem)

	twoRem := new(big.Int).Mul(rem, big.NewInt(2))
	twoRem.Abs(twoRem)
	denomAbs := new(big.Int).Abs(denom)

	if twoRem.Cmp(denomAbs) >= 0 {
		if r.Sign() > 0 {
			q.Add(q, big.NewInt(1))
		} else {
			q.Sub(q, big.NewInt(1))
		}
	}

	q.Mul(q, scale)
	return new(big.Rat).SetInt(q)
}

// roundRationalString returns the string representation of r rounded to places decimal places.
func roundRationalString(r *big.Rat, places int) string {
	rounded := roundRational(r, places)
	if places <= 0 {
		return rounded.Num().String()
	}

	// Exact decimal representation with forced `places` digits
	// rounded * 10^places is an integer
	scale := new(big.Int).Exp(big.NewInt(10), big.NewInt(int64(places)), nil)
	intVal := new(big.Int).Mul(rounded.Num(), scale)
	intVal.Quo(intVal, rounded.Denom())

	sign := ""
	if intVal.Sign() < 0 {
		sign = "-"
		intVal.Abs(intVal)
	}

	str := intVal.String()
	if len(str) <= places {
		// Pad with leading zeros: e.g. places=3, str="5" -> "0.005"
		str = strings.Repeat("0", places-len(str)+1) + str
	}
	dotPos := len(str) - places
	integerPart := str[:dotPos]
	fracPart := str[dotPos:]
	return fmt.Sprintf("%s%s.%s", sign, integerPart, fracPart)
}

// formatRational implements the display rule for exact rationals.
func formatRational(r *big.Rat, forcedFrac, forcedDec, hasRound bool, roundPlaces int) string {
	if r == nil {
		return "0"
	}

	if hasRound {
		return roundRationalString(r, roundPlaces)
	}

	if forcedFrac {
		if r.IsInt() {
			return r.Num().String()
		}
		return r.RatString()
	}

	if forcedDec {
		if r.Sign() == 0 {
			return "0"
		}
		if r.IsInt() {
			return r.Num().String()
		}
		bf := new(big.Float).SetPrec(128).SetRat(r)
		s := bf.Text('g', 6)
		if s == "-0" || s == "" {
			s = "0"
		}
		return s
	}

	if r.IsInt() {
		return r.Num().String()
	}

	// Terminating decimal check:
	// Clean terminating decimal if denominator factors as 2^a * 5^b with max(a, b) <= 6.
	q := new(big.Int).Set(r.Denom())
	a := 0
	two := big.NewInt(2)
	for new(big.Int).Mod(q, two).Int64() == 0 {
		q.Div(q, two)
		a++
	}
	b := 0
	five := big.NewInt(5)
	for new(big.Int).Mod(q, five).Int64() == 0 {
		q.Div(q, five)
		b++
	}

	if q.Cmp(big.NewInt(1)) == 0 && a <= 6 && b <= 6 {
		k := a
		if b > k {
			k = b
		}
		// Exact decimal conversion without float64:
		// Multiply numerator by 10^k / denom (an exact integer), format with k decimal places
		scale := new(big.Int).Exp(big.NewInt(10), big.NewInt(int64(k)), nil)
		factor := new(big.Int).Quo(scale, r.Denom())
		intVal := new(big.Int).Mul(r.Num(), factor)

		sign := ""
		if intVal.Sign() < 0 {
			sign = "-"
			intVal.Abs(intVal)
		}
		str := intVal.String()
		if len(str) <= k {
			str = strings.Repeat("0", k-len(str)+1) + str
		}
		dotPos := len(str) - k
		intPart := str[:dotPos]
		fracPart := strings.TrimRight(str[dotPos:], "0")
		if fracPart == "" {
			return sign + intPart
		}
		return fmt.Sprintf("%s%s.%s", sign, intPart, fracPart)
	}

	// Irreducible fraction p/q
	return r.RatString()
}

func (v CalcVal) String() string {
	switch v.Kind {
	case ValKindRat:
		if v.Str != "" {
			return v.Str
		}
		return formatRational(v.Rat, v.ForcedFrac, v.ForcedDec, v.HasRound, v.RoundPlaces)
	case ValKindSurd:
		if v.Surd == nil {
			return "0"
		}
		return v.Surd.String()
	case ValKindBool:
		if v.Bool {
			return "true"
		}
		return "false"
	case ValKindUnit:
		valStr := v.Str
		if valStr == "" {
			valStr = formatRational(v.UnitVal, v.ForcedFrac, v.ForcedDec, v.HasRound, v.RoundPlaces)
		}
		if v.UnitName == "" {
			return valStr
		}
		return fmt.Sprintf("%s %s", valStr, v.UnitName)
	case ValKindString:
		return fmt.Sprintf("%q", escapeControlTokens(v.Str))
	case ValKindRawString:
		return escapeControlTokens(v.Str)
	case ValKindDate:
		return escapeControlTokens(v.DateStr)
	case ValKindFloat:
		if v.Str != "" {
			return v.Str
		}
		res := formatFloat(v.Float, v.HasRound, v.RoundPlaces)
		if v.UnitName != "" {
			return fmt.Sprintf("%s %s", res, v.UnitName)
		}
		return res
	case ValKindList:
		var items []string
		for _, item := range v.List {
			items = append(items, item.String())
		}
		return fmt.Sprintf("[%s]", strings.Join(items, ", "))
	case ValKindExactNum:
		if v.Exact == nil {
			return "0"
		}
		return v.Exact.String()
	case ValKindPoly:
		if v.Poly == nil {
			return "0"
		}
		return v.Poly.String()
	case ValKindInterval:
		return v.Intervals.String()
	case ValKindMatrix:
		return v.Mat.String()
	case ValKindVector:
		return v.Vec.String()
	}
	return ""
}

// NumRegisters is the size of the scratchpad ring buffer .
const NumRegisters = 16

// RegisterStore manages 16 scratchpad registers (r0..r15) holding typed values.
type RegisterStore struct {
	mu         sync.RWMutex
	regs       [NumRegisters]*CalcVal
	nextRegIdx int
	clock      *time.Time
}

var (
	globalDefaultClock   *time.Time
	globalDefaultClockMu sync.RWMutex
)

// SetGlobalDefaultClock sets the process-wide default clock (e.g. from CLI flag or env var).
func SetGlobalDefaultClock(t *time.Time) {
	globalDefaultClockMu.Lock()
	defer globalDefaultClockMu.Unlock()
	if t == nil {
		globalDefaultClock = nil
	} else {
		cp := *t
		globalDefaultClock = &cp
	}
}

// GetGlobalDefaultClock returns the process-wide default clock, or nil if none is set.
func GetGlobalDefaultClock() *time.Time {
	globalDefaultClockMu.RLock()
	defer globalDefaultClockMu.RUnlock()
	if globalDefaultClock == nil {
		return nil
	}
	cp := *globalDefaultClock
	return &cp
}

// ParseClockTimestamp parses standard ISO/RFC3339 timestamps for clock injection.
func ParseClockTimestamp(s string) (time.Time, error) {
	s = strings.TrimSpace(s)
	formats := []string{
		time.RFC3339,
		time.RFC3339Nano,
		"2006-01-02T15:04:05Z07:00",
		"2006-01-02T15:04:05",
		"2006-01-02 15:04:05",
		"2006-01-02 15:04:05 MST",
		"2006-01-02",
	}
	for _, f := range formats {
		if t, err := time.Parse(f, s); err == nil {
			return t, nil
		}
	}
	return time.Time{}, fmt.Errorf("ErrSyntax: invalid clock timestamp %q", s)
}

// SetClock sets a request/session-scoped clock override on the register store.
func (rs *RegisterStore) SetClock(t *time.Time) {
	rs.mu.Lock()
	defer rs.mu.Unlock()
	if t == nil {
		rs.clock = nil
	} else {
		cp := *t
		rs.clock = &cp
	}
}

// GetClock returns the effective clock for the register store, falling back to global default or wall time.
func (rs *RegisterStore) GetClock() time.Time {
	rs.mu.RLock()
	defer rs.mu.RUnlock()
	if rs.clock != nil {
		return *rs.clock
	}
	if gClock := GetGlobalDefaultClock(); gClock != nil {
		return *gClock
	}
	return time.Now()
}

func NewRegisterStore() *RegisterStore {
	return &RegisterStore{
		nextRegIdx: 0,
	}
}

func (rs *RegisterStore) Reset() {
	rs.mu.Lock()
	defer rs.mu.Unlock()
	for i := 0; i < NumRegisters; i++ {
		rs.regs[i] = nil
	}
	rs.nextRegIdx = 0
}

// Get returns the typed value stored in register idx (0..15).
// Returns ErrUnsetRegister if register has not been initialized.
func (rs *RegisterStore) Get(idx int) (*CalcVal, error) {
	if idx < 0 || idx >= NumRegisters {
		return nil, fmt.Errorf("ErrSyntax: invalid register index %d", idx)
	}
	rs.mu.RLock()
	defer rs.mu.RUnlock()
	val := rs.regs[idx]
	if val == nil {
		return nil, fmt.Errorf("ErrUnsetRegister: <|r%d|> is empty", idx)
	}
	// Return a copy so mutations don't alter the store
	cp := *val
	return &cp, nil
}

// Set explicitly sets register idx (0..15) to val.
func (rs *RegisterStore) Set(idx int, val CalcVal) error {
	if idx < 0 || idx >= NumRegisters {
		return fmt.Errorf("ErrSyntax: invalid register index %d", idx)
	}
	if val.Kind == ValKindFloat {
		if math.IsNaN(val.Float) {
			return fmt.Errorf("ErrDomain: NaN value cannot be bound")
		}
		if math.IsInf(val.Float, 0) {
			return fmt.Errorf("ErrOverflow: infinite value cannot be bound")
		}
	}
	rs.mu.Lock()
	defer rs.mu.Unlock()
	cp := val
	rs.regs[idx] = &cp
	return nil
}

// AutoBind binds val to the next register in the ring buffer (modulo 16).
// Returns the register tag, formatted display observation, and any display error.
// If overwriting an existing register value, announces replacement:
// e.g. "<|r0|> = 5 (replaced 60)".
// Per display length limit: exact results whose display exceeds 256 characters return
// a success observation naming the register: "<|rN|> = [too large to display: N digits]",
// while still holding the exact value in the register.
func (rs *RegisterStore) AutoBind(val CalcVal) (string, string, error) {
	rs.mu.Lock()
	defer rs.mu.Unlock()
	idx := rs.nextRegIdx % NumRegisters
	rs.nextRegIdx++
	tag := fmt.Sprintf("<|r%d|>", idx)
	valStr := val.String()

	var oldStr string
	hasOld := false
	if rs.regs[idx] != nil {
		hasOld = true
		oldStr = rs.regs[idx].String()
		if len(oldStr) > 256 {
			oldStr = fmt.Sprintf("... (%d digits)", len(oldStr))
		}
	}

	cp := val
	rs.regs[idx] = &cp

	if len(valStr) > 256 {
		disp := fmt.Sprintf("%s = [too large to display: %d digits]", tag, len(valStr))
		return tag, disp, nil
	}

	var disp string
	if hasOld {
		disp = fmt.Sprintf("%s = %s (replaced %s)", tag, valStr, oldStr)
	} else {
		disp = fmt.Sprintf("%s = %s", tag, valStr)
	}
	return tag, disp, nil
}

// AutoBindNamed binds val to the next register in the ring buffer, formatted for solve:
// e.g. "x = 2.5 (<|r0|>)". If replacing, includes "(<|r0|>, replaced 60)".
// Per display length limit: exact results whose display exceeds 256 characters return
// a success observation naming the variable and register, while still holding the exact value in the register.
func (rs *RegisterStore) AutoBindNamed(varName string, val CalcVal) (string, string, error) {
	rs.mu.Lock()
	defer rs.mu.Unlock()
	idx := rs.nextRegIdx % NumRegisters
	rs.nextRegIdx++
	tag := fmt.Sprintf("<|r%d|>", idx)
	valStr := val.String()

	var oldStr string
	hasOld := false
	if rs.regs[idx] != nil {
		hasOld = true
		oldStr = rs.regs[idx].String()
		if len(oldStr) > 256 {
			oldStr = fmt.Sprintf("... (%d digits)", len(oldStr))
		}
	}

	cp := val
	rs.regs[idx] = &cp

	if len(valStr) > 256 {
		disp := fmt.Sprintf("%s = [too large to display: %d digits] (%s)", varName, len(valStr), tag)
		return tag, disp, nil
	}

	var disp string
	if hasOld {
		disp = fmt.Sprintf("%s = %s (%s, replaced %s)", varName, valStr, tag, oldStr)
	} else {
		disp = fmt.Sprintf("%s = %s (%s)", varName, valStr, tag)
	}
	return tag, disp, nil
}

// ExecuteRegisterCmd executes a reg command:
// Reading: "r0" or "<|r0|>" -> value
// Writing: "r0 = expr" or "<|r0|> = expr" -> assignment
// Accepts only registers, uses error codes, and measures lengths in runes.
func (rs *RegisterStore) ExecuteRegisterCmd(cmd string) (string, error) {
	cmd = strings.TrimSpace(cmd)
	if cmd == "clear" {
		rs.Reset()
		return "OK", nil
	}
	if cmd == "list" {
		var parts []string
		for i := 0; i < NumRegisters; i++ {
			if rs.regs[i] != nil {
				valStr := rs.regs[i].String()
				if len(valStr) > 256 {
					valStr = fmt.Sprintf("[too large to display: %d digits]", len(valStr))
				}
				parts = append(parts, fmt.Sprintf("<|r%d|> = %s", i, valStr))
			}
		}
		if len(parts) == 0 {
			return "Registers empty", nil
		}
		return strings.Join(parts, ", "), nil
	}

	fields := strings.Fields(cmd)
	if len(fields) == 2 && fields[0] == "get" {
		idx, err := parseRegName(fields[1])
		if err != nil {
			return "", err
		}
		val, err := rs.Get(idx)
		if err != nil {
			return "", err
		}
		valStr := val.String()
		if len(valStr) > 256 {
			return fmt.Sprintf("[too large to display: %d digits]", len(valStr)), nil
		}
		return valStr, nil
	}
	if len(fields) >= 3 && fields[0] == "set" {
		idx, err := parseRegName(fields[1])
		if err != nil {
			return "", err
		}
		expr := strings.TrimSpace(cmd[strings.Index(cmd, fields[1])+len(fields[1]):])
		val, err := EvaluateInfixToVal(expr, rs)
		if err != nil {
			return "", err
		}
		_ = rs.Set(idx, val)
		valStr := val.String()
		if len(valStr) > 256 {
			return fmt.Sprintf("r%d = [too large to display: %d digits]", idx, len(valStr)), nil
		}
		return fmt.Sprintf("r%d = %s", idx, valStr), nil
	}
	if len(fields) == 2 && fields[0] == "len" {
		idx, err := parseRegName(fields[1])
		if err != nil {
			return "", err
		}
		val, err := rs.Get(idx)
		if err != nil {
			return "", err
		}
		s := val.String()
		if val.Kind == ValKindString {
			s = val.Str
		}
		runeCount := utf8.RuneCountInString(s)
		return strconv.Itoa(runeCount), nil
	}

	parts := strings.Split(cmd, "=")

	if len(parts) == 1 {
		// Read command
		rawReg := strings.TrimSpace(parts[0])
		idx, err := parseRegName(rawReg)
		if err != nil {
			return "", err
		}
		val, err := rs.Get(idx)
		if err != nil {
			return "", err
		}
		return val.String(), nil
	}

	if len(parts) == 2 {
		// Assignment command: r0 = expr or <|r0|> = expr
		rawReg := strings.TrimSpace(parts[0])
		idx, err := parseRegName(rawReg)
		if err != nil {
			return "", err
		}
		expr := strings.TrimSpace(parts[1])

		// String utility functions check before arithmetic:
		// len("...") measures in runes 
		if strings.HasPrefix(expr, "len(") && strings.HasSuffix(expr, ")") {
			inner := strings.TrimSuffix(strings.TrimPrefix(expr, "len("), ")")
			inner = strings.TrimSpace(inner)
			if (strings.HasPrefix(inner, `"`) && strings.HasSuffix(inner, `"`)) ||
				(strings.HasPrefix(inner, `'`) && strings.HasSuffix(inner, `'`)) {
				inner = inner[1 : len(inner)-1]
			}
			runeCount := utf8.RuneCountInString(inner)
			val := CalcVal{
				Kind: ValKindRat,
				Rat:  big.NewRat(int64(runeCount), 1),
			}
			_ = rs.Set(idx, val)
			return fmt.Sprintf("<|r%d|> = %d", idx, runeCount), nil
		}

		if strings.HasPrefix(expr, "words(") && strings.HasSuffix(expr, ")") {
			inner := strings.TrimSuffix(strings.TrimPrefix(expr, "words("), ")")
			inner = strings.TrimSpace(inner)
			if (strings.HasPrefix(inner, `"`) && strings.HasSuffix(inner, `"`)) ||
				(strings.HasPrefix(inner, `'`) && strings.HasSuffix(inner, `'`)) {
				inner = inner[1 : len(inner)-1]
			}
			wordCount := len(strings.Fields(inner))
			val := CalcVal{
				Kind: ValKindRat,
				Rat:  big.NewRat(int64(wordCount), 1),
			}
			_ = rs.Set(idx, val)
			return fmt.Sprintf("<|r%d|> = %d", idx, wordCount), nil
		}

		// Evaluate arithmetic expression with the store
		val, err := EvaluateInfixToVal(expr, rs)
		if err != nil {
			return "", err
		}
		_ = rs.Set(idx, val)
		return fmt.Sprintf("<|r%d|> = %s", idx, val.String()), nil
	}

	return "", fmt.Errorf("ErrSyntax: invalid register command syntax: %s", cmd)
}

func parseRegName(s string) (int, error) {
	s = strings.TrimSpace(s)
	s = strings.TrimPrefix(s, "<|")
	s = strings.TrimSuffix(s, "|>")
	s = strings.ToLower(s)
	if strings.HasPrefix(s, "r") {
		n, err := strconv.Atoi(s[1:])
		if err == nil {
			if n < 0 || n >= NumRegisters {
				return -1, fmt.Errorf("ErrRegisterBounds: register out of range (r0..r%d): %s", NumRegisters-1, s)
			}
			return n, nil
		}
	}
	return -1, fmt.Errorf("ErrSyntax: invalid register name: %s", s)
}

// GetAllRegisters returns a map of current string representations for debugging / lua interop.
func (rs *RegisterStore) GetAllRegisters() map[string]string {
	rs.mu.RLock()
	defer rs.mu.RUnlock()
	res := make(map[string]string)
	for i := 0; i < NumRegisters; i++ {
		if rs.regs[i] != nil {
			tag := fmt.Sprintf("<|r%d|>", i)
			name := fmt.Sprintf("r%d", i)
			s := rs.regs[i].String()
			res[tag] = s
			res[name] = s
		}
	}
	return res
}
