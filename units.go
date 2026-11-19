package main

import (
	"fmt"
	"math/big"
	"strconv"
	"strings"
)

type UnitCategory int

const (
	CatUnknown UnitCategory = iota
	CatLength
	CatMass
	CatTime
	CatSpeed
	CatTemperature
	CatEnergy
	CatPower
	CatPressure
	CatForce
	CatFrequency
	CatElectricCharge
	CatVoltage
	CatCurrent
	CatMagneticField
	CatAmount
	CatVolume
	CatAcceleration
)

// Dimensions represents exponents of the 7 SI base units:
// 0: Length (m)
// 1: Mass (kg)
// 2: Time (s)
// 3: Electric current (A)
// 4: Thermodynamic temperature (K)
// 5: Amount of substance (mol)
// 6: Luminous intensity (cd)
type Dimensions [7]int8

var (
	DimDimensionless = Dimensions{}
	DimLength        = Dimensions{1, 0, 0, 0, 0, 0, 0}
	DimMass          = Dimensions{0, 1, 0, 0, 0, 0, 0}
	DimTime          = Dimensions{0, 0, 1, 0, 0, 0, 0}
	DimCurrent       = Dimensions{0, 0, 0, 1, 0, 0, 0}
	DimTemperature   = Dimensions{0, 0, 0, 0, 1, 0, 0}
	DimAmount        = Dimensions{0, 0, 0, 0, 0, 1, 0}
	DimLuminous      = Dimensions{0, 0, 0, 0, 0, 0, 1}

	DimSpeed        = Dimensions{1, 0, -1, 0, 0, 0, 0}
	DimAcceleration = Dimensions{1, 0, -2, 0, 0, 0, 0}
	DimForce        = Dimensions{1, 1, -2, 0, 0, 0, 0}
	DimEnergy      = Dimensions{2, 1, -2, 0, 0, 0, 0}
	DimPower       = Dimensions{2, 1, -3, 0, 0, 0, 0}
	DimPressure    = Dimensions{-1, 1, -2, 0, 0, 0, 0}
	DimFrequency   = Dimensions{0, 0, -1, 0, 0, 0, 0}
	DimCharge      = Dimensions{0, 0, 1, 1, 0, 0, 0}
	DimVoltage     = Dimensions{2, 1, -3, -1, 0, 0, 0}
	DimResistance  = Dimensions{2, 1, -3, -2, 0, 0, 0}
	DimCapacitance = Dimensions{-2, -1, 4, 2, 0, 0, 0}
	DimConductance = Dimensions{-2, -1, 3, 2, 0, 0, 0}
	DimMagField    = Dimensions{0, 1, -2, -1, 0, 0, 0}
	DimMagFlux     = Dimensions{2, 1, -2, -1, 0, 0, 0}
	DimInductance  = Dimensions{2, 1, -2, -2, 0, 0, 0}
	DimVolume      = Dimensions{3, 0, 0, 0, 0, 0, 0}
	DimArea        = Dimensions{2, 0, 0, 0, 0, 0, 0}
)

func (d Dimensions) IsDimensionless() bool {
	return d == DimDimensionless
}

func (d Dimensions) Add(o Dimensions) Dimensions {
	var res Dimensions
	for i := 0; i < 7; i++ {
		res[i] = d[i] + o[i]
	}
	return res
}

func (d Dimensions) Sub(o Dimensions) Dimensions {
	var res Dimensions
	for i := 0; i < 7; i++ {
		res[i] = d[i] - o[i]
	}
	return res
}

func (d Dimensions) MulScalar(k int8) Dimensions {
	var res Dimensions
	for i := 0; i < 7; i++ {
		res[i] = d[i] * k
	}
	return res
}

type UnitInfo struct {
	Name     string
	Category UnitCategory
	Scale    *big.Rat // Scale factor to base unit
}

var unitSpecificDimensions = map[string]Dimensions{
	"ohm":     DimResistance,
	"Ω":       DimResistance,
	"F":       DimCapacitance,
	"farad":   DimCapacitance,
	"S":       DimConductance,
	"siemens": DimConductance,
	"Wb":      DimMagFlux,
	"weber":   DimMagFlux,
	"H":       DimInductance,
	"henry":   DimInductance,
	"cd":      DimLuminous,
	"candela": DimLuminous,
	"sr":      DimDimensionless,
	"rad":     DimDimensionless,
}

func (u UnitInfo) GetDimensions() Dimensions {
	if d, ok := unitSpecificDimensions[u.Name]; ok {
		return d
	}
	switch u.Category {
	case CatLength:
		return DimLength
	case CatMass:
		return DimMass
	case CatTime:
		return DimTime
	case CatCurrent:
		return DimCurrent
	case CatTemperature:
		return DimTemperature
	case CatAmount:
		return DimAmount
	case CatSpeed:
		return DimSpeed
	case CatForce:
		return DimForce
	case CatEnergy:
		return DimEnergy
	case CatPower:
		return DimPower
	case CatPressure:
		return DimPressure
	case CatFrequency:
		return DimFrequency
	case CatElectricCharge:
		return DimCharge
	case CatVoltage:
		return DimVoltage
	case CatMagneticField:
		return DimMagField
	case CatVolume:
		return DimVolume
	case CatAcceleration:
		return DimAcceleration
	default:
		return DimDimensionless
	}
}

// exp10Rat returns 10^n as *big.Rat
func exp10Rat(n int64) *big.Rat {
	if n >= 0 {
		bi := new(big.Int).Exp(big.NewInt(10), big.NewInt(n), nil)
		return new(big.Rat).SetInt(bi)
	}
	bi := new(big.Int).Exp(big.NewInt(10), big.NewInt(-n), nil)
	return new(big.Rat).SetFrac(big.NewInt(1), bi)
}

func fracRat(num, den int64) *big.Rat {
	return big.NewRat(num, den)
}

func scaledRat(num int64, exp int64) *big.Rat {
	// num * 10^exp
	r := new(big.Rat).SetInt64(num)
	return r.Mul(r, exp10Rat(exp))
}

var unitRegistry = map[string]UnitInfo{
	// Length (base: m)
	"m":        {"m", CatLength, big.NewRat(1, 1)},
	"meter":    {"m", CatLength, big.NewRat(1, 1)},
	"meters":   {"m", CatLength, big.NewRat(1, 1)},
	"km":       {"km", CatLength, big.NewRat(1000, 1)},
	"cm":       {"cm", CatLength, big.NewRat(1, 100)},
	"mm":       {"mm", CatLength, big.NewRat(1, 1000)},
	"um":       {"um", CatLength, big.NewRat(1, 1000000)},
	"µm":       {"um", CatLength, big.NewRat(1, 1000000)},
	"nm":       {"nm", CatLength, big.NewRat(1, 1000000000)},
	"pm":       {"pm", CatLength, scaledRat(1, -12)},
	"fm":       {"fm", CatLength, scaledRat(1, -15)},
	"angstrom": {"angstrom", CatLength, scaledRat(1, -10)},
	"Å":        {"angstrom", CatLength, scaledRat(1, -10)},
	"au":       {"au", CatLength, big.NewRat(149597870700, 1)},
	"AU":       {"au", CatLength, big.NewRat(149597870700, 1)},
	"ly":       {"ly", CatLength, big.NewRat(9460730472580800, 1)},
	"in":       {"in", CatLength, big.NewRat(127, 5000)}, // 0.0254 m
	"inch":     {"in", CatLength, big.NewRat(127, 5000)},
	"inches":   {"in", CatLength, big.NewRat(127, 5000)},
	"ft":       {"ft", CatLength, big.NewRat(381, 1250)}, // 0.3048 m
	"feet":     {"ft", CatLength, big.NewRat(381, 1250)},
	"foot":     {"ft", CatLength, big.NewRat(381, 1250)},
	"mi":       {"mi", CatLength, big.NewRat(201168, 125)}, // 1609.344 m
	"mile":     {"mi", CatLength, big.NewRat(201168, 125)},
	"miles":    {"mi", CatLength, big.NewRat(201168, 125)},

	// Mass (base: kg)
	"kg":        {"kg", CatMass, big.NewRat(1, 1)},
	"kilogram":  {"kg", CatMass, big.NewRat(1, 1)},
	"kilograms": {"kg", CatMass, big.NewRat(1, 1)},
	"g":         {"g", CatMass, big.NewRat(1, 1000)},
	"gram":      {"g", CatMass, big.NewRat(1, 1000)},
	"grams":     {"g", CatMass, big.NewRat(1, 1000)},
	"mg":        {"mg", CatMass, scaledRat(1, -6)},
	"ug":        {"ug", CatMass, scaledRat(1, -9)},
	"µg":        {"ug", CatMass, scaledRat(1, -9)},
	"tonne":     {"tonne", CatMass, big.NewRat(1000, 1)},
	"t":         {"tonne", CatMass, big.NewRat(1000, 1)},
	"u":         {"u", CatMass, new(big.Rat).SetFrac(big.NewInt(166053906892), new(big.Int).Exp(big.NewInt(10), big.NewInt(38), nil))}, // 1.66053906892e-27 kg
	"Da":        {"u", CatMass, new(big.Rat).SetFrac(big.NewInt(166053906892), new(big.Int).Exp(big.NewInt(10), big.NewInt(38), nil))},
	"amu":       {"u", CatMass, new(big.Rat).SetFrac(big.NewInt(166053906892), new(big.Int).Exp(big.NewInt(10), big.NewInt(38), nil))},
	"lb":        {"lb", CatMass, big.NewRat(45359237, 100000000)}, // 0.45359237 kg
	"lbs":       {"lb", CatMass, big.NewRat(45359237, 100000000)},
	"pound":     {"lb", CatMass, big.NewRat(45359237, 100000000)},
	"pounds":    {"lb", CatMass, big.NewRat(45359237, 100000000)},
	"oz":        {"oz", CatMass, big.NewRat(45359237, 1600000000)}, // 1/16 lb
	"ounce":     {"oz", CatMass, big.NewRat(45359237, 1600000000)},
	"ounces":    {"oz", CatMass, big.NewRat(45359237, 1600000000)},

	// Time (base: s)
	"s":       {"s", CatTime, big.NewRat(1, 1)},
	"sec":     {"s", CatTime, big.NewRat(1, 1)},
	"second":  {"s", CatTime, big.NewRat(1, 1)},
	"seconds": {"s", CatTime, big.NewRat(1, 1)},
	"ms":      {"ms", CatTime, big.NewRat(1, 1000)},
	"us":      {"us", CatTime, scaledRat(1, -6)},
	"µs":      {"us", CatTime, scaledRat(1, -6)},
	"ns":      {"ns", CatTime, scaledRat(1, -9)},
	"ps":      {"ps", CatTime, scaledRat(1, -12)},
	"fs":      {"fs", CatTime, scaledRat(1, -15)},
	"min":     {"min", CatTime, big.NewRat(60, 1)},
	"minute":  {"min", CatTime, big.NewRat(60, 1)},
	"minutes": {"min", CatTime, big.NewRat(60, 1)},
	"h":       {"hr", CatTime, big.NewRat(3600, 1)},
	"hr":      {"hr", CatTime, big.NewRat(3600, 1)},
	"hour":    {"hr", CatTime, big.NewRat(3600, 1)},
	"hours":   {"hr", CatTime, big.NewRat(3600, 1)},
	"day":     {"day", CatTime, big.NewRat(86400, 1)},
	"days":    {"day", CatTime, big.NewRat(86400, 1)},
	"yr":      {"yr", CatTime, big.NewRat(31557600, 1)}, // Julian year: 365.25 * 86400
	"year":    {"yr", CatTime, big.NewRat(31557600, 1)},
	"years":   {"yr", CatTime, big.NewRat(31557600, 1)},

	// Speed (base: m/s)
	"m/s":   {"m/s", CatSpeed, big.NewRat(1, 1)},
	"mph":   {"mph", CatSpeed, big.NewRat(1397, 3125)}, // 0.44704 m/s (45 mph = 20.1168 m/s)
	"km/h":  {"km/h", CatSpeed, big.NewRat(5, 18)},     // 1000/3600 m/s
	"kmh":   {"km/h", CatSpeed, big.NewRat(5, 18)},
	"km/s":  {"km/s", CatSpeed, big.NewRat(1000, 1)},
	"m/s^2": {"m/s^2", CatAcceleration, big.NewRat(1, 1)},
	"m/s2":  {"m/s^2", CatAcceleration, big.NewRat(1, 1)},

	// Temperature
	"degC":       {"degC", CatTemperature, big.NewRat(1, 1)},
	"celsius":    {"degC", CatTemperature, big.NewRat(1, 1)},
	"F":          {"F", CatTemperature, big.NewRat(5, 9)},
	"f":          {"F", CatTemperature, big.NewRat(5, 9)},
	"degF":       {"F", CatTemperature, big.NewRat(5, 9)},
	"fahrenheit": {"F", CatTemperature, big.NewRat(5, 9)},
	"K":          {"K", CatTemperature, big.NewRat(1, 1)},
	"k":          {"K", CatTemperature, big.NewRat(1, 1)},
	"kelvin":     {"K", CatTemperature, big.NewRat(1, 1)},

	// Energy (base: J)
	"J":     {"J", CatEnergy, big.NewRat(1, 1)},
	"joule": {"J", CatEnergy, big.NewRat(1, 1)},
	"kJ":    {"kJ", CatEnergy, big.NewRat(1000, 1)},
	"mJ":    {"mJ", CatEnergy, big.NewRat(1, 1000)},
	"MJ":    {"MJ", CatEnergy, big.NewRat(1000000, 1)},
	"GJ":    {"GJ", CatEnergy, big.NewRat(1000000000, 1)},
	"eV":    {"eV", CatEnergy, new(big.Rat).SetFrac(big.NewInt(1602176634), new(big.Int).Exp(big.NewInt(10), big.NewInt(28), nil))}, // 1.602176634e-19 J
	"keV":   {"keV", CatEnergy, new(big.Rat).SetFrac(big.NewInt(1602176634), new(big.Int).Exp(big.NewInt(10), big.NewInt(25), nil))},
	"MeV":   {"MeV", CatEnergy, new(big.Rat).SetFrac(big.NewInt(1602176634), new(big.Int).Exp(big.NewInt(10), big.NewInt(22), nil))},
	"GeV":   {"GeV", CatEnergy, new(big.Rat).SetFrac(big.NewInt(1602176634), new(big.Int).Exp(big.NewInt(10), big.NewInt(19), nil))},
	"cal":   {"cal", CatEnergy, big.NewRat(4184, 1000)}, // 4.184 J
	"kcal":  {"kcal", CatEnergy, big.NewRat(4184, 1)},
	"Wh":    {"Wh", CatEnergy, big.NewRat(3600, 1)},
	"kWh":   {"kWh", CatEnergy, big.NewRat(3600000, 1)},

	// Power (base: W)
	"W":    {"W", CatPower, big.NewRat(1, 1)},
	"watt": {"W", CatPower, big.NewRat(1, 1)},
	"kW":   {"kW", CatPower, big.NewRat(1000, 1)},
	"mW":   {"mW", CatPower, big.NewRat(1, 1000)},
	"MW":   {"MW", CatPower, big.NewRat(1000000, 1)},
	"GW":   {"GW", CatPower, big.NewRat(1000000000, 1)},
	"hp":   {"hp", CatPower, big.NewRat(745699872, 1000000)}, // ~745.699872 W

	// Pressure (base: Pa)
	"Pa":     {"Pa", CatPressure, big.NewRat(1, 1)},
	"pascal": {"Pa", CatPressure, big.NewRat(1, 1)},
	"kPa":    {"kPa", CatPressure, big.NewRat(1000, 1)},
	"MPa":    {"MPa", CatPressure, big.NewRat(1000000, 1)},
	"GPa":    {"GPa", CatPressure, big.NewRat(1000000000, 1)},
	"bar":    {"bar", CatPressure, big.NewRat(100000, 1)},
	"mbar":   {"mbar", CatPressure, big.NewRat(100, 1)},
	"atm":    {"atm", CatPressure, big.NewRat(101325, 1)},
	"torr":   {"torr", CatPressure, big.NewRat(101325, 760)},
	"mmHg":   {"mmHg", CatPressure, big.NewRat(101325, 760)},
	"psi":    {"psi", CatPressure, big.NewRat(44482216152605, 6451600000)}, // 1 lbf / 1 in^2

	// Force (base: N)
	"N":      {"N", CatForce, big.NewRat(1, 1)},
	"newton": {"N", CatForce, big.NewRat(1, 1)},
	"kN":     {"kN", CatForce, big.NewRat(1000, 1)},
	"mN":     {"mN", CatForce, big.NewRat(1, 1000)},
	"dyn":    {"dyn", CatForce, big.NewRat(1, 100000)},
	"lbf":    {"lbf", CatForce, big.NewRat(4448222, 1000000)},

	// Frequency (base: Hz)
	"Hz":    {"Hz", CatFrequency, big.NewRat(1, 1)},
	"hertz": {"Hz", CatFrequency, big.NewRat(1, 1)},
	"kHz":   {"kHz", CatFrequency, big.NewRat(1000, 1)},
	"MHz":   {"MHz", CatFrequency, big.NewRat(1000000, 1)},
	"GHz":   {"GHz", CatFrequency, big.NewRat(1000000000, 1)},
	"THz":   {"THz", CatFrequency, scaledRat(1, 12)},
	"rpm":   {"rpm", CatFrequency, big.NewRat(1, 60)},

	// Electric Charge (base: C)
	"C":       {"C", CatElectricCharge, big.NewRat(1, 1)},
	"coulomb": {"C", CatElectricCharge, big.NewRat(1, 1)},
	"mC":      {"mC", CatElectricCharge, big.NewRat(1, 1000)},
	"uC":      {"uC", CatElectricCharge, scaledRat(1, -6)},
	"µC":      {"uC", CatElectricCharge, scaledRat(1, -6)},
	"nC":      {"nC", CatElectricCharge, scaledRat(1, -9)},
	"pC":      {"pC", CatElectricCharge, scaledRat(1, -12)},
	"e":       {"e", CatElectricCharge, new(big.Rat).SetFrac(big.NewInt(1602176634), new(big.Int).Exp(big.NewInt(10), big.NewInt(28), nil))},
	"Ah":      {"Ah", CatElectricCharge, big.NewRat(3600, 1)},
	"mAh":     {"mAh", CatElectricCharge, big.NewRat(18, 5)}, // 3.6 C

	// Electric Potential (base: V)
	"V":    {"V", CatVoltage, big.NewRat(1, 1)},
	"volt": {"V", CatVoltage, big.NewRat(1, 1)},
	"kV":   {"kV", CatVoltage, big.NewRat(1000, 1)},
	"mV":   {"mV", CatVoltage, big.NewRat(1, 1000)},
	"uV":   {"uV", CatVoltage, scaledRat(1, -6)},

	// Electric Current (base: A)
	"A":      {"A", CatCurrent, big.NewRat(1, 1)},
	"ampere": {"A", CatCurrent, big.NewRat(1, 1)},
	"mA":     {"mA", CatCurrent, big.NewRat(1, 1000)},
	"uA":     {"uA", CatCurrent, scaledRat(1, -6)},
	"kA":     {"kA", CatCurrent, big.NewRat(1000, 1)},

	// Magnetic Field (base: T)
	"T":     {"T", CatMagneticField, big.NewRat(1, 1)},
	"tesla": {"T", CatMagneticField, big.NewRat(1, 1)},
	"mT":    {"mT", CatMagneticField, big.NewRat(1, 1000)},
	"uT":    {"uT", CatMagneticField, scaledRat(1, -6)},
	"G":     {"G", CatMagneticField, big.NewRat(1, 10000)}, // 1 Gauss = 1e-4 T

	// Amount of Substance (base: mol)
	"mol":  {"mol", CatAmount, big.NewRat(1, 1)},
	"mole": {"mol", CatAmount, big.NewRat(1, 1)},
	"mmol": {"mmol", CatAmount, big.NewRat(1, 1000)},
	"umol": {"umol", CatAmount, scaledRat(1, -6)},
	"kmol": {"kmol", CatAmount, big.NewRat(1000, 1)},

	// Volume (base: m^3)
	"m3":     {"m3", CatVolume, big.NewRat(1, 1)},
	"m^3":    {"m3", CatVolume, big.NewRat(1, 1)},
	"L":      {"L", CatVolume, big.NewRat(1, 1000)},
	"l":      {"L", CatVolume, big.NewRat(1, 1000)},
	"liter":  {"L", CatVolume, big.NewRat(1, 1000)},
	"liters": {"L", CatVolume, big.NewRat(1, 1000)},
	"mL":     {"mL", CatVolume, scaledRat(1, -6)},
	"ml":     {"mL", CatVolume, scaledRat(1, -6)},
	"uL":     {"uL", CatVolume, scaledRat(1, -9)},
	"cm3":    {"cm3", CatVolume, scaledRat(1, -6)},
	"cm^3":   {"cm3", CatVolume, scaledRat(1, -6)},
	"cc":     {"cm3", CatVolume, scaledRat(1, -6)},
}

func init() {
	unitRegistry["ohm"] = UnitInfo{Name: "ohm", Category: CatUnknown, Scale: big.NewRat(1, 1)}
	unitRegistry["Ω"] = UnitInfo{Name: "ohm", Category: CatUnknown, Scale: big.NewRat(1, 1)}
	unitRegistry["F"] = UnitInfo{Name: "F", Category: CatUnknown, Scale: big.NewRat(1, 1)}
	unitRegistry["farad"] = UnitInfo{Name: "F", Category: CatUnknown, Scale: big.NewRat(1, 1)}
	unitRegistry["S"] = UnitInfo{Name: "S", Category: CatUnknown, Scale: big.NewRat(1, 1)}
	unitRegistry["siemens"] = UnitInfo{Name: "S", Category: CatUnknown, Scale: big.NewRat(1, 1)}
	unitRegistry["Wb"] = UnitInfo{Name: "Wb", Category: CatUnknown, Scale: big.NewRat(1, 1)}
	unitRegistry["weber"] = UnitInfo{Name: "Wb", Category: CatUnknown, Scale: big.NewRat(1, 1)}
	unitRegistry["H"] = UnitInfo{Name: "H", Category: CatUnknown, Scale: big.NewRat(1, 1)}
	unitRegistry["henry"] = UnitInfo{Name: "H", Category: CatUnknown, Scale: big.NewRat(1, 1)}
	unitRegistry["cd"] = UnitInfo{Name: "cd", Category: CatUnknown, Scale: big.NewRat(1, 1)}
	unitRegistry["candela"] = UnitInfo{Name: "cd", Category: CatUnknown, Scale: big.NewRat(1, 1)}
	unitRegistry["sr"] = UnitInfo{Name: "sr", Category: CatUnknown, Scale: big.NewRat(1, 1)}
	unitRegistry["rad"] = UnitInfo{Name: "rad", Category: CatUnknown, Scale: big.NewRat(1, 1)}
}

func NormalizeUnit(u string) (UnitInfo, bool) {
	u = strings.TrimSpace(u)
	if info, ok := unitRegistry[u]; ok {
		return info, true
	}
	if info, ok := unitRegistry[strings.ToLower(u)]; ok {
		return info, true
	}
	return UnitInfo{}, false
}

func IsKnownUnit(u string) bool {
	_, ok := NormalizeUnit(u)
	return ok
}

func findTopLevelSlash(s string) int {
	depth := 0
	lastSlash := -1
	for i := 0; i < len(s); i++ {
		switch s[i] {
		case '(':
			depth++
		case ')':
			if depth > 0 {
				depth--
			}
		case '/':
			if depth == 0 {
				lastSlash = i
			}
		}
	}
	return lastSlash
}

// ParseUnitExpression parses an arbitrary compound unit expression into a scale factor and 7-dim exponent vector.
func ParseUnitExpression(u string) (*big.Rat, Dimensions, error) {
	u = strings.TrimSpace(u)
	if u == "" || u == "1" {
		return big.NewRat(1, 1), DimDimensionless, nil
	}

	// Direct lookup for exact unit in registry
	if info, ok := unitRegistry[u]; ok {
		return new(big.Rat).Set(info.Scale), info.GetDimensions(), nil
	}
	if info, ok := unitRegistry[strings.ToLower(u)]; ok {
		return new(big.Rat).Set(info.Scale), info.GetDimensions(), nil
	}

	// Split by '/' if present outside parentheses (left-associative: a/b/c = (a/b)/c)
	slashIdx := findTopLevelSlash(u)
	if slashIdx >= 0 {
		numStr := strings.TrimSpace(u[:slashIdx])
		denStr := strings.TrimSpace(u[slashIdx+1:])

		// Strip surrounding parens from denominator if any
		if strings.HasPrefix(denStr, "(") && strings.HasSuffix(denStr, ")") {
			denStr = strings.TrimSpace(denStr[1 : len(denStr)-1])
		}

		numScale, numDim, err := parseUnitProduct(numStr)
		if err != nil {
			return nil, DimDimensionless, err
		}
		denScale, denDim, err := parseUnitProduct(denStr)
		if err != nil {
			return nil, DimDimensionless, err
		}

		resScale := new(big.Rat).Quo(numScale, denScale)
		resDim := numDim.Sub(denDim)
		return resScale, resDim, nil
	}

	return parseUnitProduct(u)
}

func parseUnitProduct(s string) (*big.Rat, Dimensions, error) {
	s = strings.TrimSpace(s)
	if s == "" || s == "1" {
		return big.NewRat(1, 1), DimDimensionless, nil
	}

	if info, ok := unitRegistry[s]; ok {
		return new(big.Rat).Set(info.Scale), info.GetDimensions(), nil
	}
	if info, ok := unitRegistry[strings.ToLower(s)]; ok {
		return new(big.Rat).Set(info.Scale), info.GetDimensions(), nil
	}

	// Tokenize preserving parenthesized groups
	var tokens []string
	var cur strings.Builder
	depth := 0

	for i := 0; i < len(s); i++ {
		ch := s[i]
		if ch == '(' {
			depth++
			cur.WriteByte(ch)
		} else if ch == ')' {
			if depth > 0 {
				depth--
			}
			cur.WriteByte(ch)
		} else if depth == 0 && (ch == ' ' || ch == '*' || ch == '·') {
			if cur.Len() > 0 {
				tokens = append(tokens, cur.String())
				cur.Reset()
			}
		} else {
			cur.WriteByte(ch)
		}
	}
	if cur.Len() > 0 {
		tokens = append(tokens, cur.String())
	}

	totalScale := big.NewRat(1, 1)
	totalDim := DimDimensionless

	for _, tok := range tokens {
		tok = strings.TrimSpace(tok)
		if tok == "" || tok == "1" {
			continue
		}

		var base string
		exp := int64(1)

		if strings.Contains(tok, "^") {
			parts := strings.SplitN(tok, "^", 2)
			base = parts[0]
			p, err := strconv.ParseInt(parts[1], 10, 64)
			if err != nil {
				return nil, DimDimensionless, fmt.Errorf("ErrInvalidUnit: invalid exponent in %q", tok)
			}
			exp = p
		} else {
			if _, ok := unitRegistry[tok]; ok {
				base = tok
			} else {
				idx := len(tok)
				for idx > 0 && tok[idx-1] >= '0' && tok[idx-1] <= '9' {
					idx--
				}
				if idx > 0 && idx < len(tok) {
					base = tok[:idx]
					p, _ := strconv.ParseInt(tok[idx:], 10, 64)
					exp = p
				} else {
					base = tok
				}
			}
		}

		var atomScale *big.Rat
		var atomDim Dimensions

		if strings.HasPrefix(base, "(") && strings.HasSuffix(base, ")") {
			inner := strings.TrimSpace(base[1 : len(base)-1])
			s, d, err := ParseUnitExpression(inner)
			if err != nil {
				return nil, DimDimensionless, err
			}
			atomScale = s
			atomDim = d
		} else {
			info, ok := unitRegistry[base]
			if !ok {
				info, ok = unitRegistry[strings.ToLower(base)]
			}
			if !ok {
				return nil, DimDimensionless, fmt.Errorf("ErrInvalidUnit: unknown unit %q", base)
			}
			atomScale = info.Scale
			atomDim = info.GetDimensions()
		}

		scaledAtom := ratPowInt(atomScale, exp)
		totalScale.Mul(totalScale, scaledAtom)
		totalDim = totalDim.Add(atomDim.MulScalar(int8(exp)))
	}

	return totalScale, totalDim, nil
}

// FormatCanonicalUnit formats an SI base dimensions vector into canonical unit string.
// Prefers named derived units (J, N, W, Pa, C, V, ohm, ...) when vector matches one exactly.
func FormatCanonicalUnit(dim Dimensions) string {
	if dim.IsDimensionless() {
		return ""
	}

	// 1. Single named derived units
	switch dim {
	case DimEnergy: // [2, 1, -2, 0, 0, 0, 0]
		return "J"
	case DimForce: // [1, 1, -2, 0, 0, 0, 0]
		return "N"
	case DimPower: // [2, 1, -3, 0, 0, 0, 0]
		return "W"
	case DimPressure: // [-1, 1, -2, 0, 0, 0, 0]
		return "Pa"
	case DimCharge: // [0, 0, 1, 1, 0, 0, 0]
		return "C"
	case DimVoltage: // [2, 1, -3, -1, 0, 0, 0]
		return "V"
	case DimResistance: // [2, 1, -3, -2, 0, 0, 0]
		return "ohm"
	case DimCapacitance: // [-2, -1, 4, 2, 0, 0, 0]
		return "F"
	case DimConductance: // [-2, -1, 3, 2, 0, 0, 0]
		return "S"
	case DimMagField: // [0, 1, -2, -1, 0, 0, 0]
		return "T"
	case DimMagFlux: // [2, 1, -2, -1, 0, 0, 0]
		return "Wb"
	case DimInductance: // [2, 1, -2, -2, 0, 0, 0]
		return "H"
	case DimLuminous: // [0, 0, 0, 0, 0, 0, 1]
		return "cd"
	}

	// 2. Single base units
	switch dim {
	case DimLength: // [1, 0, 0, 0, 0, 0, 0]
		return "m"
	case DimMass: // [0, 1, 0, 0, 0, 0, 0]
		return "kg"
	case DimTime: // [0, 0, 1, 0, 0, 0, 0]
		return "s"
	case DimCurrent: // [0, 0, 0, 1, 0, 0, 0]
		return "A"
	case DimTemperature: // [0, 0, 0, 0, 1, 0, 0]
		return "K"
	case DimAmount: // [0, 0, 0, 0, 0, 1, 0]
		return "mol"
	}

	// 3. Common canonical physics combinations
	switch dim {
	case DimSpeed: // [1, 0, -1, 0, 0, 0, 0]
		return "m/s"
	case Dimensions{1, 0, -2, 0, 0, 0, 0}:
		return "m/s^2"
	case DimArea: // [2, 0, 0, 0, 0, 0, 0]
		return "m^2"
	case DimVolume: // [3, 0, 0, 0, 0, 0, 0]
		return "m^3"
	case Dimensions{-1, 0, 0, 0, 0, 0, 0}:
		return "1/m"
	case Dimensions{-2, 0, 0, 0, 0, 0, 0}:
		return "1/m^2"
	case Dimensions{-3, 0, 0, 0, 0, 0, 0}:
		return "1/m^3"
	case DimFrequency: // [0, 0, -1, 0, 0, 0, 0]
		return "1/s"
	case Dimensions{0, 0, -2, 0, 0, 0, 0}:
		return "1/s^2"
	case Dimensions{2, 1, -1, 0, 0, 0, 0}: // J s (Planck constant)
		return "J s"
	case Dimensions{3, 1, -2, 0, 0, 0, 0}: // J m (Photon hc)
		return "J m"
	case Dimensions{2, 1, -2, 0, -1, 0, 0}: // J/K (Boltzmann constant)
		return "J/K"
	case Dimensions{2, 1, -2, 0, -1, -1, 0}: // J/(mol K) (Gas constant)
		return "J/(mol K)"
	case Dimensions{3, 1, -4, -2, 0, 0, 0}: // N m^2/C^2 (Coulomb constant)
		return "N m^2/C^2"
	case Dimensions{3, -1, -2, 0, 0, 0, 0}: // m^3/(kg s^2) (Gravitational constant G)
		return "m^3/(kg s^2)"
	case Dimensions{0, 1, -3, 0, -4, 0, 0}: // W/(m^2 K^4) (Stefan-Boltzmann constant)
		return "W/(m^2 K^4)"
	case Dimensions{0, 1, -3, 0, 0, 0, 0}: // W/m^2 (Solar irradiance)
		return "W/m^2"
	case Dimensions{0, 0, 1, 1, 0, -1, 0}: // C/mol (Faraday constant)
		return "C/mol"
	case Dimensions{0, 1, 0, 0, 0, -1, 0}: // kg/mol (Molar mass)
		return "kg/mol"
	case Dimensions{-3, 1, 0, 0, 0, 0, 0}: // kg/m^3 (Density)
		return "kg/m^3"
	case Dimensions{1, 1, -1, 0, 0, 0, 0}: // kg m/s (Momentum)
		return "kg m/s"
	case Dimensions{0, 1, -2, 0, 0, 0, 0}: // N/m (Spring constant)
		return "N/m"
	case Dimensions{1, 1, -3, -1, 0, 0, 0}: // V/m (Electric field)
		return "V/m"
	}

	// 4. General fallback: format from base dimensions
	baseNames := [7]string{"m", "kg", "s", "A", "K", "mol", "cd"}
	var numParts []string
	var denParts []string

	for i := 0; i < 7; i++ {
		p := dim[i]
		if p > 0 {
			if p == 1 {
				numParts = append(numParts, baseNames[i])
			} else {
				numParts = append(numParts, fmt.Sprintf("%s^%d", baseNames[i], p))
			}
		} else if p < 0 {
			pos := -p
			if pos == 1 {
				denParts = append(denParts, baseNames[i])
			} else {
				denParts = append(denParts, fmt.Sprintf("%s^%d", baseNames[i], pos))
			}
		}
	}

	if len(numParts) == 0 {
		if len(denParts) == 0 {
			return ""
		}
		if len(denParts) == 1 {
			return "1/" + denParts[0]
		}
		return "1/(" + strings.Join(denParts, " ") + ")"
	}

	numStr := strings.Join(numParts, " ")
	if len(denParts) == 0 {
		return numStr
	}
	if len(denParts) == 1 {
		return numStr + "/" + denParts[0]
	}
	return numStr + "/(" + strings.Join(denParts, " ") + ")"
}

// MultiplyUnits multiplies two unit quantities using general dimensional analysis.
func MultiplyUnits(v1Val *big.Rat, u1 string, v2Val *big.Rat, u2 string) (*big.Rat, string, bool, error) {
	s1, d1, err1 := ParseUnitExpression(u1)
	if err1 != nil {
		return nil, "", false, err1
	}
	s2, d2, err2 := ParseUnitExpression(u2)
	if err2 != nil {
		return nil, "", false, err2
	}

	dRes := d1.Add(d2)
	valSI1 := new(big.Rat).Mul(v1Val, s1)
	valSI2 := new(big.Rat).Mul(v2Val, s2)
	resSI := new(big.Rat).Mul(valSI1, valSI2)

	if dRes.IsDimensionless() {
		return resSI, "", true, nil
	}

	canonUnit := FormatCanonicalUnit(dRes)
	sCanon, _, errCanon := ParseUnitExpression(canonUnit)
	if errCanon != nil || sCanon.Sign() == 0 {
		return resSI, canonUnit, false, nil
	}
	resVal := new(big.Rat).Quo(resSI, sCanon)
	return resVal, canonUnit, false, nil
}

// DivideUnits divides two unit quantities using general dimensional analysis.
func DivideUnits(v1Val *big.Rat, u1 string, v2Val *big.Rat, u2 string) (*big.Rat, string, bool, error) {
	if v2Val.Sign() == 0 {
		return nil, "", false, fmt.Errorf("ErrDivZero: division by zero")
	}

	s1, d1, err1 := ParseUnitExpression(u1)
	if err1 != nil {
		return nil, "", false, err1
	}
	s2, d2, err2 := ParseUnitExpression(u2)
	if err2 != nil {
		return nil, "", false, err2
	}

	dRes := d1.Sub(d2)
	valSI1 := new(big.Rat).Mul(v1Val, s1)
	valSI2 := new(big.Rat).Mul(v2Val, s2)
	resSI := new(big.Rat).Quo(valSI1, valSI2)

	if dRes.IsDimensionless() {
		return resSI, "", true, nil
	}

	canonUnit := FormatCanonicalUnit(dRes)
	sCanon, _, errCanon := ParseUnitExpression(canonUnit)
	if errCanon != nil || sCanon.Sign() == 0 {
		return resSI, canonUnit, false, nil
	}
	resVal := new(big.Rat).Quo(resSI, sCanon)
	return resVal, canonUnit, false, nil
}

// PowerUnits computes exact integer power of a unit quantity.
func PowerUnits(vVal *big.Rat, u string, exp int64) (*big.Rat, string, bool, error) {
	if exp == 0 {
		return big.NewRat(1, 1), "", true, nil
	}

	s, d, err := ParseUnitExpression(u)
	if err != nil {
		return nil, "", false, err
	}

	dRes := d.MulScalar(int8(exp))
	valSI := new(big.Rat).Mul(vVal, s)
	resSI := ratPowInt(valSI, exp)

	if dRes.IsDimensionless() {
		return resSI, "", true, nil
	}

	canonUnit := FormatCanonicalUnit(dRes)
	sCanon, _, errCanon := ParseUnitExpression(canonUnit)
	if errCanon != nil || sCanon.Sign() == 0 {
		return resSI, canonUnit, false, nil
	}
	resVal := new(big.Rat).Quo(resSI, sCanon)
	return resVal, canonUnit, false, nil
}

func multiplyUnitNames(u1, u2 string) string {
	_, unit, _, err := MultiplyUnits(big.NewRat(1, 1), u1, big.NewRat(1, 1), u2)
	if err != nil {
		return fmt.Sprintf("%s %s", u1, u2)
	}
	return unit
}

func divideUnitNames(u1, u2 string) string {
	_, unit, _, err := DivideUnits(big.NewRat(1, 1), u1, big.NewRat(1, 1), u2)
	if err != nil {
		return fmt.Sprintf("%s/%s", u1, u2)
	}
	return unit
}

// ConvertUnits performs exact conversion of a rational value between units.
func ConvertUnits(val *big.Rat, fromUnitStr, toUnitStr string) (*big.Rat, error) {
	fromNorm := strings.TrimSpace(fromUnitStr)
	toNorm := strings.TrimSpace(toUnitStr)

	// Contextual resolution for "C" / "c": if paired with a temperature unit, treat as Celsius
	isTempUnit := func(s string) bool {
		sl := strings.ToLower(s)
		return sl == "k" || sl == "kelvin" || sl == "f" || sl == "degf" || sl == "fahrenheit" || sl == "degc" || sl == "celsius"
	}

	if (fromNorm == "C" || fromNorm == "c") && isTempUnit(toNorm) {
		fromNorm = "degC"
	}
	if (toNorm == "C" || toNorm == "c") && isTempUnit(fromNorm) {
		toNorm = "degC"
	}

	// Temperature conversions (affine)
	if isTempUnit(fromNorm) || isTempUnit(toNorm) {
		fromInfo, okFrom := NormalizeUnit(fromNorm)
		toInfo, okTo := NormalizeUnit(toNorm)
		if okFrom && okTo && fromInfo.Category == CatTemperature && toInfo.Category == CatTemperature {
			return convertTemperature(val, fromInfo.Name, toInfo.Name)
		}
	}

	fromScale, fromDim, errFrom := ParseUnitExpression(fromNorm)
	if errFrom != nil {
		return nil, fmt.Errorf("ErrInvalidUnit: unknown unit %q", fromUnitStr)
	}
	toScale, toDim, errTo := ParseUnitExpression(toNorm)
	if errTo != nil {
		return nil, fmt.Errorf("ErrInvalidUnit: unknown unit %q", toUnitStr)
	}

	if fromDim != toDim {
		return nil, fmt.Errorf("ErrUnitMismatch: cannot convert from %q to %q (incompatible categories)", fromUnitStr, toUnitStr)
	}

	if toScale.Sign() == 0 {
		return nil, fmt.Errorf("ErrDivZero: zero scale unit")
	}

	// Standard proportional conversions: val * (fromScale / toScale)
	ratio := new(big.Rat).Quo(fromScale, toScale)
	res := new(big.Rat).Mul(val, ratio)
	return res, nil
}

func convertTemperature(val *big.Rat, from, to string) (*big.Rat, error) {
	// First convert from source to Celsius
	var c *big.Rat
	switch from {
	case "degC", "C", "c", "celsius":
		c = new(big.Rat).Set(val)
	case "F", "f", "degF", "fahrenheit":
		// C = (F - 32) * 5/9
		fMinus32 := new(big.Rat).Sub(val, big.NewRat(32, 1))
		c = new(big.Rat).Mul(fMinus32, big.NewRat(5, 9))
	case "K", "k", "kelvin":
		// C = K - 273.15 = K - 5463/20
		c = new(big.Rat).Sub(val, big.NewRat(5463, 20))
	default:
		return nil, fmt.Errorf("ErrInvalidUnit: unknown temperature unit %q", from)
	}

	// Now convert Celsius to destination
	switch to {
	case "degC", "C", "c", "celsius":
		return c, nil
	case "F", "f", "degF", "fahrenheit":
		// F = C * 9/5 + 32
		cMul95 := new(big.Rat).Mul(c, big.NewRat(9, 5))
		return new(big.Rat).Add(cMul95, big.NewRat(32, 1)), nil
	case "K", "k", "kelvin":
		// K = C + 273.15 = C + 5463/20
		return new(big.Rat).Add(c, big.NewRat(5463, 20)), nil
	default:
		return nil, fmt.Errorf("ErrInvalidUnit: unknown temperature unit %q", to)
	}
}
