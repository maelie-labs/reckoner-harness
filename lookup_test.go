package main

import (
	"fmt"
	"sort"
	"strings"
	"testing"
)

func TestLookupBasic(t *testing.T) {
	rs := NewRegisterStore()

	// 1. Physical constant binding
	res, err := EvaluateLookupWithStore(`lookup("speed of light")`, rs)
	if err != nil {
		t.Fatalf("lookup speed of light failed: %v", err)
	}
	if res != "<|r0|> = 299792458 m/s" {
		t.Errorf("speed of light expected '<|r0|> = 299792458 m/s', got %q", res)
	}

	// 2. Element field binding
	res, err = EvaluateLookupWithStore(`lookup("Fe", "atomic_weight")`, rs)
	if err != nil {
		t.Fatalf("lookup Fe atomic_weight failed: %v", err)
	}
	if res != "<|r1|> = 55.84" {
		t.Errorf("Fe atomic_weight expected '<|r1|> = 55.84', got %q", res)
	}

	// 3. Bare element excerpt (NO binding!)
	res, err = EvaluateLookupWithStore(`lookup("Fe")`, rs)
	if err != nil {
		t.Fatalf("lookup Fe excerpt failed: %v", err)
	}
	if strings.Contains(res, "<|r") {
		t.Errorf("bare element excerpt must NOT bind registers, got %q", res)
	}
	if !strings.Contains(res, "Iron (Fe, Z=26)") || !strings.Contains(res, "atomic_weight=55.84") {
		t.Errorf("bare element excerpt missing expected info, got %q", res)
	}

	// Check that next register is still r2 (r2 was NOT consumed by bare excerpt)
	res, err = EvaluateLookupWithStore(`lookup("c")`, rs)
	if err != nil {
		t.Fatalf("lookup c failed: %v", err)
	}
	if !strings.HasPrefix(res, "<|r2|> = 299792458 m/s") {
		t.Errorf("next bind expected r2, got %q", res)
	}

	// 4. Astronomical body field
	res, err = EvaluateLookupWithStore(`lookup("Earth", "mass")`, rs)
	if err != nil {
		t.Fatalf("lookup Earth mass failed: %v", err)
	}
	if !strings.Contains(res, "5.97e24 kg") {
		t.Errorf("Earth mass expected 5.97e24 kg, got %q", res)
	}

	// 5. Bare astronomical excerpt (NO binding!)
	res, err = EvaluateLookupWithStore(`lookup("Earth")`, rs)
	if err != nil {
		t.Fatalf("lookup Earth excerpt failed: %v", err)
	}
	if strings.Contains(res, "<|r") {
		t.Errorf("bare astronomy excerpt must NOT bind registers, got %q", res)
	}

	// 6. Fuzzy candidate list (NO binding!)
	res, err = EvaluateLookupWithStore(`lookup("plank")`, rs)
	if err != nil {
		t.Fatalf("lookup plank candidate list failed: %v", err)
	}
	if !strings.HasPrefix(res, "candidates:") || !strings.Contains(res, "Planck constant") {
		t.Errorf("expected candidates list for plank, got %q", res)
	}

	// 7. Unknown key -> ErrNotFound
	_, err = EvaluateLookupWithStore(`lookup("xyznonexistentrandom12345")`, rs)
	if err == nil || !strings.HasPrefix(err.Error(), "ErrNotFound") {
		t.Errorf("expected ErrNotFound for unknown key, got %v", err)
	}

	// 8. Element other fields
	res, err = EvaluateLookupWithStore(`lookup("Fe", "density")`, rs)
	if err != nil || !strings.Contains(res, "7.874 g/cm3") {
		t.Errorf("Fe density expected 7.874 g/cm3, got %q (err: %v)", res, err)
	}

	res, err = EvaluateLookupWithStore(`lookup("Fe", "group")`, rs)
	if err != nil || !strings.Contains(res, "8") {
		t.Errorf("Fe group expected 8, got %q (err: %v)", res, err)
	}

	res, err = EvaluateLookupWithStore(`lookup("Fe", "period")`, rs)
	if err != nil || !strings.Contains(res, "4") {
		t.Errorf("Fe period expected 4, got %q (err: %v)", res, err)
	}
}

func TestLookupAmbiguousAndCaseSensitive(t *testing.T) {
	rs := NewRegisterStore()

	// 1. m_p (proton mass) vs m_P (Planck mass)
	res, err := EvaluateLookupWithStore(`lookup("m_p")`, rs)
	if err != nil {
		t.Fatalf("lookup(m_p) failed: %v", err)
	}
	if !strings.Contains(res, "1.67262192595e-27") && !strings.Contains(res, "1.67262") {
		t.Errorf("expected proton mass for m_p, got %q", res)
	}

	res, err = EvaluateLookupWithStore(`lookup("m_P")`, rs)
	if err != nil {
		t.Fatalf("lookup(m_P) failed: %v", err)
	}
	if !strings.Contains(res, "2.176434e-08") && !strings.Contains(res, "2.176434e-8") {
		t.Errorf("expected Planck mass for m_P, got %q", res)
	}

	// 2. F collision: bare "F" must return candidate list
	res, err = EvaluateLookupWithStore(`lookup("F")`, rs)
	if err != nil {
		t.Fatalf("lookup(F) failed: %v", err)
	}
	if !strings.HasPrefix(res, "candidates:") || !strings.Contains(res, "Faraday constant") || !strings.Contains(res, "F") {
		t.Errorf("expected candidates list for bare F, got %q", res)
	}

	// 3. F disambiguated by property "electronegativity" -> Fluorine
	res, err = EvaluateLookupWithStore(`lookup("F", "electronegativity")`, rs)
	if err != nil {
		t.Fatalf("lookup(F, electronegativity) failed: %v", err)
	}
	if !strings.Contains(res, "3.98") {
		t.Errorf("expected electronegativity 3.98 for F, got %q", res)
	}

	// 4. F disambiguated by property "unit" -> Faraday constant
	res, err = EvaluateLookupWithStore(`lookup("F", "unit")`, rs)
	if err != nil {
		t.Fatalf("lookup(F, unit) failed: %v", err)
	}
	if !strings.Contains(res, "C/mol") && !strings.Contains(res, "C mol^-1") {
		t.Errorf("expected unit C/mol for Faraday constant, got %q", res)
	}

	// 5. Full names resolve directly
	res, err = EvaluateLookupWithStore(`lookup("Fluorine")`, rs)
	if err != nil {
		t.Fatalf("lookup(Fluorine) failed: %v", err)
	}
	if !strings.Contains(res, "Fluorine") || !strings.Contains(res, "Z=9") {
		t.Errorf("expected Fluorine element record, got %q", res)
	}

	res, err = EvaluateLookupWithStore(`lookup("Faraday constant")`, rs)
	if err != nil {
		t.Fatalf("lookup(Faraday constant) failed: %v", err)
	}
	if !strings.Contains(res, "96485.33212") {
		t.Errorf("expected Faraday constant value, got %q", res)
	}

	// 6. c (speed of light) vs C (carbon)
	res, err = EvaluateLookupWithStore(`lookup("c")`, rs)
	if err != nil {
		t.Fatalf("lookup(c) failed: %v", err)
	}
	if !strings.Contains(res, "299792458") {
		t.Errorf("expected speed of light for c, got %q", res)
	}

	res, err = EvaluateLookupWithStore(`lookup("C")`, rs)
	if err != nil {
		t.Fatalf("lookup(C) failed: %v", err)
	}
	if !strings.Contains(res, "Carbon") || !strings.Contains(res, "Z=6") {
		t.Errorf("expected Carbon for C, got %q", res)
	}

	// 7. h (Planck constant) vs H (Hydrogen)
	res, err = EvaluateLookupWithStore(`lookup("h")`, rs)
	if err != nil {
		t.Fatalf("lookup(h) failed: %v", err)
	}
	if !strings.Contains(res, "6.62607015e-34") {
		t.Errorf("expected Planck constant for h, got %q", res)
	}

	res, err = EvaluateLookupWithStore(`lookup("H")`, rs)
	if err != nil {
		t.Fatalf("lookup(H) failed: %v", err)
	}
	if !strings.Contains(res, "Hydrogen") || !strings.Contains(res, "Z=1") {
		t.Errorf("expected Hydrogen for H, got %q", res)
	}

	// 8. k (Boltzmann constant) vs K (Potassium)
	res, err = EvaluateLookupWithStore(`lookup("k")`, rs)
	if err != nil {
		t.Fatalf("lookup(k) failed: %v", err)
	}
	if !strings.Contains(res, "1.380649e-23") {
		t.Errorf("expected Boltzmann constant for k, got %q", res)
	}

	res, err = EvaluateLookupWithStore(`lookup("K")`, rs)
	if err != nil {
		t.Fatalf("lookup(K) failed: %v", err)
	}
	if !strings.Contains(res, "Potassium") || !strings.Contains(res, "Z=19") {
		t.Errorf("expected Potassium for K, got %q", res)
	}

	// 9. Mercury (element vs planet)
	res, err = EvaluateLookupWithStore(`lookup("Mercury")`, rs)
	if err != nil {
		t.Fatalf("lookup(Mercury) failed: %v", err)
	}
	if !strings.HasPrefix(res, "candidates:") {
		t.Errorf("expected candidate list for ambiguous Mercury, got %q", res)
	}

	// Both have density, so density remains ambiguous -> candidates
	res, err = EvaluateLookupWithStore(`lookup("Mercury", "density")`, rs)
	if err != nil {
		t.Fatalf("lookup(Mercury, density) failed: %v", err)
	}
	if !strings.HasPrefix(res, "candidates:") {
		t.Errorf("expected candidate list for Mercury density (both have density), got %q", res)
	}

	// atomic_weight uniquely selects element Mercury
	res, err = EvaluateLookupWithStore(`lookup("Mercury", "atomic_weight")`, rs)
	if err != nil {
		t.Fatalf("lookup(Mercury, atomic_weight) failed: %v", err)
	}
	if !strings.Contains(res, "200.59") {
		t.Errorf("expected element Mercury atomic weight 200.59, got %q", res)
	}

	// orbital_period uniquely selects astronomy Mercury
	res, err = EvaluateLookupWithStore(`lookup("Mercury", "orbital_period")`, rs)
	if err != nil {
		t.Fatalf("lookup(Mercury, orbital_period) failed: %v", err)
	}
	if !strings.Contains(res, "88.0") && !strings.Contains(res, "87.97") {
		t.Errorf("expected astronomy Mercury orbital period, got %q", res)
	}
}

func TestLookupCalcComposition(t *testing.T) {
	rs := NewRegisterStore()

	// 1. Lookup speed of light: binds <|r0|> = 299792458 m/s
	_, err := EvaluateLookupWithStore(`lookup("speed of light")`, rs)
	if err != nil {
		t.Fatalf("lookup speed of light failed: %v", err)
	}

	// 2. Lookup Planck constant: binds <|r1|> = 6.62607015e-34 J s
	_, err = EvaluateLookupWithStore(`lookup("Planck constant")`, rs)
	if err != nil {
		t.Fatalf("lookup Planck constant failed: %v", err)
	}

	// 3. Evaluate E = h * c / lambda in calc:
	// Exact rational evaluation
	calcRes, err := EvaluateInfixWithStore("<|r1|> * <|r0|> / (500 * 10^-9)", rs)
	if err != nil {
		t.Fatalf("calc E = h*c/lambda failed: %v", err)
	}
	t.Logf("Calculated exact photon energy: %s", calcRes)
	if !strings.Contains(calcRes, "19864458571489287/50000000000000000000000000000000000") {
		t.Errorf("expected exact fraction, got %q", calcRes)
	}

	// 4. Evaluate float(E):
	floatRes, err := EvaluateInfixWithStore("float(<|r1|> * <|r0|> / (500 * 10^-9))", rs)
	if err != nil {
		t.Fatalf("float calc failed: %v", err)
	}
	t.Logf("Calculated float photon energy: %s", floatRes)
	if !strings.Contains(floatRes, "3.97289e-19") {
		t.Errorf("expected ~3.97289e-19, got %q", floatRes)
	}

	// 4. Multiply speed of light by 2
	scaleRes, err := EvaluateInfixWithStore("<|r0|> * 2", rs)
	if err != nil {
		t.Fatalf("calc c * 2 failed: %v", err)
	}
	if !strings.Contains(scaleRes, "599584916 m/s") {
		t.Errorf("expected 599584916 m/s, got %q", scaleRes)
	}
}

func BenchmarkLookupExact(b *testing.B) {
	rs := NewRegisterStore()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = EvaluateLookupWithStore(`lookup("speed of light")`, rs)
	}
}

func BenchmarkLookupElementField(b *testing.B) {
	rs := NewRegisterStore()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = EvaluateLookupWithStore(`lookup("Fe", "atomic_weight")`, rs)
	}
}

func BenchmarkLookupFuzzy(b *testing.B) {
	rs := NewRegisterStore()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = EvaluateLookupWithStore(`lookup("plank")`, rs)
	}
}

func FuzzLookup(f *testing.F) {
	seeds := []struct {
		key   string
		field string
	}{
		{"c", ""},
		{"speed of light", ""},
		{"Fe", ""},
		{"Fe", "atomic_weight"},
		{"Earth", ""},
		{"Earth", "mass"},
		{"plank", ""},
		{"nonexistent", ""},
		{"", ""},
		{"a", "b"},
		{"!@#$%^&*()", ""},
		{"\x00\xff\xfe", ""},
		{"Planck constant", "unit"},
		{"hydrogen", "boiling_point"},
	}
	for _, s := range seeds {
		f.Add(s.key, s.field)
	}

	f.Fuzz(func(t *testing.T, key, field string) {
		rs := NewRegisterStore()
		var call string
		if field == "" {
			call = fmt.Sprintf("lookup(%q)", key)
		} else {
			call = fmt.Sprintf("lookup(%q, %q)", key, field)
		}
		// Call EvaluateLookupWithStore. It must never panic, regardless of input.
		_, _ = EvaluateLookupWithStore(call, rs)
	})
}

func bruteForceFuzzy(rawKey string) ([]string, error) {
	initLookupTable()
	normKey := normalizeKey(rawKey)
	type rankedCandidate struct {
		id    string
		score float64
	}
	var matches []rankedCandidate
	const minScoreThreshold = 0.55

	for _, cand := range lookupTable.candidateList {
		score := scoreCandidate(normKey, cand)
		if score >= minScoreThreshold {
			matches = append(matches, rankedCandidate{
				id:    cand.id,
				score: score,
			})
		}
	}

	if len(matches) == 0 {
		return nil, fmt.Errorf("ErrNotFound: no match found for %q", rawKey)
	}

	sort.Slice(matches, func(i, j int) bool {
		if matches[i].score != matches[j].score {
			return matches[i].score > matches[j].score
		}
		return matches[i].id < matches[j].id
	})

	limit := 3
	if len(matches) < limit {
		limit = len(matches)
	}
	var candIDs []string
	for i := 0; i < limit; i++ {
		candIDs = append(candIDs, fmt.Sprintf("%q", matches[i].id))
	}
	return candIDs, nil
}

func TestFuzzyRankingConsistency(t *testing.T) {
	initLookupTable()
	rs := NewRegisterStore()

	queries := []string{
		// Physical constants misspellings (51)
		"plank", "plank constant", "speed of lite", "gravitational constnt", "avagadro",
		"avagadro number", "boltzman", "boltzman constant", "stefan boltzman", "permeability of vacum",
		"permittivty of vacuum", "fine structur", "rydbrg constant", "gas constnt", "faraday constnt",
		"electron mas", "proton mas", "neutron mas", "elementary charg", "magnetic flux quantm",
		"conductance quantm", "von klitzing", "josephson constnt", "bohr radiu", "hartree enrgy",
		"molar gas const", "molar vol", "atomic mass unit constnt", "standard gravty", "wien wavelength",
		"wien freq", "fermi coupling", "weak mixing", "deuteron mas", "triton mas",
		"helion mas", "alpha particle mas", "tau mas", "muon mas", "classical electron radiu",
		"thomson cross sect", "nuclar magneton", "bor magneton", "electrn g factor", "muon g factr",
		"protn g factor", "neutrn g factor", "comptn wavelength", "reduced compton", "astronomical unt",
		"astronomcl unit",

		// Elements misspellings (75)
		"helim", "lithum", "berylium", "boron element", "oxigen",
		"flourine", "neom", "sodum", "magnesum", "aluminm",
		"silicon element", "phosporus", "sulfur element", "chlorne", "argom",
		"potasium", "calcum", "scandim", "titaniom", "vanadim",
		"chromim", "manganse", "cobalt element", "nickle", "coper",
		"zink", "gallim", "germanim", "arsnic", "selenum",
		"bromne", "kryptom", "rubidim", "strontim", "yttrm",
		"zirconim", "niobim", "molybdenm", "technetim", "ruthenim",
		"rhodim", "paladim", "silvr", "cadmim", "indim",
		"tin element", "antimony element", "telurim", "iodne", "xenom",
		"cesim", "barim", "lanthanm", "cerim", "praseodim",
		"neodym", "europim", "gadolinim", "terbim", "dysprosim",
		"holmim", "erbm", "thulim", "ytterbim", "lutetm",
		"hafnn", "tantalm", "tungstn", "rhenim", "osmim",
		"iridim", "platium", "gold element", "mercury element", "lead element",

		// Astronomy bodies misspellings (30)
		"erth", "jupitr", "satrn", "uranus planet", "neptun",
		"mercury planet", "venis", "mars planet", "pluto dwarf", "earh",
		"juptier", "saturn body", "neptune planet", "venuss", "mercuryy",
		"earrth", "juupiter", "saturrn", "marss", "uranus body",
		"mooon", "the earth", "the sun", "the moon", "planet earth",
		"planet mars", "planet jupiter", "planet venus", "planet saturn", "dwarf pluto",

		// Realistic user misspellings and abbreviations (40)
		"planck cnst", "c speed", "g const", "mu 0 const", "eps 0 const",
		"k B boltz", "h planck", "hbar const", "speed of light in vac", "vacuum speed of light",
		"grav accel", "gravitational acc", "std atmosphere", "atm pressure", "coulomb cnst",
		"coulombs law const", "elem charge", "elec charge", "e charge", "p mass",
		"e mass", "n mass", "proton rest mass", "electron rest mass", "neutron rest mass",
		"rydberg const inf", "bohr rad", "hartree energy const", "stefan boltzmann const", "wien const",
		"faraday const", "avogadro const", "universal gas const", "molar planck", "dirac constnt",
		"reduced planck const", "newton gravitational const", "light speed", "sound speed air", "fermi coupling const",

		// Near misses and phrases (30)
		"quantum conductance", "quantum resistance", "quantm of circulation", "hyperfine splitting", "fine structure alpha",
		"planck length scale", "planck mass scale", "planck time scale", "planck temperature scale", "astronomical distance unit",
		"light travel time", "solar mass param", "earth mass param", "jupiter mass param", "lunar mass param",
		"surface gravity earth", "surface gravity moon", "surface gravity mars", "surface gravity jupiter", "sidereal day earth",
		"sidereal year earth", "semi major axis earth", "semi major axis mars", "orbital period earth", "orbital period mars",
		"escape velocity earth", "escape velocity moon", "escape velocity mars", "mean temperature earth", "mean density earth",
	}

	if len(queries) < 200 {
		t.Fatalf("expected >= 200 queries, got %d", len(queries))
	}

	discrepancies := 0
	for _, q := range queries {
		wantCands, wantErr := bruteForceFuzzy(q)
		gotResp, gotErr := EvaluateLookupWithStore(fmt.Sprintf("lookup(%q)", q), rs)

		if wantErr != nil {
			if gotErr == nil || !strings.HasPrefix(gotErr.Error(), "ErrNotFound") {
				t.Errorf("query %q: want ErrNotFound, got err=%v resp=%q", q, gotErr, gotResp)
				discrepancies++
			}
			continue
		}

		if gotErr != nil {
			t.Errorf("query %q: unexpected error: %v", q, gotErr)
			discrepancies++
			continue
		}

		wantResp := fmt.Sprintf("candidates: %s", strings.Join(wantCands, ", "))
		if gotResp != wantResp {
			t.Errorf("query %q ranking mismatch:\n  got:  %q\n  want: %q", q, gotResp, wantResp)
			discrepancies++
		}
	}

	if discrepancies == 0 {
		t.Logf("PASS: %d/%d queries matched identical brute-force top-3 candidates with 0 discrepancies.", len(queries), len(queries))
	} else {
		t.Fatalf("FAILED: %d discrepancies found across %d queries", discrepancies, len(queries))
	}
}


