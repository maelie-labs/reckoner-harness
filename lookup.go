package main

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"math/big"
	"sort"
	"strconv"
	"strings"
	"sync"
	"unicode"
)

// LookupTableVersion defines the table version compiled into the binary.
const LookupTableVersion = "2026.1"

//go:embed lookup_data/constants.json
var embeddedConstantsJSON []byte

//go:embed lookup_data/elements.json
var embeddedElementsJSON []byte

//go:embed lookup_data/astronomy.json
var embeddedAstronomyJSON []byte

type ConstantEntry struct {
	ID          string   `json:"id"`
	Aliases     []string `json:"aliases"`
	RawValue    string   `json:"raw_value"`
	Uncertainty string   `json:"uncertainty"`
	RawUnit     string   `json:"raw_unit"`
	Unit        string   `json:"unit"`
	ValRat      *big.Rat
}

type ElementEntry struct {
	AtomicNumber          int      `json:"atomic_number"`
	Symbol                string   `json:"symbol"`
	Name                  string   `json:"name"`
	AtomicWeight          string   `json:"atomic_weight"`
	Group                 int      `json:"group"`
	Period                int      `json:"period"`
	Density               string   `json:"density"`
	MeltingPoint          string   `json:"melting_point"`
	BoilingPoint          string   `json:"boiling_point"`
	Electronegativity     string   `json:"electronegativity"`
	FirstIonizationEnergy string   `json:"first_ionization_energy"`
	ElectronConfiguration string   `json:"electron_configuration"`
	Aliases               []string `json:"aliases"`
}

type AstronomyEntry struct {
	ID      string            `json:"id"`
	Aliases []string          `json:"aliases"`
	Fields  map[string]string `json:"fields"`
}

type lookupTableState struct {
	constants []ConstantEntry
	elements  []ElementEntry
	astronomy []AstronomyEntry

	// exactIndex maps lowercase normalized keys/aliases to entries
	exactIndex map[string][]interface{}
	// exactCaseIndex maps exact case keys (e.g. "c" vs "C") to entries
	exactCaseIndex map[string][]interface{}

	// candidateList for deterministic ranking
	candidateList []candidateEntry

	// Inverted indices for fast fuzzy candidate filtering
	wordIndex  map[string][]int
	ngramIndex map[string][]int
}

type candidateEntry struct {
	id         string
	normID     string
	aliases    []string
	normAlias  []string
	entryType  string // "constant", "element", "astronomy"
	entryValue interface{}
}

var (
	lookupTable     *lookupTableState
	lookupTableOnce sync.Once
)

func initLookupTable() {
	lookupTableOnce.Do(func() {
		state := &lookupTableState{
			exactIndex:     make(map[string][]interface{}),
			exactCaseIndex: make(map[string][]interface{}),
		}

		if len(embeddedConstantsJSON) > 0 {
			var rawConsts []ConstantEntry
			if err := json.Unmarshal(embeddedConstantsJSON, &rawConsts); err == nil {
				for i := range rawConsts {
					c := rawConsts[i]
					c.ValRat = parseScientificRat(c.RawValue)
					state.constants = append(state.constants, c)
				}
			}
		}

		if len(embeddedElementsJSON) > 0 {
			var rawElems []ElementEntry
			if err := json.Unmarshal(embeddedElementsJSON, &rawElems); err == nil {
				state.elements = rawElems
			}
		}

		if len(embeddedAstronomyJSON) > 0 {
			var rawAstro []AstronomyEntry
			if err := json.Unmarshal(embeddedAstronomyJSON, &rawAstro); err == nil {
				state.astronomy = rawAstro
			}
		}

		addIndexEntry := func(m map[string][]interface{}, key string, entry interface{}) {
			key = strings.TrimSpace(key)
			if key == "" {
				return
			}
			for _, existing := range m[key] {
				if existing == entry {
					return
				}
			}
			m[key] = append(m[key], entry)
		}

		// Build index & candidates
		for i := range state.constants {
			c := &state.constants[i]
			normID := normalizeKey(c.ID)
			addIndexEntry(state.exactCaseIndex, c.ID, c)
			addIndexEntry(state.exactIndex, normID, c)
			cand := candidateEntry{
				id:         c.ID,
				normID:     normID,
				aliases:    c.Aliases,
				entryType:  "constant",
				entryValue: c,
			}
			for _, a := range c.Aliases {
				na := normalizeKey(a)
				addIndexEntry(state.exactCaseIndex, a, c)
				addIndexEntry(state.exactIndex, na, c)
				cand.normAlias = append(cand.normAlias, na)
			}
			state.candidateList = append(state.candidateList, cand)
		}

		for i := range state.elements {
			e := &state.elements[i]
			normSymbol := normalizeKey(e.Symbol)
			normName := normalizeKey(e.Name)
			addIndexEntry(state.exactCaseIndex, e.Symbol, e)
			addIndexEntry(state.exactCaseIndex, e.Name, e)
			addIndexEntry(state.exactIndex, normSymbol, e)
			addIndexEntry(state.exactIndex, normName, e)
			cand := candidateEntry{
				id:         e.Symbol,
				normID:     normSymbol,
				aliases:    e.Aliases,
				entryType:  "element",
				entryValue: e,
			}
			for _, a := range e.Aliases {
				na := normalizeKey(a)
				addIndexEntry(state.exactCaseIndex, a, e)
				addIndexEntry(state.exactIndex, na, e)
				cand.normAlias = append(cand.normAlias, na)
			}
			state.candidateList = append(state.candidateList, cand)
		}

		for i := range state.astronomy {
			a := &state.astronomy[i]
			normID := normalizeKey(a.ID)
			addIndexEntry(state.exactCaseIndex, a.ID, a)
			addIndexEntry(state.exactIndex, normID, a)
			cand := candidateEntry{
				id:         a.ID,
				normID:     normID,
				aliases:    a.Aliases,
				entryType:  "astronomy",
				entryValue: a,
			}
			for _, alias := range a.Aliases {
				na := normalizeKey(alias)
				addIndexEntry(state.exactCaseIndex, alias, a)
				addIndexEntry(state.exactIndex, na, a)
				cand.normAlias = append(cand.normAlias, na)
			}
			state.candidateList = append(state.candidateList, cand)
		}

		// Precompute inverted index for fuzzy candidate filtering
		state.wordIndex = make(map[string][]int)
		state.ngramIndex = make(map[string][]int)

		for cIdx, cand := range state.candidateList {
			seenWords := make(map[string]bool)
			seenGrams := make(map[string]bool)

			indexText := func(text string) {
				words := strings.Fields(text)
				for _, w := range words {
					if !seenWords[w] {
						seenWords[w] = true
						state.wordIndex[w] = append(state.wordIndex[w], cIdx)
					}
					// Character bigrams
					for k := 0; k < len(w)-1; k++ {
						bg := w[k : k+2]
						if !seenGrams[bg] {
							seenGrams[bg] = true
							state.ngramIndex[bg] = append(state.ngramIndex[bg], cIdx)
						}
					}
					// Character trigrams
					for k := 0; k < len(w)-2; k++ {
						tg := w[k : k+3]
						if !seenGrams[tg] {
							seenGrams[tg] = true
							state.ngramIndex[tg] = append(state.ngramIndex[tg], cIdx)
						}
					}
				}
			}

			indexText(cand.normID)
			for _, na := range cand.normAlias {
				indexText(na)
			}
		}

		lookupTable = state
	})
}

func normalizeKey(s string) string {
	s = strings.TrimSpace(strings.ToLower(s))
	// Replace punctuation with spaces, clean multiple spaces
	var b strings.Builder
	for _, r := range s {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			b.WriteRune(r)
		} else {
			b.WriteRune(' ')
		}
	}
	return strings.Join(strings.Fields(b.String()), " ")
}

// parseScientificRat parses a scientific or decimal string into an exact *big.Rat
func parseScientificRat(s string) *big.Rat {
	s = strings.TrimSpace(s)
	s = strings.ReplaceAll(s, "...", "")
	s = strings.ReplaceAll(s, " ", "")
	if s == "" {
		return big.NewRat(0, 1)
	}

	exp := int64(0)
	if eIdx := strings.IndexAny(s, "eE"); eIdx != -1 {
		expStr := s[eIdx+1:]
		s = s[:eIdx]
		if n, err := strconv.ParseInt(expStr, 10, 64); err == nil {
			exp = n
		}
	}

	// Parse mantissa into big.Rat
	var mantissaRat *big.Rat
	if strings.Contains(s, ".") {
		parts := strings.Split(s, ".")
		intPart := parts[0]
		fracPart := parts[1]
		decLen := int64(len(fracPart))

		combinedStr := intPart + fracPart
		combinedInt, ok := new(big.Int).SetString(combinedStr, 10)
		if !ok {
			f, _ := strconv.ParseFloat(s, 64)
			return new(big.Rat).SetFloat64(f)
		}
		den := new(big.Int).Exp(big.NewInt(10), big.NewInt(decLen), nil)
		mantissaRat = new(big.Rat).SetFrac(combinedInt, den)
	} else {
		intVal, ok := new(big.Int).SetString(s, 10)
		if !ok {
			f, _ := strconv.ParseFloat(s, 64)
			return new(big.Rat).SetFloat64(f)
		}
		mantissaRat = new(big.Rat).SetInt(intVal)
	}

	if exp == 0 {
		return mantissaRat
	}

	scale := exp10Rat(exp)
	return new(big.Rat).Mul(mantissaRat, scale)
}

// ParseLookupPayload parses lookup("key"[, "field"]) or bare arguments
func ParseLookupPayload(payload string) (key, field string, err error) {
	payload = strings.TrimSpace(payload)
	if strings.HasPrefix(payload, "lookup(") && strings.HasSuffix(payload, ")") {
		payload = payload[7 : len(payload)-1]
	}

	payload = strings.TrimSpace(payload)
	if payload == "" {
		return "", "", fmt.Errorf("ErrSyntax: empty lookup query")
	}

	// Tokenize comma-separated string arguments
	var args []string
	inQuote := false
	quoteChar := byte(0)
	var current strings.Builder

	for i := 0; i < len(payload); i++ {
		ch := payload[i]
		if !inQuote && (ch == '"' || ch == '\'') {
			inQuote = true
			quoteChar = ch
			continue
		}
		if inQuote && ch == quoteChar {
			inQuote = false
			continue
		}
		if !inQuote && ch == ',' {
			args = append(args, strings.TrimSpace(current.String()))
			current.Reset()
			continue
		}
		current.WriteByte(ch)
	}
	if current.Len() > 0 || len(args) > 0 {
		args = append(args, strings.TrimSpace(current.String()))
	}

	if len(args) == 0 {
		return "", "", fmt.Errorf("ErrSyntax: malformed lookup arguments")
	}

	key = strings.Trim(args[0], `"'`)
	if len(args) > 1 {
		field = strings.Trim(args[1], `"'`)
	}
	return key, field, nil
}

// EvaluateLookup evaluates a lookup tool call against the global register store.
func EvaluateLookup(payload string) (string, error) {
	return EvaluateLookupWithStore(payload, globalRegisters)
}

func entrySupportsField(entry interface{}, field string) bool {
	field = strings.TrimSpace(strings.ToLower(field))
	if field == "" {
		return false
	}
	switch v := entry.(type) {
	case *ConstantEntry:
		return field == "unit" || field == "uncertainty" || field == "value"
	case *ElementEntry:
		switch field {
		case "atomic_weight", "atomic_mass", "weight", "mass",
			"atomic_number", "z", "number",
			"group", "period", "density",
			"melting_point", "mp", "boiling_point", "bp",
			"electronegativity", "first_ionization_energy", "ionization_energy", "ie",
			"electron_configuration", "config", "symbol", "name":
			return true
		default:
			return false
		}
	case *AstronomyEntry:
		if v.Fields != nil {
			_, ok := v.Fields[field]
			return ok
		}
		return false
	default:
		return false
	}
}

type candidateItem struct {
	id    string
	label string
}

func formatCandidateEntry(entry interface{}) (string, string) {
	switch v := entry.(type) {
	case *ElementEntry:
		return v.Symbol, fmt.Sprintf("element %q (%s)", v.Symbol, v.Name)
	case *AstronomyEntry:
		switch v.ID {
		case "Sun":
			return v.ID, fmt.Sprintf("star %q", v.ID)
		case "Moon":
			return v.ID, fmt.Sprintf("moon %q", v.ID)
		case "Pluto":
			return v.ID, fmt.Sprintf("dwarf planet %q", v.ID)
		default:
			return v.ID, fmt.Sprintf("planet %q", v.ID)
		}
	case *ConstantEntry:
		return v.ID, fmt.Sprintf("%q", v.ID)
	default:
		return "", ""
	}
}

func formatCandidateList(entries []interface{}) string {
	var items []candidateItem
	seen := make(map[string]bool)
	for _, e := range entries {
		id, label := formatCandidateEntry(e)
		if id != "" && !seen[id] {
			seen[id] = true
			items = append(items, candidateItem{id: id, label: label})
		}
	}
	sort.Slice(items, func(i, j int) bool {
		return items[i].id < items[j].id
	})
	var formatted []string
	for _, item := range items {
		formatted = append(formatted, item.label)
	}
	return fmt.Sprintf("candidates: %s", strings.Join(formatted, ", "))
}

func resolveLookupMatches(matches []interface{}, rawKey, normField string, store *RegisterStore) (string, error) {
	if len(matches) == 1 {
		return formatLookupResult(matches[0], rawKey, normField, store)
	}

	// Check if all matches are ConstantEntries with identical value and unit
	// (e.g. CODATA duplicate entries for "atomic mass constant" and "unified atomic mass unit")
	if c0, ok := matches[0].(*ConstantEntry); ok {
		allSameConst := true
		for _, m := range matches[1:] {
			if cM, okM := m.(*ConstantEntry); okM {
				if cM.RawValue != c0.RawValue || cM.Unit != c0.Unit {
					allSameConst = false
					break
				}
			} else {
				allSameConst = false
				break
			}
		}
		if allSameConst {
			return formatLookupResult(matches[0], rawKey, normField, store)
		}
	}

	// Check if exactly one match has an exact full ID/Name matching rawKey (not just an alias)
	normRaw := normalizeKey(rawKey)
	var exactIDMatches []interface{}
	for _, m := range matches {
		switch e := m.(type) {
		case *ConstantEntry:
			if strings.EqualFold(e.ID, rawKey) || normalizeKey(e.ID) == normRaw {
				exactIDMatches = append(exactIDMatches, m)
			}
		case *ElementEntry:
			if strings.EqualFold(e.Name, rawKey) || normalizeKey(e.Name) == normRaw {
				exactIDMatches = append(exactIDMatches, m)
			}
		case *AstronomyEntry:
			if strings.EqualFold(e.ID, rawKey) || normalizeKey(e.ID) == normRaw {
				exactIDMatches = append(exactIDMatches, m)
			}
		}
	}
	if len(exactIDMatches) == 1 {
		return formatLookupResult(exactIDMatches[0], rawKey, normField, store)
	}

	// More than 1 match: check if field disambiguates
	if normField != "" {
		var fieldMatches []interface{}
		for _, m := range matches {
			if entrySupportsField(m, normField) {
				fieldMatches = append(fieldMatches, m)
			}
		}
		if len(fieldMatches) == 1 {
			return formatLookupResult(fieldMatches[0], rawKey, normField, store)
		}
		if len(fieldMatches) > 1 {
			return formatCandidateList(fieldMatches), nil
		}
	}
	return formatCandidateList(matches), nil
}

// EvaluateLookupWithStore evaluates a lookup call against a specific RegisterStore.
func EvaluateLookupWithStore(payload string, store *RegisterStore) (string, error) {
	initLookupTable()

	rawKey, field, err := ParseLookupPayload(payload)
	if err != nil {
		return "", err
	}

	normKey := normalizeKey(rawKey)
	normField := strings.TrimSpace(strings.ToLower(field))

	// 1a. Check exact case index first (for single-letter disambiguation like "c" vs "C")
	if matches, ok := lookupTable.exactCaseIndex[rawKey]; ok && len(matches) > 0 {
		return resolveLookupMatches(matches, rawKey, normField, store)
	}

	// 1b. Exact match on normalized key
	if matches, ok := lookupTable.exactIndex[normKey]; ok && len(matches) > 0 {
		return resolveLookupMatches(matches, rawKey, normField, store)
	}

	// Also check rawKey without normalization (case-insensitive)
	lowerRaw := strings.ToLower(rawKey)
	if matches, ok := lookupTable.exactIndex[lowerRaw]; ok && len(matches) > 0 {
		return resolveLookupMatches(matches, rawKey, normField, store)
	}

	// 2. Fuzzy match: deterministic ranking with inverted index filtering
	type rankedCandidate struct {
		id    string
		score float64
	}

	var matches []rankedCandidate
	const minScoreThreshold = 0.55

	qWords := strings.Fields(normKey)
	var filteredCands []int
	seenCand := make([]bool, len(lookupTable.candidateList))

	for _, qw := range qWords {
		if cands, ok := lookupTable.wordIndex[qw]; ok {
			for _, cIdx := range cands {
				if !seenCand[cIdx] {
					seenCand[cIdx] = true
					filteredCands = append(filteredCands, cIdx)
				}
			}
		}
		// N-grams of query word: trigrams for len >= 3, bigrams only for len == 2
		if len(qw) >= 3 {
			for k := 0; k < len(qw)-2; k++ {
				tg := qw[k : k+3]
				if cands, ok := lookupTable.ngramIndex[tg]; ok {
					for _, cIdx := range cands {
						if !seenCand[cIdx] {
							seenCand[cIdx] = true
							filteredCands = append(filteredCands, cIdx)
						}
					}
				}
			}
		} else if len(qw) == 2 {
			bg := qw
			if cands, ok := lookupTable.ngramIndex[bg]; ok {
				for _, cIdx := range cands {
					if !seenCand[cIdx] {
						seenCand[cIdx] = true
						filteredCands = append(filteredCands, cIdx)
					}
				}
			}
		}
	}

	if len(filteredCands) < 10 {
		// Expand with bigrams when trigrams produce fewer than 10 candidate matches
		for _, qw := range qWords {
			for k := 0; k < len(qw)-1; k++ {
				bg := qw[k : k+2]
				if cands, ok := lookupTable.ngramIndex[bg]; ok {
					for _, cIdx := range cands {
						if !seenCand[cIdx] {
							seenCand[cIdx] = true
							filteredCands = append(filteredCands, cIdx)
						}
					}
				}
			}
		}
	}

	if len(filteredCands) == 0 {
		return "", fmt.Errorf("ErrNotFound: no match found for %q", rawKey)
	}

	for _, cIdx := range filteredCands {
		cand := lookupTable.candidateList[cIdx]
		score := scoreCandidate(normKey, cand)
		if score >= minScoreThreshold {
			matches = append(matches, rankedCandidate{
				id:    cand.id,
				score: score,
			})
		}
	}

	if len(matches) == 0 {
		return "", fmt.Errorf("ErrNotFound: no match found for %q", rawKey)
	}

	// Deterministic sort: score DESC, id ASC
	sort.Slice(matches, func(i, j int) bool {
		if matches[i].score != matches[j].score {
			return matches[i].score > matches[j].score
		}
		return matches[i].id < matches[j].id
	})

	// Pick top 3 candidates
	limit := 3
	if len(matches) < limit {
		limit = len(matches)
	}
	var candIDs []string
	for i := 0; i < limit; i++ {
		candIDs = append(candIDs, fmt.Sprintf("%q", matches[i].id))
	}

	return fmt.Sprintf("candidates: %s", strings.Join(candIDs, ", ")), nil
}

func scoreCandidate(queryNorm string, cand candidateEntry) float64 {
	best := scoreString(queryNorm, cand.normID)
	for _, na := range cand.normAlias {
		s := scoreString(queryNorm, na)
		if s > best {
			best = s
		}
	}
	return best
}

func levenshtein(s1, s2 string) int {
	if len(s1) < len(s2) {
		s1, s2 = s2, s1
	}
	if len(s2) == 0 {
		return len(s1)
	}
	prev := make([]int, len(s2)+1)
	for j := range prev {
		prev[j] = j
	}
	for i := 0; i < len(s1); i++ {
		curr := make([]int, len(s2)+1)
		curr[0] = i + 1
		for j := 0; j < len(s2); j++ {
			ins := prev[j+1] + 1
			del := curr[j] + 1
			sub := prev[j]
			if s1[i] != s2[j] {
				sub++
			}
			min := ins
			if del < min {
				min = del
			}
			if sub < min {
				min = sub
			}
			curr[j+1] = min
		}
		prev = curr
	}
	return prev[len(s2)]
}

func wordSimilarity(w1, w2 string) float64 {
	if w1 == w2 {
		return 1.0
	}
	dist := levenshtein(w1, w2)
	maxLen := len(w1)
	if len(w2) > maxLen {
		maxLen = len(w2)
	}
	return 1.0 - float64(dist)/float64(maxLen)
}

func scoreString(query, target string) float64 {
	if query == target {
		return 1.0
	}
	if len(query) == 0 || len(target) == 0 {
		return 0.0
	}

	qWords := strings.Fields(query)
	tWords := strings.Fields(target)
	if len(qWords) == 0 || len(tWords) == 0 {
		return 0.0
	}

	// Exact substring match
	if strings.Contains(target, query) && len(query) >= 3 {
		return 0.65 + 0.30*(float64(len(query))/float64(len(target)))
	}

	total := 0.0
	for _, qw := range qWords {
		best := 0.0
		for _, tw := range tWords {
			sim := wordSimilarity(qw, tw)
			if sim > best {
				best = sim
			}
		}
		total += best
	}
	return total / float64(len(qWords))
}

func formatLookupResult(entry interface{}, rawKey, field string, store *RegisterStore) (string, error) {
	switch v := entry.(type) {
	case *ConstantEntry:
		return formatConstantResult(v, field, store)
	case *ElementEntry:
		return formatElementResult(v, field, store)
	case *AstronomyEntry:
		return formatAstronomyResult(v, field, store)
	default:
		return "", fmt.Errorf("ErrInternal: unexpected entry type")
	}
}

func formatConstantResult(c *ConstantEntry, field string, store *RegisterStore) (string, error) {
	if field == "unit" {
		val := CalcVal{Kind: ValKindString, Str: c.Unit}
		_, res, err := store.AutoBind(val)
		return res, err
	}
	if field == "uncertainty" {
		unc := c.Uncertainty
		if unc == "" || unc == "0" || unc == "(exact)" {
			unc = "exact"
		}
		val := CalcVal{Kind: ValKindString, Str: unc}
		_, res, err := store.AutoBind(val)
		return res, err
	}

	// Default or field == "value": binds typed register value
	val := CalcVal{
		Kind:     ValKindUnit,
		UnitVal:  c.ValRat,
		UnitName: c.Unit,
		Str:      c.RawValue,
	}
	if c.Unit == "" {
		if c.ValRat != nil && c.ValRat.IsInt() {
			val = CalcVal{Kind: ValKindRat, Rat: c.ValRat}
		} else {
			f, _ := c.ValRat.Float64()
			val = CalcVal{Kind: ValKindFloat, Float: f, Str: c.RawValue}
		}
	}

	_, res, err := store.AutoBind(val)
	return res, err
}

func formatElementResult(e *ElementEntry, field string, store *RegisterStore) (string, error) {
	if field == "" {
		// Bare element returns one-line record excerpt with NO binding (spec §4)
		var parts []string
		parts = append(parts, fmt.Sprintf("%s (%s, Z=%d)", e.Name, e.Symbol, e.AtomicNumber))
		parts = append(parts, fmt.Sprintf("atomic_weight=%s", e.AtomicWeight))
		if e.Group > 0 {
			parts = append(parts, fmt.Sprintf("group=%d", e.Group))
		}
		if e.Period > 0 {
			parts = append(parts, fmt.Sprintf("period=%d", e.Period))
		}
		if e.Density != "" {
			parts = append(parts, fmt.Sprintf("density=%s g/cm3", e.Density))
		}
		if e.MeltingPoint != "" {
			parts = append(parts, fmt.Sprintf("mp=%s K", e.MeltingPoint))
		}
		if e.BoilingPoint != "" {
			parts = append(parts, fmt.Sprintf("bp=%s K", e.BoilingPoint))
		}
		if e.ElectronConfiguration != "" {
			parts = append(parts, fmt.Sprintf("config=%s", e.ElectronConfiguration))
		}
		return strings.Join(parts, ", "), nil
	}

	// Specific field requested -> binds typed value
	switch field {
	case "atomic_weight", "atomic_mass", "weight", "mass":
		rat := parseScientificRat(e.AtomicWeight)
		val := CalcVal{Kind: ValKindRat, Rat: rat, Str: e.AtomicWeight}
		_, res, err := store.AutoBind(val)
		return res, err

	case "atomic_number", "z", "number":
		val := CalcVal{Kind: ValKindRat, Rat: big.NewRat(int64(e.AtomicNumber), 1)}
		_, res, err := store.AutoBind(val)
		return res, err

	case "group":
		val := CalcVal{Kind: ValKindRat, Rat: big.NewRat(int64(e.Group), 1)}
		_, res, err := store.AutoBind(val)
		return res, err

	case "period":
		val := CalcVal{Kind: ValKindRat, Rat: big.NewRat(int64(e.Period), 1)}
		_, res, err := store.AutoBind(val)
		return res, err

	case "density":
		if e.Density == "" {
			return "", fmt.Errorf("ErrNotFound: %s has no known density", e.Name)
		}
		rat := parseScientificRat(e.Density)
		val := CalcVal{Kind: ValKindUnit, UnitVal: rat, UnitName: "g/cm3", Str: e.Density}
		_, res, err := store.AutoBind(val)
		return res, err

	case "melting_point", "mp":
		if e.MeltingPoint == "" {
			return "", fmt.Errorf("ErrNotFound: %s has no known melting point", e.Name)
		}
		rat := parseScientificRat(e.MeltingPoint)
		val := CalcVal{Kind: ValKindUnit, UnitVal: rat, UnitName: "K", Str: e.MeltingPoint}
		_, res, err := store.AutoBind(val)
		return res, err

	case "boiling_point", "bp":
		if e.BoilingPoint == "" {
			return "", fmt.Errorf("ErrNotFound: %s has no known boiling point", e.Name)
		}
		rat := parseScientificRat(e.BoilingPoint)
		val := CalcVal{Kind: ValKindUnit, UnitVal: rat, UnitName: "K", Str: e.BoilingPoint}
		_, res, err := store.AutoBind(val)
		return res, err

	case "electronegativity":
		if e.Electronegativity == "" {
			return "", fmt.Errorf("ErrNotFound: %s has no known electronegativity", e.Name)
		}
		rat := parseScientificRat(e.Electronegativity)
		val := CalcVal{Kind: ValKindRat, Rat: rat, Str: e.Electronegativity}
		_, res, err := store.AutoBind(val)
		return res, err

	case "first_ionization_energy", "ionization_energy", "ie":
		if e.FirstIonizationEnergy == "" {
			return "", fmt.Errorf("ErrNotFound: %s has no known ionization energy", e.Name)
		}
		rat := parseScientificRat(e.FirstIonizationEnergy)
		val := CalcVal{Kind: ValKindUnit, UnitVal: rat, UnitName: "eV", Str: e.FirstIonizationEnergy}
		_, res, err := store.AutoBind(val)
		return res, err

	case "electron_configuration", "config":
		val := CalcVal{Kind: ValKindString, Str: e.ElectronConfiguration}
		_, res, err := store.AutoBind(val)
		return res, err

	case "symbol":
		val := CalcVal{Kind: ValKindString, Str: e.Symbol}
		_, res, err := store.AutoBind(val)
		return res, err

	case "name":
		val := CalcVal{Kind: ValKindString, Str: e.Name}
		_, res, err := store.AutoBind(val)
		return res, err

	default:
		return "", fmt.Errorf("ErrNotFound: unknown field %q for element %s", field, e.Name)
	}
}

func formatAstronomyResult(a *AstronomyEntry, field string, store *RegisterStore) (string, error) {
	if field == "" {
		// Bare astronomical body returns one-line record excerpt with NO binding
		var parts []string
		parts = append(parts, a.ID)
		for _, k := range []string{"mass", "diameter", "radius", "orbital_period", "gravity"} {
			if v, ok := a.Fields[k]; ok && v != "" {
				parts = append(parts, fmt.Sprintf("%s=%s", k, v))
			}
		}
		return strings.Join(parts, ", "), nil
	}

	valStr, ok := a.Fields[field]
	if !ok || valStr == "" {
		// Check alias for field
		switch field {
		case "semi_major_axis":
			valStr = a.Fields["distance_from_sun"]
		case "surface_gravity":
			valStr = a.Fields["gravity"]
		case "gm":
			valStr = a.Fields["GM"]
		}
	}
	if valStr == "" {
		return "", fmt.Errorf("ErrNotFound: unknown field %q for %s", field, a.ID)
	}

	// Parse field value with unit
	fields := strings.Fields(valStr)
	if len(fields) >= 2 {
		numPart := fields[0]
		unitPart := strings.Join(fields[1:], " ")
		rat := parseScientificRat(numPart)
		val := CalcVal{Kind: ValKindUnit, UnitVal: rat, UnitName: unitPart, Str: numPart}
		_, res, err := store.AutoBind(val)
		return res, err
	}

	// Dimensionless number or text
	if rat := parseScientificRat(valStr); rat != nil {
		val := CalcVal{Kind: ValKindRat, Rat: rat, Str: valStr}
		_, res, err := store.AutoBind(val)
		return res, err
	}

	val := CalcVal{Kind: ValKindString, Str: valStr}
	_, res, err := store.AutoBind(val)
	return res, err
}
