package main

import (
	"embed"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

//go:embed profiles/*.json
var embeddedProfilesFS embed.FS

// ModelProfile defines per-model prompt, tokenizer, control-token, and context-anchoring defaults for the Reckoner harness.
type ModelProfile struct {
	Name                    string         `json:"name"`
	DefaultSystem           *string        `json:"default_system"`
	GradedAnswerInstruction *string        `json:"graded_answer_instruction"`
	Anchors                 bool           `json:"anchors"`
	EffortLine              string         `json:"effort_line,omitempty"`
	Tokenizer               string         `json:"tokenizer,omitempty"`
	NewlineTokenID          *int           `json:"newline_token_id,omitempty"`
	ControlTokens           map[string]int `json:"control_tokens,omitempty"`
	Notes                   string         `json:"notes"`
	SourceDir               string         `json:"-"`
	SourcePath              string         `json:"-"`
}

// TokenTable holds the resolved special and control token IDs for prompt rendering,
// output parsing, and in-path/stop-and-continue serving.
type TokenTable struct {
	EndOfText       int
	Pad             int
	TurnUser        int
	TurnAgent       int
	EndTurn         int
	TurnSystem      int
	ToolsOpen       int
	ToolsClose      int
	Deliberate      int
	Act             int
	Ready           int
	Clarify         int
	Pause           int
	CallExternal    int
	EndCall         int
	Obs             int
	ObsErr          int
	EndObs          int
	CallCalc        int
	CallSolve       int
	CallLua         int
	CallStr         int
	CallDate        int
	CallLookup      int
	CallFind        int
	CallTools       int
	CallDef         int
	Quote           int
	EndQuote        int
	Newline         int
	Registers       [16]int
	CallTokenToTool map[int]string
}

// DefaultTokenTable returns the canonical 1.4B (36,864-vocab) token table matching
// the package-level Tok* constants in serve.go byte-for-byte.
func DefaultTokenTable() TokenTable {
	return TokenTable{
		EndOfText:    TokEndOfText,
		Pad:          TokPad,
		TurnUser:     TokTurnUser,
		TurnAgent:    TokTurnAgent,
		EndTurn:      TokEndTurn,
		TurnSystem:   TokTurnSystem,
		ToolsOpen:    TokToolsOpen,
		ToolsClose:   TokToolsClose,
		Deliberate:   TokDeliberate,
		Act:          TokAct,
		Ready:        TokReady,
		Clarify:      TokClarify,
		Pause:        TokPause,
		CallExternal: TokCallExternal,
		EndCall:      TokEndCall,
		Obs:          TokObs,
		ObsErr:       TokObsErr,
		EndObs:       TokEndObs,
		CallCalc:     TokCallCalc,
		CallSolve:    TokCallSolve,
		CallLua:      TokCallLua,
		CallStr:      TokCallStr,
		CallDate:     TokCallDate,
		CallLookup:   TokCallLookup,
		CallFind:     TokCallFind,
		CallTools:    35,
		CallDef:      36,
		Quote:        TokQuote,
		EndQuote:     TokEndQuote,
		Newline:      TokNewline,
		Registers: [16]int{
			21, 22, 23, 24, 25, 26, 27, 28,
			37, 38, 39, 40, 41, 42, 43, 44,
		},
		CallTokenToTool: map[int]string{
			TokCallCalc:     "calc",
			TokCallSolve:    "solve",
			TokCallLua:      "lua",
			TokCallStr:      "str",
			TokCallDate:     "date",
			TokCallLookup:   "lookup",
			TokCallFind:     "find",
			TokCallExternal: "external",
		},
	}
}

var requiredControlTokenNames = []string{
	"<|endoftext|>",
	"<|pad|>",
	"<|turn:user|>",
	"<|turn:agent|>",
	"<|/turn|>",
	"<|turn:system|>",
	"<|tools|>",
	"<|/tools|>",
	"<|deliberate|>",
	"<|act|>",
	"<|ready|>",
	"<|clarify|>",
	"<|pause|>",
	"<|call|>",
	"<|/call|>",
	"<|obs|>",
	"<|obs:err|>",
	"<|/obs|>",
	"<|call:calc|>",
	"<|call:solve|>",
	"<|call:lua|>",
	"<|call:str|>",
	"<|call:date|>",
	"<|call:lookup|>",
	"<|call:find|>",
	"<|call:tools|>",
	"<|call:def|>",
	"<|r0|>", "<|r1|>", "<|r2|>", "<|r3|>",
	"<|r4|>", "<|r5|>", "<|r6|>", "<|r7|>",
	"<|r8|>", "<|r9|>", "<|r10|>", "<|r11|>",
	"<|r12|>", "<|r13|>", "<|r14|>", "<|r15|>",
}

// BuildProfileTokenTable constructs and validates a TokenTable from a ModelProfile.
// For nil profile or a profile using the embedded tokenizer without custom control_tokens
// (such as reckoner-1.4b), returns DefaultTokenTable() unchanged.
// Fails closed with ErrArgs if:
//   - the profile declares an external tokenizer without both control_tokens and newline_token_id;
//   - the profile specifies control_tokens or newline_token_id with an empty or "embedded" tokenizer.
func validateProfileEffortLine(profile *ModelProfile) (string, error) {
	if profile == nil || profile.EffortLine == "" {
		return "off", nil
	}
	switch profile.EffortLine {
	case "off", "none_only", "all":
		return profile.EffortLine, nil
	default:
		return "", fmt.Errorf("ErrArgs: profile %q has invalid effort_line %q (allowed: \"off\", \"none_only\", \"all\")", profile.Name, profile.EffortLine)
	}
}

func BuildProfileTokenTable(profile *ModelProfile) (TokenTable, error) {
	if profile == nil {
		return DefaultTokenTable(), nil
	}
	if _, err := validateProfileEffortLine(profile); err != nil {
		return TokenTable{}, err
	}
	isExt := profileUsesExternalTokenizer(profile)
	if !isExt {
		if len(profile.ControlTokens) > 0 {
			return TokenTable{}, fmt.Errorf("ErrArgs: profile %q specifies control_tokens with %q tokenizer (control_tokens require an external tokenizer)", profile.Name, profile.Tokenizer)
		}
		if profile.NewlineTokenID != nil {
			return TokenTable{}, fmt.Errorf("ErrArgs: profile %q specifies newline_token_id without an external tokenizer and control_tokens", profile.Name)
		}
		return DefaultTokenTable(), nil
	}
	if len(profile.ControlTokens) == 0 || profile.NewlineTokenID == nil {
		return TokenTable{}, fmt.Errorf("ErrArgs: profile %q specifies external tokenizer %q without both control_tokens and newline_token_id", profile.Name, profile.Tokenizer)
	}

	required := append([]string(nil), requiredControlTokenNames...)
	if profile.Anchors {
		required = append(required, "<|quote|>", "<|/quote|>")
	}
	for _, reqName := range required {
		if _, ok := profile.ControlTokens[reqName]; !ok {
			return TokenTable{}, fmt.Errorf("ErrArgs: profile %q is missing required control token %q", profile.Name, reqName)
		}
	}

	seenIDToName := make(map[int]string, len(profile.ControlTokens)+1)
	names := make([]string, 0, len(profile.ControlTokens))
	for k := range profile.ControlTokens {
		names = append(names, k)
	}
	sort.Strings(names)
	for _, name := range names {
		id := profile.ControlTokens[name]
		if id < 0 {
			return TokenTable{}, fmt.Errorf("ErrArgs: profile %q control token %q has negative ID %d", profile.Name, name, id)
		}
		if prev, dup := seenIDToName[id]; dup {
			return TokenTable{}, fmt.Errorf("ErrArgs: profile %q has duplicate control token ID %d for %q and %q", profile.Name, id, prev, name)
		}
		seenIDToName[id] = name
	}

	nlID := *profile.NewlineTokenID
	if nlID < 0 {
		return TokenTable{}, fmt.Errorf("ErrArgs: profile %q has negative newline_token_id %d", profile.Name, nlID)
	}
	if prev, dup := seenIDToName[nlID]; dup {
		return TokenTable{}, fmt.Errorf("ErrArgs: profile %q newline_token_id %d collides with control token %q", profile.Name, nlID, prev)
	}

	quoteID := -1
	if q, ok := profile.ControlTokens["<|quote|>"]; ok {
		quoteID = q
	}
	endQuoteID := -1
	if eq, ok := profile.ControlTokens["<|/quote|>"]; ok {
		endQuoteID = eq
	}

	var regs [16]int
	for r := 0; r < 16; r++ {
		regs[r] = profile.ControlTokens[fmt.Sprintf("<|r%d|>", r)]
	}

	tbl := TokenTable{
		EndOfText:    profile.ControlTokens["<|endoftext|>"],
		Pad:          profile.ControlTokens["<|pad|>"],
		TurnUser:     profile.ControlTokens["<|turn:user|>"],
		TurnAgent:    profile.ControlTokens["<|turn:agent|>"],
		EndTurn:      profile.ControlTokens["<|/turn|>"],
		TurnSystem:   profile.ControlTokens["<|turn:system|>"],
		ToolsOpen:    profile.ControlTokens["<|tools|>"],
		ToolsClose:   profile.ControlTokens["<|/tools|>"],
		Deliberate:   profile.ControlTokens["<|deliberate|>"],
		Act:          profile.ControlTokens["<|act|>"],
		Ready:        profile.ControlTokens["<|ready|>"],
		Clarify:      profile.ControlTokens["<|clarify|>"],
		Pause:        profile.ControlTokens["<|pause|>"],
		CallExternal: profile.ControlTokens["<|call|>"],
		EndCall:      profile.ControlTokens["<|/call|>"],
		Obs:          profile.ControlTokens["<|obs|>"],
		ObsErr:       profile.ControlTokens["<|obs:err|>"],
		EndObs:       profile.ControlTokens["<|/obs|>"],
		CallCalc:     profile.ControlTokens["<|call:calc|>"],
		CallSolve:    profile.ControlTokens["<|call:solve|>"],
		CallLua:      profile.ControlTokens["<|call:lua|>"],
		CallStr:      profile.ControlTokens["<|call:str|>"],
		CallDate:     profile.ControlTokens["<|call:date|>"],
		CallLookup:   profile.ControlTokens["<|call:lookup|>"],
		CallFind:     profile.ControlTokens["<|call:find|>"],
		CallTools:    profile.ControlTokens["<|call:tools|>"],
		CallDef:      profile.ControlTokens["<|call:def|>"],
		Quote:        quoteID,
		EndQuote:     endQuoteID,
		Newline:      nlID,
		Registers:    regs,
	}
	tbl.CallTokenToTool = map[int]string{
		tbl.CallCalc:     "calc",
		tbl.CallSolve:    "solve",
		tbl.CallLua:      "lua",
		tbl.CallStr:      "str",
		tbl.CallDate:     "date",
		tbl.CallLookup:   "lookup",
		tbl.CallFind:     "find",
		tbl.CallExternal: "external",
	}
	return tbl, nil
}

// AllowedAgentTurnTokenIDs returns the set of control token IDs from the profile's TokenTable
// that are permitted inside a pre-formatted assistant/agent turn:
//   - Mode tokens: Deliberate, Act, Ready, Clarify
//   - Tool call delimiters: CallExternal (<|call|>), EndCall (<|/call|>), CallCalc, CallSolve,
//     CallLua, CallStr, CallDate, CallLookup, CallFind, CallTools, CallDef
//   - Observation delimiters: Obs (<|obs|>), ObsErr (<|obs:err|>), EndObs (<|/obs|>)
//   - Register tokens: Registers[0..15] (<|r0|>..<|r15|>)
//   - Quote delimiters (only when enabled on the profile, i.e. Quote >= 0 and EndQuote >= 0):
//     Quote (<|quote|>), EndQuote (<|/quote|>)
func (tbl TokenTable) AllowedAgentTurnTokenIDs() map[int]bool {
	allowed := map[int]bool{
		tbl.Deliberate:   true,
		tbl.Act:          true,
		tbl.Ready:        true,
		tbl.Clarify:      true,
		tbl.CallExternal: true,
		tbl.EndCall:      true,
		tbl.Obs:          true,
		tbl.ObsErr:       true,
		tbl.EndObs:       true,
		tbl.CallCalc:     true,
		tbl.CallSolve:    true,
		tbl.CallLua:      true,
		tbl.CallStr:      true,
		tbl.CallDate:     true,
		tbl.CallLookup:   true,
		tbl.CallFind:     true,
		tbl.CallTools:    true,
		tbl.CallDef:      true,
	}
	for _, rID := range tbl.Registers {
		if rID >= 0 {
			allowed[rID] = true
		}
	}
	if tbl.Quote >= 0 {
		allowed[tbl.Quote] = true
	}
	if tbl.EndQuote >= 0 {
		allowed[tbl.EndQuote] = true
	}
	if tbl.Quote >= 0 && tbl.EndQuote >= 0 {
		for id := M0TokenID; id <= L1023TokenID; id++ {
			allowed[id] = true
		}
	}
	return allowed
}

// ForbiddenAgentTurnTokenIDs derives the map of token IDs (to token names) from the profile's
// ControlTokens (TokenTable) and tok.AddedTokens that are forbidden inside a pre-formatted
// assistant/agent turn:
//   - Turn, system, tools, pad, pause, and end-of-text control tokens:
//     EndOfText (<|endoftext|>), Pad (<|pad|>), TurnUser (<|turn:user|>), TurnAgent (<|turn:agent|>),
//     EndTurn (<|/turn|>), TurnSystem (<|turn:system|>), ToolsOpen (<|tools|>), ToolsClose (<|/tools|>),
//     Pause (<|pause|>)
//   - Every token in tok.AddedTokens whose ID is not in tbl.AllowedAgentTurnTokenIDs()
//     (including all Qwen special/added tokens 248044..248069).
func (tbl TokenTable) ForbiddenAgentTurnTokenIDs(tok *Tokenizer) map[int]string {
	allowed := tbl.AllowedAgentTurnTokenIDs()
	forbidden := map[int]string{
		tbl.EndOfText:  "<|endoftext|>",
		tbl.Pad:        "<|pad|>",
		tbl.TurnUser:   "<|turn:user|>",
		tbl.TurnAgent:  "<|turn:agent|>",
		tbl.EndTurn:    "<|/turn|>",
		tbl.TurnSystem: "<|turn:system|>",
		tbl.ToolsOpen:  "<|tools|>",
		tbl.ToolsClose: "<|/tools|>",
		tbl.Pause:      "<|pause|>",
	}
	if tok != nil {
		for name, id := range tok.AddedTokens {
			if !allowed[id] {
				forbidden[id] = name
			}
		}
	}
	return forbidden
}

// ValidateAndEncodePreformattedAssistantContent validates and encodes a pre-formatted assistant turn
// (whose content begins with <|ready|>, <|act|>, <|deliberate|>, or <|clarify|>) on an external-tokenizer profile,
// rejecting any forbidden control token, disabled quote token, or Qwen special token with ErrArgs.
func ValidateAndEncodePreformattedAssistantContent(tok *Tokenizer, tbl TokenTable, content string) ([]int, error) {
	if (tbl.Quote < 0 && strings.Contains(content, "<|quote|>")) ||
		(tbl.EndQuote < 0 && strings.Contains(content, "<|/quote|>")) {
		return nil, fmt.Errorf("ErrArgs: pre-formatted assistant content contains quote token (<|quote|> or <|/quote|>) which is disabled on this profile")
	}
	ids := tok.Encode(content)
	forbidden := tbl.ForbiddenAgentTurnTokenIDs(tok)
	for _, id := range ids {
		if name, bad := forbidden[id]; bad {
			return nil, fmt.Errorf("ErrArgs: pre-formatted assistant content contains forbidden token %q (ID %d)", name, id)
		}
	}
	return ids, nil
}

// ValidateTokenizerWithProfile verifies that every required control token and newline token
// exists in tok with its exact expected ID, that all IDs are distinct, and that decode(encode(tokStr))
// round-trips back to tokStr as a single token. Never returns nil vacuously for an external profile.
func ValidateTokenizerWithProfile(tok *Tokenizer, profile *ModelProfile, table TokenTable) error {
	if tok == nil {
		return fmt.Errorf("ErrArgs: nil tokenizer")
	}
	if profile == nil {
		return nil
	}
	if _, err := BuildProfileTokenTable(profile); err != nil {
		return err
	}
	if !profileUsesExternalTokenizer(profile) {
		return nil
	}

	names := make([]string, 0, len(profile.ControlTokens))
	for k := range profile.ControlTokens {
		names = append(names, k)
	}
	sort.Strings(names)

	for _, name := range names {
		wantID := profile.ControlTokens[name]
		gotID := tok.TokenToID(name)
		if gotID != wantID {
			return fmt.Errorf("ErrArgs: tokenizer control token %q has ID %d in vocab, profile %q expects %d", name, gotID, profile.Name, wantID)
		}
		if addID, inAdded := tok.AddedTokens[name]; !inAdded || addID != wantID {
			return fmt.Errorf("ErrArgs: tokenizer control token %q missing from added_tokens or ID mismatch (got %d, want %d)", name, addID, wantID)
		}
		if invStr := tok.IDToToken(wantID); invStr != name {
			return fmt.Errorf("ErrArgs: tokenizer ID %d maps to %q, expected %q", wantID, invStr, name)
		}
		enc := tok.Encode(name)
		if len(enc) != 1 || enc[0] != wantID {
			return fmt.Errorf("ErrArgs: control token %q encode round-trip produced %v, expected [%d]", name, enc, wantID)
		}
		dec := tok.DecodeTokens(enc)
		if dec != name {
			return fmt.Errorf("ErrArgs: control token %q decode(encode) round-trip produced %q, expected %q", name, dec, name)
		}
	}

	nlEnc := tok.Encode("\n")
	if len(nlEnc) != 1 || nlEnc[0] != table.Newline {
		return fmt.Errorf("ErrArgs: newline encode round-trip produced %v, profile %q expects [%d]", nlEnc, profile.Name, table.Newline)
	}
	if nlDec := tok.DecodeTokens(nlEnc); nlDec != "\n" {
		return fmt.Errorf("ErrArgs: newline decode(encode) round-trip produced %q, expected \"\\n\"", nlDec)
	}

	return nil
}

func profileUsesExternalTokenizer(profile *ModelProfile) bool {
	if profile == nil {
		return false
	}
	tokRef := strings.TrimSpace(profile.Tokenizer)
	return tokRef != "" && tokRef != "embedded"
}

func resolveFileOrDirWithTokenizer(candidatePath string, tokFileName string) string {
	candidatePath = strings.TrimSpace(candidatePath)
	if candidatePath == "" {
		return ""
	}
	if info, err := os.Stat(candidatePath); err == nil && info.IsDir() {
		return filepath.Join(candidatePath, tokFileName)
	}
	return candidatePath
}

// ResolveProfileTokenizerPath resolves the disk path to a profile's external tokenizer.json
// only when the profile declares an external tokenizer (profile.Tokenizer != "" && != "embedded").
// Resolution order:
//  1. Explicit --tokenizer PATH (accepted as a file or a directory joined with profile.Tokenizer)
//  2. RECKONER_TOKENIZER_PATH environment variable (accepted as a file or a directory joined with profile.Tokenizer)
//  3. Absolute profile.Tokenizer path if profile.Tokenizer is absolute
//  4. Model directory when known (only if the target tokenizer file exists in that directory)
//  5. Directory of a disk profile file (profile.SourceDir, resolved to an absolute path at profile load time,
//     only if the target tokenizer file exists in that directory)
//
// It never resolves a relative profile.Tokenizer against the current working directory,
// and returns "" when profile is nil or uses the embedded 1.4B tokenizer.
func ResolveProfileTokenizerPath(profile *ModelProfile, explicitPath string, modelDirs ...string) string {
	if !profileUsesExternalTokenizer(profile) {
		return ""
	}
	tokRef := strings.TrimSpace(profile.Tokenizer)
	tokBase := filepath.Base(tokRef)

	if strings.TrimSpace(explicitPath) != "" {
		return resolveFileOrDirWithTokenizer(explicitPath, tokBase)
	}
	if envPath := strings.TrimSpace(os.Getenv("RECKONER_TOKENIZER_PATH")); envPath != "" {
		return resolveFileOrDirWithTokenizer(envPath, tokBase)
	}
	if filepath.IsAbs(tokRef) {
		return tokRef
	}
	for _, md := range modelDirs {
		md = strings.TrimSpace(md)
		if md == "" {
			continue
		}
		dir := md
		if info, err := os.Stat(md); err == nil && !info.IsDir() {
			dir = filepath.Dir(md)
		}
		cand := filepath.Join(dir, tokRef)
		if info, err := os.Stat(cand); err == nil && !info.IsDir() {
			return cand
		}
	}
	if profile.SourceDir != "" {
		cand := filepath.Join(profile.SourceDir, tokRef)
		if info, err := os.Stat(cand); err == nil && !info.IsDir() {
			return cand
		}
	}
	return ""
}

// LoadTokenizerForProfile loads and validates the tokenizer and TokenTable for the given profile
// and optional explicit --tokenizer path override and model directory.
//   - When profile is nil or uses the embedded 1.4B tokenizer:
//   - Rejects any non-empty explicitPath with ErrArgs.
//   - Ignores RECKONER_TOKENIZER_PATH and returns GetGlobalTokenizer() and DefaultTokenTable() with zero disk I/O.
//   - When profile declares an external tokenizer (e.g. reckoner-q2b):
//   - Resolves the tokenizer path via ResolveProfileTokenizerPath.
//   - Caches the validated tokenizer per profile (at most one entry per profile, invalidated on replacement)
//     only AFTER ValidateTokenizerWithProfile succeeds.
func LoadTokenizerForProfile(profile *ModelProfile, explicitPath string, modelDirs ...string) (*Tokenizer, TokenTable, error) {
	tbl, err := BuildProfileTokenTable(profile)
	if err != nil {
		return nil, TokenTable{}, err
	}

	if !profileUsesExternalTokenizer(profile) {
		if strings.TrimSpace(explicitPath) != "" {
			profName := "<none>"
			if profile != nil && profile.Name != "" {
				profName = profile.Name
			}
			return nil, TokenTable{}, fmt.Errorf(
				"ErrArgs: --tokenizer cannot be used with profile %q (which uses the embedded 36,864-vocab 1.4B tokenizer)",
				profName,
			)
		}
		tok, gErr := GetGlobalTokenizer()
		if gErr != nil {
			return nil, TokenTable{}, gErr
		}
		return tok, tbl, nil
	}

	tokPath := ResolveProfileTokenizerPath(profile, explicitPath, modelDirs...)
	if tokPath == "" {
		return nil, TokenTable{}, fmt.Errorf(
			"ErrArgs: profile %q requires external tokenizer %q, but no tokenizer path could be resolved; pass --tokenizer PATH, set RECKONER_TOKENIZER_PATH, pass --model-dir DIR, or load a profile file from disk",
			profile.Name, profile.Tokenizer,
		)
	}

	absTokPath := tokPath
	if abs, absErr := filepath.Abs(tokPath); absErr == nil {
		absTokPath = abs
	}

	info, statErr := os.Stat(absTokPath)
	if statErr != nil {
		return nil, TokenTable{}, fmt.Errorf("ErrArgs: failed to load tokenizer for profile %q from %q (override with --tokenizer PATH): %w", profile.Name, tokPath, statErr)
	}
	if info.IsDir() {
		return nil, TokenTable{}, fmt.Errorf("ErrArgs: tokenizer path %q for profile %q is a directory", tokPath, profile.Name)
	}

	cacheKey := profile.Name
	mtimeNano := info.ModTime().UnixNano()
	fSize := info.Size()

	tokenizerFileCacheMu.Lock()
	if cached, ok := tokenizerFileCache[cacheKey]; ok &&
		cached.path == absTokPath &&
		cached.mtimeNano == mtimeNano &&
		cached.size == fSize &&
		cached.tok != nil {
		tokCopy := *cached.tok
		tokenizerFileCacheMu.Unlock()
		if err := ValidateTokenizerWithProfile(&tokCopy, profile, tbl); err != nil {
			return nil, TokenTable{}, err
		}
		tblCopy := tbl
		tokCopy.Table = &tblCopy
		tokCopy.UsesExternalTokenizer = true
		return &tokCopy, tbl, nil
	}
	tokenizerFileCacheMu.Unlock()

	parsed, err := LoadTokenizerFromFile(absTokPath)
	if err != nil {
		return nil, TokenTable{}, fmt.Errorf("ErrArgs: failed to load tokenizer for profile %q from %q (override with --tokenizer PATH): %w", profile.Name, tokPath, err)
	}
	if err := ValidateTokenizerWithProfile(parsed, profile, tbl); err != nil {
		return nil, TokenTable{}, err
	}

	tokenizerFileCacheMu.Lock()
	tokenizerFileCache[cacheKey] = tokenizerCacheEntry{
		path:      absTokPath,
		mtimeNano: mtimeNano,
		size:      fSize,
		tok:       parsed,
	}
	tokenizerFileCacheMu.Unlock()

	tokCopy := *parsed
	tblCopy := tbl
	tokCopy.Table = &tblCopy
	tokCopy.UsesExternalTokenizer = true
	return &tokCopy, tbl, nil
}

// LoadProfile loads a ModelProfile by built-in name (e.g. "reckoner-1.4b", "reckoner-q2b")
// or from a JSON file path on disk. Returns (nil, nil) if nameOrPath is empty.
func LoadProfile(nameOrPath string) (*ModelProfile, error) {
	spec := strings.TrimSpace(nameOrPath)
	if spec == "" {
		return nil, nil
	}

	var data []byte
	var readErr error
	var sourceDir string
	var sourcePath string

	// 1. Check if spec points to an existing file on disk.
	if info, err := os.Stat(spec); err == nil && !info.IsDir() {
		data, readErr = os.ReadFile(spec)
		if readErr != nil {
			return nil, fmt.Errorf("ErrArgs: failed to read profile file %q: %w", spec, readErr)
		}
		if absSpec, absErr := filepath.Abs(spec); absErr == nil {
			sourceDir = filepath.Dir(absSpec)
			sourcePath = absSpec
		} else {
			sourceDir = filepath.Dir(spec)
			sourcePath = spec
		}
	} else {
		// 2. Check embedded profiles/<name>.json.
		base := strings.TrimPrefix(spec, "profiles/")
		base = strings.TrimSuffix(base, ".json")
		if base == "" || strings.ContainsAny(base, `/\`) {
			return nil, fmt.Errorf("ErrArgs: unknown profile or missing file %q", spec)
		}
		embPath := "profiles/" + base + ".json"
		data, readErr = embeddedProfilesFS.ReadFile(embPath)
		if readErr != nil {
			// Also check relative to profiles/ on disk if not embedded.
			diskFallback := filepath.Join("profiles", base+".json")
			if d2, err2 := os.ReadFile(diskFallback); err2 == nil {
				data = d2
				if absFallback, absErr := filepath.Abs(diskFallback); absErr == nil {
					sourceDir = filepath.Dir(absFallback)
					sourcePath = absFallback
				} else {
					sourceDir = "profiles"
					sourcePath = diskFallback
				}
			} else {
				return nil, fmt.Errorf("ErrArgs: unknown profile %q", spec)
			}
		}
	}

	var prof ModelProfile
	if err := json.Unmarshal(data, &prof); err != nil {
		return nil, fmt.Errorf("ErrSyntax: invalid profile JSON in %q: %w", spec, err)
	}
	if strings.TrimSpace(prof.Name) == "" {
		return nil, fmt.Errorf("ErrArgs: profile %q is missing required 'name' field", spec)
	}
	if prof.EffortLine == "" {
		prof.EffortLine = "off"
	}
	prof.SourceDir = sourceDir
	prof.SourcePath = sourcePath
	if _, err := BuildProfileTokenTable(&prof); err != nil {
		return nil, err
	}
	return &prof, nil
}

// ListBuiltinProfiles returns the sorted names of all embedded built-in model profiles.
func ListBuiltinProfiles() ([]string, error) {
	entries, err := embeddedProfilesFS.ReadDir("profiles")
	if err != nil {
		return nil, err
	}
	var names []string
	for _, e := range entries {
		if !e.IsDir() && strings.HasSuffix(e.Name(), ".json") {
			names = append(names, strings.TrimSuffix(e.Name(), ".json"))
		}
	}
	sort.Strings(names)
	return names, nil
}

// ResolveEffectiveAnchors determines whether context anchors (<|mN|>, <|Lk|>) are enabled.
//   - With no profile (profile == nil), defaults to true (the original default) and explicitAnchors wins over noAnchorsOverride.
//   - With a profile, effective_anchors = profile.Anchors && !noAnchorsOverride && (explicitAnchors == nil || *explicitAnchors).
//     A profile with Anchors == false (e.g. reckoner-q2b) always returns false even if explicitAnchors is true.
func ResolveEffectiveAnchors(profile *ModelProfile, noAnchorsOverride bool, explicitAnchors *bool) bool {
	if profile != nil {
		return profile.Anchors && !noAnchorsOverride && (explicitAnchors == nil || *explicitAnchors)
	}
	anchors := !noAnchorsOverride
	if explicitAnchors != nil {
		anchors = *explicitAnchors
	}
	return anchors
}

// ApplyProfileToMessages applies a ModelProfile's default_system and graded_answer_instruction
// to a list of OpenAIMessage turns without mutating the input slice:
//  1. When no system turn is present in messages and profile.DefaultSystem is non-null,
//     inserts {Role: "system", Content: *profile.DefaultSystem} at index 0.
//  2. When graded is true and profile.GradedAnswerInstruction is non-null,
//     appends "\n\n" + *profile.GradedAnswerInstruction to the last user turn.
func ApplyProfileToMessages(messages []OpenAIMessage, profile *ModelProfile, graded bool) []OpenAIMessage {
	out := make([]OpenAIMessage, len(messages))
	copy(out, messages)
	if profile == nil {
		return out
	}

	hasSystem := false
	for _, msg := range out {
		if strings.EqualFold(strings.TrimSpace(msg.Role), "system") {
			hasSystem = true
			break
		}
	}

	if !hasSystem && profile.DefaultSystem != nil && *profile.DefaultSystem != "" {
		sysMsg := OpenAIMessage{
			Role:    "system",
			Content: *profile.DefaultSystem,
		}
		out = append([]OpenAIMessage{sysMsg}, out...)
	}

	if graded && profile.GradedAnswerInstruction != nil && *profile.GradedAnswerInstruction != "" {
		lastUserIdx := -1
		for i := len(out) - 1; i >= 0; i-- {
			if strings.EqualFold(strings.TrimSpace(out[i].Role), "user") {
				lastUserIdx = i
				break
			}
		}
		if lastUserIdx >= 0 {
			out[lastUserIdx].Content = out[lastUserIdx].Content + "\n\n" + *profile.GradedAnswerInstruction
		}
	}

	return out
}

// ApplyEffortLineToMessages applies the profile's effort_line rule ("off", "none_only", or "all")
// after ApplyProfileToMessages without mutating the input slice:
//  1. If effort_line == "off", or effort_line == "none_only" and the effective effort is not none/minimal: do nothing.
//  2. Otherwise let line = "reasoning_effort: " + name (where minimal -> none, aliases -> target, empty -> medium).
//  3. Find the first role == "system" message. If none exists, insert {Role: "system", Content: line} at index 0 and stop.
//  4. If the first system message's content already begins with "reasoning_effort:": leave it unchanged.
//  5. If the first system message's content is empty: set it to line (no trailing "\n").
//  6. If the first system message's content begins with "<|tools|>": set it to line + "\n" + content.
//  7. Otherwise: set it to line + "\n" + content.
//  8. Leave every other role == "system" message unchanged.
func ApplyEffortLineToMessages(messages []OpenAIMessage, profile *ModelProfile, rawEffort string) ([]OpenAIMessage, error) {
	mode, err := validateProfileEffortLine(profile)
	if err != nil {
		return nil, err
	}
	canonEffort, _, _, err := resolveEffortOrErr(rawEffort)
	if err != nil {
		return nil, err
	}
	promptName := canonEffort
	if promptName == "minimal" {
		promptName = "none"
	}

	out := make([]OpenAIMessage, len(messages))
	copy(out, messages)

	if mode == "off" || (mode == "none_only" && promptName != "none") {
		return out, nil
	}

	line := "reasoning_effort: " + promptName

	firstSysIdx := -1
	for i, msg := range out {
		if strings.EqualFold(strings.TrimSpace(msg.Role), "system") {
			firstSysIdx = i
			break
		}
	}
	if firstSysIdx < 0 {
		sysMsg := OpenAIMessage{
			Role:    "system",
			Content: line,
		}
		return append([]OpenAIMessage{sysMsg}, out...), nil
	}

	content := out[firstSysIdx].Content
	if strings.HasPrefix(content, "reasoning_effort:") {
		return out, nil
	}
	if content == "" {
		out[firstSysIdx].Content = line
		return out, nil
	}
	out[firstSysIdx].Content = line + "\n" + content
	out[firstSysIdx].harnessEffortPrefix = true
	return out, nil
}

// RenderChatPromptWithProfile applies profile rules (default_system, graded_answer_instruction, effort_line, and anchors)
// and renders the resulting messages into canonical Reckoner token IDs using the tokenizer's TokenTable.
// The profile's anchors setting may only switch anchors off (never turn on anchors a no-profile request wouldn't get),
// and rendered.ContextStore keeps the prompt's anchor state.
func RenderChatPromptWithProfile(
	tok *Tokenizer,
	messages []OpenAIMessage,
	toolsJSON json.RawMessage,
	anchorContext bool,
	profile *ModelProfile,
	graded bool,
	noAnchorsOverride bool,
) (*RenderedPrompt, []OpenAIMessage, error) {
	return RenderChatPromptWithProfileAndEffort(tok, messages, toolsJSON, anchorContext, profile, graded, noAnchorsOverride, "")
}

// RenderChatPromptWithProfileAndEffort applies profile rules (default_system, graded_answer_instruction,
// effort_line with explicit reasoningEffort, and anchors) and renders the resulting messages into canonical
// Reckoner token IDs using the tokenizer's TokenTable.
func RenderChatPromptWithProfileAndEffort(
	tok *Tokenizer,
	messages []OpenAIMessage,
	toolsJSON json.RawMessage,
	anchorContext bool,
	profile *ModelProfile,
	graded bool,
	noAnchorsOverride bool,
	reasoningEffort string,
) (*RenderedPrompt, []OpenAIMessage, error) {
	if _, err := validateProfileEffortLine(profile); err != nil {
		return nil, nil, err
	}
	applied := ApplyProfileToMessages(messages, profile, graded)
	var err error
	applied, err = ApplyEffortLineToMessages(applied, profile, reasoningEffort)
	if err != nil {
		return nil, nil, err
	}
	promptAnchorCtx := anchorContext
	if profile != nil && (!profile.Anchors || noAnchorsOverride) {
		promptAnchorCtx = false
	}
	table := tok.TokenTable()
	renderTok := tok
	ext := profileUsesExternalTokenizer(profile)
	if tok != nil && tok.UsesExternalTokenizer != ext {
		tokCopy := *tok
		tokCopy.UsesExternalTokenizer = ext
		renderTok = &tokCopy
	}
	rendered, err := RenderChatPromptWithTable(renderTok, table, applied, toolsJSON, promptAnchorCtx)
	if err != nil {
		return nil, nil, err
	}
	return rendered, applied, nil
}
