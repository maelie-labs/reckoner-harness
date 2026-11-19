package main

import (
	"bytes"
	"compress/gzip"
	_ "embed"
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
	"unicode"
	"unicode/utf8"

	"golang.org/x/text/unicode/norm"
)

//go:embed tokenizer_data/tokenizer.json.gz
var defaultTokenizerJSONGz []byte

const (
	NumMessageTokens = 256
	NumLineAnchors   = 1024

	M0TokenID    = 32896
	M255TokenID  = 33151
	L0TokenID    = 33152
	L1023TokenID = 34175

	IgnoreIndex = -100

	VocabSize = 36864

	// MaxVocabID caps the maximum token ID allowed when allocating InvVocab
	// to prevent unbounded memory allocation on malformed tokenizer files.
	MaxVocabID = 4_000_000

	// Qwen35SplitRegex is the exact 7-branch ordered PCRE alternation required by Qwen3.5's Split pre-tokenizer.
	Qwen35SplitRegex = `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
)

type bpePair struct {
	a, b string
}

// Tokenizer wraps vocabulary mapping, newline/whitespace classification,
// and byte-level BPE encoding and decoding for context anchoring and quote operations.
type Tokenizer struct {
	Vocab                 map[string]int
	InvVocab              []string
	AddedTokens           map[string]int
	AddedByFirstByte      [256][]string
	NonAngleAddedTokens   []string
	MergeRanks            map[bpePair]int
	NewlineTokens         []bool
	WhitespaceTokens      []bool
	RuneToByte            [324]byte
	ValidRune             [324]bool
	ByteToRune            [256]rune
	IsQwenPreTokenizer    bool
	UsesExternalTokenizer bool
	UseNFC                bool
	HasAnchors            bool
	Table                 *TokenTable
}

// TokenTable returns the tokenizer's bound control-token table, or DefaultTokenTable() for 1.4B.
func (tok *Tokenizer) TokenTable() TokenTable {
	if tok != nil && tok.Table != nil {
		return *tok.Table
	}
	return DefaultTokenTable()
}

// escapeNonPipeAngleAddedTokens escapes registered non-"<|...|>" added_tokens
// starting with "<" (such as Qwen3.5's "<think>", "</think>", "<tool_call>",
// "</tool_call>", "<tool_response>", "</tool_response>") by replacing their
// leading "<" with "<\", without applying a second "<|" -> "<\|" replacement
// (preserving the harness's own "<|rN|> =" prefix in tool observations).
func (tok *Tokenizer) escapeNonPipeAngleAddedTokens(s string) string {
	if tok != nil && len(tok.NonAngleAddedTokens) > 0 && strings.IndexByte(s, '<') >= 0 {
		for _, tag := range tok.NonAngleAddedTokens {
			if len(tag) > 1 && tag[0] == '<' && strings.Contains(s, tag) {
				s = strings.ReplaceAll(s, tag, `<\`+tag[1:])
			}
		}
	}
	return s
}

// escapeUserSystemTokens escapes "<|" to "<\|" and, when the tokenizer has non-"<|...|>"
// added_tokens starting with "<" (such as Qwen3.5's "<think>", "</think>", "<tool_call>",
// "</tool_call>", "<tool_response>", "</tool_response>"), escapes their leading "<" to "<\"
// so untrusted user/system text cannot inject special/added token IDs.
// Because escapeControlTokens replaces "<|" with "<\|" (where the character after "<" is "\"
// rather than "|"), escapeUserSystemTokens is strictly idempotent: calling it on a string whose
// "<|" sequences were already escaped by value formatting never double-escapes to "<\\|".
func (tok *Tokenizer) escapeUserSystemTokens(s string) string {
	return tok.escapeNonPipeAngleAddedTokens(escapeControlTokens(s))
}

// splitLeadingRegisterPrefix splits a harness register-binding prefix ("<|r0|> = " .. "<|r15|> = ")
// from the tool-produced body if present.
func splitLeadingRegisterPrefix(s string) (prefix string, body string, ok bool) {
	if !strings.HasPrefix(s, "<|r") {
		return "", s, false
	}
	endIdx := strings.Index(s, "|> = ")
	if endIdx < 4 || endIdx > 5 {
		return "", s, false
	}
	numStr := s[3:endIdx]
	n, err := strconv.Atoi(numStr)
	if err != nil || n < 0 || n >= NumRegisters {
		return "", s, false
	}
	cut := endIdx + len("|> = ")
	return s[:cut], s[cut:], true
}

// escapeSolveSuccessBody escapes a non-error solve result while preserving the harness's own
// register annotations ("(<|r0|>)" .. "(<|r15|>)" or "(<|r0|>, replaced ...)").
func (tok *Tokenizer) escapeSolveSuccessBody(s string) string {
	var b strings.Builder
	rem := s
	for {
		idx := strings.Index(rem, "(<|r")
		if idx < 0 {
			b.WriteString(tok.escapeUserSystemTokens(rem))
			break
		}
		after := rem[idx+4:]
		closeIdx := strings.Index(after, "|>")
		if closeIdx >= 1 && closeIdx <= 2 {
			if n, err := strconv.Atoi(after[:closeIdx]); err == nil && n >= 0 && n < NumRegisters {
				tail := after[closeIdx+2:]
				if strings.HasPrefix(tail, ")") || strings.HasPrefix(tail, ", replaced ") {
					b.WriteString(tok.escapeUserSystemTokens(rem[:idx]))
					b.WriteString(rem[idx : idx+4+closeIdx+2])
					rem = tail
					continue
				}
			}
		}
		b.WriteString(tok.escapeUserSystemTokens(rem[:idx+4]))
		rem = after
	}
	return b.String()
}

// sanitize14BFindTokens preserves normal 1.4B BPE text tokens (128..32895) and context anchor tokens
// (M0TokenID..L1023TokenID, 32896..34175) in rawFindTokens while escaping any non-anchor control or
// reserved token ID (< 128 or > L1023TokenID, such as <|tools|> or <|/tools|> from the system turn).
// When no such token ID is present, it returns rawFindTokens unchanged.
func sanitize14BFindTokens(tok *Tokenizer, rawFindTokens []int) []int {
	if tok == nil || len(rawFindTokens) == 0 {
		return rawFindTokens
	}
	needsSanitize := false
	for _, id := range rawFindTokens {
		if id < 128 || id > L1023TokenID {
			needsSanitize = true
			break
		}
	}
	if !needsSanitize {
		return rawFindTokens
	}
	out := make([]int, 0, len(rawFindTokens)+8)
	for _, id := range rawFindTokens {
		if id < 128 || id > L1023TokenID {
			out = append(out, tok.Encode(tok.escapeUserSystemTokens(tok.IDToToken(id)))...)
		} else {
			out = append(out, id)
		}
	}
	return out
}

// EncodeToolObservation is the single choke point for encoding in-path tool observations into token IDs
// across both 1.4B / no-profile and external-tokenizer profiles (such as reckoner-q2b).
// Every tool observation (calc, solve, str, date, lookup, find, lines, quote, lua, reg, or client tool,
// whether success or error) passes through this function: the harness's own structure (<|obs|> / <|obs:err|>,
// the <|rN|> = register prefix, and <|/obs|>) is emitted as fixed template tokens, while the entire
// tool-produced body is escaped exactly once for both "<|" and registered angle-bracket added_tokens.
// Whenever no tool body contains "<|" or a registered angle token, 1.4B output is byte-identical to the no-escaping baseline.
func EncodeToolObservation(tok *Tokenizer, table TokenTable, toolName string, resultStr string, rawFindTokens []int, isErr bool) []int {
	if tok == nil {
		if !isErr && (toolName == "find" || toolName == "lines") {
			obsIDs := make([]int, 0, len(rawFindTokens)+2)
			obsIDs = append(obsIDs, table.Obs)
			obsIDs = append(obsIDs, rawFindTokens...)
			obsIDs = append(obsIDs, table.EndObs)
			return obsIDs
		}
		return nil
	}

	if isErr {
		escapedErr := tok.escapeUserSystemTokens(resultStr)
		return tok.Encode(fmt.Sprintf("<|obs:err|>%s<|/obs|>", escapedErr))
	}

	switch toolName {
	case "calc", "str", "date", "lua", "lookup", "reg":
		if regPrefix, body, ok := splitLeadingRegisterPrefix(resultStr); ok {
			escapedBody := tok.escapeUserSystemTokens(body)
			return tok.Encode(fmt.Sprintf("<|obs|>%s%s<|/obs|>", regPrefix, escapedBody))
		}
		escapedBody := tok.escapeUserSystemTokens(resultStr)
		return tok.Encode(fmt.Sprintf("<|obs|>%s<|/obs|>", escapedBody))
	case "solve":
		if regPrefix, body, ok := splitLeadingRegisterPrefix(resultStr); ok {
			escapedBody := tok.escapeUserSystemTokens(body)
			return tok.Encode(fmt.Sprintf("<|obs|>%s%s<|/obs|>", regPrefix, escapedBody))
		}
		escapedBody := tok.escapeSolveSuccessBody(resultStr)
		return tok.Encode(fmt.Sprintf("<|obs|>%s<|/obs|>", escapedBody))
	case "quote":
		body := resultStr
		if strings.HasPrefix(body, "<|obs|>") && strings.HasSuffix(body, "<|/obs|>") {
			body = body[len("<|obs|>") : len(body)-len("<|/obs|>")]
		}
		escapedBody := tok.escapeUserSystemTokens(body)
		return tok.Encode(fmt.Sprintf("<|obs|>%s<|/obs|>", escapedBody))
	case "find", "lines":
		if !tok.UsesExternalTokenizer {
			if len(rawFindTokens) == 0 && resultStr != "" && resultStr != "no matches" {
				escapedBody := tok.escapeUserSystemTokens(resultStr)
				return tok.Encode(fmt.Sprintf("<|obs|>%s<|/obs|>", escapedBody))
			}
			cleanTokens := sanitize14BFindTokens(tok, rawFindTokens)
			obsIDs := make([]int, 0, len(cleanTokens)+2)
			obsIDs = append(obsIDs, table.Obs)
			obsIDs = append(obsIDs, cleanTokens...)
			obsIDs = append(obsIDs, table.EndObs)
			return obsIDs
		}
		escapedBody := tok.escapeUserSystemTokens(resultStr)
		return tok.Encode(fmt.Sprintf("<|obs|>%s<|/obs|>", escapedBody))
	default:
		escapedBody := tok.escapeUserSystemTokens(resultStr)
		return tok.Encode(fmt.Sprintf("<|obs|>%s<|/obs|>", escapedBody))
	}
}

type tokenizerCacheEntry struct {
	path      string
	mtimeNano int64
	size      int64
	tok       *Tokenizer
}

var (
	globalTokenizer      *Tokenizer
	globalTokenizerOnce  sync.Once
	globalTokenizerErr   error
	tokenizerFileCacheMu sync.Mutex
	// Keyed by profile name (at most one validated entry per profile; invalidated on replacement).
	tokenizerFileCache = make(map[string]tokenizerCacheEntry)
)

// GetGlobalTokenizer returns the lazily initialized singleton tokenizer.
func GetGlobalTokenizer() (*Tokenizer, error) {
	globalTokenizerOnce.Do(func() {
		globalTokenizer, globalTokenizerErr = LoadTokenizerFromGz(defaultTokenizerJSONGz)
	})
	return globalTokenizer, globalTokenizerErr
}

// LoadTokenizerFromFile reads and parses an uncompressed or .gz tokenizer.json file without caching.
// Validated profile tokenizers are cached per-profile by LoadTokenizerForProfile after validation succeeds.
func LoadTokenizerFromFile(path string) (*Tokenizer, error) {
	info, statErr := os.Stat(path)
	if statErr != nil {
		return nil, fmt.Errorf("ErrArgs: failed to read tokenizer file %q: %w", path, statErr)
	}
	if info.IsDir() {
		return nil, fmt.Errorf("ErrArgs: tokenizer path %q is a directory", path)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("ErrArgs: failed to read tokenizer file %q: %w", path, err)
	}
	if strings.HasSuffix(path, ".gz") {
		return LoadTokenizerFromGz(data)
	}
	return ParseTokenizerJSON(data)
}

// LoadTokenizerFromGz decompresses and parses a gzipped tokenizer.json.
func LoadTokenizerFromGz(gzData []byte) (*Tokenizer, error) {
	gz, err := gzip.NewReader(bytes.NewReader(gzData))
	if err != nil {
		return nil, fmt.Errorf("failed to create gzip reader: %w", err)
	}
	defer gz.Close()

	var buf bytes.Buffer
	if _, err := buf.ReadFrom(gz); err != nil {
		return nil, fmt.Errorf("failed to decompress tokenizer: %w", err)
	}
	return ParseTokenizerJSON(buf.Bytes())
}

// ParseTokenizerJSON parses tokenizer.json payload into a Tokenizer.
// Fails closed with ErrArgs/ErrSyntax if normalizer, pre_tokenizer, post_processor, decoder,
// added_tokens flags, model parameters, vocab IDs, or merges do not match one of the supported
// configurations (Reckoner 1.4B or Qwen3.5).
func ParseTokenizerJSON(data []byte) (*Tokenizer, error) {
	type rawAddedToken struct {
		ID         int    `json:"id"`
		Content    string `json:"content"`
		SingleWord bool   `json:"single_word"`
		LStrip     bool   `json:"lstrip"`
		RStrip     bool   `json:"rstrip"`
		Normalized bool   `json:"normalized"`
		Special    bool   `json:"special"`
	}
	type rawSplitPattern struct {
		Regex  *string `json:"Regex"`
		String *string `json:"String"`
	}
	type rawPreTokItem struct {
		Type             string           `json:"type"`
		Pattern          *rawSplitPattern `json:"pattern"`
		Behavior         string           `json:"behavior"`
		Invert           *bool            `json:"invert"`
		IndividualDigits *bool            `json:"individual_digits"`
		AddPrefixSpace   *bool            `json:"add_prefix_space"`
		TrimOffsets      *bool            `json:"trim_offsets"`
		UseRegex         *bool            `json:"use_regex"`
	}
	type rawNormItem struct {
		Type string `json:"type"`
	}
	type rawByteLevelStage struct {
		Type           string `json:"type"`
		AddPrefixSpace *bool  `json:"add_prefix_space"`
		TrimOffsets    *bool  `json:"trim_offsets"`
		UseRegex       *bool  `json:"use_regex"`
	}
	type rawTokenizer struct {
		AddedTokens []rawAddedToken `json:"added_tokens"`
		Normalizer  *struct {
			Type        string        `json:"type"`
			Normalizers []rawNormItem `json:"normalizers"`
		} `json:"normalizer"`
		PreTokenizer *struct {
			Type             string           `json:"type"`
			Pattern          *rawSplitPattern `json:"pattern"`
			Behavior         string           `json:"behavior"`
			Invert           *bool            `json:"invert"`
			IndividualDigits *bool            `json:"individual_digits"`
			AddPrefixSpace   *bool            `json:"add_prefix_space"`
			TrimOffsets      *bool            `json:"trim_offsets"`
			UseRegex         *bool            `json:"use_regex"`
			PreTokenizers    []rawPreTokItem  `json:"pretokenizers"`
		} `json:"pre_tokenizer"`
		PostProcessor *rawByteLevelStage `json:"post_processor"`
		Decoder       *rawByteLevelStage `json:"decoder"`
		Model         struct {
			Type                    string            `json:"type"`
			Vocab                   map[string]int    `json:"vocab"`
			Merges                  []json.RawMessage `json:"merges"`
			ContinuingSubwordPrefix *string           `json:"continuing_subword_prefix"`
			EndOfWordSuffix         *string           `json:"end_of_word_suffix"`
			ByteFallback            *bool             `json:"byte_fallback"`
			IgnoreMerges            *bool             `json:"ignore_merges"`
		} `json:"model"`
	}

	var raw rawTokenizer
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, fmt.Errorf("ErrSyntax: failed to decode tokenizer json: %w", err)
	}
	if len(raw.Model.Vocab) == 0 {
		return nil, fmt.Errorf("ErrSyntax: tokenizer json model.vocab is empty")
	}

	// Validate model BPE configuration (fail closed on unsupported BPE flags).
	if raw.Model.Type != "" && raw.Model.Type != "BPE" {
		return nil, fmt.Errorf("ErrArgs: unsupported tokenizer model.type %q (expected \"BPE\")", raw.Model.Type)
	}
	if raw.Model.ByteFallback != nil && *raw.Model.ByteFallback {
		return nil, fmt.Errorf("ErrArgs: unsupported tokenizer model.byte_fallback=true")
	}
	if raw.Model.IgnoreMerges != nil && *raw.Model.IgnoreMerges {
		return nil, fmt.Errorf("ErrArgs: unsupported tokenizer model.ignore_merges=true")
	}
	if raw.Model.ContinuingSubwordPrefix != nil && *raw.Model.ContinuingSubwordPrefix != "" {
		return nil, fmt.Errorf("ErrArgs: unsupported tokenizer model.continuing_subword_prefix %q", *raw.Model.ContinuingSubwordPrefix)
	}
	if raw.Model.EndOfWordSuffix != nil && *raw.Model.EndOfWordSuffix != "" {
		return nil, fmt.Errorf("ErrArgs: unsupported tokenizer model.end_of_word_suffix %q", *raw.Model.EndOfWordSuffix)
	}

	// Fail-closed detection of normalizer + pre_tokenizer pipeline:
	// Supported Pipeline A (Reckoner 1.4B):
	//   normalizer: nil or Sequence[NFKC] / NFKC
	//   pre_tokenizer: Sequence[Digits(individual_digits=true), ByteLevel(add_prefix_space=false, use_regex=true)]
	// Supported Pipeline B (Qwen3.5):
	//   normalizer: NFC
	//   pre_tokenizer: Sequence[Split(pattern.Regex == Qwen35SplitRegex, behavior == "Isolated", invert == false), ByteLevel(add_prefix_space=false, use_regex=false)]
	//   post_processor: nil or ByteLevel(add_prefix_space=false, use_regex=false)
	//   decoder: ByteLevel(add_prefix_space=false, use_regex=false)
	useNFC := false
	isQwenPreTok := false

	normType := ""
	if raw.Normalizer != nil {
		normType = raw.Normalizer.Type
	}
	switch normType {
	case "NFC":
		useNFC = true
	case "", "NFKC":
		useNFC = false
	case "Sequence":
		if len(raw.Normalizer.Normalizers) == 1 && raw.Normalizer.Normalizers[0].Type == "NFC" {
			useNFC = true
		} else if len(raw.Normalizer.Normalizers) == 1 && raw.Normalizer.Normalizers[0].Type == "NFKC" {
			useNFC = false
		} else if len(raw.Normalizer.Normalizers) > 0 {
			return nil, fmt.Errorf("ErrArgs: unsupported tokenizer normalizer Sequence %+v", raw.Normalizer.Normalizers)
		}
	default:
		return nil, fmt.Errorf("ErrArgs: unsupported tokenizer normalizer.type %q", normType)
	}

	if raw.PreTokenizer == nil {
		return nil, fmt.Errorf("ErrArgs: tokenizer json missing pre_tokenizer")
	}
	if raw.PreTokenizer.Type != "Sequence" || len(raw.PreTokenizer.PreTokenizers) != 2 {
		return nil, fmt.Errorf("ErrArgs: unsupported pre_tokenizer structure (type=%q, len=%d); expected 2-stage Sequence", raw.PreTokenizer.Type, len(raw.PreTokenizer.PreTokenizers))
	}
	stage0 := raw.PreTokenizer.PreTokenizers[0]
	stage1 := raw.PreTokenizer.PreTokenizers[1]
	if stage1.Type != "ByteLevel" {
		return nil, fmt.Errorf("ErrArgs: unsupported second pre_tokenizer stage %q (expected \"ByteLevel\")", stage1.Type)
	}
	if stage1.AddPrefixSpace == nil || *stage1.AddPrefixSpace {
		return nil, fmt.Errorf("ErrArgs: ByteLevel pre_tokenizer must have add_prefix_space=false")
	}

	switch stage0.Type {
	case "Split":
		if !useNFC {
			return nil, fmt.Errorf("ErrArgs: Split pre_tokenizer requires NFC normalizer")
		}
		if stage0.Pattern == nil || stage0.Pattern.Regex == nil || *stage0.Pattern.Regex != Qwen35SplitRegex {
			return nil, fmt.Errorf("ErrArgs: unsupported Split pre_tokenizer regex pattern")
		}
		if stage0.Behavior != "Isolated" {
			return nil, fmt.Errorf("ErrArgs: unsupported Split pre_tokenizer behavior %q (expected \"Isolated\")", stage0.Behavior)
		}
		if stage0.Invert != nil && *stage0.Invert {
			return nil, fmt.Errorf("ErrArgs: Split pre_tokenizer must have invert=false")
		}
		if stage1.UseRegex == nil || *stage1.UseRegex {
			return nil, fmt.Errorf("ErrArgs: Qwen3.5 ByteLevel pre_tokenizer must have use_regex=false")
		}
		if raw.PostProcessor != nil {
			if raw.PostProcessor.Type != "ByteLevel" {
				return nil, fmt.Errorf("ErrArgs: unsupported Qwen3.5 post_processor.type %q (expected null or \"ByteLevel\")", raw.PostProcessor.Type)
			}
			if raw.PostProcessor.AddPrefixSpace != nil && *raw.PostProcessor.AddPrefixSpace {
				return nil, fmt.Errorf("ErrArgs: Qwen3.5 ByteLevel post_processor must have add_prefix_space=false")
			}
			if raw.PostProcessor.UseRegex != nil && *raw.PostProcessor.UseRegex {
				return nil, fmt.Errorf("ErrArgs: Qwen3.5 ByteLevel post_processor must have use_regex=false")
			}
		}
		if raw.Decoder == nil || raw.Decoder.Type != "ByteLevel" {
			decType := "<nil>"
			if raw.Decoder != nil {
				decType = raw.Decoder.Type
			}
			return nil, fmt.Errorf("ErrArgs: unsupported Qwen3.5 decoder.type %q (expected \"ByteLevel\")", decType)
		}
		if raw.Decoder.AddPrefixSpace != nil && *raw.Decoder.AddPrefixSpace {
			return nil, fmt.Errorf("ErrArgs: Qwen3.5 ByteLevel decoder must have add_prefix_space=false")
		}
		if raw.Decoder.UseRegex != nil && *raw.Decoder.UseRegex {
			return nil, fmt.Errorf("ErrArgs: Qwen3.5 ByteLevel decoder must have use_regex=false")
		}
		isQwenPreTok = true

	case "Digits":
		if useNFC {
			return nil, fmt.Errorf("ErrArgs: NFC normalizer requires Qwen3.5 Split pre_tokenizer")
		}
		if stage0.IndividualDigits == nil || !*stage0.IndividualDigits {
			return nil, fmt.Errorf("ErrArgs: Digits pre_tokenizer must have individual_digits=true")
		}
		if stage1.UseRegex == nil || !*stage1.UseRegex {
			return nil, fmt.Errorf("ErrArgs: 1.4B ByteLevel pre_tokenizer must have use_regex=true")
		}
		isQwenPreTok = false

	default:
		return nil, fmt.Errorf("ErrArgs: unsupported first pre_tokenizer stage %q", stage0.Type)
	}

	vocab := raw.Model.Vocab
	added := make(map[string]int, len(raw.AddedTokens))
	var addedByByte [256][]string
	var nonAngleAdded []string
	for _, at := range raw.AddedTokens {
		if at.ID < 0 || at.ID > MaxVocabID {
			return nil, fmt.Errorf("ErrArgs: added_token %q has out-of-bounds ID %d (max %d)", at.Content, at.ID, MaxVocabID)
		}
		if isQwenPreTok && (at.LStrip || at.RStrip || at.SingleWord || at.Normalized) {
			return nil, fmt.Errorf(
				"ErrArgs: added_token %q has unsupported flags (lstrip=%v, rstrip=%v, single_word=%v, normalized=%v; all must be false)",
				at.Content, at.LStrip, at.RStrip, at.SingleWord, at.Normalized,
			)
		}
		added[at.Content] = at.ID
		if _, exists := vocab[at.Content]; !exists {
			vocab[at.Content] = at.ID
		}
		if len(at.Content) > 0 {
			fb := at.Content[0]
			addedByByte[fb] = append(addedByByte[fb], at.Content)
		}
		if strings.HasPrefix(at.Content, "<") && !strings.HasPrefix(at.Content, "<|") {
			nonAngleAdded = append(nonAngleAdded, at.Content)
		}
	}
	sort.Slice(nonAngleAdded, func(i, j int) bool {
		if len(nonAngleAdded[i]) != len(nonAngleAdded[j]) {
			return len(nonAngleAdded[i]) > len(nonAngleAdded[j])
		}
		return nonAngleAdded[i] < nonAngleAdded[j]
	})
	for b := 0; b < 256; b++ {
		if len(addedByByte[b]) > 1 {
			s := addedByByte[b]
			sort.Slice(s, func(i, j int) bool {
				if len(s[i]) != len(s[j]) {
					return len(s[i]) > len(s[j])
				}
				return s[i] < s[j]
			})
		}
	}

	maxID := -1
	for tokStr, id := range vocab {
		if id < 0 || id > MaxVocabID {
			return nil, fmt.Errorf("ErrArgs: vocab token %q has out-of-bounds ID %d (max %d)", tokStr, id, MaxVocabID)
		}
		if id > maxID {
			maxID = id
		}
	}
	if maxID < 0 {
		return nil, fmt.Errorf("ErrSyntax: tokenizer json contains no valid token IDs")
	}

	invVocab := make([]string, maxID+1)
	for s, id := range vocab {
		if id >= 0 && id < len(invVocab) {
			invVocab[id] = s
		}
	}

	mergeRanks := make(map[bpePair]int, len(raw.Model.Merges))
	for rank, rawMerge := range raw.Model.Merges {
		if len(rawMerge) == 0 {
			return nil, fmt.Errorf("ErrSyntax: empty merge entry at rank %d", rank)
		}
		if rawMerge[0] == '[' {
			var pair []string
			if err := json.Unmarshal(rawMerge, &pair); err != nil {
				return nil, fmt.Errorf("ErrSyntax: invalid merge array at rank %d: %w", rank, err)
			}
			if len(pair) != 2 || pair[0] == "" || pair[1] == "" {
				return nil, fmt.Errorf("ErrSyntax: merge array at rank %d must have exactly 2 non-empty strings", rank)
			}
			mergeRanks[bpePair{a: pair[0], b: pair[1]}] = rank
		} else {
			var s string
			if err := json.Unmarshal(rawMerge, &s); err != nil {
				return nil, fmt.Errorf("ErrSyntax: invalid merge string at rank %d: %w", rank, err)
			}
			spIdx := strings.IndexByte(s, ' ')
			if spIdx <= 0 || spIdx >= len(s)-1 || strings.IndexByte(s[spIdx+1:], ' ') != -1 {
				return nil, fmt.Errorf("ErrSyntax: malformed merge string %q at rank %d (expected \"<a> <b>\")", s, rank)
			}
			mergeRanks[bpePair{a: s[:spIdx], b: s[spIdx+1:]}] = rank
		}
	}

	m0ID, hasM0 := vocab["<|m0|>"]
	l0ID, hasL0 := vocab["<|L0|>"]
	hasAnchors := hasM0 && m0ID == M0TokenID && hasL0 && l0ID == L0TokenID

	tok := &Tokenizer{
		Vocab:               vocab,
		InvVocab:            invVocab,
		AddedTokens:         added,
		AddedByFirstByte:    addedByByte,
		NonAngleAddedTokens: nonAngleAdded,
		MergeRanks:          mergeRanks,
		NewlineTokens:       make([]bool, len(invVocab)),
		WhitespaceTokens:    make([]bool, len(invVocab)),
		IsQwenPreTokenizer:  isQwenPreTok,
		UseNFC:              useNFC,
		HasAnchors:          hasAnchors,
	}

	for id, tstr := range invVocab {
		if tstr == "" {
			continue
		}
		if strings.Contains(tstr, "Ċ") || strings.Contains(tstr, "\n") {
			tok.NewlineTokens[id] = true
		}
		if strings.HasPrefix(tstr, "Ġ") || strings.HasPrefix(tstr, " ") || strings.Contains(tstr, "\t") {
			tok.WhitespaceTokens[id] = true
		}
	}

	initByteDecoder(tok)
	return tok, nil
}

func initByteDecoder(tok *Tokenizer) {
	bs := make([]int, 0, 256)
	for b := int('!'); b <= int('~'); b++ {
		bs = append(bs, b)
	}
	for b := int('¡'); b <= int('¬'); b++ {
		bs = append(bs, b)
	}
	for b := int('®'); b <= int('ÿ'); b++ {
		bs = append(bs, b)
	}
	cs := make([]int, len(bs), 256)
	copy(cs, bs)
	n := 0
	for b := 0; b < 256; b++ {
		found := false
		for _, existing := range bs {
			if existing == b {
				found = true
				break
			}
		}
		if !found {
			bs = append(bs, b)
			cs = append(cs, 256+n)
			n++
		}
	}
	for i := 0; i < len(bs); i++ {
		c := cs[i]
		b := byte(bs[i])
		tok.ByteToRune[b] = rune(c)
		if c < len(tok.RuneToByte) {
			tok.RuneToByte[c] = b
			tok.ValidRune[c] = true
		}
	}
}

// TokenToID looks up a token string in the vocabulary. Returns -1 if not found.
func (tok *Tokenizer) TokenToID(token string) int {
	if id, ok := tok.Vocab[token]; ok {
		return id
	}
	return -1
}

// IDToToken looks up a token ID. Returns empty string if out of range.
func (tok *Tokenizer) IDToToken(id int) string {
	if id >= 0 && id < len(tok.InvVocab) {
		return tok.InvVocab[id]
	}
	return ""
}

// IsAnchorToken returns true if tokenID is a message token or line anchor.
func (tok *Tokenizer) IsAnchorToken(tokenID int) bool {
	if tok != nil && !tok.HasAnchors {
		return false
	}
	isMsg := tokenID >= M0TokenID && tokenID < M0TokenID+NumMessageTokens
	isLine := tokenID >= L0TokenID && tokenID < L0TokenID+NumLineAnchors
	return isMsg || isLine
}

// GetMessageTokenID returns the token ID for a message index (0..255).
func (tok *Tokenizer) GetMessageTokenID(msgIdx int) int {
	return M0TokenID + (msgIdx % NumMessageTokens)
}

// GetLineTokenID returns the token ID for a line index (0..1023).
func (tok *Tokenizer) GetLineTokenID(lineIdx int) int {
	if lineIdx < 0 || lineIdx >= NumLineAnchors {
		panic(fmt.Sprintf("Line index %d out of range 0..%d", lineIdx, NumLineAnchors-1))
	}
	return L0TokenID + lineIdx
}

// DecodeTokens decodes a sequence of BPE token IDs back into the original UTF-8 text.
func (tok *Tokenizer) DecodeTokens(tokens []int) string {
	var buf bytes.Buffer
	for _, t := range tokens {
		if t < 0 || t >= len(tok.InvVocab) {
			continue
		}
		tstr := tok.InvVocab[t]
		for _, r := range tstr {
			if int(r) < len(tok.RuneToByte) && tok.ValidRune[r] {
				buf.WriteByte(tok.RuneToByte[r])
			} else {
				buf.WriteRune(r)
			}
		}
	}
	return buf.String()
}

// InsertAnchors inserts message tokens and line anchors into a token sequence,
// matching the training data renderer (data/anchor_inserter.py) token-for-token.
func (tok *Tokenizer) InsertAnchors(tokens []int, targets []int, messageID int) ([]int, []int, int, error) {
	if len(tokens) == 0 {
		return []int{}, []int{}, messageID, nil
	}
	if targets == nil {
		targets = make([]int, len(tokens))
		copy(targets, tokens)
	}
	if len(tokens) != len(targets) {
		return nil, nil, 0, fmt.Errorf("ErrArgs: length mismatch: %d tokens vs %d targets", len(tokens), len(targets))
	}
	for _, t := range tokens {
		if t < 0 || t >= len(tok.InvVocab) {
			return nil, nil, 0, fmt.Errorf("ErrArgs: token id out of vocabulary: %d", t)
		}
		if tok.IsAnchorToken(t) {
			return nil, nil, 0, fmt.Errorf("ErrArgs: input tokens contain raw anchor token: %d", t)
		}
	}

	if messageID < 0 {
		return nil, nil, 0, fmt.Errorf("ErrArgs: message_id must be non-negative: %d", messageID)
	}

	outTokens := make([]int, 0, len(tokens)+len(tokens)/16+4)
	outTargets := make([]int, 0, cap(outTokens))

	curMsgID := ((messageID % NumMessageTokens) + NumMessageTokens) % NumMessageTokens
	curLineIdx := 0

	// Start message with <|mN|> and <|L0|>
	outTokens = append(outTokens, M0TokenID+curMsgID)
	outTargets = append(outTargets, IgnoreIndex)

	outTokens = append(outTokens, L0TokenID+curLineIdx)
	outTargets = append(outTargets, IgnoreIndex)
	curLineIdx++

	tokensSinceAnchor := 0
	prevWasNewline := false

	for i, t := range tokens {
		y := targets[i]

		isStartOfLine := prevWasNewline
		isMidLine := (tokensSinceAnchor >= 64) && (t >= 0 && t < len(tok.WhitespaceTokens) && tok.WhitespaceTokens[t])

		if isStartOfLine || isMidLine {
			if curLineIdx >= NumLineAnchors {
				curMsgID = (curMsgID + 1) % NumMessageTokens
				outTokens = append(outTokens, M0TokenID+curMsgID)
				outTargets = append(outTargets, IgnoreIndex)
				curLineIdx = 0
			}

			outTokens = append(outTokens, L0TokenID+curLineIdx)
			outTargets = append(outTargets, IgnoreIndex)
			curLineIdx++
			tokensSinceAnchor = 0
		}

		outTokens = append(outTokens, t)
		outTargets = append(outTargets, y)
		tokensSinceAnchor++

		prevWasNewline = (t >= 0 && t < len(tok.NewlineTokens) && tok.NewlineTokens[t])
	}

	nextMsgID := (curMsgID + 1) % NumMessageTokens
	return outTokens, outTargets, nextMsgID, nil
}

// StripAnchors removes <|mN|> and <|Lk|> anchor tokens from a token sequence.
func (tok *Tokenizer) StripAnchors(tokens []int) []int {
	out := make([]int, 0, len(tokens))
	for _, t := range tokens {
		if !tok.IsAnchorToken(t) {
			out = append(out, t)
		}
	}
	return out
}

// Encode encodes UTF-8 text into BPE token IDs matching HuggingFace PreTrainedTokenizerFast
// (added_tokens split -> normalizer/pre_tokenizer -> BPE merges).
func (tok *Tokenizer) Encode(text string) []int {
	if text == "" {
		return []int{}
	}
	if tok.IsQwenPreTokenizer {
		return tok.encodeQwen(text)
	}
	var out []int
	i := 0
	n := len(text)
	for i < n {
		// Find next special token <|...|>
		idx := strings.Index(text[i:], "<|")
		if idx == -1 {
			out = append(out, tok.encodePlainSegment(text[i:])...)
			break
		}
		absIdx := i + idx
		endIdx := strings.Index(text[absIdx+2:], "|>")
		if endIdx != -1 {
			candEnd := absIdx + 2 + endIdx + 2
			cand := text[absIdx:candEnd]
			if spID, ok := tok.AddedTokens[cand]; ok {
				if absIdx > i {
					out = append(out, tok.encodePlainSegment(text[i:absIdx])...)
				}
				out = append(out, spID)
				i = candEnd
				continue
			}
		}
		// Not a recognized special token; encode up past "<|"
		out = append(out, tok.encodePlainSegment(text[i:absIdx+2])...)
		i = absIdx + 2
	}
	return out
}

// encodeQwen reproduces the Qwen3.5 Hugging Face tokenizer pipeline:
//  1. Split exact non-normalized added_tokens (both <|...|> and XML-style <tool_call>/<think> etc.)
//     without breaking surrounding non-added text when '<' or '<|' is not a registered added token.
//  2. Apply Unicode NFC normalization (without Stream-Safe U+034F insertion) to each non-added segment.
//  3. Apply the 7-branch ordered PCRE alternation from Qwen3.5's Split pre-tokenizer:
//     (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
//  4. Run byte-level BPE merges on each piece.
func (tok *Tokenizer) encodeQwen(text string) []int {
	var out []int
	segStart := 0
	i := 0
	n := len(text)
	for i < n {
		cands := tok.AddedByFirstByte[text[i]]
		matched := ""
		for _, cand := range cands {
			if strings.HasPrefix(text[i:], cand) {
				matched = cand
				break
			}
		}
		if matched != "" {
			if i > segStart {
				out = append(out, tok.encodeQwenPlainSegment(text[segStart:i])...)
			}
			out = append(out, tok.AddedTokens[matched])
			i += len(matched)
			segStart = i
		} else {
			i++
		}
	}
	if segStart < n {
		out = append(out, tok.encodeQwenPlainSegment(text[segStart:])...)
	}
	return out
}

func isQwenLetterOrMark(r rune) bool {
	return unicode.IsLetter(r) || unicode.IsMark(r)
}

func isQwenPunct(r rune) bool {
	return !unicode.IsSpace(r) && !unicode.IsLetter(r) && !unicode.IsMark(r) && !unicode.IsNumber(r)
}

func runeCCC(r rune) uint8 {
	var buf [4]byte
	n := utf8.EncodeRune(buf[:], r)
	return norm.NFC.Properties(buf[:n]).CCC()
}

func appendNFDNoCGJ(dst []rune, r rune) []rune {
	var buf [4]byte
	n := utf8.EncodeRune(buf[:], r)
	p := norm.NFD.Properties(buf[:n])
	if dec := p.Decomposition(); len(dec) > 0 {
		for _, dr := range string(dec) {
			dst = appendNFDNoCGJ(dst, dr)
		}
		return dst
	}
	// Hangul precomposed syllables U+AC00..U+D7A3
	const (
		sBase  = 0xAC00
		lBase  = 0x1100
		vBase  = 0x1161
		tBase  = 0x11A7
		lCount = 19
		vCount = 21
		tCount = 28
		nCount = vCount * tCount // 588
		sCount = lCount * nCount // 11172
	)
	if r >= sBase && r < sBase+sCount {
		sIdx := r - sBase
		l := lBase + sIdx/nCount
		v := vBase + (sIdx%nCount)/tCount
		t := tBase + sIdx%tCount
		dst = append(dst, l, v)
		if t != tBase {
			dst = append(dst, t)
		}
		return dst
	}
	return append(dst, r)
}

func composePair(a, b rune) (rune, bool) {
	s := norm.NFC.String(string([]rune{a, b}))
	r, sz := utf8.DecodeRuneInString(s)
	if sz == len(s) && r != utf8.RuneError {
		return r, true
	}
	return 0, false
}

// nfcNormalizeWithoutStreamSafe applies standard UAX #15 NFC normalization without the
// Stream-Safe Text Process (UAX #15 §12), which Go's norm.NFC applies by inserting U+034F (CGJ)
// after 30 consecutive non-starters. Hugging Face's Rust `unicode-normalization` crate performs
// pure NFC without Stream-Safe CGJ insertion.
func nfcNormalizeWithoutStreamSafe(s string) string {
	fast := norm.NFC.String(s)
	// Fast path: if norm.NFC did not insert any U+034F (CGJ) runes, return fast directly.
	if !strings.ContainsRune(fast, '\u034f') || strings.Count(s, "\u034f") == strings.Count(fast, "\u034f") {
		return fast
	}
	// Slow path (>30 consecutive non-starters):
	// 1. Canonical decomposition (NFD) per rune without CGJ insertion.
	var runes []rune
	for _, r := range s {
		runes = appendNFDNoCGJ(runes, r)
	}
	// 2. Canonical Ordering Algorithm: stable sort contiguous runs of non-starters (CCC > 0) by CCC.
	i := 0
	for i < len(runes) {
		if runeCCC(runes[i]) == 0 {
			i++
			continue
		}
		j := i + 1
		for j < len(runes) && runeCCC(runes[j]) > 0 {
			j++
		}
		if j-i > 1 {
			sub := runes[i:j]
			sort.SliceStable(sub, func(a, b int) bool {
				return runeCCC(sub[a]) < runeCCC(sub[b])
			})
		}
		i = j
	}
	// 3. Canonical Composition Algorithm (UAX #15 D115-D117).
	var out []rune
	starterIdx := -1
	var lastCCC uint8
	for _, r := range runes {
		ccc := runeCCC(r)
		if starterIdx >= 0 && (lastCCC == 0 || lastCCC < ccc) {
			if comp, ok := composePair(out[starterIdx], r); ok {
				out[starterIdx] = comp
				continue
			}
		}
		if ccc == 0 {
			starterIdx = len(out)
			lastCCC = 0
		} else if starterIdx >= 0 {
			if ccc > lastCCC {
				lastCCC = ccc
			}
		}
		out = append(out, r)
	}
	return string(out)
}

func (tok *Tokenizer) encodeQwenPlainSegment(seg string) []int {
	if seg == "" {
		return nil
	}
	if tok.UseNFC {
		seg = nfcNormalizeWithoutStreamSafe(seg)
	}
	runes := []rune(seg)
	n := len(runes)
	var out []int
	i := 0
	for i < n {
		r0 := runes[i]

		// Branch 1: (?i:'s|'t|'re|'ve|'m|'ll|'d)
		if r0 == '\'' && i+1 < n {
			r1 := unicode.ToLower(runes[i+1])
			if i+2 < n {
				r2 := unicode.ToLower(runes[i+2])
				if (r1 == 'r' && r2 == 'e') || (r1 == 'v' && r2 == 'e') || (r1 == 'l' && r2 == 'l') {
					out = append(out, tok.encodeBPEPiece(string(runes[i:i+3]))...)
					i += 3
					continue
				}
			}
			if r1 == 's' || r1 == 't' || r1 == 'm' || r1 == 'd' {
				out = append(out, tok.encodeBPEPiece(string(runes[i:i+2]))...)
				i += 2
				continue
			}
		}

		// Branch 2: [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
		if isQwenLetterOrMark(r0) || (r0 != '\r' && r0 != '\n' && !unicode.IsLetter(r0) && !unicode.IsNumber(r0) && i+1 < n && isQwenLetterOrMark(runes[i+1])) {
			j := i + 1
			for j < n && isQwenLetterOrMark(runes[j]) {
				j++
			}
			out = append(out, tok.encodeBPEPiece(string(runes[i:j]))...)
			i = j
			continue
		}

		// Branch 3: \p{N} (single Unicode number rune)
		if unicode.IsNumber(r0) {
			out = append(out, tok.encodeBPEPiece(string(runes[i:i+1]))...)
			i++
			continue
		}

		// Branch 4:  ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
		if (r0 == ' ' && i+1 < n && isQwenPunct(runes[i+1])) || isQwenPunct(r0) {
			j := i
			if runes[j] == ' ' {
				j++
			}
			for j < n && isQwenPunct(runes[j]) {
				j++
			}
			for j < n && (runes[j] == '\r' || runes[j] == '\n') {
				j++
			}
			out = append(out, tok.encodeBPEPiece(string(runes[i:j]))...)
			i = j
			continue
		}

		// Branches 5, 6, 7: \s*[\r\n]+|\s+(?!\S)|\s+
		if unicode.IsSpace(r0) {
			j := i
			lastCRLF := -1
			for j < n && unicode.IsSpace(runes[j]) {
				if runes[j] == '\r' || runes[j] == '\n' {
					lastCRLF = j
				}
				j++
			}
			if lastCRLF != -1 {
				out = append(out, tok.encodeBPEPiece(string(runes[i:lastCRLF+1]))...)
				i = lastCRLF + 1
			} else if j < n && j-i > 1 {
				out = append(out, tok.encodeBPEPiece(string(runes[i:j-1]))...)
				i = j - 1
			} else {
				out = append(out, tok.encodeBPEPiece(string(runes[i:j]))...)
				i = j
			}
			continue
		}

		// Fallback single rune
		out = append(out, tok.encodeBPEPiece(string(runes[i:i+1]))...)
		i++
	}
	return out
}

func (tok *Tokenizer) encodePlainSegment(seg string) []int {
	if seg == "" {
		return nil
	}
	var out []int
	// Step 1: Digits(individual_digits=True) splits around every unicode.IsNumber rune
	runes := []rune(seg)
	start := 0
	for idx, r := range runes {
		if unicode.IsNumber(r) {
			if idx > start {
				out = append(out, tok.encodeByteLevelSubsegment(runes[start:idx])...)
			}
			out = append(out, tok.encodeBPEPiece(string(r))...)
			start = idx + 1
		}
	}
	if start < len(runes) {
		out = append(out, tok.encodeByteLevelSubsegment(runes[start:])...)
	}
	return out
}

func isByteLevelPunct(r rune) bool {
	return !unicode.IsSpace(r) && !unicode.IsLetter(r) && !unicode.IsNumber(r)
}

// encodeByteLevelSubsegment splits a non-digit rune slice using GPT-2 ByteLevel regex:
// 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
func (tok *Tokenizer) encodeByteLevelSubsegment(runes []rune) []int {
	var out []int
	n := len(runes)
	i := 0
	for i < n {
		// 1. Contractions: 's|'t|'re|'ve|'m|'ll|'d
		if runes[i] == '\'' && i+1 < n {
			if i+2 < n {
				two := string(runes[i+1 : i+3])
				if two == "re" || two == "ve" || two == "ll" {
					out = append(out, tok.encodeBPEPiece(string(runes[i:i+3]))...)
					i += 3
					continue
				}
			}
			one := runes[i+1]
			if one == 's' || one == 't' || one == 'm' || one == 'd' {
				out = append(out, tok.encodeBPEPiece(string(runes[i:i+2]))...)
				i += 2
				continue
			}
		}

		// 2. Optional space followed by letters:  ?\p{L}+
		if (runes[i] == ' ' && i+1 < n && unicode.IsLetter(runes[i+1])) || unicode.IsLetter(runes[i]) {
			j := i
			if runes[j] == ' ' {
				j++
			}
			for j < n && unicode.IsLetter(runes[j]) {
				j++
			}
			out = append(out, tok.encodeBPEPiece(string(runes[i:j]))...)
			i = j
			continue
		}

		// 3. Optional space followed by numbers:  ?\p{N}+
		if (runes[i] == ' ' && i+1 < n && unicode.IsNumber(runes[i+1])) || unicode.IsNumber(runes[i]) {
			j := i
			if runes[j] == ' ' {
				j++
			}
			for j < n && unicode.IsNumber(runes[j]) {
				j++
			}
			out = append(out, tok.encodeBPEPiece(string(runes[i:j]))...)
			i = j
			continue
		}

		// 4. Optional space followed by non-space/non-letter/non-number:  ?[^\s\p{L}\p{N}]+
		if (runes[i] == ' ' && i+1 < n && isByteLevelPunct(runes[i+1])) || isByteLevelPunct(runes[i]) {
			j := i
			if runes[j] == ' ' {
				j++
			}
			for j < n && isByteLevelPunct(runes[j]) {
				j++
			}
			out = append(out, tok.encodeBPEPiece(string(runes[i:j]))...)
			i = j
			continue
		}

		// 5. Whitespace: \s+(?!\S)|\s+
		if unicode.IsSpace(runes[i]) {
			j := i
			for j < n && unicode.IsSpace(runes[j]) {
				j++
			}
			if j < n && j-i > 1 {
				// \s+(?!\S) leaves the final whitespace rune for the next token
				out = append(out, tok.encodeBPEPiece(string(runes[i:j-1]))...)
				i = j - 1
			} else {
				out = append(out, tok.encodeBPEPiece(string(runes[i:j]))...)
				i = j
			}
			continue
		}

		// Fallback single rune
		out = append(out, tok.encodeBPEPiece(string(runes[i:i+1]))...)
		i++
	}
	return out
}

func (tok *Tokenizer) encodeBPEPiece(piece string) []int {
	if piece == "" {
		return nil
	}
	rawBytes := []byte(piece)
	word := make([]string, len(rawBytes))
	for i, b := range rawBytes {
		word[i] = string(tok.ByteToRune[b])
	}

	if len(word) == 1 {
		if id, ok := tok.Vocab[word[0]]; ok {
			return []int{id}
		}
		return nil
	}

	for len(word) > 1 {
		bestRank := int(^uint(0) >> 1)
		bestPair := bpePair{}
		found := false

		for i := 0; i < len(word)-1; i++ {
			p := bpePair{a: word[i], b: word[i+1]}
			if rank, ok := tok.MergeRanks[p]; ok && rank < bestRank {
				bestRank = rank
				bestPair = p
				found = true
			}
		}
		if !found {
			break
		}

		merged := bestPair.a + bestPair.b
		nextWord := make([]string, 0, len(word)-1)
		i := 0
		for i < len(word) {
			if i < len(word)-1 && word[i] == bestPair.a && word[i+1] == bestPair.b {
				nextWord = append(nextWord, merged)
				i += 2
			} else {
				nextWord = append(nextWord, word[i])
				i++
			}
		}
		word = nextWord
	}

	ids := make([]int, 0, len(word))
	for _, sym := range word {
		if id, ok := tok.Vocab[sym]; ok {
			ids = append(ids, id)
		}
	}
	return ids
}
