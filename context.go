package main

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"unicode/utf8"
)

// MessageSegment represents an anchored message or continuation chunk in context.
type MessageSegment struct {
	MessageID int      `json:"message_id"` // 0..255
	TurnID    int      `json:"turn_id"`
	Active    bool     `json:"active"`
	IsDerived bool     `json:"is_derived,omitempty"` // true if created by find or lines
	RawTokens []int    `json:"raw_tokens"`
	Anchored  []int    `json:"anchored_tokens"`
	Lines     [][]int  `json:"lines"`
	LineTexts []string `json:"line_texts"`
	NormLines []string `json:"norm_lines,omitempty"`
	FullText  string   `json:"full_text"`
}

// QuoteProvenance contains metadata for an expanded quote.
type QuoteProvenance struct {
	MessageID    int    `json:"message_id"`
	MessageToken string `json:"message_token"`
	StartLine    int    `json:"start_line"`
	EndLine      int    `json:"end_line"`
	NumLines     int    `json:"num_lines"`
	NumTokens    int    `json:"num_tokens"`
	CharStart    int    `json:"char_start"`
	CharEnd      int    `json:"char_end"`
}

// QuoteResult holds the full resolution of a quote operation.
type QuoteResult struct {
	Success    bool            `json:"success"`
	Result     string          `json:"result"`
	Receipt    string          `json:"receipt"`
	Text       string          `json:"text"`
	Tokens     []int           `json:"tokens"`
	Provenance QuoteProvenance `json:"provenance"`
}

// QuoteMaskResult represents constrained decoding options for a quote step.
type QuoteMaskResult struct {
	Success         bool     `json:"success"`
	AllowedTokens   []string `json:"allowed_tokens"`
	AllowedTokenIDs []int    `json:"allowed_token_ids"`
	Step            string   `json:"step"`
}

// FindResult represents the outcome of a find operation.
type FindResult struct {
	Success    bool   `json:"success"`
	Result     string `json:"result"`
	MessageID  int    `json:"message_id,omitempty"`
	Tokens     []int  `json:"tokens,omitempty"`
	NumMatches int    `json:"num_matches"`
}

// ContextStore manages the conversation context, message segments, line indexing,
// recycling past 256 messages, quote expansion, and find matching.
type ContextStore struct {
	mu             sync.RWMutex
	tok            *Tokenizer
	activeSlots    [NumMessageTokens]*MessageSegment
	history        []*MessageSegment
	bm25Index      *BM25Index
	nextMsgID      int
	turnCounter    int
	anchorsEnabled bool
}

var globalContextStore *ContextStore
var globalContextOnce sync.Once

// GetGlobalContextStore returns the singleton ContextStore.
func GetGlobalContextStore() (*ContextStore, error) {
	var initErr error
	globalContextOnce.Do(func() {
		tok, err := GetGlobalTokenizer()
		if err != nil {
			initErr = err
			return
		}
		globalContextStore = NewContextStore(tok)
	})
	if initErr != nil {
		return nil, initErr
	}
	return globalContextStore, nil
}

// NewContextStore creates a new ContextStore using the provided Tokenizer.
func NewContextStore(tok *Tokenizer) *ContextStore {
	return &ContextStore{
		tok:            tok,
		bm25Index:      NewBM25Index(),
		nextMsgID:      0,
		anchorsEnabled: true,
	}
}

// SetAnchorsEnabled configures whether anchors are inserted and expected in context operations.
func (cs *ContextStore) SetAnchorsEnabled(enabled bool) {
	cs.mu.Lock()
	defer cs.mu.Unlock()
	cs.anchorsEnabled = enabled
}

// AnchorsEnabled reports whether anchors are currently enabled.
func (cs *ContextStore) AnchorsEnabled() bool {
	cs.mu.RLock()
	defer cs.mu.RUnlock()
	return cs.anchorsEnabled
}

// NextMessageID returns the next message ID to be assigned.
func (cs *ContextStore) NextMessageID() int {
	cs.mu.RLock()
	defer cs.mu.RUnlock()
	return cs.nextMsgID
}

// Reset clears all active message slots and conversation history.
func (cs *ContextStore) Reset() {
	cs.mu.Lock()
	defer cs.mu.Unlock()

	for i := range cs.activeSlots {
		cs.activeSlots[i] = nil
	}
	cs.history = nil
	cs.bm25Index.Reset()
	cs.nextMsgID = 0
	cs.turnCounter = 0
}

// StoreMessage anchors a token sequence and commits it into the active context.
// Returns the anchored token IDs, target IDs, the assigned message ID, and any continuation segments.
func (cs *ContextStore) StoreMessage(tokens []int, targets []int, startMsgID *int) ([]int, []int, int, error) {
	cs.mu.Lock()
	defer cs.mu.Unlock()

	reqMsgID := cs.nextMsgID
	if startMsgID != nil {
		if *startMsgID < 0 {
			return nil, nil, 0, fmt.Errorf("ErrArgs: message_id must be non-negative: %d", *startMsgID)
		}
		reqMsgID = *startMsgID % NumMessageTokens
	}

	if !cs.anchorsEnabled {
		// Anchor-free mode: do NOT insert <|mN|> or <|LN|> tokens
		cs.indexPlainTokensLocked(tokens, reqMsgID, false)
		nextMsg := (reqMsgID + 1) % NumMessageTokens
		cs.nextMsgID = nextMsg
		return tokens, targets, nextMsg, nil
	}

	anchoredTokens, anchoredTargets, nextMsg, err := cs.tok.InsertAnchors(tokens, targets, reqMsgID)
	if err != nil {
		return nil, nil, 0, err
	}

	// Parse message segments and line structures from the anchored token sequence
	cs.indexSegmentsLocked(anchoredTokens, false)
	cs.nextMsgID = nextMsg

	return anchoredTokens, anchoredTargets, nextMsg, nil
}

// indexPlainTokensLocked indexes lines and segments from a plain token stream without anchors.
func (cs *ContextStore) indexPlainTokensLocked(tokens []int, msgID int, isDerived bool) {
	if len(tokens) == 0 {
		return
	}

	var lines [][]int
	var curLine []int
	for _, t := range tokens {
		curLine = append(curLine, t)
		if t >= 0 && t < len(cs.tok.NewlineTokens) && cs.tok.NewlineTokens[t] {
			lines = append(lines, curLine)
			curLine = nil
		}
	}
	if len(curLine) > 0 {
		lines = append(lines, curLine)
	}

	cs.turnCounter++
	seg := &MessageSegment{
		MessageID: msgID,
		TurnID:    cs.turnCounter,
		Active:    true,
		IsDerived: isDerived,
		RawTokens: tokens,
		Anchored:  tokens,
		Lines:     lines,
		LineTexts: make([]string, len(lines)),
		NormLines: make([]string, len(lines)),
	}

	for i, line := range lines {
		txt := cs.tok.DecodeTokens(line)
		seg.LineTexts[i] = txt
		seg.NormLines[i] = NormalizeText(txt)
	}
	seg.FullText = cs.tok.DecodeTokens(tokens)

	slotIdx := msgID % NumMessageTokens
	if old := cs.activeSlots[slotIdx]; old != nil {
		old.Active = false
	}
	cs.activeSlots[slotIdx] = seg
	cs.history = append(cs.history, seg)

	if !isDerived && cs.bm25Index != nil {
		cs.bm25Index.AddSegment(seg)
	}
}

// indexSegmentsLocked indexes lines and segments from an anchored token stream.
func (cs *ContextStore) indexSegmentsLocked(anchoredTokens []int, isDerived bool) {
	var curMsgID = -1
	var curSegment *MessageSegment
	var curLineTokens []int

	flushLine := func() {
		if curSegment != nil && curLineTokens != nil {
			curSegment.Lines = append(curSegment.Lines, curLineTokens)
			curSegment.RawTokens = append(curSegment.RawTokens, curLineTokens...)
			curLineTokens = nil
		}
	}

	flushSegment := func() {
		flushLine()
		if curSegment != nil {
			curSegment.LineTexts = make([]string, len(curSegment.Lines))
			curSegment.NormLines = make([]string, len(curSegment.Lines))
			for i, line := range curSegment.Lines {
				txt := cs.tok.DecodeTokens(line)
				curSegment.LineTexts[i] = txt
				curSegment.NormLines[i] = NormalizeText(txt)
			}
			curSegment.FullText = cs.tok.DecodeTokens(curSegment.RawTokens)

			// Update active slots (replaces old assignment if recycled)
			slotIdx := curSegment.MessageID % NumMessageTokens
			if old := cs.activeSlots[slotIdx]; old != nil {
				old.Active = false
			}
			cs.activeSlots[slotIdx] = curSegment
			cs.history = append(cs.history, curSegment)
			if !isDerived && cs.bm25Index != nil {
				cs.bm25Index.AddSegment(curSegment)
			}
		}
	}

	for _, t := range anchoredTokens {
		if t >= M0TokenID && t < M0TokenID+NumMessageTokens {
			// Start new message token segment
			flushSegment()
			curMsgID = t - M0TokenID
			cs.turnCounter++
			curSegment = &MessageSegment{
				MessageID: curMsgID,
				TurnID:    cs.turnCounter,
				Active:    true,
				IsDerived: isDerived,
			}
			curSegment.Anchored = append(curSegment.Anchored, t)
			continue
		}

		if t >= L0TokenID && t < L0TokenID+NumLineAnchors {
			// Line anchor
			flushLine()
			curLineTokens = []int{}
			if curSegment != nil {
				curSegment.Anchored = append(curSegment.Anchored, t)
			}
			continue
		}

		// Text token
		if curSegment != nil {
			curSegment.Anchored = append(curSegment.Anchored, t)
			curLineTokens = append(curLineTokens, t)
		}
	}

	flushSegment()
}

var quoteRegex = regexp.MustCompile(`<\|quote\|><\|m(\d+)\|><\|L(\d+)\|><\|L(\d+)\|><\|/quote\|>`)

// ParseQuoteCall parses a quote call string like "<|quote|><|m0|><|L0|><|L5|><|/quote|>"
// or simplified CLI forms like "m0 L0 L5" or "0 0 5".
func ParseQuoteCall(call string) (msgID, startLine, endLine int, err error) {
	call = strings.TrimSpace(call)
	if match := quoteRegex.FindStringSubmatch(call); match != nil {
		m, _ := strconv.Atoi(match[1])
		s, _ := strconv.Atoi(match[2])
		e, _ := strconv.Atoi(match[3])
		return m, s, e, nil
	}

	// Try CLI form: "m0 L0 L5" or "0 0 5"
	call = strings.ReplaceAll(call, "m", "")
	call = strings.ReplaceAll(call, "L", "")
	call = strings.ReplaceAll(call, ",", " ")
	parts := strings.Fields(call)
	if len(parts) == 3 {
		m, err1 := strconv.Atoi(parts[0])
		s, err2 := strconv.Atoi(parts[1])
		e, err3 := strconv.Atoi(parts[2])
		if err1 == nil && err2 == nil && err3 == nil {
			return m, s, e, nil
		}
	}

	return 0, 0, 0, fmt.Errorf("ErrSyntax: malformed quote syntax: %s", call)
}

// formatLineSample produces a clean, truncated sample for quote receipt observations.
func formatLineSample(text string) string {
	text = strings.TrimSpace(text)
	text = strings.ReplaceAll(text, "\n", " ")
	text = strings.ReplaceAll(text, "\r", "")
	text = strings.ReplaceAll(text, "\t", " ")

	const maxRunes = 20
	runeCount := utf8.RuneCountInString(text)
	if runeCount > maxRunes {
		runes := []rune(text)
		return fmt.Sprintf("%q", string(runes[:maxRunes-3])+"…")
	}
	return fmt.Sprintf("%q", text)
}

// Quote resolves a quote reference against the active context.
func (cs *ContextStore) Quote(call string) (*QuoteResult, error) {
	msgID, startLine, endLine, err := ParseQuoteCall(call)
	if err != nil {
		return nil, err
	}
	if msgID < 0 || msgID >= NumMessageTokens {
		return nil, fmt.Errorf("ErrArgs: message id must be in range 0..%d: got %d", NumMessageTokens-1, msgID)
	}

	cs.mu.RLock()
	defer cs.mu.RUnlock()

	if !cs.anchorsEnabled {
		return nil, fmt.Errorf("ErrUnsupported: quote is not supported in anchor-free mode")
	}

	slotIdx := msgID % NumMessageTokens
	segment := cs.activeSlots[slotIdx]
	if segment == nil || !segment.Active || segment.MessageID != msgID {
		return nil, fmt.Errorf("ErrNotFound: message token <|m%d|> not found in active context", msgID)
	}

	if startLine < 0 || endLine < 0 {
		return nil, fmt.Errorf("ErrOutOfBounds: negative line index: start=%d end=%d", startLine, endLine)
	}
	if startLine > endLine {
		return nil, fmt.Errorf("ErrOutOfBounds: start line L%d > end line L%d", startLine, endLine)
	}
	numLines := len(segment.Lines)
	if endLine >= numLines {
		return nil, fmt.Errorf("ErrOutOfBounds: line index L%d out of bounds (message <|m%d|> has %d lines: L0..L%d)", endLine, msgID, numLines, numLines-1)
	}

	// Extract raw tokens for lines startLine..endLine
	var quotedTokens []int
	for i := startLine; i <= endLine; i++ {
		quotedTokens = append(quotedTokens, segment.Lines[i]...)
	}

	expandedText := cs.tok.DecodeTokens(quotedTokens)

	// Build receipt observation
	var receipt string
	lineCount := endLine - startLine + 1
	if startLine == endLine {
		sample := formatLineSample(segment.LineTexts[startLine])
		if cs.tok != nil {
			sample = cs.tok.escapeUserSystemTokens(sample)
		}
		receipt = fmt.Sprintf("<|obs|>Quoted 1 line (L%d): %s<|/obs|>", startLine, sample)
	} else {
		sample1 := formatLineSample(segment.LineTexts[startLine])
		sample2 := formatLineSample(segment.LineTexts[endLine])
		if cs.tok != nil {
			sample1 = cs.tok.escapeUserSystemTokens(sample1)
			sample2 = cs.tok.escapeUserSystemTokens(sample2)
		}
		receipt = fmt.Sprintf("<|obs|>Quoted %d lines (L%d-L%d): %s … %s<|/obs|>", lineCount, startLine, endLine, sample1, sample2)
	}

	charOffset := 0
	for l := 0; l < startLine && l < len(segment.LineTexts); l++ {
		charOffset += len([]rune(segment.LineTexts[l]))
	}
	charLen := len([]rune(expandedText))

	prov := QuoteProvenance{
		MessageID:    msgID,
		MessageToken: fmt.Sprintf("<|m%d|>", msgID),
		StartLine:    startLine,
		EndLine:      endLine,
		NumLines:     lineCount,
		NumTokens:    len(quotedTokens),
		CharStart:    charOffset,
		CharEnd:      charOffset + charLen,
	}

	return &QuoteResult{
		Success:    true,
		Result:     expandedText,
		Receipt:    receipt,
		Text:       expandedText,
		Tokens:     quotedTokens,
		Provenance: prov,
	}, nil
}

// QuoteMask computes the allowed next tokens at each step of a quote sequence.
func (cs *ContextStore) QuoteMask(prefix []string) (*QuoteMaskResult, error) {
	cs.mu.RLock()
	defer cs.mu.RUnlock()

	if !cs.anchorsEnabled {
		return nil, fmt.Errorf("ErrUnsupported: quote is not supported in anchor-free mode")
	}

	// Step 0: prefix has 1 element: ["<|quote|>"]
	if len(prefix) == 1 && prefix[0] == "<|quote|>" {
		var tokens []string
		var ids []int
		for i, seg := range cs.activeSlots {
			if seg != nil && seg.Active {
				mToken := fmt.Sprintf("<|m%d|>", i)
				tokens = append(tokens, mToken)
				ids = append(ids, M0TokenID+i)
			}
		}
		return &QuoteMaskResult{
			Success:         true,
			AllowedTokens:   tokens,
			AllowedTokenIDs: ids,
			Step:            "message",
		}, nil
	}

	// Step 1: prefix has 2 elements: ["<|quote|>", "<|mN|>"]
	if len(prefix) == 2 && prefix[0] == "<|quote|>" {
		mStr := strings.TrimSuffix(strings.TrimPrefix(prefix[1], "<|m"), "|>")
		mID, err := strconv.Atoi(mStr)
		if err != nil || mID < 0 || mID >= NumMessageTokens {
			return nil, fmt.Errorf("ErrSyntax: invalid message token %q", prefix[1])
		}
		seg := cs.activeSlots[mID]
		if seg == nil || !seg.Active {
			return nil, fmt.Errorf("ErrNotFound: message token <|m%d|> not found in active context", mID)
		}

		numLines := len(seg.Lines)
		tokens := make([]string, numLines)
		ids := make([]int, numLines)
		for i := 0; i < numLines; i++ {
			tokens[i] = fmt.Sprintf("<|L%d|>", i)
			ids[i] = L0TokenID + i
		}
		return &QuoteMaskResult{
			Success:         true,
			AllowedTokens:   tokens,
			AllowedTokenIDs: ids,
			Step:            "start_line",
		}, nil
	}

	// Step 2: prefix has 3 elements: ["<|quote|>", "<|mN|>", "<|La|>"]
	if len(prefix) == 3 && prefix[0] == "<|quote|>" {
		mStr := strings.TrimSuffix(strings.TrimPrefix(prefix[1], "<|m"), "|>")
		mID, err := strconv.Atoi(mStr)
		if err != nil || mID < 0 || mID >= NumMessageTokens {
			return nil, fmt.Errorf("ErrSyntax: invalid message token %q", prefix[1])
		}
		seg := cs.activeSlots[mID]
		if seg == nil || !seg.Active {
			return nil, fmt.Errorf("ErrNotFound: message token <|m%d|> not found in active context", mID)
		}

		lStr := strings.TrimSuffix(strings.TrimPrefix(prefix[2], "<|L"), "|>")
		startLine, err := strconv.Atoi(lStr)
		numLines := len(seg.Lines)
		if err != nil || startLine < 0 || startLine >= numLines {
			return nil, fmt.Errorf("ErrOutOfBounds: invalid start line index %q", prefix[2])
		}

		// End line must be >= startLine and < numLines
		count := numLines - startLine
		tokens := make([]string, count)
		ids := make([]int, count)
		for i := 0; i < count; i++ {
			lineIdx := startLine + i
			tokens[i] = fmt.Sprintf("<|L%d|>", lineIdx)
			ids[i] = L0TokenID + lineIdx
		}
		return &QuoteMaskResult{
			Success:         true,
			AllowedTokens:   tokens,
			AllowedTokenIDs: ids,
			Step:            "end_line",
		}, nil
	}

	// Step 3: prefix has 4 elements: ["<|quote|>", "<|mN|>", "<|La|>", "<|Lb|>"]
	if len(prefix) == 4 && prefix[0] == "<|quote|>" {
		closeID := cs.tok.TokenToID("<|/quote|>")
		if closeID == -1 {
			closeID = 34
		}
		return &QuoteMaskResult{
			Success:         true,
			AllowedTokens:   []string{"<|/quote|>"},
			AllowedTokenIDs: []int{closeID},
			Step:            "close",
		}, nil
	}

	return nil, fmt.Errorf("ErrSyntax: unrecognized quote prefix sequence: %v", prefix)
}

var findRegex = regexp.MustCompile(`^find\(\s*"([^"]+)"(?:\s*,\s*(\d+))?\s*\)$`)
var linesRegex = regexp.MustCompile(`^lines\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)$`)

// ParseFindOrLinesCall parses "find(\"keyword\")", "find(\"key\", 12)", or "lines(12, 10, 25)".
func ParseFindOrLinesCall(call string) (isLines bool, query string, msgID *int, startLine, endLine int, err error) {
	call = strings.TrimSpace(call)

	if match := linesRegex.FindStringSubmatch(call); match != nil {
		m, _ := strconv.Atoi(match[1])
		s, _ := strconv.Atoi(match[2])
		e, _ := strconv.Atoi(match[3])
		return true, "", &m, s, e, nil
	}

	if match := findRegex.FindStringSubmatch(call); match != nil {
		q := match[1]
		if match[2] != "" {
			m, _ := strconv.Atoi(match[2])
			return false, q, &m, 0, 0, nil
		}
		return false, q, nil, 0, 0, nil
	}

	// CLI fallbacks:
	// lines 12 10 25
	if strings.HasPrefix(call, "lines ") {
		parts := strings.Fields(strings.TrimPrefix(call, "lines "))
		if len(parts) == 3 {
			m, err1 := strconv.Atoi(parts[0])
			s, err2 := strconv.Atoi(parts[1])
			e, err3 := strconv.Atoi(parts[2])
			if err1 == nil && err2 == nil && err3 == nil {
				return true, "", &m, s, e, nil
			}
		}
	}

	// find "text" [msg]
	if strings.HasPrefix(call, "find ") {
		arg := strings.TrimSpace(strings.TrimPrefix(call, "find "))
		if strings.HasPrefix(arg, `"`) {
			idx := strings.Index(arg[1:], `"`)
			if idx != -1 {
				q := arg[1 : idx+1]
				rest := strings.TrimSpace(arg[idx+2:])
				if rest != "" {
					if m, err := strconv.Atoi(rest); err == nil {
						return false, q, &m, 0, 0, nil
					}
				}
				return false, q, nil, 0, 0, nil
			}
		} else {
			parts := strings.Fields(arg)
			if len(parts) == 1 {
				return false, parts[0], nil, 0, 0, nil
			}
			if len(parts) == 2 {
				if m, err := strconv.Atoi(parts[1]); err == nil {
					return false, parts[0], &m, 0, 0, nil
				}
			}
		}
	}

	return false, "", nil, 0, 0, fmt.Errorf("ErrSyntax: malformed find/lines syntax: %s", call)
}

// Find executes a normalized exact search across context messages with BM25 ranked fallback.
// Returns up to 3 matches with ±2 lines of context, returned as a newly anchored message.
func (cs *ContextStore) Find(query string, scopeMsgID *int) (*FindResult, error) {
	query = strings.TrimSpace(query)
	if query == "" {
		return nil, fmt.Errorf("ErrArgs: find query cannot be empty")
	}

	normQuery := NormalizeText(query)
	if normQuery == "" {
		return &FindResult{
			Success:    true,
			Result:     "no matches",
			NumMatches: 0,
		}, nil
	}

	cs.mu.Lock()
	defer cs.mu.Unlock()

	type matchExcerpt struct {
		segment   *MessageSegment
		startLine int
		endLine   int
	}

	var excerpts []matchExcerpt
	isExact := false

	var targets []*MessageSegment
	if scopeMsgID != nil {
		if *scopeMsgID < 0 || *scopeMsgID >= NumMessageTokens {
			return nil, fmt.Errorf("ErrArgs: message id must be in range 0..%d: got %d", NumMessageTokens-1, *scopeMsgID)
		}
		slotIdx := *scopeMsgID % NumMessageTokens
		seg := cs.activeSlots[slotIdx]
		if seg != nil && seg.Active {
			targets = []*MessageSegment{seg}
		} else {
			// Search history for matching msgID if not in active slot
			for _, h := range cs.history {
				if h != nil && h.MessageID == *scopeMsgID {
					targets = []*MessageSegment{h}
					break
				}
			}
		}
	} else {
		// Search historical conversation messages in chronological order (exclude derived search/lines excerpts to prevent duplicate results)
		for _, seg := range cs.history {
			if seg != nil && !seg.IsDerived {
				targets = append(targets, seg)
			}
		}
	}

	// 1. Normalized exact match first
	for _, seg := range targets {
		numLines := len(seg.Lines)
		var matchedLineIndices []int
		for lineIdx := range seg.Lines {
			normLine := ""
			if lineIdx < len(seg.NormLines) {
				normLine = seg.NormLines[lineIdx]
			} else {
				normLine = NormalizeText(seg.LineTexts[lineIdx])
			}
			if strings.Contains(normLine, normQuery) {
				matchedLineIndices = append(matchedLineIndices, lineIdx)
			}
		}

		if len(matchedLineIndices) == 0 {
			continue
		}

		isExact = true
		// Merge overlapping ranges (±2 lines)
		var mergedRanges [][2]int
		for _, idx := range matchedLineIndices {
			s := idx - 2
			if s < 0 {
				s = 0
			}
			e := idx + 2
			if e >= numLines {
				e = numLines - 1
			}

			if len(mergedRanges) == 0 {
				mergedRanges = append(mergedRanges, [2]int{s, e})
			} else {
				last := &mergedRanges[len(mergedRanges)-1]
				if s <= last[1]+1 {
					if e > last[1] {
						last[1] = e
					}
				} else {
					mergedRanges = append(mergedRanges, [2]int{s, e})
				}
			}
		}

		for _, r := range mergedRanges {
			excerpts = append(excerpts, matchExcerpt{
				segment:   seg,
				startLine: r[0],
				endLine:   r[1],
			})
			if len(excerpts) >= 3 {
				break
			}
		}
		if len(excerpts) >= 3 {
			break
		}
	}

	// 2. BM25 fallback only when there's no exact hit
	if !isExact && cs.bm25Index != nil {
		scored := cs.bm25Index.ScoreQuery(query, scopeMsgID)
		for _, sc := range scored {
			seg := sc.Chunk.Segment
			numLines := len(seg.Lines)
			s := sc.Chunk.LineIdx - 2
			if s < 0 {
				s = 0
			}
			e := sc.Chunk.LineIdx + 2
			if e >= numLines {
				e = numLines - 1
			}

			// Check if this chunk overlaps an existing excerpt from the same segment
			merged := false
			for i := range excerpts {
				if excerpts[i].segment == seg {
					if s <= excerpts[i].endLine+1 && e >= excerpts[i].startLine-1 {
						if s < excerpts[i].startLine {
							excerpts[i].startLine = s
						}
						if e > excerpts[i].endLine {
							excerpts[i].endLine = e
						}
						merged = true
						break
					}
				}
			}
			if !merged {
				excerpts = append(excerpts, matchExcerpt{
					segment:   seg,
					startLine: s,
					endLine:   e,
				})
				if len(excerpts) >= 3 {
					break
				}
			}
		}
	}

	if len(excerpts) == 0 {
		return &FindResult{
			Success:    true,
			Result:     "no matches",
			NumMatches: 0,
		}, nil
	}

	// 3. Labels: Each result in the observation says [exact] or [ranked]
	labelStr := "[ranked] "
	if isExact {
		labelStr = "[exact] "
	}
	labelTokens := cs.tok.Encode(labelStr)

	// Build raw token sequence for the excerpts
	var combinedTokens []int
	newlineTokenID := cs.tok.TokenToID("Ċ")
	if newlineTokenID == -1 {
		newlineTokenID = cs.tok.TokenToID("\n")
	}

	for i, exc := range excerpts {
		if i > 0 && newlineTokenID != -1 {
			combinedTokens = append(combinedTokens, newlineTokenID)
		}
		// Prepend label tokens to each result excerpt
		combinedTokens = append(combinedTokens, labelTokens...)
		for lineIdx := exc.startLine; lineIdx <= exc.endLine; lineIdx++ {
			combinedTokens = append(combinedTokens, exc.segment.Lines[lineIdx]...)
		}
	}

	if !cs.anchorsEnabled {
		newMsgID := cs.nextMsgID
		cs.indexPlainTokensLocked(combinedTokens, newMsgID, true)
		cs.nextMsgID = (cs.nextMsgID + 1) % NumMessageTokens
		resultStr := cs.tok.DecodeTokens(combinedTokens)
		return &FindResult{
			Success:    true,
			Result:     resultStr,
			MessageID:  newMsgID,
			Tokens:     combinedTokens,
			NumMatches: len(excerpts),
		}, nil
	}

	// Anchor this sequence as a NEW message with cs.nextMsgID
	anchoredTokens, _, nextMsg, err := cs.tok.InsertAnchors(combinedTokens, nil, cs.nextMsgID)
	if err != nil {
		return nil, err
	}
	newMsgID := cs.nextMsgID
	cs.indexSegmentsLocked(anchoredTokens, true)
	cs.nextMsgID = nextMsg

	resultStr := cs.tok.DecodeTokens(anchoredTokens)

	return &FindResult{
		Success:    true,
		Result:     resultStr,
		MessageID:  newMsgID,
		Tokens:     anchoredTokens,
		NumMatches: len(excerpts),
	}, nil
}

// Lines returns lines a..b of message msgID anchored as a new message.
func (cs *ContextStore) Lines(msgID int, startLine, endLine int) (*FindResult, error) {
	if msgID < 0 || msgID >= NumMessageTokens {
		return nil, fmt.Errorf("ErrArgs: message id must be in range 0..%d: got %d", NumMessageTokens-1, msgID)
	}
	cs.mu.Lock()
	defer cs.mu.Unlock()

	slotIdx := msgID % NumMessageTokens
	seg := cs.activeSlots[slotIdx]
	if seg == nil || !seg.Active || seg.MessageID != msgID {
		return nil, fmt.Errorf("ErrNotFound: message token <|m%d|> not found in active context", msgID)
	}

	if startLine < 0 || endLine < 0 {
		return nil, fmt.Errorf("ErrOutOfBounds: negative line index: start=%d end=%d", startLine, endLine)
	}
	if startLine > endLine {
		return nil, fmt.Errorf("ErrOutOfBounds: start line L%d > end line L%d", startLine, endLine)
	}
	numLines := len(seg.Lines)
	if endLine >= numLines {
		return nil, fmt.Errorf("ErrOutOfBounds: line index L%d out of bounds (message <|m%d|> has %d lines: L0..L%d)", endLine, msgID, numLines, numLines-1)
	}

	var tokens []int
	for i := startLine; i <= endLine; i++ {
		tokens = append(tokens, seg.Lines[i]...)
	}

	if !cs.anchorsEnabled {
		newMsgID := cs.nextMsgID
		cs.indexPlainTokensLocked(tokens, newMsgID, true)
		cs.nextMsgID = (cs.nextMsgID + 1) % NumMessageTokens
		resultStr := cs.tok.DecodeTokens(tokens)
		return &FindResult{
			Success:    true,
			Result:     resultStr,
			MessageID:  newMsgID,
			Tokens:     tokens,
			NumMatches: 1,
		}, nil
	}

	anchoredTokens, _, nextMsg, err := cs.tok.InsertAnchors(tokens, nil, cs.nextMsgID)
	if err != nil {
		return nil, err
	}
	newMsgID := cs.nextMsgID
	cs.indexSegmentsLocked(anchoredTokens, true)
	cs.nextMsgID = nextMsg

	resultStr := cs.tok.DecodeTokens(anchoredTokens)

	return &FindResult{
		Success:    true,
		Result:     resultStr,
		MessageID:  newMsgID,
		Tokens:     anchoredTokens,
		NumMatches: 1,
	}, nil
}

// DispatchContextTool dispatches anchor, quote, find, lines, quote_mask, context_reset operations.
func DispatchContextTool(cs *ContextStore, toolName, payload string) (string, error) {
	toolName = strings.TrimSpace(toolName)
	payload = strings.TrimSpace(payload)

	switch toolName {
	case "quote", "<|quote|>":
		res, err := cs.Quote(payload)
		if err != nil {
			return "", err
		}
		return res.Text, nil

	case "find", "<|call:find|>":
		isLines, q, scopeMsg, s, e, err := ParseFindOrLinesCall(payload)
		if err != nil {
			return "", err
		}
		if isLines {
			res, err := cs.Lines(*scopeMsg, s, e)
			if err != nil {
				return "", err
			}
			return res.Result, nil
		}
		res, err := cs.Find(q, scopeMsg)
		if err != nil {
			return "", err
		}
		return res.Result, nil

	case "lines":
		isLines, _, scopeMsg, s, e, err := ParseFindOrLinesCall(payload)
		if err != nil {
			return "", err
		}
		if !isLines || scopeMsg == nil {
			return "", fmt.Errorf("ErrSyntax: expected lines(msg, start, end)")
		}
		res, err := cs.Lines(*scopeMsg, s, e)
		if err != nil {
			return "", err
		}
		return res.Result, nil

	case "quote_mask":
		type maskReq struct {
			Prefix []string `json:"prefix"`
		}
		var req maskReq
		if err := json.Unmarshal([]byte(payload), &req); err == nil && len(req.Prefix) > 0 {
			res, err := cs.QuoteMask(req.Prefix)
			if err != nil {
				return "", err
			}
			data, _ := json.Marshal(res)
			return string(data), nil
		}
		// Try space-separated prefix
		fields := strings.Fields(payload)
		if len(fields) > 0 {
			res, err := cs.QuoteMask(fields)
			if err != nil {
				return "", err
			}
			data, _ := json.Marshal(res)
			return string(data), nil
		}
		return "", fmt.Errorf("ErrSyntax: malformed quote_mask payload")

	case "context_reset":
		cs.Reset()
		return "OK", nil

	default:
		return "", fmt.Errorf("ErrSyntax: unknown context tool: %s", toolName)
	}
}
