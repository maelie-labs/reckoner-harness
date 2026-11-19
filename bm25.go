package main

import (
	"math"
	"sort"
	"strings"
	"unicode"

	"golang.org/x/text/unicode/norm"
)

// NormalizeText normalizes text for exact and BM25 matching according to SPEC §6.4:
// 1. Unicode NFKC normalization
// 2. Folding quotation marks (curly single/double quotes to straight quotes)
// 3. Folding dash variants to ASCII hyphen '-'
// 4. Removing thousands separators inside digit groups (e.g. 1,250 -> 1250)
// 5. Converting whitespace runs to a single ASCII space and trimming ends
// 6. Case folding via Unicode-aware lowercasing
func NormalizeText(s string) string {
	// 1. Unicode NFKC
	s = norm.NFKC.String(s)

	// 2. Fold quotes, 3. Fold dashes, 4. Remove thousands separators
	runes := []rune(s)
	var out []rune
	for i := 0; i < len(runes); i++ {
		r := runes[i]
		// Remove thousands separator inside digit groups: 1,250 -> 1250
		if r == ',' && i > 0 && i+1 < len(runes) && unicode.IsDigit(runes[i-1]) && unicode.IsDigit(runes[i+1]) {
			continue
		}
		// Curly single quotes: ‘, ’, ‚, ‛ -> '
		switch r {
		case '\u2018', '\u2019', '\u201a', '\u201b':
			r = '\''
		// Curly double quotes: “, ”, „, ‟ -> "
		case '\u201c', '\u201d', '\u201e', '\u201f':
			r = '"'
		// Dash variants: ‐, ‑, ‒, –, —, ―, − -> -
		case '\u2010', '\u2011', '\u2012', '\u2013', '\u2014', '\u2015', '\u2212':
			r = '-'
		}
		out = append(out, r)
	}

	// 5. Whitespace runs to one ASCII space
	var wsClean []rune
	inWhitespace := false
	for _, r := range out {
		if unicode.IsSpace(r) {
			if !inWhitespace {
				wsClean = append(wsClean, ' ')
				inWhitespace = true
			}
		} else {
			wsClean = append(wsClean, r)
			inWhitespace = false
		}
	}
	s = strings.TrimSpace(string(wsClean))

	// 6. Case folding
	return strings.ToLower(s)
}

// TokenizeBM25 splits normalized text on runs of Unicode letters, digits, and '_',
// plus '.' that sits between two digits (e.g. 3.14, v1.2.3).
// No stemming, no stop words.
func TokenizeBM25(normText string) []string {
	var tokens []string
	runes := []rune(normText)
	var cur []rune
	n := len(runes)

	for i := 0; i < n; i++ {
		r := runes[i]
		isTokenRune := unicode.IsLetter(r) || unicode.IsDigit(r) || r == '_'
		if !isTokenRune && r == '.' {
			if i > 0 && unicode.IsDigit(runes[i-1]) && i+1 < n && unicode.IsDigit(runes[i+1]) {
				isTokenRune = true
			}
		}

		if isTokenRune {
			cur = append(cur, r)
		} else {
			if len(cur) > 0 {
				tokens = append(tokens, string(cur))
				cur = nil
			}
		}
	}
	if len(cur) > 0 {
		tokens = append(tokens, string(cur))
	}
	return tokens
}

// UniqueStrings returns unique elements of a string slice in order of appearance.
func UniqueStrings(slice []string) []string {
	seen := make(map[string]struct{}, len(slice))
	var out []string
	for _, s := range slice {
		if _, ok := seen[s]; !ok {
			seen[s] = struct{}{}
			out = append(out, s)
		}
	}
	return out
}

// BM25Chunk represents an indexed line chunk.
type BM25Chunk struct {
	ChunkID   int
	MsgID     int
	TurnID    int
	LineIdx   int
	Length    int
	TermFreqs map[string]int
	Segment   *MessageSegment
}

// BM25Posting stores posting information for a term occurrence in a chunk.
type BM25Posting struct {
	ChunkID int
	Freq    int
}

// BM25Index manages an incremental index over message line chunks for BM25 ranking.
type BM25Index struct {
	chunks      []*BM25Chunk
	postings    map[string][]BM25Posting
	termDocFreq map[string]int
	totalTokens int
	msgChunks   map[int][]*BM25Chunk
}

// NewBM25Index creates an empty BM25 incremental index.
func NewBM25Index() *BM25Index {
	return &BM25Index{
		postings:    make(map[string][]BM25Posting),
		termDocFreq: make(map[string]int),
		msgChunks:   make(map[int][]*BM25Chunk),
	}
}

// Reset clears the BM25 index.
func (idx *BM25Index) Reset() {
	idx.chunks = nil
	idx.postings = make(map[string][]BM25Posting)
	idx.termDocFreq = make(map[string]int)
	idx.totalTokens = 0
	idx.msgChunks = make(map[int][]*BM25Chunk)
}

// Rebuild clears and completely re-indexes all non-derived segments from history.
func (idx *BM25Index) Rebuild(history []*MessageSegment) {
	idx.Reset()
	for _, seg := range history {
		if seg != nil && !seg.IsDerived {
			idx.AddSegment(seg)
		}
	}
}

// AddSegment incrementally indexes all lines of a message segment.
func (idx *BM25Index) AddSegment(seg *MessageSegment) {
	if seg == nil || seg.IsDerived {
		return
	}

	for lineIdx := range seg.LineTexts {
		normLine := ""
		if lineIdx < len(seg.NormLines) {
			normLine = seg.NormLines[lineIdx]
		} else {
			normLine = NormalizeText(seg.LineTexts[lineIdx])
		}

		terms := TokenizeBM25(normLine)
		tfMap := make(map[string]int)
		for _, t := range terms {
			tfMap[t]++
		}

		chunkID := len(idx.chunks)
		chunk := &BM25Chunk{
			ChunkID:   chunkID,
			MsgID:     seg.MessageID,
			TurnID:    seg.TurnID,
			LineIdx:   lineIdx,
			Length:    len(terms),
			TermFreqs: tfMap,
			Segment:   seg,
		}

		idx.chunks = append(idx.chunks, chunk)
		idx.totalTokens += len(terms)
		idx.msgChunks[seg.MessageID] = append(idx.msgChunks[seg.MessageID], chunk)

		for t, freq := range tfMap {
			idx.postings[t] = append(idx.postings[t], BM25Posting{
				ChunkID: chunkID,
				Freq:    freq,
			})
			idx.termDocFreq[t]++
		}
	}
}

// ScoredChunk pairs a chunk with its computed BM25 score.
type ScoredChunk struct {
	Chunk *BM25Chunk
	Score float64
}

// ScoreQuery computes BM25 scores for query across chunks in scope.
// BM25 parameters: k1 = 1.2, b = 0.75
// IDF = ln(1 + (N - n + 0.5)/(n + 0.5))
func (idx *BM25Index) ScoreQuery(query string, scopeMsgID *int) []ScoredChunk {
	normQuery := NormalizeText(query)
	queryTokens := TokenizeBM25(normQuery)
	uniqueTerms := UniqueStrings(queryTokens)
	if len(uniqueTerms) == 0 {
		return nil
	}

	const (
		k1 = 1.2
		b  = 0.75
	)

	if scopeMsgID == nil {
		N := len(idx.chunks)
		if N == 0 || idx.totalTokens == 0 {
			return nil
		}
		avgdl := float64(idx.totalTokens) / float64(N)

		// Accumulate scores for chunks containing query terms
		scores := make([]float64, N)
		var seenChunks []int

		for _, t := range uniqueTerms {
			n := idx.termDocFreq[t]
			if n == 0 {
				continue
			}

			idf := math.Log(1.0 + (float64(N-n)+0.5)/(float64(n)+0.5))
			postings := idx.postings[t]

			for _, p := range postings {
				if scores[p.ChunkID] == 0 {
					seenChunks = append(seenChunks, p.ChunkID)
				}
				chunk := idx.chunks[p.ChunkID]
				tf := float64(p.Freq)
				docLen := float64(chunk.Length)

				denom := tf + k1*(1.0-b+b*(docLen/avgdl))
				termScore := idf * (tf * (k1 + 1.0)) / denom
				scores[p.ChunkID] += termScore
			}
		}

		var scored []ScoredChunk
		for _, chunkID := range seenChunks {
			score := scores[chunkID]
			if score > 0 {
				scored = append(scored, ScoredChunk{
					Chunk: idx.chunks[chunkID],
					Score: score,
				})
			}
		}

		// Sort by score desc, then earlier message (TurnID asc), then earlier line (LineIdx asc)
		sort.Slice(scored, func(i, j int) bool {
			diff := scored[i].Score - scored[j].Score
			if math.Abs(diff) > 1e-13 {
				return scored[i].Score > scored[j].Score
			}
			if scored[i].Chunk.TurnID != scored[j].Chunk.TurnID {
				return scored[i].Chunk.TurnID < scored[j].Chunk.TurnID
			}
			return scored[i].Chunk.LineIdx < scored[j].Chunk.LineIdx
		})

		return scored
	}

	// Scoped query for a specific message
	scoped := idx.msgChunks[*scopeMsgID]
	N := len(scoped)
	if N == 0 {
		return nil
	}

	totalScopedTokens := 0
	for _, c := range scoped {
		totalScopedTokens += c.Length
	}
	if totalScopedTokens == 0 {
		return nil
	}
	avgdl := float64(totalScopedTokens) / float64(N)

	scores := make(map[int]float64)
	for _, t := range uniqueTerms {
		// Document frequency within scoped message chunks
		n := 0
		for _, c := range scoped {
			if c.TermFreqs[t] > 0 {
				n++
			}
		}
		if n == 0 {
			continue
		}

		idf := math.Log(1.0 + (float64(N-n)+0.5)/(float64(n)+0.5))
		for _, c := range scoped {
			freq := c.TermFreqs[t]
			if freq > 0 {
				tf := float64(freq)
				docLen := float64(c.Length)
				denom := tf + k1*(1.0-b+b*(docLen/avgdl))
				termScore := idf * (tf * (k1 + 1.0)) / denom
				scores[c.ChunkID] += termScore
			}
		}
	}

	var scored []ScoredChunk
	for _, c := range scoped {
		if s, ok := scores[c.ChunkID]; ok && s > 0 {
			scored = append(scored, ScoredChunk{
				Chunk: c,
				Score: s,
			})
		}
	}

	sort.Slice(scored, func(i, j int) bool {
		diff := scored[i].Score - scored[j].Score
		if math.Abs(diff) > 1e-13 {
			return scored[i].Score > scored[j].Score
		}
		if scored[i].Chunk.TurnID != scored[j].Chunk.TurnID {
			return scored[i].Chunk.TurnID < scored[j].Chunk.TurnID
		}
		return scored[i].Chunk.LineIdx < scored[j].Chunk.LineIdx
	})

	return scored
}
