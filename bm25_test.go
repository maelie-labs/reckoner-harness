package main

import (
	"math"
	"strings"
	"testing"
)

func TestNormalizationRules(t *testing.T) {
	cases := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "NFKC ligature and fullwidth",
			input:    "ﬁle １２３",
			expected: "file 123",
		},
		{
			name:     "Case folding",
			input:    "INVOICE Ch_9812",
			expected: "invoice ch_9812",
		},
		{
			name:     "Thousands separator removal",
			input:    "Total: $1,250 and 1,000,000 items, not 1, 2, 3",
			expected: "total: $1250 and 1000000 items, not 1, 2, 3",
		},
		{
			name:     "Curly single quotes to straight",
			input:    "‘hello’ and ’world’ and ‚quote‛",
			expected: "'hello' and 'world' and 'quote'",
		},
		{
			name:     "Curly double quotes to straight",
			input:    "“hello” and ”world” and „test‟",
			expected: "\"hello\" and \"world\" and \"test\"",
		},
		{
			name:     "Dash variants to ASCII hyphen",
			input:    "a‐b‑c‒d–e—f―g−h",
			expected: "a-b-c-d-e-f-g-h",
		},
		{
			name:     "Whitespace runs to single space",
			input:    "  hello \t\n  world   test  ",
			expected: "hello world test",
		},
		{
			name:     "Combined normalization",
			input:    "  “Invoice” ch—9812:   $1,250   ",
			expected: "\"invoice\" ch-9812: $1250",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := NormalizeText(tc.input)
			if got != tc.expected {
				t.Errorf("NormalizeText(%q) = %q, expected %q", tc.input, got, tc.expected)
			}
		})
	}
}

func TestTokenizeBM25(t *testing.T) {
	cases := []struct {
		input    string
		expected []string
	}{
		{
			input:    "ch_9812",
			expected: []string{"ch_9812"},
		},
		{
			input:    "Invoice ch_9812 Amount: $500",
			expected: []string{"invoice", "ch_9812", "amount", "500"},
		},
		{
			input:    "pi is 3.14 approximately",
			expected: []string{"pi", "is", "3.14", "approximately"},
		},
		{
			input:    "release v1.2.3-final",
			expected: []string{"release", "v1.2.3", "final"},
		},
		{
			input:    "3 and 14 separate numbers",
			expected: []string{"3", "and", "14", "separate", "numbers"},
		},
		{
			input:    "sentence with a period. and ... ellipses.",
			expected: []string{"sentence", "with", "a", "period", "and", "ellipses"},
		},
	}

	for _, tc := range cases {
		norm := NormalizeText(tc.input)
		got := TokenizeBM25(norm)
		if len(got) != len(tc.expected) {
			t.Fatalf("TokenizeBM25(%q) length mismatch: got %v, expected %v", tc.input, got, tc.expected)
		}
		for i := range got {
			if got[i] != tc.expected[i] {
				t.Errorf("TokenizeBM25(%q)[%d] = %q, expected %q", tc.input, i, got[i], tc.expected[i])
			}
		}
	}
}

func TestBM25EdgeCases(t *testing.T) {
	tok, err := GetGlobalTokenizer()
	if err != nil {
		t.Fatalf("Tokenizer error: %v", err)
	}

	// (a) context has ch_9812, query ch_9821 -> no matches
	{
		cs := NewContextStore(tok)
		tokens := tok.Encode("Invoice ch_9812\nAmount: $500")
		cs.StoreMessage(tokens, nil, nil)
		res, err := cs.Find("ch_9821", nil)
		if err != nil {
			t.Fatalf("Find failed: %v", err)
		}
		if res.Result != "no matches" || res.NumMatches != 0 {
			t.Errorf("Case (a) expected 'no matches', got result=%q, matches=%d", res.Result, res.NumMatches)
		}
	}

	// (b) query 3.14 doesn't match a line with only 3 and 14
	{
		cs := NewContextStore(tok)
		tokens := tok.Encode("values 3 and 14 are separate\nother unrelated text")
		cs.StoreMessage(tokens, nil, nil)
		res, err := cs.Find("3.14", nil)
		if err != nil {
			t.Fatalf("Find failed: %v", err)
		}
		if res.Result != "no matches" || res.NumMatches != 0 {
			t.Errorf("Case (b) expected 'no matches', got result=%q, matches=%d", res.Result, res.NumMatches)
		}
	}

	// (c) a multi-word query still ranks as before
	{
		cs := NewContextStore(tok)
		tokens := tok.Encode("feature v1.2.3 released\nuser_id_100 active\nother line")
		cs.StoreMessage(tokens, nil, nil)
		res, err := cs.Find("user_id_100 v1.2.3", nil)
		if err != nil {
			t.Fatalf("Find failed: %v", err)
		}
		if res.NumMatches != 1 || !strings.Contains(res.Result, "[ranked]") {
			t.Errorf("Case (c) expected 1 ranked merged match, got result=%q, matches=%d", res.Result, res.NumMatches)
		}
	}
}

func TestBM25HandCheckedFiveLine(t *testing.T) {
	// Hand-checked 5-line example:
	// Line 0: "cat dog cat"       (tokens: cat, dog, cat; len=3)
	// Line 1: "dog fish"          (tokens: dog, fish; len=2)
	// Line 2: "cat bird"          (tokens: cat, bird; len=2)
	// Line 3: "fish bird fish"    (tokens: fish, bird, fish; len=3)
	// Line 4: "frog"              (tokens: frog; len=1)
	//
	// Total tokens = 3 + 2 + 2 + 3 + 1 = 11
	// N = 5, avgdl = 11 / 5 = 2.2
	//
	// Query: "cat fish"
	// n(cat) = 2 (lines 0, 2)
	// n(fish) = 2 (lines 1, 3)
	// IDF(cat) = IDF(fish) = ln(1 + (5 - 2 + 0.5) / (2 + 0.5)) = ln(1 + 3.5 / 2.5) = ln(2.4)
	// ln(2.4) in float64 = 0.8754687373538999
	//
	// Expected float64 values:
	// TF(cat, line 0) = (2 * 2.2) / (2 + 1.2 * (0.25 + 0.75 * 3 / 2.2)) = 4.4 / (2 + 16.8/11) = 48.4 / 38.8 = 121 / 97
	// Score(line 0) = ln(2.4) * (121 / 97)
	//
	// TF(fish, line 1) = (1 * 2.2) / (1 + 1.2 * (0.25 + 0.75 * 2 / 2.2)) = 2.2 / (1 + 123/110) = 242 / 233
	// Score(line 1) = ln(2.4) * (242 / 233)
	//
	// Score(line 2) = ln(2.4) * (242 / 233)  [tied with line 1; line 1 wins tie-break]
	// Score(line 3) = ln(2.4) * (121 / 97)   [tied with line 0; line 0 wins tie-break]
	// Score(line 4) = 0.0

	expectedIDF := math.Log(2.4)
	expectedScore0 := expectedIDF * (121.0 / 97.0)
	expectedScore1 := expectedIDF * (242.0 / 233.0)
	expectedScore2 := expectedIDF * (242.0 / 233.0)
	expectedScore3 := expectedIDF * (121.0 / 97.0)

	lines := []string{
		"cat dog cat",
		"dog fish",
		"cat bird",
		"fish bird fish",
		"frog",
	}

	seg := &MessageSegment{
		MessageID: 0,
		TurnID:    1,
		Active:    true,
		LineTexts: lines,
		NormLines: []string{
			NormalizeText(lines[0]),
			NormalizeText(lines[1]),
			NormalizeText(lines[2]),
			NormalizeText(lines[3]),
			NormalizeText(lines[4]),
		},
	}

	idx := NewBM25Index()
	idx.AddSegment(seg)

	scored := idx.ScoreQuery("cat fish", nil)

	// We expect 4 lines with score > 0 (line 0, 3, 1, 2)
	if len(scored) != 4 {
		t.Fatalf("Expected 4 scored chunks, got %d", len(scored))
	}

	const tol = 1e-12

	// Check Rank 1: Line 0
	if scored[0].Chunk.LineIdx != 0 {
		t.Errorf("Rank 0: expected line 0, got line %d", scored[0].Chunk.LineIdx)
	}
	if math.Abs(scored[0].Score-expectedScore0) > tol {
		t.Errorf("Rank 0 score: expected %.15f, got %.15f (diff: %e)", expectedScore0, scored[0].Score, math.Abs(scored[0].Score-expectedScore0))
	}

	// Check Rank 2: Line 3 (tie with line 0, but line 0 appeared earlier)
	if scored[1].Chunk.LineIdx != 3 {
		t.Errorf("Rank 1: expected line 3, got line %d", scored[1].Chunk.LineIdx)
	}
	if math.Abs(scored[1].Score-expectedScore3) > tol {
		t.Errorf("Rank 1 score: expected %.15f, got %.15f (diff: %e)", expectedScore3, scored[1].Score, math.Abs(scored[1].Score-expectedScore3))
	}

	// Check Rank 3: Line 1 (tie with line 2, but line 1 appeared earlier)
	if scored[2].Chunk.LineIdx != 1 {
		t.Errorf("Rank 2: expected line 1, got line %d", scored[2].Chunk.LineIdx)
	}
	if math.Abs(scored[2].Score-expectedScore1) > tol {
		t.Errorf("Rank 2 score: expected %.15f, got %.15f (diff: %e)", expectedScore1, scored[2].Score, math.Abs(scored[2].Score-expectedScore1))
	}

	// Check Rank 4: Line 2
	if scored[3].Chunk.LineIdx != 2 {
		t.Errorf("Rank 3: expected line 2, got line %d", scored[3].Chunk.LineIdx)
	}
	if math.Abs(scored[3].Score-expectedScore2) > tol {
		t.Errorf("Rank 3 score: expected %.15f, got %.15f (diff: %e)", expectedScore2, scored[3].Score, math.Abs(scored[3].Score-expectedScore2))
	}
}

func TestBM25CrossMessageTieBreaking(t *testing.T) {
	// Line in Message 0 (Turn 1) vs Line in Message 1 (Turn 2) with exact same score
	seg0 := &MessageSegment{
		MessageID: 0,
		TurnID:    1,
		Active:    true,
		LineTexts: []string{"identical token line"},
		NormLines: []string{"identical token line"},
	}
	seg1 := &MessageSegment{
		MessageID: 1,
		TurnID:    2,
		Active:    true,
		LineTexts: []string{"identical token line"},
		NormLines: []string{"identical token line"},
	}

	idx := NewBM25Index()
	idx.AddSegment(seg0)
	idx.AddSegment(seg1)

	scored := idx.ScoreQuery("identical", nil)
	if len(scored) != 2 {
		t.Fatalf("Expected 2 scored chunks, got %d", len(scored))
	}
	if scored[0].Chunk.MsgID != 0 {
		t.Errorf("Earlier message should win tie-break: expected msg 0, got msg %d", scored[0].Chunk.MsgID)
	}
	if scored[1].Chunk.MsgID != 1 {
		t.Errorf("Later message should be 2nd: expected msg 1, got msg %d", scored[1].Chunk.MsgID)
	}
}

func TestIncrementalVsRebuild(t *testing.T) {
	vocab := []string{"alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet"}

	// Pseudo-random deterministic generator without crypto/rand or math/rand dependency
	seed := uint64(123456789)
	nextRand := func(max int) int {
		seed = seed*6364136223846793005 + 1442695040888963407
		return int((seed >> 33) % uint64(max))
	}

	for trial := 0; trial < 1000; trial++ {
		numMessages := 1 + nextRand(5)
		var history []*MessageSegment
		incIdx := NewBM25Index()

		for m := 0; m < numMessages; m++ {
			numLines := 1 + nextRand(6)
			var lines []string
			var normLines []string
			for l := 0; l < numLines; l++ {
				numWords := 1 + nextRand(5)
				var words []string
				for w := 0; w < numWords; w++ {
					words = append(words, vocab[nextRand(len(vocab))])
				}
				lineStr := ""
				for i, w := range words {
					if i > 0 {
						lineStr += " "
					}
					lineStr += w
				}
				lines = append(lines, lineStr)
				normLines = append(normLines, NormalizeText(lineStr))
			}

			seg := &MessageSegment{
				MessageID: m,
				TurnID:    m + 1,
				Active:    true,
				LineTexts: lines,
				NormLines: normLines,
			}
			history = append(history, seg)
			incIdx.AddSegment(seg)
		}

		rebuildIdx := NewBM25Index()
		rebuildIdx.Rebuild(history)

		// Compare total stats
		if incIdx.totalTokens != rebuildIdx.totalTokens {
			t.Fatalf("Trial %d: totalTokens mismatch: inc=%d rebuild=%d", trial, incIdx.totalTokens, rebuildIdx.totalTokens)
		}
		if len(incIdx.chunks) != len(rebuildIdx.chunks) {
			t.Fatalf("Trial %d: chunks count mismatch: inc=%d rebuild=%d", trial, len(incIdx.chunks), len(rebuildIdx.chunks))
		}

		// Query random term
		queryTerm := vocab[nextRand(len(vocab))]
		incScored := incIdx.ScoreQuery(queryTerm, nil)
		rebuildScored := rebuildIdx.ScoreQuery(queryTerm, nil)

		if len(incScored) != len(rebuildScored) {
			t.Fatalf("Trial %d: scored length mismatch for query %q: inc=%d rebuild=%d", trial, queryTerm, len(incScored), len(rebuildScored))
		}

		for i := range incScored {
			if incScored[i].Chunk.ChunkID != rebuildScored[i].Chunk.ChunkID {
				t.Fatalf("Trial %d rank %d: chunk ID mismatch: inc=%d rebuild=%d", trial, i, incScored[i].Chunk.ChunkID, rebuildScored[i].Chunk.ChunkID)
			}
			if math.Abs(incScored[i].Score-rebuildScored[i].Score) > 1e-12 {
				t.Fatalf("Trial %d rank %d: score mismatch: inc=%.15f rebuild=%.15f", trial, i, incScored[i].Score, rebuildScored[i].Score)
			}
		}
	}
}

func setupBenchmarkContext(numTokens int) (*ContextStore, error) {
	tok, err := GetGlobalTokenizer()
	if err != nil {
		return nil, err
	}
	cs := NewContextStore(tok)

	// Build realistic text passages of ~numTokens tokens
	// Sentences of ~10 tokens each
	vocab := []string{
		"algorithm", "binary", "computation", "deterministic", "execution",
		"function", "graph", "heuristic", "iteration", "kernel",
		"matrix", "network", "optimization", "polynomial", "quadratic",
	}

	curTokens := 0
	msgIdx := 0
	for curTokens < numTokens {
		// Each message has ~10 lines
		var lines []string
		for l := 0; l < 10 && curTokens < numTokens; l++ {
			w1 := vocab[(curTokens+1)%len(vocab)]
			w2 := vocab[(curTokens+3)%len(vocab)]
			w3 := vocab[(curTokens+7)%len(vocab)]
			lines = append(lines, "item "+w1+" and "+w2+" processing "+w3+" step")
			curTokens += 10
		}
		text := ""
		for i, line := range lines {
			if i > 0 {
				text += "\n"
			}
			text += line
		}
		toks := tok.Encode(text)
		cs.StoreMessage(toks, nil, &msgIdx)
		msgIdx++
	}
	return cs, nil
}

func BenchmarkFind_8K_Exact(b *testing.B) {
	cs, err := setupBenchmarkContext(8192)
	if err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res, err := cs.Find("computation", nil)
		if err != nil || res.NumMatches == 0 {
			b.Fatalf("failed: %v, res: %+v", err, res)
		}
	}
}

func BenchmarkFind_8K_BM25(b *testing.B) {
	cs, err := setupBenchmarkContext(8192)
	if err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res, err := cs.Find("computation matrix missingword", nil)
		if err != nil || res.NumMatches == 0 {
			b.Fatalf("failed: %v, res: %+v", err, res)
		}
	}
}

func BenchmarkFind_64K_Exact(b *testing.B) {
	cs, err := setupBenchmarkContext(65536)
	if err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res, err := cs.Find("computation", nil)
		if err != nil || res.NumMatches == 0 {
			b.Fatalf("failed: %v, res: %+v", err, res)
		}
	}
}

func BenchmarkFind_64K_BM25(b *testing.B) {
	cs, err := setupBenchmarkContext(65536)
	if err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res, err := cs.Find("computation matrix missingword", nil)
		if err != nil || res.NumMatches == 0 {
			b.Fatalf("failed: %v, res: %+v", err, res)
		}
	}
}

func BenchmarkFind_256K_Exact(b *testing.B) {
	cs, err := setupBenchmarkContext(262144)
	if err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res, err := cs.Find("computation", nil)
		if err != nil || res.NumMatches == 0 {
			b.Fatalf("failed: %v, res: %+v", err, res)
		}
	}
}

func BenchmarkFind_256K_BM25(b *testing.B) {
	cs, err := setupBenchmarkContext(262144)
	if err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res, err := cs.Find("computation matrix missingword", nil)
		if err != nil || res.NumMatches == 0 {
			b.Fatalf("failed: %v, res: %+v", err, res)
		}
	}
}

