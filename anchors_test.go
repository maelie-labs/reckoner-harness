package main

import (
	"strings"
	"testing"
)

func TestTokenizerLoading(t *testing.T) {
	tok, err := GetGlobalTokenizer()
	if err != nil {
		t.Fatalf("Failed to load global tokenizer: %v", err)
	}
	if len(tok.Vocab) != VocabSize {
		t.Errorf("Expected vocab size %d, got %d", VocabSize, len(tok.Vocab))
	}
	if len(tok.InvVocab) != VocabSize {
		t.Errorf("Expected inv vocab size %d, got %d", VocabSize, len(tok.InvVocab))
	}

	// Verify key special tokens
	tests := []struct {
		token string
		id    int
	}{
		{"<|m0|>", M0TokenID},
		{"<|m255|>", M255TokenID},
		{"<|L0|>", L0TokenID},
		{"<|L1023|>", L1023TokenID},
		{"<|call:calc|>", 18},
		{"<|call:find|>", 32},
		{"<|quote|>", 33},
		{"<|/quote|>", 34},
	}

	for _, tc := range tests {
		gotID := tok.TokenToID(tc.token)
		if gotID != tc.id {
			t.Errorf("TokenToID(%s): expected %d, got %d", tc.token, tc.id, gotID)
		}
		gotStr := tok.IDToToken(tc.id)
		if gotStr != tc.token {
			t.Errorf("IDToToken(%d): expected %s, got %s", tc.id, tc.token, gotStr)
		}
	}
}

func TestByteLevelDecoder(t *testing.T) {
	tok, err := GetGlobalTokenizer()
	if err != nil {
		t.Fatalf("Failed to load global tokenizer: %v", err)
	}

	m0ID := tok.TokenToID("<|m0|>")
	l0ID := tok.TokenToID("<|L0|>")
	decoded := tok.DecodeTokens([]int{m0ID, l0ID})
	if decoded != "<|m0|><|L0|>" {
		t.Errorf("Expected '<|m0|><|L0|>', got %q", decoded)
	}
}

func TestInsertAnchorsBasic(t *testing.T) {
	tok, err := GetGlobalTokenizer()
	if err != nil {
		t.Fatalf("Failed to load global tokenizer: %v", err)
	}

	// Empty tokens
	outTokens, outTargets, nextMsg, err := tok.InsertAnchors([]int{}, nil, 0)
	if err != nil || len(outTokens) != 0 || len(outTargets) != 0 || nextMsg != 0 {
		t.Errorf("Empty tokens failed: %v, %v, %d, err=%v", outTokens, outTargets, nextMsg, err)
	}

	// Single token
	dummyToken := 500
	outTokens, outTargets, nextMsg, err = tok.InsertAnchors([]int{dummyToken}, nil, 0)
	if err != nil {
		t.Fatalf("Single token insert failed: %v", err)
	}
	// Expected: <|m0|>, <|L0|>, dummyToken
	expectedTokens := []int{M0TokenID, L0TokenID, dummyToken}
	expectedTargets := []int{IgnoreIndex, IgnoreIndex, dummyToken}

	if len(outTokens) != len(expectedTokens) {
		t.Fatalf("Length mismatch: got %d, expected %d", len(outTokens), len(expectedTokens))
	}
	for i := range expectedTokens {
		if outTokens[i] != expectedTokens[i] {
			t.Errorf("Token %d: got %d, expected %d", i, outTokens[i], expectedTokens[i])
		}
		if outTargets[i] != expectedTargets[i] {
			t.Errorf("Target %d: got %d, expected %d", i, outTargets[i], expectedTargets[i])
		}
	}
	if nextMsg != 1 {
		t.Errorf("Expected nextMsg 1, got %d", nextMsg)
	}
}

func TestInsertAnchorsContinuationPast1024(t *testing.T) {
	tok, err := GetGlobalTokenizer()
	if err != nil {
		t.Fatalf("Failed to load global tokenizer: %v", err)
	}

	// Find a newline token ID
	newlineID := -1
	for id, isNL := range tok.NewlineTokens {
		if isNL {
			newlineID = id
			break
		}
	}
	if newlineID == -1 {
		t.Fatal("Could not find a newline token")
	}

	// Create 1,050 newline tokens -> should create 1,051 lines
	// Lines 0..1023 under m0, then line 0..26 under m1!
	numNewlines := 1050
	tokens := make([]int, numNewlines)
	for i := range tokens {
		tokens[i] = newlineID
	}

	outTokens, outTargets, nextMsg, err := tok.InsertAnchors(tokens, nil, 0)
	if err != nil {
		t.Fatalf("InsertAnchors continuation failed: %v", err)
	}

	// Check that m0 was emitted at start, and m1 was emitted when curLineIdx hit 1024
	foundM0 := false
	foundM1 := false
	m0Idx := -1
	m1Idx := -1

	for idx, tid := range outTokens {
		if tid == M0TokenID {
			foundM0 = true
			m0Idx = idx
		}
		if tid == M0TokenID+1 {
			foundM1 = true
			m1Idx = idx
		}
	}

	if !foundM0 || m0Idx != 0 {
		t.Errorf("Expected m0 at start, found at %d", m0Idx)
	}
	if !foundM1 || m1Idx <= m0Idx {
		t.Errorf("Expected m1 continuation token after m0, found m1 at %d, m0 at %d", m1Idx, m0Idx)
	}
	if nextMsg != 2 {
		t.Errorf("Expected nextMsg 2, got %d", nextMsg)
	}

	// Verify all anchor targets are masked to -100
	for i, tid := range outTokens {
		if tok.IsAnchorToken(tid) {
			if outTargets[i] != IgnoreIndex {
				t.Errorf("Anchor token at %d (%d) was not masked to -100: got %d", i, tid, outTargets[i])
			}
		}
	}
}

func TestInsertAnchorsRecyclingPast256(t *testing.T) {
	tok, err := GetGlobalTokenizer()
	if err != nil {
		t.Fatalf("Failed to load global tokenizer: %v", err)
	}

	tokens := []int{100, 200}
	outTokens, _, nextMsg, err := tok.InsertAnchors(tokens, nil, 255)
	if err != nil {
		t.Fatalf("InsertAnchors failed: %v", err)
	}
	if outTokens[0] != M0TokenID+255 {
		t.Errorf("Expected m255 (%d), got %d", M0TokenID+255, outTokens[0])
	}
	if nextMsg != 0 {
		t.Errorf("Expected nextMsg 0 after 255, got %d", nextMsg)
	}

	// Start at 256
	outTokens2, _, nextMsg2, err := tok.InsertAnchors(tokens, nil, 256)
	if err != nil {
		t.Fatalf("InsertAnchors failed: %v", err)
	}
	if outTokens2[0] != M0TokenID {
		t.Errorf("Expected m0 (%d) for messageID 256, got %d", M0TokenID, outTokens2[0])
	}
	if nextMsg2 != 1 {
		t.Errorf("Expected nextMsg 1, got %d", nextMsg2)
	}
}

func TestInsertAnchorsInvalidInputs(t *testing.T) {
	tok, err := GetGlobalTokenizer()
	if err != nil {
		t.Fatalf("Failed to load global tokenizer: %v", err)
	}

	// Length mismatch
	_, _, _, err = tok.InsertAnchors([]int{1, 2}, []int{1}, 0)
	if err == nil || !strings.Contains(err.Error(), "length mismatch") {
		t.Errorf("Expected length mismatch error, got %v", err)
	}

	// Out of vocabulary token
	_, _, _, err = tok.InsertAnchors([]int{99999999}, nil, 0)
	if err == nil || !strings.Contains(err.Error(), "out of vocabulary") {
		t.Errorf("Expected out of vocabulary error, got %v", err)
	}

	// Negative token
	_, _, _, err = tok.InsertAnchors([]int{-5}, nil, 0)
	if err == nil || !strings.Contains(err.Error(), "out of vocabulary") {
		t.Errorf("Expected out of vocabulary error, got %v", err)
	}

	// Raw anchor token
	_, _, _, err = tok.InsertAnchors([]int{M0TokenID}, nil, 0)
	if err == nil || !strings.Contains(err.Error(), "raw anchor token") {
		t.Errorf("Expected raw anchor token error, got %v", err)
	}
}

func TestTokenizerEncodeParity(t *testing.T) {
	tok, err := GetGlobalTokenizer()
	if err != nil {
		t.Fatalf("Failed to load global tokenizer: %v", err)
	}

	// Golden token sequences produced by HuggingFace AutoTokenizer (tokenizer/tokenizer_weights_v2)
	cases := []struct {
		text string
		want []int
	}{
		{
			text: "<|turn:user|>Calculate 482 * 19 - 14 and give the answer in \\boxed{}.<|/turn|><|turn:agent|>",
			want: []int{2, 2520, 348, 147, 151, 145, 424, 348, 144, 152, 439, 348, 144, 147, 445, 1967, 388, 621, 437, 421, 715, 1931, 4, 3},
		},
		{
			text: "<|obs|>r0 = 9144<|/obs|>",
			want: []int{15, 209, 143, 403, 348, 152, 144, 147, 147, 17},
		},
		{
			text: "<|obs:err|>ErrDivZero: division by zero<|/obs|>",
			want: []int{16, 164, 15360, 2909, 23289, 153, 2509, 534, 1623, 17},
		},
		{
			text: "<|turn:system|>\n<|tools|>\ndef get_weather(city: str) -> str:\n    \"\"\"Get weather.\"\"\"\n<|/tools|><|/turn|>\n<|turn:user|>Hello, world! 12345 != 67.89\nLine 2\twith tab<|/turn|><|turn:agent|>",
			want: []int{5, 326, 6, 326, 13199, 973, 190, 794, 3302, 135, 31986, 153, 781, 136, 1216, 781, 153, 454, 498, 129, 129, 27386, 14894, 1029, 129, 129, 326, 7, 4, 326, 2, 167, 24465, 139, 8883, 128, 348, 144, 145, 146, 147, 148, 13870, 348, 149, 150, 141, 151, 152, 326, 10762, 348, 145, 325, 1806, 31535, 4, 3},
		},
	}

	for idx, tc := range cases {
		got := tok.Encode(tc.text)
		decoded := tok.DecodeTokens(got)
		if decoded != tc.text {
			t.Errorf("case %d round-trip decode mismatch: got %q, want %q", idx, decoded, tc.text)
		}
		if len(got) != len(tc.want) {
			t.Errorf("case %d len mismatch: got %v, want %v", idx, got, tc.want)
			continue
		}
		for i := range tc.want {
			if got[i] != tc.want[i] {
				t.Errorf("case %d token[%d] mismatch: got %d (%q), want %d (%q)",
					idx, i, got[i], tok.IDToToken(got[i]), tc.want[i], tok.IDToToken(tc.want[i]))
			}
		}
	}
}

func TestStripAnchors(t *testing.T) {
	tok, err := GetGlobalTokenizer()
	if err != nil {
		t.Fatalf("Failed to load global tokenizer: %v", err)
	}
	raw := tok.Encode("Line 1\nLine 2\nLine 3")
	anchored, _, _, err := tok.InsertAnchors(raw, nil, 0)
	if err != nil {
		t.Fatalf("InsertAnchors failed: %v", err)
	}
	if len(anchored) <= len(raw) {
		t.Fatalf("Expected anchored tokens > raw tokens")
	}
	stripped := tok.StripAnchors(anchored)
	if len(stripped) != len(raw) {
		t.Fatalf("StripAnchors length mismatch: got %d, want %d", len(stripped), len(raw))
	}
	for i := range raw {
		if stripped[i] != raw[i] {
			t.Errorf("StripAnchors[%d]: got %d, want %d", i, stripped[i], raw[i])
		}
	}
}

