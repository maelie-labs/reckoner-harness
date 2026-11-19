package main

import (
	"encoding/json"
	"os"
	"strings"
	"testing"
)

func TestContextStoreBasic(t *testing.T) {
	tok, err := GetGlobalTokenizer()
	if err != nil {
		t.Fatalf("Failed to load tokenizer: %v", err)
	}

	cs := NewContextStore(tok)

	// Create a message: 3 lines of text
	// Line 1: "Hello world"
	// Line 2: "Second line of text"
	// Line 3: "Third and final line"
	newlineID := tok.TokenToID("Ċ")
	if newlineID == -1 {
		newlineID = tok.TokenToID("\n")
	}

	// We can find some token IDs
	t1 := 1000
	t2 := 1001
	t3 := 1002

	msgTokens := []int{t1, newlineID, t2, newlineID, t3}
	anchored, targets, nextMsg, err := cs.StoreMessage(msgTokens, nil, nil)
	if err != nil {
		t.Fatalf("StoreMessage failed: %v", err)
	}

	if nextMsg != 1 {
		t.Errorf("Expected nextMsg 1, got %d", nextMsg)
	}
	if len(anchored) == 0 || len(targets) != len(anchored) {
		t.Errorf("Invalid anchored tokens: %v", anchored)
	}

	// Active slot 0 should have 3 lines
	seg := cs.activeSlots[0]
	if seg == nil || !seg.Active {
		t.Fatalf("Slot 0 should be active")
	}
	if len(seg.Lines) != 3 {
		t.Fatalf("Expected 3 lines, got %d", len(seg.Lines))
	}
}

func TestQuoteSingleAndMultiLine(t *testing.T) {
	tok, err := GetGlobalTokenizer()
	if err != nil {
		t.Fatalf("Failed to load tokenizer: %v", err)
	}

	cs := NewContextStore(tok)

	newlineID := tok.TokenToID("Ċ")
	if newlineID == -1 {
		newlineID = tok.TokenToID("\n")
	}

	// 5 lines
	tokens := []int{
		100, newlineID,
		200, newlineID,
		300, newlineID,
		400, newlineID,
		500,
	}

	_, _, _, err = cs.StoreMessage(tokens, nil, nil)
	if err != nil {
		t.Fatalf("StoreMessage failed: %v", err)
	}

	// Test single line quote: L1
	q1, err := cs.Quote("<|quote|><|m0|><|L1|><|L1|><|/quote|>")
	if err != nil {
		t.Fatalf("Quote L1 failed: %v", err)
	}
	if q1.Provenance.StartLine != 1 || q1.Provenance.EndLine != 1 || q1.Provenance.NumLines != 1 {
		t.Errorf("Invalid single line provenance: %+v", q1.Provenance)
	}
	if !strings.Contains(q1.Receipt, "Quoted 1 line (L1):") {
		t.Errorf("Unexpected single line receipt: %s", q1.Receipt)
	}

	// Test multi-line quote: L0-L3
	qMulti, err := cs.Quote("<|quote|><|m0|><|L0|><|L3|><|/quote|>")
	if err != nil {
		t.Fatalf("Quote L0-L3 failed: %v", err)
	}
	if qMulti.Provenance.NumLines != 4 {
		t.Errorf("Expected 4 lines, got %d", qMulti.Provenance.NumLines)
	}
	if !strings.Contains(qMulti.Receipt, "Quoted 4 lines (L0-L3):") {
		t.Errorf("Unexpected multi-line receipt: %s", qMulti.Receipt)
	}

	// Test error: out of bounds end line
	_, err = cs.Quote("<|quote|><|m0|><|L0|><|L10|><|/quote|>")
	if err == nil || !strings.Contains(err.Error(), "ErrOutOfBounds") {
		t.Errorf("Expected ErrOutOfBounds, got %v", err)
	}

	// Test error: start > end
	_, err = cs.Quote("<|quote|><|m0|><|L3|><|L1|><|/quote|>")
	if err == nil || !strings.Contains(err.Error(), "ErrOutOfBounds") {
		t.Errorf("Expected ErrOutOfBounds, got %v", err)
	}

	// Test error: inactive message token
	_, err = cs.Quote("<|quote|><|m5|><|L0|><|L1|><|/quote|>")
	if err == nil || !strings.Contains(err.Error(), "ErrNotFound") {
		t.Errorf("Expected ErrNotFound, got %v", err)
	}
}

func TestQuoteMaskStateMachine(t *testing.T) {
	tok, err := GetGlobalTokenizer()
	if err != nil {
		t.Fatalf("Failed to load tokenizer: %v", err)
	}

	cs := NewContextStore(tok)

	newlineID := tok.TokenToID("Ċ")
	if newlineID == -1 {
		newlineID = tok.TokenToID("\n")
	}

	// Add 2 messages: m0 has 3 lines, m1 has 2 lines
	cs.StoreMessage([]int{10, newlineID, 20, newlineID, 30}, nil, nil)
	cs.StoreMessage([]int{40, newlineID, 50}, nil, nil)

	// Step 0: prefix ["<|quote|>"]
	m0, err := cs.QuoteMask([]string{"<|quote|>"})
	if err != nil {
		t.Fatalf("QuoteMask step 0 failed: %v", err)
	}
	if len(m0.AllowedTokens) != 2 || m0.AllowedTokens[0] != "<|m0|>" || m0.AllowedTokens[1] != "<|m1|>" {
		t.Errorf("Unexpected step 0 tokens: %v", m0.AllowedTokens)
	}

	// Step 1: prefix ["<|quote|>", "<|m0|>"]
	m1, err := cs.QuoteMask([]string{"<|quote|>", "<|m0|>"})
	if err != nil {
		t.Fatalf("QuoteMask step 1 failed: %v", err)
	}
	if len(m1.AllowedTokens) != 3 || m1.AllowedTokens[0] != "<|L0|>" || m1.AllowedTokens[2] != "<|L2|>" {
		t.Errorf("Unexpected step 1 tokens: %v", m1.AllowedTokens)
	}

	// Step 2: prefix ["<|quote|>", "<|m0|>", "<|L1|>"] -> allowed L1, L2
	m2, err := cs.QuoteMask([]string{"<|quote|>", "<|m0|>", "<|L1|>"})
	if err != nil {
		t.Fatalf("QuoteMask step 2 failed: %v", err)
	}
	if len(m2.AllowedTokens) != 2 || m2.AllowedTokens[0] != "<|L1|>" || m2.AllowedTokens[1] != "<|L2|>" {
		t.Errorf("Unexpected step 2 tokens: %v", m2.AllowedTokens)
	}

	// Step 3: prefix ["<|quote|>", "<|m0|>", "<|L1|>", "<|L2|>"] -> allowed ["<|/quote|>"]
	m3, err := cs.QuoteMask([]string{"<|quote|>", "<|m0|>", "<|L1|>", "<|L2|>"})
	if err != nil {
		t.Fatalf("QuoteMask step 3 failed: %v", err)
	}
	if len(m3.AllowedTokens) != 1 || m3.AllowedTokens[0] != "<|/quote|>" {
		t.Errorf("Unexpected step 3 tokens: %v", m3.AllowedTokens)
	}
}

func TestContextRecycling(t *testing.T) {
	tok, err := GetGlobalTokenizer()
	if err != nil {
		t.Fatalf("Failed to load tokenizer: %v", err)
	}

	cs := NewContextStore(tok)

	// Fill all 256 slots
	for i := 0; i < 256; i++ {
		start := i
		cs.StoreMessage([]int{100 + i}, nil, &start)
	}

	// Check m0 points to turn 1
	seg0 := cs.activeSlots[0]
	if seg0.TurnID != 1 {
		t.Errorf("Expected TurnID 1, got %d", seg0.TurnID)
	}

	// Store 257th message (slot 0 recycled!)
	zero := 0
	cs.StoreMessage([]int{999}, nil, &zero)

	// Now m0 points to turn 257
	segRecycled := cs.activeSlots[0]
	if segRecycled.TurnID != 257 {
		t.Errorf("Expected TurnID 257 for recycled slot 0, got %d", segRecycled.TurnID)
	}
	// Old segment should be marked inactive
	if seg0.Active {
		t.Errorf("Old segment should be inactive")
	}

	// Quoting m0 should resolve against turn 257
	q, err := cs.Quote("<|quote|><|m0|><|L0|><|L0|><|/quote|>")
	if err != nil {
		t.Fatalf("Quote on recycled slot failed: %v", err)
	}
	if len(q.Tokens) != 1 || q.Tokens[0] != 999 {
		t.Errorf("Expected recycled token 999, got %v", q.Tokens)
	}
}

func TestFindAndLines(t *testing.T) {
	tok, err := GetGlobalTokenizer()
	if err != nil {
		t.Fatalf("Failed to load tokenizer: %v", err)
	}

	cs := NewContextStore(tok)

	newlineID := tok.TokenToID("Ċ")
	if newlineID == -1 {
		newlineID = tok.TokenToID("\n")
	}

	// Message with keyword invoice ch_9812
	// Encoded: "Invoice ch_9812\nAmount: $500"
	m0ID := 0
	cs.StoreMessage([]int{
		25049, 12511, 659, 190, 152, 151, 144, 145,
		newlineID,
		8977, 153, 404, 148, 143, 143,
	}, nil, &m0ID)

	// Search for "invoice"
	res, err := cs.Find("invoice", nil)
	if err != nil {
		t.Fatalf("Find failed: %v", err)
	}
	if res.NumMatches == 0 {
		t.Errorf("Expected match for 'invoice', got 0 matches")
	}
	if !strings.HasPrefix(res.Result, "<|m1|><|L0|>") {
		t.Errorf("Expected newly anchored message starting with <|m1|><|L0|>, got %q", res.Result)
	}

	// Search for non-existent keyword
	noMatch, err := cs.Find("nonexistent_needle_xyz", nil)
	if err != nil {
		t.Fatalf("Find nonexistent failed: %v", err)
	}
	if noMatch.Result != "no matches" || noMatch.NumMatches != 0 {
		t.Errorf("Expected 'no matches', got %v", noMatch)
	}

	// Test lines(0, 0, 1)
	linesRes, err := cs.Lines(0, 0, 1)
	if err != nil {
		t.Fatalf("Lines failed: %v", err)
	}
	if linesRes.NumMatches != 1 {
		t.Errorf("Expected 1 match, got %d", linesRes.NumMatches)
	}
	if !strings.HasPrefix(linesRes.Result, "<|m") {
		t.Errorf("Expected anchored message, got %q", linesRes.Result)
	}
}

func TestContextConformanceSuite(t *testing.T) {
	tok, err := GetGlobalTokenizer()
	if err != nil {
		t.Fatalf("Failed to load tokenizer: %v", err)
	}

	cs := NewContextStore(tok)

	casesData, err := os.ReadFile("conformance/context_cases.jsonl")
	if err != nil {
		t.Fatalf("Failed to read context cases: %v", err)
	}

	lines := strings.Split(string(casesData), "\n")
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		type contextCase struct {
			ID                    string   `json:"id"`
			Tool                  string   `json:"tool"`
			Category              string   `json:"category"`
			Tokens                []int    `json:"tokens"`
			Targets               []int    `json:"targets"`
			MessageID             *int     `json:"message_id"`
			SetupMessages         [][]int  `json:"setup_messages"`
			Call                  string   `json:"call"`
			Payload               string   `json:"payload"`
			Prefix                []string `json:"prefix"`
			ExpectedTokens        []int    `json:"expected_tokens"`
			ExpectedTargets       []int    `json:"expected_targets"`
			ExpectedNextMessageID *int     `json:"expected_next_message_id"`
			ExpectedText          string   `json:"expected_text"`
			ExpectedReceipt       string   `json:"expected_receipt"`
			ExpectedReceiptPrefix string   `json:"expected_receipt_prefix"`
			ExpectedErrorPrefix   string   `json:"expected_error_prefix"`
			ExpectedAllowedTokens []string `json:"expected_allowed_tokens"`
			ExpectedStep          string   `json:"expected_step"`
			ExpectedPrefix        string   `json:"expected_prefix"`
			ExpectedResult        string   `json:"expected_result"`
			ExpectedMatches       *int     `json:"expected_matches"`
			Anchors               *bool    `json:"anchors,omitempty"`
		}

		var cc contextCase
		if err := json.Unmarshal([]byte(line), &cc); err != nil {
			t.Fatalf("Failed to parse case: %v", err)
		}

		t.Run(cc.ID+"_"+cc.Category, func(t *testing.T) {
			cs.Reset()
			if cc.Anchors != nil {
				cs.SetAnchorsEnabled(*cc.Anchors)
			} else {
				cs.SetAnchorsEnabled(true)
			}
			for _, setup := range cc.SetupMessages {
				cs.StoreMessage(setup, nil, nil)
			}

			switch cc.Tool {
			case "anchor":
				reqID := 0
				if cc.MessageID != nil {
					reqID = *cc.MessageID
				}
				outToks, outTargets, nextMsg, err := tok.InsertAnchors(cc.Tokens, cc.Targets, reqID)
				if cc.ExpectedErrorPrefix != "" {
					if err == nil || !strings.HasPrefix(err.Error(), cc.ExpectedErrorPrefix) {
						t.Errorf("Expected error prefix %s, got %v", cc.ExpectedErrorPrefix, err)
					}
					return
				}
				if err != nil {
					t.Fatalf("InsertAnchors failed: %v", err)
				}
				if len(cc.ExpectedTokens) > 0 {
					if len(outToks) != len(cc.ExpectedTokens) {
						t.Fatalf("Tokens length mismatch: got %d, expected %d", len(outToks), len(cc.ExpectedTokens))
					}
					for i := range cc.ExpectedTokens {
						if outToks[i] != cc.ExpectedTokens[i] {
							t.Errorf("Token[%d]: got %d, expected %d", i, outToks[i], cc.ExpectedTokens[i])
						}
					}
				}
				if len(cc.ExpectedTargets) > 0 {
					for i := range cc.ExpectedTargets {
						if outTargets[i] != cc.ExpectedTargets[i] {
							t.Errorf("Target[%d]: got %d, expected %d", i, outTargets[i], cc.ExpectedTargets[i])
						}
					}
				}
				if cc.ExpectedNextMessageID != nil && nextMsg != *cc.ExpectedNextMessageID {
					t.Errorf("Expected next msg %d, got %d", *cc.ExpectedNextMessageID, nextMsg)
				}

			case "quote":
				call := cc.Call
				if call == "" {
					call = cc.Payload
				}
				res, err := cs.Quote(call)
				if cc.ExpectedErrorPrefix != "" {
					if err == nil || !strings.HasPrefix(err.Error(), cc.ExpectedErrorPrefix) {
						t.Errorf("Expected error prefix %s, got %v", cc.ExpectedErrorPrefix, err)
					}
				} else {
					if err != nil {
						t.Fatalf("Quote failed: %v", err)
					}
					if cc.ExpectedText != "" && res.Text != cc.ExpectedText {
						t.Errorf("Expected text %q, got %q", cc.ExpectedText, res.Text)
					}
					if cc.ExpectedReceipt != "" && res.Receipt != cc.ExpectedReceipt {
						t.Errorf("Expected receipt exact %q, got %q", cc.ExpectedReceipt, res.Receipt)
					}
					if cc.ExpectedReceiptPrefix != "" && !strings.HasPrefix(res.Receipt, cc.ExpectedReceiptPrefix) {
						t.Errorf("Expected receipt prefix %q, got %q", cc.ExpectedReceiptPrefix, res.Receipt)
					}
				}

			case "quote_mask":
				res, err := cs.QuoteMask(cc.Prefix)
				if cc.ExpectedErrorPrefix != "" {
					if err == nil || !strings.HasPrefix(err.Error(), cc.ExpectedErrorPrefix) {
						t.Errorf("Expected error prefix %s, got %v", cc.ExpectedErrorPrefix, err)
					}
				} else {
					if err != nil {
						t.Fatalf("QuoteMask failed: %v", err)
					}
					if len(cc.ExpectedAllowedTokens) > 0 {
						if len(res.AllowedTokens) != len(cc.ExpectedAllowedTokens) {
							t.Fatalf("Allowed tokens length mismatch: got %v, expected %v", res.AllowedTokens, cc.ExpectedAllowedTokens)
						}
						for i := range cc.ExpectedAllowedTokens {
							if res.AllowedTokens[i] != cc.ExpectedAllowedTokens[i] {
								t.Errorf("Allowed token[%d]: got %s, expected %s", i, res.AllowedTokens[i], cc.ExpectedAllowedTokens[i])
							}
						}
					}
					if cc.ExpectedStep != "" && res.Step != cc.ExpectedStep {
						t.Errorf("Expected step %s, got %s", cc.ExpectedStep, res.Step)
					}
				}

			case "find":
				isLines, q, scope, s, e, pErr := ParseFindOrLinesCall(cc.Payload)
				if pErr != nil && cc.ExpectedErrorPrefix != "" {
					return
				}
				var res *FindResult
				var fErr error
				if isLines {
					res, fErr = cs.Lines(*scope, s, e)
				} else {
					res, fErr = cs.Find(q, scope)
				}
				if cc.ExpectedErrorPrefix != "" {
					if fErr == nil || !strings.HasPrefix(fErr.Error(), cc.ExpectedErrorPrefix) {
						t.Errorf("Expected error prefix %s, got %v", cc.ExpectedErrorPrefix, fErr)
					}
				} else {
					if fErr != nil {
						t.Fatalf("Find failed: %v", fErr)
					}
					if cc.ExpectedPrefix != "" && !strings.HasPrefix(res.Result, cc.ExpectedPrefix) {
						t.Errorf("Expected prefix %q, got %q", cc.ExpectedPrefix, res.Result)
					}
					if cc.ExpectedResult != "" && res.Result != cc.ExpectedResult {
						t.Errorf("Expected result %q, got %q", cc.ExpectedResult, res.Result)
					}
					if cc.ExpectedMatches != nil && res.NumMatches != *cc.ExpectedMatches {
						t.Errorf("Expected %d matches, got %d", *cc.ExpectedMatches, res.NumMatches)
					}
				}

			case "lines":
				isLines, _, scope, s, e, pErr := ParseFindOrLinesCall(cc.Payload)
				if pErr != nil {
					if cc.ExpectedErrorPrefix != "" {
						return
					}
					t.Fatalf("Parse failed: %v", pErr)
				}
				if !isLines || scope == nil {
					t.Fatal("Expected lines syntax")
				}
				res, err := cs.Lines(*scope, s, e)
				if cc.ExpectedErrorPrefix != "" {
					if err == nil || !strings.HasPrefix(err.Error(), cc.ExpectedErrorPrefix) {
						t.Errorf("Expected error prefix %s, got %v", cc.ExpectedErrorPrefix, err)
					}
				} else {
					if err != nil {
						t.Fatalf("Lines failed: %v", err)
					}
					if cc.ExpectedPrefix != "" && !strings.HasPrefix(res.Result, cc.ExpectedPrefix) {
						t.Errorf("Expected prefix %q, got %q", cc.ExpectedPrefix, res.Result)
					}
				}
			}
		})
	}
}

func TestContextEdgeCases(t *testing.T) {
	tok, err := GetGlobalTokenizer()
	if err != nil {
		t.Fatalf("Failed to load tokenizer: %v", err)
	}

	cs := NewContextStore(tok)

	// 1. Negative messageID in InsertAnchors -> ErrArgs
	_, _, _, err = tok.InsertAnchors([]int{100, 101}, nil, -3)
	if err == nil || !strings.Contains(err.Error(), "ErrArgs") {
		t.Errorf("Expected ErrArgs for negative message_id in InsertAnchors, got %v", err)
	}

	// 2. Math-mod messageID in InsertAnchors (e.g. 256 wraps to 0)
	tokens, _, nextMsg, err := tok.InsertAnchors([]int{100, 101}, nil, 256)
	if err != nil {
		t.Fatalf("InsertAnchors with message_id=256 failed: %v", err)
	}
	if tokens[0] != M0TokenID {
		t.Errorf("Expected token 0 to be M0TokenID (%d), got %d", M0TokenID, tokens[0])
	}
	if nextMsg != 1 {
		t.Errorf("Expected nextMsg to be 1, got %d", nextMsg)
	}

	// 3. Negative messageID in StoreMessage -> ErrArgs
	negMsg := -1
	_, _, _, err = cs.StoreMessage([]int{100, 101}, nil, &negMsg)
	if err == nil || !strings.Contains(err.Error(), "ErrArgs") {
		t.Errorf("Expected ErrArgs for negative message_id in StoreMessage, got %v", err)
	}

	// 4. Quote m-1 L0 L0 -> ErrArgs (not panic!)
	_, err = cs.Quote("m-1 L0 L0")
	if err == nil || !strings.Contains(err.Error(), "ErrArgs") {
		t.Errorf("Expected ErrArgs for m-1 quote, got %v", err)
	}

	// 5. Lines with -1 -> ErrArgs
	_, err = cs.Lines(-1, 0, 0)
	if err == nil || !strings.Contains(err.Error(), "ErrArgs") {
		t.Errorf("Expected ErrArgs for lines -1, got %v", err)
	}

	// 6. Find with negative scopeMsgID -> ErrArgs
	negScope := -1
	_, err = cs.Find("hello", &negScope)
	if err == nil || !strings.Contains(err.Error(), "ErrArgs") {
		t.Errorf("Expected ErrArgs for find with negative scope, got %v", err)
	}

	// 7. Find deduplication: store message, find it, find it again
	storeMsg := []int{25049, 12511} // "Invoice"
	_, _, _, err = cs.StoreMessage(storeMsg, nil, nil)
	if err != nil {
		t.Fatalf("StoreMessage failed: %v", err)
	}

	// First find for "invoice"
	f1, err := cs.Find("invoice", nil)
	if err != nil {
		t.Fatalf("First find failed: %v", err)
	}
	if f1.NumMatches != 1 {
		t.Errorf("Expected 1 match on first find, got %d", f1.NumMatches)
	}

	// Second find for same query: should still match ONLY 1 excerpt from original message, NOT the excerpt message
	f2, err := cs.Find("invoice", nil)
	if err != nil {
		t.Fatalf("Second find failed: %v", err)
	}
	if f2.NumMatches != 1 {
		t.Errorf("Expected 1 match on second find (excluding derived search results), got %d", f2.NumMatches)
	}

	// 8. Main handler tests: malformed anchor payload and negative message_id
	store := NewRegisterStore()
	resp := DispatchJSONRequest(ToolRequest{
		Tool:    "anchor",
		Payload: "not json",
	}, store)
	if resp.Success || !strings.Contains(resp.Error, "ErrSyntax") {
		t.Errorf("Expected ErrSyntax for malformed anchor payload, got: %+v", resp)
	}

	badMsg := -3
	resp2 := DispatchJSONRequest(ToolRequest{
		Tool:      "anchor",
		Tokens:    []int{100, 101},
		MessageID: &badMsg,
	}, store)
	if resp2.Success || !strings.Contains(resp2.Error, "ErrArgs") {
		t.Errorf("Expected ErrArgs for message_id -3, got: %+v", resp2)
	}
}

func TestAnchorFreeMode(t *testing.T) {
	tok, err := GetGlobalTokenizer()
	if err != nil {
		t.Fatalf("Failed to load tokenizer: %v", err)
	}

	cs := NewContextStore(tok)
	cs.SetAnchorsEnabled(false)

	if cs.AnchorsEnabled() {
		t.Fatalf("Expected AnchorsEnabled() == false")
	}

	// 1. Store a message in anchor-free mode
	text := "Invoice ch_9812\nAmount: $500\nStatus: Paid\n"
	encoded := tok.Encode(text)
	tokens, _, nextMsg, err := cs.StoreMessage(encoded, nil, nil)
	if err != nil {
		t.Fatalf("StoreMessage in anchor-free mode failed: %v", err)
	}
	if nextMsg != 1 {
		t.Errorf("Expected nextMsg 1, got %d", nextMsg)
	}
	seg := cs.activeSlots[0]
	if seg == nil {
		t.Fatalf("Expected active segment at slot 0")
	}
	if len(seg.Lines) != 3 {
		t.Errorf("Expected 3 lines, got %d", len(seg.Lines))
	}

	// Verify no anchor tokens in output
	for _, tokID := range tokens {
		if (tokID >= M0TokenID && tokID < M0TokenID+NumMessageTokens) || (tokID >= L0TokenID && tokID < L0TokenID+NumLineAnchors) {
			t.Errorf("Found anchor token %d in anchor-free tokens", tokID)
		}
	}

	// 2. Find exact match
	fRes, err := cs.Find("Invoice", nil)
	if err != nil {
		t.Fatalf("Find failed: %v", err)
	}
	if strings.Contains(fRes.Result, "<|m") || strings.Contains(fRes.Result, "<|L") {
		t.Errorf("Find result contains anchor tokens: %q", fRes.Result)
	}
	if !strings.HasPrefix(fRes.Result, "[exact]") {
		t.Errorf("Expected [exact] prefix, got %q", fRes.Result)
	}
	if !strings.Contains(fRes.Result, "Invoice ch_9812") {
		t.Errorf("Expected excerpt to contain 'Invoice ch_9812', got %q", fRes.Result)
	}

	// 3. Find ranked BM25 match
	bmRes, err := cs.Find("Invoice 9812 Paid", nil)
	if err != nil {
		t.Fatalf("Find BM25 failed: %v", err)
	}
	if strings.Contains(bmRes.Result, "<|m") || strings.Contains(bmRes.Result, "<|L") {
		t.Errorf("BM25 result contains anchor tokens: %q", bmRes.Result)
	}

	// 4. Lines in anchor-free mode
	lRes, err := cs.Lines(0, 0, 1)
	if err != nil {
		t.Fatalf("Lines failed: %v", err)
	}
	if strings.Contains(lRes.Result, "<|m") || strings.Contains(lRes.Result, "<|L") {
		t.Errorf("Lines result contains anchor tokens: %q", lRes.Result)
	}
	if !strings.Contains(lRes.Result, "Invoice ch_9812") {
		t.Errorf("Lines result should contain line 0, got %q", lRes.Result)
	}

	// 5. Quote and QuoteMask must return ErrUnsupported
	_, err = cs.Quote("m0 L0 L0")
	if err == nil || !strings.Contains(err.Error(), "ErrUnsupported") {
		t.Errorf("Expected ErrUnsupported for Quote in anchor-free mode, got %v", err)
	}

	_, err = cs.QuoteMask([]string{"m0", "L0"})
	if err == nil || !strings.Contains(err.Error(), "ErrUnsupported") {
		t.Errorf("Expected ErrUnsupported for QuoteMask in anchor-free mode, got %v", err)
	}

	// 6. Test JSON dispatch with session_config
	store := NewRegisterStore()
	cfgResp := DispatchJSONRequest(ToolRequest{
		Tool:    "session_config",
		Anchors: boolPtr(false),
	}, store)
	if !cfgResp.Success {
		t.Fatalf("session_config failed: %+v", cfgResp)
	}

	qResp := DispatchJSONRequest(ToolRequest{
		Tool:    "quote",
		Payload: "m0 L0 L0",
	}, store)
	if qResp.Success || !strings.Contains(qResp.Error, "ErrUnsupported") {
		t.Errorf("Expected ErrUnsupported via dispatch, got %+v", qResp)
	}

	// Re-enable anchors
	cfgResp2 := DispatchJSONRequest(ToolRequest{
		Tool:    "session_config",
		Anchors: boolPtr(true),
	}, store)
	if !cfgResp2.Success {
		t.Fatalf("session_config re-enable failed: %+v", cfgResp2)
	}
}

func boolPtr(b bool) *bool {
	return &b
}

