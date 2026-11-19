package main

import (
	"strings"
	"testing"
)

func TestStrTool(t *testing.T) {
	// Spec §8.1: count("strawberry", "r") == 3 (case-sensitive, non-overlapping)
	res, err := EvaluateStr("count(\"strawberry\", \"r\")")
	if err != nil || res != "3" {
		t.Fatalf("count failed: got %q, err %v", res, err)
	}

	// Spec §8.1: slice("mathematics", 0, 4) == "math" (0-indexed)
	res, err = EvaluateStr("slice(\"mathematics\", 0, 4)")
	if err != nil || res != "\"math\"" {
		t.Fatalf("slice failed: got %q, err %v", res, err)
	}

	// Reversal (UTF-8 safe)
	res, err = EvaluateStr("reverse(\"hello\")")
	if err != nil || res != "\"olleh\"" {
		t.Fatalf("reverse failed: got %q, err %v", res, err)
	}

	// Length (runes)
	res, err = EvaluateStr("len(\"strawberry\")")
	if err != nil || res != "10" {
		t.Fatalf("len failed: got %q, err %v", res, err)
	}

	// Base conversions
	res, err = EvaluateStr("to_hex(255)")
	if err != nil || res != "\"FF\"" {
		t.Fatalf("to_hex failed: got %q, err %v", res, err)
	}

	res, err = EvaluateStr("to_bin(5)")
	if err != nil || res != "\"101\"" {
		t.Fatalf("to_bin failed: got %q, err %v", res, err)
	}

	res, err = EvaluateStr("to_ascii(\"A\")")
	if err != nil || res != "65" {
		t.Fatalf("to_ascii failed: got %q, err %v", res, err)
	}

	// replace
	res, err = EvaluateStr("replace(\"strawberry\", \"r\", \"x\")")
	if err != nil || res != "\"stxawbexxy\"" {
		t.Fatalf("replace failed: got %q, err %v", res, err)
	}

	// find (0-indexed, -1 if absent)
	res, err = EvaluateStr("find(\"strawberry\", \"a\")")
	if err != nil || res != "3" {
		t.Fatalf("find 'a' failed: got %q, err %v", res, err)
	}
	res, err = EvaluateStr("find(\"strawberry\", \"z\")")
	if err != nil || res != "-1" {
		t.Fatalf("find 'z' failed: got %q, err %v", res, err)
	}

	// count_vowels (a, e, i, o, u, case-insensitive)
	res, err = EvaluateStr("count_vowels(\"strawberry\")")
	if err != nil || res != "2" {
		t.Fatalf("count_vowels 'strawberry' failed: got %q, err %v", res, err)
	}
	res, err = EvaluateStr("count_vowels(\"Encyclopedia\")")
	if err != nil || res != "5" {
		t.Fatalf("count_vowels 'Encyclopedia' failed: got %q, err %v", res, err)
	}

	// is_palindrome
	res, err = EvaluateStr("is_palindrome(\"radar\")")
	if err != nil || res != "true" {
		t.Fatalf("is_palindrome 'radar' failed: got %q, err %v", res, err)
	}
	res, err = EvaluateStr("is_palindrome(\"strawberry\")")
	if err != nil || res != "false" {
		t.Fatalf("is_palindrome 'strawberry' failed: got %q, err %v", res, err)
	}

	// Error handling: slice out of bounds
	_, err = EvaluateStr("slice(\"short\", 0, 10)")
	if err == nil || !strings.Contains(err.Error(), "ErrOutOfBounds") {
		t.Fatalf("expected ErrOutOfBounds, got %v", err)
	}
}
