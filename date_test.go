package main

import (
	"strings"
	"testing"
)

func TestDateTool(t *testing.T) {
	// Spec §8.1: add("2024-01-31", 1, "months") == "2024-02-29 (Thursday)" (month-end clamping)
	res, err := EvaluateDate("add(\"2024-01-31\", 1, \"months\")")
	if err != nil || res != "2024-02-29 (Thursday)" {
		t.Fatalf("add 1 month to 2024-01-31 failed: got %q, err %v", res, err)
	}

	// Non-leap year month end: 2023-01-31 + 1 month == 2023-02-28 (Tuesday)
	res, err = EvaluateDate("add(\"2023-01-31\", 1, \"months\")")
	if err != nil || res != "2023-02-28 (Tuesday)" {
		t.Fatalf("add 1 month to 2023-01-31 failed: got %q, err %v", res, err)
	}

	// Spec §3.3B: add("2024-03-14", -87, "days") == "2023-12-18 (Monday)"
	res, err = EvaluateDate("add(\"2024-03-14\", -87, \"days\")")
	if err != nil || res != "2023-12-18 (Monday)" {
		t.Fatalf("add -87 days failed: got %q, err %v", res, err)
	}

	// Spec §8.1: diff("2024-01-01", "2024-01-05") == 4
	res, err = EvaluateDate("diff(\"2024-01-01\", \"2024-01-05\")")
	if err != nil || res != "4" {
		t.Fatalf("diff failed: got %q, err %v", res, err)
	}

	// day_of_week (quoted string result)
	res, err = EvaluateDate("day_of_week(\"2024-08-25\")")
	if err != nil || res != "\"Sunday\"" {
		t.Fatalf("day_of_week failed: got %q, err %v", res, err)
	}

	// diff_weeks
	res, err = EvaluateDate("diff_weeks(\"2024-01-01\", \"2024-01-29\")")
	if err != nil || res != "4" {
		t.Fatalf("diff_weeks failed: got %q, err %v", res, err)
	}

	// next_weekday
	res, err = EvaluateDate("next_weekday(\"2026-01-01\", \"Friday\")")
	if err != nil || res != "2026-01-02 (Friday)" {
		t.Fatalf("next_weekday failed: got %q, err %v", res, err)
	}

	// quarter
	res, err = EvaluateDate("quarter(\"2026-02-14\")")
	if err != nil || res != "1" {
		t.Fatalf("quarter failed: got %q, err %v", res, err)
	}
	res, err = EvaluateDate("quarter(\"2026-11-05\")")
	if err != nil || res != "4" {
		t.Fatalf("quarter Q4 failed: got %q, err %v", res, err)
	}

	// days_in_month
	res, err = EvaluateDate("days_in_month(2024, \"February\")")
	if err != nil || res != "29" {
		t.Fatalf("days_in_month leap Feb failed: got %q, err %v", res, err)
	}
	res, err = EvaluateDate("days_in_month(2025, 2)")
	if err != nil || res != "28" {
		t.Fatalf("days_in_month non-leap Feb failed: got %q, err %v", res, err)
	}

	// is_leap_year
	res, err = EvaluateDate("is_leap_year(2024)")
	if err != nil || res != "true" {
		t.Fatalf("is_leap_year 2024 failed: got %q, err %v", res, err)
	}
	res, err = EvaluateDate("is_leap_year(2025)")
	if err != nil || res != "false" {
		t.Fatalf("is_leap_year 2025 failed: got %q, err %v", res, err)
	}

	// now() with deterministic clock injection
	store := NewRegisterStore()
	clk, _ := ParseClockTimestamp("2026-10-10T23:42:00Z")
	store.SetClock(&clk)

	valNow, err := EvaluateDateWithStore("now()", store)
	if err != nil || valNow.String() != "\"2026-10-10 23:42 UTC\"" {
		t.Fatalf("now() UTC failed: got %q, err %v", valNow.String(), err)
	}

	valNY, err := EvaluateDateWithStore("now(\"America/New_York\")", store)
	if err != nil || valNY.String() != "\"2026-10-10 19:42 EDT\"" {
		t.Fatalf("now() NY EDT failed: got %q, err %v", valNY.String(), err)
	}

	// Winter time (EST / GMT)
	clkWinter, _ := ParseClockTimestamp("2026-01-15T17:30:00Z")
	store.SetClock(&clkWinter)
	valNYWinter, err := EvaluateDateWithStore("now(\"America/New_York\")", store)
	if err != nil || valNYWinter.String() != "\"2026-01-15 12:30 EST\"" {
		t.Fatalf("now() NY EST failed: got %q, err %v", valNYWinter.String(), err)
	}

	valLonWinter, err := EvaluateDateWithStore("now(\"Europe/London\")", store)
	if err != nil || valLonWinter.String() != "\"2026-01-15 17:30 GMT\"" {
		t.Fatalf("now() London GMT failed: got %q, err %v", valLonWinter.String(), err)
	}

	// Summer time (BST)
	clkSummer, _ := ParseClockTimestamp("2026-07-01T12:00:00Z")
	store.SetClock(&clkSummer)
	valLonSummer, err := EvaluateDateWithStore("now(\"Europe/London\")", store)
	if err != nil || valLonSummer.String() != "\"2026-07-01 13:00 BST\"" {
		t.Fatalf("now() London BST failed: got %q, err %v", valLonSummer.String(), err)
	}

	// Tokyo JST (no DST)
	store.SetClock(&clk)
	valTok, err := EvaluateDateWithStore("now(\"Asia/Tokyo\")", store)
	if err != nil || valTok.String() != "\"2026-10-11 08:42 JST\"" {
		t.Fatalf("now() Tokyo JST failed: got %q, err %v", valTok.String(), err)
	}

	// Unknown timezone
	_, err = EvaluateDateWithStore("now(\"Unknown/Zone\")", store)
	if err == nil || !strings.Contains(err.Error(), "ErrSyntax: unknown IANA timezone") {
		t.Fatalf("expected ErrSyntax for unknown timezone, got %v", err)
	}

	// Error handling: invalid date
	_, err = EvaluateDate("add(\"invalid-date\", 1, \"days\")")
	if err == nil || !strings.Contains(err.Error(), "ErrInvalidDate") {
		t.Fatalf("expected ErrInvalidDate, got %v", err)
	}

	// Error handling: invalid unit
	_, err = EvaluateDate("add(\"2024-01-01\", 1, \"decades\")")
	if err == nil || !strings.Contains(err.Error(), "ErrInvalidUnit") {
		t.Fatalf("expected ErrInvalidUnit, got %v", err)
	}
}
