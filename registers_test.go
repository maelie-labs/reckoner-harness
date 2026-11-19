package main

import (
	"math/big"
	"strings"
	"testing"
)

func TestRegisterTypingAndSelectiveBinding(t *testing.T) {
	rs := NewRegisterStore()

	// 1. Scalars auto-bind: number/rational
	tag0, bound0, _ := rs.AutoBind(CalcVal{Kind: ValKindRat, Rat: big.NewRat(68, 77)})
	if tag0 != "<|r0|>" || bound0 != "<|r0|> = 68/77" {
		t.Fatalf("scalar rational failed: got %q, %q", tag0, bound0)
	}

	// 2. Scalars auto-bind: unit value
	tag1, bound1, _ := rs.AutoBind(CalcVal{Kind: ValKindUnit, UnitVal: big.NewRat(18480, 1), UnitName: "ft"})
	if tag1 != "<|r1|>" || bound1 != "<|r1|> = 18480 ft" {
		t.Fatalf("unit scalar failed: got %q, %q", tag1, bound1)
	}

	// 3. Scalars auto-bind: date
	tag2, bound2, _ := rs.AutoBind(CalcVal{Kind: ValKindDate, DateStr: "2024-03-14"})
	if tag2 != "<|r2|>" || bound2 != "<|r2|> = 2024-03-14" {
		t.Fatalf("date scalar failed: got %q, %q", tag2, bound2)
	}

	// 4. Scalars auto-bind: boolean
	tag3, bound3, _ := rs.AutoBind(CalcVal{Kind: ValKindBool, Bool: true})
	if tag3 != "<|r3|>" || bound3 != "<|r3|> = true" {
		t.Fatalf("boolean scalar failed: got %q, %q", tag3, bound3)
	}

	// 5. Unit safety via register evaluation: r1 is "18480 ft"
	_, err := EvaluateInfixToVal("<|r1|> + 10 m", rs)
	if err == nil || !strings.Contains(err.Error(), "ErrUnitMismatch") {
		t.Fatalf("expected ErrUnitMismatch for <|r1|> + 10 m, got %v", err)
	}
}

func TestDispatchToolErrorsDoNotBindRegisters(t *testing.T) {
	globalRegisters.Reset()

	// Error call: 10 / 0
	resp := DispatchTool("calc", "10 / 0")
	if resp.Success || !strings.Contains(resp.Error, "ErrDivZero") {
		t.Fatalf("expected ErrDivZero, got %v", resp)
	}

	// Ensure r0 is uninitialized
	allRegs := globalRegisters.GetAllRegisters()
	if len(allRegs) != 0 {
		t.Fatalf("expected 0 registers bound on error, found %d: %v", len(allRegs), allRegs)
	}

	// Success call: 40 * 2
	resp = DispatchTool("calc", "40 * 2")
	if !resp.Success || resp.Result != "<|r0|> = 80" {
		t.Fatalf("expected <|r0|> = 80, got %v", resp)
	}

	// Error call after success should not disturb r0 or bind r1
	resp = DispatchTool("calc", "10 / 0")
	if resp.Success {
		t.Fatalf("expected error")
	}
	allRegs = globalRegisters.GetAllRegisters()
	if allRegs["<|r0|>"] != "80" || allRegs["<|r1|>"] != "" {
		t.Fatalf("registers corrupted after error: %v", allRegs)
	}
}
