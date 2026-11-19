package main

import (
	"math"
	"math/big"
	"strings"
	"testing"
)

func TestLuaExecution(t *testing.T) {
	rs := NewRegisterStore()

	tests := []struct {
		name    string
		script  string
		want    string
		wantErr bool
	}{
		{
			name:   "Basic arithmetic and return",
			script: "local a = 14 * 3; return a",
			want:   "42",
		},
		{
			name:   "Table operations and loops",
			script: `
				local t = {3, 1, 4, 1, 5}
				table.sort(t)
				local sum = 0
				for _, v in ipairs(t) do
					sum = sum + v
				end
				return sum
			`,
			want: "14",
		},
		{
			name:   "Table concat and tonumber",
			script: `
				local digits = {"4", "2"}
				local s = table.concat(digits, "")
				return tonumber(s)
			`,
			want: "42",
		},
		{
			name:   "Injected math helpers",
			script: `
				local g = gcd(48, 18)
				local p = isprime(17)
				local c = comb(5, 2)
				return g, p, c
			`,
			want: "6, true, 10",
		},
		{
			name: "Register variable reading",
			script: `
				return r0 * 2 + r1
			`,
			want: "94", // r0=42, r1=10 -> 42*2 + 10 = 94
		},
		{
			name:    "Missing return statement",
			script:  "local x = 5",
			wantErr: true,
		},
		{
			name:    "Timeout on infinite loop",
			script:  "while true do end",
			wantErr: true,
		},
		{
			name:    "Memory limit string.rep heap bomb",
			script:  `return string.rep("a", 15000000)`,
			wantErr: true,
		},
		{
			name: "Table.insert 20k elements succeeds under 10MB heap cap",
			script: `
				local t = {}
				for i = 1, 20000 do
					table.insert(t, i)
				end
				return #t
			`,
			want: "20000",
		},
		{
			name:   "Native 64-bit integers past 2^53 (Lua 5.4)",
			script: "return 9007199254740994",
			want:   "9007199254740994",
		},
		{
			name:   "1e20 evaluates as float per float formatting specification",
			script: "return 1e20",
			want:   "1e+20",
		},
		{
			name:   "Lua 5.4 integer division //",
			script: "return 14 // 3",
			want:   "4",
		},
		{
			name:   "Lua 5.4 bitwise operators",
			script: "return (5 & 3), (5 | 2), (5 ~ 1), (1 << 4)",
			want:   "1, 7, 4, 16",
		},
		{
			name:   "Lua 5.4 utf8 library",
			script: `return utf8.len("hello world")`,
			want:   "11",
		},
		{
			name: "Lua 5.4 goto statement",
			script: `
				local x = 0
				goto skip
				x = 10
				::skip::
				return x
			`,
			want: "0",
		},
		{
			name: "Timeout on goto-based infinite loop",
			script: `
				::loop::
				goto loop
			`,
			wantErr: true,
		},
		{
			name:    "Float format float(2^53)",
			script:  "return 9007199254740992",
			want:    "9007199254740992",
			wantErr: false,
		},
		{
			name:    "Float format 1e300",
			script:  "return 1e300",
			want:    "1e+300",
			wantErr: false,
		},
		{
			name:    "Float format 2.0^60",
			script:  "return 2.0^60",
			want:    "1.15292e+18",
			wantErr: false,
		},
		{
			name:    "Float format exp(-20)",
			script:  "return math.exp(-20)",
			want:    "2.06115e-09",
			wantErr: false,
		},
		{
			name:    "Float format -0.0",
			script:  "return -0.0",
			want:    "0",
			wantErr: false,
		},
		{
			name:    "Float format NaN rejection",
			script:  "return 0/0",
			wantErr: true,
		},
		{
			name:    "Float format Inf rejection",
			script:  "return 1/0",
			wantErr: true,
		},
		{
			name: "Memory limit allocator heap bomb",
			script: `
				local s = "abcdefghij"
				for i = 1, 25 do
					s = s .. s
				end
				return #s
			`,
			wantErr: true,
		},
		{
			name:    "pcall is stripped",
			script:  "return pcall(function() return 1 end)",
			wantErr: true,
		},
		{
			name:    "xpcall is stripped",
			script:  "return xpcall(function() return 1 end, function() end)",
			wantErr: true,
		},
		{
			name:    "load is stripped",
			script:  `return load("return 1")()`,
			wantErr: true,
		},
		{
			name:    "os is stripped",
			script:  `return os.execute("ls")`,
			wantErr: true,
		},
		{
			name:    "io is stripped",
			script:  `return io.open("/etc/passwd")`,
			wantErr: true,
		},
		{
			name:    "string.dump is stripped",
			script:  `return string.dump(function() return 1 end)`,
			wantErr: true,
		},
		{
			name:    "metatable protection on strings",
			script:  `getmetatable("").__index = {}`,
			wantErr: true,
		},
		{
			name:    "setmetatable on strings rejected",
			script:  `setmetatable("foo", {})`,
			wantErr: true,
		},
		{
			name:   "reg_multi_return",
			script: "return 1, 2, 3",
			want:   "1, 2, 3",
		},
	}

	// Pre-populate r0=42, r1=10 for register test
	rs.AutoBind(CalcVal{Kind: ValKindRat, Rat: big.NewRat(42, 1)}) // r0
	rs.AutoBind(CalcVal{Kind: ValKindRat, Rat: big.NewRat(10, 1)}) // r1

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ExecuteLua(tt.script, rs)
			if (err != nil) != tt.wantErr {
				t.Fatalf("ExecuteLua() error = %v, wantErr %v", err, tt.wantErr)
			}
			if !tt.wantErr && got != tt.want {
				t.Errorf("ExecuteLua() got = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestLuaDeterminism(t *testing.T) {
	// math.random is deterministically seeded with 42
	got1, err1 := ExecuteLua("return math.random(1, 100000)", nil)
	if err1 != nil {
		t.Fatalf("ExecuteLua failed: %v", err1)
	}
	got2, err2 := ExecuteLua("return math.random(1, 100000)", nil)
	if err2 != nil {
		t.Fatalf("ExecuteLua failed: %v", err2)
	}
	if got1 != got2 {
		t.Errorf("math.random is non-deterministic: got %v vs %v", got1, got2)
	}
}

func TestLuaDispatchRingBuffer(t *testing.T) {
	globalRegisters.Reset()

	resp := DispatchTool("lua", "local x = 25 * 4; return x")
	if !resp.Success {
		t.Fatalf("DispatchTool failed: %s", resp.Error)
	}
	if resp.Result != "<|r0|> = 100" {
		t.Errorf("Expected <|r0|> = 100, got %s", resp.Result)
	}

	// Verify next call auto-binds to r1 and can read r0
	resp2 := DispatchTool("lua", "return r0 + 50")
	if !resp2.Success {
		t.Fatalf("DispatchTool failed: %s", resp2.Error)
	}
	if resp2.Result != "<|r1|> = 150" {
		t.Errorf("Expected <|r1|> = 150, got %s", resp2.Result)
	}
}

func TestF1_FloatDisplay(t *testing.T) {
	// Tests for float formatting: 2^53, 2.0^60, 1 << 60, 1e20, 1e300, 1/3, 0.1+0.2, exp(-20), -0.0
	cases := []struct {
		script string
		want   string
	}{
		{"return 2^53", "9007199254740992"},
		{"return 2.0^60", "1.15292e+18"},
		{"return 1 << 60", "1152921504606846976"},
		{"return 1e20", "1e+20"},
		{"return 1e300", "1e+300"},
		{"return 1/3", "0.333333"},
		{"return 0.1 + 0.2", "0.3"},
		{"return math.exp(-20)", "2.06115e-09"},
		{"return 1e15 + 0.25", "1e+15"},
		{"return -0.0", "0"},
	}
	for _, tc := range cases {
		got, err := ExecuteLua(tc.script, nil)
		if err != nil {
			t.Errorf("ExecuteLua(%q) unexpected error: %v", tc.script, err)
			continue
		}
		if got != tc.want {
			t.Errorf("ExecuteLua(%q) = %q, want %q", tc.script, got, tc.want)
		}
	}

	// Shared FormatFloatStandard unit tests
	if got, _ := FormatFloatStandard(math.Pow(2, 53)); got != "9007199254740992" {
		t.Errorf("FormatFloatStandard(2^53) = %q, want 9007199254740992", got)
	}
	if got, _ := FormatFloatStandard(math.Pow(2, 60)); got != "1.15292e+18" {
		t.Errorf("FormatFloatStandard(2^60) = %q, want 1.15292e+18", got)
	}
	if got, _ := FormatFloatStandard(1e20); got != "1e+20" {
		t.Errorf("FormatFloatStandard(1e20) = %q, want 1e+20", got)
	}
	if got, _ := FormatFloatStandard(1e300); got != "1e+300" {
		t.Errorf("FormatFloatStandard(1e300) = %q, want 1e+300", got)
	}
	if got, _ := FormatFloatStandard(1.0 / 3.0); got != "0.333333" {
		t.Errorf("FormatFloatStandard(1/3) = %q, want 0.333333", got)
	}
	if got, _ := FormatFloatStandard(0.1 + 0.2); got != "0.3" {
		t.Errorf("FormatFloatStandard(0.1+0.2) = %q, want 0.3", got)
	}
	if got, _ := FormatFloatStandard(-0.0); got != "0" {
		t.Errorf("FormatFloatStandard(-0.0) = %q, want 0", got)
	}

	// NaN and Inf rejection
	if _, err := FormatFloatStandard(math.NaN()); err == nil {
		t.Errorf("FormatFloatStandard(NaN) expected error, got nil")
	}
	if _, err := FormatFloatStandard(math.Inf(1)); err == nil {
		t.Errorf("FormatFloatStandard(+Inf) expected error, got nil")
	}
	if _, err := FormatFloatStandard(math.Inf(-1)); err == nil {
		t.Errorf("FormatFloatStandard(-Inf) expected error, got nil")
	}
	if got := formatFloat(0.0, false, 0); got != "0" {
		t.Errorf("formatFloat(0.0) = %q, want 0", got)
	}
	if got := formatFloat(math.NaN(), false, 0); got != "NaN" {
		t.Errorf("formatFloat(NaN) = %q, want NaN", got)
	}
	if got := formatFloat(math.Inf(1), false, 0); got != "Inf" {
		t.Errorf("formatFloat(Inf) = %q, want Inf", got)
	}
	if got := formatFloat(math.Inf(-1), false, 0); got != "-Inf" {
		t.Errorf("formatFloat(-Inf) = %q, want -Inf", got)
	}
}

func TestF2_DeterministicInstructionBudget(t *testing.T) {
	// Heavy loop from F2 review finding: 50,000 iterations must complete reproducibly
	script := `local s = ""; for i = 1, 50000 do s = s .. "x" end; return #s`
	got, err := ExecuteLua(script, nil)
	if err != nil {
		t.Fatalf("ExecuteLua(50,000 loop) unexpectedly timed out: %v", err)
	}
	if got != "50000" {
		t.Errorf("ExecuteLua(50,000 loop) = %q, want 50000", got)
	}

	// Infinite loop must terminate on instruction budget exceeded
	_, err = ExecuteLua("while true do end", nil)
	if err == nil {
		t.Fatal("expected error on while true loop")
	}
	if err.Error() != "ErrTimeout: instruction budget exceeded" {
		t.Errorf("while true error = %q, want %q", err.Error(), "ErrTimeout: instruction budget exceeded")
	}

	// Goto loop must terminate on instruction budget exceeded
	_, err = ExecuteLua("::loop:: goto loop", nil)
	if err == nil {
		t.Fatal("expected error on goto loop")
	}
	if err.Error() != "ErrTimeout: instruction budget exceeded" {
		t.Errorf("goto loop error = %q, want %q", err.Error(), "ErrTimeout: instruction budget exceeded")
	}
}

func TestF3_ArbitraryCapsRemoved(t *testing.T) {
	// table.insert beyond old 10,000 limit
	scriptTbl := `
		local t = {}
		for i = 1, 15000 do
			table.insert(t, i)
		end
		return #t
	`
	gotTbl, err := ExecuteLua(scriptTbl, nil)
	if err != nil {
		t.Fatalf("table.insert(15,000) failed: %v", err)
	}
	if gotTbl != "15000" {
		t.Errorf("table.insert(15,000) = %q, want 15000", gotTbl)
	}

	// string.rep beyond old 100 KB limit
	scriptStr := `return #string.rep("a", 150000)`
	gotStr, err := ExecuteLua(scriptStr, nil)
	if err != nil {
		t.Fatalf("string.rep(150,000) failed: %v", err)
	}
	if gotStr != "150000" {
		t.Errorf("string.rep(150,000) = %q, want 150000", gotStr)
	}

	// Allocator 10 MB heap cap is the single memory guard
	_, err = ExecuteLua(`return string.rep("a", 15000000)`, nil)
	if err == nil {
		t.Fatal("expected memory limit error for 15MB string")
	}
	if err.Error() != "ErrOverflow: memory limit" {
		t.Errorf("memory bomb error = %q, want %q", err.Error(), "ErrOverflow: memory limit")
	}
}

func TestF4_ErrorContract(t *testing.T) {
	// Syntax error: chunk prefix stripped, keep line N, no code echo
	_, err := ExecuteLua("local x =", nil)
	if err == nil {
		t.Fatal("expected syntax error")
	}
	expectedSyntax := "ErrLua: line 1: unexpected symbol near <eof>"
	if err.Error() != expectedSyntax {
		t.Errorf("syntax error = %q, want %q", err.Error(), expectedSyntax)
	}

	// Runtime error: chunk prefix stripped, keep line N
	_, err = ExecuteLua("local a = nil; return a + 1", nil)
	if err == nil {
		t.Fatal("expected runtime error")
	}
	if !strings.HasPrefix(err.Error(), "ErrLua: line 1:") {
		t.Errorf("runtime error = %q, want prefix ErrLua: line 1:", err.Error())
	}
	if strings.Contains(err.Error(), "\n") {
		t.Errorf("error contains newline: %q", err.Error())
	}

	// error({}) returns ErrLua: table
	_, err = ExecuteLua("error({})", nil)
	if err == nil {
		t.Fatal("expected error for error({})")
	}
	if err.Error() != "ErrLua: table" {
		t.Errorf("error({}) = %q, want ErrLua: table", err.Error())
	}

	// error("custom", 0) returns ErrLua: custom failure
	_, err = ExecuteLua("error('custom failure', 0)", nil)
	if err == nil {
		t.Fatal("expected error")
	}
	if err.Error() != "ErrLua: custom failure" {
		t.Errorf("error('custom failure', 0) = %q, want ErrLua: custom failure", err.Error())
	}

	// tostring of table and function returns without memory addresses
	gotTbl, err := ExecuteLua("return tostring({})", nil)
	if err != nil {
		t.Fatalf("tostring({}) failed: %v", err)
	}
	if gotTbl != `"table"` {
		t.Errorf("tostring({}) = %q, want \"table\"", gotTbl)
	}

	gotFn, err := ExecuteLua("return tostring(print)", nil)
	if err != nil {
		t.Fatalf("tostring(print) failed: %v", err)
	}
	if gotFn != `"function"` {
		t.Errorf("tostring(print) = %q, want \"function\"", gotFn)
	}

	// Control token escaping: control tokens escaped in values
	gotEsc, err := ExecuteLua(`return "<|obs|>"`, nil)
	if err != nil {
		t.Fatalf("control token test failed: %v", err)
	}
	if gotEsc != `"<\|obs|>"` {
		t.Errorf("control token escaping = %q, want \"<\\|obs|>\"", gotEsc)
	}
}

func TestF5_QuotedStrings(t *testing.T) {
	cases := []struct {
		script string
		want   string
	}{
		{`return "abc"`, `"abc"`},
		{`return "hello world"`, `"hello world"`},
		{`return "a", "b"`, `"a", "b"`},
		{`return {"a", "b"}`, `["a", "b"]`},
		{`return "newline\ntest"`, `"newline\ntest"`},
	}
	for _, tc := range cases {
		got, err := ExecuteLua(tc.script, nil)
		if err != nil {
			t.Errorf("ExecuteLua(%q) error: %v", tc.script, err)
			continue
		}
		if got != tc.want {
			t.Errorf("ExecuteLua(%q) = %q, want %q", tc.script, got, tc.want)
		}
	}
}

func TestF6_IntegerWrapAround(t *testing.T) {
	// math.maxinteger + 1 wraps silently to math.mininteger per Lua 5.4 semantics
	got, err := ExecuteLua("return math.maxinteger + 1", nil)
	if err != nil {
		t.Fatalf("math.maxinteger + 1 failed: %v", err)
	}
	if got != "-9223372036854775808" {
		t.Errorf("math.maxinteger + 1 = %q, want -9223372036854775808", got)
	}

	// return 1e20 is a float, displays per float formatting specification as "1e+20"
	gotFloat, err := ExecuteLua("return 1e20", nil)
	if err != nil {
		t.Fatalf("return 1e20 failed: %v", err)
	}
	if gotFloat != "1e+20" {
		t.Errorf("return 1e20 = %q, want 1e+20", gotFloat)
	}
}

func TestF7_TypedRegistersAndMultiReturn(t *testing.T) {
	// 1. Multi-returns bind as list in DispatchTool
	resp := DispatchTool("lua", "return 1, 2, 3")
	if !resp.Success {
		t.Fatalf("DispatchTool lua multi-return failed: %v", resp.Error)
	}
	if !strings.Contains(resp.Result, "[1, 2, 3]") {
		t.Errorf("DispatchTool lua multi-return result = %q, want '[1, 2, 3]'", resp.Result)
	}

	// 2. Typed register injection: bool, int, float, string
	rs := NewRegisterStore()
	rs.AutoBind(CalcVal{Kind: ValKindBool, Bool: true})            // r0
	rs.AutoBind(CalcVal{Kind: ValKindBool, Bool: false})           // r1
	rs.AutoBind(CalcVal{Kind: ValKindRat, Rat: big.NewRat(42, 1)}) // r2
	rs.AutoBind(CalcVal{Kind: ValKindFloat, Float: 3.14})          // r3
	rs.AutoBind(CalcVal{Kind: ValKindString, Str: "hello"})        // r4

	// Verify types inside Lua
	typeScript := `return type(r0), type(r1), type(r2), type(r3), type(r4)`
	typeGot, err := ExecuteLua(typeScript, rs)
	if err != nil {
		t.Fatalf("ExecuteLua(typeScript) failed: %v", err)
	}
	wantTypes := `"boolean", "boolean", "number", "number", "string"`
	if typeGot != wantTypes {
		t.Errorf("type script got %q, want %q", typeGot, wantTypes)
	}

	// Verify conditional branches on boolean registers
	boolTrueScript := `if r0 then return "is_true" else return "is_false" end`
	gotTrue, err := ExecuteLua(boolTrueScript, rs)
	if err != nil || gotTrue != `"is_true"` {
		t.Errorf("if r0 got %q, err %v; want \"is_true\"", gotTrue, err)
	}

	boolFalseScript := `if r1 then return "is_true" else return "is_false" end`
	gotFalse, err := ExecuteLua(boolFalseScript, rs)
	if err != nil || gotFalse != `"is_false"` {
		t.Errorf("if r1 got %q, err %v; want \"is_false\"", gotFalse, err)
	}
}

