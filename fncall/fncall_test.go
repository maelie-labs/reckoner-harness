package fncall

import (
	"bufio"
	"encoding/json"
	"os"
	"reflect"
	"strings"
	"testing"
)

type ConformanceCase struct {
	ID                  string          `json:"id"`
	Category            string          `json:"category"`
	Description         string          `json:"description"`
	Tools               json.RawMessage `json:"tools,omitempty"`
	ExpectedRender      string          `json:"expected_render,omitempty"`
	Call                string          `json:"call,omitempty"`
	ExpectedName        string          `json:"expected_name,omitempty"`
	ExpectedArguments   json.RawMessage `json:"expected_arguments,omitempty"`
	CanonicalCall       string          `json:"canonical_call,omitempty"`
	InvalidReason       string          `json:"invalid_reason,omitempty"`
	ExpectedErrorPrefix string          `json:"expected_error_prefix,omitempty"`
}

func loadConformanceCases(t *testing.T) []ConformanceCase {
	// Look for cases in conformance/fncall_cases.jsonl
	candidates := []string{
		"../../conformance/fncall_cases.jsonl",
		"../conformance/fncall_cases.jsonl",
		"conformance/fncall_cases.jsonl",
	}

	var path string
	for _, c := range candidates {
		if _, err := os.Stat(c); err == nil {
			path = c
			break
		}
	}
	if path == "" {
		t.Fatalf("Could not locate conformance/fncall_cases.jsonl")
	}

	f, err := os.Open(path)
	if err != nil {
		t.Fatalf("Failed to open %s: %v", path, err)
	}
	defer f.Close()

	var cases []ConformanceCase
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var c ConformanceCase
		if err := json.Unmarshal([]byte(line), &c); err != nil {
			t.Fatalf("Failed to unmarshal line: %v", err)
		}
		cases = append(cases, c)
	}
	if err := scanner.Err(); err != nil {
		t.Fatalf("Scanner error: %v", err)
	}
	return cases
}

func TestConformanceSuite(t *testing.T) {
	cases := loadConformanceCases(t)
	if len(cases) < 300 {
		t.Fatalf("Expected at least 300 conformance cases, got %d", len(cases))
	}

	renderCount := 0
	parseCount := 0
	invalidCount := 0

	for _, tc := range cases {
		tc := tc
		switch tc.Category {
		case "render":
			renderCount++
			t.Run(tc.ID+"_"+tc.Description, func(t *testing.T) {
				rendered, _, _, err := Render(tc.Tools)
				if err != nil {
					t.Fatalf("Render failed: %v", err)
				}
				if rendered != tc.ExpectedRender {
					t.Errorf("Render mismatch for %s:\nGot:\n%s\nExpected:\n%s", tc.ID, rendered, tc.ExpectedRender)
				}
			})

		case "parse_roundtrip":
			parseCount++
			t.Run(tc.ID+"_"+tc.Description, func(t *testing.T) {
				var tools []ParsedTool
				var mapper *NameMapper
				if len(tc.Tools) > 0 {
					var err error
					mapper = NewNameMapper()
					tools, err = ParseToolsList(tc.Tools, mapper)
					if err != nil {
						t.Fatalf("ParseToolsList failed: %v", err)
					}
				}

				name, args, err := Parse(tc.Call, tools, mapper)
				if err != nil {
					t.Fatalf("Parse failed: %v", err)
				}
				if name != tc.ExpectedName {
					t.Errorf("Name mismatch: got %s, expected %s", name, tc.ExpectedName)
				}

				// Check arguments JSON
				if len(tc.ExpectedArguments) > 0 {
					var expectedArgs map[string]interface{}
					if err := json.Unmarshal(tc.ExpectedArguments, &expectedArgs); err != nil {
						t.Fatalf("Failed to unmarshal expected args: %v", err)
					}
					// Compare normalized JSON
					argsBytes, _ := json.Marshal(args)
					var actualNorm map[string]interface{}
					_ = json.Unmarshal(argsBytes, &actualNorm)

					expBytes, _ := json.Marshal(expectedArgs)
					var expNorm map[string]interface{}
					_ = json.Unmarshal(expBytes, &expNorm)

					if !reflect.DeepEqual(actualNorm, expNorm) {
						t.Errorf("Args mismatch for %s:\nGot: %s\nExpected: %s", tc.ID, string(argsBytes), string(expBytes))
					}
				}

				// Check serialization
				ser, err := Serialize(name, args, tools)
				if err != nil {
					t.Fatalf("Serialize failed: %v", err)
				}
				if ser != tc.CanonicalCall {
					t.Errorf("Serialize mismatch for %s:\nGot: %s\nExpected: %s", tc.ID, ser, tc.CanonicalCall)
				}
			})

		case "invalid_call":
			invalidCount++
			t.Run(tc.ID+"_"+tc.Description, func(t *testing.T) {
				var tools []ParsedTool
				var mapper *NameMapper
				if len(tc.Tools) > 0 {
					mapper = NewNameMapper()
					var err error
					tools, err = ParseToolsList(tc.Tools, mapper)
					if err != nil {
						t.Fatalf("ParseToolsList failed: %v", err)
					}
				}

				name, args, parseErr := Parse(tc.Call, tools, mapper)
				var valErr error
				if parseErr == nil {
					valErr = Validate(name, args, tools)
				}

				var actualErr error
				if parseErr != nil {
					actualErr = parseErr
				} else {
					actualErr = valErr
				}

				if actualErr == nil {
					t.Fatalf("Expected call to fail, but it succeeded for %s: %s", tc.ID, tc.Call)
				}

				errMsg := actualErr.Error()
				if tc.ExpectedErrorPrefix != "" && !strings.HasPrefix(errMsg, tc.ExpectedErrorPrefix) {
					t.Errorf("Error prefix mismatch for %s: got '%s', expected prefix '%s'", tc.ID, errMsg, tc.ExpectedErrorPrefix)
				}
			})
		}
	}

	t.Logf("Tested %d total conformance cases (render: %d, parse_roundtrip: %d, invalid_call: %d)",
		len(cases), renderCount, parseCount, invalidCount)
}

func TestBijectiveNameMapping(t *testing.T) {
	mapper := NewNameMapper()
	s1 := mapper.SanitizeToolName("search-flights")
	if s1 != "search_flights" {
		t.Errorf("Expected search_flights, got %s", s1)
	}
	s2 := mapper.SanitizeToolName("search_flights")
	if s2 != "search_flights_2" {
		t.Errorf("Expected search_flights_2 on collision, got %s", s2)
	}
	s3 := mapper.SanitizeToolName("search.flights")
	if s3 != "search_flights_3" {
		t.Errorf("Expected search_flights_3 on second collision, got %s", s3)
	}

	// Reverse mapping
	if mapper.OriginalToolName("search_flights") != "search-flights" {
		t.Errorf("Reverse mapping failed for search_flights")
	}
	if mapper.OriginalToolName("search_flights_2") != "search_flights" {
		t.Errorf("Reverse mapping failed for search_flights_2")
	}
	if mapper.OriginalToolName("search_flights_3") != "search.flights" {
		t.Errorf("Reverse mapping failed for search_flights_3")
	}
}

func TestDescriptionCapping(t *testing.T) {
	desc := "First sentence. Second sentence. A very long third sentence that keeps going on and on."
	capped := CapDescription(desc, 35)
	if capped != "First sentence. Second sentence." {
		t.Errorf("Expected cut at second sentence, got '%s'", capped)
	}
}

func TestGrammarEmission(t *testing.T) {
	rawJSON := `[{"name": "test_fn", "parameters": {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}}]`
	tools, err := ParseToolsList([]byte(rawJSON), NewNameMapper())
	if err != nil {
		t.Fatalf("Failed to parse tools: %v", err)
	}
	ebnf := EmitGrammar(tools, "ebnf")
	if !strings.Contains(ebnf, "root ::= call_test_fn") {
		t.Errorf("EBNF missing root rule: %s", ebnf)
	}
	if !strings.Contains(ebnf, "call_test_fn ::=") {
		t.Errorf("EBNF missing call_test_fn: %s", ebnf)
	}
}

func TestIPCInterface(t *testing.T) {
	rawJSON := `[{"name": "ping", "description": "health check", "parameters": {"type": "object", "properties": {}}}]`
	rendered, err := HandleIPC(`{"op": "render", "tools": ` + rawJSON + `}`)
	if err != nil {
		t.Fatalf("HandleIPC render failed: %v", err)
	}
	if !strings.Contains(rendered, "def ping():") {
		t.Errorf("Rendered result unexpected: %s", rendered)
	}

	parseOut, err := HandleIPC(`{"op": "parse", "call": "ping()"}`)
	if err != nil {
		t.Fatalf("HandleIPC parse failed: %v", err)
	}
	var parseRes map[string]interface{}
	if err := json.Unmarshal([]byte(parseOut), &parseRes); err != nil {
		t.Fatalf("Parse output invalid JSON: %v", err)
	}
	if parseRes["name"] != "ping" {
		t.Errorf("Expected name 'ping', got %v", parseRes["name"])
	}
}

// A tool whose real name is a valid identifier keeps it; the sanitized name of another tool must not
// take it, in either input order.
func TestNameMapperReservesValidNames(t *testing.T) {
	for _, raw := range []string{
		`[{"name":"github.create-issue","parameters":{"type":"object","properties":{"from-date":{"type":"string"},"from_date":{"type":"string"}}}},{"name":"github_create_issue","parameters":{"type":"object","properties":{}}}]`,
		`[{"name":"github_create_issue","parameters":{"type":"object","properties":{}}},{"name":"github.create-issue","parameters":{"type":"object","properties":{"from-date":{"type":"string"},"from_date":{"type":"string"}}}}]`,
	} {
		tools, err := ParseToolsList([]byte(raw), NewNameMapper())
		if err != nil {
			t.Fatal(err)
		}
		got := map[string]string{}
		for _, tl := range tools {
			got[tl.OriginalName] = tl.SanitizedName
			if tl.OriginalName == "github.create-issue" {
				ps := map[string]string{}
				for _, p := range tl.Properties {
					ps[p.OriginalName] = p.SanitizedName
				}
				if ps["from_date"] != "from_date" || ps["from-date"] != "from_date_2" {
					t.Errorf("param mapping %v", ps)
				}
			}
		}
		if got["github_create_issue"] != "github_create_issue" || got["github.create-issue"] != "github_create_issue_2" {
			t.Errorf("tool mapping %v", got)
		}
	}
}

func TestNumericSerializationParityWithPython(t *testing.T) {
	testCases := []struct {
		name     string
		rawArgs  string
		expected string
	}{
		{"int_1", `{"x": 1}`, "test_fn(x=1)"},
		{"float_int_1", `{"x": 1.0}`, "test_fn(x=1)"},
		{"int_0", `{"x": 0}`, "test_fn(x=0)"},
		{"float_neg_zero", `{"x": -0.0}`, "test_fn(x=0)"},
		{"float_int_300", `{"x": 300.0}`, "test_fn(x=300)"},
		{"float_2_5", `{"x": 2.5}`, "test_fn(x=2.5)"},
		{"float_neg_5_5", `{"x": -5.5}`, "test_fn(x=-5.5)"},
		{"float_1e21", `{"x": 1e+21}`, "test_fn(x=1000000000000000000000)"},
		{"float_1e_minus_5", `{"x": 1e-05}`, "test_fn(x=1e-05)"},
		{"float_1234567_5", `{"x": 1234567.5}`, "test_fn(x=1234567.5)"},
		{"float_neg_1234567_5", `{"x": -1234567.5}`, "test_fn(x=-1234567.5)"},
		{"float_12345678_9", `{"x": 12345678.9}`, "test_fn(x=12345678.9)"},
		{"float_0_0001", `{"x": 0.0001}`, "test_fn(x=0.0001)"},
		{"float_neg_0_0001", `{"x": -0.0001}`, "test_fn(x=-0.0001)"},
		{"float_1e16", `{"x": 1e16}`, "test_fn(x=10000000000000000)"},
		{"float_1_25e16", `{"x": 1.25e16}`, "test_fn(x=12500000000000000)"},
		{"large_int_beyond_53", `{"x": 9007199254740993}`, "test_fn(x=9007199254740993)"},
		{"neg_large_int_beyond_53", `{"x": -9007199254740993}`, "test_fn(x=-9007199254740993)"},
		{"large_int_64", `{"x": 999999999999999999}`, "test_fn(x=999999999999999999)"},
		{"large_int_beyond_64", `{"x": 123456789012345678901234567890}`, "test_fn(x=123456789012345678901234567890)"},
		{"neg_large_int_beyond_64", `{"x": -123456789012345678901234567890}`, "test_fn(x=-123456789012345678901234567890)"},
		{"mixed_list", `{"l": [1, 2.5, 300.0, 1234567.5, 0.0001]}`, "test_fn(l=[1, 2.5, 300, 1234567.5, 0.0001])"},
		{"nested_dict", `{"d": {"a": 1.0, "b": 1234567.5, "c": 0.0001, "d": 1e-05}}`, `test_fn(d={"a": 1, "b": 1234567.5, "c": 0.0001, "d": 1e-05})`},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			payload := `{"op": "serialize", "name": "test_fn", "arguments": ` + tc.rawArgs + `}`
			out, err := HandleIPC(payload)
			if err != nil {
				t.Fatalf("HandleIPC failed: %v", err)
			}
			if out != tc.expected {
				t.Errorf("Mismatch for %s:\nGot:      %s\nExpected: %s", tc.name, out, tc.expected)
			}
		})
	}

	// Verify that NaN and Infinity in JSON are rejected with syntax error
	t.Run("reject_nan_infinity_json", func(t *testing.T) {
		nanPayload := `{"op": "serialize", "name": "test_fn", "arguments": {"x": NaN}}`
		if _, err := HandleIPC(nanPayload); err == nil {
			t.Errorf("Expected HandleIPC to reject JSON NaN, but got success")
		}
		infPayload := `{"op": "serialize", "name": "test_fn", "arguments": {"x": Infinity}}`
		if _, err := HandleIPC(infPayload); err == nil {
			t.Errorf("Expected HandleIPC to reject JSON Infinity, but got success")
		}
	})
}

