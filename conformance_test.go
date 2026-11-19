package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

type ConformanceCase struct {
	Tool                string            `json:"tool"`
	Input               string            `json:"input"`
	ExpectedOutput      string            `json:"expected_output"`
	ExpectedErrorPrefix *string           `json:"expected_error_prefix"`
	RegistersBefore     map[string]string `json:"registers_before"`
	RegistersAfter      map[string]string `json:"registers_after"`
	Tags                []string          `json:"tags"`
	Clock               string            `json:"clock,omitempty"`
}

func TestConformanceSuite(t *testing.T) {
	candidates := []string{
		filepath.Join("conformance", "cases.jsonl"),
		filepath.Join("..", "conformance", "cases.jsonl"),
	}
	var casesPath string
	for _, c := range candidates {
		if _, err := os.Stat(c); err == nil {
			casesPath = c
			break
		}
	}
	if casesPath == "" {
		t.Fatalf("Could not locate conformance/cases.jsonl in candidate paths: %v", candidates)
	}
	file, err := os.Open(casesPath)
	if err != nil {
		t.Fatalf("Failed to open conformance suite at %s: %v", casesPath, err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	count := 0

	for scanner.Scan() {
		line := scanner.Text()
		if len(line) == 0 {
			continue
		}

		var tc ConformanceCase
		if err := json.Unmarshal([]byte(line), &tc); err != nil {
			t.Fatalf("Failed to decode JSON conformance case: %v", err)
		}
		count++

		t.Run(fmt.Sprintf("%s_%d", tc.Tool, count), func(t *testing.T) {
			globalRegisters.Reset()
			if tc.Clock != "" {
				clk, err := ParseClockTimestamp(tc.Clock)
				if err != nil {
					t.Fatalf("Failed to parse test case clock %q: %v", tc.Clock, err)
				}
				globalRegisters.SetClock(&clk)
				defer globalRegisters.SetClock(nil)
			} else {
				globalRegisters.SetClock(nil)
			}

			// Preload registers
			for k, v := range tc.RegistersBefore {
				_, _ = globalRegisters.ExecuteRegisterCmd(fmt.Sprintf("%s = %s", k, v))
			}

			resp := DispatchTool(tc.Tool, tc.Input)

			if tc.ExpectedErrorPrefix != nil && *tc.ExpectedErrorPrefix != "" {
				if resp.Success {
					t.Fatalf("Expected error prefix %q, but tool succeeded with result %q", *tc.ExpectedErrorPrefix, resp.Result)
				}
				if !strings.HasPrefix(resp.Error, *tc.ExpectedErrorPrefix) {
					t.Errorf("Error prefix mismatch: got %q, want prefix %q", resp.Error, *tc.ExpectedErrorPrefix)
				}
			} else {
				if !resp.Success {
					t.Fatalf("Expected success with %q, but tool failed with error %q", tc.ExpectedOutput, resp.Error)
				}
				if resp.Result != tc.ExpectedOutput {
					t.Errorf("Result mismatch:\n  got:  %q\n  want: %q", resp.Result, tc.ExpectedOutput)
				}
			}

			// Validate registers after
			for rk, wantVal := range tc.RegistersAfter {
				gotVal, err := globalRegisters.ExecuteRegisterCmd(rk)
				if err != nil || gotVal != wantVal {
					t.Errorf("Register %s mismatch: got %q (err: %v), want %q", rk, gotVal, err, wantVal)
				}
			}
		})
	}

	if err := scanner.Err(); err != nil {
		t.Fatalf("Error reading conformance cases: %v", err)
	}

	if count < 400 {
		t.Errorf("Expected >= 400 conformance cases, only executed %d", count)
	}
	t.Logf("Executed %d conformance cases successfully against Go reference harness.", count)
}
