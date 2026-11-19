package main

import (
	"crypto/sha256"
	"strings"
	"sync"
	"testing"
	"time"
)

// TestDeterminismUnderStress verifies that heavy-but-legal inputs produce identical
// results under idle conditions and under simulated CPU stress .
func TestDeterminismUnderStress(t *testing.T) {
	testCases := []struct {
		name    string
		tool    string
		payload string
	}{
		{"HeavyFactor", "calc", "factor(1000000007)"},
		{"HeavyTotient", "calc", "totient(1000000000)"},
		{"CubicSolve", "solve", "x^3 - 6*x^2 + 11*x - 6 = 0, x"},
		{"ComplexCalc", "calc", "sum_range(1, 1000) + comb(30, 10)"},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			store1 := NewRegisterStore()
			respIdle := DispatchToolWithStore(store1, tc.tool, tc.payload)
			if !respIdle.Success {
				t.Fatalf("Idle evaluation failed: %s", respIdle.Error)
			}

			// Launch concurrent CPU stress goroutines
			stopStress := make(chan struct{})
			var wg sync.WaitGroup
			numWorkers := 8
			for i := 0; i < numWorkers; i++ {
				wg.Add(1)
				go func(id int) {
					defer wg.Done()
					h := sha256.New()
					buf := make([]byte, 1024)
					for {
						select {
						case <-stopStress:
							return
						default:
							h.Write(buf)
							_ = h.Sum(nil)
						}
					}
				}(i)
			}

			// Small sleep to ensure stress goroutines are fully active
			time.Sleep(10 * time.Millisecond)

			// Execute under stress
			store2 := NewRegisterStore()
			respStress := DispatchToolWithStore(store2, tc.tool, tc.payload)

			// Stop stress workers
			close(stopStress)
			wg.Wait()

			if !respStress.Success {
				t.Fatalf("Stress evaluation failed: %s", respStress.Error)
			}

			if respIdle.Result != respStress.Result {
				t.Errorf("Determinism failure on %s: idle=%q, stress=%q", tc.payload, respIdle.Result, respStress.Result)
			}
		})
	}
}

// TestStepBudgetDeterminism verifies that budget exhaustion is strictly deterministic
// and errors with ErrTimeout: step budget exceeded rather than wall-clock dependent.
func TestStepBudgetDeterminism(t *testing.T) {
	// A prime near 10^14 requires > 10^6 steps in trial division
	payload := "factor(99999999999973)"

	store1 := NewRegisterStore()
	respIdle := DispatchToolWithStore(store1, "calc", payload)
	if respIdle.Success || !strings.Contains(respIdle.Error, "step budget exceeded") {
		t.Fatalf("Expected 'step budget exceeded', got success=%v, err=%s", respIdle.Success, respIdle.Error)
	}

	// Under stress
	stopStress := make(chan struct{})
	var wg sync.WaitGroup
	for i := 0; i < 6; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			h := sha256.New()
			b := make([]byte, 512)
			for {
				select {
				case <-stopStress:
					return
				default:
					h.Write(b)
					_ = h.Sum(nil)
				}
			}
		}()
	}

	time.Sleep(5 * time.Millisecond)
	store2 := NewRegisterStore()
	respStress := DispatchToolWithStore(store2, "calc", payload)
	close(stopStress)
	wg.Wait()

	if respStress.Success || !strings.Contains(respStress.Error, "step budget exceeded") {
		t.Fatalf("Expected 'step budget exceeded' under stress, got success=%v, err=%s", respStress.Success, respStress.Error)
	}

	if respIdle.Error != respStress.Error {
		t.Errorf("Error string mismatch: idle=%q, stress=%q", respIdle.Error, respStress.Error)
	}
}

// TestTimeoutDoesNotBindRegisters verifies that a timed-out tool call never binds to registers
// and never mutates the register store, even if the worker goroutine completes later.
func TestTimeoutDoesNotBindRegisters(t *testing.T) {
	store := NewRegisterStore()

	// Dispatch an algebraic factorization with a 1µs deadline so it times out,
	// while the background worker computation succeeds.
	resp := DispatchToolWithStoreAndDeadline(store, "calc", "factor(x^4 - 10*x^2 + 9)", 1*time.Microsecond)

	if resp.Success {
		t.Fatalf("Expected timeout failure, got success: %v", resp.Result)
	}
	if !strings.Contains(resp.Error, "ErrTimeout") {
		t.Fatalf("Expected ErrTimeout, got %q", resp.Error)
	}

	// Wait 50ms for worker goroutine to complete its work
	time.Sleep(50 * time.Millisecond)

	// Verify r0 was NOT bound because late-completing workers never commit bindings
	_, err := store.Get(0)
	if err == nil {
		t.Fatalf("Expected r0 to be unbound after timeout, but found bound value")
	}

	// Verify next successful call binds cleanly to r0
	respNext := DispatchToolWithStore(store, "calc", "35 * 12 + 14")
	if !respNext.Success {
		t.Fatalf("Subsequent call failed: %v", respNext.Error)
	}
	if respNext.Result != "<|r0|> = 434" {
		t.Fatalf("Subsequent call expected '<|r0|> = 434', got %q", respNext.Result)
	}
}
