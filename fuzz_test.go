package main

import (
	"testing"
)

func FuzzCalc(f *testing.F) {
	seeds := []string{
		"1 + 1",
		"482 * 19 - 14",
		"2^-3",
		"sqrt(144) * (2^3)",
		"5 // 2",
		"7 % 3",
		"det(1, 2, 3, 4)",
		"det3(1, 2, 3, 4, 5, 6, 7, 8, 9)",
		"dot(1, 2, 3, 4)",
		"base(255, 16)",
		"cbrt(27)",
		"convert(10 km, m)",
		"max(10 m, 5 m)",
		"abs(-42)",
		"sum([1, 2, 3, 4])",
		"2^1000000",
		"1 / 0",
		"((((1 + 2))))",
		"25 * pi",
		"(1 + sqrt(5)) / 2",
		"3 - 4*i",
		"expand((x + 1)*(x + 2))",
		"factor(x^2 - 5*x + 6)",
		"dist([1, 2], [4, 6])",
		"[[1, 2], [3, 4]] @ [[2, 0], [1, 2]]",
		"det([[1, 2], [3, 4]])",
		"sum(k, k, 1, 10)",
		"sin(pi/6)",
		"cos(pi/4)",
		"tan(pi/3)",
		"log(8, 2)",
		"heron(3, 4, 5)",
		"circ_area(5)",
		"sqrt(3 + 2*sqrt(2))",
	}
	for _, seed := range seeds {
		f.Add(seed)
	}

	f.Fuzz(func(t *testing.T, expr string) {
		defer func() {
			if r := recover(); r != nil {
				t.Fatalf("panic in FuzzCalc on expr %q: %v", expr, r)
			}
		}()
		resp := DispatchTool("calc", expr)
		_ = resp
	})
}

func FuzzSolve(f *testing.F) {
	seeds := []string{
		"x + 2 = 5, x",
		"x^2 - 4 = 0, x",
		"2*x^2 + 5*x - 3 = 0, x",
		"x^2 + 1 = 0, x",
		"3*x + 10 = 25, x",
		"x = x, x",
		"0 = 1, x",
		"-x^2 + 4 = 0, x",
		"x^3 - 8 = 0, x",
		"x^2 - 16 > 0",
		"x^2 + 5*x < 6",
		"2*x + 1 >= 7",
		"1/x > 1",
		"abs(x - 3) < 5",
	}
	for _, seed := range seeds {
		f.Add(seed)
	}

	f.Fuzz(func(t *testing.T, expr string) {
		defer func() {
			if r := recover(); r != nil {
				t.Fatalf("panic in FuzzSolve on expr %q: %v", expr, r)
			}
		}()
		resp := DispatchTool("solve", expr)
		_ = resp
	})
}

func FuzzStr(f *testing.F) {
	seeds := []string{
		`len("hello")`,
		`len("")`,
		`slice("hello", 0, 2)`,
		`reverse("hello")`,
		`lower("HELLO")`,
		`upper("world")`,
		`to_hex("café")`,
		`from_hex("636166c3a9")`,
		`count("banana", "a")`,
		`find("hello", "l")`,
		`contains("hello", "ll")`,
		`replace("hello world", "world", "there")`,
	}
	for _, seed := range seeds {
		f.Add(seed)
	}

	f.Fuzz(func(t *testing.T, expr string) {
		defer func() {
			if r := recover(); r != nil {
				t.Fatalf("panic in FuzzStr on expr %q: %v", expr, r)
			}
		}()
		resp := DispatchTool("str", expr)
		_ = resp
	})
}

func FuzzDate(f *testing.F) {
	seeds := []string{
		`add("2024-01-31", 1, "month")`,
		`add("2024-01-01", 10, "day")`,
		`diff("2024-01-01", "2024-01-10")`,
		`day_of_week("2024-03-14")`,
		`next_weekday("2024-03-14", "thursday")`,
		`add("2024-02-29", 1, "year")`,
		`diff("2025-02-30", "2025-03-01")`,
	}
	for _, seed := range seeds {
		f.Add(seed)
	}

	f.Fuzz(func(t *testing.T, expr string) {
		defer func() {
			if r := recover(); r != nil {
				t.Fatalf("panic in FuzzDate on expr %q: %v", expr, r)
			}
		}()
		resp := DispatchTool("date", expr)
		_ = resp
	})
}

func FuzzFind(f *testing.F) {
	seeds := []string{
		`find("invoice")`,
		`find("1250")`,
		`find("1,250")`,
		`find("ch_9812")`,
		`find("nonexistent_key_xyz")`,
		`find("target_alpha", 0)`,
		`find("")`,
		`lines(0, 0, 1)`,
		`find "invoice"`,
		`find “quoted” 0`,
		`find("a-b-c")`,
		`find("a—b—c")`,
	}
	for _, seed := range seeds {
		f.Add(seed)
	}

	tok, err := GetGlobalTokenizer()
	if err != nil {
		f.Fatalf("Failed to get tokenizer: %v", err)
	}

	f.Fuzz(func(t *testing.T, payload string) {
		defer func() {
			if r := recover(); r != nil {
				t.Fatalf("panic in FuzzFind on payload %q: %v", payload, r)
			}
		}()

		cs := NewContextStore(tok)
		// Store a couple sample messages
		cs.StoreMessage([]int{25049, 12511, 326, 8977, 153}, nil, nil)
		cs.StoreMessage([]int{100, 200, 326, 300, 400}, nil, nil)

		_, _ = DispatchContextTool(cs, "find", payload)
	})
}
