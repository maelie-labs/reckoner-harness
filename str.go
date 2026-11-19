package main

import (
	"encoding/hex"
	"fmt"
	"math/big"
	"strconv"
	"strings"
	"unicode/utf8"
)

// StrArg represents an argument parsed from a str or date function call.
type StrArg struct {
	IsQuoted bool
	RawStr   string
	IsReg    bool
	RegIdx   int
}

// EvaluateStr evaluates a str tool call: e.g. count("strawberry", "r"), slice("mathematics", 0, 4), to_hex(255).
func EvaluateStr(payload string) (string, error) {
	val, err := EvaluateStrWithStore(payload, globalRegisters)
	if err != nil {
		return "", err
	}
	return val.String(), nil
}

// EvaluateStrWithStore evaluates a str tool call against the given typed RegisterStore.
func EvaluateStrWithStore(payload string, store *RegisterStore) (CalcVal, error) {
	payload = strings.TrimSpace(payload)
	openParen := strings.IndexByte(payload, '(')
	if openParen == -1 || !strings.HasSuffix(payload, ")") {
		return CalcVal{}, fmt.Errorf("ErrSyntax: malformed str call, expected func(args...)")
	}

	fnName := strings.TrimSpace(payload[:openParen])
	argsBody := strings.TrimSpace(payload[openParen+1 : len(payload)-1])

	args, err := parseStrArgsWithStore(argsBody)
	if err != nil {
		return CalcVal{}, err
	}

	resolveString := func(arg StrArg, fn string) (string, error) {
		if arg.IsReg {
			if store == nil {
				return "", fmt.Errorf("ErrUnsetRegister: <|r%d|> is empty", arg.RegIdx)
			}
			val, err := store.Get(arg.RegIdx)
			if err != nil {
				return "", err
			}
			if val.Kind == ValKindString {
				return val.Str, nil
			}
			if val.Kind == ValKindDate {
				return val.DateStr, nil
			}
			return "", fmt.Errorf("ErrTypeMismatch: %s expects string argument, got %s", fn, val.String())
		}
		return arg.RawStr, nil
	}

	resolveInt := func(arg StrArg, fn string) (int64, error) {
		if arg.IsReg {
			if store == nil {
				return 0, fmt.Errorf("ErrUnsetRegister: <|r%d|> is empty", arg.RegIdx)
			}
			val, err := store.Get(arg.RegIdx)
			if err != nil {
				return 0, err
			}
			if val.Kind == ValKindRat && val.Rat != nil && val.Rat.IsInt() {
				return val.Rat.Num().Int64(), nil
			}
			return 0, fmt.Errorf("ErrTypeMismatch: %s expects integer argument, got %s", fn, val.String())
		}
		n, err := strconv.ParseInt(strings.TrimSpace(arg.RawStr), 10, 64)
		if err != nil {
			return 0, fmt.Errorf("ErrSyntax: %s expects integer argument, got %q", fn, arg.RawStr)
		}
		return n, nil
	}

	switch fnName {
	case "count":
		if len(args) != 2 {
			return CalcVal{}, fmt.Errorf("ErrSyntax: count expects 2 arguments: (target_str, search_char)")
		}
		target, err := resolveString(args[0], "count")
		if err != nil {
			return CalcVal{}, err
		}
		search, err := resolveString(args[1], "count")
		if err != nil {
			return CalcVal{}, err
		}
		if !utf8.ValidString(target) || !utf8.ValidString(search) {
			return CalcVal{}, fmt.Errorf("ErrSyntax: malformed UTF-8 byte sequence")
		}
		count := strings.Count(target, search)
		return CalcVal{
			Kind: ValKindRat,
			Rat:  big.NewRat(int64(count), 1),
		}, nil

	case "reverse":
		if len(args) != 1 {
			return CalcVal{}, fmt.Errorf("ErrSyntax: reverse expects 1 argument: (str)")
		}
		s, err := resolveString(args[0], "reverse")
		if err != nil {
			return CalcVal{}, err
		}
		if !utf8.ValidString(s) {
			return CalcVal{}, fmt.Errorf("ErrSyntax: malformed UTF-8 byte sequence")
		}
		runes := []rune(s)
		for i, j := 0, len(runes)-1; i < j; i, j = i+1, j-1 {
			runes[i], runes[j] = runes[j], runes[i]
		}
		return CalcVal{
			Kind: ValKindString,
			Str:  string(runes),
		}, nil

	case "slice":
		if len(args) != 3 {
			return CalcVal{}, fmt.Errorf("ErrSyntax: slice expects 3 arguments: (str, start, end)")
		}
		s, err := resolveString(args[0], "slice")
		if err != nil {
			return CalcVal{}, err
		}
		if !utf8.ValidString(s) {
			return CalcVal{}, fmt.Errorf("ErrSyntax: malformed UTF-8 byte sequence")
		}
		start, err1 := resolveInt(args[1], "slice")
		end, err2 := resolveInt(args[2], "slice")
		if err1 != nil {
			return CalcVal{}, err1
		}
		if err2 != nil {
			return CalcVal{}, err2
		}
		runes := []rune(s)
		n := int64(len(runes))
		if start < 0 || end < start || end > n {
			return CalcVal{}, fmt.Errorf("ErrOutOfBounds: slice index [%d:%d] out of range for length %d", start, end, n)
		}
		return CalcVal{
			Kind: ValKindString,
			Str:  string(runes[start:end]),
		}, nil

	case "len":
		if len(args) != 1 {
			return CalcVal{}, fmt.Errorf("ErrSyntax: len expects 1 argument: (str)")
		}
		s, err := resolveString(args[0], "len")
		if err != nil {
			return CalcVal{}, err
		}
		if !utf8.ValidString(s) {
			return CalcVal{}, fmt.Errorf("ErrSyntax: malformed UTF-8 byte sequence")
		}
		return CalcVal{
			Kind: ValKindRat,
			Rat:  big.NewRat(int64(utf8.RuneCountInString(s)), 1),
		}, nil

	case "to_hex":
		if len(args) != 1 {
			return CalcVal{}, fmt.Errorf("ErrSyntax: to_hex expects 1 argument: (val)")
		}
		if args[0].IsQuoted {
			s, err := resolveString(args[0], "to_hex")
			if err != nil {
				return CalcVal{}, err
			}
			return CalcVal{
				Kind: ValKindString,
				Str:  strings.ToUpper(hex.EncodeToString([]byte(s))),
			}, nil
		}
		val, err := resolveInt(args[0], "to_hex")
		if err == nil {
			return CalcVal{
				Kind: ValKindString,
				Str:  strings.ToUpper(strconv.FormatInt(val, 16)),
			}, nil
		}
		s, err := resolveString(args[0], "to_hex")
		if err != nil {
			return CalcVal{}, err
		}
		return CalcVal{
			Kind: ValKindString,
			Str:  strings.ToUpper(hex.EncodeToString([]byte(s))),
		}, nil

	case "from_hex":
		if len(args) != 1 {
			return CalcVal{}, fmt.Errorf("ErrSyntax: from_hex expects 1 argument: (hex_str)")
		}
		s, err := resolveString(args[0], "from_hex")
		if err != nil {
			return CalcVal{}, err
		}
		bytes, err := hex.DecodeString(s)
		if err != nil {
			return CalcVal{}, fmt.Errorf("ErrSyntax: invalid hex string: %v", err)
		}
		return CalcVal{
			Kind: ValKindString,
			Str:  string(bytes),
		}, nil

	case "upper":
		if len(args) != 1 {
			return CalcVal{}, fmt.Errorf("ErrSyntax: upper expects 1 argument: (str)")
		}
		s, err := resolveString(args[0], "upper")
		if err != nil {
			return CalcVal{}, err
		}
		return CalcVal{
			Kind: ValKindString,
			Str:  strings.ToUpper(s),
		}, nil

	case "lower":
		if len(args) != 1 {
			return CalcVal{}, fmt.Errorf("ErrSyntax: lower expects 1 argument: (str)")
		}
		s, err := resolveString(args[0], "lower")
		if err != nil {
			return CalcVal{}, err
		}
		return CalcVal{
			Kind: ValKindString,
			Str:  strings.ToLower(s),
		}, nil

	case "trim":
		if len(args) != 1 {
			return CalcVal{}, fmt.Errorf("ErrSyntax: trim expects 1 argument: (str)")
		}
		s, err := resolveString(args[0], "trim")
		if err != nil {
			return CalcVal{}, err
		}
		return CalcVal{
			Kind: ValKindString,
			Str:  strings.TrimSpace(s),
		}, nil

	case "contains":
		if len(args) != 2 {
			return CalcVal{}, fmt.Errorf("ErrSyntax: contains expects 2 arguments: (target, search)")
		}
		target, err := resolveString(args[0], "contains")
		if err != nil {
			return CalcVal{}, err
		}
		search, err := resolveString(args[1], "contains")
		if err != nil {
			return CalcVal{}, err
		}
		return CalcVal{
			Kind: ValKindBool,
			Bool: strings.Contains(target, search),
		}, nil

	case "to_bin":
		if len(args) != 1 {
			return CalcVal{}, fmt.Errorf("ErrSyntax: to_bin expects 1 argument: (val)")
		}
		val, err := resolveInt(args[0], "to_bin")
		if err != nil {
			return CalcVal{}, err
		}
		return CalcVal{
			Kind: ValKindString,
			Str:  strconv.FormatInt(val, 2),
		}, nil

	case "to_ascii":
		if len(args) != 1 {
			return CalcVal{}, fmt.Errorf("ErrSyntax: to_ascii expects 1 argument: (char)")
		}
		s, err := resolveString(args[0], "to_ascii")
		if err != nil {
			return CalcVal{}, err
		}
		if !utf8.ValidString(s) || utf8.RuneCountInString(s) == 0 {
			return CalcVal{}, fmt.Errorf("ErrSyntax: expected non-empty character")
		}
		r := []rune(s)[0]
		return CalcVal{
			Kind: ValKindRat,
			Rat:  big.NewRat(int64(r), 1),
		}, nil

	case "replace":
		if len(args) != 3 {
			return CalcVal{}, fmt.Errorf("ErrSyntax: replace expects 3 arguments: (str, old, new)")
		}
		s, err := resolveString(args[0], "replace")
		if err != nil {
			return CalcVal{}, err
		}
		old, err := resolveString(args[1], "replace")
		if err != nil {
			return CalcVal{}, err
		}
		newStr, err := resolveString(args[2], "replace")
		if err != nil {
			return CalcVal{}, err
		}
		if !utf8.ValidString(s) || !utf8.ValidString(old) || !utf8.ValidString(newStr) {
			return CalcVal{}, fmt.Errorf("ErrSyntax: malformed UTF-8 byte sequence")
		}
		return CalcVal{
			Kind: ValKindString,
			Str:  strings.ReplaceAll(s, old, newStr),
		}, nil

	case "find":
		if len(args) != 2 {
			return CalcVal{}, fmt.Errorf("ErrSyntax: find expects 2 arguments: (str, sub)")
		}
		s, err := resolveString(args[0], "find")
		if err != nil {
			return CalcVal{}, err
		}
		sub, err := resolveString(args[1], "find")
		if err != nil {
			return CalcVal{}, err
		}
		if !utf8.ValidString(s) || !utf8.ValidString(sub) {
			return CalcVal{}, fmt.Errorf("ErrSyntax: malformed UTF-8 byte sequence")
		}
		byteIdx := strings.Index(s, sub)
		if byteIdx == -1 {
			return CalcVal{
				Kind: ValKindRat,
				Rat:  big.NewRat(-1, 1),
			}, nil
		}
		runeIdx := utf8.RuneCountInString(s[:byteIdx])
		return CalcVal{
			Kind: ValKindRat,
			Rat:  big.NewRat(int64(runeIdx), 1),
		}, nil

	case "count_vowels":
		if len(args) != 1 {
			return CalcVal{}, fmt.Errorf("ErrSyntax: count_vowels expects 1 argument: (str)")
		}
		s, err := resolveString(args[0], "count_vowels")
		if err != nil {
			return CalcVal{}, err
		}
		if !utf8.ValidString(s) {
			return CalcVal{}, fmt.Errorf("ErrSyntax: malformed UTF-8 byte sequence")
		}
		cnt := 0
		for _, r := range s {
			switch r {
			case 'a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U':
				cnt++
			}
		}
		return CalcVal{
			Kind: ValKindRat,
			Rat:  big.NewRat(int64(cnt), 1),
		}, nil

	case "is_palindrome":
		if len(args) != 1 {
			return CalcVal{}, fmt.Errorf("ErrSyntax: is_palindrome expects 1 argument: (str)")
		}
		s, err := resolveString(args[0], "is_palindrome")
		if err != nil {
			return CalcVal{}, err
		}
		if !utf8.ValidString(s) {
			return CalcVal{}, fmt.Errorf("ErrSyntax: malformed UTF-8 byte sequence")
		}
		runes := []rune(s)
		for i, j := 0, len(runes)-1; i < j; i, j = i+1, j-1 {
			if runes[i] != runes[j] {
				return CalcVal{
					Kind: ValKindBool,
					Bool: false,
				}, nil
			}
		}
		return CalcVal{
			Kind: ValKindBool,
			Bool: true,
		}, nil

	default:
		return CalcVal{}, fmt.Errorf("ErrSyntax: unknown str function %q", fnName)
	}
}

func parseStrArgsWithStore(argsBody string) ([]StrArg, error) {
	argsBody = strings.TrimSpace(argsBody)
	if argsBody == "" {
		return nil, nil
	}

	var args []StrArg
	runes := []rune(argsBody)
	n := len(runes)
	i := 0

	for i < n {
		// Skip whitespace before argument
		for i < n && (runes[i] == ' ' || runes[i] == '\t' || runes[i] == '\r' || runes[i] == '\n') {
			i++
		}
		if i >= n {
			break
		}

		if runes[i] == '"' || runes[i] == '\'' {
			// Quoted string argument: keep whitespace inside byte-exact 
			quote := runes[i]
			i++
			var buf strings.Builder
			closed := false
			for i < n {
				r := runes[i]
				if r == '\\' && i+1 < n {
					nextR := runes[i+1]
					switch nextR {
					case 'n':
						buf.WriteRune('\n')
					case 't':
						buf.WriteRune('\t')
					case 'r':
						buf.WriteRune('\r')
					case '\\':
						buf.WriteRune('\\')
					case '"':
						buf.WriteRune('"')
					case '\'':
						buf.WriteRune('\'')
					default:
						buf.WriteRune('\\')
						buf.WriteRune(nextR)
					}
					i += 2
					continue
				}
				if r == quote {
					closed = true
					i++
					break
				}
				buf.WriteRune(r)
				i++
			}
			if !closed {
				return nil, fmt.Errorf("ErrSyntax: unclosed string literal in arguments")
			}
			args = append(args, StrArg{
				IsQuoted: true,
				RawStr:   buf.String(),
			})
		} else {
			// Unquoted argument: could be a register <|rN|>, number, or identifier
			start := i
			for i < n && runes[i] != ',' {
				i++
			}
			raw := strings.TrimSpace(string(runes[start:i]))
			// Check if register tag <|r0|> .. <|r15|>
			if strings.HasPrefix(raw, "<|r") && strings.HasSuffix(raw, "|>") && (len(raw) == 6 || len(raw) == 7) {
				numStr := raw[3 : len(raw)-2]
				idx, err := strconv.Atoi(numStr)
				if err == nil && idx >= 0 && idx < NumRegisters {
					args = append(args, StrArg{
						IsReg:  true,
						RegIdx: idx,
					})
				} else {
					args = append(args, StrArg{
						IsQuoted: false,
						RawStr:   raw,
					})
				}
			} else {
				args = append(args, StrArg{
					IsQuoted: false,
					RawStr:   raw,
				})
			}
		}

		// Skip trailing whitespace after argument
		for i < n && (runes[i] == ' ' || runes[i] == '\t' || runes[i] == '\r' || runes[i] == '\n') {
			i++
		}
		if i < n && runes[i] == ',' {
			i++ // Skip comma
		}
	}

	return args, nil
}
