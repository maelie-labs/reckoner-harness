package fncall

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

// IsValidIdentifier reports whether s is a valid ASCII Python identifier ([a-zA-Z_][a-zA-Z0-9_]*).
func IsValidIdentifier(s string) bool {
	return isValidIdentifier(s)
}

// SerializeCall formats a tool name and arguments map into the canonical pythonic call string (§3.2).
func SerializeCall(toolName string, args map[string]interface{}, tool *ParsedTool) (string, error) {
	return serializeCallInternal(toolName, args, tool, false, nil)
}

// SerializeCallWithEscape formats a tool name and arguments map into the canonical pythonic call string (§3.2)
// without identifier validation, applying escapeFn (if non-nil) to undeclared tool names, argument keys,
// and multi-line triple-quoted string bodies.
func SerializeCallWithEscape(toolName string, args map[string]interface{}, tool *ParsedTool, escapeFn func(string) string) (string, error) {
	return serializeCallInternal(toolName, args, tool, false, escapeFn)
}

// SerializeCallStrict formats a tool name and arguments map into the canonical pythonic call string,
// enforcing identifier validation on unknown tool names and extra/unknown argument keys, and applying
// escapeFn (if non-nil) to multi-line triple-quoted string bodies.
func SerializeCallStrict(toolName string, args map[string]interface{}, tool *ParsedTool, escapeFn func(string) string) (string, error) {
	return serializeCallInternal(toolName, args, tool, true, escapeFn)
}

func serializeCallInternal(toolName string, args map[string]interface{}, tool *ParsedTool, strict bool, escapeFn func(string) string) (string, error) {
	var orderedKeys []string
	callName := toolName

	if tool != nil {
		callName = tool.SanitizedName
		// Use parameter order from tool schema
		seen := make(map[string]bool)
		for _, prop := range tool.Properties {
			k := prop.SanitizedName
			origK := prop.OriginalName
			if _, exists := args[k]; exists {
				orderedKeys = append(orderedKeys, k)
				seen[k] = true
				seen[origK] = true
			} else if _, exists := args[origK]; exists {
				orderedKeys = append(orderedKeys, k)
				seen[k] = true
				seen[origK] = true
			}
		}
		// Any remaining extra keys in args
		var extraKeys []string
		for k := range args {
			if !seen[k] {
				if strict && !isValidIdentifier(k) {
					return "", fmt.Errorf("ErrArgs: invalid tool argument key %q (must be a valid identifier)", k)
				}
				extraKeys = append(extraKeys, k)
			}
		}
		sort.Strings(extraKeys)
		orderedKeys = append(orderedKeys, extraKeys...)
	} else {
		if strict && !isValidIdentifier(toolName) {
			return "", fmt.Errorf("ErrArgs: invalid undeclared tool name %q (must match a declared tool or be a valid identifier)", toolName)
		}
		if !strict && escapeFn != nil {
			callName = escapeFn(toolName)
		}
		// Sort keys alphabetically
		for k := range args {
			if strict && !isValidIdentifier(k) {
				return "", fmt.Errorf("ErrArgs: invalid tool argument key %q (must be a valid identifier)", k)
			}
			orderedKeys = append(orderedKeys, k)
		}
		sort.Strings(orderedKeys)
	}

	var kwargParts []string
	for _, k := range orderedKeys {
		val, exists := args[k]
		if !exists && tool != nil {
			for _, prop := range tool.Properties {
				if prop.SanitizedName == k {
					val = args[prop.OriginalName]
					break
				}
			}
		}
		valStr, err := SerializeValueWithEscape(val, escapeFn)
		if err != nil {
			return "", err
		}
		keyStr := k
		if !strict && escapeFn != nil {
			keyStr = escapeFn(k)
		}
		kwargParts = append(kwargParts, fmt.Sprintf("%s=%s", keyStr, valStr))
	}

	return fmt.Sprintf("%s(%s)", callName, strings.Join(kwargParts, ", ")), nil
}

// SerializeValue serializes an interface{} value into a Pythonic literal string.
func SerializeValue(val interface{}) (string, error) {
	return SerializeValueWithEscape(val, nil)
}

// SerializeValueWithEscape serializes an interface{} value into a Pythonic literal string,
// applying escapeFn (if non-nil) to multi-line triple-quoted string bodies.
func SerializeValueWithEscape(val interface{}, escapeFn func(string) string) (string, error) {
	astVal, err := ValueFromInterface(val)
	if err != nil {
		return "", err
	}
	return SerializeASTValueWithEscape(astVal, escapeFn), nil
}

// SerializeASTValue serializes an AST Value into its canonical Python literal representation.
func SerializeASTValue(v Value) string {
	return SerializeASTValueWithEscape(v, nil)
}

// SerializeASTValueWithEscape serializes an AST Value into its canonical Python literal representation,
// applying escapeFn (if non-nil) to multi-line triple-quoted string bodies.
func SerializeASTValueWithEscape(v Value, escapeFn func(string) string) string {
	switch v.Type {
	case ValueTypeString:
		// If string contains newline and no triple quote, use """..."""
		if strings.Contains(v.StrVal, "\n") && !strings.Contains(v.StrVal, "\"\"\"") {
			s := v.StrVal
			if escapeFn != nil {
				s = escapeFn(s)
			}
			return fmt.Sprintf("\"\"\"%s\"\"\"", s)
		}
		// Otherwise standard JSON string
		b, _ := json.Marshal(v.StrVal)
		return string(b)
	case ValueTypeNumber:
		return v.NumStr
	case ValueTypeBool:
		if v.BoolVal {
			return "True"
		}
		return "False"
	case ValueTypeNone:
		return "None"
	case ValueTypeList:
		var items []string
		for _, item := range v.ListVal {
			items = append(items, SerializeASTValueWithEscape(item, escapeFn))
		}
		return fmt.Sprintf("[%s]", strings.Join(items, ", "))
	case ValueTypeDict:
		var entries []string
		// Sort dict keys for deterministic output
		entriesSorted := make([]DictEntry, len(v.DictVal))
		copy(entriesSorted, v.DictVal)
		sort.SliceStable(entriesSorted, func(i, j int) bool {
			return entriesSorted[i].Key < entriesSorted[j].Key
		})
		for _, entry := range entriesSorted {
			kB, _ := json.Marshal(entry.Key)
			vStr := SerializeASTValueWithEscape(entry.Val, escapeFn)
			entries = append(entries, fmt.Sprintf("%s: %s", string(kB), vStr))
		}
		return fmt.Sprintf("{%s}", strings.Join(entries, ", "))
	default:
		return "None"
	}
}
