package main

/*
#cgo CFLAGS: -I${SRCDIR}/third_party/lua-5.4.9/src -DLUA_COMPAT_5_3
#cgo LDFLAGS: -lm
#include <stdlib.h>
#include "lua_bridge.h"
*/
import "C"
import (
	"fmt"
	"strconv"
	"strings"
	"unsafe"
)

// stripChunkPrefix removes chunk name prefixes like '[string "..."]:' while preserving 'line N:'.
func stripChunkPrefix(errStr string) string {
	errStr = strings.TrimSpace(errStr)
	if strings.HasPrefix(errStr, "[string ") {
		idx := strings.Index(errStr, "]:")
		if idx != -1 {
			remainder := strings.TrimSpace(errStr[idx+2:])
			colonIdx := strings.IndexByte(remainder, ':')
			if colonIdx != -1 {
				lineNumStr := strings.TrimSpace(remainder[:colonIdx])
				if _, err := strconv.Atoi(lineNumStr); err == nil {
					msg := strings.TrimSpace(remainder[colonIdx+1:])
					return fmt.Sprintf("line %s: %s", lineNumStr, msg)
				}
			}
			return remainder
		}
	}
	return errStr
}

// formatLuaError enforces the error contract: single-line Err<Code>: <message>.
func formatLuaError(rawErr string) string {
	if strings.Contains(rawErr, "instruction budget exceeded") {
		return "ErrTimeout: instruction budget exceeded"
	}
	if strings.Contains(rawErr, "execution deadline exceeded") {
		return "ErrTimeout: execution deadline exceeded"
	}
	if strings.Contains(rawErr, "not enough memory") || strings.Contains(rawErr, "memory limit") {
		return "ErrOverflow: memory limit"
	}

	cleaned := stripChunkPrefix(rawErr)
	// Flatten any newlines or multiple whitespace to a single line per error contract
	cleaned = strings.Join(strings.Fields(cleaned), " ")
	cleaned = escapeControlTokens(cleaned)

	if strings.HasPrefix(cleaned, "Err") {
		return cleaned
	}
	return fmt.Sprintf("ErrLua: %s", cleaned)
}

// ExecuteLua runs a sandboxed Lua 5.4 script with deterministic instruction budget and strict resource caps.
func ExecuteLua(script string, rs *RegisterStore) (string, error) {
	cScript := C.CString(script)
	defer C.free(unsafe.Pointer(cScript))

	var regNames []*C.char
	var regVals []*C.char
	var regTypes []C.int

	// Bind typed registers r0..r15 into Lua environment 
	if rs != nil {
		for i := 0; i < NumRegisters; i++ {
			val, err := rs.Get(i)
			if err != nil || val == nil {
				continue
			}
			regName := fmt.Sprintf("r%d", i)
			var valStr string
			var regType C.int
			switch val.Kind {
			case ValKindRat:
				if val.Rat != nil && val.Rat.IsInt() {
					valStr = val.Rat.Num().String()
					regType = C.REG_TYPE_INT
				} else if val.Rat != nil {
					f, _ := val.Rat.Float64()
					valStr, _ = FormatFloatStandard(f)
					regType = C.REG_TYPE_FLOAT
				}
			case ValKindFloat:
				valStr, _ = FormatFloatStandard(val.Float)
				regType = C.REG_TYPE_FLOAT
			case ValKindBool:
				valStr = fmt.Sprintf("%t", val.Bool)
				regType = C.REG_TYPE_BOOL
			case ValKindString:
				valStr = val.Str
				regType = C.REG_TYPE_STRING
			case ValKindRawString:
				valStr = val.Str
				regType = C.REG_TYPE_STRING
			case ValKindDate:
				valStr = val.DateStr
				regType = C.REG_TYPE_STRING
			case ValKindUnit:
				if val.UnitVal != nil {
					f, _ := val.UnitVal.Float64()
					valStr, _ = FormatFloatStandard(f)
					regType = C.REG_TYPE_FLOAT
				}
			case ValKindList:
				var items []string
				for _, item := range val.List {
					items = append(items, item.String())
				}
				valStr = fmt.Sprintf("[%s]", strings.Join(items, ", "))
				regType = C.REG_TYPE_STRING
			default:
				valStr = val.String()
				regType = C.REG_TYPE_STRING
			}

			cName := C.CString(regName)
			cVal := C.CString(valStr)
			defer C.free(unsafe.Pointer(cName))
			defer C.free(unsafe.Pointer(cVal))
			regNames = append(regNames, cName)
			regVals = append(regVals, cVal)
			regTypes = append(regTypes, regType)
		}
	}

	var regNamesPtr **C.char
	var regValsPtr **C.char
	var regTypesPtr *C.int
	if len(regNames) > 0 {
		regNamesPtr = &regNames[0]
		regValsPtr = &regVals[0]
		regTypesPtr = &regTypes[0]
	}

	var cResult *C.char
	var cErr *C.char

	status := C.run_lua_sandboxed(
		cScript,
		C.int(len(regNames)),
		regNamesPtr,
		regValsPtr,
		regTypesPtr,
		&cResult,
		&cErr,
	)

	if status != 0 {
		var errStr string
		if cErr != nil {
			errStr = C.GoString(cErr)
			C.free_lua_string(cErr)
		} else {
			errStr = "unknown Lua execution error"
		}
		return "", fmt.Errorf("%s", formatLuaError(errStr))
	}

	var resultStr string
	if cResult != nil {
		resultStr = C.GoString(cResult)
		C.free_lua_string(cResult)
	}
	return escapeControlTokens(resultStr), nil
}
