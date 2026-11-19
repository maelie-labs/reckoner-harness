package fncall

import (
	"encoding/json"
	"fmt"
	"math"
	"math/big"
	"sort"
	"strconv"
	"strings"
)

type ValueType int

const (
	ValueTypeString ValueType = iota
	ValueTypeNumber
	ValueTypeBool
	ValueTypeNone
	ValueTypeList
	ValueTypeDict
)

type Value struct {
	Type        ValueType
	StrVal      string
	IsMultiLine bool
	NumStr      string
	BoolVal     bool
	ListVal     []Value
	DictVal     []DictEntry
}

type DictEntry struct {
	Key string
	Val Value
}

type Kwarg struct {
	Key string
	Val Value
}

type Call struct {
	Name   string
	Kwargs []Kwarg
}

// ToJSONArguments converts Call kwargs into a JSON-compatible map[string]interface{}.
func (c *Call) ToJSONArguments() map[string]interface{} {
	result := make(map[string]interface{})
	for _, kw := range c.Kwargs {
		result[kw.Key] = kw.Val.ToInterface()
	}
	return result
}

// ToInterface converts Value to a standard Go interface{} representing the JSON value.
func (v *Value) ToInterface() interface{} {
	switch v.Type {
	case ValueTypeString:
		return v.StrVal
	case ValueTypeNumber:
		// Check if it's an integer
		if !strings.ContainsAny(v.NumStr, ".eE") {
			if bi, ok := new(big.Int).SetString(v.NumStr, 10); ok {
				if bi.IsInt64() {
					return bi.Int64()
				}
				// Return as json.Number or string to preserve big integer precision
				return json.Number(v.NumStr)
			}
		}
		// Return as json.Number to preserve exact decimals
		return json.Number(v.NumStr)
	case ValueTypeBool:
		return v.BoolVal
	case ValueTypeNone:
		return nil
	case ValueTypeList:
		list := make([]interface{}, len(v.ListVal))
		for i, item := range v.ListVal {
			list[i] = item.ToInterface()
		}
		return list
	case ValueTypeDict:
		dict := make(map[string]interface{})
		for _, entry := range v.DictVal {
			dict[entry.Key] = entry.Val.ToInterface()
		}
		return dict
	default:
		return nil
	}
}

// ValueFromInterface converts an interface{} into a Value AST.
func ValueFromInterface(val interface{}) (Value, error) {
	switch v := val.(type) {
	case string:
		return Value{Type: ValueTypeString, StrVal: v}, nil
	case bool:
		return Value{Type: ValueTypeBool, BoolVal: v}, nil
	case nil:
		return Value{Type: ValueTypeNone}, nil
	case int:
		return Value{Type: ValueTypeNumber, NumStr: strconv.FormatInt(int64(v), 10)}, nil
	case int64:
		return Value{Type: ValueTypeNumber, NumStr: strconv.FormatInt(v, 10)}, nil
	case int32:
		return Value{Type: ValueTypeNumber, NumStr: strconv.FormatInt(int64(v), 10)}, nil
	case float64:
		return Value{Type: ValueTypeNumber, NumStr: formatFloat64(v)}, nil
	case json.Number:
		return Value{Type: ValueTypeNumber, NumStr: formatNumberFromRaw(v.String())}, nil
	case *big.Int:
		return Value{Type: ValueTypeNumber, NumStr: v.String()}, nil
	case []interface{}:
		var list []Value
		for _, item := range v {
			converted, err := ValueFromInterface(item)
			if err != nil {
				return Value{}, err
			}
			list = append(list, converted)
		}
		return Value{Type: ValueTypeList, ListVal: list}, nil
	case map[string]interface{}:
		keys := make([]string, 0, len(v))
		for k := range v {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		var entries []DictEntry
		for _, k := range keys {
			converted, err := ValueFromInterface(v[k])
			if err != nil {
				return Value{}, err
			}
			entries = append(entries, DictEntry{Key: k, Val: converted})
		}
		return Value{Type: ValueTypeDict, DictVal: entries}, nil
	default:
		return Value{}, fmt.Errorf("unsupported value type: %T", val)
	}
}

func formatFloat64(f float64) string {
	if math.IsNaN(f) {
		return "nan"
	}
	if math.IsInf(f, 1) {
		return "inf"
	}
	if math.IsInf(f, -1) {
		return "-inf"
	}
	if math.Floor(f) == f {
		if f == 0.0 || f == -0.0 {
			return "0"
		}
		if f >= math.MinInt64 && f <= math.MaxInt64 {
			return strconv.FormatInt(int64(f), 10)
		}
		bf := new(big.Float).SetFloat64(f)
		bi, _ := bf.Int(nil)
		return bi.String()
	}
	abs := math.Abs(f)
	if abs >= 1e-4 && abs < 1e16 {
		return strconv.FormatFloat(f, 'f', -1, 64)
	}
	return strconv.FormatFloat(f, 'g', -1, 64)
}

func formatNumberFromRaw(s string) string {
	if !strings.ContainsAny(s, ".eE") {
		if s == "-0" {
			return "0"
		}
		return s
	}
	f, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return s
	}
	return formatFloat64(f)
}

