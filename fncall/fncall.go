package fncall

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
)

type IPCPayload struct {
	Op        string          `json:"op"`
	Operation string          `json:"operation"`
	Tools     json.RawMessage `json:"tools"`
	Call      string          `json:"call"`
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
	Format    string          `json:"format"`
	JSON      bool            `json:"json"`
}

// Render processes a raw tools JSON and returns the rendered Python catalog stubs.
func Render(toolsJSON []byte) (string, *NameMapper, []ParsedTool, error) {
	mapper := NewNameMapper()
	tools, err := ParseToolsList(toolsJSON, mapper)
	if err != nil {
		return "", nil, nil, err
	}
	rendered := RenderCatalog(tools, mapper)
	return rendered, mapper, tools, nil
}

// Parse parses a Pythonic call string into tool name and arguments map.
// If tools and mapper are provided, sanitized identifiers are mapped back to their original names.
func Parse(callStr string, tools []ParsedTool, mapper *NameMapper) (string, map[string]interface{}, error) {
	call, err := ParseCall(callStr)
	if err != nil {
		return "", nil, err
	}

	sanitizedName := call.Name
	origName := sanitizedName
	if mapper != nil {
		origName = mapper.OriginalToolName(sanitizedName)
	}

	args := call.ToJSONArguments()
	if mapper != nil {
		remappedArgs := make(map[string]interface{})
		for k, v := range args {
			origK := mapper.OriginalParamName(sanitizedName, k)
			remappedArgs[origK] = v
		}
		args = remappedArgs
	}

	return origName, args, nil
}

// Serialize formats tool name and arguments into the canonical Pythonic call string.
func Serialize(name string, args map[string]interface{}, tools []ParsedTool) (string, error) {
	var matchedTool *ParsedTool
	for i := range tools {
		if tools[i].OriginalName == name || tools[i].SanitizedName == name {
			matchedTool = &tools[i]
			break
		}
	}
	return SerializeCall(name, args, matchedTool)
}

// SerializeWithEscape formats tool name and arguments into the canonical Pythonic call string
// without identifier validation, applying escapeFn (if non-nil) to undeclared tool names,
// argument keys, and multi-line triple-quoted string bodies.
func SerializeWithEscape(name string, args map[string]interface{}, tools []ParsedTool, escapeFn func(string) string) (string, error) {
	var matchedTool *ParsedTool
	for i := range tools {
		if tools[i].OriginalName == name || tools[i].SanitizedName == name {
			matchedTool = &tools[i]
			break
		}
	}
	return SerializeCallWithEscape(name, args, matchedTool, escapeFn)
}

// SerializeStrict formats tool name and arguments into the canonical Pythonic call string
// with strict identifier validation on unknown tool names and extra/unknown argument keys,
// and applies escapeFn (if non-nil) to multi-line triple-quoted string bodies.
func SerializeStrict(name string, args map[string]interface{}, tools []ParsedTool, escapeFn func(string) string) (string, error) {
	var matchedTool *ParsedTool
	for i := range tools {
		if tools[i].OriginalName == name || tools[i].SanitizedName == name {
			matchedTool = &tools[i]
			break
		}
	}
	return SerializeCallStrict(name, args, matchedTool, escapeFn)
}

// Validate validates arguments against the tool's JSON Schema.
func Validate(name string, args map[string]interface{}, tools []ParsedTool) error {
	var matchedTool *ParsedTool
	for i := range tools {
		if tools[i].OriginalName == name || tools[i].SanitizedName == name {
			matchedTool = &tools[i]
			break
		}
	}
	if matchedTool == nil {
		return fmt.Errorf("ErrArgs: unknown tool: %s", name)
	}

	return ValidateArguments(matchedTool, args)
}

// EmitGrammar emits GBNF or xgrammar-compatible EBNF for the tool list.
func EmitGrammar(tools []ParsedTool, format string) string {
	g := BuildGrammar(tools)
	if strings.ToLower(format) == "gbnf" {
		return g.EmitGBNF()
	}
	return g.EmitEBNF()
}

// HandleIPC handles the "fncall" tool requests in reckoner-harness.
func HandleIPC(payloadStr string) (string, error) {
	payloadStr = strings.TrimSpace(payloadStr)
	var req IPCPayload

	if strings.HasPrefix(payloadStr, "{") {
		if err := json.Unmarshal([]byte(payloadStr), &req); err != nil {
			return "", fmt.Errorf("ErrSyntax: invalid JSON payload: %w", err)
		}
	} else {
		// CLI-style: "op <rest>"
		parts := strings.SplitN(payloadStr, " ", 2)
		req.Op = parts[0]
		if len(parts) > 1 {
			rest := strings.TrimSpace(parts[1])
			if req.Op == "parse" {
				req.Call = rest
			} else if req.Op == "render" || req.Op == "grammar" {
				req.Tools = json.RawMessage(rest)
			}
		}
	}

	op := req.Op
	if op == "" {
		op = req.Operation
	}

	switch strings.ToLower(op) {
	case "render":
		if len(req.Tools) == 0 {
			return "", fmt.Errorf("ErrArgs: missing 'tools' for render")
		}
		rendered, mapper, _, err := Render(req.Tools)
		if err != nil {
			return "", err
		}
		if req.JSON {
			outMap := map[string]interface{}{
				"rendered": rendered,
				"mapper":   mapper,
			}
			b, _ := json.Marshal(outMap)
			return string(b), nil
		}
		return rendered, nil

	case "parse":
		if req.Call == "" {
			return "", fmt.Errorf("ErrArgs: missing 'call' for parse")
		}
		var tools []ParsedTool
		var mapper *NameMapper
		if len(req.Tools) > 0 {
			var err error
			mapper = NewNameMapper()
			tools, err = ParseToolsList(req.Tools, mapper)
			if err != nil {
				return "", err
			}
		}
		name, args, err := Parse(req.Call, tools, mapper)
		if err != nil {
			return "", err
		}
		resMap := map[string]interface{}{
			"name":      name,
			"arguments": args,
		}
		b, err := json.Marshal(resMap)
		if err != nil {
			return "", err
		}
		return string(b), nil

	case "serialize":
		if req.Name == "" {
			return "", fmt.Errorf("ErrArgs: missing 'name' for serialize")
		}
		var args map[string]interface{}
		if len(req.Arguments) > 0 {
			dec := json.NewDecoder(bytes.NewReader(req.Arguments))
			dec.UseNumber()
			if err := dec.Decode(&args); err != nil {
				return "", fmt.Errorf("ErrArgs: invalid 'arguments': %w", err)
			}
		}
		var tools []ParsedTool
		if len(req.Tools) > 0 {
			var err error
			tools, err = ParseToolsList(req.Tools, NewNameMapper())
			if err != nil {
				return "", err
			}
		}
		return Serialize(req.Name, args, tools)

	case "validate":
		var tools []ParsedTool
		if len(req.Tools) == 0 {
			return "", fmt.Errorf("ErrArgs: missing 'tools' for validate")
		}
		var err error
		mapper := NewNameMapper()
		tools, err = ParseToolsList(req.Tools, mapper)
		if err != nil {
			return "", err
		}

		name := req.Name
		var args map[string]interface{}

		if req.Call != "" {
			name, args, err = Parse(req.Call, tools, mapper)
			if err != nil {
				res := map[string]interface{}{"valid": false, "error": err.Error()}
				b, _ := json.Marshal(res)
				return string(b), nil
			}
		} else {
			if len(req.Arguments) > 0 {
				dec := json.NewDecoder(bytes.NewReader(req.Arguments))
				dec.UseNumber()
				if err := dec.Decode(&args); err != nil {
					res := map[string]interface{}{"valid": false, "error": fmt.Sprintf("ErrArgs: %v", err)}
					b, _ := json.Marshal(res)
					return string(b), nil
				}
			}
		}

		if err := Validate(name, args, tools); err != nil {
			res := map[string]interface{}{"valid": false, "error": err.Error()}
			b, _ := json.Marshal(res)
			return string(b), nil
		}
		res := map[string]interface{}{"valid": true}
		b, _ := json.Marshal(res)
		return string(b), nil

	case "grammar":
		if len(req.Tools) == 0 {
			return "", fmt.Errorf("ErrArgs: missing 'tools' for grammar")
		}
		tools, err := ParseToolsList(req.Tools, NewNameMapper())
		if err != nil {
			return "", err
		}
		fmtChoice := req.Format
		if fmtChoice == "" {
			fmtChoice = "ebnf"
		}
		return EmitGrammar(tools, fmtChoice), nil

	default:
		return "", fmt.Errorf("ErrArgs: unknown fncall operation: %s", op)
	}
}
