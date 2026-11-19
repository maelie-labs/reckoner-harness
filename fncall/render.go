package fncall

import (
	"bytes"
	"encoding/json"
	"fmt"
	"sort"
	"strconv"
	"strings"
)

// NameMapper manages bijective mappings between original names and sanitized Python identifiers.
type NameMapper struct {
	ToolOriginalToSanitized  map[string]string            `json:"tool_original_to_sanitized"`
	ToolSanitizedToOriginal  map[string]string            `json:"tool_sanitized_to_original"`
	ParamOriginalToSanitized map[string]map[string]string `json:"param_original_to_sanitized"`
	ParamSanitizedToOriginal map[string]map[string]string `json:"param_sanitized_to_original"`

	// Names that are already valid identifiers keep their spelling: they are reserved before any
	// invalid name is sanitized, so e.g. "github.create-issue" can never take "github_create_issue"
	// from a tool that is really called that (it gets "github_create_issue_2" instead).
	reservedTools  map[string]bool
	reservedParams map[string]map[string]bool
}

// ReserveTool reserves a tool name that is already a valid identifier.
func (m *NameMapper) ReserveTool(orig string) {
	if !isValidIdentifier(orig) {
		return
	}
	if m.reservedTools == nil {
		m.reservedTools = make(map[string]bool)
	}
	m.reservedTools[orig] = true
}

// ReserveParam reserves a parameter name (per tool) that is already a valid identifier.
func (m *NameMapper) ReserveParam(toolOrig, paramOrig string) {
	if !isValidIdentifier(paramOrig) {
		return
	}
	if m.reservedParams == nil {
		m.reservedParams = make(map[string]map[string]bool)
	}
	if m.reservedParams[toolOrig] == nil {
		m.reservedParams[toolOrig] = make(map[string]bool)
	}
	m.reservedParams[toolOrig][paramOrig] = true
}

func NewNameMapper() *NameMapper {
	return &NameMapper{
		ToolOriginalToSanitized:  make(map[string]string),
		ToolSanitizedToOriginal:  make(map[string]string),
		ParamOriginalToSanitized: make(map[string]map[string]string),
		ParamSanitizedToOriginal: make(map[string]map[string]string),
	}
}

func isValidIdentifier(s string) bool {
	if s == "" {
		return false
	}
	for i, r := range s {
		if i == 0 {
			if !((r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || r == '_') {
				return false
			}
		} else {
			if !((r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '_') {
				return false
			}
		}
	}
	return true
}

func sanitizeString(s string, used map[string]bool) string {
	var candidate string
	if isValidIdentifier(s) {
		candidate = s
	} else {
		var sb strings.Builder
		for i, r := range s {
			if i == 0 && (r >= '0' && r <= '9') {
				sb.WriteByte('_')
			}
			if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '_' {
				sb.WriteRune(r)
			} else {
				sb.WriteByte('_')
			}
		}
		candidate = sb.String()
		if candidate == "" || candidate == "_" {
			candidate = "_val"
		}
	}

	if !used[candidate] {
		used[candidate] = true
		return candidate
	}
	idx := 2
	for {
		cand := fmt.Sprintf("%s_%d", candidate, idx)
		if !used[cand] {
			used[cand] = true
			return cand
		}
		idx++
	}
}

func (m *NameMapper) SanitizeToolName(orig string) string {
	if s, ok := m.ToolOriginalToSanitized[orig]; ok {
		return s
	}
	used := make(map[string]bool)
	for s := range m.ToolSanitizedToOriginal {
		used[s] = true
	}
	if !isValidIdentifier(orig) {
		for r := range m.reservedTools {
			used[r] = true
		}
	}
	s := sanitizeString(orig, used)
	m.ToolOriginalToSanitized[orig] = s
	m.ToolSanitizedToOriginal[s] = orig
	return s
}

func (m *NameMapper) SanitizeParamName(toolOrig, paramOrig string) string {
	if m.ParamOriginalToSanitized[toolOrig] == nil {
		m.ParamOriginalToSanitized[toolOrig] = make(map[string]string)
		m.ParamSanitizedToOriginal[toolOrig] = make(map[string]string)
	}
	if s, ok := m.ParamOriginalToSanitized[toolOrig][paramOrig]; ok {
		return s
	}
	used := make(map[string]bool)
	for s := range m.ParamSanitizedToOriginal[toolOrig] {
		used[s] = true
	}
	if !isValidIdentifier(paramOrig) {
		for r := range m.reservedParams[toolOrig] {
			used[r] = true
		}
	}
	s := sanitizeString(paramOrig, used)
	m.ParamOriginalToSanitized[toolOrig][paramOrig] = s
	m.ParamSanitizedToOriginal[toolOrig][s] = paramOrig
	return s
}

func (m *NameMapper) OriginalToolName(sanitized string) string {
	if orig, ok := m.ToolSanitizedToOriginal[sanitized]; ok {
		return orig
	}
	return sanitized
}

func (m *NameMapper) OriginalParamName(toolSanitizedOrOrig, paramSanitized string) string {
	toolOrig := m.OriginalToolName(toolSanitizedOrOrig)
	if m.ParamSanitizedToOriginal[toolOrig] != nil {
		if orig, ok := m.ParamSanitizedToOriginal[toolOrig][paramSanitized]; ok {
			return orig
		}
	}
	return paramSanitized
}

// CapDescription caps description at limit characters, cutting at a sentence boundary when possible.
func CapDescription(s string, limit int) string {
	s = strings.TrimSpace(s)
	if len(s) <= limit {
		return s
	}
	sub := s[:limit]
	lastPunct := -1
	for i := len(sub) - 1; i >= 0; i-- {
		c := sub[i]
		if c == '.' || c == '!' || c == '?' {
			if i == len(sub)-1 || s[i+1] == ' ' || s[i+1] == '\n' {
				lastPunct = i
				break
			}
		}
	}
	if lastPunct >= limit/3 {
		return strings.TrimSpace(sub[:lastPunct+1])
	}
	lastSpace := strings.LastIndex(sub, " ")
	if lastSpace > 0 {
		return strings.TrimSpace(sub[:lastSpace])
	}
	return strings.TrimSpace(sub)
}

// Tool Definition models
type RawTool struct {
	Type         string                 `json:"type,omitempty"`
	Function     *RawFunction           `json:"function,omitempty"`
	Name         string                 `json:"name,omitempty"`
	Description  string                 `json:"description,omitempty"`
	Parameters   map[string]interface{} `json:"parameters,omitempty"`
	InputSchema  map[string]interface{} `json:"inputSchema,omitempty"`
	OutputSchema map[string]interface{} `json:"outputSchema,omitempty"`
}

type RawFunction struct {
	Name         string                 `json:"name"`
	Description  string                 `json:"description,omitempty"`
	Parameters   map[string]interface{} `json:"parameters,omitempty"`
	OutputSchema map[string]interface{} `json:"outputSchema,omitempty"`
}

// PropertyInfo holds metadata and ordered properties.
type PropertyInfo struct {
	OriginalName  string
	SanitizedName string
	Schema        map[string]interface{}
	Required      bool
}

type ParsedTool struct {
	OriginalName     string
	SanitizedName    string
	Description      string
	Properties       []PropertyInfo
	ParametersSchema map[string]interface{}
	OutputSchema     map[string]interface{}
}

// TypeHoister tracks object types that need hoisting (depth > 2).
type TypeHoister struct {
	Definitions  []string
	DefinedTypes map[string]string
	Counter      int
}

func NewTypeHoister() *TypeHoister {
	return &TypeHoister{
		DefinedTypes: make(map[string]string),
	}
}

func (h *TypeHoister) GetOrHoist(schema map[string]interface{}, defaultName string) string {
	name := defaultName
	if t, ok := schema["title"].(string); ok && isValidIdentifier(t) {
		name = t
	}
	if name == "" {
		h.Counter++
		name = fmt.Sprintf("Type_%d", h.Counter)
	}
	if _, ok := h.DefinedTypes[name]; ok {
		return name
	}
	// Render the object type at depth 1
	body := renderObjectProperties(schema, 1, h)
	rendered := fmt.Sprintf("%s = {%s}", name, body)
	h.DefinedTypes[name] = rendered
	h.Definitions = append(h.Definitions, rendered)
	return name
}

func formatLiteralValue(v interface{}) string {
	switch val := v.(type) {
	case string:
		b, _ := json.Marshal(val)
		return string(b)
	case bool:
		if val {
			return "True"
		}
		return "False"
	case nil:
		return "None"
	case float64:
		if float64(int64(val)) == val {
			return strconv.FormatInt(int64(val), 10)
		}
		return strconv.FormatFloat(val, 'f', -1, 64)
	case json.Number:
		return val.String()
	default:
		return fmt.Sprintf("%v", val)
	}
}

func renderObjectProperties(schema map[string]interface{}, depth int, hoister *TypeHoister) string {
	rawProps, _ := schema["properties"].(map[string]interface{})
	if len(rawProps) == 0 {
		return ""
	}
	reqSet := make(map[string]bool)
	if reqList, ok := schema["required"].([]interface{}); ok {
		for _, r := range reqList {
			if s, ok := r.(string); ok {
				reqSet[s] = true
			}
		}
	}

	keys := make([]string, 0, len(rawProps))
	for k := range rawProps {
		keys = append(keys, k)
	}
	// Sort required first, then optional, maintaining alphabetical or schema stability
	sort.SliceStable(keys, func(i, j int) bool {
		reqI := reqSet[keys[i]]
		reqJ := reqSet[keys[j]]
		if reqI && !reqJ {
			return true
		}
		if !reqI && reqJ {
			return false
		}
		return keys[i] < keys[j]
	})

	var parts []string
	for _, k := range keys {
		propSchema, _ := rawProps[k].(map[string]interface{})
		tStr := renderType(propSchema, depth+1, hoister, toPascal(k))
		if reqSet[k] {
			parts = append(parts, fmt.Sprintf("%s: %s", k, tStr))
		} else {
			parts = append(parts, fmt.Sprintf("%s?: %s", k, tStr))
		}
	}
	return strings.Join(parts, ", ")
}

func toPascal(s string) string {
	parts := strings.FieldsFunc(s, func(r rune) bool {
		return r == '_' || r == '-' || r == '.'
	})
	var sb strings.Builder
	for _, p := range parts {
		if len(p) > 0 {
			sb.WriteString(strings.ToUpper(p[:1]) + p[1:])
		}
	}
	res := sb.String()
	if res == "" || !isValidIdentifier(res) {
		res = "Item"
	}
	// If it ends in "s", singularize common words if simple
	if strings.HasSuffix(res, "s") && len(res) > 3 && !strings.HasSuffix(res, "ss") {
		res = res[:len(res)-1]
	}
	return res
}

func renderType(schema map[string]interface{}, depth int, hoister *TypeHoister, hintName string) string {
	if schema == nil {
		return "Any"
	}

	// 1. Enum
	if enumRaw, ok := schema["enum"].([]interface{}); ok && len(enumRaw) > 0 {
		var litParts []string
		for _, item := range enumRaw {
			litParts = append(litParts, formatLiteralValue(item))
		}
		return fmt.Sprintf("Literal[%s]", strings.Join(litParts, ", "))
	}

	// 2. anyOf / oneOf
	for _, combKey := range []string{"anyOf", "oneOf"} {
		if combRaw, ok := schema[combKey].([]interface{}); ok && len(combRaw) > 0 {
			var unionParts []string
			for _, sub := range combRaw {
				if subMap, ok := sub.(map[string]interface{}); ok {
					unionParts = append(unionParts, renderType(subMap, depth, hoister, hintName))
				}
			}
			return strings.Join(unionParts, " | ")
		}
	}

	// 3. Type
	tRaw, ok := schema["type"]
	if !ok {
		if _, hasProps := schema["properties"]; hasProps {
			tRaw = "object"
		} else if _, hasItems := schema["items"]; hasItems {
			tRaw = "array"
		} else {
			return "Any"
		}
	}

	// Type might be list e.g. ["string", "null"]
	if tList, ok := tRaw.([]interface{}); ok {
		var parts []string
		for _, item := range tList {
			if s, ok := item.(string); ok {
				subSchema := map[string]interface{}{"type": s}
				parts = append(parts, renderType(subSchema, depth, hoister, hintName))
			}
		}
		return strings.Join(parts, " | ")
	}

	typeName, _ := tRaw.(string)
	switch typeName {
	case "string":
		return "str"
	case "integer":
		return "int"
	case "number":
		return "float"
	case "boolean":
		return "bool"
	case "null":
		return "None"
	case "array":
		items, hasItems := schema["items"].(map[string]interface{})
		if hasItems {
			subT := renderType(items, depth+1, hoister, hintName)
			return fmt.Sprintf("list[%s]", subT)
		}
		return "list"
	case "object":
		rawProps, hasProps := schema["properties"].(map[string]interface{})
		if !hasProps || len(rawProps) == 0 {
			return "dict"
		}
		if depth <= 2 {
			return fmt.Sprintf("{%s}", renderObjectProperties(schema, depth+1, hoister))
		}
		// Deeper than depth 2: hoist!
		return hoister.GetOrHoist(schema, hintName)
	default:
		return "Any"
	}
}

func renderDefaultValue(v interface{}) string {
	return formatLiteralValue(v)
}

func renderConstraintsNote(schema map[string]interface{}) string {
	var notes []string

	// Range minimum / maximum
	hasMin := false
	hasMax := false
	var minStr, maxStr string
	if minVal, ok := schema["minimum"]; ok {
		hasMin = true
		minStr = formatNumConstraint(minVal)
	}
	if maxVal, ok := schema["maximum"]; ok {
		hasMax = true
		maxStr = formatNumConstraint(maxVal)
	}
	if hasMin && hasMax {
		notes = append(notes, fmt.Sprintf("%s..%s", minStr, maxStr))
	} else if hasMin {
		notes = append(notes, fmt.Sprintf(">= %s", minStr))
	} else if hasMax {
		notes = append(notes, fmt.Sprintf("<= %s", maxStr))
	}

	// format
	if fmtVal, ok := schema["format"].(string); ok && fmtVal != "" {
		notes = append(notes, fmt.Sprintf("(format: %s)", fmtVal))
	}
	// pattern
	if patVal, ok := schema["pattern"].(string); ok && patVal != "" {
		notes = append(notes, fmt.Sprintf("(pattern: %s)", patVal))
	}
	// length
	hasMinL := false
	hasMaxL := false
	var minL, maxL string
	if v, ok := schema["minLength"]; ok {
		hasMinL = true
		minL = formatNumConstraint(v)
	}
	if v, ok := schema["maxLength"]; ok {
		hasMaxL = true
		maxL = formatNumConstraint(v)
	}
	if hasMinL && hasMaxL {
		notes = append(notes, fmt.Sprintf("(length: %s..%s)", minL, maxL))
	} else if hasMinL {
		notes = append(notes, fmt.Sprintf("(min length: %s)", minL))
	} else if hasMaxL {
		notes = append(notes, fmt.Sprintf("(max length: %s)", maxL))
	}

	// items
	hasMinI := false
	hasMaxI := false
	var minI, maxI string
	if v, ok := schema["minItems"]; ok {
		hasMinI = true
		minI = formatNumConstraint(v)
	}
	if v, ok := schema["maxItems"]; ok {
		hasMaxI = true
		maxI = formatNumConstraint(v)
	}
	if hasMinI && hasMaxI {
		notes = append(notes, fmt.Sprintf("(items: %s..%s)", minI, maxI))
	} else if hasMinI {
		notes = append(notes, fmt.Sprintf("(min items: %s)", minI))
	} else if hasMaxI {
		notes = append(notes, fmt.Sprintf("(max items: %s)", maxI))
	}

	return strings.Join(notes, " ")
}

func formatNumConstraint(v interface{}) string {
	switch val := v.(type) {
	case float64:
		if float64(int64(val)) == val {
			return strconv.FormatInt(int64(val), 10)
		}
		return strconv.FormatFloat(val, 'f', -1, 64)
	case json.Number:
		return val.String()
	default:
		return fmt.Sprintf("%v", val)
	}
}

// ParseToolsList decodes a list of tool definitions from raw JSON, preserving properties order.
func ParseToolsList(data []byte, mapper *NameMapper) ([]ParsedTool, error) {
	data = bytes.TrimSpace(data)
	if len(data) == 0 {
		return nil, nil
	}

	var rawList []json.RawMessage
	if data[0] == '[' {
		if err := json.Unmarshal(data, &rawList); err != nil {
			return nil, fmt.Errorf("invalid json list: %w", err)
		}
	} else if data[0] == '{' {
		var wrapper struct {
			Tools []json.RawMessage `json:"tools"`
		}
		if err := json.Unmarshal(data, &wrapper); err == nil && len(wrapper.Tools) > 0 {
			rawList = wrapper.Tools
		} else {
			rawList = []json.RawMessage{json.RawMessage(data)}
		}
	} else {
		return nil, fmt.Errorf("expected json list or object")
	}

	// Pre-pass: reserve every name that is already a valid identifier (see NameMapper).
	for _, rawItem := range rawList {
		var raw RawTool
		if err := json.Unmarshal(rawItem, &raw); err != nil {
			return nil, err
		}
		name, params := raw.Name, raw.Parameters
		if raw.Function != nil {
			if raw.Function.Name != "" {
				name = raw.Function.Name
			}
			if raw.Function.Parameters != nil {
				params = raw.Function.Parameters
			}
		} else if raw.InputSchema != nil && params == nil {
			params = raw.InputSchema
		}
		mapper.ReserveTool(name)
		if props, ok := params["properties"].(map[string]interface{}); ok {
			for k := range props {
				mapper.ReserveParam(name, k)
			}
		}
	}

	var parsed []ParsedTool
	for _, rawItem := range rawList {
		tool, err := parseSingleTool(rawItem, mapper)
		if err != nil {
			return nil, err
		}
		parsed = append(parsed, tool)
	}
	return parsed, nil
}

func parseSingleTool(data []byte, mapper *NameMapper) (ParsedTool, error) {
	var raw RawTool
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.UseNumber()
	if err := dec.Decode(&raw); err != nil {
		return ParsedTool{}, err
	}

	name := raw.Name
	desc := raw.Description
	params := raw.Parameters
	outSchema := raw.OutputSchema

	if raw.Function != nil {
		if raw.Function.Name != "" {
			name = raw.Function.Name
		}
		if raw.Function.Description != "" {
			desc = raw.Function.Description
		}
		if raw.Function.Parameters != nil {
			params = raw.Function.Parameters
		}
		if raw.Function.OutputSchema != nil {
			outSchema = raw.Function.OutputSchema
		}
	} else if raw.InputSchema != nil && params == nil {
		params = raw.InputSchema
	}

	sanitizedToolName := mapper.SanitizeToolName(name)

	// Extract properties in JSON appearance order
	orderedProps := extractOrderedProperties(data, params, mapper, name)

	return ParsedTool{
		OriginalName:     name,
		SanitizedName:    sanitizedToolName,
		Description:      desc,
		Properties:       orderedProps,
		ParametersSchema: params,
		OutputSchema:     outSchema,
	}, nil
}

func extractOrderedProperties(fullData []byte, params map[string]interface{}, mapper *NameMapper, toolName string) []PropertyInfo {
	if params == nil {
		return nil
	}
	rawProps, _ := params["properties"].(map[string]interface{})
	if len(rawProps) == 0 {
		return nil
	}

	reqSet := make(map[string]bool)
	if reqList, ok := params["required"].([]interface{}); ok {
		for _, r := range reqList {
			if s, ok := r.(string); ok {
				reqSet[s] = true
			}
		}
	}

	// Use json.Decoder to scan for key order inside "properties": { ... }
	propOrder := scanPropertyKeyOrder(fullData)
	// If scan failed or missed some keys, fall back to remaining keys
	seen := make(map[string]bool)
	var orderedKeys []string
	for _, k := range propOrder {
		if _, exists := rawProps[k]; exists && !seen[k] {
			orderedKeys = append(orderedKeys, k)
			seen[k] = true
		}
	}
	for k := range rawProps {
		if !seen[k] {
			orderedKeys = append(orderedKeys, k)
			seen[k] = true
		}
	}

	// Partition into required first (in schema order), then optional (in schema order)
	var reqKeys, optKeys []string
	for _, k := range orderedKeys {
		if reqSet[k] {
			reqKeys = append(reqKeys, k)
		} else {
			optKeys = append(optKeys, k)
		}
	}
	allKeys := append(reqKeys, optKeys...)

	var result []PropertyInfo
	for _, k := range allKeys {
		pSchema, _ := rawProps[k].(map[string]interface{})
		sanitizedPName := mapper.SanitizeParamName(toolName, k)
		result = append(result, PropertyInfo{
			OriginalName:  k,
			SanitizedName: sanitizedPName,
			Schema:        pSchema,
			Required:      reqSet[k],
		})
	}
	return result
}

func scanPropertyKeyOrder(data []byte) []string {
	dec := json.NewDecoder(bytes.NewReader(data))
	var keys []string
	for {
		t, err := dec.Token()
		if err != nil {
			break
		}
		if str, ok := t.(string); ok && str == "properties" {
			// Next token should be '{'
			delim, err := dec.Token()
			if err != nil {
				break
			}
			if d, ok := delim.(json.Delim); ok && d == '{' {
				depth := 1
				for dec.More() {
					kt, err := dec.Token()
					if err != nil {
						break
					}
					if k, ok := kt.(string); ok && depth == 1 {
						keys = append(keys, k)
					}
					// Skip the value
					var discard interface{}
					if err := dec.Decode(&discard); err != nil {
						break
					}
				}
				break
			}
		}
	}
	return keys
}

// RenderCatalog renders the list of parsed tools into the compact Python-stub catalog (§3.1).
func RenderCatalog(tools []ParsedTool, mapper *NameMapper) string {
	hoister := NewTypeHoister()
	var funcStubs []string

	for _, tool := range tools {
		var paramStrs []string
		var descLines []string

		for _, p := range tool.Properties {
			tStr := renderType(p.Schema, 1, hoister, toPascal(p.SanitizedName))
			if p.Required {
				paramStrs = append(paramStrs, fmt.Sprintf("%s: %s", p.SanitizedName, tStr))
			} else {
				if defVal, hasDef := p.Schema["default"]; hasDef {
					paramStrs = append(paramStrs, fmt.Sprintf("%s: %s = %s", p.SanitizedName, tStr, renderDefaultValue(defVal)))
				} else {
					paramStrs = append(paramStrs, fmt.Sprintf("%s: %s = None", p.SanitizedName, tStr))
				}
			}

			// Parameter description line
			rawPDesc, _ := p.Schema["description"].(string)
			constraintNote := renderConstraintsNote(p.Schema)
			pDesc := CapDescription(rawPDesc, 160)

			var lineDesc string
			if pDesc != "" && constraintNote != "" {
				lineDesc = CapDescription(pDesc+" "+constraintNote, 160)
			} else if pDesc != "" {
				lineDesc = pDesc
			} else if constraintNote != "" {
				lineDesc = constraintNote
			}

			if lineDesc != "" {
				descLines = append(descLines, fmt.Sprintf("%s: %s", p.SanitizedName, lineDesc))
			}
		}

		// Tool return annotation
		retAnnotation := ""
		if tool.OutputSchema != nil {
			retT := renderType(tool.OutputSchema, 1, hoister, "Output")
			retAnnotation = fmt.Sprintf(" -> %s", retT)
		}

		sig := fmt.Sprintf("def %s(%s)%s:", tool.SanitizedName, strings.Join(paramStrs, ", "), retAnnotation)

		toolDesc := CapDescription(tool.Description, 300)

		var stubLines []string
		stubLines = append(stubLines, sig)

		if toolDesc != "" || len(descLines) > 0 {
			if toolDesc != "" && len(descLines) == 0 {
				stubLines = append(stubLines, fmt.Sprintf("    \"\"\"%s\"\"\"", toolDesc))
			} else if toolDesc != "" && len(descLines) > 0 {
				stubLines = append(stubLines, fmt.Sprintf("    \"\"\"%s", toolDesc))
				for i, dl := range descLines {
					if i == len(descLines)-1 {
						stubLines = append(stubLines, fmt.Sprintf("    %s\"\"\"", dl))
					} else {
						stubLines = append(stubLines, fmt.Sprintf("    %s", dl))
					}
				}
			} else {
				// No toolDesc, but has descLines
				for i, dl := range descLines {
					if i == 0 && len(descLines) == 1 {
						stubLines = append(stubLines, fmt.Sprintf("    \"\"\"%s\"\"\"", dl))
					} else if i == 0 {
						stubLines = append(stubLines, fmt.Sprintf("    \"\"\"%s", dl))
					} else if i == len(descLines)-1 {
						stubLines = append(stubLines, fmt.Sprintf("    %s\"\"\"", dl))
					} else {
						stubLines = append(stubLines, fmt.Sprintf("    %s", dl))
					}
				}
			}
		} else {
			stubLines = append(stubLines, "    pass")
		}

		funcStubs = append(funcStubs, strings.Join(stubLines, "\n"))
	}

	var allBlocks []string
	if len(hoister.Definitions) > 0 {
		allBlocks = append(allBlocks, strings.Join(hoister.Definitions, "\n"))
	}
	allBlocks = append(allBlocks, strings.Join(funcStubs, "\n"))
	return strings.Join(allBlocks, "\n")
}
