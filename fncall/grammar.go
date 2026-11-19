package fncall

import (
	"fmt"
	"sort"
	"strings"
)

// Grammar AST types
type Expr interface {
	Emit() string
}

type LitExpr struct {
	Val string
}

func (e LitExpr) Emit() string {
	var sb strings.Builder
	sb.WriteByte('"')
	for _, r := range e.Val {
		switch r {
		case '"':
			sb.WriteString(`\"`)
		case '\\':
			sb.WriteString(`\\`)
		case '\n':
			sb.WriteString(`\n`)
		case '\r':
			sb.WriteString(`\r`)
		case '\t':
			sb.WriteString(`\t`)
		default:
			sb.WriteRune(r)
		}
	}
	sb.WriteByte('"')
	return sb.String()
}

type RawExpr struct {
	Raw string
}

func (e RawExpr) Emit() string {
	return e.Raw
}

type RefExpr struct {
	Name string
}

func (e RefExpr) Emit() string {
	return e.Name
}

type SeqExpr struct {
	Items []Expr
}

func (e SeqExpr) Emit() string {
	var parts []string
	for _, item := range e.Items {
		parts = append(parts, item.Emit())
	}
	return strings.Join(parts, " ")
}

type AltExpr struct {
	Alts []Expr
}

func (e AltExpr) Emit() string {
	if len(e.Alts) == 1 {
		return e.Alts[0].Emit()
	}
	var parts []string
	for _, a := range e.Alts {
		parts = append(parts, a.Emit())
	}
	return fmt.Sprintf("(%s)", strings.Join(parts, " | "))
}

type OptExpr struct {
	Inner Expr
}

func (e OptExpr) Emit() string {
	return fmt.Sprintf("(%s)?", e.Inner.Emit())
}

type Rep0Expr struct {
	Inner Expr
}

func (e Rep0Expr) Emit() string {
	return fmt.Sprintf("(%s)*", e.Inner.Emit())
}

type GrammarRule struct {
	Name string
	Expr Expr
}

type Grammar struct {
	Rules []GrammarRule
}

func (g *Grammar) EmitEBNF() string {
	var lines []string
	for _, r := range g.Rules {
		lines = append(lines, fmt.Sprintf("%s ::= %s", r.Name, r.Expr.Emit()))
	}
	return strings.Join(lines, "\n")
}

func (g *Grammar) EmitGBNF() string {
	return g.EmitEBNF()
}

// BuildGrammar generates the specialized GBNF/EBNF grammar for the tool list (§4.4).
func BuildGrammar(tools []ParsedTool) *Grammar {
	g := &Grammar{}

	// Standard base rules
	baseRules := []GrammarRule{
		{Name: "str_single", Expr: RawExpr{Raw: `"\"" ([^"\\\0-\x1f] | "\\" ["\\/bfnrt])* "\""`}},
		{Name: "str_multi", Expr: RawExpr{Raw: `"\"\"\"" ( [^"] | "\"" [^"] | "\"\"" [^"] )* "\"\"\""`}},
		{Name: "str_val", Expr: RawExpr{Raw: `(str_single | str_multi)`}},
		{Name: "int_val", Expr: RawExpr{Raw: `"-"? ("0" | [1-9] [0-9]*)`}},
		{Name: "float_val", Expr: RawExpr{Raw: `"-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?`}},
		{Name: "bool_val", Expr: RawExpr{Raw: `("True" | "False")`}},
		{Name: "none_val", Expr: RawExpr{Raw: `"None"`}},
		{Name: "any_val", Expr: RawExpr{Raw: `(str_val | float_val | bool_val | none_val | any_list | any_dict)`}},
		{Name: "any_list", Expr: RawExpr{Raw: `"[" (any_val (", " any_val)*)? "]"`}},
		{Name: "any_dict", Expr: RawExpr{Raw: `"{" (str_val ": " any_val (", " str_val ": " any_val)*)? "}"`}},
	}

	if len(tools) == 0 {
		g.Rules = append(g.Rules, GrammarRule{Name: "root", Expr: LitExpr{Val: ""}})
		g.Rules = append(g.Rules, baseRules...)
		return g
	}

	var rootAlts []Expr
	for _, tool := range tools {
		callRuleName := fmt.Sprintf("call_%s", tool.SanitizedName)
		rootAlts = append(rootAlts, RefExpr{Name: callRuleName})
	}

	if len(rootAlts) == 1 {
		g.Rules = append(g.Rules, GrammarRule{Name: "root", Expr: rootAlts[0]})
	} else {
		g.Rules = append(g.Rules, GrammarRule{Name: "root", Expr: AltExpr{Alts: rootAlts}})
	}

	// For each tool, build call rule and parameter type rules
	for _, tool := range tools {
		buildToolRules(tool, g)
	}

	g.Rules = append(g.Rules, baseRules...)
	return g
}

func buildToolRules(tool ParsedTool, g *Grammar) {
	callRuleName := fmt.Sprintf("call_%s", tool.SanitizedName)

	var reqProps []PropertyInfo
	var optProps []PropertyInfo
	for _, p := range tool.Properties {
		if p.Required {
			reqProps = append(reqProps, p)
		} else {
			optProps = append(optProps, p)
		}
	}

	// Register type rules for properties
	for _, p := range tool.Properties {
		typeRuleName := fmt.Sprintf("type_%s_%s", tool.SanitizedName, p.SanitizedName)
		typeExpr := buildTypeExpr(p.Schema, typeRuleName, g)
		g.Rules = append(g.Rules, GrammarRule{Name: typeRuleName, Expr: typeExpr})
	}

	openParen := LitExpr{Val: fmt.Sprintf("%s(", tool.SanitizedName)}
	closeParen := LitExpr{Val: ")"}

	if len(reqProps) == 0 && len(optProps) == 0 {
		g.Rules = append(g.Rules, GrammarRule{
			Name: callRuleName,
			Expr: SeqExpr{Items: []Expr{openParen, closeParen}},
		})
		return
	}

	if len(reqProps) > 0 {
		var reqItems []Expr
		for i, p := range reqProps {
			prefix := fmt.Sprintf("%s=", p.SanitizedName)
			if i > 0 {
				prefix = fmt.Sprintf(", %s=", p.SanitizedName)
			}
			pTypeRef := RefExpr{Name: fmt.Sprintf("type_%s_%s", tool.SanitizedName, p.SanitizedName)}
			reqItems = append(reqItems, SeqExpr{Items: []Expr{LitExpr{Val: prefix}, pTypeRef}})
		}

		var optItems []Expr
		for _, p := range optProps {
			prefix := fmt.Sprintf(", %s=", p.SanitizedName)
			pTypeRef := RefExpr{Name: fmt.Sprintf("type_%s_%s", tool.SanitizedName, p.SanitizedName)}
			optItems = append(optItems, OptExpr{Inner: SeqExpr{Items: []Expr{LitExpr{Val: prefix}, pTypeRef}}})
		}

		allCallItems := []Expr{openParen}
		allCallItems = append(allCallItems, reqItems...)
		allCallItems = append(allCallItems, optItems...)
		allCallItems = append(allCallItems, closeParen)

		g.Rules = append(g.Rules, GrammarRule{
			Name: callRuleName,
			Expr: SeqExpr{Items: allCallItems},
		})
		return
	}

	// n == 0, only optional properties
	argsRuleName := fmt.Sprintf("args_%s", tool.SanitizedName)
	var altBranches []Expr
	altBranches = append(altBranches, LitExpr{Val: ""})

	for i := 0; i < len(optProps); i++ {
		optBranchName := fmt.Sprintf("opt_%s_%d", tool.SanitizedName, i)
		altBranches = append(altBranches, RefExpr{Name: optBranchName})

		var branchItems []Expr
		firstProp := optProps[i]
		firstPrefix := fmt.Sprintf("%s=", firstProp.SanitizedName)
		firstTypeRef := RefExpr{Name: fmt.Sprintf("type_%s_%s", tool.SanitizedName, firstProp.SanitizedName)}
		branchItems = append(branchItems, SeqExpr{Items: []Expr{LitExpr{Val: firstPrefix}, firstTypeRef}})

		for j := i + 1; j < len(optProps); j++ {
			subProp := optProps[j]
			subPrefix := fmt.Sprintf(", %s=", subProp.SanitizedName)
			subTypeRef := RefExpr{Name: fmt.Sprintf("type_%s_%s", tool.SanitizedName, subProp.SanitizedName)}
			branchItems = append(branchItems, OptExpr{Inner: SeqExpr{Items: []Expr{LitExpr{Val: subPrefix}, subTypeRef}}})
		}

		g.Rules = append(g.Rules, GrammarRule{
			Name: optBranchName,
			Expr: SeqExpr{Items: branchItems},
		})
	}

	g.Rules = append(g.Rules, GrammarRule{
		Name: argsRuleName,
		Expr: AltExpr{Alts: altBranches},
	})

	g.Rules = append(g.Rules, GrammarRule{
		Name: callRuleName,
		Expr: SeqExpr{Items: []Expr{openParen, RefExpr{Name: argsRuleName}, closeParen}},
	})
}

func buildTypeExpr(schema map[string]interface{}, ruleNameHint string, g *Grammar) Expr {
	if schema == nil {
		return RefExpr{Name: "any_val"}
	}

	// 1. Enum
	if enumRaw, ok := schema["enum"].([]interface{}); ok && len(enumRaw) > 0 {
		var enumLits []Expr
		for _, item := range enumRaw {
			litStr := formatLiteralValue(item)
			enumLits = append(enumLits, LitExpr{Val: litStr})
		}
		return AltExpr{Alts: enumLits}
	}

	// 2. anyOf / oneOf
	for _, combKey := range []string{"anyOf", "oneOf"} {
		if combRaw, ok := schema[combKey].([]interface{}); ok && len(combRaw) > 0 {
			var unionAlts []Expr
			for i, sub := range combRaw {
				if subMap, ok := sub.(map[string]interface{}); ok {
					subHint := fmt.Sprintf("%s_sub%d", ruleNameHint, i)
					unionAlts = append(unionAlts, buildTypeExpr(subMap, subHint, g))
				}
			}
			return AltExpr{Alts: unionAlts}
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
			return RefExpr{Name: "any_val"}
		}
	}

	// Multiple types e.g. ["string", "null"]
	if tList, ok := tRaw.([]interface{}); ok {
		var alts []Expr
		for i, item := range tList {
			if s, ok := item.(string); ok {
				subHint := fmt.Sprintf("%s_t%d", ruleNameHint, i)
				alts = append(alts, buildTypeExpr(map[string]interface{}{"type": s}, subHint, g))
			}
		}
		return AltExpr{Alts: alts}
	}

	typeName, _ := tRaw.(string)
	switch typeName {
	case "string":
		return RefExpr{Name: "str_val"}
	case "integer":
		return RefExpr{Name: "int_val"}
	case "number":
		return RefExpr{Name: "float_val"}
	case "boolean":
		return RefExpr{Name: "bool_val"}
	case "null":
		return RefExpr{Name: "none_val"}
	case "array":
		items, hasItems := schema["items"].(map[string]interface{})
		if hasItems {
			itemRuleName := fmt.Sprintf("%s_item", ruleNameHint)
			itemExpr := buildTypeExpr(items, itemRuleName, g)
			g.Rules = append(g.Rules, GrammarRule{Name: itemRuleName, Expr: itemExpr})

			// "[" (itemRule (", " itemRule)*)? "]"
			repPart := SeqExpr{Items: []Expr{LitExpr{Val: ", "}, RefExpr{Name: itemRuleName}}}
			innerSeq := SeqExpr{Items: []Expr{RefExpr{Name: itemRuleName}, Rep0Expr{Inner: repPart}}}
			return SeqExpr{Items: []Expr{
				LitExpr{Val: "["},
				OptExpr{Inner: innerSeq},
				LitExpr{Val: "]"},
			}}
		}
		return RefExpr{Name: "any_list"}
	case "object":
		rawProps, hasProps := schema["properties"].(map[string]interface{})
		if !hasProps || len(rawProps) == 0 {
			return RefExpr{Name: "any_dict"}
		}
		return buildObjectStructureExpr(schema, ruleNameHint, g)
	default:
		return RefExpr{Name: "any_val"}
	}
}

func buildObjectStructureExpr(schema map[string]interface{}, ruleNameHint string, g *Grammar) Expr {
	rawProps, _ := schema["properties"].(map[string]interface{})
	reqSet := make(map[string]bool)
	if reqList, ok := schema["required"].([]interface{}); ok {
		for _, r := range reqList {
			if s, ok := r.(string); ok {
				reqSet[s] = true
			}
		}
	}

	var reqKeys, optKeys []string
	for k := range rawProps {
		if reqSet[k] {
			reqKeys = append(reqKeys, k)
		} else {
			optKeys = append(optKeys, k)
		}
	}
	sort.Strings(reqKeys)
	sort.Strings(optKeys)

	// Create property type rules
	for k, propVal := range rawProps {
		propSchema, _ := propVal.(map[string]interface{})
		pRuleName := fmt.Sprintf("%s_prop_%s", ruleNameHint, sanitizePropRuleName(k))
		pTypeExpr := buildTypeExpr(propSchema, pRuleName, g)
		g.Rules = append(g.Rules, GrammarRule{Name: pRuleName, Expr: pTypeExpr})
	}

	openBrace := LitExpr{Val: "{"}
	closeBrace := LitExpr{Val: "}"}

	if len(reqKeys) == 0 && len(optKeys) == 0 {
		return SeqExpr{Items: []Expr{openBrace, closeBrace}}
	}

	if len(reqKeys) > 0 {
		var reqItems []Expr
		for i, k := range reqKeys {
			prefix := fmt.Sprintf("\"%s\": ", k)
			if i > 0 {
				prefix = fmt.Sprintf(", \"%s\": ", k)
			}
			pRuleName := fmt.Sprintf("%s_prop_%s", ruleNameHint, sanitizePropRuleName(k))
			reqItems = append(reqItems, SeqExpr{Items: []Expr{LitExpr{Val: prefix}, RefExpr{Name: pRuleName}}})
		}

		var optItems []Expr
		for _, k := range optKeys {
			prefix := fmt.Sprintf(", \"%s\": ", k)
			pRuleName := fmt.Sprintf("%s_prop_%s", ruleNameHint, sanitizePropRuleName(k))
			optItems = append(optItems, OptExpr{Inner: SeqExpr{Items: []Expr{LitExpr{Val: prefix}, RefExpr{Name: pRuleName}}}})
		}

		allObjItems := []Expr{openBrace}
		allObjItems = append(allObjItems, reqItems...)
		allObjItems = append(allObjItems, optItems...)
		allObjItems = append(allObjItems, closeBrace)

		return SeqExpr{Items: allObjItems}
	}

	// All optional properties in object
	var altBranches []Expr
	altBranches = append(altBranches, LitExpr{Val: ""})

	for i := 0; i < len(optKeys); i++ {
		optBranchName := fmt.Sprintf("%s_opt_%d", ruleNameHint, i)
		altBranches = append(altBranches, RefExpr{Name: optBranchName})

		var branchItems []Expr
		firstKey := optKeys[i]
		firstPrefix := fmt.Sprintf("\"%s\": ", firstKey)
		firstRef := RefExpr{Name: fmt.Sprintf("%s_prop_%s", ruleNameHint, sanitizePropRuleName(firstKey))}
		branchItems = append(branchItems, SeqExpr{Items: []Expr{LitExpr{Val: firstPrefix}, firstRef}})

		for j := i + 1; j < len(optKeys); j++ {
			subKey := optKeys[j]
			subPrefix := fmt.Sprintf(", \"%s\": ", subKey)
			subRef := RefExpr{Name: fmt.Sprintf("%s_prop_%s", ruleNameHint, sanitizePropRuleName(subKey))}
			branchItems = append(branchItems, OptExpr{Inner: SeqExpr{Items: []Expr{LitExpr{Val: subPrefix}, subRef}}})
		}

		g.Rules = append(g.Rules, GrammarRule{
			Name: optBranchName,
			Expr: SeqExpr{Items: branchItems},
		})
	}

	argsRuleName := fmt.Sprintf("%s_body", ruleNameHint)
	g.Rules = append(g.Rules, GrammarRule{
		Name: argsRuleName,
		Expr: AltExpr{Alts: altBranches},
	})

	return SeqExpr{Items: []Expr{openBrace, RefExpr{Name: argsRuleName}, closeBrace}}
}

func sanitizePropRuleName(name string) string {
	var sb strings.Builder
	for _, r := range name {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '_' {
			sb.WriteRune(r)
		} else {
			sb.WriteByte('_')
		}
	}
	res := sb.String()
	if res == "" {
		return "val"
	}
	return res
}
