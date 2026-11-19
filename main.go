package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"math/big"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/maelie-labs/reckoner-harness/fncall"
)

type ToolRequest struct {
	ToolName  string          `json:"tool_name"`
	Tool      string          `json:"tool"`
	Payload   string          `json:"payload"`
	Input     string          `json:"input"`
	Call      string          `json:"call,omitempty"`
	Tokens    []int           `json:"tokens,omitempty"`
	Targets   []int           `json:"targets,omitempty"`
	MessageID *int            `json:"message_id,omitempty"`
	Prefix    []string        `json:"prefix,omitempty"`
	Store     *bool           `json:"store,omitempty"`
	Clock     string          `json:"clock,omitempty"`
	Anchors   *bool           `json:"anchors,omitempty"`
	NoAnchors *bool           `json:"no_anchors,omitempty"`
	Profile         *string         `json:"profile,omitempty"`
	Tokenizer       *string         `json:"tokenizer,omitempty"`
	Graded          *bool           `json:"graded,omitempty"`
	ReasoningEffort string          `json:"reasoning_effort,omitempty"`
	Messages        []OpenAIMessage `json:"messages,omitempty"`
	Tools           json.RawMessage `json:"tools,omitempty"`
}

type ToolResponse struct {
	Success         bool             `json:"success"`
	Result          string           `json:"result"`
	Error           string           `json:"error,omitempty"`
	ElapsedUs       int64            `json:"elapsed_us"`
	Tokens          []int            `json:"tokens,omitempty"`
	Targets         []int            `json:"targets,omitempty"`
	NextMessageID   *int             `json:"next_message_id,omitempty"`
	Receipt         string           `json:"receipt,omitempty"`
	Text            string           `json:"text,omitempty"`
	Provenance      *QuoteProvenance `json:"provenance,omitempty"`
	AllowedTokens   []string         `json:"allowed_tokens,omitempty"`
	AllowedTokenIDs []int            `json:"allowed_token_ids,omitempty"`
	Step            string           `json:"step,omitempty"`
	NumMatches      int              `json:"num_matches,omitempty"`
}

var globalRegisters = NewRegisterStore()

// DispatchTool dispatches against the global register store.
func DispatchTool(toolName, payload string) ToolResponse {
	return DispatchToolWithStore(globalRegisters, toolName, payload)
}

// DispatchToolWithStore dispatches a tool request against the specified RegisterStore with default 2s deadline.
func DispatchToolWithStore(store *RegisterStore, toolName, payload string) (resp ToolResponse) {
	return DispatchToolWithStoreAndDeadline(store, toolName, payload, 2*time.Second)
}

// DispatchToolWithStoreAndDeadline dispatches a tool request against the specified RegisterStore with custom deadline.
// Wrapped in recover() to ensure the harness process never terminates abruptly.
func DispatchToolWithStoreAndDeadline(store *RegisterStore, toolName, payload string, deadline time.Duration) (resp ToolResponse) {
	t0 := time.Now()
	defer func() {
		if r := recover(); r != nil {
			resp = ToolResponse{
				Success:   false,
				Error:     fmt.Sprintf("ErrInternal: panic recovered: %v", r),
				ElapsedUs: time.Since(t0).Microseconds(),
			}
		}
	}()

	toolName = strings.TrimSpace(toolName)
	payload = strings.TrimSpace(payload)

	if len(payload) > 1024*1024 {
		return ToolResponse{
			Success:   false,
			Error:     "ErrTooLong: payload exceeds 1 MiB limit",
			ElapsedUs: time.Since(t0).Microseconds(),
		}
	}

	type workerOutput struct {
		commit func() (string, error)
		result string
		err    error
	}
	outCh := make(chan workerOutput, 1)

	go func() {
		var out workerOutput

		defer func() {
			if r := recover(); r != nil {
				out.err = fmt.Errorf("ErrInternal: %v", r)
				out.commit = nil
				out.result = ""
			}
			outCh <- out
		}()

		switch toolName {
		case "calc", "<|call:calc|>":
			val, err := EvaluateInfixToVal(payload, store)
			if err != nil {
				out.err = err
			} else {
				out.commit = func() (string, error) {
					_, res, err := store.AutoBind(val)
					return res, err
				}
			}

		case "str", "<|call:str|>":
			val, err := EvaluateStrWithStore(payload, store)
			if err != nil {
				out.err = err
			} else {
				out.commit = func() (string, error) {
					_, res, err := store.AutoBind(val)
					return res, err
				}
			}

		case "date", "<|call:date|>":
			val, err := EvaluateDateWithStore(payload, store)
			if err != nil {
				out.err = err
			} else {
				out.commit = func() (string, error) {
					_, res, err := store.AutoBind(val)
					return res, err
				}
			}

		case "solve", "<|call:solve|>":
			solns, err := SolveEquationWithStore(payload, store)
			if err != nil {
				out.err = err
			} else if len(solns) == 0 {
				out.result = "no real solutions"
			} else if len(solns) == 1 && solns[0].Val.Kind == ValKindInterval {
				if len(solns[0].Val.Intervals) == 0 {
					out.result = "no solution"
				} else {
					v := solns[0].Val
					out.commit = func() (string, error) {
						_, res, err := store.AutoBind(v)
						return res, err
					}
				}
			} else {
				out.commit = func() (string, error) {
					var boundParts []string
					for _, s := range solns {
						_, boundStr, bErr := store.AutoBindNamed(s.VarName, s.Val)
						if bErr != nil {
							return "", bErr
						}
						boundParts = append(boundParts, boundStr)
					}
					return strings.Join(boundParts, ", "), nil
				}
			}

		case "reg", "<|call:reg|>":
			out.commit = func() (string, error) {
				return store.ExecuteRegisterCmd(payload)
			}

		case "lua", "<|call:lua|>":
			rawVal, err := ExecuteLua(payload, store)
			if err != nil {
				out.err = err
			} else {
				val := parseLuaResultToVal(rawVal)
				out.commit = func() (string, error) {
					_, res, err := store.AutoBind(val)
					return res, err
				}
			}

		case "fncall", "<|call:fncall|>":
			res, err := fncall.HandleIPC(payload)
			out.result = res
			out.err = err

		case "lookup", "<|call:lookup|>":
			out.commit = func() (string, error) {
				return EvaluateLookupWithStore(payload, store)
			}

		case "reset":
			out.commit = func() (string, error) {
				store.Reset()
				if ctxStore, err := GetGlobalContextStore(); err == nil {
					ctxStore.Reset()
				}
				return "OK", nil
			}

		case "quote", "<|quote|>":
			ctxStore, cErr := GetGlobalContextStore()
			if cErr != nil {
				out.err = cErr
			} else {
				qRes, err := ctxStore.Quote(payload)
				if err != nil {
					out.err = err
				} else {
					out.result = qRes.Text
				}
			}

		case "find", "<|call:find|>":
			ctxStore, cErr := GetGlobalContextStore()
			if cErr != nil {
				out.err = cErr
			} else {
				res, err := DispatchContextTool(ctxStore, "find", payload)
				out.result = res
				out.err = err
			}

		case "lines":
			ctxStore, cErr := GetGlobalContextStore()
			if cErr != nil {
				out.err = cErr
			} else {
				res, err := DispatchContextTool(ctxStore, "lines", payload)
				out.result = res
				out.err = err
			}

		case "quote_mask":
			ctxStore, cErr := GetGlobalContextStore()
			if cErr != nil {
				out.err = cErr
			} else {
				res, err := DispatchContextTool(ctxStore, "quote_mask", payload)
				out.result = res
				out.err = err
			}

		case "context_reset":
			ctxStore, cErr := GetGlobalContextStore()
			if cErr != nil {
				out.err = cErr
			} else {
				ctxStore.Reset()
				out.result = "OK"
			}

		case "anchor":
			ctxStore, cErr := GetGlobalContextStore()
			if cErr != nil {
				out.err = cErr
			} else {
				var rawTokens []int
				if err := json.Unmarshal([]byte(payload), &rawTokens); err == nil {
					anchored, _, _, err := ctxStore.StoreMessage(rawTokens, nil, nil)
					if err != nil {
						out.err = err
					} else {
						data, _ := json.Marshal(anchored)
						out.result = string(data)
					}
				} else {
					out.err = fmt.Errorf("ErrSyntax: anchor payload must be json array of token ids")
				}
			}

		case "format_prompt", "render_prompt":
			resStr, _, rErr := executeFormatPrompt(payload, nil, nil, nil, false, nil, nil, nil, nil, "")
			out.result = resStr
			out.err = rErr

		default:
			out.err = fmt.Errorf("ErrSyntax: unknown tool: %s", toolName)
		}
	}()

	var result string
	var err error

	select {
	case out := <-outCh:
		if out.err != nil {
			err = out.err
		} else if out.commit != nil {
			result, err = out.commit()
		} else {
			result = out.result
		}
	case <-time.After(deadline):
		err = fmt.Errorf("ErrTimeout: %s execution timed out (%v ceiling)", toolName, deadline)
	}

	elapsed := time.Since(t0).Microseconds()
	if err != nil {
		return ToolResponse{
			Success:   false,
			Error:     err.Error(),
			ElapsedUs: elapsed,
		}
	}

	return ToolResponse{
		Success:   true,
		Result:    result,
		ElapsedUs: elapsed,
	}
}

func resolveTokenizerForPrompt(prof *ModelProfile, tokPath string, modelDirs ...string) (*Tokenizer, error) {
	tok, _, err := LoadTokenizerForProfile(prof, tokPath, modelDirs...)
	return tok, err
}

func executeFormatPrompt(
	payload string,
	reqMessages []OpenAIMessage,
	reqTools json.RawMessage,
	reqProfile *ModelProfile,
	reqProfileSet bool,
	reqTokenizer *string,
	reqGraded *bool,
	reqAnchors *bool,
	reqNoAnchors *bool,
	reqReasoningEffort string,
) (string, []int, error) {
	prof := globalProfile
	if reqProfileSet {
		prof = reqProfile
	}
	explicitTokenizerProvided := reqTokenizer != nil
	var explicitTokPath string
	if reqTokenizer != nil {
		explicitTokPath = *reqTokenizer
	} else if !reqProfileSet || (prof != nil && globalProfile != nil && prof.Name == globalProfile.Name) {
		explicitTokPath = globalTokenizerPath
	}
	graded := globalGraded
	if reqGraded != nil {
		graded = *reqGraded
	}
	reasoningEffort := reqReasoningEffort
	messages := reqMessages
	toolsJSON := reqTools
	anchorCtx := false
	noAnchors := globalNoAnchorsFlag
	explicitAnchors := reqAnchors
	if reqAnchors != nil {
		anchorCtx = *reqAnchors
	}
	if reqNoAnchors != nil {
		if *reqNoAnchors {
			anchorCtx = false
			noAnchors = true
		} else if reqAnchors == nil {
			anchorCtx = true
		}
	}

	if len(messages) == 0 && payload != "" {
		type formatPayload struct {
			Messages        []OpenAIMessage `json:"messages"`
			Tools           json.RawMessage `json:"tools,omitempty"`
			Profile         *string         `json:"profile,omitempty"`
			Tokenizer       *string         `json:"tokenizer,omitempty"`
			Graded          *bool           `json:"graded,omitempty"`
			ReasoningEffort *string         `json:"reasoning_effort,omitempty"`
			AnchorContext   *bool           `json:"anchor_context,omitempty"`
			Anchors         *bool           `json:"anchors,omitempty"`
			NoAnchors       *bool           `json:"no_anchors,omitempty"`
		}
		var fp formatPayload
		if err := json.Unmarshal([]byte(payload), &fp); err == nil && len(fp.Messages) > 0 {
			messages = fp.Messages
			if len(fp.Tools) > 0 {
				toolsJSON = fp.Tools
			}
			if fp.Profile != nil && !reqProfileSet {
				if *fp.Profile == "" {
					prof = nil
				} else {
					p, pErr := LoadProfile(*fp.Profile)
					if pErr != nil {
						return "", nil, pErr
					}
					prof = p
				}
				if reqTokenizer == nil && fp.Tokenizer == nil {
					if prof != nil && globalProfile != nil && prof.Name == globalProfile.Name {
						explicitTokPath = globalTokenizerPath
					} else {
						explicitTokPath = ""
					}
				}
			}
			if fp.Tokenizer != nil && reqTokenizer == nil {
				explicitTokenizerProvided = true
				explicitTokPath = *fp.Tokenizer
			}
			if fp.Graded != nil && reqGraded == nil {
				graded = *fp.Graded
			}
			if fp.ReasoningEffort != nil && reasoningEffort == "" {
				reasoningEffort = *fp.ReasoningEffort
			}
			if fp.AnchorContext != nil {
				anchorCtx = *fp.AnchorContext
			}
			if fp.Anchors != nil {
				anchorCtx = *fp.Anchors
				explicitAnchors = fp.Anchors
			}
			if fp.NoAnchors != nil {
				if *fp.NoAnchors {
					anchorCtx = false
					noAnchors = true
				} else if fp.Anchors == nil && fp.AnchorContext == nil {
					anchorCtx = true
				}
			}
		} else {
			var msgArr []OpenAIMessage
			if err := json.Unmarshal([]byte(payload), &msgArr); err == nil && len(msgArr) > 0 {
				messages = msgArr
			} else {
				messages = []OpenAIMessage{{Role: "user", Content: payload}}
			}
		}
	}

	if _, _, _, err := resolveEffortOrErr(reasoningEffort); err != nil {
		return "", nil, err
	}

	if explicitTokenizerProvided && profileUsesExternalTokenizer(prof) && strings.TrimSpace(explicitTokPath) == "" {
		return "", nil, fmt.Errorf("ErrArgs: empty or whitespace-only tokenizer path for profile %q", prof.Name)
	}

	if prof != nil && explicitAnchors != nil && !*explicitAnchors {
		anchorCtx = false
	}

	tok, err := resolveTokenizerForPrompt(prof, explicitTokPath, globalModelDir)
	if err != nil {
		return "", nil, err
	}

	rendered, _, err := RenderChatPromptWithProfileAndEffort(
		tok,
		messages,
		toolsJSON,
		anchorCtx,
		prof,
		graded,
		noAnchors,
		reasoningEffort,
	)
	if err != nil {
		return "", nil, err
	}
	return tok.DecodeTokens(rendered.InputIDs), rendered.InputIDs, nil
}

func parseLuaResultToVal(s string) CalcVal {
	s = strings.TrimSpace(s)
	if s == "true" {
		return CalcVal{Kind: ValKindBool, Bool: true}
	}
	if s == "false" {
		return CalcVal{Kind: ValKindBool, Bool: false}
	}
	if n, err := strconv.ParseInt(s, 10, 64); err == nil {
		return CalcVal{Kind: ValKindRat, Rat: big.NewRat(n, 1)}
	}
	if f, err := strconv.ParseFloat(s, 64); err == nil {
		return CalcVal{Kind: ValKindFloat, Float: f}
	}
	if strings.HasPrefix(s, `"`) && strings.HasSuffix(s, `"`) && len(s) >= 2 {
		if unquoted, err := strconv.Unquote(s); err == nil {
			return CalcVal{Kind: ValKindString, Str: unquoted}
		}
	}
	if strings.HasPrefix(s, "[") && strings.HasSuffix(s, "]") {
		return CalcVal{Kind: ValKindRawString, Str: s}
	}
	if strings.Contains(s, ", ") {
		return CalcVal{Kind: ValKindRawString, Str: "[" + s + "]"}
	}
	return CalcVal{Kind: ValKindString, Str: s}
}

var (
	version             = "1.1.3"
	globalProfile       *ModelProfile
	globalTokenizerPath string
	globalModelDir      string
	globalGraded        bool
	globalNoAnchorsFlag bool
)

func DispatchJSONRequest(req ToolRequest, store *RegisterStore) (resp ToolResponse) {
	t0 := time.Now()
	defer func() {
		if r := recover(); r != nil {
			resp = ToolResponse{
				Success:   false,
				Error:     fmt.Sprintf("ErrInternal: panic recovered: %v", r),
				ElapsedUs: time.Since(t0).Microseconds(),
			}
		}
	}()

	tool := req.ToolName
	if tool == "" {
		tool = req.Tool
	}
	payload := req.Payload
	if payload == "" && req.Input != "" {
		payload = req.Input
	}
	if payload == "" && req.Call != "" {
		payload = req.Call
	}

	if req.Clock != "" {
		clkTime, err := ParseClockTimestamp(req.Clock)
		if err != nil {
			return ToolResponse{
				Success:   false,
				Error:     err.Error(),
				ElapsedUs: time.Since(t0).Microseconds(),
			}
		}
		if store != nil {
			store.SetClock(&clkTime)
			defer store.SetClock(nil)
		}
	}

	reqProfile := globalProfile
	reqProfileSet := false
	if req.Profile != nil {
		reqProfileSet = true
		if *req.Profile == "" {
			reqProfile = nil
		} else {
			var pErr error
			reqProfile, pErr = LoadProfile(*req.Profile)
			if pErr != nil {
				return ToolResponse{
					Success:   false,
					Error:     pErr.Error(),
					ElapsedUs: time.Since(t0).Microseconds(),
				}
			}
		}
	}

	if req.Profile != nil || req.Anchors != nil || req.NoAnchors != nil {
		if ctxStore, err := GetGlobalContextStore(); err == nil {
			if reqProfile != nil {
				noAnchors := globalNoAnchorsFlag || (req.NoAnchors != nil && *req.NoAnchors)
				ctxStore.SetAnchorsEnabled(ResolveEffectiveAnchors(reqProfile, noAnchors, req.Anchors))
			} else {
				// Preserve the original flag precedence when no profile is active:
				// req.Anchors wins over req.NoAnchors, and per-request fields override CLI flags.
				if req.Anchors != nil {
					ctxStore.SetAnchorsEnabled(*req.Anchors)
				} else if req.NoAnchors != nil {
					ctxStore.SetAnchorsEnabled(!*req.NoAnchors)
				} else if req.Profile != nil {
					ctxStore.SetAnchorsEnabled(!globalNoAnchorsFlag)
				}
			}
		}
	}

	switch tool {
	case "session_config", "config":
		oldProfName := ""
		oldProfPath := ""
		if globalProfile != nil {
			oldProfName = globalProfile.Name
			oldProfPath = globalProfile.SourcePath
		}
		newProfile := globalProfile
		if req.Profile != nil {
			newProfile = reqProfile
		}
		newProfName := ""
		newProfPath := ""
		if newProfile != nil {
			newProfName = newProfile.Name
			newProfPath = newProfile.SourcePath
		}
		newTokPath := globalTokenizerPath
		if req.Profile != nil && (newProfName != oldProfName || newProfPath != oldProfPath) && req.Tokenizer == nil {
			newTokPath = ""
		}
		if req.Tokenizer != nil {
			newTokPath = *req.Tokenizer
		}
		if strings.TrimSpace(newTokPath) != "" && !profileUsesExternalTokenizer(newProfile) {
			profName := "default (no profile)"
			if newProfile != nil && newProfile.Name != "" {
				profName = fmt.Sprintf("profile %q", newProfile.Name)
			}
			return ToolResponse{
				Success:   false,
				Error:     fmt.Sprintf("ErrArgs: external tokenizer path %q cannot be used with %s (which uses the embedded tokenizer)", newTokPath, profName),
				ElapsedUs: time.Since(t0).Microseconds(),
			}
		}
		if req.Tokenizer != nil && profileUsesExternalTokenizer(newProfile) {
			if strings.TrimSpace(newTokPath) == "" {
				return ToolResponse{
					Success:   false,
					Error:     fmt.Sprintf("ErrArgs: empty or whitespace-only tokenizer path for profile %q", newProfile.Name),
					ElapsedUs: time.Since(t0).Microseconds(),
				}
			}
			if _, _, err := LoadTokenizerForProfile(newProfile, newTokPath, globalModelDir); err != nil {
				return ToolResponse{
					Success:   false,
					Error:     err.Error(),
					ElapsedUs: time.Since(t0).Microseconds(),
				}
			}
		}
		globalProfile = newProfile
		globalTokenizerPath = newTokPath
		if req.Graded != nil {
			globalGraded = *req.Graded
		}
		if payload != "" {
			if strings.Contains(payload, "no_anchors") || strings.Contains(payload, "anchors=false") {
				if ctxStore, err := GetGlobalContextStore(); err == nil {
					ctxStore.SetAnchorsEnabled(false)
				}
			} else if strings.Contains(payload, "anchors=true") {
				if ctxStore, err := GetGlobalContextStore(); err == nil {
					if globalProfile != nil {
						t := true
						noAnchors := globalNoAnchorsFlag || (req.NoAnchors != nil && *req.NoAnchors)
						ctxStore.SetAnchorsEnabled(ResolveEffectiveAnchors(globalProfile, noAnchors, &t))
					} else {
						ctxStore.SetAnchorsEnabled(true)
					}
				}
			}
		}
		return ToolResponse{
			Success:   true,
			Result:    "OK",
			ElapsedUs: time.Since(t0).Microseconds(),
		}

	case "anchor":
		ctxStore, err := GetGlobalContextStore()
		if err != nil {
			return ToolResponse{
				Success:   false,
				Error:     fmt.Sprintf("ErrInternal: context store init failed: %v", err),
				ElapsedUs: time.Since(t0).Microseconds(),
			}
		}

		tokens := req.Tokens
		targets := req.Targets
		msgID := req.MessageID

		if len(tokens) == 0 && payload != "" {
			type anchorPayload struct {
				Tokens    []int `json:"tokens"`
				Targets   []int `json:"targets"`
				MessageID *int  `json:"message_id"`
			}
			var ap anchorPayload
			if err := json.Unmarshal([]byte(payload), &ap); err == nil && (len(ap.Tokens) > 0 || ap.MessageID != nil) {
				tokens = ap.Tokens
				targets = ap.Targets
				msgID = ap.MessageID
			} else {
				var rawTokens []int
				if err := json.Unmarshal([]byte(payload), &rawTokens); err == nil {
					tokens = rawTokens
				} else {
					return ToolResponse{
						Success:   false,
						Error:     "ErrSyntax: anchor payload must be valid JSON object or token array",
						ElapsedUs: time.Since(t0).Microseconds(),
					}
				}
			}
		}

		if msgID != nil && *msgID < 0 {
			return ToolResponse{
				Success:   false,
				Error:     fmt.Sprintf("ErrArgs: message_id must be non-negative: %d", *msgID),
				ElapsedUs: time.Since(t0).Microseconds(),
			}
		}

		shouldStore := true
		if req.Store != nil {
			shouldStore = *req.Store
		}

		var outTokens, outTargets []int
		var nextMsg int
		if shouldStore {
			outTokens, outTargets, nextMsg, err = ctxStore.StoreMessage(tokens, targets, msgID)
		} else {
			if !ctxStore.AnchorsEnabled() {
				outTokens = tokens
				outTargets = targets
				reqID := ctxStore.NextMessageID()
				if msgID != nil {
					reqID = *msgID
				}
				nextMsg = (reqID + 1) % NumMessageTokens
			} else {
				reqID := 0
				if msgID != nil {
					reqID = *msgID
				}
				outTokens, outTargets, nextMsg, err = ctxStore.tok.InsertAnchors(tokens, targets, reqID)
			}
		}

		if err != nil {
			return ToolResponse{
				Success:   false,
				Error:     err.Error(),
				ElapsedUs: time.Since(t0).Microseconds(),
			}
		}

		resMap := map[string]interface{}{
			"tokens":          outTokens,
			"targets":         outTargets,
			"next_message_id": nextMsg,
		}
		data, _ := json.Marshal(resMap)

		return ToolResponse{
			Success:       true,
			Result:        string(data),
			Tokens:        outTokens,
			Targets:       outTargets,
			NextMessageID: &nextMsg,
			ElapsedUs:     time.Since(t0).Microseconds(),
		}

	case "quote", "<|quote|>":
		ctxStore, err := GetGlobalContextStore()
		if err != nil {
			return ToolResponse{
				Success:   false,
				Error:     fmt.Sprintf("ErrInternal: context store init failed: %v", err),
				ElapsedUs: time.Since(t0).Microseconds(),
			}
		}
		call := payload
		if call == "" {
			call = req.Call
		}
		qRes, err := ctxStore.Quote(call)
		if err != nil {
			return ToolResponse{
				Success:   false,
				Error:     err.Error(),
				ElapsedUs: time.Since(t0).Microseconds(),
			}
		}
		return ToolResponse{
			Success:    true,
			Result:     qRes.Text,
			Text:       qRes.Text,
			Receipt:    qRes.Receipt,
			Tokens:     qRes.Tokens,
			Provenance: &qRes.Provenance,
			ElapsedUs:  time.Since(t0).Microseconds(),
		}

	case "quote_mask":
		ctxStore, err := GetGlobalContextStore()
		if err != nil {
			return ToolResponse{
				Success:   false,
				Error:     fmt.Sprintf("ErrInternal: context store init failed: %v", err),
				ElapsedUs: time.Since(t0).Microseconds(),
			}
		}
		prefix := req.Prefix
		if len(prefix) == 0 && payload != "" {
			var p []string
			if err := json.Unmarshal([]byte(payload), &p); err == nil {
				prefix = p
			} else {
				prefix = strings.Fields(payload)
			}
		}
		maskRes, err := ctxStore.QuoteMask(prefix)
		if err != nil {
			return ToolResponse{
				Success:   false,
				Error:     err.Error(),
				ElapsedUs: time.Since(t0).Microseconds(),
			}
		}
		return ToolResponse{
			Success:         true,
			AllowedTokens:   maskRes.AllowedTokens,
			AllowedTokenIDs: maskRes.AllowedTokenIDs,
			Step:            maskRes.Step,
			ElapsedUs:       time.Since(t0).Microseconds(),
		}

	case "find", "<|call:find|>":
		ctxStore, err := GetGlobalContextStore()
		if err != nil {
			return ToolResponse{
				Success:   false,
				Error:     fmt.Sprintf("ErrInternal: context store init failed: %v", err),
				ElapsedUs: time.Since(t0).Microseconds(),
			}
		}
		isLines, q, scopeMsg, s, e, err := ParseFindOrLinesCall(payload)
		if err != nil {
			return ToolResponse{
				Success:   false,
				Error:     err.Error(),
				ElapsedUs: time.Since(t0).Microseconds(),
			}
		}
		var fRes *FindResult
		if isLines {
			fRes, err = ctxStore.Lines(*scopeMsg, s, e)
		} else {
			fRes, err = ctxStore.Find(q, scopeMsg)
		}
		if err != nil {
			return ToolResponse{
				Success:   false,
				Error:     err.Error(),
				ElapsedUs: time.Since(t0).Microseconds(),
			}
		}
		return ToolResponse{
			Success:    true,
			Result:     fRes.Result,
			Tokens:     fRes.Tokens,
			NumMatches: fRes.NumMatches,
			ElapsedUs:  time.Since(t0).Microseconds(),
		}

	case "lines":
		ctxStore, err := GetGlobalContextStore()
		if err != nil {
			return ToolResponse{
				Success:   false,
				Error:     fmt.Sprintf("ErrInternal: context store init failed: %v", err),
				ElapsedUs: time.Since(t0).Microseconds(),
			}
		}
		isLines, _, scopeMsg, s, e, err := ParseFindOrLinesCall(payload)
		if err != nil || !isLines || scopeMsg == nil {
			return ToolResponse{
				Success:   false,
				Error:     "ErrSyntax: expected lines(msg, start, end)",
				ElapsedUs: time.Since(t0).Microseconds(),
			}
		}
		fRes, err := ctxStore.Lines(*scopeMsg, s, e)
		if err != nil {
			return ToolResponse{
				Success:   false,
				Error:     err.Error(),
				ElapsedUs: time.Since(t0).Microseconds(),
			}
		}
		return ToolResponse{
			Success:    true,
			Result:     fRes.Result,
			Tokens:     fRes.Tokens,
			NumMatches: fRes.NumMatches,
			ElapsedUs:  time.Since(t0).Microseconds(),
		}

	case "context_reset":
		ctxStore, err := GetGlobalContextStore()
		if err != nil {
			return ToolResponse{
				Success:   false,
				Error:     fmt.Sprintf("ErrInternal: context store init failed: %v", err),
				ElapsedUs: time.Since(t0).Microseconds(),
			}
		}
		ctxStore.Reset()
		return ToolResponse{
			Success:   true,
			Result:    "OK",
			ElapsedUs: time.Since(t0).Microseconds(),
		}

	case "format_prompt", "render_prompt":
		resStr, tokIDs, rErr := executeFormatPrompt(payload, req.Messages, req.Tools, reqProfile, reqProfileSet, req.Tokenizer, req.Graded, req.Anchors, req.NoAnchors, req.ReasoningEffort)
		if rErr != nil {
			return ToolResponse{
				Success:   false,
				Error:     rErr.Error(),
				ElapsedUs: time.Since(t0).Microseconds(),
			}
		}
		return ToolResponse{
			Success:   true,
			Result:    resStr,
			Tokens:    tokIDs,
			ElapsedUs: time.Since(t0).Microseconds(),
		}

	case "tokenize", "encode":
		// Use untrimmed req.Payload/req.Input directly so leading/trailing whitespace is preserved.
		rawText := req.Payload
		if rawText == "" && req.Input != "" {
			rawText = req.Input
		}
		var tokPath string
		if req.Tokenizer != nil {
			tokPath = *req.Tokenizer
			if profileUsesExternalTokenizer(reqProfile) && strings.TrimSpace(tokPath) == "" {
				return ToolResponse{
					Success:   false,
					Error:     fmt.Sprintf("ErrArgs: empty or whitespace-only tokenizer path for profile %q", reqProfile.Name),
					ElapsedUs: time.Since(t0).Microseconds(),
				}
			}
		} else if !reqProfileSet || (reqProfile != nil && globalProfile != nil && reqProfile.Name == globalProfile.Name) {
			tokPath = globalTokenizerPath
		}
		tok, _, tErr := LoadTokenizerForProfile(reqProfile, tokPath, globalModelDir)
		if tErr != nil {
			return ToolResponse{
				Success:   false,
				Error:     tErr.Error(),
				ElapsedUs: time.Since(t0).Microseconds(),
			}
		}
		ids := tok.Encode(rawText)
		if ids == nil {
			ids = []int{}
		}
		decoded := tok.DecodeTokens(ids)
		return ToolResponse{
			Success:   true,
			Result:    decoded,
			Text:      decoded,
			Tokens:    ids,
			ElapsedUs: time.Since(t0).Microseconds(),
		}

	default:
		return DispatchToolWithStore(store, tool, payload)
	}
}

func printUsage() {
	fmt.Println("Reckoner Harness: Deterministic tool executive engine")
	fmt.Println("")
	fmt.Println("Usage:")
	fmt.Println("  reckoner-harness [options] <tool> <payload>   # Single-shot CLI execution")
	fmt.Println("  reckoner-harness [options]                    # Interactive newline-delimited JSON streaming over stdin/stdout")
	fmt.Println("  reckoner-harness serve [options]              # OpenAI-compatible HTTP/SSE front door for SGLang")
	fmt.Println("")
	fmt.Println("Options:")
	fmt.Println("  --profile <name|path> Load model profile (e.g., reckoner-1.4b, reckoner-q2b, or JSON file path)")
	fmt.Println("  --tokenizer <path>    Override tokenizer.json file or directory path for profile-driven tokenization")
	fmt.Println("  --model-dir <dir>     Model directory for profile-driven tokenizer resolution")
	fmt.Println("  --graded              Append the profile's graded_answer_instruction to the last user turn")
	fmt.Println("  --clock <timestamp>   Override wall clock (ISO 8601 / RFC 3339) for deterministic execution")
	fmt.Println("  --no-anchors          Disable context reference anchors (<|mN|>, <|LN|>) for plain-text find/lines")
	fmt.Println("  -h, --help            Show this help message")
	fmt.Println("  -v, --version         Show version information")
	fmt.Println("")
	fmt.Println("Supported tools:")
	fmt.Println("  calc        Exact arithmetic, polynomials, linear algebra, geometry, units")
	fmt.Println("  solve       Symbolic equation and inequality solver")
	fmt.Println("  lua         Sandboxed Lua 5.4 runtime")
	fmt.Println("  str         Rune-aware string manipulation")
	fmt.Println("  date        Calendar arithmetic")
	fmt.Println("  lookup      Offline scientific constants, elements, astronomical data")
	fmt.Println("  reg         Register operations (r0-r15)")
	fmt.Println("  fncall      External JSON tool call parsing, validation and rendering")
	fmt.Println("  anchor      Deterministic context reference anchor insertion (<|m0|>..<|m255|>, <|L0|>..<|L1023|>)")
	fmt.Println("  quote       Byte-exact context range resolution and receipt generation (<|quote|>..<|/quote|>)")
	fmt.Println("  quote_mask  Constrained decoding next-token mask for quote sequences")
	fmt.Println("  find        Normalized exact substring search and BM25 ranked fallback across context messages (top 3 matches with ±2 lines)")
	fmt.Println("  lines       Extract line range from context message as newly anchored message")
}

func main() {
	var filteredArgs []string
	clockFlagVal := os.Getenv("RECKONER_HARNESS_CLOCK")
	profileFlagVal := ""
	tokenizerFlagVal := ""
	modelDirFlagVal := ""
	var explicitAnchors *bool
	noAnchorsFlag := false
	gradedFlag := false

	for i := 1; i < len(os.Args); i++ {
		arg := os.Args[i]
		if arg == "--clock" && i+1 < len(os.Args) {
			clockFlagVal = os.Args[i+1]
			i++
		} else if strings.HasPrefix(arg, "--clock=") {
			clockFlagVal = strings.TrimPrefix(arg, "--clock=")
		} else if arg == "--profile" && i+1 < len(os.Args) {
			profileFlagVal = os.Args[i+1]
			i++
		} else if strings.HasPrefix(arg, "--profile=") {
			profileFlagVal = strings.TrimPrefix(arg, "--profile=")
		} else if arg == "--tokenizer" && i+1 < len(os.Args) {
			tokenizerFlagVal = os.Args[i+1]
			i++
		} else if strings.HasPrefix(arg, "--tokenizer=") {
			tokenizerFlagVal = strings.TrimPrefix(arg, "--tokenizer=")
		} else if (arg == "--model-dir" || arg == "--model-path" || arg == "--model") && i+1 < len(os.Args) {
			modelDirFlagVal = os.Args[i+1]
			i++
		} else if strings.HasPrefix(arg, "--model-dir=") {
			modelDirFlagVal = strings.TrimPrefix(arg, "--model-dir=")
		} else if strings.HasPrefix(arg, "--model-path=") {
			modelDirFlagVal = strings.TrimPrefix(arg, "--model-path=")
		} else if strings.HasPrefix(arg, "--model=") {
			modelDirFlagVal = strings.TrimPrefix(arg, "--model=")
		} else if arg == "--graded" || arg == "--graded=true" {
			gradedFlag = true
		} else if arg == "--graded=false" {
			gradedFlag = false
		} else if arg == "--no-anchors" || arg == "--anchors=false" {
			noAnchorsFlag = true
			f := false
			explicitAnchors = &f
		} else if arg == "--anchors" || arg == "--anchors=true" {
			noAnchorsFlag = false
			t := true
			explicitAnchors = &t
		} else {
			filteredArgs = append(filteredArgs, arg)
		}
	}

	if profileFlagVal != "" {
		prof, err := LoadProfile(profileFlagVal)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %s\n", err.Error())
			os.Exit(1)
		}
		globalProfile = prof
	}
	globalTokenizerPath = tokenizerFlagVal
	globalModelDir = modelDirFlagVal
	globalGraded = gradedFlag
	globalNoAnchorsFlag = noAnchorsFlag

	effectiveAnchors := ResolveEffectiveAnchors(globalProfile, noAnchorsFlag, explicitAnchors)
	if !effectiveAnchors {
		if ctxStore, err := GetGlobalContextStore(); err == nil {
			ctxStore.SetAnchorsEnabled(false)
		}
	}

	if clockFlagVal != "" {
		clkTime, err := ParseClockTimestamp(clockFlagVal)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %s\n", err.Error())
			os.Exit(1)
		}
		SetGlobalDefaultClock(&clkTime)
	}

	if len(filteredArgs) >= 1 && filteredArgs[0] == "serve" {
		if err := RunServeCommand(filteredArgs[1:]); err != nil {
			fmt.Fprintf(os.Stderr, "Error: %s\n", err.Error())
			os.Exit(1)
		}
		return
	}

	if strings.TrimSpace(tokenizerFlagVal) != "" && !profileUsesExternalTokenizer(globalProfile) {
		_, _, err := LoadTokenizerForProfile(globalProfile, tokenizerFlagVal, modelDirFlagVal)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %s\n", err.Error())
			os.Exit(1)
		}
	}

	if len(filteredArgs) == 1 {
		arg := filteredArgs[0]
		if arg == "-h" || arg == "--help" || arg == "help" {
			printUsage()
			return
		}
		if arg == "-v" || arg == "--version" || arg == "version" {
			fmt.Printf("reckoner-harness %s\n", version)
			return
		}
	}

	if len(filteredArgs) >= 2 {
		// CLI mode: ./reckoner-harness calc "482 * 19 - 14"
		toolName := filteredArgs[0]
		payload := strings.Join(filteredArgs[1:], " ")
		resp := DispatchTool(toolName, payload)
		if resp.Success {
			fmt.Println(resp.Result)
		} else {
			fmt.Fprintf(os.Stderr, "Error: %s\n", resp.Error)
			os.Exit(1)
		}
		return
	}

	// Interactive Stdin/Stdout JSON streaming mode for Python subprocess integration.
	// Requests up to 1 MiB are read with bufio.Reader. Longer requests get ErrTooLong.
	const maxRequestBytes = 1024 * 1024 // 1 MiB
	reader := bufio.NewReaderSize(os.Stdin, maxRequestBytes)

	// Eagerly warm tokenizer in background so first context tool call is microsecond-scale
	go func() {
		_, _ = GetGlobalContextStore()
	}()

	for {
		lineBytes, isPrefix, err := reader.ReadLine()
		if err != nil {
			break
		}

		if isPrefix {
			// Line exceeded 1 MiB limit! Discard remainder of line and return ErrTooLong
			for isPrefix && err == nil {
				_, isPrefix, err = reader.ReadLine()
			}
			resp := ToolResponse{
				Success: false,
				Error:   "ErrTooLong: request line exceeds 1 MiB limit",
			}
			data, _ := json.Marshal(resp)
			fmt.Println(string(data))
			continue
		}

		line := strings.TrimSpace(string(lineBytes))
		if line == "" {
			continue
		}

		var req ToolRequest
		if err := json.Unmarshal([]byte(line), &req); err != nil {
			resp := ToolResponse{
				Success: false,
				Error:   fmt.Sprintf("ErrSyntax: invalid json: %v", err),
			}
			data, _ := json.Marshal(resp)
			fmt.Println(string(data))
			continue
		}

		resp := DispatchJSONRequest(req, globalRegisters)
		data, _ := json.Marshal(resp)
		fmt.Println(string(data))
	}
}
