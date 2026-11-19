package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/maelie-labs/reckoner-harness/fncall"
)

// Special token IDs from tokenizer_data/tokenizer.json.gz
const (
	TokEndOfText    = 0
	TokPad          = 1
	TokTurnUser     = 2
	TokTurnAgent    = 3
	TokEndTurn      = 4
	TokTurnSystem   = 5
	TokToolsOpen    = 6
	TokToolsClose   = 7
	TokDeliberate   = 8
	TokAct          = 9
	TokReady        = 10
	TokClarify      = 11
	TokPause        = 12
	TokCallExternal = 13
	TokEndCall      = 14
	TokObs          = 15
	TokObsErr       = 16
	TokEndObs       = 17
	TokCallCalc     = 18
	TokCallSolve    = 19
	TokCallLua      = 20
	TokCallStr      = 29
	TokCallDate     = 30
	TokCallLookup   = 31
	TokCallFind     = 32
	TokQuote        = 33
	TokEndQuote     = 34
	TokNewline      = 326
)

// DefaultReckonerProcessorStr is the deterministic CustomLogitProcessor.to_str()
// payload for serve.sglang_reckoner.tool_processor.ReckonerToolLogitProcessor.
const DefaultReckonerProcessorStr = `{"callable": "80049547000000000000008c2473657276652e73676c616e675f7265636b6f6e65722e746f6f6c5f70726f636573736f72948c1a5265636b6f6e6572546f6f6c4c6f67697450726f636573736f729493942e"}`

const DefaultProcessorInfoPath = "/dev/shm/reckoner_sgl_processor.json"

var callTokenToToolName = map[int]string{
	TokCallCalc:     "calc",
	TokCallSolve:    "solve",
	TokCallLua:      "lua",
	TokCallStr:      "str",
	TokCallDate:     "date",
	TokCallLookup:   "lookup",
	TokCallFind:     "find",
	TokCallExternal: "external",
}

// ServeConfig holds configuration for `reckoner-harness serve`.
type ServeConfig struct {
	Host              string
	Port              int
	SGLangAddr        string // e.g. "127.0.0.1:30000", "http://127.0.0.1:30000", "grpc://127.0.0.1:30000", "unix:///tmp/sgl.sock"
	SGLangProto       string // "auto", "http", or "grpc"
	DefaultMode       string // "native" (in-path) or "stop-and-continue"
	ProcessorStr      string
	AnchorByDefault   bool
	NoAnchorsOverride bool
	GradedByDefault   bool
	Profile           *ModelProfile
	TokenizerPath     string
	ModelDir          string
	Tokenizer         *Tokenizer
	MaxToolCalls      int
}

// OpenAIMessage represents an input or output chat message in /v1/chat/completions.
type OpenAIMessage struct {
	Role                string           `json:"role"`
	Content             string           `json:"content"`
	ReasoningContent    string           `json:"reasoning_content,omitempty"`
	ToolCalls           []OpenAIToolCall `json:"tool_calls,omitempty"`
	ToolCallID          string           `json:"tool_call_id,omitempty"`
	Name                string           `json:"name,omitempty"`
	harnessEffortPrefix bool             `json:"-"`
}

// OpenAIToolCall represents an external function call in OpenAI format.
type OpenAIToolCall struct {
	Index    *int               `json:"index,omitempty"`
	ID       string             `json:"id"`
	Type     string             `json:"type"`
	Function OpenAIFunctionCall `json:"function"`
}

// OpenAIFunctionCall holds the function name and JSON-encoded arguments string.
type OpenAIFunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// OpenAIChatCompletionRequest represents a POST /v1/chat/completions payload.
type OpenAIChatCompletionRequest struct {
	Model           string          `json:"model"`
	Messages        []OpenAIMessage `json:"messages"`
	Tools           json.RawMessage `json:"tools,omitempty"`
	Stream          bool            `json:"stream,omitempty"`
	Temperature     *float64        `json:"temperature,omitempty"`
	TopP            *float64        `json:"top_p,omitempty"`
	MaxTokens       *int            `json:"max_tokens,omitempty"`
	MaxCompletion   *int            `json:"max_completion_tokens,omitempty"`
	ReasoningEffort string          `json:"reasoning_effort,omitempty"`
	ServingMode     string          `json:"serving_mode,omitempty"` // optional override: "native" or "stop_and_continue"
	ToolMode        string          `json:"tool_mode,omitempty"`    // "all", "none", "lua_only"
	ForceSystem     string          `json:"force_system,omitempty"` // "sys1", "sys2", "ready", "act"
	AnchorContext   *bool           `json:"anchor_context,omitempty"`
	Anchors         *bool           `json:"anchors,omitempty"`
	NoAnchors       *bool           `json:"no_anchors,omitempty"`
	Graded          *bool           `json:"graded,omitempty"`
	Profile         *string         `json:"profile,omitempty"`
}

// ToolCallTelemetry records latency breakdown for a single in-path or stop-and-continue tool call.
type ToolCallTelemetry struct {
	Tool              string  `json:"tool"`
	Payload           string  `json:"payload"`
	Result            string  `json:"result"`
	IsError           bool    `json:"is_error"`
	LatencyUs         float64 `json:"latency_us"`
	HarnessIPCMs      float64 `json:"harness_ipc_ms"`
	PauseResumeMs     float64 `json:"pause_resume_ms"`
	ObsPrefillMs      float64 `json:"obs_prefill_ms"`
	CallToNextTokenMs float64 `json:"call_to_next_token_ms"`
	ObsTokens         int     `json:"obs_tokens"`
	CachedTokens      int     `json:"cached_tokens,omitempty"`
	RecomputedTokens  int     `json:"recomputed_tokens,omitempty"`
}

// ReckonerTelemetry provides detailed serving and tool execution metrics per completion.
type ReckonerTelemetry struct {
	ServingMode      string              `json:"serving_mode"`
	ActionMode       string              `json:"action_mode"`
	TTFTMs           float64             `json:"ttft_ms"`
	ElapsedMs        float64             `json:"elapsed_ms"`
	DecodeTokPerSec  float64             `json:"decode_tok_s"`
	PromptTokens     int                 `json:"prompt_tokens"`
	CompletionTokens int                 `json:"completion_tokens"`
	ThinkingTokens   int                 `json:"thinking_tokens"`
	ToolCalls        []ToolCallTelemetry `json:"tool_calls"`
	CachedTokens     int                 `json:"cached_tokens"`
	RecomputedTokens int                 `json:"recomputed_tokens"`
	FormatErrors     []string            `json:"format_errors,omitempty"`
}

// OpenAIUsage reports token usage counts.
type OpenAIUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// OpenAIChatChoice is a single completion choice in non-streaming mode.
type OpenAIChatChoice struct {
	Index        int           `json:"index"`
	Message      OpenAIMessage `json:"message"`
	FinishReason string        `json:"finish_reason"`
}

// OpenAIChatCompletionResponse is the full non-streaming /v1/chat/completions response.
type OpenAIChatCompletionResponse struct {
	ID                string             `json:"id"`
	Object            string             `json:"object"`
	Created           int64              `json:"created"`
	Model             string             `json:"model"`
	Choices           []OpenAIChatChoice `json:"choices"`
	Usage             OpenAIUsage        `json:"usage"`
	ReckonerTelemetry *ReckonerTelemetry `json:"reckoner_telemetry,omitempty"`
}

// OpenAIStreamDelta represents a streaming token delta in SSE mode.
type OpenAIStreamDelta struct {
	Role             string           `json:"role,omitempty"`
	Content          string           `json:"content,omitempty"`
	ReasoningContent string           `json:"reasoning_content,omitempty"`
	ToolCalls        []OpenAIToolCall `json:"tool_calls,omitempty"`
}

// OpenAIStreamChoice wraps a single streaming choice.
type OpenAIStreamChoice struct {
	Index        int               `json:"index"`
	Delta        OpenAIStreamDelta `json:"delta"`
	FinishReason *string           `json:"finish_reason"`
}

// OpenAIStreamChunk is a single SSE event payload in streaming mode.
type OpenAIStreamChunk struct {
	ID                string               `json:"id"`
	Object            string               `json:"object"`
	Created           int64                `json:"created"`
	Model             string               `json:"model"`
	Choices           []OpenAIStreamChoice `json:"choices"`
	Usage             *OpenAIUsage         `json:"usage,omitempty"`
	ReckonerTelemetry *ReckonerTelemetry   `json:"reckoner_telemetry,omitempty"`
}

var reqCounter uint64

func nextSessionID() string {
	n := atomic.AddUint64(&reqCounter, 1)
	return fmt.Sprintf("sgl-req-%d-%d", time.Now().UnixNano(), n)
}

func resolveProcessorStr(configured string) string {
	if configured != "" {
		return configured
	}
	if data, err := os.ReadFile(DefaultProcessorInfoPath); err == nil {
		var info struct {
			CustomLogitProcessor string `json:"custom_logit_processor"`
		}
		if json.Unmarshal(data, &info) == nil && info.CustomLogitProcessor != "" {
			return info.CustomLogitProcessor
		}
	}
	return DefaultReckonerProcessorStr
}

var stopContinueBudgetWarnOnce sync.Once

// resolveEffortOrErr resolves a reasoning_effort name or alias against the canonical spec table:
//   - empty or whitespace-only -> ("medium", 1024, false, nil)
//   - "none" -> ("none", 0, true, nil)
//   - "minimal" -> ("minimal", 0, true, nil)
//   - "low", "fast" -> ("low", 256, false, nil)
//   - "medium" -> ("medium", 1024, false, nil)
//   - "high", "careful", "xhigh" -> ("high", 4096, false, nil)
//   - any other name -> ErrArgs listing the allowed names.
func resolveEffortOrErr(effort string) (canonical string, budget int, maskDeliberate bool, err error) {
	switch strings.ToLower(strings.TrimSpace(effort)) {
	case "", "medium":
		return "medium", 1024, false, nil
	case "none":
		return "none", 0, true, nil
	case "minimal":
		return "minimal", 0, true, nil
	case "low", "fast":
		return "low", 256, false, nil
	case "high", "careful", "xhigh":
		return "high", 4096, false, nil
	default:
		return "", -1, false, fmt.Errorf(
			"ErrArgs: invalid reasoning_effort %q (allowed names: none, minimal, low, medium, high; aliases: fast, careful, xhigh)",
			effort,
		)
	}
}

func effortBudgetAndMask(effort string) (int, bool) {
	_, budget, mask, err := resolveEffortOrErr(effort)
	if err != nil {
		return -1, false
	}
	return budget, mask
}

// RenderedPrompt holds the tokenized prompt and context state for a single request.
type RenderedPrompt struct {
	InputIDs          []int
	ContextStore      *ContextStore
	RawMessagesTokens [][]int
	ParsedTools       []fncall.ParsedTool
	NameMapper        *fncall.NameMapper
}

// RenderChatPrompt renders OpenAI messages and optional tools into Reckoner token IDs,
// applying context anchoring (<|mN|>, <|Lk|>) when anchorContext is true.
// Emits canonical turn boundaries matching harness_bridge.py format_prompt token-for-token:
//   - table.Newline immediately after <|turn:user|> (before any <|mN|> anchor tokens)
//   - table.Newline immediately after <|turn:system|> unless <|tools|> follows
//   - table.Newline immediately after every <|/turn|> followed by another turn
//   - Nothing between <|turn:agent|> and the gate token
func RenderChatPrompt(
	tok *Tokenizer,
	messages []OpenAIMessage,
	toolsJSON json.RawMessage,
	anchorContext bool,
) (*RenderedPrompt, error) {
	return RenderChatPromptWithTable(tok, tok.TokenTable(), messages, toolsJSON, anchorContext)
}

// RenderChatPromptWithTable renders OpenAI messages and optional tools using an explicit TokenTable.
func RenderChatPromptWithTable(
	tok *Tokenizer,
	table TokenTable,
	messages []OpenAIMessage,
	toolsJSON json.RawMessage,
	anchorContext bool,
) (*RenderedPrompt, error) {
	ctxStore := NewContextStore(tok)
	ctxStore.SetAnchorsEnabled(anchorContext)
	var parsedTools []fncall.ParsedTool
	var mapper *fncall.NameMapper
	var toolsBlock string

	if len(toolsJSON) > 0 && string(bytes.TrimSpace(toolsJSON)) != "null" && string(bytes.TrimSpace(toolsJSON)) != "[]" {
		var err error
		toolsBlock, mapper, parsedTools, err = fncall.Render(toolsJSON)
		if err != nil {
			return nil, fmt.Errorf("ErrArgs: invalid tools schema: %w", err)
		}
		if toolsBlock != "" && !strings.HasPrefix(toolsBlock, "<|tools|>") {
			toolsBlock = tok.escapeUserSystemTokens(toolsBlock)
			toolsBlock = "<|tools|>\n" + toolsBlock + "\n<|/tools|>"
		}
	}

	var inputIDs []int
	var rawMsgsTokens [][]int
	systemRendered := false

	appendTurnStart := func(turnTok int) {
		if len(inputIDs) > 0 && inputIDs[len(inputIDs)-1] == table.EndTurn {
			inputIDs = append(inputIDs, table.Newline)
		}
		inputIDs = append(inputIDs, turnTok)
	}

	for i, msg := range messages {
		role := strings.ToLower(strings.TrimSpace(msg.Role))
		content := msg.Content

		if role == "system" {
			var sysBody string
			prefix := ""
			rest := content
			if msg.harnessEffortPrefix && !strings.HasPrefix(rest, "<|tools|>") && strings.HasPrefix(rest, "reasoning_effort:") {
				if nl := strings.IndexByte(rest, '\n'); nl >= 0 && strings.HasPrefix(rest[nl+1:], "<|tools|>") {
					prefix = tok.escapeUserSystemTokens(rest[:nl+1])
					rest = rest[nl+1:]
				}
			}
			if strings.HasPrefix(rest, "<|tools|>") {
				if endIdx := strings.Index(rest, "<|/tools|>"); endIdx >= 0 {
					toolsPart := rest[len("<|tools|>"):endIdx]
					restPart := rest[endIdx+len("<|/tools|>"):]
					sysBody = prefix + "<|tools|>" + tok.escapeUserSystemTokens(toolsPart) + "<|/tools|>" + tok.escapeUserSystemTokens(restPart)
				} else {
					sysBody = prefix + "<|tools|>" + tok.escapeUserSystemTokens(rest[len("<|tools|>"):])
				}
			} else {
				sysBody = tok.escapeUserSystemTokens(content)
			}
			if toolsBlock != "" && !systemRendered {
				if sysBody != "" {
					sysBody = sysBody + "\n" + toolsBlock
				} else {
					sysBody = toolsBlock
				}
				systemRendered = true
			}
			appendTurnStart(table.TurnSystem)
			bodyTokens := tok.Encode(sysBody)
			// Emit table.Newline after <|turn:system|> unless <|tools|> is the very first token of the system body
			// (when a leading "reasoning_effort: ...\n" line precedes <|tools|>, bodyTokens[0] != table.ToolsOpen,
			// so table.Newline is emitted before "reasoning_effort:" and <|tools|> follows the effort line's "\n").
			if len(bodyTokens) == 0 || bodyTokens[0] != table.ToolsOpen {
				inputIDs = append(inputIDs, table.Newline)
			}
			if len(bodyTokens) > 0 {
				rawMsgsTokens = append(rawMsgsTokens, bodyTokens)
				stored, _, _, err := ctxStore.StoreMessage(bodyTokens, nil, nil)
				if err != nil {
					return nil, err
				}
				inputIDs = append(inputIDs, stored...)
			}
			inputIDs = append(inputIDs, table.EndTurn)
			continue
		}

		if i == 0 && toolsBlock != "" && !systemRendered {
			appendTurnStart(table.TurnSystem)
			toolsTokens := tok.Encode(toolsBlock)
			if len(toolsTokens) == 0 || toolsTokens[0] != table.ToolsOpen {
				inputIDs = append(inputIDs, table.Newline)
			}
			inputIDs = append(inputIDs, toolsTokens...)
			inputIDs = append(inputIDs, table.EndTurn)
			systemRendered = true
		}

		switch role {
		case "user":
			appendTurnStart(table.TurnUser)
			inputIDs = append(inputIDs, table.Newline)
			bodyTokens := tok.Encode(tok.escapeUserSystemTokens(content))
			if len(bodyTokens) > 0 {
				rawMsgsTokens = append(rawMsgsTokens, bodyTokens)
				stored, _, _, err := ctxStore.StoreMessage(bodyTokens, nil, nil)
				if err != nil {
					return nil, err
				}
				inputIDs = append(inputIDs, stored...)
			}
			inputIDs = append(inputIDs, table.EndTurn)

		case "assistant", "agent":
			appendTurnStart(table.TurnAgent)
			if msg.ReasoningContent != "" {
				inputIDs = append(inputIDs, table.Deliberate)
				inputIDs = append(inputIDs, tok.Encode(tok.escapeUserSystemTokens(msg.ReasoningContent))...)
			}
			if len(msg.ToolCalls) > 0 {
				inputIDs = append(inputIDs, table.Act)
				for _, tc := range msg.ToolCalls {
					if tok != nil && tok.UsesExternalTokenizer {
						resolvedName := ""
						for i := range parsedTools {
							if parsedTools[i].OriginalName == tc.Function.Name || parsedTools[i].SanitizedName == tc.Function.Name {
								resolvedName = parsedTools[i].SanitizedName
								break
							}
						}
						if resolvedName == "" {
							if !fncall.IsValidIdentifier(tc.Function.Name) {
								return nil, fmt.Errorf("ErrArgs: invalid tool call function name %q (must match a declared tool or be a valid identifier)", tc.Function.Name)
							}
							resolvedName = tc.Function.Name
						}
						var argsMap map[string]interface{}
						unmarshalErr := json.Unmarshal([]byte(tc.Function.Arguments), &argsMap)
						var callStr string
						if strings.TrimSpace(tc.Function.Arguments) == "" || (unmarshalErr == nil && argsMap != nil) {
							var serErr error
							callStr, serErr = fncall.SerializeStrict(tc.Function.Name, argsMap, parsedTools, tok.escapeUserSystemTokens)
							if serErr != nil {
								return nil, serErr
							}
						} else {
							callStr = fmt.Sprintf("%s(%s)", resolvedName, tok.escapeUserSystemTokens(tc.Function.Arguments))
						}
						inputIDs = append(inputIDs, table.CallExternal)
						inputIDs = append(inputIDs, tok.Encode(callStr)...)
						inputIDs = append(inputIDs, table.EndCall)
						continue
					}
					var argsMap map[string]interface{}
					_ = json.Unmarshal([]byte(tc.Function.Arguments), &argsMap)
					callStr, err := fncall.SerializeWithEscape(tc.Function.Name, argsMap, parsedTools, tok.escapeUserSystemTokens)
					if err != nil {
						callStr = fmt.Sprintf("%s(%s)", tok.escapeUserSystemTokens(tc.Function.Name), tok.escapeUserSystemTokens(tc.Function.Arguments))
					}
					inputIDs = append(inputIDs, table.CallExternal)
					inputIDs = append(inputIDs, tok.Encode(callStr)...)
					inputIDs = append(inputIDs, table.EndCall)
				}
			} else if strings.HasPrefix(content, "<|ready|>") ||
				strings.HasPrefix(content, "<|act|>") ||
				strings.HasPrefix(content, "<|deliberate|>") ||
				strings.HasPrefix(content, "<|clarify|>") {
				preIDs, preErr := ValidateAndEncodePreformattedAssistantContent(tok, table, content)
				if preErr != nil {
					return nil, preErr
				}
				inputIDs = append(inputIDs, preIDs...)
			} else {
				inputIDs = append(inputIDs, table.Ready)
				inputIDs = append(inputIDs, tok.Encode(tok.escapeUserSystemTokens(content))...)
			}
			inputIDs = append(inputIDs, table.EndTurn)

		case "tool":
			inputIDs = append(inputIDs, table.Obs)
			bodyTokens := tok.Encode(tok.escapeUserSystemTokens(content))
			if len(bodyTokens) > 0 {
				rawMsgsTokens = append(rawMsgsTokens, bodyTokens)
				stored, _, _, err := ctxStore.StoreMessage(bodyTokens, nil, nil)
				if err != nil {
					return nil, err
				}
				inputIDs = append(inputIDs, stored...)
			}
			inputIDs = append(inputIDs, table.EndObs)
		}
	}

	if toolsBlock != "" && !systemRendered {
		sysPrefix := []int{table.TurnSystem}
		toolsTokens := tok.Encode(toolsBlock)
		if len(toolsTokens) == 0 || toolsTokens[0] != table.ToolsOpen {
			sysPrefix = append(sysPrefix, table.Newline)
		}
		sysPrefix = append(sysPrefix, toolsTokens...)
		sysPrefix = append(sysPrefix, table.EndTurn)
		if len(inputIDs) > 0 {
			sysPrefix = append(sysPrefix, table.Newline)
		}
		inputIDs = append(sysPrefix, inputIDs...)
	}

	// Open the assistant turn for generation
	appendTurnStart(table.TurnAgent)

	return &RenderedPrompt{
		InputIDs:          inputIDs,
		ContextStore:      ctxStore,
		RawMessagesTokens: rawMsgsTokens,
		ParsedTools:       parsedTools,
		NameMapper:        mapper,
	}, nil
}

// ParsedAgentTurn holds the structured post-processed fields from an agent's output token IDs.
type ParsedAgentTurn struct {
	ActionMode       string
	Content          string
	ReasoningContent string
	ExternalCalls    []OpenAIToolCall
	InternalCalls    []ToolCallTelemetry
	ThinkingTokens   int
	FinishReason     string
	FormatErrors     []string
}

// ParseAgentOutputTokens post-processes generated token IDs into clean OpenAI response fields:
// - Extracts <|deliberate|> thought trace into ReasoningContent
// - Expands <|quote|>...<|/quote|> citations into byte-exact text from ctxStore
// - Strips <|mN|> and <|Lk|> anchor tokens
// - Parses <|call|>...<|/call|> external tool calls via fncall.Parse
func ParseAgentOutputTokens(
	tok *Tokenizer,
	outputIDs []int,
	ctxStore *ContextStore,
	parsedTools []fncall.ParsedTool,
	mapper *fncall.NameMapper,
) ParsedAgentTurn {
	return ParseAgentOutputTokensWithTable(tok, tok.TokenTable(), outputIDs, ctxStore, parsedTools, mapper)
}

// ParseAgentOutputTokensWithTable post-processes generated token IDs using an explicit TokenTable.
func ParseAgentOutputTokensWithTable(
	tok *Tokenizer,
	table TokenTable,
	outputIDs []int,
	ctxStore *ContextStore,
	parsedTools []fncall.ParsedTool,
	mapper *fncall.NameMapper,
) ParsedAgentTurn {
	res := ParsedAgentTurn{
		ActionMode:   "READY",
		FinishReason: "stop",
	}

	callMap := table.CallTokenToTool
	if len(callMap) == 0 {
		callMap = callTokenToToolName
	}

	curMode := "ready"
	var readyTokens []int
	var deliberateTokens []int
	var contentParts []string

	flushReadyTokens := func() {
		if len(readyTokens) > 0 {
			stripped := tok.StripAnchors(readyTokens)
			contentParts = append(contentParts, tok.DecodeTokens(stripped))
			readyTokens = nil
		}
	}

	n := len(outputIDs)
	i := 0
	for i < n {
		t := outputIDs[i]

		if i == 0 {
			switch t {
			case table.Ready:
				res.ActionMode = "READY"
				curMode = "ready"
				i++
				continue
			case table.Act:
				res.ActionMode = "ACT"
				curMode = "act"
				i++
				continue
			case table.Deliberate:
				res.ActionMode = "DELIBERATE"
				curMode = "deliberate"
				i++
				continue
			case table.Clarify:
				res.ActionMode = "CLARIFY"
				curMode = "clarify"
				i++
				continue
			default:
				res.ActionMode = "DIRECT"
				curMode = "ready"
			}
		}

		if t == table.EndTurn || t == table.EndOfText {
			break
		}

		switch t {
		case table.Ready:
			curMode = "ready"
			i++
			continue
		case table.Act:
			curMode = "act"
			i++
			continue
		case table.Deliberate:
			curMode = "deliberate"
			i++
			continue
		case table.Clarify:
			curMode = "clarify"
			i++
			continue
		}

		// Handle <|quote|>...<|/quote|> followed by optional <|obs|>...<|/obs|>
		if table.Quote >= 0 && t == table.Quote {
			flushReadyTokens()
			j := i + 1
			for j < n && outputIDs[j] != table.EndQuote && outputIDs[j] != table.EndTurn {
				j++
			}
			quoteSpan := outputIDs[i+1 : j]
			if j < n && outputIDs[j] == table.EndQuote {
				j++
			}
			// Skip trailing receipt observation <|obs|>...<|/obs|> if present
			if j < n && (outputIDs[j] == table.Obs || outputIDs[j] == table.ObsErr) {
				j++
				for j < n && outputIDs[j] != table.EndObs && outputIDs[j] != table.EndTurn {
					j++
				}
				if j < n && outputIDs[j] == table.EndObs {
					j++
				}
			}
			// Expand quote from ctxStore
			if ctxStore != nil {
				mID := -1
				var lineIDs []int
				for _, qt := range quoteSpan {
					if qt >= M0TokenID && qt < M0TokenID+NumMessageTokens {
						mID = qt - M0TokenID
					} else if qt >= L0TokenID && qt < L0TokenID+NumLineAnchors {
						lineIDs = append(lineIDs, qt-L0TokenID)
					}
				}
				if mID >= 0 && len(lineIDs) > 0 {
					lo, hi := lineIDs[0], lineIDs[0]
					for _, lid := range lineIDs[1:] {
						if lid < lo {
							lo = lid
						}
						if lid > hi {
							hi = lid
						}
					}
					qPayload := fmt.Sprintf("m%d, L%d, L%d", mID, lo, hi)
					if qRes, err := ctxStore.Quote(qPayload); err == nil && qRes.Success {
						contentParts = append(contentParts, qRes.Text)
					}
				}
			}
			i = j
			continue
		}

		// Handle tool call <|call:...| >...<|/call|> followed by optional <|obs|>...<|/obs|>
		if toolName, isCall := callMap[t]; isCall {
			flushReadyTokens()
			j := i + 1
			for j < n && outputIDs[j] != table.EndCall && outputIDs[j] != table.EndTurn {
				j++
			}
			callSpan := outputIDs[i+1 : j]
			callPayload := strings.TrimSpace(tok.DecodeTokens(tok.StripAnchors(callSpan)))
			if j < n && outputIDs[j] == table.EndCall {
				j++
			}

			var obsText string
			var isObsErr bool
			var obsCount int
			if j < n && (outputIDs[j] == table.Obs || outputIDs[j] == table.ObsErr) {
				isObsErr = (outputIDs[j] == table.ObsErr)
				obsStart := j
				j++
				obsInnerStart := j
				for j < n && outputIDs[j] != table.EndObs && outputIDs[j] != table.EndTurn {
					j++
				}
				obsInner := outputIDs[obsInnerStart:j]
				obsText = tok.DecodeTokens(tok.StripAnchors(obsInner))
				if j < n && outputIDs[j] == table.EndObs {
					j++
				}
				obsCount = j - obsStart
			}

			if toolName == "external" && len(parsedTools) > 0 {
				origName, argsMap, err := fncall.Parse(callPayload, parsedTools, mapper)
				if err != nil {
					res.FormatErrors = append(res.FormatErrors, err.Error())
				} else {
					argsBytes, _ := json.Marshal(argsMap)
					idx := len(res.ExternalCalls)
					res.ExternalCalls = append(res.ExternalCalls, OpenAIToolCall{
						Index: &idx,
						ID:    fmt.Sprintf("call_%d", idx),
						Type:  "function",
						Function: OpenAIFunctionCall{
							Name:      origName,
							Arguments: string(argsBytes),
						},
					})
					res.FinishReason = "tool_calls"
				}
			} else {
				res.InternalCalls = append(res.InternalCalls, ToolCallTelemetry{
					Tool:      toolName,
					Payload:   callPayload,
					Result:    obsText,
					IsError:   isObsErr,
					ObsTokens: obsCount,
				})
			}
			i = j
			continue
		}

		// Skip standalone <|obs|>...<|/obs|> if any
		if t == table.Obs || t == table.ObsErr {
			flushReadyTokens()
			j := i + 1
			for j < n && outputIDs[j] != table.EndObs && outputIDs[j] != table.EndTurn {
				j++
			}
			if j < n && outputIDs[j] == table.EndObs {
				j++
			}
			i = j
			continue
		}

		if curMode == "deliberate" {
			res.ThinkingTokens++
			deliberateTokens = append(deliberateTokens, t)
		} else {
			readyTokens = append(readyTokens, t)
		}
		i++
	}

	flushReadyTokens()
	if len(deliberateTokens) > 0 {
		res.ReasoningContent = strings.TrimSpace(tok.DecodeTokens(tok.StripAnchors(deliberateTokens)))
	}
	res.Content = strings.TrimSpace(strings.Join(contentParts, ""))
	return res
}

// sglangTokenRequest represents a token-level generation request to SGLang.
type sglangTokenRequest struct {
	RequestID            string                 `json:"rid,omitempty"`
	InputIDs             []int                  `json:"input_ids"`
	SamplingParams       map[string]interface{} `json:"sampling_params"`
	CustomLogitProcessor string                 `json:"custom_logit_processor,omitempty"`
	Stream               bool                   `json:"stream"`
}

// sglangTokenChunk represents a streaming or final token-level response from SGLang.
type sglangTokenChunk struct {
	DeltaTokenIDs    []int
	FullOutputIDs    []int
	PromptTokens     int
	CompletionTokens int
	CachedTokens     int
	FinishReason     string
	MatchedStopToken int
	Done             bool
}

// SGLangEngineClient drives SGLang strictly through token-level interfaces
// (either HTTP/UDS `/generate` with `input_ids` or gRPC `SglangScheduler/Generate`).
type SGLangEngineClient struct {
	addr       string
	proto      string // "http" or "grpc"
	httpClient *http.Client
	grpcClient *http.Client
}

// NewSGLangEngineClient creates a token-level SGLang client.
func NewSGLangEngineClient(addr, proto string) *SGLangEngineClient {
	addr = strings.TrimSpace(addr)
	if proto == "" || proto == "auto" {
		if strings.HasPrefix(addr, "grpc://") {
			proto = "grpc"
		} else {
			proto = "http"
		}
	}

	var httpTransport *http.Transport
	var baseURL string
	if strings.HasPrefix(addr, "unix://") {
		sockPath := strings.TrimPrefix(addr, "unix://")
		httpTransport = &http.Transport{
			DialContext: func(ctx context.Context, _, _ string) (net.Conn, error) {
				var d net.Dialer
				return d.DialContext(ctx, "unix", sockPath)
			},
			MaxIdleConns:        256,
			MaxIdleConnsPerHost: 256,
		}
		baseURL = "http://localhost"
	} else {
		cleanHost := strings.TrimPrefix(strings.TrimPrefix(addr, "http://"), "grpc://")
		cleanHost = strings.TrimRight(cleanHost, "/")
		httpTransport = &http.Transport{
			MaxIdleConns:        256,
			MaxIdleConnsPerHost: 256,
		}
		baseURL = "http://" + cleanHost
	}

	var grpcProtocols http.Protocols
	grpcProtocols.SetUnencryptedHTTP2(true)
	grpcTransport := httpTransport.Clone()
	grpcTransport.Protocols = &grpcProtocols

	return &SGLangEngineClient{
		addr:       baseURL,
		proto:      proto,
		httpClient: &http.Client{Transport: httpTransport, Timeout: 300 * time.Second},
		grpcClient: &http.Client{Transport: grpcTransport, Timeout: 300 * time.Second},
	}
}

// GenerateStream sends a token-level request to SGLang and invokes onChunk for each token chunk.
func (c *SGLangEngineClient) GenerateStream(
	ctx context.Context,
	req sglangTokenRequest,
	onChunk func(chunk sglangTokenChunk) error,
) error {
	if c.proto == "grpc" {
		return c.generateStreamGRPC(ctx, req, onChunk)
	}
	return c.generateStreamHTTP(ctx, req, onChunk)
}

func (c *SGLangEngineClient) generateStreamHTTP(
	ctx context.Context,
	req sglangTokenRequest,
	onChunk func(chunk sglangTokenChunk) error,
) error {
	bodyBytes, err := json.Marshal(req)
	if err != nil {
		return err
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.addr+"/generate", bytes.NewReader(bodyBytes))
	if err != nil {
		return err
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return fmt.Errorf("SGLang /generate request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		errBody, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("SGLang /generate HTTP %d: %s", resp.StatusCode, string(errBody))
	}

	type rawGenerateOut struct {
		OutputIDs []int `json:"output_ids"`
		MetaInfo  struct {
			PromptTokens     int         `json:"prompt_tokens"`
			CompletionTokens int         `json:"completion_tokens"`
			CachedTokens     int         `json:"cached_tokens"`
			FinishReason     interface{} `json:"finish_reason"`
		} `json:"meta_info"`
	}

	parseFinish := func(fr interface{}) (string, int) {
		if fr == nil {
			return "", -1
		}
		switch v := fr.(type) {
		case string:
			return v, -1
		case map[string]interface{}:
			tStr, _ := v["type"].(string)
			matched := -1
			if mFloat, ok := v["matched"].(float64); ok {
				matched = int(mFloat)
			}
			return tStr, matched
		}
		return "", -1
	}

	if !req.Stream {
		var raw rawGenerateOut
		if err := json.NewDecoder(resp.Body).Decode(&raw); err != nil {
			return fmt.Errorf("failed to decode /generate JSON: %w", err)
		}
		fReason, matched := parseFinish(raw.MetaInfo.FinishReason)
		if fReason == "" {
			fReason = "stop"
		}
		return onChunk(sglangTokenChunk{
			DeltaTokenIDs:    raw.OutputIDs,
			FullOutputIDs:    raw.OutputIDs,
			PromptTokens:     raw.MetaInfo.PromptTokens,
			CompletionTokens: raw.MetaInfo.CompletionTokens,
			CachedTokens:     raw.MetaInfo.CachedTokens,
			FinishReason:     fReason,
			MatchedStopToken: matched,
			Done:             true,
		})
	}

	var accumulated []int
	reader := bufio.NewReaderSize(resp.Body, 64*1024)
	for {
		line, err := reader.ReadString('\n')
		if len(line) > 0 {
			line = strings.TrimSpace(line)
			if strings.HasPrefix(line, "data:") {
				dataStr := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
				if dataStr == "[DONE]" {
					break
				}
				var raw rawGenerateOut
				if jsonErr := json.Unmarshal([]byte(dataStr), &raw); jsonErr == nil {
					var delta []int
					if len(raw.OutputIDs) >= len(accumulated) && hasIntPrefix(raw.OutputIDs, accumulated) {
						delta = raw.OutputIDs[len(accumulated):]
						accumulated = append(accumulated[:0], raw.OutputIDs...)
					} else {
						delta = raw.OutputIDs
						accumulated = append(accumulated, delta...)
					}
					fReason, matched := parseFinish(raw.MetaInfo.FinishReason)
					done := (fReason != "")
					if callErr := onChunk(sglangTokenChunk{
						DeltaTokenIDs:    delta,
						FullOutputIDs:    accumulated,
						PromptTokens:     raw.MetaInfo.PromptTokens,
						CompletionTokens: len(accumulated),
						CachedTokens:     raw.MetaInfo.CachedTokens,
						FinishReason:     fReason,
						MatchedStopToken: matched,
						Done:             done,
					}); callErr != nil {
						return callErr
					}
				}
			}
		}
		if err != nil {
			if err == io.EOF {
				break
			}
			return err
		}
	}
	return nil
}

func hasIntPrefix(slice, prefix []int) bool {
	if len(slice) < len(prefix) {
		return false
	}
	for i := range prefix {
		if slice[i] != prefix[i] {
			return false
		}
	}
	return true
}

// ============================================================================
// Pure-Go Protobuf Wire Codec for sglang.grpc.scheduler.SglangScheduler/Generate
// ============================================================================

func appendVarintField(buf []byte, fieldNum int, val uint64) []byte {
	tag := uint64(fieldNum<<3) | 0
	buf = binary.AppendUvarint(buf, tag)
	buf = binary.AppendUvarint(buf, val)
	return buf
}

func appendFloat32Field(buf []byte, fieldNum int, val float32) []byte {
	tag := uint64(fieldNum<<3) | 5
	buf = binary.AppendUvarint(buf, tag)
	var b [4]byte
	binary.LittleEndian.PutUint32(b[:], math.Float32bits(val))
	return append(buf, b[:]...)
}

func appendBytesField(buf []byte, fieldNum int, data []byte) []byte {
	tag := uint64(fieldNum<<3) | 2
	buf = binary.AppendUvarint(buf, tag)
	buf = binary.AppendUvarint(buf, uint64(len(data)))
	return append(buf, data...)
}

func appendStringField(buf []byte, fieldNum int, s string) []byte {
	if s == "" {
		return buf
	}
	return appendBytesField(buf, fieldNum, []byte(s))
}

func encodeProtobufValue(v interface{}) []byte {
	var valBuf []byte
	switch tv := v.(type) {
	case nil:
		valBuf = appendVarintField(valBuf, 1, 0)
	case float64:
		tag := uint64(2<<3) | 1
		valBuf = binary.AppendUvarint(valBuf, tag)
		var b [8]byte
		binary.LittleEndian.PutUint64(b[:], math.Float64bits(tv))
		valBuf = append(valBuf, b[:]...)
	case int:
		tag := uint64(2<<3) | 1
		valBuf = binary.AppendUvarint(valBuf, tag)
		var b [8]byte
		binary.LittleEndian.PutUint64(b[:], math.Float64bits(float64(tv)))
		valBuf = append(valBuf, b[:]...)
	case string:
		valBuf = appendBytesField(valBuf, 3, []byte(tv))
	case bool:
		bVal := uint64(0)
		if tv {
			bVal = 1
		}
		valBuf = appendVarintField(valBuf, 4, bVal)
	case []int:
		var listBuf []byte
		for _, elem := range tv {
			listBuf = appendBytesField(listBuf, 1, encodeProtobufValue(elem))
		}
		valBuf = appendBytesField(valBuf, 6, listBuf)
	case [][]int:
		var listBuf []byte
		for _, elem := range tv {
			listBuf = appendBytesField(listBuf, 1, encodeProtobufValue(elem))
		}
		valBuf = appendBytesField(valBuf, 6, listBuf)
	}
	return valBuf
}

func encodeProtobufStruct(m map[string]interface{}) []byte {
	var structBuf []byte
	for k, v := range m {
		if v == nil {
			continue
		}
		var entryBuf []byte
		entryBuf = appendStringField(entryBuf, 1, k)
		entryBuf = appendBytesField(entryBuf, 2, encodeProtobufValue(v))
		structBuf = appendBytesField(structBuf, 1, entryBuf)
	}
	return structBuf
}

func encodeSGLangGenerateRequest(req sglangTokenRequest) []byte {
	var msg []byte
	reqID := req.RequestID
	if reqID == "" {
		reqID = nextSessionID()
	}
	msg = appendStringField(msg, 1, reqID)

	// Field 2: TokenizedInput { string original_text = 1; repeated uint32 input_ids = 2; }
	var packedIDs []byte
	for _, id := range req.InputIDs {
		packedIDs = binary.AppendUvarint(packedIDs, uint64(id))
	}
	var tokInput []byte
	tokInput = appendBytesField(tokInput, 2, packedIDs)
	msg = appendBytesField(msg, 2, tokInput)

	// Field 4: SamplingParams
	var spBuf []byte
	temp := float32(0.0)
	if t, ok := req.SamplingParams["temperature"].(float64); ok {
		temp = float32(t)
	}
	topP := float32(1.0)
	if tp, ok := req.SamplingParams["top_p"].(float64); ok {
		topP = float32(tp)
	}
	spBuf = appendFloat32Field(spBuf, 1, temp)
	spBuf = appendFloat32Field(spBuf, 2, topP)
	spBuf = appendVarintField(spBuf, 3, ^uint64(0)) // top_k = -1 (10-byte two's complement varint)
	spBuf = appendFloat32Field(spBuf, 7, 1.0)       // repetition_penalty = 1.0

	maxNew := 512
	if mn, ok := req.SamplingParams["max_new_tokens"].(int); ok {
		maxNew = mn
	}
	spBuf = appendVarintField(spBuf, 8, uint64(maxNew))

	if stopIDs, ok := req.SamplingParams["stop_token_ids"].([]int); ok && len(stopIDs) > 0 {
		var packedStops []byte
		for _, sid := range stopIDs {
			packedStops = binary.AppendUvarint(packedStops, uint64(sid))
		}
		spBuf = appendBytesField(spBuf, 10, packedStops)
	}
	spBuf = appendVarintField(spBuf, 11, 0) // skip_special_tokens = false
	spBuf = appendVarintField(spBuf, 17, 1) // n = 1
	if noTrim, ok := req.SamplingParams["no_stop_trim"].(bool); ok && noTrim {
		spBuf = appendVarintField(spBuf, 20, 1)
	}
	if cp, ok := req.SamplingParams["custom_params"].(map[string]interface{}); ok && len(cp) > 0 {
		spBuf = appendBytesField(spBuf, 23, encodeProtobufStruct(cp))
	}
	msg = appendBytesField(msg, 4, spBuf)

	if req.CustomLogitProcessor != "" {
		msg = appendStringField(msg, 11, req.CustomLogitProcessor)
	}
	if req.Stream {
		msg = appendVarintField(msg, 17, 1)
	}
	return msg
}

func decodeRepeatedUint32(valBytes []byte) []int {
	var out []int
	for len(valBytes) > 0 {
		v, n := binary.Uvarint(valBytes)
		if n <= 0 {
			break
		}
		out = append(out, int(v))
		valBytes = valBytes[n:]
	}
	return out
}

func decodeSGLangGenerateResponse(data []byte) (sglangTokenChunk, error) {
	var chunk sglangTokenChunk
	chunk.MatchedStopToken = -1

	for len(data) > 0 {
		tag, n := binary.Uvarint(data)
		if n <= 0 {
			break
		}
		data = data[n:]
		fieldNum := int(tag >> 3)
		wireType := int(tag & 7)

		switch wireType {
		case 0:
			_, vn := binary.Uvarint(data)
			if vn <= 0 {
				return chunk, fmt.Errorf("invalid varint in GenerateResponse")
			}
			data = data[vn:]
		case 2:
			bLen, ln := binary.Uvarint(data)
			if ln <= 0 || int(bLen) > len(data[ln:]) {
				return chunk, fmt.Errorf("invalid bytes field in GenerateResponse")
			}
			sub := data[ln : ln+int(bLen)]
			data = data[ln+int(bLen):]

			if fieldNum == 2 {
				// GenerateStreamChunk
				for len(sub) > 0 {
					stag, sn := binary.Uvarint(sub)
					if sn <= 0 {
						break
					}
					sub = sub[sn:]
					sField := int(stag >> 3)
					sWire := int(stag & 7)
					if sWire == 0 {
						sv, svn := binary.Uvarint(sub)
						if svn <= 0 {
							break
						}
						sub = sub[svn:]
						switch sField {
						case 1:
							chunk.DeltaTokenIDs = append(chunk.DeltaTokenIDs, int(sv))
						case 2:
							chunk.PromptTokens = int(sv)
						case 3:
							chunk.CompletionTokens = int(sv)
						case 4:
							chunk.CachedTokens = int(sv)
						}
					} else if sWire == 2 {
						sLen, sln := binary.Uvarint(sub)
						if sln <= 0 || int(sLen) > len(sub[sln:]) {
							break
						}
						sBytes := sub[sln : sln+int(sLen)]
						sub = sub[sln+int(sLen):]
						if sField == 1 {
							chunk.DeltaTokenIDs = append(chunk.DeltaTokenIDs, decodeRepeatedUint32(sBytes)...)
						}
					} else if sWire == 5 {
						if len(sub) < 4 {
							break
						}
						sub = sub[4:]
					} else if sWire == 1 {
						if len(sub) < 8 {
							break
						}
						sub = sub[8:]
					}
				}
			} else if fieldNum == 3 {
				// GenerateComplete
				chunk.Done = true
				for len(sub) > 0 {
					stag, sn := binary.Uvarint(sub)
					if sn <= 0 {
						break
					}
					sub = sub[sn:]
					sField := int(stag >> 3)
					sWire := int(stag & 7)
					if sWire == 0 {
						sv, svn := binary.Uvarint(sub)
						if svn <= 0 {
							break
						}
						sub = sub[svn:]
						switch sField {
						case 1:
							chunk.FullOutputIDs = append(chunk.FullOutputIDs, int(sv))
						case 3:
							chunk.PromptTokens = int(sv)
						case 4:
							chunk.CompletionTokens = int(sv)
						case 5:
							chunk.CachedTokens = int(sv)
						case 8:
							chunk.MatchedStopToken = int(sv)
						}
					} else if sWire == 2 {
						sLen, sln := binary.Uvarint(sub)
						if sln <= 0 || int(sLen) > len(sub[sln:]) {
							break
						}
						sBytes := sub[sln : sln+int(sLen)]
						sub = sub[sln+int(sLen):]
						switch sField {
						case 1:
							chunk.FullOutputIDs = append(chunk.FullOutputIDs, decodeRepeatedUint32(sBytes)...)
						case 2:
							chunk.FinishReason = string(sBytes)
						}
					} else if sWire == 5 {
						if len(sub) < 4 {
							break
						}
						sub = sub[4:]
					} else if sWire == 1 {
						if len(sub) < 8 {
							break
						}
						sub = sub[8:]
					}
				}
			}
		case 5:
			if len(data) < 4 {
				break
			}
			data = data[4:]
		case 1:
			if len(data) < 8 {
				break
			}
			data = data[8:]
		}
	}
	return chunk, nil
}

func (c *SGLangEngineClient) generateStreamGRPC(
	ctx context.Context,
	req sglangTokenRequest,
	onChunk func(chunk sglangTokenChunk) error,
) error {
	protoPayload := encodeSGLangGenerateRequest(req)
	framed := make([]byte, 5+len(protoPayload))
	framed[0] = 0
	binary.BigEndian.PutUint32(framed[1:5], uint32(len(protoPayload)))
	copy(framed[5:], protoPayload)

	url := c.addr + "/sglang.grpc.scheduler.SglangScheduler/Generate"
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(framed))
	if err != nil {
		return err
	}
	httpReq.Header.Set("Content-Type", "application/grpc")
	httpReq.Header.Set("Te", "trailers")

	resp, err := c.grpcClient.Do(httpReq)
	if err != nil {
		return fmt.Errorf("SGLang gRPC Generate failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("SGLang gRPC HTTP status %d", resp.StatusCode)
	}

	var accumulated []int
	var header [5]byte
	for {
		_, err := io.ReadFull(resp.Body, header[:])
		if err != nil {
			if err == io.EOF {
				break
			}
			return err
		}
		msgLen := binary.BigEndian.Uint32(header[1:5])
		msgBuf := make([]byte, msgLen)
		if _, err := io.ReadFull(resp.Body, msgBuf); err != nil {
			return err
		}
		chunk, err := decodeSGLangGenerateResponse(msgBuf)
		if err != nil {
			return err
		}
		if chunk.Done {
			if len(chunk.FullOutputIDs) >= len(accumulated) {
				chunk.DeltaTokenIDs = chunk.FullOutputIDs[len(accumulated):]
				accumulated = chunk.FullOutputIDs
			} else {
				chunk.FullOutputIDs = accumulated
			}
		} else {
			accumulated = append(accumulated, chunk.DeltaTokenIDs...)
			chunk.FullOutputIDs = accumulated
		}
		if err := onChunk(chunk); err != nil {
			return err
		}
	}

	if grpcStatus := resp.Trailer.Get("Grpc-Status"); grpcStatus != "" && grpcStatus != "0" {
		return fmt.Errorf("gRPC status %s: %s", grpcStatus, resp.Trailer.Get("Grpc-Message"))
	}
	return nil
}

// ============================================================================
// Front-Door Execution Engine (Native In-Path & Stop-and-Continue Modes)
// ============================================================================

type FrontDoorServer struct {
	cfg    ServeConfig
	tok    *Tokenizer
	table  TokenTable
	engine *SGLangEngineClient
}

func NewFrontDoorServer(cfg ServeConfig) (*FrontDoorServer, error) {
	var tok *Tokenizer
	var tbl TokenTable
	var err error
	if cfg.Tokenizer != nil {
		tbl, err = BuildProfileTokenTable(cfg.Profile)
		if err != nil {
			return nil, err
		}
		if err := ValidateTokenizerWithProfile(cfg.Tokenizer, cfg.Profile, tbl); err != nil {
			return nil, err
		}
		tokCopy := *cfg.Tokenizer
		tblCopy := tbl
		tokCopy.Table = &tblCopy
		tokCopy.UsesExternalTokenizer = profileUsesExternalTokenizer(cfg.Profile)
		tok = &tokCopy
	} else {
		tok, tbl, err = LoadTokenizerForProfile(cfg.Profile, cfg.TokenizerPath, cfg.ModelDir)
		if err != nil {
			return nil, err
		}
	}
	if cfg.Port <= 0 {
		cfg.Port = 8753
	}
	if cfg.Host == "" {
		cfg.Host = "127.0.0.1"
	}
	if cfg.SGLangAddr == "" {
		cfg.SGLangAddr = "127.0.0.1:30000"
	}
	if cfg.DefaultMode == "" {
		cfg.DefaultMode = "native"
	}
	if cfg.MaxToolCalls <= 0 {
		cfg.MaxToolCalls = 16
	}
	return &FrontDoorServer{
		cfg:    cfg,
		tok:    tok,
		table:  tbl,
		engine: NewSGLangEngineClient(cfg.SGLangAddr, cfg.SGLangProto),
	}, nil
}

func readAndCleanSessionShmJSON(sessionID string) *struct {
	ActionMode         string              `json:"action_mode"`
	ThinkingTokens     int                 `json:"thinking_tokens"`
	ToolCalls          []ToolCallTelemetry `json:"tool_calls"`
	FormatErrors       []string            `json:"format_errors"`
	TotalToolLatencyUs float64             `json:"total_tool_latency_us"`
} {
	var safe strings.Builder
	for _, r := range sessionID {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '-' || r == '_' || r == '.' {
			safe.WriteRune(r)
		} else {
			safe.WriteByte('_')
		}
	}
	jsonPath := fmt.Sprintf("/dev/shm/reckoner_sgl_sess_%s.json", safe.String())
	ptPath := fmt.Sprintf("/dev/shm/reckoner_sgl_sess_%s.pt", safe.String())
	defer func() {
		_ = os.Remove(jsonPath)
		_ = os.Remove(ptPath)
	}()

	data, err := os.ReadFile(jsonPath)
	if err != nil {
		return nil
	}
	var parsed struct {
		ActionMode         string              `json:"action_mode"`
		ThinkingTokens     int                 `json:"thinking_tokens"`
		ToolCalls          []ToolCallTelemetry `json:"tool_calls"`
		FormatErrors       []string            `json:"format_errors"`
		TotalToolLatencyUs float64             `json:"total_tool_latency_us"`
	}
	if json.Unmarshal(data, &parsed) != nil {
		return nil
	}
	return &parsed
}

// executeNativeMode drives SGLang with in-path ReckonerToolLogitProcessor.
func (s *FrontDoorServer) executeNativeMode(
	ctx context.Context,
	req OpenAIChatCompletionRequest,
	reqTok *Tokenizer,
	reqTable TokenTable,
	rendered *RenderedPrompt,
	onStreamDelta func(delta OpenAIStreamDelta) error,
) ([]int, *ReckonerTelemetry, error) {
	t0 := time.Now()
	sessionID := nextSessionID()
	temp := 0.0
	if req.Temperature != nil {
		temp = *req.Temperature
	}
	topP := 1.0
	if req.TopP != nil {
		topP = *req.TopP
	}
	maxNew := 512
	if req.MaxTokens != nil && *req.MaxTokens > 0 {
		maxNew = *req.MaxTokens
	} else if req.MaxCompletion != nil && *req.MaxCompletion > 0 {
		maxNew = *req.MaxCompletion
	}

	canonEffort, budget, maskDelib, err := resolveEffortOrErr(req.ReasoningEffort)
	if err != nil {
		return nil, nil, err
	}
	toolMode := req.ToolMode
	if toolMode == "" {
		toolMode = "all"
	}

	customParams := map[string]interface{}{
		"session_id":               sessionID,
		"reckoner_in_engine_tools": true,
		"effort":                   canonEffort,
		"thinking_budget":          budget,
		"gate_mask_deliberate":     maskDelib,
		"tool_mode":                toolMode,
		"max_tool_calls":           s.cfg.MaxToolCalls,
		"allow_external_tools":     len(rendered.ParsedTools) > 0,
	}
	if req.ForceSystem != "" {
		customParams["force_system"] = req.ForceSystem
	}
	if len(rendered.RawMessagesTokens) > 0 {
		customParams["context_messages_tokens"] = rendered.RawMessagesTokens
	}

	samplingParams := map[string]interface{}{
		"temperature":         temp,
		"top_p":               topP,
		"max_new_tokens":      maxNew,
		"stop_token_ids":      []int{reqTable.EndTurn, reqTable.EndOfText},
		"no_stop_trim":        true,
		"skip_special_tokens": false,
		"custom_params":       customParams,
	}

	sglReq := sglangTokenRequest{
		RequestID:            sessionID,
		InputIDs:             rendered.InputIDs,
		SamplingParams:       samplingParams,
		CustomLogitProcessor: resolveProcessorStr(s.cfg.ProcessorStr),
		Stream:               true,
	}

	var outputIDs []int
	var ttftMs float64
	var cachedTokens int
	var promptTokens = len(rendered.InputIDs)
	firstTokenSeen := false

	callMap := reqTable.CallTokenToTool
	if len(callMap) == 0 {
		callMap = callTokenToToolName
	}

	// Stream state for live SSE emission
	streamCurMode := "ready"
	streamInCall := false
	streamInObs := false
	streamInQuote := false

	// Fallback client-side tool call timing tracker when SGLang runs on a remote host
	var streamCallEnter time.Time
	var streamObsEnter time.Time
	var streamToolLatencies []struct {
		pauseResumeMs float64
		obsPrefillMs  float64
		totalMs       float64
	}
	awaitingNextModelTok := false

	err = s.engine.GenerateStream(ctx, sglReq, func(chunk sglangTokenChunk) error {
		now := time.Now()
		if !firstTokenSeen && len(chunk.DeltaTokenIDs) > 0 {
			firstTokenSeen = true
			ttftMs = float64(now.Sub(t0).Microseconds()) / 1000.0
		}
		if chunk.PromptTokens > 0 {
			promptTokens = chunk.PromptTokens
		}
		if chunk.CachedTokens > 0 {
			cachedTokens = chunk.CachedTokens
		}

		for _, tokID := range chunk.DeltaTokenIDs {
			if awaitingNextModelTok && tokID != reqTable.Obs && tokID != reqTable.ObsErr && tokID != reqTable.EndObs {
				obsMs := float64(now.Sub(streamObsEnter).Microseconds()) / 1000.0
				totMs := float64(now.Sub(streamCallEnter).Microseconds()) / 1000.0
				prMs := math.Max(0.0, totMs-obsMs)
				streamToolLatencies = append(streamToolLatencies, struct {
					pauseResumeMs float64
					obsPrefillMs  float64
					totalMs       float64
				}{pauseResumeMs: prMs, obsPrefillMs: obsMs, totalMs: totMs})
				awaitingNextModelTok = false
			}

			if tokID == reqTable.EndCall || (reqTable.EndQuote >= 0 && tokID == reqTable.EndQuote) {
				streamCallEnter = now
			}
			if (tokID == reqTable.Obs || tokID == reqTable.ObsErr) && !streamInObs {
				streamObsEnter = now
				awaitingNextModelTok = true
			}

			outputIDs = append(outputIDs, tokID)

			if onStreamDelta != nil {
				if tokID == reqTable.EndTurn || tokID == reqTable.EndOfText {
					continue
				}
				switch tokID {
				case reqTable.Ready:
					streamCurMode = "ready"
					continue
				case reqTable.Act:
					streamCurMode = "act"
					continue
				case reqTable.Deliberate:
					streamCurMode = "deliberate"
					continue
				case reqTable.Clarify:
					streamCurMode = "clarify"
					continue
				}
				if _, isCall := callMap[tokID]; isCall {
					streamInCall = true
					continue
				}
				if tokID == reqTable.EndCall {
					streamInCall = false
					continue
				}
				if tokID == reqTable.Obs || tokID == reqTable.ObsErr {
					streamInObs = true
					continue
				}
				if tokID == reqTable.EndObs {
					streamInObs = false
					continue
				}
				if reqTable.Quote >= 0 && tokID == reqTable.Quote {
					streamInQuote = true
					continue
				}
				if reqTable.EndQuote >= 0 && tokID == reqTable.EndQuote {
					streamInQuote = false
					continue
				}
				if streamInCall || streamInObs || streamInQuote || reqTok.IsAnchorToken(tokID) {
					continue
				}
				piece := reqTok.DecodeTokens([]int{tokID})
				if piece != "" {
					if streamCurMode == "deliberate" {
						if err := onStreamDelta(OpenAIStreamDelta{ReasoningContent: piece}); err != nil {
							return err
						}
					} else {
						if err := onStreamDelta(OpenAIStreamDelta{Content: piece}); err != nil {
							return err
						}
					}
				}
			}
		}
		return nil
	})
	if err != nil {
		return nil, nil, err
	}

	elapsedMs := float64(time.Since(t0).Microseconds()) / 1000.0
	decodeWindowMs := math.Max(0.1, elapsedMs-ttftMs)
	decodeTokS := 0.0
	if len(outputIDs) > 1 {
		decodeTokS = float64(len(outputIDs)-1) / (decodeWindowMs / 1000.0)
	} else if len(outputIDs) == 1 && elapsedMs > 0 {
		decodeTokS = 1000.0 / elapsedMs
	}

	parsed := ParseAgentOutputTokensWithTable(
		reqTok,
		reqTable,
		outputIDs,
		rendered.ContextStore,
		rendered.ParsedTools,
		rendered.NameMapper,
	)

	shmInfo := readAndCleanSessionShmJSON(sessionID)
	toolCalls := parsed.InternalCalls
	if shmInfo != nil && len(shmInfo.ToolCalls) > 0 {
		toolCalls = shmInfo.ToolCalls
	} else {
		for idx := range toolCalls {
			if idx < len(streamToolLatencies) {
				toolCalls[idx].PauseResumeMs = round4(streamToolLatencies[idx].pauseResumeMs)
				toolCalls[idx].ObsPrefillMs = round4(streamToolLatencies[idx].obsPrefillMs)
				toolCalls[idx].CallToNextTokenMs = round4(streamToolLatencies[idx].totalMs)
			}
		}
	}

	recomputed := promptTokens - cachedTokens
	if recomputed < 0 {
		recomputed = 0
	}

	telemetry := &ReckonerTelemetry{
		ServingMode:      "native",
		ActionMode:       parsed.ActionMode,
		TTFTMs:           round4(ttftMs),
		ElapsedMs:        round4(elapsedMs),
		DecodeTokPerSec:  round4(decodeTokS),
		PromptTokens:     promptTokens,
		CompletionTokens: len(outputIDs),
		ThinkingTokens:   parsed.ThinkingTokens,
		ToolCalls:        toolCalls,
		CachedTokens:     cachedTokens,
		RecomputedTokens: recomputed,
		FormatErrors:     parsed.FormatErrors,
	}
	return outputIDs, telemetry, nil
}

// executeInPathToolAndEncodeObservation dispatches a single in-path tool call (find, lines, calc, solve,
// str, date, lookup, lua, or external) and encodes its observation via the single EncodeToolObservation choke point.
func executeInPathToolAndEncodeObservation(
	reqTok *Tokenizer,
	reqTable TokenTable,
	ctxStore *ContextStore,
	reqRegStore *RegisterStore,
	toolName string,
	payloadStr string,
	tH0 time.Time,
) (resolvedToolName string, resultStr string, isErr bool, latUs float64, obsIDs []int) {
	resolvedToolName = toolName
	var rawFindTokens []int

	if toolName == "find" {
		isLines, q, scopeMsg, lStart, lEnd, parseErr := ParseFindOrLinesCall(payloadStr)
		if parseErr != nil {
			isErr = true
			resultStr = parseErr.Error()
		} else if isLines && scopeMsg != nil {
			if ctxStore == nil {
				isErr = true
				resultStr = "ErrUnsupported: context store is not initialized"
			} else {
				fRes, lErr := ctxStore.Lines(*scopeMsg, lStart, lEnd)
				if lErr != nil {
					isErr = true
					resultStr = lErr.Error()
				} else {
					resultStr = fRes.Result
					rawFindTokens = fRes.Tokens
				}
			}
			resolvedToolName = "lines"
		} else {
			if ctxStore == nil {
				isErr = true
				resultStr = "ErrUnsupported: context store is not initialized"
			} else {
				fRes, fErr := ctxStore.Find(q, scopeMsg)
				if fErr != nil {
					isErr = true
					resultStr = fErr.Error()
				} else {
					resultStr = fRes.Result
					rawFindTokens = fRes.Tokens
				}
			}
		}
		latUs = float64(time.Since(tH0).Nanoseconds()) / 1000.0
	} else {
		resp := DispatchToolWithStore(reqRegStore, toolName, payloadStr)
		latUs = float64(resp.ElapsedUs)
		if latUs <= 0 {
			latUs = float64(time.Since(tH0).Nanoseconds()) / 1000.0
		}
		if resp.Success {
			resultStr = resp.Result
		} else {
			isErr = true
			resultStr = resp.Error
		}
	}

	obsIDs = EncodeToolObservation(reqTok, reqTable, resolvedToolName, resultStr, rawFindTokens, isErr)
	return resolvedToolName, resultStr, isErr, latUs, obsIDs
}

// executeAndEncodeInPathToolCall is a convenience wrapper around executeInPathToolAndEncodeObservation
// returning the encoded observation token IDs.
func executeAndEncodeInPathToolCall(
	reqTok *Tokenizer,
	reqTable TokenTable,
	ctxStore *ContextStore,
	reqRegStore *RegisterStore,
	toolName string,
	payloadStr string,
) []int {
	_, _, _, _, obsIDs := executeInPathToolAndEncodeObservation(
		reqTok, reqTable, ctxStore, reqRegStore, toolName, payloadStr, time.Now(),
	)
	return obsIDs
}

// executeStopAndContinueMode runs the serving loop where SGLang halts at <|/call|> or <|/quote|>,
// Go executes the tool or quote expansion in <50us, appends the observation tokens,
// and immediately starts the continuation generation on SGLang.
func (s *FrontDoorServer) executeStopAndContinueMode(
	ctx context.Context,
	req OpenAIChatCompletionRequest,
	reqTok *Tokenizer,
	reqTable TokenTable,
	rendered *RenderedPrompt,
	onStreamDelta func(delta OpenAIStreamDelta) error,
) ([]int, *ReckonerTelemetry, error) {
	t0 := time.Now()
	temp := 0.0
	if req.Temperature != nil {
		temp = *req.Temperature
	}
	topP := 1.0
	if req.TopP != nil {
		topP = *req.TopP
	}
	maxNew := 512
	if req.MaxTokens != nil && *req.MaxTokens > 0 {
		maxNew = *req.MaxTokens
	} else if req.MaxCompletion != nil && *req.MaxCompletion > 0 {
		maxNew = *req.MaxCompletion
	}

	canonEffort, _, maskDelib, err := resolveEffortOrErr(req.ReasoningEffort)
	if err != nil {
		return nil, nil, err
	}
	if !maskDelib {
		stopContinueBudgetWarnOnce.Do(func() {
			fmt.Fprintf(os.Stderr, "[reckoner-harness] WARNING: reasoning_effort=%q thinking_budget is not enforced in stop_and_continue mode (use native mode for in-engine budget enforcement)\n", canonEffort)
		})
	}

	callMap := reqTable.CallTokenToTool
	if len(callMap) == 0 {
		callMap = callTokenToToolName
	}

	reqRegStore := NewRegisterStore()
	curInputIDs := make([]int, len(rendered.InputIDs))
	copy(curInputIDs, rendered.InputIDs)

	var allOutputIDs []int
	var toolCalls []ToolCallTelemetry
	var ttftMs float64
	firstTokenSeen := false
	totalCachedTokens := 0
	totalRecomputedTokens := 0
	streamCurMode := "ready"
	streamInCall := false
	streamInQuote := false

	var pendingCallStopTime time.Time
	var pendingResubmitTime time.Time
	hasPendingCallTiming := false

	stopToks := []int{reqTable.EndCall}
	if reqTable.EndQuote >= 0 {
		stopToks = append(stopToks, reqTable.EndQuote)
	}
	stopToks = append(stopToks, reqTable.EndTurn, reqTable.EndOfText)

	for round := 0; round <= s.cfg.MaxToolCalls; round++ {
		remBudget := maxNew - len(allOutputIDs)
		if remBudget <= 0 {
			break
		}

		samplingParams := map[string]interface{}{
			"temperature":         temp,
			"top_p":               topP,
			"max_new_tokens":      remBudget,
			"stop_token_ids":      stopToks,
			"no_stop_trim":        true,
			"skip_special_tokens": false,
		}
		if maskDelib {
			samplingParams["logit_bias"] = map[string]int{
				strconv.Itoa(reqTable.Deliberate): -100,
			}
		}

		sglReq := sglangTokenRequest{
			RequestID:      nextSessionID(),
			InputIDs:       curInputIDs,
			SamplingParams: samplingParams,
			Stream:         true,
		}

		var segIDs []int
		var segCached int
		var segPrompt int
		segFirstSeen := false

		err := s.engine.GenerateStream(ctx, sglReq, func(chunk sglangTokenChunk) error {
			now := time.Now()
			if !firstTokenSeen && len(chunk.DeltaTokenIDs) > 0 {
				firstTokenSeen = true
				ttftMs = float64(now.Sub(t0).Microseconds()) / 1000.0
			}
			if !segFirstSeen && len(chunk.DeltaTokenIDs) > 0 {
				segFirstSeen = true
				if hasPendingCallTiming && len(toolCalls) > 0 {
					tcIdx := len(toolCalls) - 1
					contTTFTMs := float64(now.Sub(pendingResubmitTime).Microseconds()) / 1000.0
					callToNextMs := float64(now.Sub(pendingCallStopTime).Microseconds()) / 1000.0
					latMs := toolCalls[tcIdx].LatencyUs / 1000.0
					obsTokens := toolCalls[tcIdx].ObsTokens
					estObsPrefillMs := contTTFTMs * 0.35
					if obsTokens > 0 && segPrompt > 0 {
						uncached := segPrompt - chunk.CachedTokens
						if uncached <= 0 {
							uncached = obsTokens
						}
						estObsPrefillMs = contTTFTMs * math.Min(0.85, math.Max(0.25, float64(obsTokens)/float64(uncached+obsTokens)))
					}
					pauseResumeMs := math.Max(0.0, callToNextMs-latMs-estObsPrefillMs)
					toolCalls[tcIdx].ObsPrefillMs = round4(estObsPrefillMs)
					toolCalls[tcIdx].PauseResumeMs = round4(pauseResumeMs)
					toolCalls[tcIdx].CallToNextTokenMs = round4(callToNextMs)
					toolCalls[tcIdx].CachedTokens = chunk.CachedTokens
					if segPrompt >= chunk.CachedTokens {
						toolCalls[tcIdx].RecomputedTokens = segPrompt - chunk.CachedTokens
					}
					hasPendingCallTiming = false
				}
			}
			if chunk.PromptTokens > 0 {
				segPrompt = chunk.PromptTokens
			}
			if chunk.CachedTokens > 0 {
				segCached = chunk.CachedTokens
			}

			for _, tokID := range chunk.DeltaTokenIDs {
				segIDs = append(segIDs, tokID)
				if onStreamDelta != nil {
					if tokID == reqTable.EndTurn || tokID == reqTable.EndOfText {
						continue
					}
					switch tokID {
					case reqTable.Ready:
						streamCurMode = "ready"
						continue
					case reqTable.Act:
						streamCurMode = "act"
						continue
					case reqTable.Deliberate:
						streamCurMode = "deliberate"
						continue
					case reqTable.Clarify:
						streamCurMode = "clarify"
						continue
					}
					if _, isCall := callMap[tokID]; isCall {
						streamInCall = true
						continue
					}
					if tokID == reqTable.EndCall {
						streamInCall = false
						continue
					}
					if reqTable.Quote >= 0 && tokID == reqTable.Quote {
						streamInQuote = true
						continue
					}
					if reqTable.EndQuote >= 0 && tokID == reqTable.EndQuote {
						streamInQuote = false
						continue
					}
					if streamInCall || streamInQuote || reqTok.IsAnchorToken(tokID) {
						continue
					}
					piece := reqTok.DecodeTokens([]int{tokID})
					if piece != "" {
						if streamCurMode == "deliberate" {
							if err := onStreamDelta(OpenAIStreamDelta{ReasoningContent: piece}); err != nil {
								return err
							}
						} else {
							if err := onStreamDelta(OpenAIStreamDelta{Content: piece}); err != nil {
								return err
							}
						}
					}
				}
			}
			return nil
		})
		if err != nil {
			return nil, nil, err
		}

		tStop := time.Now()
		if segPrompt == 0 {
			segPrompt = len(curInputIDs)
		}
		totalCachedTokens += segCached
		if segPrompt > segCached {
			totalRecomputedTokens += (segPrompt - segCached)
		}

		allOutputIDs = append(allOutputIDs, segIDs...)
		if len(segIDs) == 0 {
			break
		}

		lastTok := segIDs[len(segIDs)-1]
		if lastTok == reqTable.EndCall {
			// Find matching <|call:...|>
			callStartIdx := -1
			var toolName string
			for k := len(segIDs) - 2; k >= 0; k-- {
				if tn, ok := callMap[segIDs[k]]; ok {
					callStartIdx = k
					toolName = tn
					break
				}
			}
			if callStartIdx < 0 {
				break
			}
			payloadTokens := segIDs[callStartIdx+1 : len(segIDs)-1]
			payloadStr := strings.TrimSpace(reqTok.DecodeTokens(reqTok.StripAnchors(payloadTokens)))

			// If external tool call and client provided tools, stop here and return tool_calls
			if toolName == "external" && len(rendered.ParsedTools) > 0 {
				allOutputIDs = append(allOutputIDs, reqTable.EndTurn)
				break
			}

			tH0 := time.Now()
			var obsIDs []int
			resolvedToolName, resultStr, isErr, latUs, obsIDs := executeInPathToolAndEncodeObservation(
				reqTok, reqTable, rendered.ContextStore, reqRegStore, toolName, payloadStr, tH0,
			)
			toolName = resolvedToolName
			harnessIPCMs := float64(time.Since(tH0).Microseconds()) / 1000.0

			toolCalls = append(toolCalls, ToolCallTelemetry{
				Tool:         toolName,
				Payload:      payloadStr,
				Result:       resultStr,
				IsError:      isErr,
				LatencyUs:    round4(latUs),
				HarnessIPCMs: round4(harnessIPCMs),
				ObsTokens:    len(obsIDs),
			})

			allOutputIDs = append(allOutputIDs, obsIDs...)
			curInputIDs = append(curInputIDs, segIDs...)
			curInputIDs = append(curInputIDs, obsIDs...)
			pendingCallStopTime = tStop
			pendingResubmitTime = time.Now()
			hasPendingCallTiming = true
			continue
		}

		if reqTable.EndQuote >= 0 && lastTok == reqTable.EndQuote {
			quoteStartIdx := -1
			for k := len(segIDs) - 2; k >= 0; k-- {
				if segIDs[k] == reqTable.Quote {
					quoteStartIdx = k
					break
				}
			}
			if quoteStartIdx < 0 {
				break
			}
			quoteTokens := segIDs[quoteStartIdx+1 : len(segIDs)-1]
			tH0 := time.Now()
			mID := -1
			var lineIDs []int
			for _, qt := range quoteTokens {
				if qt >= M0TokenID && qt < M0TokenID+NumMessageTokens {
					mID = qt - M0TokenID
				} else if qt >= L0TokenID && qt < L0TokenID+NumLineAnchors {
					lineIDs = append(lineIDs, qt-L0TokenID)
				}
			}
			var obsIDs []int
			var qText string
			var qPayload string
			isErr := false
			if mID >= 0 && len(lineIDs) > 0 {
				lo, hi := lineIDs[0], lineIDs[0]
				for _, lid := range lineIDs[1:] {
					if lid < lo {
						lo = lid
					}
					if lid > hi {
						hi = lid
					}
				}
				qPayload = fmt.Sprintf("m%d, L%d, L%d", mID, lo, hi)
				qRes, qErr := rendered.ContextStore.Quote(qPayload)
				if qErr != nil || !qRes.Success {
					isErr = true
					errMsg := "quote failed"
					if qErr != nil {
						errMsg = qErr.Error()
					}
					obsIDs = EncodeToolObservation(reqTok, reqTable, "quote", errMsg, nil, true)
				} else {
					qText = qRes.Text
					obsIDs = EncodeToolObservation(reqTok, reqTable, "quote", qRes.Receipt, nil, false)
				}
			} else {
				isErr = true
				obsIDs = EncodeToolObservation(reqTok, reqTable, "quote", "malformed quote", nil, true)
			}
			latUs := float64(time.Since(tH0).Nanoseconds()) / 1000.0
			toolCalls = append(toolCalls, ToolCallTelemetry{
				Tool:         "quote",
				Payload:      qPayload,
				Result:       qText,
				IsError:      isErr,
				LatencyUs:    round4(latUs),
				HarnessIPCMs: round4(latUs / 1000.0),
				ObsTokens:    len(obsIDs),
			})
			allOutputIDs = append(allOutputIDs, obsIDs...)
			curInputIDs = append(curInputIDs, segIDs...)
			curInputIDs = append(curInputIDs, obsIDs...)
			pendingCallStopTime = tStop
			pendingResubmitTime = time.Now()
			hasPendingCallTiming = true
			continue
		}

		break
	}

	elapsedMs := float64(time.Since(t0).Microseconds()) / 1000.0
	decodeWindowMs := math.Max(0.1, elapsedMs-ttftMs)
	decodeTokS := 0.0
	if len(allOutputIDs) > 1 {
		decodeTokS = float64(len(allOutputIDs)-1) / (decodeWindowMs / 1000.0)
	}

	parsed := ParseAgentOutputTokensWithTable(
		reqTok,
		reqTable,
		allOutputIDs,
		rendered.ContextStore,
		rendered.ParsedTools,
		rendered.NameMapper,
	)

	telemetry := &ReckonerTelemetry{
		ServingMode:      "stop_and_continue",
		ActionMode:       parsed.ActionMode,
		TTFTMs:           round4(ttftMs),
		ElapsedMs:        round4(elapsedMs),
		DecodeTokPerSec:  round4(decodeTokS),
		PromptTokens:     len(rendered.InputIDs),
		CompletionTokens: len(allOutputIDs),
		ThinkingTokens:   parsed.ThinkingTokens,
		ToolCalls:        toolCalls,
		CachedTokens:     totalCachedTokens,
		RecomputedTokens: totalRecomputedTokens,
		FormatErrors:     parsed.FormatErrors,
	}
	return allOutputIDs, telemetry, nil
}

func round4(v float64) float64 {
	return math.Round(v*10000.0) / 10000.0
}

// Handler returns an http.Handler serving /v1/chat/completions, /v1/models, and /health.
func (s *FrontDoorServer) Handler() http.Handler {
	mux := http.NewServeMux()

	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		resp := map[string]interface{}{
			"status":       "ok",
			"sglang_addr":  s.cfg.SGLangAddr,
			"sglang_proto": s.engine.proto,
			"default_mode": s.cfg.DefaultMode,
		}
		if s.cfg.Profile != nil {
			resp["profile"] = s.cfg.Profile.Name
		}
		_ = json.NewEncoder(w).Encode(resp)
	})

	mux.HandleFunc("/v1/models", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		modelID := "reckoner-1.4b"
		if s.cfg.Profile != nil && s.cfg.Profile.Name != "" {
			modelID = s.cfg.Profile.Name
		}
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"object": "list",
			"data": []map[string]interface{}{
				{
					"id":       modelID,
					"object":   "model",
					"created":  1760700000,
					"owned_by": "maelie-labs",
				},
			},
		})
	})

	mux.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		body, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, "Failed to read request body", http.StatusBadRequest)
			return
		}

		var req OpenAIChatCompletionRequest
		if err := json.Unmarshal(body, &req); err != nil {
			http.Error(w, fmt.Sprintf("Invalid JSON: %v", err), http.StatusBadRequest)
			return
		}
		if _, _, _, err := resolveEffortOrErr(req.ReasoningEffort); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		prof := s.cfg.Profile
		reqTok := s.tok
		reqTable := s.table
		if req.Profile != nil {
			if strings.TrimSpace(*req.Profile) == "" {
				prof = nil
			} else {
				loadedProf, pErr := LoadProfile(*req.Profile)
				if pErr != nil {
					http.Error(w, pErr.Error(), http.StatusBadRequest)
					return
				}
				prof = loadedProf
			}
			tokPath := ""
			if profileUsesExternalTokenizer(prof) {
				tokPath = s.cfg.TokenizerPath
			}
			var tErr error
			reqTok, reqTable, tErr = LoadTokenizerForProfile(prof, tokPath, s.cfg.ModelDir)
			if tErr != nil {
				http.Error(w, tErr.Error(), http.StatusBadRequest)
				return
			}
		}

		if req.Model == "" {
			if prof != nil && prof.Name != "" {
				req.Model = prof.Name
			} else {
				req.Model = "reckoner-1.4b"
			}
		}

		graded := s.cfg.GradedByDefault
		if req.Graded != nil {
			graded = *req.Graded
		}

		messages := req.Messages
		if prof != nil {
			messages = ApplyProfileToMessages(req.Messages, prof, graded)
			var effLineErr error
			messages, effLineErr = ApplyEffortLineToMessages(messages, prof, req.ReasoningEffort)
			if effLineErr != nil {
				http.Error(w, effLineErr.Error(), http.StatusBadRequest)
				return
			}
		}

		anchorCtx := s.cfg.AnchorByDefault
		if req.AnchorContext != nil {
			anchorCtx = *req.AnchorContext
		}
		if req.Anchors != nil {
			anchorCtx = *req.Anchors
		}
		if req.NoAnchors != nil {
			anchorCtx = !*req.NoAnchors
		}
		if prof != nil && (!prof.Anchors || s.cfg.NoAnchorsOverride || (req.NoAnchors != nil && *req.NoAnchors) || (req.Anchors != nil && !*req.Anchors)) {
			anchorCtx = false
		}

		rendered, err := RenderChatPromptWithTable(reqTok, reqTable, messages, req.Tools, anchorCtx)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		mode := strings.ToLower(strings.TrimSpace(req.ServingMode))
		if mode == "" {
			mode = strings.ToLower(strings.TrimSpace(s.cfg.DefaultMode))
		}
		useStopContinue := (mode == "stop-and-continue" || mode == "stop_and_continue")

		reqID := fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano())
		created := time.Now().Unix()

		if req.Stream {
			w.Header().Set("Content-Type", "text/event-stream")
			w.Header().Set("Cache-Control", "no-cache")
			w.Header().Set("Connection", "keep-alive")
			flusher, ok := w.(http.Flusher)
			if !ok {
				http.Error(w, "Streaming unsupported", http.StatusInternalServerError)
				return
			}

			// Emit initial role chunk
			initChunk := OpenAIStreamChunk{
				ID:      reqID,
				Object:  "chat.completion.chunk",
				Created: created,
				Model:   req.Model,
				Choices: []OpenAIStreamChoice{
					{
						Index: 0,
						Delta: OpenAIStreamDelta{Role: "assistant"},
					},
				},
			}
			initBytes, _ := json.Marshal(initChunk)
			_, _ = fmt.Fprintf(w, "data: %s\n\n", string(initBytes))
			flusher.Flush()

			onDelta := func(delta OpenAIStreamDelta) error {
				chunk := OpenAIStreamChunk{
					ID:      reqID,
					Object:  "chat.completion.chunk",
					Created: created,
					Model:   req.Model,
					Choices: []OpenAIStreamChoice{
						{
							Index: 0,
							Delta: delta,
						},
					},
				}
				b, _ := json.Marshal(chunk)
				_, wErr := fmt.Fprintf(w, "data: %s\n\n", string(b))
				flusher.Flush()
				return wErr
			}

			var outputIDs []int
			var telemetry *ReckonerTelemetry
			if useStopContinue {
				outputIDs, telemetry, err = s.executeStopAndContinueMode(r.Context(), req, reqTok, reqTable, rendered, onDelta)
			} else {
				outputIDs, telemetry, err = s.executeNativeMode(r.Context(), req, reqTok, reqTable, rendered, onDelta)
			}
			if err != nil {
				errChunk := map[string]interface{}{"error": err.Error()}
				b, _ := json.Marshal(errChunk)
				_, _ = fmt.Fprintf(w, "data: %s\n\n", string(b))
				flusher.Flush()
				return
			}

			parsed := ParseAgentOutputTokensWithTable(
				reqTok,
				reqTable,
				outputIDs,
				rendered.ContextStore,
				rendered.ParsedTools,
				rendered.NameMapper,
			)
			if len(parsed.ExternalCalls) > 0 {
				_ = onDelta(OpenAIStreamDelta{ToolCalls: parsed.ExternalCalls})
			}

			finishReason := parsed.FinishReason
			finalChunk := OpenAIStreamChunk{
				ID:      reqID,
				Object:  "chat.completion.chunk",
				Created: created,
				Model:   req.Model,
				Choices: []OpenAIStreamChoice{
					{
						Index:        0,
						Delta:        OpenAIStreamDelta{},
						FinishReason: &finishReason,
					},
				},
				Usage: &OpenAIUsage{
					PromptTokens:     telemetry.PromptTokens,
					CompletionTokens: telemetry.CompletionTokens,
					TotalTokens:      telemetry.PromptTokens + telemetry.CompletionTokens,
				},
				ReckonerTelemetry: telemetry,
			}
			finalBytes, _ := json.Marshal(finalChunk)
			_, _ = fmt.Fprintf(w, "data: %s\n\n", string(finalBytes))
			_, _ = fmt.Fprintf(w, "data: [DONE]\n\n")
			flusher.Flush()
			return
		}

		// Non-streaming JSON response
		var outputIDs []int
		var telemetry *ReckonerTelemetry
		if useStopContinue {
			outputIDs, telemetry, err = s.executeStopAndContinueMode(r.Context(), req, reqTok, reqTable, rendered, nil)
		} else {
			outputIDs, telemetry, err = s.executeNativeMode(r.Context(), req, reqTok, reqTable, rendered, nil)
		}
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadGateway)
			return
		}

		parsed := ParseAgentOutputTokensWithTable(
			reqTok,
			reqTable,
			outputIDs,
			rendered.ContextStore,
			rendered.ParsedTools,
			rendered.NameMapper,
		)

		resp := OpenAIChatCompletionResponse{
			ID:      reqID,
			Object:  "chat.completion",
			Created: created,
			Model:   req.Model,
			Choices: []OpenAIChatChoice{
				{
					Index: 0,
					Message: OpenAIMessage{
						Role:             "assistant",
						Content:          parsed.Content,
						ReasoningContent: parsed.ReasoningContent,
						ToolCalls:        parsed.ExternalCalls,
					},
					FinishReason: parsed.FinishReason,
				},
			},
			Usage: OpenAIUsage{
				PromptTokens:     telemetry.PromptTokens,
				CompletionTokens: telemetry.CompletionTokens,
				TotalTokens:      telemetry.PromptTokens + telemetry.CompletionTokens,
			},
			ReckonerTelemetry: telemetry,
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	})

	return mux
}

// RunServeCommand parses `reckoner-harness serve` flags and starts the HTTP server.
func RunServeCommand(args []string) error {
	cfg := ServeConfig{
		Host:              "127.0.0.1",
		Port:              8753,
		SGLangAddr:        "127.0.0.1:30000",
		SGLangProto:       "auto",
		DefaultMode:       "native",
		AnchorByDefault:   false,
		NoAnchorsOverride: globalNoAnchorsFlag,
		GradedByDefault:   globalGraded,
		Profile:           globalProfile,
		TokenizerPath:     globalTokenizerPath,
		ModelDir:          globalModelDir,
		MaxToolCalls:      16,
	}

	for i := 0; i < len(args); i++ {
		arg := args[i]
		switch {
		case (arg == "--port" || arg == "-p") && i+1 < len(args):
			p, err := strconv.Atoi(args[i+1])
			if err != nil {
				return fmt.Errorf("invalid --port: %w", err)
			}
			cfg.Port = p
			i++
		case strings.HasPrefix(arg, "--port="):
			p, err := strconv.Atoi(strings.TrimPrefix(arg, "--port="))
			if err != nil {
				return fmt.Errorf("invalid --port: %w", err)
			}
			cfg.Port = p
		case arg == "--host" && i+1 < len(args):
			cfg.Host = args[i+1]
			i++
		case strings.HasPrefix(arg, "--host="):
			cfg.Host = strings.TrimPrefix(arg, "--host=")
		case (arg == "--sglang-addr" || arg == "--sglang") && i+1 < len(args):
			cfg.SGLangAddr = args[i+1]
			i++
		case strings.HasPrefix(arg, "--sglang-addr="):
			cfg.SGLangAddr = strings.TrimPrefix(arg, "--sglang-addr=")
		case arg == "--sglang-proto" && i+1 < len(args):
			cfg.SGLangProto = args[i+1]
			i++
		case strings.HasPrefix(arg, "--sglang-proto="):
			cfg.SGLangProto = strings.TrimPrefix(arg, "--sglang-proto=")
		case arg == "--mode" && i+1 < len(args):
			cfg.DefaultMode = args[i+1]
			i++
		case strings.HasPrefix(arg, "--mode="):
			cfg.DefaultMode = strings.TrimPrefix(arg, "--mode=")
		case arg == "--processor-str" && i+1 < len(args):
			cfg.ProcessorStr = args[i+1]
			i++
		case arg == "--profile" && i+1 < len(args):
			prof, err := LoadProfile(args[i+1])
			if err != nil {
				return err
			}
			cfg.Profile = prof
			i++
		case strings.HasPrefix(arg, "--profile="):
			prof, err := LoadProfile(strings.TrimPrefix(arg, "--profile="))
			if err != nil {
				return err
			}
			cfg.Profile = prof
		case arg == "--tokenizer" && i+1 < len(args):
			cfg.TokenizerPath = args[i+1]
			i++
		case strings.HasPrefix(arg, "--tokenizer="):
			cfg.TokenizerPath = strings.TrimPrefix(arg, "--tokenizer=")
		case (arg == "--model-dir" || arg == "--model-path" || arg == "--model") && i+1 < len(args):
			cfg.ModelDir = args[i+1]
			i++
		case strings.HasPrefix(arg, "--model-dir="):
			cfg.ModelDir = strings.TrimPrefix(arg, "--model-dir=")
		case strings.HasPrefix(arg, "--model-path="):
			cfg.ModelDir = strings.TrimPrefix(arg, "--model-path=")
		case strings.HasPrefix(arg, "--model="):
			cfg.ModelDir = strings.TrimPrefix(arg, "--model=")
		case arg == "--graded" || arg == "--graded=true":
			cfg.GradedByDefault = true
		case arg == "--graded=false":
			cfg.GradedByDefault = false
		case arg == "--anchor-context" || arg == "--anchors":
			cfg.AnchorByDefault = true
		case arg == "--no-anchors" || arg == "--anchor-context=false" || arg == "--anchors=false":
			cfg.AnchorByDefault = false
			if arg == "--no-anchors" || arg == "--anchors=false" {
				cfg.NoAnchorsOverride = true
			}
		}
	}

	srv, err := NewFrontDoorServer(cfg)
	if err != nil {
		return err
	}
	listenAddr := fmt.Sprintf("%s:%d", cfg.Host, cfg.Port)
	fmt.Printf("Reckoner Harness Front Door listening on http://%s (SGLang=%s, proto=%s, mode=%s)\n",
		listenAddr, cfg.SGLangAddr, srv.engine.proto, cfg.DefaultMode)
	return http.ListenAndServe(listenAddr, srv.Handler())
}
