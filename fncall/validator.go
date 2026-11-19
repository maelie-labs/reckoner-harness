package fncall

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/santhosh-tekuri/jsonschema/v5"
)

// ValidateArguments validates the arguments against the tool's JSON Schema.
func ValidateArguments(tool *ParsedTool, args map[string]interface{}) error {
	if tool == nil || tool.ParametersSchema == nil {
		return nil
	}

	// Reject unknown/invented parameter keys
	knownKeys := make(map[string]bool)
	for _, prop := range tool.Properties {
		knownKeys[prop.SanitizedName] = true
		knownKeys[prop.OriginalName] = true
	}
	for k := range args {
		if !knownKeys[k] {
			return fmt.Errorf("ErrArgs: unknown parameter '%s' for tool '%s'", k, tool.SanitizedName)
		}
	}

	schemaBytes, err := json.Marshal(tool.ParametersSchema)
	if err != nil {
		return fmt.Errorf("ErrArgs: failed to encode schema: %w", err)
	}

	compiler := jsonschema.NewCompiler()
	schemaURI := fmt.Sprintf("schema://%s.json", tool.SanitizedName)
	if err := compiler.AddResource(schemaURI, strings.NewReader(string(schemaBytes))); err != nil {
		return fmt.Errorf("ErrArgs: failed to add schema resource: %w", err)
	}

	schema, err := compiler.Compile(schemaURI)
	if err != nil {
		return fmt.Errorf("ErrArgs: failed to compile schema: %w", err)
	}

	// Remap sanitized argument names to original schema names if needed
	remappedArgs := make(map[string]interface{})
	for k, v := range args {
		origK := k
		for _, prop := range tool.Properties {
			if prop.SanitizedName == k {
				origK = prop.OriginalName
				break
			}
		}
		remappedArgs[origK] = v
	}

	// Normalize arguments through JSON marshal/unmarshal for jsonschema library
	argsBytes, err := json.Marshal(remappedArgs)
	if err != nil {
		return fmt.Errorf("ErrArgs: failed to marshal arguments: %w", err)
	}

	var jsonVal interface{}
	dec := json.NewDecoder(strings.NewReader(string(argsBytes)))
	if err := dec.Decode(&jsonVal); err != nil {
		return fmt.Errorf("ErrArgs: failed to decode arguments: %w", err)
	}

	if err := schema.Validate(jsonVal); err != nil {
		if valErr, ok := err.(*jsonschema.ValidationError); ok {
			return fmt.Errorf("ErrArgs: %s", formatValidationError(valErr))
		}
		return fmt.Errorf("ErrArgs: %w", err)
	}

	return nil
}

func formatValidationError(vErr *jsonschema.ValidationError) string {
	// Find the most informative leaf error
	leaf := vErr
	for len(leaf.Causes) > 0 {
		leaf = leaf.Causes[0]
	}
	msg := leaf.Message
	if leaf.InstanceLocation != "" {
		field := strings.TrimPrefix(leaf.InstanceLocation, "/")
		field = strings.ReplaceAll(field, "/", ".")
		return fmt.Sprintf("parameter '%s': %s", field, msg)
	}
	return msg
}
