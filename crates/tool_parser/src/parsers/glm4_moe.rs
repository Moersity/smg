use std::{
    collections::{BTreeSet, HashMap},
    sync::LazyLock,
};

use async_trait::async_trait;
use openai_protocol::common::Tool;
use regex::Regex;
use serde_json::Value;

use crate::{
    errors::{ParserError, ParserResult},
    parsers::helpers,
    traits::ToolParser,
    types::{FunctionCall, StreamingParseResult, ToolCall, ToolCallItem},
};

/// GLM-4 MoE format parser for tool calls
///
/// Handles both GLM-4 MoE and GLM-4.7 MoE formats:
/// - GLM-4: `<tool_call>{name}\n<arg_key>{key}</arg_key>\n<arg_value>{value}</arg_value>\n</tool_call>`
/// - GLM-4.7: `<tool_call>{name}<arg_key>{key}</arg_key><arg_value>{value}</arg_value></tool_call>`
///
/// Features:
/// - XML-style tags for tool calls
/// - Key-value pairs for arguments
/// - Support for multiple sequential tool calls
pub struct Glm4MoeParser {
    /// Regex for extracting complete tool calls
    tool_call_extractor: Regex,
    /// Regex for extracting function details
    func_detail_extractor: Regex,
    /// Regex for extracting argument key-value pairs
    arg_extractor: Regex,

    /// Buffer for accumulating incomplete patterns across chunks
    buffer: String,

    /// Stores complete tool call info (name and arguments) for each tool being parsed
    prev_tool_call_arr: Vec<Value>,

    /// Index of currently streaming tool call (-1 means no active tool)
    current_tool_id: i32,

    /// Tracks raw JSON string content streamed to client for each tool's arguments
    streamed_args_for_tool: Vec<String>,

    /// Token configuration
    bot_token: &'static str,
    eot_token: &'static str,
}

impl Glm4MoeParser {
    /// Create a new generic GLM MoE parser with a custom func_detail_extractor pattern
    ///
    /// # Arguments
    /// - `func_detail_pattern`: Regex pattern for extracting function name and arguments
    ///   - For GLM-4: `r"(?s)<tool_call>([^\n]*)\n(.*)</tool_call>"`
    ///   - For GLM-4.7: `r"(?s)<tool_call>\s*([^<\s]+)\s*(.*?)</tool_call>"`
    #[expect(
        clippy::expect_used,
        reason = "regex patterns are compile-time string literals"
    )]
    pub(crate) fn new(func_detail_pattern: &str) -> Self {
        // Use (?s) flag for DOTALL mode to handle newlines
        let tool_call_pattern = r"(?s)<tool_call>.*?</tool_call>";
        let tool_call_extractor = Regex::new(tool_call_pattern).expect("Valid regex pattern");

        let func_detail_extractor = Regex::new(func_detail_pattern).expect("Valid regex pattern");

        let arg_pattern = r"(?s)<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>";
        let arg_extractor = Regex::new(arg_pattern).expect("Valid regex pattern");

        Self {
            tool_call_extractor,
            func_detail_extractor,
            arg_extractor,
            buffer: String::new(),
            prev_tool_call_arr: Vec::new(),
            current_tool_id: -1,
            streamed_args_for_tool: Vec::new(),
            bot_token: "<tool_call>",
            eot_token: "</tool_call>",
        }
    }

    /// Create a new GLM-4.5/4.6 MoE parser (with newline-based format)
    pub fn glm45() -> Self {
        Self::new(r"(?s)<tool_call>([^\n]*)\n(.*)</tool_call>")
    }

    /// Create a new GLM-4.7 MoE parser (with whitespace-based format)
    pub fn glm47() -> Self {
        Self::new(r"(?s)<tool_call>\s*([^<\s]+)\s*(.*?)</tool_call>")
    }

    /// Parse arguments, coercing each value by its declared schema type and
    /// falling back to [`infer_value`] when the type is unknown.
    fn parse_arguments(
        &self,
        args_text: &str,
        param_types: &HashMap<String, String>,
    ) -> serde_json::Map<String, Value> {
        let mut arguments = serde_json::Map::new();

        for capture in self.arg_extractor.captures_iter(args_text) {
            let key = capture.get(1).map_or("", |m| m.as_str()).trim();
            let value_str = capture.get(2).map_or("", |m| m.as_str()).trim();

            let value =
                helpers::coerce_by_schema_type(value_str, param_types.get(key).map(String::as_str))
                    .unwrap_or_else(|| infer_value(value_str));

            arguments.insert(key.to_string(), value);
        }

        arguments
    }

    /// Parse a single tool call block
    fn parse_tool_call(&self, block: &str, tools: &[Tool]) -> ParserResult<Option<ToolCall>> {
        if let Some(captures) = self.func_detail_extractor.captures(block) {
            // Get function name
            let func_name = captures.get(1).map_or("", |m| m.as_str()).trim();

            // Get arguments text
            let args_text = captures.get(2).map_or("", |m| m.as_str());

            let param_types = helpers::param_types_for_function(tools, func_name);
            let arguments = self.parse_arguments(args_text, &param_types);

            let arguments_str = serde_json::to_string(&arguments)
                .map_err(|e| ParserError::ParsingFailed(e.to_string()))?;

            Ok(Some(ToolCall {
                function: FunctionCall {
                    name: func_name.to_string(),
                    arguments: arguments_str,
                },
            }))
        } else {
            Ok(None)
        }
    }

    /// Parse all tool calls from text (shared logic for complete and incremental parsing)
    fn parse_tool_calls_from_text(&self, text: &str, tools: &[Tool]) -> Vec<ToolCall> {
        let mut parsed = Vec::new();

        for mat in self.tool_call_extractor.find_iter(text) {
            match self.parse_tool_call(mat.as_str(), tools) {
                Ok(Some(tool)) => parsed.push(tool),
                Ok(None) => continue,
                Err(e) => {
                    tracing::debug!("Failed to parse tool call: {}", e);
                    continue;
                }
            }
        }

        parsed
    }
}

impl Glm4MoeParser {
    /// Shared non-streaming parse, schema-aware when `tools` are provided.
    fn parse_complete_inner(
        &self,
        text: &str,
        tools: &[Tool],
    ) -> ParserResult<(String, Vec<ToolCall>)> {
        if !self.has_tool_markers(text) {
            return Ok((text.to_string(), vec![]));
        }

        // Find where tool calls begin
        // Safe: has_tool_markers() already confirmed the marker exists
        let idx = text
            .find("<tool_call>")
            .ok_or_else(|| ParserError::ParsingFailed("tool call marker not found".to_string()))?;
        let normal_text = text[..idx].to_string();

        let parsed = self.parse_tool_calls_from_text(text, tools);

        // If no tools were successfully parsed despite having markers, return entire text as fallback
        if parsed.is_empty() {
            return Ok((text.to_string(), vec![]));
        }

        Ok((normal_text, parsed))
    }
}

/// Infer a JSON value from raw text when the schema type is unknown: JSON
/// (numbers/bools/null/objects/arrays), then Python-style literals, then string.
fn infer_value(value_str: &str) -> Value {
    if let Ok(json_val) = serde_json::from_str::<Value>(value_str) {
        return json_val;
    }
    match value_str {
        "true" | "True" => Value::Bool(true),
        "false" | "False" => Value::Bool(false),
        "null" | "None" => Value::Null,
        _ => {
            if let Ok(num) = value_str.parse::<i64>() {
                Value::Number(num.into())
            } else if let Ok(num) = value_str.parse::<f64>() {
                serde_json::Number::from_f64(num)
                    .map_or_else(|| Value::String(value_str.to_string()), Value::Number)
            } else {
                Value::String(value_str.to_string())
            }
        }
    }
}

impl Default for Glm4MoeParser {
    fn default() -> Self {
        Self::glm45()
    }
}

#[async_trait]
impl ToolParser for Glm4MoeParser {
    async fn parse_complete(&self, text: &str) -> ParserResult<(String, Vec<ToolCall>)> {
        self.parse_complete_inner(text, &[])
    }

    async fn parse_complete_with_tools(
        &self,
        text: &str,
        tools: &[Tool],
    ) -> ParserResult<(String, Vec<ToolCall>)> {
        self.parse_complete_inner(text, tools)
    }

    async fn parse_incremental(
        &mut self,
        chunk: &str,
        tools: &[Tool],
    ) -> ParserResult<StreamingParseResult> {
        // Python logic: Wait for complete tool call, then parse it all at once
        self.buffer.push_str(chunk);
        let current_text = &self.buffer.clone();

        // Check if we have bot_token
        let start = current_text.find(self.bot_token);
        if start.is_none() {
            self.buffer.clear();
            // If we're in the middle of streaming (current_tool_id > 0), don't return text
            let normal_text = if self.current_tool_id > 0 {
                String::new()
            } else {
                current_text.clone()
            };
            return Ok(StreamingParseResult {
                normal_text,
                calls: vec![],
            });
        }

        // Check if we have eot_token (end of tool call)
        let end = current_text.find(self.eot_token);
        if let Some(end_pos) = end {
            // We have a complete tool call!

            // Initialize state if this is the first tool call
            if self.current_tool_id == -1 {
                self.current_tool_id = 0;
                self.prev_tool_call_arr = Vec::new();
                self.streamed_args_for_tool = vec![String::new()];
            }

            // Ensure we have enough entries in our tracking arrays
            helpers::ensure_capacity(
                self.current_tool_id,
                &mut self.prev_tool_call_arr,
                &mut self.streamed_args_for_tool,
            );

            // Parse the complete block using shared helper
            let block_end = end_pos + self.eot_token.len();
            let parsed_tools = self.parse_tool_calls_from_text(&current_text[..block_end], tools);

            // Extract normal text before tool calls
            let idx = current_text.find(self.bot_token);
            let normal_text = if let Some(pos) = idx {
                current_text[..pos].trim().to_string()
            } else {
                String::new()
            };

            // Build tool indices for validation
            let tool_indices = helpers::get_tool_indices(tools);

            let mut calls = Vec::new();

            if !parsed_tools.is_empty() {
                // Take the first tool and convert to ToolCallItem
                let tool_call = &parsed_tools[0];
                let tool_id = self.current_tool_id as usize;

                // Validate tool name
                if !tool_indices.contains_key(&tool_call.function.name) {
                    // Invalid tool name - skip this tool, preserve indexing for next tool
                    tracing::debug!("Invalid tool name '{}' - skipping", tool_call.function.name);
                    helpers::reset_current_tool_state(
                        &mut self.buffer,
                        &mut false, // glm45_moe/glm47_moe doesn't track name_sent per tool
                        &mut self.streamed_args_for_tool,
                        &self.prev_tool_call_arr,
                    );
                    return Ok(StreamingParseResult::default());
                }

                calls.push(ToolCallItem {
                    tool_index: tool_id,
                    name: Some(tool_call.function.name.clone()),
                    parameters: tool_call.function.arguments.clone(),
                });

                // Store in tracking arrays
                if self.prev_tool_call_arr.len() <= tool_id {
                    self.prev_tool_call_arr
                        .resize_with(tool_id + 1, || Value::Null);
                }

                // Parse parameters as JSON and store
                if let Ok(args) = serde_json::from_str::<Value>(&tool_call.function.arguments) {
                    self.prev_tool_call_arr[tool_id] = serde_json::json!({
                        "name": tool_call.function.name,
                        "arguments": args,
                    });
                }

                if self.streamed_args_for_tool.len() <= tool_id {
                    self.streamed_args_for_tool
                        .resize_with(tool_id + 1, String::new);
                }
                self.streamed_args_for_tool[tool_id].clone_from(&tool_call.function.arguments);

                self.current_tool_id += 1;
            }

            // Remove processed portion from buffer
            self.buffer = current_text[block_end..].to_string();
            return Ok(StreamingParseResult { normal_text, calls });
        }

        // No complete tool call yet - return normal text before start token
        // Safe: start.is_none() case was handled above (early return)
        let Some(start_pos) = start else {
            return Ok(StreamingParseResult::default());
        };
        let normal_text = current_text[..start_pos].to_string();
        self.buffer = current_text[start_pos..].to_string();

        Ok(StreamingParseResult {
            normal_text,
            calls: vec![],
        })
    }

    fn has_tool_markers(&self, text: &str) -> bool {
        text.contains(self.bot_token)
    }

    fn get_unstreamed_tool_args(&self) -> Option<Vec<ToolCallItem>> {
        helpers::get_unstreamed_args(&self.prev_tool_call_arr, &self.streamed_args_for_tool)
    }

    fn reset(&mut self) {
        self.buffer.clear();
        self.prev_tool_call_arr.clear();
        self.current_tool_id = -1;
        self.streamed_args_for_tool.clear();
    }
}

// ============================================================================
// Full-assistant-turn EBNF for non-strict GLM47 tool calls
// ============================================================================
//
// Mirrors the upstream engine grammar (SGLang `glm4_moe_detector`): one EBNF
// covering the whole assistant turn — thinking block, plain text, and
// zero-or-more native `<tool_call>` blocks — handed to the engine's grammar
// backend so free-form generation cannot drift off the native format.
// Selected by the registry factory via `uses_full_assistant_constraint`.

/// Framing tokens the free-text and thinking rules must not emit.
const SPECIAL_TOKENS: &[&str] = &[
    "<think>",
    "</think>",
    "<tool_call>",
    "</tool_call>",
    "<arg_key>",
    "</arg_key>",
    "<arg_value>",
    "</arg_value>",
    "<|assistant|>",
];

static TEXT_RULES: LazyLock<String> = LazyLock::new(|| string_excluding("text", SPECIAL_TOKENS));
static THINKING_RULES: LazyLock<String> =
    LazyLock::new(|| string_excluding("thinking", &SPECIAL_TOKENS[1..8]));

const JSON_RULES: &str = r#"
json_string ::= "\"" ([^"\\\x00-\x1F] | "\\" (["\\/bfnrt] | "u" [a-fA-F0-9]{4}))* "\""
number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
boolean ::= "true" | "false"
null ::= "null"
ws ::= [ \n\t]*
array ::= "[" (ws value (ws "," ws value)*)? ws "]"
object ::= "{" (ws json_string ws ":" ws value (ws "," ws json_string ws ":" ws value)*)? ws "}"
value ::= number | boolean | null | json_string | array | object
"#;

fn literal(value: &str) -> String {
    Value::String(value.to_owned()).to_string()
}

/// Build string rules that accept anything except the framing `patterns`,
/// via a failure-link prefix automaton over the excluded alphabet.
fn string_excluding(name: &str, patterns: &[&str]) -> String {
    let mut prefixes = BTreeSet::from([String::new()]);
    let mut alphabet = BTreeSet::new();
    for pattern in patterns {
        for (index, character) in pattern.char_indices() {
            prefixes.insert(pattern[..index].to_owned());
            alphabet.insert(character);
        }
    }
    let prefixes: Vec<_> = prefixes.into_iter().collect();
    let excluded: String = alphabet
        .iter()
        .map(|character| match character {
            '\\' | ']' | '^' | '-' => format!("\\{character}"),
            _ => character.to_string(),
        })
        .collect();
    let mut rules = vec![format!("{name} ::= {name}_0")];
    for (index, prefix) in prefixes.iter().enumerate() {
        let mut choices = vec![format!("[^{excluded}] {name}_0"), "\"\"".to_owned()];
        for character in &alphabet {
            let candidate = format!("{prefix}{character}");
            if patterns.iter().any(|pattern| candidate.ends_with(pattern)) {
                continue;
            }
            let target = prefixes
                .iter()
                .enumerate()
                .filter(|(_, suffix)| candidate.ends_with(suffix.as_str()))
                .max_by_key(|(_, suffix)| suffix.len())
                .map_or(0, |(target, _)| target);
            choices.push(format!(
                "{} {name}_{target}",
                literal(&character.to_string())
            ));
        }
        rules.push(format!("{name}_{index} ::= {}", choices.join(" | ")));
    }
    rules.join("\n")
}

fn type_rule(schema_type: &str) -> &'static str {
    match schema_type {
        "integer" | "number" => "number",
        "boolean" => "boolean",
        "null" => "null",
        "array" => "array",
        "object" => "object",
        _ => "text",
    }
}

fn value_rule(schema: &Value) -> Result<String, String> {
    if let Some(values) = schema.get("enum").and_then(Value::as_array) {
        if values.is_empty() {
            return Err("Tool parameter enum must not be empty".to_owned());
        }
        return Ok(values
            .iter()
            .map(|value| match value {
                Value::String(value) => literal(value),
                _ => literal(&value.to_string()),
            })
            .collect::<Vec<_>>()
            .join(" | "));
    }
    match schema.get("type") {
        Some(Value::String(schema_type)) => Ok(type_rule(schema_type).to_owned()),
        Some(Value::Array(schema_types)) if !schema_types.is_empty() => Ok(schema_types
            .iter()
            .map(|schema_type| type_rule(schema_type.as_str().unwrap_or("string")))
            .collect::<Vec<_>>()
            .join(" | ")),
        _ => Ok("text".to_owned()),
    }
}

impl Glm4MoeParser {
    /// Generate the full-assistant-turn EBNF for `tools` (non-strict GLM47
    /// calls): `assistant_turn ::= thinking_block text tool_calls`. Argument
    /// pairs stay non-strict — omittable, repeatable, any order — with
    /// shallow schema constraints (enums, JSON types) on values. An empty
    /// `tools` list yields a grammar with no tool-call branch; otherwise
    /// `at_least_one` selects `tool_call+` — a call is forced, matching
    /// required / named / required-mode tool choices — over `tool_call*`.
    pub(crate) fn generate_chat_ebnf(
        tools: &[Tool],
        enable_thinking: bool,
        at_least_one: bool,
    ) -> Result<String, String> {
        let mut rules = vec![
            "root ::= assistant_turn".to_owned(),
            "assistant_turn ::= thinking_block text tool_calls".to_owned(),
            if enable_thinking {
                "thinking_block ::= thinking \"</think>\"".to_owned()
            } else {
                "thinking_block ::= \"\"".to_owned()
            },
            TEXT_RULES.clone(),
        ];
        if enable_thinking {
            rules.push(THINKING_RULES.clone());
        }
        if tools.is_empty() {
            rules.push("tool_calls ::= \"\"".to_owned());
        } else {
            rules.push(if at_least_one {
                "tool_calls ::= tool_call+".to_owned()
            } else {
                "tool_calls ::= tool_call*".to_owned()
            });
            rules.push(format!(
                "tool_call ::= \"<tool_call>\" ({}) \"</tool_call>\"",
                (0..tools.len())
                    .map(|index| format!("call_{index}"))
                    .collect::<Vec<_>>()
                    .join(" | ")
            ));
            for (index, tool) in tools.iter().enumerate() {
                let mut arguments = Vec::new();
                if let Some(properties) = tool
                    .function
                    .parameters
                    .get("properties")
                    .and_then(Value::as_object)
                {
                    for (key, schema) in properties {
                        arguments.push(format!(
                            "{} ({}) \"</arg_value>\"",
                            literal(&format!("<arg_key>{key}</arg_key><arg_value>")),
                            value_rule(schema)?,
                        ));
                    }
                }
                let arguments = if arguments.is_empty() {
                    "\"\"".to_owned()
                } else {
                    format!("({})*", arguments.join(" | "))
                };
                rules.push(format!(
                    "call_{index} ::= {} {arguments}",
                    literal(&tool.function.name),
                ));
            }
            rules.push(JSON_RULES.to_owned());
        }
        Ok(rules.join("\n"))
    }
}

#[cfg(test)]
mod tests {
    use openai_protocol::common::Function;

    use super::*;

    fn tool_with_props(props: Value) -> Vec<Tool> {
        vec![Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "f".to_string(),
                description: None,
                parameters: serde_json::json!({"type": "object", "properties": props}),
                strict: None,
            },
        }]
    }

    // A forcing tool_choice must not permit a tool-less assistant turn.
    #[test]
    fn chat_ebnf_requires_a_call_only_when_forced() {
        let tools = tool_with_props(serde_json::json!({
            "city": {"type": "string"}
        }));
        let grammar = Glm4MoeParser::generate_chat_ebnf(&tools, false, false).unwrap();
        assert!(grammar.contains("tool_calls ::= tool_call*"));
        assert!(!grammar.contains("tool_calls ::= tool_call+"));

        let grammar = Glm4MoeParser::generate_chat_ebnf(&tools, true, true).unwrap();
        assert!(grammar.contains("tool_calls ::= tool_call+"));

        // No tools permitted → no tool-call branch regardless of forcing.
        let grammar = Glm4MoeParser::generate_chat_ebnf(&[], true, true).unwrap();
        assert!(grammar.contains("tool_calls ::= \"\""));
    }

    // String-typed params stay strings even when they look numeric/bool/array.
    #[tokio::test]
    async fn test_schema_aware_coercion_keeps_strings() {
        let tools = tool_with_props(serde_json::json!({
            "limit": {"type": "string"},
            "flag": {"type": "string"},
            "coords": {"type": "string"},
            "count": {"type": "integer"},
        }));
        let text = "<tool_call>f\n\
            <arg_key>limit</arg_key>\n<arg_value>4</arg_value>\n\
            <arg_key>flag</arg_key>\n<arg_value>true</arg_value>\n\
            <arg_key>coords</arg_key>\n<arg_value>[60,30]</arg_value>\n\
            <arg_key>count</arg_key>\n<arg_value>5</arg_value>\n\
            </tool_call>";
        let (_, calls) = Glm4MoeParser::glm45()
            .parse_complete_with_tools(text, &tools)
            .await
            .unwrap();
        assert_eq!(calls.len(), 1);
        let args: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["limit"], Value::String("4".to_string()));
        assert_eq!(args["flag"], Value::String("true".to_string()));
        assert_eq!(args["coords"], Value::String("[60,30]".to_string()));
        assert_eq!(args["count"], Value::Number(5.into()));
    }

    // The streaming path threads `tools` separately, so cover it too.
    #[tokio::test]
    async fn test_streaming_schema_aware_coercion() {
        let tools = tool_with_props(serde_json::json!({
            "limit": {"type": "string"},
            "count": {"type": "integer"},
        }));
        let text = "<tool_call>f\n\
            <arg_key>limit</arg_key>\n<arg_value>4</arg_value>\n\
            <arg_key>count</arg_key>\n<arg_value>5</arg_value>\n\
            </tool_call>";
        let result = Glm4MoeParser::glm45()
            .parse_incremental(text, &tools)
            .await
            .unwrap();
        let args: Value = serde_json::from_str(&result.calls[0].parameters).unwrap();
        assert_eq!(args["limit"], Value::String("4".to_string()));
        assert_eq!(args["count"], Value::Number(5.into()));
    }
}
