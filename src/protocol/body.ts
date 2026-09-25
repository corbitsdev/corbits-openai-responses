import {
  BEARER_CREDENTIAL_SENTINEL,
  encodeToolName,
  type BuiltRequest,
  type ToolNameLimit,
} from "@intx/inference";
import { formatSafetyRatingText } from "@intx/types/runtime";
import type {
  ContentBlock,
  ConversationTurn,
  InferenceOptions,
} from "@intx/types/runtime";

import type { ResolvedResponsesQuirks, ResponsesHooks } from "../responses.js";

// Wire-charset limit for function names on the Responses surface:
// `^[a-zA-Z0-9_-]{1,64}$` across every vendor that speaks this protocol.
const RESPONSES_TOOL_NAME_LIMIT: ToolNameLimit = {
  provider: "responses",
  maxLength: 64,
};

type ResponsesContentPart =
  | { type: "input_text"; text: string }
  | { type: "output_text"; text: string }
  | { type: "input_image"; image_url: string }
  | { type: "input_image"; file_id: string }
  | { type: "input_file"; filename: string; file_data: string }
  | { type: "input_file"; file_id: string };

type ResponsesInputItem =
  | {
      type: "message";
      role: "user" | "assistant" | "system" | "developer";
      content: string | ResponsesContentPart[];
    }
  | { type: "function_call"; name: string; arguments: string; call_id: string }
  | { type: "function_call_output"; call_id: string; output: string }
  | { type: "reasoning"; summary: never[]; encrypted_content: string };

// Tool results carry a content array; the Responses API wants a string. Join
// the text parts; non-text content is not representable here and is dropped
// with a marker so the model is not misled into thinking it went missing
// silently.
function toolResultText(
  block: Extract<ContentBlock, { type: "tool_result" }>,
): string {
  const parts: string[] = [];
  for (const c of block.content) {
    if (c.type === "text") parts.push(c.text);
    else parts.push(`[unsupported ${c.type} content omitted]`);
  }
  return parts.join("");
}

// Grounded on the OpenAI Responses API document input shape: a base64
// document synthesizes a deterministic filename from its mimeType, since
// MediaSource carries no filename field of its own. Only PDF is supported
// today, matching the Chat Completions adapter's own document support.
function filenameForDocumentMime(mimeType: string): string {
  if (mimeType === "application/pdf") return "document.pdf";
  throw new Error(
    `@corbits/openai-responses: document input currently supports application/pdf only; received mimeType: ${mimeType}`,
  );
}

// Converts one content block to zero or one Responses content part. Blocks
// handled as whole `ResponsesInputItem`s elsewhere (tool_call, tool_result,
// signed thinking) are not passed here.
function toResponsesContentPart(
  block: ContentBlock,
  textKind: "input_text" | "output_text",
): ResponsesContentPart | undefined {
  switch (block.type) {
    case "text":
      return { type: textKind, text: block.text };
    case "safety_rating":
      // No input wire shape for safety_rating; render it as text so a
      // safety_rating-only turn is not a silently empty message.
      return { type: textKind, text: formatSafetyRatingText(block) };
    case "image": {
      const source = block.source;
      if (source.kind === "base64") {
        return {
          type: "input_image",
          image_url: `data:${source.mimeType};base64,${source.data}`,
        };
      }
      if (source.kind === "url") {
        return { type: "input_image", image_url: source.url };
      }
      // Unlike Chat Completions' image_url-only surface, the Responses API
      // accepts an uploaded file_id directly on an input_image part.
      return { type: "input_image", file_id: source.reference };
    }
    case "document": {
      const source = block.source;
      if (source.kind === "base64") {
        return {
          type: "input_file",
          filename: filenameForDocumentMime(source.mimeType),
          file_data: `data:${source.mimeType};base64,${source.data}`,
        };
      }
      if (source.kind === "file-reference") {
        return { type: "input_file", file_id: source.reference };
      }
      throw new Error(
        `@corbits/openai-responses: input_file does not accept url document sources; the Responses API only takes base64 data (file_data) or an uploaded file_id. Received url: ${source.url}`,
      );
    }
    case "audio":
    case "video":
      throw new Error(
        `@corbits/openai-responses: adapter does not yet handle ${block.type} content blocks.`,
      );
    case "citation":
    case "redacted_thinking":
      // Server-emitted attribution/opaque metadata with no Responses input
      // wire shape; dropping it is a silent no-op, not data loss the caller
      // needs to see, matching the Chat Completions adapter's own policy.
      return undefined;
    case "code_execution_request":
    case "code_execution_result":
      throw new Error(
        `@corbits/openai-responses: adapter does not handle ${block.type} content blocks.`,
      );
    case "refusal":
      throw new Error(
        "@corbits/openai-responses: adapter does not handle refusal content blocks.",
      );
    case "thinking":
    case "tool_call":
    case "tool_result":
      // Handled as whole items in toResponsesItems, not as content parts.
      return undefined;
  }
}

// Map one internal turn to zero or more Responses items. Reasoning blocks
// are echoed back only when they carry the opaque `encrypted_content` the
// backend issued AND that backend is the one this request is going to —
// replaying it to a different provider gets a 400 it cannot recover from.
function toResponsesItems(
  turn: ConversationTurn,
  requestModel: string,
  requestProvider: string,
  contentShape: "typed" | "flat",
): ResponsesInputItem[] {
  const items: ResponsesInputItem[] = [];
  const role = turn.role;
  const textKind: "input_text" | "output_text" =
    contentShape === "typed" && role === "assistant"
      ? "output_text"
      : "input_text";
  const parts: ResponsesContentPart[] = [];
  let hasNonTextPart = false;
  // A reasoning block whose signature could not be replayed (foreign
  // provider, model switch, or a missing/untagged signature) leaves any
  // function_call it produced without the reasoning item the Responses API
  // expects to precede it — the exact orphaned shape that degenerates
  // reasoning models. Suppress function_call items until the next text or
  // successfully-replayed reasoning item re-establishes a clean turn shape;
  // tool results are unaffected since they never need a preceding reasoning
  // item.
  let suppressOrphanedCalls = false;

  const flush = (): void => {
    if (parts.length === 0) return;
    const content: string | ResponsesContentPart[] =
      contentShape === "flat" && !hasNonTextPart
        ? parts.map((part) => ("text" in part ? part.text : "")).join("")
        : [...parts];
    items.push({ type: "message", role, content });
    parts.length = 0;
    hasNonTextPart = false;
    suppressOrphanedCalls = false;
  };

  for (const block of turn.content) {
    if (block.type === "tool_call") {
      if (suppressOrphanedCalls) continue;
      flush();
      items.push({
        type: "function_call",
        name: encodeToolName(block.name, RESPONSES_TOOL_NAME_LIMIT),
        arguments: JSON.stringify(block.arguments ?? {}),
        call_id: block.id,
      });
      continue;
    }
    if (block.type === "tool_result") {
      flush();
      suppressOrphanedCalls = false;
      items.push({
        type: "function_call_output",
        call_id: block.callId,
        output: toolResultText(block),
      });
      continue;
    }
    if (block.type === "thinking") {
      if (typeof block.signature !== "string" || block.signature.length === 0)
        continue;
      flush();
      const encryptedContent = signatureForModel(
        turn,
        requestModel,
        requestProvider,
        block.signature,
      );
      if (encryptedContent !== undefined) {
        items.push({
          type: "reasoning",
          summary: [],
          encrypted_content: encryptedContent,
        });
        suppressOrphanedCalls = false;
      } else {
        suppressOrphanedCalls = true;
      }
      continue;
    }
    const part = toResponsesContentPart(block, textKind);
    if (part === undefined) continue;
    if (part.type !== "input_text" && part.type !== "output_text")
      hasNonTextPart = true;
    parts.push(part);
  }
  flush();
  return items;
}

// Keeps the LAST occurrence of each duplicate function_call /
// function_call_output call_id, not the first: a duplicate is most often a
// corrected retry, and discarding the retry in favor of the stale original
// silently replays the wrong tool result. Both item types are covered — a
// duplicated function_call is just as invalid on the wire as a duplicated
// output.
function dedupeToolItems(items: ResponsesInputItem[]): ResponsesInputItem[] {
  const lastIndexForCall = new Map<string, number>();
  items.forEach((item, i) => {
    if (item.type === "function_call" || item.type === "function_call_output") {
      lastIndexForCall.set(`${item.type}:${item.call_id}`, i);
    }
  });
  return items.filter((item, i) => {
    if (item.type === "function_call" || item.type === "function_call_output") {
      return lastIndexForCall.get(`${item.type}:${item.call_id}`) === i;
    }
    return true;
  });
}

function toResponsesTools(options: InferenceOptions): unknown[] | undefined {
  if (options.tools === undefined || options.tools.length === 0) {
    return undefined;
  }
  // Responses function tools are FLAT — name/description/parameters sit
  // beside `type`, not nested under a `function` key (unlike Chat
  // Completions).
  return options.tools.map((t) => ({
    type: "function",
    name: encodeToolName(t.name, RESPONSES_TOOL_NAME_LIMIT),
    description: t.description,
    parameters: t.inputSchema,
  }));
}

function optionString(
  options: InferenceOptions,
  key: string,
): string | undefined {
  const value = options.providerOptions?.[key];
  return typeof value === "string" && value.length > 0 ? value : undefined;
}

function systemPromptItem(
  quirks: ResolvedResponsesQuirks,
  systemPrompt: string,
  hooks: ResponsesHooks | undefined,
): ResponsesInputItem {
  const placement = quirks.systemPrompt;
  const text =
    hooks?.wrapSystemPrompt !== undefined
      ? hooks.wrapSystemPrompt(systemPrompt)
      : systemPrompt;
  const content: string | ResponsesContentPart[] =
    placement.shape === "string" ? text : [{ type: "input_text", text }];
  return { type: "message", role: placement.role, content };
}

function reasoningField(
  quirks: ResolvedResponsesQuirks,
  options: InferenceOptions,
  hooks: ResponsesHooks | undefined,
): Record<string, unknown> | undefined {
  const reasoning = quirks.reasoning;
  if (reasoning === undefined) return undefined;
  const field: Record<string, unknown> = {};
  if (reasoning.summary !== undefined) field["summary"] = reasoning.summary;
  const effortOption = reasoning.effortOption;
  if (effortOption !== undefined) {
    const effort = optionString(options, effortOption);
    if (
      effort !== undefined &&
      (hooks?.includeReasoningEffort === undefined ||
        hooks.includeReasoningEffort(effort))
    ) {
      field["effort"] = effort;
    }
  }
  return Object.keys(field).length > 0 ? field : undefined;
}

export function buildResponsesRequest(
  quirks: ResolvedResponsesQuirks,
  messages: ConversationTurn[],
  model: string,
  options: InferenceOptions,
  requestProvider: string,
  hooks: ResponsesHooks | undefined,
): BuiltRequest {
  const conversation = dedupeToolItems(
    messages.flatMap((turn) =>
      toResponsesItems(turn, model, requestProvider, quirks.contentShape),
    ),
  );
  const leadingSystemItem =
    options.systemPrompt !== undefined
      ? systemPromptItem(quirks, options.systemPrompt, hooks)
      : undefined;
  const input =
    leadingSystemItem !== undefined
      ? [leadingSystemItem, ...conversation]
      : conversation;
  const tools = toResponsesTools(options);

  const body: Record<string, unknown> = {
    model,
    input,
    store: quirks.store,
    stream: quirks.stream,
    include: ["reasoning.encrypted_content"],
  };
  if (quirks.instructions !== undefined) {
    body["instructions"] = quirks.instructions;
  }
  if (quirks.parallelToolCalls !== undefined) {
    body["parallel_tool_calls"] = quirks.parallelToolCalls;
  }
  if (tools !== undefined) {
    body["tools"] = tools;
    body["tool_choice"] = "auto";
  }
  if (quirks.maxOutputTokens && options.maxTokens !== undefined) {
    body["max_output_tokens"] = options.maxTokens;
  }
  if (quirks.temperature && options.temperature !== undefined) {
    body["temperature"] = options.temperature;
  }
  const reasoning = reasoningField(quirks, options, hooks);
  if (reasoning !== undefined) body["reasoning"] = reasoning;

  const sessionId =
    quirks.sessionIdOption !== undefined
      ? optionString(options, quirks.sessionIdOption)
      : undefined;
  // With store:false this is the only cache-routing signal; keying it to the
  // inference thread's session id keeps every request on the same cache
  // shard.
  if (sessionId !== undefined) body["prompt_cache_key"] = sessionId;

  const headers: Record<string, string> = {
    "content-type": "application/json",
    // A non-streaming request must ask for a plain JSON body: sending
    // `text/event-stream` on `stream: false` gets some backends to 406, and
    // others to silently ignore the header and return JSON anyway — neither
    // is a wire contract to depend on.
    accept: quirks.stream ? "text/event-stream" : "application/json",
    authorization: BEARER_CREDENTIAL_SENTINEL,
  };
  for (const [name, value] of Object.entries(quirks.headers.static)) {
    headers[name] = value;
  }
  if (quirks.headers.modelHeader !== undefined) {
    headers[quirks.headers.modelHeader] = model;
  }
  for (const { optionKey, header } of quirks.headers.fromOption) {
    const value = optionString(options, optionKey);
    if (value !== undefined) headers[header] = value;
  }
  if (sessionId !== undefined && quirks.sessionIdHeader !== undefined) {
    headers[quirks.sessionIdHeader] = sessionId;
  }

  return { url: quirks.path, headers, body: JSON.stringify(body) };
}

// A thinking block's `signature` is opaque ciphertext a specific backend
// issued for a specific model; only that backend can decrypt it. `model` is
// arbitrary catalog/user-supplied text — nothing stops two distinct backends
// (proxy aliases, two OpenAI-compatible endpoints) from declaring the same
// literal model name, so comparing `turn.model` alone treats a foreign
// signature as safe to replay. `ConversationTurn` carries no field for which
// provider produced it, so provenance rides inside the signature string
// itself: capture tags it `<provider>:<ciphertext>` (see `tagSignature`),
// and replay only unwraps the ciphertext when both the tagged provider and
// the model match the current request.
//
// Provider, not the per-account source id, is the unit of decrypt
// capability — a backend shared across accounts can decrypt a signature
// issued to any of them, so keying on provider (rather than source id) is
// what lets an account switch keep reasoning continuity while a genuine
// cross-provider collision still gets dropped. A poisoned history self-heals
// on the next request instead of being replayed forever.
const SIGNATURE_TAG_SEPARATOR = ":";

export function tagSignature(
  provider: string,
  encryptedContent: string,
): string {
  return `${provider}${SIGNATURE_TAG_SEPARATOR}${encryptedContent}`;
}

function untagSignature(
  tagged: string,
): { provider: string; encryptedContent: string } | undefined {
  const idx = tagged.indexOf(SIGNATURE_TAG_SEPARATOR);
  if (idx === -1) return undefined;
  return {
    provider: tagged.slice(0, idx),
    encryptedContent: tagged.slice(idx + 1),
  };
}

// Returns the replayable `encrypted_content` for a thinking block's
// signature, or `undefined` when it must not be replayed on this request
// (persisted-turn model mismatch, or a signature tagged for a different
// provider).
function signatureForModel(
  turn: ConversationTurn,
  requestModel: string,
  requestProvider: string,
  signature: string,
): string | undefined {
  // `model` is optional on the persisted turn schema; a turn saved before
  // that field existed (or otherwise missing it) is not evidence of a model
  // switch — treat the absence as benign and fall through to the provider
  // check, rather than dropping reasoning that never actually crossed
  // models.
  if (turn.model !== undefined && turn.model !== requestModel) {
    return undefined;
  }
  const tagged = untagSignature(signature);
  if (tagged === undefined) return undefined;
  return tagged.provider === requestProvider
    ? tagged.encryptedContent
    : undefined;
}
