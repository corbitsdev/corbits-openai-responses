import { type } from "arktype";

import { decodeToolName, ProtocolMismatchError } from "@intx/inference";
import type {
  InferenceEvent,
  LastCycleSource,
  PartialMessage,
  TokenUsage,
} from "@intx/types/runtime";

import { tagSignature } from "./body.js";

// ---------------------------------------------------------------------------
// Event schemas
// ---------------------------------------------------------------------------
//
// Every `response.*` event type this adapter acts on gets a schema here. An
// event whose `type` is not one of these is protocol-legal but uninteresting
// (a lifecycle envelope this adapter has nothing to do with) and is ignored;
// an event whose `type` IS one of these but whose shape fails validation is
// a genuine protocol mismatch and throws.

const EMPTY_PARTIAL: PartialMessage = { text: "" };

const ResponsesEnvelope = type({ type: "string" });

const TextDeltaEvent = type({ item_id: "string > 0", delta: "string" });
// Every reasoning delta the Responses API emits carries the item_id of the
// reasoning item it belongs to and a delta string (possibly empty); an event
// of this type missing either field is not a lenient variant, it's a
// protocol mismatch this adapter cannot attribute to any block.
const ReasoningDeltaEvent = type({ item_id: "string > 0", delta: "string" });
const RefusalDeltaEvent = type({ item_id: "string > 0", delta: "string" });
const RefusalDoneEvent = type({ item_id: "string > 0", refusal: "string" });
const FunctionCallArgsDeltaEvent = type({
  item_id: "string > 0",
  delta: "string",
});

const OutputItemAddedEvent = type({
  item: {
    type: "string",
    "id?": "string > 0",
    "call_id?": "string > 0",
    "name?": "string > 0",
  },
});

const OutputItemDoneEvent = type({
  item: {
    type: "string",
    "id?": "string > 0",
    "encrypted_content?": "string",
  },
});

const ResponsesUsage = type({
  "input_tokens?": "number",
  "output_tokens?": "number",
  "input_tokens_details?": type({
    "cached_tokens?": "number",
    "cache_write_tokens?": "number",
    "cache_creation_tokens?": "number",
  }).or("null"),
  "output_tokens_details?": type({ "reasoning_tokens?": "number" }).or("null"),
}).or("null");

const CompletedEvent = type({ response: { "usage?": ResponsesUsage } });

const FailedEvent = type({
  "response?": { "error?": { "message?": "string" } },
});
const ErrorEvent = type({ "message?": "string" });

// Maps the Responses API's usage object onto the internal TokenUsage,
// splitting the cached-token subset out of input_tokens. Shared by the SSE
// `response.completed` handler and the non-streaming parseJSONResponse,
// whose usage objects carry the same field names.
function toInferenceUsage(
  usage: typeof ResponsesUsage.infer | undefined,
): TokenUsage {
  if (usage === null || usage === undefined) {
    return { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, thinking: 0 };
  }
  // Responses-API `input_tokens` counts the full prompt and `cached_tokens`
  // is a subset of it. Downstream consumers treat `TokenUsage` fields as
  // non-overlapping and sum them, so the cached subset must be split out of
  // input here — emitting the wire counts verbatim double-counts every
  // cached token and inflates context occupancy up to ~2x on high
  // cache-hit sessions.
  const totalInput = usage.input_tokens ?? 0;
  const details = usage.input_tokens_details;
  // Counters are non-negative quantities: a negative count off the wire is
  // garbage, clamped to zero rather than propagated into sums or allowed to
  // inflate input by subtracting a negative.
  const cachedTokens = Math.max(0, details?.cached_tokens ?? 0);
  // OpenAI (GPT-5.6+) reports cache writes as `cache_write_tokens` and
  // documents them as a subset of `input_tokens`, so they must be split out
  // of input exactly like `cached_tokens`. Gateways fronting OpenAI-shaped
  // endpoints report the Anthropic-shaped `cache_creation_tokens` instead;
  // its subset relationship to `input_tokens` is unobservable from here, so
  // that fallback keeps the historic behavior of not reducing input.
  const rawWriteTokens = details?.cache_write_tokens;
  const openaiWriteTokens =
    rawWriteTokens === undefined ? undefined : Math.max(0, rawWriteTokens);
  return {
    input: Math.max(0, totalInput - cachedTokens - (openaiWriteTokens ?? 0)),
    output: Math.max(0, usage.output_tokens ?? 0),
    cacheRead: cachedTokens,
    cacheWrite: Math.max(
      0,
      openaiWriteTokens ?? details?.cache_creation_tokens ?? 0,
    ),
    thinking: Math.max(0, usage.output_tokens_details?.reasoning_tokens ?? 0),
  };
}

// ---------------------------------------------------------------------------
// Streaming parse
// ---------------------------------------------------------------------------

// Per-request block indexing. The Responses stream tags every streaming item
// with an `item_id`, so one content-block index is allocated per distinct
// item id (regardless of kind). Keying by item id — rather than one sticky
// index per kind — preserves true arrival order when reasoning, text, and
// tool calls interleave, and lets `response.output_item.done` attach an
// encrypted-reasoning signature to the exact thinking block it belongs to.
// `kind` is recorded so a signature is only emitted against a real thinking
// block.
type BlockKind = "text" | "thinking" | "tool_call" | "refusal";
type ResponsesBlockIndexer = {
  nextIndex: number;
  items: Map<string, { index: number; kind: BlockKind }>;
};

export function createResponsesBlockIndexer(): ResponsesBlockIndexer {
  return { nextIndex: 0, items: new Map() };
}

function getOrAssignBlockIndex(
  state: ResponsesBlockIndexer,
  itemId: string,
  kind: BlockKind,
): number {
  const existing = state.items.get(itemId);
  if (existing !== undefined) return existing.index;
  const index = state.nextIndex;
  state.nextIndex += 1;
  state.items.set(itemId, { index, kind });
  return index;
}

function protocolMismatch(
  provider: string,
  message: string,
  raw: unknown,
): ProtocolMismatchError {
  return new ProtocolMismatchError(
    `${provider} parseResponse: ${message}`,
    raw,
  );
}

export function parseResponse(
  sseData: string,
  indexer: ResponsesBlockIndexer,
  source: LastCycleSource,
): InferenceEvent[] {
  const provider = source.provider;
  let parsed: unknown;
  try {
    parsed = JSON.parse(sseData);
  } catch (cause) {
    throw new ProtocolMismatchError(
      `${provider} parseResponse: malformed JSON in SSE data payload: ${cause instanceof Error ? cause.message : String(cause)}`,
      sseData,
    );
  }
  const envelope = ResponsesEnvelope(parsed);
  if (envelope instanceof type.errors) {
    throw protocolMismatch(
      provider,
      `SSE envelope failed schema validation: ${envelope.summary}`,
      parsed,
    );
  }
  const eventType = envelope.type;

  const seq = 0;
  const events: InferenceEvent[] = [];

  switch (eventType) {
    case "response.output_text.delta": {
      const validated = TextDeltaEvent(parsed);
      if (validated instanceof type.errors) {
        throw protocolMismatch(
          provider,
          `output_text.delta failed schema validation: ${validated.summary}`,
          parsed,
        );
      }
      if (validated.delta.length > 0) {
        events.push({
          type: "inference.text.delta",
          seq,
          data: {
            token: validated.delta,
            partial: EMPTY_PARTIAL,
            index: getOrAssignBlockIndex(indexer, validated.item_id, "text"),
          },
        });
      }
      return events;
    }
    case "response.reasoning_summary_text.delta":
    case "response.reasoning_text.delta": {
      const validated = ReasoningDeltaEvent(parsed);
      if (validated instanceof type.errors) {
        throw protocolMismatch(
          provider,
          `${eventType} failed schema validation: ${validated.summary}`,
          parsed,
        );
      }
      // Always register the block and emit a thinking delta (even for empty
      // tokens). This ensures a preceding thinking block exists for any
      // subsequent signature, supporting reasoning items whose visible
      // summary may be empty or delivered only via the done envelope.
      const index = getOrAssignBlockIndex(
        indexer,
        validated.item_id,
        "thinking",
      );
      events.push({
        type: "inference.thinking.delta",
        seq,
        data: { token: validated.delta, partial: EMPTY_PARTIAL, index },
      });
      return events;
    }
    case "response.refusal.delta": {
      const validated = RefusalDeltaEvent(parsed);
      if (validated instanceof type.errors) {
        throw protocolMismatch(
          provider,
          `refusal.delta failed schema validation: ${validated.summary}`,
          parsed,
        );
      }
      if (validated.delta.length > 0) {
        events.push({
          type: "inference.refusal.delta",
          seq,
          data: {
            token: validated.delta,
            partial: EMPTY_PARTIAL,
            index: getOrAssignBlockIndex(indexer, validated.item_id, "refusal"),
          },
        });
      }
      return events;
    }
    case "response.refusal.done": {
      const validated = RefusalDoneEvent(parsed);
      if (validated instanceof type.errors) {
        throw protocolMismatch(
          provider,
          `refusal.done failed schema validation: ${validated.summary}`,
          parsed,
        );
      }
      // The deltas already streamed the full text; the done envelope carries
      // no incremental payload this adapter needs.
      return events;
    }
    case "response.output_item.added": {
      const validated = OutputItemAddedEvent(parsed);
      if (validated instanceof type.errors) {
        throw protocolMismatch(
          provider,
          `output_item.added failed schema validation: ${validated.summary}`,
          parsed,
        );
      }
      const { item } = validated;
      if (item.type === "function_call") {
        // A function_call start missing id/call_id/name is not a lenient
        // partial to skip past — the harness would then see a
        // function_call_arguments.delta with no corresponding
        // inference.tool_call.start and silently drop the fragments.
        if (
          item.id === undefined ||
          item.call_id === undefined ||
          item.name === undefined
        ) {
          throw protocolMismatch(
            provider,
            "output_item.added function_call is missing id, call_id, or name",
            parsed,
          );
        }
        events.push({
          type: "inference.tool_call.start",
          seq,
          data: {
            callId: item.call_id,
            name: decodeToolName(item.name),
            partial: EMPTY_PARTIAL,
            index: getOrAssignBlockIndex(indexer, item.id, "tool_call"),
          },
        });
      } else if (item.type === "reasoning") {
        // A reasoning item with no id cannot be pre-registered, and the
        // signature it will later carry on output_item.done could never be
        // attached to a block — that's a protocol mismatch, not a skip.
        if (item.id === undefined) {
          throw protocolMismatch(
            provider,
            "output_item.added reasoning item is missing id",
            parsed,
          );
        }
        // Pre-register reasoning items on added so the index is stable even
        // if no text deltas follow (pure-encrypted case).
        const index = getOrAssignBlockIndex(indexer, item.id, "thinking");
        events.push({
          type: "inference.thinking.delta",
          seq,
          data: { token: "", partial: EMPTY_PARTIAL, index },
        });
      }
      return events;
    }
    case "response.output_item.done": {
      const validated = OutputItemDoneEvent(parsed);
      if (validated instanceof type.errors) {
        throw protocolMismatch(
          provider,
          `output_item.done failed schema validation: ${validated.summary}`,
          parsed,
        );
      }
      // Capture the encrypted reasoning blob (signature) so it can be echoed
      // back on the next turn. Required for multi-turn continuity when the
      // backend uses store:false + reasoning.encrypted_content. A thinking
      // block is ensured to exist (emitting an empty delta if this is the
      // first signal for the item) so the harness can attach the signature
      // without a protocol-mismatch error.
      const { item } = validated;
      if (item.type === "reasoning") {
        // Neither field present means the backend closed a reasoning item
        // this adapter can neither index (no id) nor carry forward (no
        // encrypted_content) — silently doing nothing here would drop a
        // reasoning turn's continuity with no signal anywhere.
        if (item.id === undefined && item.encrypted_content === undefined) {
          throw protocolMismatch(
            provider,
            "output_item.done reasoning item has neither id nor encrypted_content",
            parsed,
          );
        }
        if (item.encrypted_content !== undefined && item.id === undefined) {
          throw protocolMismatch(
            provider,
            "output_item.done reasoning item carries encrypted_content but is missing id",
            parsed,
          );
        }
        if (item.id !== undefined && item.encrypted_content !== undefined) {
          const itemId = item.id;
          const hadPrior = indexer.items.has(itemId);
          const index = getOrAssignBlockIndex(indexer, itemId, "thinking");
          if (!hadPrior) {
            events.push({
              type: "inference.thinking.delta",
              seq,
              data: { token: "", partial: EMPTY_PARTIAL, index },
            });
          }
          events.push({
            type: "inference.block.signature",
            seq,
            data: {
              signature: tagSignature(provider, item.encrypted_content),
              index,
            },
          });
        }
      }
      return events;
    }
    case "response.function_call_arguments.delta": {
      const validated = FunctionCallArgsDeltaEvent(parsed);
      if (validated instanceof type.errors) {
        throw protocolMismatch(
          provider,
          `function_call_arguments.delta failed schema validation: ${validated.summary}`,
          parsed,
        );
      }
      // The block must already have been registered by
      // `response.output_item.added`; an argument fragment for an item_id
      // this adapter never saw start is an orphan the harness has no
      // tool_call.start to route it against, not a fresh block to
      // synthesize silently.
      const existing = indexer.items.get(validated.item_id);
      if (existing === undefined) {
        throw protocolMismatch(
          provider,
          `function_call_arguments.delta for item_id ${validated.item_id} was never announced by output_item.added`,
          parsed,
        );
      }
      if (validated.delta.length > 0) {
        const blockIndex = existing.index;
        events.push({
          type: "inference.tool_call.delta",
          seq,
          // The harness routes argument fragments by a per-stream
          // placeholder keyed to the block index registered on the start
          // event.
          data: {
            callId: String(blockIndex),
            argumentFragment: validated.delta,
            partial: EMPTY_PARTIAL,
            index: blockIndex,
          },
        });
      }
      return events;
    }
    case "response.completed": {
      const validated = CompletedEvent(parsed);
      if (validated instanceof type.errors) {
        throw protocolMismatch(
          provider,
          `response.completed failed schema validation: ${validated.summary}`,
          parsed,
        );
      }
      if (validated.response.usage !== undefined) {
        events.push({
          type: "inference.usage",
          seq,
          data: { usage: toInferenceUsage(validated.response.usage), source },
        });
      }
      return events;
    }
    case "response.failed": {
      const validated = FailedEvent(parsed);
      if (validated instanceof type.errors) {
        throw protocolMismatch(
          provider,
          `response.failed failed schema validation: ${validated.summary}`,
          parsed,
        );
      }
      const message = validated.response?.error?.message ?? "response failed";
      throw new ProtocolMismatchError(`${provider}: ${message}`, parsed);
    }
    case "error": {
      const validated = ErrorEvent(parsed);
      if (validated instanceof type.errors) {
        throw protocolMismatch(
          provider,
          `error event failed schema validation: ${validated.summary}`,
          parsed,
        );
      }
      throw new ProtocolMismatchError(
        `${provider}: ${validated.message ?? "stream error"}`,
        parsed,
      );
    }
    default:
      // Lifecycle envelopes (response.created, response.in_progress,
      // content_part.*, *_text.done, response.incomplete) carry no
      // incremental payload a caller needs; ignore them. Unknown event
      // types are protocol-legal — the vocabulary is expected to grow.
      return events;
  }
}

// The Responses stream ends on a semantic lifecycle event, not `[DONE]` or a
// socket close: `response.completed` on success, `response.incomplete` when
// the backend truncates, `response.done` as an alias some backends emit.
// Failure envelopes (`response.failed`, `error`) already throw in
// `parseResponse`.
const RESPONSES_TERMINAL_EVENTS = new Set([
  "response.completed",
  "response.incomplete",
  "response.done",
]);

/** True when an SSE event payload is a Responses stream's terminal event. */
export function isResponsesStreamTerminal(sseData: string): boolean {
  let parsed: unknown;
  try {
    parsed = JSON.parse(sseData);
  } catch {
    // parseResponse re-parses the same payload and raises the protocol
    // error; reporting "not terminal" here defers to that single throw
    // site.
    return false;
  }
  const envelope = ResponsesEnvelope(parsed);
  if (envelope instanceof type.errors) return false;
  return RESPONSES_TERMINAL_EVENTS.has(envelope.type);
}

// ---------------------------------------------------------------------------
// Non-streaming JSON parse
//
// A complete non-streaming Responses object returns the whole turn in one
// JSON body. parseJSONResponse re-expresses it as the same InferenceEvent
// vocabulary parseResponse emits from the stream, walking `output[]` in
// order and synthesizing indices the same way the streaming path assigns
// them on first observation.
// ---------------------------------------------------------------------------

const OutputTextPart = type({ type: "'output_text'", text: "string" });
const RefusalPart = type({ type: "'refusal'", refusal: "string" });

// Content parts are validated individually by kind, same rationale as
// output items above: an "unknown kind, ignore it" fallback branch defeats
// narrowing on the known branches.
const MessageItem = type({
  type: "'message'",
  content: type({ type: "string" }).array(),
});
const FunctionCallItem = type({
  type: "'function_call'",
  id: "string",
  call_id: "string",
  name: "string",
  arguments: "string",
});
const ReasoningSummaryPart = type({ type: "'summary_text'", text: "string" });
const ReasoningItem = type({
  type: "'reasoning'",
  id: "string",
  "summary?": ReasoningSummaryPart.array(),
  "encrypted_content?": "string",
});

// Output items are validated individually by kind rather than as one
// discriminated array schema: a fallback "unknown item kind, ignore it"
// branch has a non-literal `type: string` discriminant, which defeats
// TypeScript's narrowing on every known branch. Validating per-item lets an
// unrecognized `type` skip validation entirely (protocol-legal, like an
// unrecognized SSE event) while a recognized `type` with the wrong shape
// still throws.
const CompleteResponse = type({
  output: type({ type: "string" }).array(),
  "usage?": ResponsesUsage,
  "status?":
    "'completed' | 'failed' | 'incomplete' | 'in_progress' | 'cancelled' | 'queued'",
  "error?": { "message?": "string" },
  "incomplete_details?": { "reason?": "string" },
});

export function parseJSONResponse(
  body: string,
  source: LastCycleSource,
): InferenceEvent[] {
  const provider = source.provider;
  let parsed: unknown;
  try {
    parsed = JSON.parse(body);
  } catch (cause) {
    throw new ProtocolMismatchError(
      `${provider} parseJSONResponse: malformed JSON response body: ${cause instanceof Error ? cause.message : String(cause)}`,
      body,
    );
  }

  const response = CompleteResponse(parsed);
  if (response instanceof type.errors) {
    throw new ProtocolMismatchError(
      `${provider} parseJSONResponse: response failed schema validation: ${response.summary}`,
      parsed,
    );
  }

  // A non-streaming response can report failure inside a 200 body instead of
  // an HTTP error code or an SSE `response.failed` event; decoding `output`
  // (typically empty) as a successful empty turn would hide the failure from
  // the caller entirely.
  if (response.status === "failed") {
    throw new ProtocolMismatchError(
      `${provider} parseJSONResponse: response status is "failed": ${response.error?.message ?? "no error message"}`,
      parsed,
    );
  }
  if (response.status === "incomplete") {
    throw new ProtocolMismatchError(
      `${provider} parseJSONResponse: response status is "incomplete": ${response.incomplete_details?.reason ?? "no reason given"}`,
      parsed,
    );
  }

  const seq = 0;
  const events: InferenceEvent[] = [];
  // A fresh indexer per body, synthesizing indices in output[] arrival
  // order — the same convention the SSE path uses on first observation of
  // each item id.
  let nextIndex = 0;

  for (const rawItem of response.output) {
    if (rawItem.type === "message") {
      const item = MessageItem(rawItem);
      if (item instanceof type.errors) {
        throw new ProtocolMismatchError(
          `${provider} parseJSONResponse: message output item failed schema validation: ${item.summary}`,
          rawItem,
        );
      }
      for (const rawPart of item.content) {
        if (rawPart.type === "output_text") {
          const part = OutputTextPart(rawPart);
          if (part instanceof type.errors) {
            throw new ProtocolMismatchError(
              `${provider} parseJSONResponse: output_text content part failed schema validation: ${part.summary}`,
              rawPart,
            );
          }
          if (part.text.length > 0) {
            events.push({
              type: "inference.text.delta",
              seq,
              data: {
                token: part.text,
                partial: EMPTY_PARTIAL,
                index: nextIndex,
              },
            });
            nextIndex += 1;
          }
        } else if (rawPart.type === "refusal") {
          const part = RefusalPart(rawPart);
          if (part instanceof type.errors) {
            throw new ProtocolMismatchError(
              `${provider} parseJSONResponse: refusal content part failed schema validation: ${part.summary}`,
              rawPart,
            );
          }
          if (part.refusal.length > 0) {
            events.push({
              type: "inference.refusal.delta",
              seq,
              data: {
                token: part.refusal,
                partial: EMPTY_PARTIAL,
                index: nextIndex,
              },
            });
            nextIndex += 1;
          }
        }
      }
    } else if (rawItem.type === "reasoning") {
      const item = ReasoningItem(rawItem);
      if (item instanceof type.errors) {
        throw new ProtocolMismatchError(
          `${provider} parseJSONResponse: reasoning output item failed schema validation: ${item.summary}`,
          rawItem,
        );
      }
      const summaryText = (item.summary ?? []).map((s) => s.text).join("");
      if (summaryText.length > 0 || item.encrypted_content !== undefined) {
        const index = nextIndex;
        nextIndex += 1;
        events.push({
          type: "inference.thinking.delta",
          seq,
          data: { token: summaryText, partial: EMPTY_PARTIAL, index },
        });
        if (item.encrypted_content !== undefined) {
          events.push({
            type: "inference.block.signature",
            seq,
            data: {
              signature: tagSignature(provider, item.encrypted_content),
              index,
            },
          });
        }
      }
    } else if (rawItem.type === "function_call") {
      const item = FunctionCallItem(rawItem);
      if (item instanceof type.errors) {
        throw new ProtocolMismatchError(
          `${provider} parseJSONResponse: function_call output item failed schema validation: ${item.summary}`,
          rawItem,
        );
      }
      const index = nextIndex;
      nextIndex += 1;
      events.push({
        type: "inference.tool_call.start",
        seq,
        data: {
          callId: item.call_id,
          name: decodeToolName(item.name),
          partial: EMPTY_PARTIAL,
          index,
        },
      });
      if (item.arguments.length > 0) {
        events.push({
          type: "inference.tool_call.delta",
          seq,
          data: {
            callId: String(index),
            argumentFragment: item.arguments,
            partial: EMPTY_PARTIAL,
            index,
          },
        });
      }
    }
    // Unknown output item kinds are protocol-legal extensions this adapter
    // has nothing to do with; ignored, mirroring the SSE default case.
  }

  events.push({
    type: "inference.usage",
    seq,
    data: { usage: toInferenceUsage(response.usage), source },
  });

  return events;
}
