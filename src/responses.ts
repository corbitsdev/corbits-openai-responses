import { type } from "arktype";

import {
  type AdapterFactory,
  type ProviderAdapter,
  type RequestBuilder,
  type ResponseParser,
} from "@intx/inference";
import type { InferenceEvent, LastCycleSource } from "@intx/types/runtime";

import { buildResponsesRequest } from "./protocol/body.js";
import {
  createResponsesBlockIndexer,
  isResponsesStreamTerminal,
  parseJSONResponse,
  parseResponse,
} from "./protocol/iterator.js";

// ---------------------------------------------------------------------------
// Quirks
// ---------------------------------------------------------------------------
//
// Everything a vendor (Codex, xAI/Grok, plain OpenAI, ...) needs to bend the
// shared Responses wire protocol to its own backend, expressed as data
// instead of a fork of the adapter. Every field here must be JSON-safe: the
// quirks bag rides inside an `InferenceSource`, which is persisted and sent
// over the wire, so a function-valued field would silently fail to survive
// that round trip. Code-shaped accommodations live in `ResponsesHooks`
// instead — see `responsesAdapterFactory` below.

const HeaderFromOption = type({
  optionKey: "string",
  header: "string",
});
type HeaderFromOption = typeof HeaderFromOption.infer;

const ResponsesHeaders = type({
  "static?": { "[string]": "string" },
  "modelHeader?": "string",
  "fromOption?": HeaderFromOption.array(),
});

// `shape: "string"` sends plain string content (Grok/OpenAI); `"parts"`
// sends a single `input_text` content part (Codex, which addresses the item
// as a `developer` message).
const SystemPromptPlacement = type({
  role: "'system' | 'developer'",
  shape: "'string' | 'parts'",
});
type SystemPromptPlacement = typeof SystemPromptPlacement.infer;

const ReasoningQuirks = type({
  "summary?": "'auto' | 'detailed'",
  "effortOption?": "string",
});
type ReasoningQuirks = typeof ReasoningQuirks.infer;

// Per-source accommodations for the OpenAI Responses API wire protocol.
// Every field is optional; an absent field resolves to the strict protocol
// default, so a source that supplies no quirks gets no accommodation and
// must opt into lenient behavior explicitly.
export const ResponsesQuirks = type({
  "path?": "string",
  "headers?": ResponsesHeaders,
  "sessionIdOption?": "string",
  "sessionIdHeader?": "string",
  "systemPrompt?": SystemPromptPlacement,
  // `typed` is the protocol-native shape (`output_text` for assistant items,
  // `input_text` otherwise); `flat` collapses text-only content to a plain
  // string for backends that reject typed parts.
  "contentShape?": "'typed' | 'flat'",
  // Sent verbatim when present; absent omits the field so the backend's own
  // default applies. Some backends require an explicit `false`.
  "parallelToolCalls?": "boolean",
  // The Responses protocol forwards a caller-supplied maxTokens/temperature
  // verbatim; these default true. Set `false` only when a vendor's backend
  // rejects the field outright (the quirk opts a source OUT of protocol-
  // native forwarding, not into it).
  "maxOutputTokens?": "boolean",
  "temperature?": "boolean",
  "store?": "boolean",
  "stream?": "boolean",
  "reasoning?": ReasoningQuirks,
  "instructions?": "string",
  // Reject unknown keys so a mistyped quirk name fails loudly at
  // construction rather than being silently ignored and running with
  // default behavior.
  "+": "reject",
});
export type ResponsesQuirks = typeof ResponsesQuirks.infer;

// Quirks resolved to concrete values at the factory edge, so interior code
// never re-decides a default.
export type ResolvedResponsesQuirks = {
  path: string;
  headers: {
    static: Record<string, string>;
    modelHeader: string | undefined;
    fromOption: HeaderFromOption[];
  };
  sessionIdOption: string | undefined;
  sessionIdHeader: string | undefined;
  systemPrompt: SystemPromptPlacement;
  contentShape: "typed" | "flat";
  parallelToolCalls: boolean | undefined;
  maxOutputTokens: boolean;
  temperature: boolean;
  store: boolean;
  stream: boolean;
  reasoning: ReasoningQuirks | undefined;
  instructions: string | undefined;
};

// Parses an adapter's `quirks` argument (an `InferenceSource.quirks` bag)
// into a validated `ResponsesQuirks` and fills in defaults. An absent bag
// (`undefined`) resolves the same as `{}` — protocol-native defaults, not a
// thrown error. Throws on shape violations rather than silently ignoring an
// unrecognized or misspelled field.
function parseResponsesQuirks(raw: unknown): ResolvedResponsesQuirks {
  const validated = ResponsesQuirks(raw ?? {});
  if (validated instanceof type.errors) {
    throw new Error(
      `openai-responses adapter: invalid quirks: ${validated.summary}`,
    );
  }
  return {
    path: validated.path ?? "/responses",
    headers: {
      static: validated.headers?.static ?? {},
      modelHeader: validated.headers?.modelHeader,
      fromOption: validated.headers?.fromOption ?? [],
    },
    sessionIdOption: validated.sessionIdOption,
    sessionIdHeader: validated.sessionIdHeader,
    // Protocol-native behaviour always sends the caller's system prompt as a
    // plain `system` message; a vendor whose backend addresses it
    // differently (Codex: `developer` role, parts shape) overrides this
    // quirk instead of opting out of sending it at all.
    systemPrompt: validated.systemPrompt ?? {
      role: "system",
      shape: "string",
    },
    contentShape: validated.contentShape ?? "typed",
    parallelToolCalls: validated.parallelToolCalls,
    maxOutputTokens: validated.maxOutputTokens ?? true,
    temperature: validated.temperature ?? true,
    store: validated.store ?? false,
    stream: validated.stream ?? true,
    reasoning: validated.reasoning,
    instructions: validated.instructions,
  };
}

/**
 * TS-only composition hooks a caller can bake into a factory-built adapter.
 * Never JSON, never part of the quirks bag: quirks ride inside an
 * `InferenceSource`, which is persisted and sent over the wire, so a
 * function-valued field would not survive that round trip.
 */
export type ResponsesHooks = {
  wrapSystemPrompt?: (systemPrompt: string) => string;
  includeReasoningEffort?: (effort: string) => boolean;
};

// ---------------------------------------------------------------------------
// Header extractors
//
// Duplicated (not imported) from Interchange's OpenAI Chat Completions
// adapter (`packages/inference/src/providers/openai.ts`, read-only
// reference): these read OpenAI-family rate-limit headers, which the
// Responses surface shares byte-for-byte, but the functions are module-
// private there and not part of `@intx/inference`'s public surface.
// ---------------------------------------------------------------------------

function extractRetryAfterMs(headers: Headers): number | undefined {
  // OpenAI's non-standard millisecond header takes priority
  const retryMs = headers.get("retry-after-ms");
  if (retryMs !== null) {
    const ms = Number(retryMs);
    if (Number.isFinite(ms) && ms > 0) return Math.ceil(ms);
  }
  const raw = headers.get("retry-after");
  if (raw !== null) {
    const seconds = Number(raw);
    if (Number.isFinite(seconds) && seconds > 0) {
      return Math.ceil(seconds * 1000);
    }
  }
  return undefined;
}

function parseDuration(value: string): number | undefined {
  let total = 0;
  const pattern = /(\d+(?:\.\d+)?)(ms|s|m|h)/g;
  let match;
  while ((match = pattern.exec(value)) !== null) {
    const num = Number(match[1]);
    switch (match[2]) {
      case "ms":
        total += num;
        break;
      case "s":
        total += num * 1000;
        break;
      case "m":
        total += num * 60_000;
        break;
      case "h":
        total += num * 3_600_000;
        break;
    }
  }
  return total > 0 ? Math.ceil(total) : undefined;
}

function extractPacingDelayMs(headers: Headers): number | undefined {
  const remaining = headers.get("x-ratelimit-remaining-requests");
  if (remaining === null) return undefined;
  const n = Number(remaining);
  if (!Number.isFinite(n) || n > 0) return undefined;

  const reset = headers.get("x-ratelimit-reset-requests");
  if (reset === null) return undefined;
  const ms = parseDuration(reset);
  return ms !== undefined && ms > 0 ? ms : undefined;
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

function buildAdapter(
  source: LastCycleSource,
  quirks: ResolvedResponsesQuirks,
  hooks: ResponsesHooks | undefined,
): ProviderAdapter {
  // Re-created per request in buildRequest, not just once here — otherwise
  // block indices accumulate across every request this adapter instance
  // ever serves, growing the map for the life of the conversation.
  let indexer = createResponsesBlockIndexer();
  const buildRequest: RequestBuilder = (messages, model, options) => {
    indexer = createResponsesBlockIndexer();
    return buildResponsesRequest(
      quirks,
      messages,
      model,
      options,
      source.provider,
      hooks,
    );
  };
  const parseResponseFromStream: ResponseParser = (sseData) =>
    parseResponse(sseData, indexer, source);
  const parseCompleteResponse: (responseBody: string) => InferenceEvent[] = (
    responseBody,
  ) => parseJSONResponse(responseBody, source);
  // Declared as a plain const rather than a typed object literal: the
  // `ProviderAdapter` type in `@intx/inference` 0.4.0 has no
  // `isStreamTerminal` field, so a literal typed as `ProviderAdapter` here
  // would fail an excess-property check. Returning a widened variable
  // instead lets a host running the semantic-terminal harness patch read
  // `isStreamTerminal` off the returned value while everyone else's
  // `ProviderAdapter`-typed usage ignores the extra field.
  const adapter = {
    buildRequest,
    parseResponse: parseResponseFromStream,
    parseJSONResponse: parseCompleteResponse,
    extractRetryAfterMs,
    extractPacingDelayMs,
    isStreamTerminal: isResponsesStreamTerminal,
  };
  return adapter;
}

/**
 * `AdapterFactory` for the OpenAI Responses API wire protocol. `quirks` is
 * this package's own {@link ResponsesQuirks} (an `InferenceSource.quirks`
 * bag) describing exactly how this vendor's backend deviates from the
 * shared protocol — headers, system-prompt placement, session-id routing,
 * and the handful of request-body switches that differ between backends. An
 * absent `quirks` bag resolves the same as `{}`: protocol-native defaults,
 * not an error.
 */
export const createOpenAIResponsesAdapter: AdapterFactory = (
  source: LastCycleSource,
  quirks?: unknown,
): ProviderAdapter =>
  buildAdapter(source, parseResponsesQuirks(quirks), undefined);

/**
 * Curries a fixed {@link ResponsesQuirks} bag into a plain `AdapterFactory`,
 * so a provider package can bake its vendor's config in once and hand
 * Interchange a factory that takes only `(source)` — no quirks bag needed at
 * the call site.
 *
 * `hooks` carries the handful of accommodations that must be code rather
 * than data (wrapping a system prompt in vendor-specific markup, deciding
 * whether a reasoning effort value is safe to forward for a given model).
 * They are TypeScript-only and deliberately never part of `quirks`: quirks
 * ride inside an `InferenceSource`, which is JSON — a function-valued field
 * would not survive being persisted or sent over the wire, and a caller with
 * a genuine code-shaped need reaches for `hooks` instead of smuggling a
 * closure into a bag that has to stay serializable.
 */
export function responsesAdapterFactory(
  quirks: ResponsesQuirks,
  hooks?: ResponsesHooks,
): AdapterFactory {
  const resolved = parseResponsesQuirks(quirks);
  return (source: LastCycleSource): ProviderAdapter =>
    buildAdapter(source, resolved, hooks);
}
