import { describe, expect, test } from "bun:test";
import { ProtocolMismatchError } from "@intx/inference";
import type {
  ConversationTurn,
  InferenceEvent,
  LastCycleSource,
} from "@intx/types/runtime";
import {
  createOpenAIResponsesAdapter,
  responsesAdapterFactory,
  type ResponsesQuirks,
} from "./index";

const source: LastCycleSource = {
  sourceId: "test/source",
  provider: "test-provider",
  model: "model",
};
const turns: ConversationTurn[] = [
  { role: "user", timestamp: 0, content: [{ type: "text", text: "hi" }] },
];

// Inline narrowing utilities, following the `pickFirst*` pattern upstream
// uses in `providers/anthropic.test.ts`: each exhausts its failure modes
// (empty result, wrong type/shape) with a descriptive throw, leaving the body
// of a test free of `as` casts.

function isRecord(x: unknown): x is Record<string, unknown> {
  return typeof x === "object" && x !== null;
}

function bodyOf(request: { body: string }): Record<string, unknown> {
  const parsed: unknown = JSON.parse(request.body);
  if (!isRecord(parsed)) throw new Error("request body is not a JSON object");
  return parsed;
}

function inputItemsOf(request: { body: string }): Record<string, unknown>[] {
  const input = bodyOf(request)["input"];
  if (!Array.isArray(input))
    throw new Error("expected body.input to be an array");
  return input.map((item) => {
    if (!isRecord(item))
      throw new Error("expected every input item to be an object");
    return item;
  });
}

function pickFirstTextDelta(
  events: InferenceEvent[],
): Extract<InferenceEvent, { type: "inference.text.delta" }> {
  const ev = events[0];
  if (ev === undefined) throw new Error("expected at least one event");
  if (ev.type !== "inference.text.delta")
    throw new Error(`expected inference.text.delta, got ${ev.type}`);
  return ev;
}

describe("Responses adapter — block-index lifecycle", () => {
  test("resets block indices on a new buildRequest instead of accumulating across requests", () => {
    const adapter = createOpenAIResponsesAdapter(source, {
      contentShape: "flat",
    });

    adapter.buildRequest(turns, "model", {});
    adapter.parseResponse(
      JSON.stringify({
        type: "response.output_text.delta",
        item_id: "item_1",
        delta: "a",
      }),
    );
    adapter.parseResponse(
      JSON.stringify({
        type: "response.output_text.delta",
        item_id: "item_2",
        delta: "b",
      }),
    );

    // A new HTTP round trip with a brand-new item id should start indexing
    // from 0 again, not continue accumulating the prior request's indexer
    // state.
    adapter.buildRequest(turns, "model", {});
    const secondRequestDelta = adapter.parseResponse(
      JSON.stringify({
        type: "response.output_text.delta",
        item_id: "item_3",
        delta: "c",
      }),
    );
    expect(pickFirstTextDelta(secondRequestDelta).data.index).toBe(0);
  });

  test("rejects an unrecognized quirks field instead of ignoring it", () => {
    expect(() =>
      createOpenAIResponsesAdapter(source, {
        contentShape: "flat",
        bogus: true,
      }),
    ).toThrow();
  });
});

describe("Responses request builder — quirk matrix", () => {
  const options = { maxTokens: 100, temperature: 0.5 };
  const cases: {
    name: string;
    quirks: ResponsesQuirks;
    present: Record<string, unknown>;
    absent: string[];
    accept: string;
  }[] = [
    {
      name: "no quirks sends protocol-native defaults",
      quirks: {},
      present: {
        store: false,
        stream: true,
        max_output_tokens: 100,
        temperature: 0.5,
        input: [{ content: [{ type: "input_text", text: "hi" }] }],
      },
      absent: ["parallel_tool_calls", "instructions"],
      accept: "text/event-stream",
    },
    {
      name: "stream:false asks for a JSON body",
      quirks: { stream: false },
      present: { stream: false },
      absent: [],
      accept: "application/json",
    },
    {
      name: "store:true is sent verbatim",
      quirks: { store: true },
      present: { store: true },
      absent: [],
      accept: "text/event-stream",
    },
    {
      // Some backends reject a request that omits the field.
      name: "parallelToolCalls:false is sent verbatim",
      quirks: { parallelToolCalls: false },
      present: { parallel_tool_calls: false },
      absent: [],
      accept: "text/event-stream",
    },
    {
      name: "maxOutputTokens:false and temperature:false omit their fields",
      quirks: { maxOutputTokens: false, temperature: false },
      present: {},
      absent: ["max_output_tokens", "temperature"],
      accept: "text/event-stream",
    },
    {
      name: "instructions is sent verbatim",
      quirks: { instructions: "be terse" },
      present: { instructions: "be terse" },
      absent: [],
      accept: "text/event-stream",
    },
    {
      name: "flat contentShape flattens text-only content to a string",
      quirks: { contentShape: "flat" },
      present: { input: [{ content: "hi" }] },
      absent: [],
      accept: "text/event-stream",
    },
  ];

  test.each(cases)("$name", ({ quirks, present, absent, accept }) => {
    const adapter = createOpenAIResponsesAdapter(source, quirks);
    const request = adapter.buildRequest(turns, "model", options);
    const body = bodyOf(request);
    expect(body).toMatchObject(present);
    for (const key of absent) expect(body).not.toHaveProperty(key);
    expect(request.headers["accept"]).toBe(accept);
  });

  test("static headers override defaults and per-request headers override static ones", () => {
    const adapter = createOpenAIResponsesAdapter(source, {
      headers: {
        static: { accept: "application/x-ndjson", "x-model": "static" },
        modelHeader: "x-model",
      },
    });
    const { headers } = adapter.buildRequest(turns, "m", {});
    expect(headers["accept"]).toBe("application/x-ndjson");
    expect(headers["x-model"]).toBe("m");
  });

  test("typed shape (the default) splits assistant text into output_text", () => {
    const adapter = createOpenAIResponsesAdapter(source, {});
    const assistantTurn: ConversationTurn[] = [
      {
        role: "assistant",
        timestamp: 0,
        content: [{ type: "text", text: "hi" }],
      },
    ];
    const request = adapter.buildRequest(assistantTurn, "model", {});
    expect(inputItemsOf(request)[0]?.["content"]).toEqual([
      { type: "output_text", text: "hi" },
    ]);
  });
});

describe("Responses parser — event schema validation", () => {
  test("a function_call output_item.added missing id throws rather than silently dropping the start", () => {
    const adapter = createOpenAIResponsesAdapter(source, {});
    adapter.buildRequest(turns, "model", {});
    const added = JSON.stringify({
      type: "response.output_item.added",
      item: { type: "function_call", call_id: "call_1", name: "shell" },
    });
    expect(() => adapter.parseResponse(added)).toThrow(ProtocolMismatchError);
  });

  // A reasoning item that later carries a signature must have been
  // registered with an id on `added`; an id-less reasoning item is a
  // protocol mismatch, not a block silently skipped.
  test("output_item.added reasoning item missing id throws instead of skipping registration", () => {
    const adapter = createOpenAIResponsesAdapter(source, {});
    adapter.buildRequest(turns, "model", {});
    const added = JSON.stringify({
      type: "response.output_item.added",
      item: { type: "reasoning" },
    });
    expect(() => adapter.parseResponse(added)).toThrow(ProtocolMismatchError);
  });

  // The prior behavior dropped the encrypted signature silently when the
  // reasoning item carried no id; that left a reasoning turn's replayable
  // ciphertext lost with no error raised anywhere.
  test("output_item.done reasoning with encrypted_content but no id throws instead of dropping the signature", () => {
    const adapter = createOpenAIResponsesAdapter(source, {});
    adapter.buildRequest(turns, "model", {});
    const done = JSON.stringify({
      type: "response.output_item.done",
      item: { type: "reasoning", encrypted_content: "CIPHER" },
    });
    expect(() => adapter.parseResponse(done)).toThrow(ProtocolMismatchError);
  });

  // A reasoning item closing with NEITHER id nor encrypted_content is not a
  // benign no-op either: the adapter cannot index it and cannot carry its
  // ciphertext forward, so it must throw rather than silently doing nothing.
  test("output_item.done reasoning with neither id nor encrypted_content throws instead of being skipped", () => {
    const adapter = createOpenAIResponsesAdapter(source, {});
    adapter.buildRequest(turns, "model", {});
    const done = JSON.stringify({
      type: "response.output_item.done",
      item: { type: "reasoning" },
    });
    expect(() => adapter.parseResponse(done)).toThrow(ProtocolMismatchError);
  });

  // An SSE payload that fails envelope validation (no `type` field, or a
  // non-object payload) is a genuine protocol mismatch, not a payload to
  // quietly ignore — only a well-formed envelope with an unrecognized `type`
  // is protocol-legal to skip.
  test("an SSE payload with no `type` field throws a protocol mismatch instead of being ignored", () => {
    const adapter = createOpenAIResponsesAdapter(source, {});
    adapter.buildRequest(turns, "model", {});
    expect(() =>
      adapter.parseResponse(JSON.stringify({ item_id: "x", delta: "hi" })),
    ).toThrow(ProtocolMismatchError);
    expect(() => adapter.parseResponse(JSON.stringify(42))).toThrow(
      ProtocolMismatchError,
    );
  });

  test("output_item.added function_call with an empty-string id is a protocol mismatch, not accepted", () => {
    const adapter = createOpenAIResponsesAdapter(source, {});
    adapter.buildRequest(turns, "model", {});
    const added = JSON.stringify({
      type: "response.output_item.added",
      item: { type: "function_call", id: "", call_id: "", name: "" },
    });
    expect(() => adapter.parseResponse(added)).toThrow(ProtocolMismatchError);
  });

  test("ignores an unrecognized but well-formed event type rather than throwing", () => {
    const adapter = createOpenAIResponsesAdapter(source, {});
    adapter.buildRequest(turns, "model", {});
    expect(
      adapter.parseResponse(
        JSON.stringify({ type: "response.some_future_event" }),
      ),
    ).toEqual([]);
  });

  // A function_call_arguments.delta is only routable to the tool_call.start
  // the harness already saw; one for an item_id that never arrived via
  // output_item.added is an orphan fragment, not a fresh block to
  // synthesize silently.
  test("a function_call_arguments.delta for an item_id never announced by output_item.added throws", () => {
    const adapter = createOpenAIResponsesAdapter(source, {});
    adapter.buildRequest(turns, "model", {});
    const orphanDelta = JSON.stringify({
      type: "response.function_call_arguments.delta",
      item_id: "never_announced",
      delta: '{"cmd":"ls"}',
    });
    expect(() => adapter.parseResponse(orphanDelta)).toThrow(
      ProtocolMismatchError,
    );
  });
});

describe("Responses parser — non-streaming failure states", () => {
  test("throws instead of decoding an empty successful turn on status:failed", () => {
    const adapter = createOpenAIResponsesAdapter(source, {});
    const failed = JSON.stringify({
      id: "resp_1",
      object: "response",
      status: "failed",
      error: { code: "server_error", message: "boom" },
      output: [],
      usage: null,
    });
    expect(() => adapter.parseJSONResponse(failed)).toThrow(
      ProtocolMismatchError,
    );
  });

  test("throws on status:incomplete", () => {
    const adapter = createOpenAIResponsesAdapter(source, {});
    const incomplete = JSON.stringify({
      output: [],
      status: "incomplete",
      incomplete_details: { reason: "max_output_tokens" },
      usage: null,
    });
    expect(() => adapter.parseJSONResponse(incomplete)).toThrow(
      ProtocolMismatchError,
    );
  });

  test("throws on a response missing the required output field", () => {
    const adapter = createOpenAIResponsesAdapter(source, {});
    expect(() =>
      adapter.parseJSONResponse(JSON.stringify({ usage: {} })),
    ).toThrow(ProtocolMismatchError);
  });
});

describe("Responses request builder — tool-item dedupe", () => {
  // A duplicate call_id is most often a corrected retry; keeping the first
  // occurrence instead of the last would silently replay a stale tool
  // call/result the model already moved past.
  test("keeps the latest function_call_output on a duplicate call_id", () => {
    const adapter = createOpenAIResponsesAdapter(source, {
      contentShape: "flat",
    });
    const duplicated: ConversationTurn[] = [
      {
        role: "user",
        timestamp: 0,
        content: [
          {
            type: "tool_result",
            callId: "call_1",
            content: [{ type: "text", text: "stale" }],
          },
          {
            type: "tool_result",
            callId: "call_1",
            content: [{ type: "text", text: "fresh" }],
          },
        ],
      },
    ];
    const request = adapter.buildRequest(duplicated, "model", {});
    const outputs = inputItemsOf(request).filter(
      (item) => item["type"] === "function_call_output",
    );
    expect(outputs).toHaveLength(1);
    expect(outputs[0]?.["output"]).toBe("fresh");
  });
});

describe("Responses request builder — tool-name codec", () => {
  // The Responses function-name charset is a hard 64-char wire limit; an
  // over-length or non-conforming internal tool name must be encoded, not
  // sent verbatim and rejected by the backend.
  test("encodes a non-wire-safe tool name on the outgoing function tool definition", () => {
    const adapter = createOpenAIResponsesAdapter(source, {});
    const request = adapter.buildRequest(turns, "model", {
      tools: [
        {
          name: "@intx/tools-posix/sidecar-bundle:run_shell",
          description: "run",
          inputSchema: {},
        },
      ],
    });
    const tools = bodyOf(request)["tools"];
    if (!Array.isArray(tools))
      throw new Error("expected body.tools to be an array");
    const [tool] = tools;
    if (!isRecord(tool) || typeof tool["name"] !== "string")
      throw new Error("expected a named tool definition");
    expect(tool["name"]).toMatch(/^[A-Za-z0-9_-]{1,64}$/);
    expect(tool["name"]).not.toBe("@intx/tools-posix/sidecar-bundle:run_shell");
  });
});

describe("responsesAdapterFactory — hooks composition", () => {
  test("applies a wrapSystemPrompt hook without the hook ever riding in the quirks bag", () => {
    const factory = responsesAdapterFactory(
      {
        contentShape: "flat",
        systemPrompt: { role: "system", shape: "string" },
      },
      { wrapSystemPrompt: (s) => `<wrapped>${s}</wrapped>` },
    );
    const adapter = factory(source);
    const request = adapter.buildRequest(turns, "model", {
      systemPrompt: "be nice",
    });
    expect(inputItemsOf(request)[0]?.["content"]).toBe(
      "<wrapped>be nice</wrapped>",
    );
  });
});

describe("Responses parser — usage mapping", () => {
  type UsageEvent = Extract<InferenceEvent, { type: "inference.usage" }>;

  // Narrow the emitted `inference.usage` event, following this file's
  // pickFirst* pattern: exhaust failure modes with a descriptive throw.
  function usageEventOf(events: InferenceEvent[]): UsageEvent {
    const event = events.find(
      (e): e is UsageEvent => e.type === "inference.usage",
    );
    if (event === undefined)
      throw new Error("expected an inference.usage event");
    return event;
  }

  // The `response.completed` handler reads no request-scoped state, so no
  // buildRequest priming is needed — usage mapping starts from a fresh
  // adapter every time.
  function usageFromCompleted(usage: unknown): UsageEvent["data"]["usage"] {
    const adapter = createOpenAIResponsesAdapter(source, {});
    const events = adapter.parseResponse(
      JSON.stringify({ type: "response.completed", response: { usage } }),
    );
    return usageEventOf(events).data.usage;
  }

  // A gateway fronting an OpenAI-shaped endpoint reports the
  // Anthropic-shaped `cache_creation_tokens`, whose subset relationship to
  // `input_tokens` is unobservable from the client; that path keeps the
  // historic behavior of not reducing input.
  test("gateway-shaped cache_creation_tokens maps to cacheWrite without reducing input", () => {
    const usage = usageFromCompleted({
      input_tokens: 1000,
      input_tokens_details: { cached_tokens: 100, cache_creation_tokens: 50 },
    });
    expect(usage.cacheWrite).toBe(50);
    expect(usage.cacheRead).toBe(100);
    expect(usage.input).toBe(900);
  });

  test("usage without a cache-write field reports cacheWrite zero", () => {
    const usage = usageFromCompleted({
      input_tokens: 500,
      input_tokens_details: { cached_tokens: 500 },
    });
    expect(usage.cacheWrite).toBe(0);
    expect(usage.cacheRead).toBe(500);
    expect(usage.input).toBe(0);
  });

  // When both field names appear, the OpenAI-native one wins for both the
  // reported value and the input subtraction — the two are one linkage,
  // not independent choices.
  test("both write fields present prefers cache_write_tokens for value and input split", () => {
    const usage = usageFromCompleted({
      input_tokens: 2600,
      input_tokens_details: {
        cached_tokens: 2000,
        cache_write_tokens: 400,
        cache_creation_tokens: 999,
      },
    });
    expect(usage.cacheWrite).toBe(400);
    expect(usage.input).toBe(200);
  });

  // An explicit zero is a real reported value, not an omission: the
  // preference chain is nullish (`??`), so `cache_write_tokens: 0` must
  // suppress the `cache_creation_tokens` fallback rather than fall through
  // to it (a rewrite to `||` would silently report the fallback instead).
  test("explicit cache_write_tokens zero wins over a nonzero cache_creation_tokens", () => {
    const usage = usageFromCompleted({
      input_tokens: 100,
      input_tokens_details: {
        cache_write_tokens: 0,
        cache_creation_tokens: 77,
      },
    });
    expect(usage).toEqual({
      input: 100,
      output: 0,
      cacheRead: 0,
      cacheWrite: 0,
      thinking: 0,
    });
  });

  // A backend reporting subset counts larger than the total must clamp at
  // zero rather than emit a negative input that corrupts downstream sums.
  test("write tokens exceeding input clamp input at zero, not negative", () => {
    const usage = usageFromCompleted({
      input_tokens: 300,
      input_tokens_details: { cached_tokens: 100, cache_write_tokens: 400 },
    });
    expect(usage).toEqual({
      input: 0,
      output: 0,
      cacheRead: 100,
      cacheWrite: 400,
      thinking: 0,
    });
  });

  // The schema accepts the whole details object as null (the SSE envelope's
  // own convention), which must behave like an absent object.
  test("null input_tokens_details yields zero cache fields with input preserved", () => {
    const usage = usageFromCompleted({
      input_tokens: 250,
      output_tokens: 5,
      input_tokens_details: null,
    });
    expect(usage).toEqual({
      input: 250,
      output: 5,
      cacheRead: 0,
      cacheWrite: 0,
      thinking: 0,
    });
  });

  // A completed frame may carry `usage: null`; the zeroed event must still
  // be emitted so downstream usage accounting sees the turn.
  test("null usage emits a zeroed usage event", () => {
    const usage = usageFromCompleted(null);
    expect(usage).toEqual({
      input: 0,
      output: 0,
      cacheRead: 0,
      cacheWrite: 0,
      thinking: 0,
    });
  });

  // Usage counters are non-negative quantities. A backend reporting a
  // negative write count is wire garbage: report zero rather than
  // propagate it into sums, and never let subtracting a negative inflate
  // input. A clamped zero still counts as an explicit value and suppresses
  // the gateway fallback, matching the explicit-zero case above.
  test("negative cache_write_tokens is clamped to zero without inflating input", () => {
    const usage = usageFromCompleted({
      input_tokens: 100,
      input_tokens_details: {
        cache_write_tokens: -5,
        cache_creation_tokens: 77,
      },
    });
    expect(usage).toEqual({
      input: 100,
      output: 0,
      cacheRead: 0,
      cacheWrite: 0,
      thinking: 0,
    });
  });

  // The final cacheWrite clamp above also covers a negative gateway
  // fallback; pin that by name so a future refactor of the chain cannot
  // reintroduce a negative report on the gateway path alone.
  test("negative cache_creation_tokens reports cacheWrite zero on the gateway path", () => {
    const usage = usageFromCompleted({
      input_tokens: 100,
      input_tokens_details: { cache_creation_tokens: -9 },
    });
    expect(usage).toEqual({
      input: 100,
      output: 0,
      cacheRead: 0,
      cacheWrite: 0,
      thinking: 0,
    });
  });

  // Every counter negative at once: each must clamp independently so no
  // sum goes negative and no subtraction of a negative inflates input.
  test("all-negative counters clamp to zero without distorting any field", () => {
    const usage = usageFromCompleted({
      input_tokens: 100,
      output_tokens: -3,
      input_tokens_details: { cached_tokens: -20, cache_creation_tokens: -5 },
      output_tokens_details: { reasoning_tokens: -2 },
    });
    expect(usage).toEqual({
      input: 100,
      output: 0,
      cacheRead: 0,
      cacheWrite: 0,
      thinking: 0,
    });
  });

  // All five counters populated at once: each TokenUsage field must come
  // from its own wire field, with no cross-wiring between them.
  test("a fully populated usage object maps every field without cross-wiring", () => {
    const usage = usageFromCompleted({
      input_tokens: 3000,
      output_tokens: 25,
      input_tokens_details: { cached_tokens: 1500, cache_write_tokens: 500 },
      output_tokens_details: { reasoning_tokens: 12 },
    });
    expect(usage).toEqual({
      input: 1000,
      output: 25,
      cacheRead: 1500,
      cacheWrite: 500,
      thinking: 12,
    });
  });

  // The non-streaming path shares the usage schema and mapping with SSE,
  // but a future split between the two would silently lose the subset
  // arithmetic; pin the same OpenAI shape through parseJSONResponse.
  test("non-streaming response maps cache_write_tokens identically", () => {
    const adapter = createOpenAIResponsesAdapter(source, {});
    const events = adapter.parseJSONResponse(
      JSON.stringify({
        status: "completed",
        output: [],
        usage: {
          input_tokens: 2600,
          input_tokens_details: {
            cached_tokens: 2000,
            cache_write_tokens: 400,
          },
        },
      }),
    );
    expect(usageEventOf(events).data.usage).toEqual({
      input: 200,
      output: 0,
      cacheRead: 2000,
      cacheWrite: 400,
      thinking: 0,
    });
  });
});
