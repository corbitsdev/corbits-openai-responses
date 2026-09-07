import { describe, expect, test } from "bun:test";
import { ProtocolMismatchError } from "@intx/inference";
import type {
  ConversationTurn,
  InferenceEvent,
  LastCycleSource,
} from "@intx/types/runtime";
import { createOpenAIResponsesAdapter, responsesAdapterFactory } from "./index";

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

describe("Responses adapter — protocol-native defaults", () => {
  test("stream:false requests accept application/json, not text/event-stream", () => {
    const adapter = createOpenAIResponsesAdapter(source, { stream: false });
    const request = adapter.buildRequest(turns, "gpt-x", {});
    expect(bodyOf(request)["stream"]).toBe(false);
    expect(request.headers["accept"]).toBe("application/json");
  });

  // An explicit `parallelToolCalls: false` must reach the wire verbatim —
  // some backends require it and reject a request that omits the field.
  test("an explicit parallelToolCalls:false reaches the wire as parallel_tool_calls:false", () => {
    const adapter = createOpenAIResponsesAdapter(source, {
      parallelToolCalls: false,
    });
    const body = bodyOf(adapter.buildRequest(turns, "m", {}));
    expect(body["parallel_tool_calls"]).toBe(false);
  });

  test("typed shape (the default) splits assistant text into output_text and never flattens to a string", () => {
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

  test("flat contentShape flattens text-only content to a plain string", () => {
    const adapter = createOpenAIResponsesAdapter(source, {
      contentShape: "flat",
    });
    const request = adapter.buildRequest(turns, "model", {});
    expect(inputItemsOf(request)[0]?.["content"]).toBe("hi");
  });
});

describe("Responses parser — event schema validation", () => {
  test("a function_call output_item.added missing id throws rather than silently dropping the start", () => {
    const adapter = createOpenAIResponsesAdapter(source, {});
    adapter.buildRequest(turns, "gpt-x", {});
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
    adapter.buildRequest(turns, "gpt-x", {});
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
    adapter.buildRequest(turns, "gpt-x", {});
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
    adapter.buildRequest(turns, "gpt-x", {});
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
    adapter.buildRequest(turns, "gpt-x", {});
    expect(() =>
      adapter.parseResponse(JSON.stringify({ item_id: "x", delta: "hi" })),
    ).toThrow(ProtocolMismatchError);
    expect(() => adapter.parseResponse(JSON.stringify(42))).toThrow(
      ProtocolMismatchError,
    );
  });

  test("output_item.added function_call with an empty-string id is a protocol mismatch, not accepted", () => {
    const adapter = createOpenAIResponsesAdapter(source, {});
    adapter.buildRequest(turns, "gpt-x", {});
    const added = JSON.stringify({
      type: "response.output_item.added",
      item: { type: "function_call", id: "", call_id: "", name: "" },
    });
    expect(() => adapter.parseResponse(added)).toThrow(ProtocolMismatchError);
  });

  test("ignores an unrecognized but well-formed event type rather than throwing", () => {
    const adapter = createOpenAIResponsesAdapter(source, {});
    adapter.buildRequest(turns, "gpt-x", {});
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
    adapter.buildRequest(turns, "gpt-x", {});
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
