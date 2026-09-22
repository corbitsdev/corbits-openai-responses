// End-to-end proof through production `runInference` with scripted Responses
// SSE wire bytes, in the style @intx/inference-testing's README prescribes.
// The adapter is exercised only through its public factory — no internal
// module is imported here.

import { afterEach, describe, expect, test } from "bun:test";
import { type } from "arktype";
import type { AdapterRegistry } from "@intx/inference";
import { setupHarness, type Harness } from "@intx/inference-testing";
import type {
  ConversationTurn,
  InferenceEvent,
  InferenceSource,
} from "@intx/types/runtime";
import type { CredentialMaterial } from "@intx/types";
import {
  createOpenAIResponsesAdapter,
  OPENAI_RESPONSES_PROVIDER,
} from "./index";

// Narrows the replayed request body for the reasoning-replay assertion below
// without an `as` cast on data that came off a real `Request`.
const ReplayedRequestBody = type({
  input: type({
    type: "string",
    "encrypted_content?": "string",
    "call_id?": "string",
  }).array(),
});

const utf8 = (s: string): Uint8Array => new TextEncoder().encode(s);
const sse = (event: Record<string, unknown>): Uint8Array =>
  utf8(`event: ${String(event["type"])}\ndata: ${JSON.stringify(event)}\n\n`);

// @intx/inference 0.3.0 does not export `createAdapterRegistry` from either
// of its two published subpaths ("." or "./providers"); a registry is built
// by hand here the same way a host's own `AdapterRegistry` implementation
// would.
const registry: AdapterRegistry = {
  has: (provider) => provider === OPENAI_RESPONSES_PROVIDER,
  resolve: (source) => createOpenAIResponsesAdapter(source, {}),
};

const source: InferenceSource = {
  id: "openai-responses:model",
  provider: OPENAI_RESPONSES_PROVIDER,
  baseURL: "https://example.test/v1",
  credentialId: "key",
  model: "model",
};

const readMaterial = (credentialId: string): CredentialMaterial => ({
  secret: credentialId,
});

const userTurn = (text: string): ConversationTurn => ({
  role: "user",
  timestamp: 0,
  content: [{ type: "text", text }],
});

function doneEvent(
  events: InferenceEvent[],
): Extract<InferenceEvent, { type: "inference.done" }> {
  const ev = events.find((e) => e.type === "inference.done");
  if (ev === undefined || ev.type !== "inference.done")
    throw new Error("expected inference.done");
  return ev;
}

function errorEvent(
  events: InferenceEvent[],
): Extract<InferenceEvent, { type: "inference.error" }> {
  const ev = events.find((e) => e.type === "inference.error");
  if (ev === undefined || ev.type !== "inference.error")
    throw new Error("expected inference.error");
  return ev;
}

async function drainRun(
  harness: Harness,
  turns: ConversationTurn[],
  events: InferenceEvent[],
): Promise<void> {
  let seq = 0;
  for await (const ev of harness.runInference({
    turns,
    source,
    nextSeq: () => seq++,
    readMaterial,
  }))
    events.push(ev);
}

async function collect(
  harness: Harness,
  turns: ConversationTurn[],
): Promise<InferenceEvent[]> {
  const events: InferenceEvent[] = [];
  const drain = drainRun(harness, turns, events);
  await harness.run();
  await drain;
  return events;
}

let harness: Harness | undefined;
afterEach(() => {
  harness?.dispose();
  harness = undefined;
});

describe("openai-responses adapter through runInference", () => {
  test("reasoning + text + function_call stream assembles a complete turn", async () => {
    harness = setupHarness({ adapters: registry });
    harness.scenario.onTool("shell", () => ({ ok: true }));
    const stream = harness.scenario.createStream();
    harness.scenario.whenRequestMatches(
      (req) => new URL(req.url).pathname === "/v1/responses",
      stream,
    );
    stream.enqueueAll(
      [
        sse({ type: "response.created", response: { id: "resp_1" } }),
        sse({
          type: "response.output_item.added",
          output_index: 0,
          item: { type: "reasoning", id: "rs_1" },
        }),
        sse({
          type: "response.reasoning_summary_text.delta",
          item_id: "rs_1",
          delta: "thinking",
        }),
        sse({
          type: "response.output_item.done",
          output_index: 0,
          item: { type: "reasoning", id: "rs_1", encrypted_content: "CIPHER" },
        }),
        sse({
          type: "response.output_item.added",
          output_index: 1,
          item: { type: "message", id: "msg_1" },
        }),
        sse({
          type: "response.output_text.delta",
          item_id: "msg_1",
          delta: "Hello",
        }),
        sse({
          type: "response.output_item.added",
          output_index: 2,
          item: {
            type: "function_call",
            id: "fc_1",
            call_id: "call_1",
            name: "shell",
          },
        }),
        sse({
          type: "response.function_call_arguments.delta",
          item_id: "fc_1",
          delta: '{"cmd":',
        }),
        sse({
          type: "response.function_call_arguments.delta",
          item_id: "fc_1",
          delta: '"ls"}',
        }),
        sse({
          type: "response.completed",
          response: {
            usage: {
              input_tokens: 100,
              output_tokens: 20,
              input_tokens_details: {
                cached_tokens: 40,
                cache_write_tokens: 10,
              },
            },
          },
        }),
      ],
      { startAt: 1 },
    );

    const events = await collect(harness, [userTurn("hi")]);
    const done = doneEvent(events);
    const kinds = done.data.turn.content.map((b) => b.type);
    expect(kinds).toEqual(["thinking", "text", "tool_call"]);
    const thinking = done.data.turn.content[0];
    if (thinking?.type !== "thinking")
      throw new Error("expected thinking block first");
    expect(thinking.signature).toBe(`${OPENAI_RESPONSES_PROVIDER}:CIPHER`);
    const call = done.data.turn.content[2];
    if (call?.type !== "tool_call")
      throw new Error("expected tool_call block third");
    expect(call.id).toBe("call_1");
    expect(call.arguments).toEqual({ cmd: "ls" });
    expect(done.data.usage).toEqual({
      input: 50,
      output: 20,
      cacheRead: 40,
      cacheWrite: 10,
      thinking: 0,
    });

    const request = harness.scenario.matchedRequests()[0];
    if (request === undefined) throw new Error("expected one matched request");
    expect(request.headers.get("authorization")).toBe("Bearer key");
    expect(request.headers.get("accept")).toBe("text/event-stream");
  });

  test("response.failed surfaces as a protocol_mismatch inference.error", async () => {
    harness = setupHarness({ adapters: registry });
    const stream = harness.scenario.createStream();
    harness.scenario.whenRequestMatches(() => true, stream);
    stream.enqueueAll(
      [
        sse({
          type: "response.failed",
          response: { error: { message: "backend exploded" } },
        }),
      ],
      {
        startAt: 1,
      },
    );
    const events = await collect(harness, [userTurn("hi")]);
    const error = errorEvent(events);
    expect(error.data.error.category).toBe("protocol_mismatch");
    expect(error.data.error.message).toContain("backend exploded");
  });

  test("a prior signed reasoning turn replays as a reasoning item ahead of its function_call", async () => {
    harness = setupHarness({ adapters: registry });
    const stream = harness.scenario.createStream();
    harness.scenario.whenRequestMatches(() => true, stream);
    stream.enqueueAll(
      [
        sse({
          type: "response.output_text.delta",
          item_id: "msg_2",
          delta: "done",
        }),
        sse({
          type: "response.completed",
          response: { usage: { input_tokens: 1, output_tokens: 1 } },
        }),
      ],
      { startAt: 1 },
    );
    const history: ConversationTurn[] = [
      userTurn("hi"),
      {
        role: "assistant",
        model: "model",
        timestamp: 1,
        content: [
          {
            type: "thinking",
            thinking: "thinking",
            signature: `${OPENAI_RESPONSES_PROVIDER}:CIPHER`,
          },
          {
            type: "tool_call",
            id: "call_1",
            name: "shell",
            arguments: { cmd: "ls" },
          },
        ],
      },
      {
        role: "user",
        timestamp: 2,
        content: [
          {
            type: "tool_result",
            callId: "call_1",
            content: [{ type: "text", text: "a.txt" }],
          },
        ],
      },
    ];
    await collect(harness, history);
    const request = harness.scenario.matchedRequests()[0];
    if (request === undefined) throw new Error("expected one matched request");
    const rawBody: unknown = await request.json();
    const requestBody = ReplayedRequestBody(rawBody);
    if (requestBody instanceof type.errors) {
      throw new Error(
        `replayed request body failed schema validation: ${requestBody.summary}`,
      );
    }
    const types = requestBody.input.map((i) => i.type);
    expect(types).toEqual([
      "message",
      "reasoning",
      "function_call",
      "function_call_output",
    ]);
    expect(requestBody.input[1]?.encrypted_content).toBe("CIPHER");
  });
});
