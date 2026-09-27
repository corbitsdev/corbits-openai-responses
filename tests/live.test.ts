// Live calls against a real Responses API. Opt in by setting
// OPENAI_RESPONSES_LIVE_URL (e.g. https://api.openai.com/v1, or Ollama's
// /v1). OPENAI_RESPONSES_LIVE_KEY and OPENAI_RESPONSES_LIVE_MODEL are
// optional; servers that ignore auth accept the placeholder key.

import { describe, expect, test } from "bun:test";
import { createDependencies, runInference } from "@intx/inference";
import type {
  InferenceEvent,
  InferenceOptions,
  InferenceSource,
} from "@intx/types/runtime";
import {
  createOpenAIResponsesAdapter,
  OPENAI_RESPONSES_PROVIDER,
} from "../src/index";

const baseURL = process.env["OPENAI_RESPONSES_LIVE_URL"] ?? "";
const apiKey = process.env["OPENAI_RESPONSES_LIVE_KEY"] ?? "unused";
const model = process.env["OPENAI_RESPONSES_LIVE_MODEL"] ?? "gpt-5-mini";

const source: InferenceSource = {
  id: `openai-responses:${model}`,
  provider: OPENAI_RESPONSES_PROVIDER,
  baseURL,
  credentialId: "openai",
  model,
};

const deps = createDependencies({
  has: (provider) => provider === OPENAI_RESPONSES_PROVIDER,
  resolve: (s) => createOpenAIResponsesAdapter(s, {}),
});

async function doneTurn(text: string, inferenceOptions: InferenceOptions) {
  let seq = 0;
  const events: InferenceEvent[] = [];
  for await (const ev of runInference({
    turns: [{ role: "user", timestamp: 0, content: [{ type: "text", text }] }],
    source,
    inferenceOptions,
    nextSeq: () => seq++,
    readMaterial: () => ({ secret: apiKey }),
    deps,
  }))
    events.push(ev);
  const done = events.find((e) => e.type === "inference.done");
  if (done?.type !== "inference.done")
    throw new Error(`expected inference.done, got ${JSON.stringify(events)}`);
  return { events, turn: done.data.turn };
}

describe.skipIf(baseURL === "")("live Responses API", () => {
  test("streams a text turn", async () => {
    const { events, turn } = await doneTurn("Reply with the word pong.", {});
    expect(events.some((e) => e.type === "inference.text.delta")).toBe(true);
    const text = turn.content.find((b) => b.type === "text");
    expect(text?.type === "text" && text.text.toLowerCase()).toContain("pong");
  }, 60_000);

  test("emits a tool call", async () => {
    const { turn } = await doneTurn("What is the weather in Paris?", {
      tools: [
        {
          name: "get_weather",
          description: "Get the current weather for a city.",
          inputSchema: {
            type: "object",
            properties: { city: { type: "string" } },
            required: ["city"],
            additionalProperties: false,
          },
        },
      ],
    });
    const call = turn.content.find((b) => b.type === "tool_call");
    if (call?.type !== "tool_call") throw new Error("expected a tool_call");
    expect(call.name).toBe("get_weather");
    expect(call.arguments).toMatchObject({
      city: expect.stringMatching(/paris/i),
    });
  }, 60_000);
});
