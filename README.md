# @corbits/openai-responses

An Interchange `ProviderAdapter` for the OpenAI Responses API wire protocol:
text, tool calls, reasoning with `encrypted_content` replay, image and PDF
input, SSE and non-streaming. Vendor differences (Codex, xAI/Grok, plain
OpenAI) are a `ResponsesQuirks` bag, not a forked adapter. Current scope covers
text, tool calls, reasoning replay, and image/PDF input.

## Runtime support

Bun >= 1.2 is the development runtime and consumes this package's TypeScript
source directly. Node >= 24 is the engines floor; native Node does not load
this extensionless TypeScript source as-is. `@intx/inference` and `@intx/types`
are peer dependencies and must resolve to the host's own copy.

## Quickstart

```sh
npm add @corbits/openai-responses
pnpm add @corbits/openai-responses
yarn add @corbits/openai-responses
bun add @corbits/openai-responses
```

Bake a vendor's wire shape into a factory, then build an adapter for one
source each: the Grok factory below, and the protocol-native
`createOpenAIResponsesAdapter` — loaded by provider id — for plain OpenAI.
The host keeps its own provider id.

```ts
import {
  OPENAI_RESPONSES_PROVIDER,
  createOpenAIResponsesAdapter,
  responsesAdapterFactory,
} from "@corbits/openai-responses";
import type { AdapterManifest } from "@intx/inference";
import { loadAdapterRegistry } from "@intx/inference/providers";
import type { LastCycleSource } from "@intx/types/runtime";

export const createGrokResponsesAdapter = responsesAdapterFactory(
  {
    contentShape: "flat",
    sessionIdOption: "sessionId",
    sessionIdHeader: "x-grok-session",
    reasoning: { summary: "auto" },
    headers: { static: { "x-grok-client": "my-harness" } },
    maxOutputTokens: true,
    temperature: false,
  },
  {
    wrapSystemPrompt: (prompt) =>
      `<grok-instructions>${prompt}</grok-instructions>`,
  },
);

const manifest: AdapterManifest = [
  {
    provider: OPENAI_RESPONSES_PROVIDER,
    specifier: "@corbits/openai-responses",
    export: "createOpenAIResponsesAdapter",
  },
];

await loadAdapterRegistry(manifest);

const grok: LastCycleSource = {
  sourceId: "grok/1",
  provider: "grok",
  model: "grok-4",
};

export const grokAdapter = createGrokResponsesAdapter(grok);

const openai: LastCycleSource = {
  sourceId: "openai/1",
  provider: OPENAI_RESPONSES_PROVIDER,
  model: "gpt-5",
};

export const openaiAdapter = createOpenAIResponsesAdapter(openai);
```

`responsesAdapterFactories` maps `openai-responses` and
`openai-compatible-responses` onto `createOpenAIResponsesAdapter`, so a host
can register either provider id with the same factory.

## How it works

`quirks` are JSON on `InferenceSource` (persisted, sent over the wire).
`hooks` are code, applied once at `responsesAdapterFactory` construction.
Defaults are protocol-native: system prompt, `maxTokens`, and `temperature`
go through unless a quirk opts a backend out. The host owns provider ids; a
reasoning signature is tagged with the id in effect when it was issued.

## Development

```sh
git clone https://github.com/corbitsdev/corbits-openai-responses.git
cd corbits-openai-responses
bun install
bun run typecheck
bun run lint
bun run format:check
bun run test
bun run check
```

`bun run format` rewrites the tree. `bun run check` is typecheck + lint +
format:check + test.

## License

LGPL-2.1-only.
