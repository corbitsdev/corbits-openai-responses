# @corbits/openai-responses

An Interchange `ProviderAdapter` for the OpenAI Responses API wire protocol:
text, tool calls, reasoning with `encrypted_content` replay, image and PDF
input, SSE and non-streaming. Vendor differences (Codex, xAI/Grok, plain
OpenAI) are a `ResponsesQuirks` bag, not a forked adapter. Structured output
(`text.format`) and url-form file input are not implemented.

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

Bake a vendor's wire shape into a factory. The host keeps its own provider id.

```ts
import { responsesAdapterFactory } from "@corbits/openai-responses";

export const createGrokResponsesAdapter = responsesAdapterFactory(
  {
    contentShape: "flat",
    sessionIdOption: "sessionId",
    sessionIdHeader: "x-grok-session",
    reasoning: { summary: "auto" },
    headers: { static: { "x-grok-client": "workbench" } },
    maxOutputTokens: true,
    temperature: false,
  },
  {
    wrapSystemPrompt: (prompt) =>
      `<grok-instructions>${prompt}</grok-instructions>`,
  },
);
```

For a source with no prior wire shape, load the protocol-native factory by
provider id:

```ts
import type { AdapterManifest } from "@intx/inference";
import {
  OPENAI_RESPONSES_PROVIDER,
  createOpenAIResponsesAdapter,
  responsesAdapterFactories,
} from "@corbits/openai-responses";

const manifest: AdapterManifest = [
  {
    provider: OPENAI_RESPONSES_PROVIDER,
    specifier: "@corbits/openai-responses",
    export: "createOpenAIResponsesAdapter",
  },
];

void createOpenAIResponsesAdapter;
void responsesAdapterFactories;
void manifest;
```

`responsesAdapterFactories` maps `openai-responses` and
`openai-compatible-responses` onto `createOpenAIResponsesAdapter`.

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
