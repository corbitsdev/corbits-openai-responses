# @corbits/openai-responses — Implementation

## Package

- Name: `@corbits/openai-responses`
- Public export: `./src/index.ts` (TypeScript source; no `dist/`)
- License: LGPL-2.1-only

## Runtime

- Bun >= 1.2 is the development runtime and consumes this package's
  TypeScript source directly.
- Node >= 24 is the engines floor; native Node does not load this
  extensionless TypeScript source as-is.
- Peer dependencies: `@intx/inference` and `@intx/types`. They must resolve
  to the host's own copy.

## Install

```sh
npm add @corbits/openai-responses
pnpm add @corbits/openai-responses
yarn add @corbits/openai-responses
bun add @corbits/openai-responses
```

## Public surface

From `@corbits/openai-responses`:

- `responsesAdapterFactory(quirks, hooks?)` — curry a `ResponsesQuirks` bag
  (and optional `ResponsesHooks`) into an `AdapterFactory`.
- `createOpenAIResponsesAdapter` — protocol-native `AdapterFactory`.
- `responsesAdapterFactories` — maps `openai-responses` and
  `openai-compatible-responses` onto `createOpenAIResponsesAdapter`.
- `OPENAI_RESPONSES_PROVIDER` — `"openai-responses"`.
- `ResponsesQuirks` — JSON-safe vendor bag (arktype).
- `ResponsesHooks` — TypeScript-only (`wrapSystemPrompt`, …).

## Factory: bake a vendor wire shape

The host keeps its own provider id. Example (Grok-shaped quirks):

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

## Factory: protocol-native by provider id

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
```

## Wire

Implemented: text, tool calls, reasoning with `encrypted_content` replay,
image and PDF input, SSE and non-streaming.

Not implemented: `text.format` structured output; url-form file input.

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
