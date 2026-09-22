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

Bake a vendor's wire shape into a quirks bag, then hand it to
`responsesAdapterFactory` to get back an `AdapterFactory`. This mirrors how
`@corbits/xai-provider` wires up Grok's Responses-speaking CLI proxy:

```ts
import {
  responsesAdapterFactory,
  type ResponsesQuirks,
} from "@corbits/openai-responses";
import type { AdapterFactory } from "@intx/inference";

// Mirrors the vendor's own request shape: headers, system-prompt
// placement, reasoning summary depth, which stock fields to suppress.
const grokResponsesQuirks: ResponsesQuirks = {
  path: "/v1/responses",
  headers: {
    static: { "x-grok-client-identifier": "my-harness" },
  },
  sessionIdOption: "sessionId",
  systemPrompt: { role: "system", shape: "string" },
  contentShape: "flat",
  reasoning: { summary: "detailed" },
  maxOutputTokens: false,
  temperature: false,
};

export const createGrokResponsesAdapter: AdapterFactory =
  responsesAdapterFactory(grokResponsesQuirks);
```

A sidecar host registers the resulting export on its
`SIDECAR_ADAPTER_MANIFEST` (one entry per provider id, `specifier` naming an
already-installed module) — this package ships no manifest entry itself; the
vendor package wrapping it does, e.g.
`{"provider":"xai","specifier":"@corbits/xai-provider","export":"createXaiResponsesAdapter"}`.

For a source that already speaks the protocol natively — plain OpenAI, or an
OpenAI-compatible Responses endpoint — no quirks bag is needed at all:
register `createOpenAIResponsesAdapter` directly. `responsesAdapterFactories`
maps both `openai-responses` and `openai-compatible-responses` provider ids
onto it, so a host can register either id with the same factory:

```ts
import { responsesAdapterFactories } from "@corbits/openai-responses";

responsesAdapterFactories["openai-responses"]; // === createOpenAIResponsesAdapter
```

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
