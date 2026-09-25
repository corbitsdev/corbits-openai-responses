# AGENTS.md

## Purpose

A vendor-agnostic Interchange `ProviderAdapter` for the OpenAI Responses API
wire protocol. Vendor differences (headers, system-prompt placement,
session-id routing, request-body switches) are config, not forked code.

## Layout

`src/responses.ts` holds the quirks schema + resolver, the header
extractors, and the factory that wires `src/protocol/` together, following
Interchange's `inference-discovery-openai` layout. `src/index.ts` is
re-exports plus the provider ids and factory record:

- `src/responses.ts` — `ResponsesQuirks` and its resolver, `ResponsesHooks`,
  the rate-limit header extractors, and `createOpenAIResponsesAdapter` /
  `responsesAdapterFactory`.
- `src/protocol/body.ts` — `buildResponsesRequest` with its content, tool,
  system-prompt, and reasoning helpers, plus reasoning `encrypted_content`
  signature tag/replay.
- `src/protocol/iterator.ts` — event schemas, SSE streaming parse,
  non-streaming JSON parse, block indexing, and terminal-event detection.
- `src/index.ts` — re-exports of the above plus the two provider-id
  constants (`OPENAI_COMPATIBLE_RESPONSES_PROVIDER` is deprecated, removed
  in 0.3.0) and `responsesAdapterFactories`.
- `src/*.test.ts` — unit tests for the wire codec and parser, excluded from
  the build.
- `e2e/` — `runInference` through `@intx/inference-testing`, plus the
  opt-in live suite (`OPENAI_RESPONSES_LIVE_URL`).

## Rules

- Consume `@intx/inference` and `@intx/types` as `peerDependencies`
  (`^0.4.0`), pinned `0.4.0` in `devDependencies` for typecheck — never
  vendor, never `workspace:`. A host must resolve exactly one copy; a second
  copy breaks `instanceof ProtocolMismatchError`.
- Parse every trust boundary (the `quirks` bag, every `response.*` SSE event,
  the non-streaming JSON body) with arktype; never `as T` untrusted input.
- `exactOptionalPropertyTypes` is on: omit optional keys, never assign
  `undefined` to one.
- No product strings baked in — anything vendor-specific is a config field
  the caller supplies.
- Public surface is `src/index.ts`'s export list only:
  `createOpenAIResponsesAdapter`, `responsesAdapterFactory`,
  `responsesAdapterFactories`, `ResponsesQuirks`, `ResponsesHooks`, and the
  two provider-id constants.
  Everything else in `src/responses.ts` and `src/protocol/`
  (`parseResponse`, `parseJSONResponse`, `isResponsesStreamTerminal`,
  block indexing, signature tag/replay, the resolved quirks type) is
  package-private. Tests go through the public factory only — never import
  `./responses` directly from a test.
- Tests only for load-bearing risk (wire-format encoding, state machines,
  hostile-input parsing) — not for trivial mapping or "returns what I passed
  in".

## Local development

```
bun install
bun run check   # typecheck + lint + format:check + test
```

## Distribution

The package ships compiled `dist/` on npm as `@corbits/openai-responses`:
`main`/`types` and `exports` point at `dist/index.js` / `dist/index.d.ts`
(built with `bun run build`, i.e. `tsc -p tsconfig.build.json`).
Consumers install it with `bun add @corbits/openai-responses` and Bun or
Node loads the built output. It requires Node >=24 and Bun >=1.2.0 per the
`engines` field. To
publish, bump the version and run `npm publish --access public`.
