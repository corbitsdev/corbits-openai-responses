# AGENTS.md

## Purpose

A vendor-agnostic Interchange `ProviderAdapter` for the OpenAI Responses API
wire protocol. Vendor differences (headers, system-prompt placement,
session-id routing, request-body switches) are config, not forked code.

## Layout

`src/responses.ts` mirrors Interchange's own OpenAI Chat Completions adapter
section order in one file — quirks schema + resolver → request building →
event schemas → streaming parse → JSON parse → header extractors → factory.
`src/index.ts` is re-exports plus registry-shaped values:

- `src/responses.ts` — everything: `ResponsesQuirks` and its resolver,
  `ResponsesHooks`, request building, event schemas, SSE streaming parse,
  non-streaming JSON parse, block indexing, terminal-event detection, the
  rate-limit header extractors, reasoning `encrypted_content` tag/replay,
  and `createOpenAIResponsesAdapter` / `responsesAdapterFactory`.
- `src/index.ts` — re-exports of the above plus the two provider-id
  constants and `responsesAdapterFactories`.
- `*.test.ts` next to the source they cover.

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
- Public surface is `src/index.ts`'s export list only: `createOpenAIResponsesAdapter`,
  `responsesAdapterFactory`, `responsesAdapterFactories`, `ResponsesQuirks`,
  `ResponsesHooks`, `isResponsesStreamTerminal`, and the two provider-id
  constants. Everything else in `src/responses.ts` (`parseResponse`,
  `parseJSONResponse`, block indexing, signature tag/replay, the resolved
  quirks type) is module-private. Tests go through the public factory only —
  never import `./responses` directly from a test.
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
