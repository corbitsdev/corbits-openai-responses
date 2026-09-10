# @corbits/openai-responses

An Interchange `ProviderAdapter` for the OpenAI Responses API wire protocol:
text, tool calls, reasoning with `encrypted_content` replay, image and PDF
input, SSE and non-streaming. It knows no vendor by name — every difference
between backends that speak this protocol (Codex's ChatGPT backend, xAI/Grok's
proxy, plain OpenAI) is a `ResponsesQuirks` config object, not a forked copy of
the adapter. Structured output (`text.format`) and url-form file input are not
implemented.

## Install

```
bun add github:corbitsdev/corbits-openai-responses
```

The package ships TypeScript source and needs no build step; Bun consumes it
directly. `@intx/inference` and `@intx/types` are peer dependencies and
resolve to the host's own copy.

## Usage

Migrating an existing vendor adapter onto this package means reproducing its
live wire shape under the host's own provider id, so bake the vendor's
quirks into a factory rather than registering the bare default:

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

A host keeps its existing provider id (`"grok-responses"` above) and registers
`createGrokResponsesAdapter` against it exactly as it did its old adapter.

For a fresh source with no prior wire shape to match, load the bare factory
by provider id through an `AdapterManifest` entry instead — it applies
protocol-native defaults and is not a drop-in replacement for an adapter
whose wire shape already exists:

```ts
import type { AdapterManifest } from "@intx/inference";

const manifest: AdapterManifest = [
  {
    provider: "openai-responses",
    specifier: "@corbits/openai-responses",
    export: "createOpenAIResponsesAdapter",
  },
];
```

## API

See `src/index.ts` for the full public surface and its TSDoc; every quirk and
hook field is documented individually in `src/responses.ts`.

## Design notes

- `quirks` are JSON (an `InferenceSource.quirks` bag, persisted and sent over
  the wire); `hooks` are code, applied once at `responsesAdapterFactory`
  construction, never smuggled into the serializable bag.
- Defaults are protocol-native: the caller's system prompt, `maxTokens`, and
  `temperature` are forwarded unless a quirk explicitly opts a backend out.
- The host owns provider ids, not this package. A reasoning signature is
  tagged with the provider id in effect when it was issued; renaming that id
  later invalidates every signature's replay silently rather than erroring.
- `isStreamTerminal` rides on the returned adapter value (not the
  `ProviderAdapter` type, which 0.3.0 doesn't declare it on) for hosts running
  a semantic-terminal harness; it's also exported standalone.
- `parallelToolCalls` is tri-state: absent omits the field, `true`/`false`
  send verbatim — some backends require an explicit `false`.
- Usage mapping splits every field OpenAI documents as a subset of
  `input_tokens` (`cached_tokens`, `cache_write_tokens`) out of `input`, so
  the OpenAI-native fields stay non-overlapping when summed; the
  Anthropic-shaped `cache_creation_tokens` that gateways emit is reported
  as `cacheWrite` without reducing `input`, since its subset relationship
  is unobservable from here.
- A host must resolve one copy of `@intx/inference`: it's a peer dependency,
  and an adapter built against a second copy fails `instanceof
ProtocolMismatchError` checks in the host's harness.

## Not supported

- Structured output (`text.format`) and url-form `input_file` document input.
- Document input is PDF only, matching upstream's Chat Completions adapter.
- The `previous_response_id` / `store: true` path is unexercised by this
  package's tests.
- `extractRetryAfterMs` / `extractPacingDelayMs` duplicate logic from
  upstream's OpenAI Chat Completions adapter, which doesn't export them;
  delete these once `@intx/inference` does.

## License

LGPL-2.1-or-later.
