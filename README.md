# @corbits/openai-responses

An `@intx/inference` provider adapter for the OpenAI Responses API (`/v1/responses`): SSE and JSON responses, text, tool calls, image and PDF input, and reasoning replay. An inference provider for Corbits and Interchange agents that also works in any host that runs `@intx/inference`.

## Why @corbits/openai-responses?

1. **One adapter for every Responses backend.** OpenAI, Codex, xAI and Ollama's `/v1` differ in paths, headers and body fields. Those differences are a JSON `quirks` object on the source, not forked code.
2. **Reasoning survives across turns.** Encrypted reasoning items are tagged with the provider that issued them. They are replayed only to the same provider and model, so multi-turn reasoning keeps its context and the backend never rejects a signature it did not issue.
3. **Interchange error and retry semantics.** Requests run through `runInference`, so rate limits, pacing headers and auth failures behave the same as for the built-in providers.

It speaks only the Responses protocol. For Chat Completions, use the built-in OpenAI adapter in `@intx/inference`.

## Install

```bash
bun add @corbits/openai-responses @intx/inference@^0.4.0 @intx/types@^0.4.0
```

Runs on Bun >= 1.2 or Node >= 24.

## Quickstart

Needs `OPENAI_API_KEY` set.

```ts
import { createDependencies, runInference } from "@intx/inference";
import {
  createOpenAIResponsesAdapter,
  OPENAI_RESPONSES_PROVIDER,
} from "@corbits/openai-responses";

const deps = createDependencies({
  has: (provider) => provider === OPENAI_RESPONSES_PROVIDER,
  resolve: (source, quirks) => createOpenAIResponsesAdapter(source, quirks),
});

let seq = 0;
for await (const event of runInference({
  deps,
  source: {
    id: "openai",
    provider: OPENAI_RESPONSES_PROVIDER,
    baseURL: "https://api.openai.com/v1",
    credentialId: "OPENAI_API_KEY",
    model: "gpt-5-mini",
  },
  turns: [
    {
      role: "user",
      timestamp: Date.now(),
      content: [{ type: "text", text: "Say hello." }],
    },
  ],
  nextSeq: () => seq++,
  readMaterial: (id) => {
    const secret = process.env[id];
    if (secret === undefined) throw new Error(`${id} is not set`);
    return { secret };
  },
})) {
  if (event.type === "inference.text.delta")
    process.stdout.write(event.data.token);
  if (event.type === "inference.error")
    throw new Error(event.data.error.message);
}
process.stdout.write("\n");
```

## Where it fits

[Interchange](https://github.com/faremeter/interchange) runs AI agents as principals (accounts that hold their own identity, permissions and credentials). Corbits packages add what an agent product needs around it.

- **Runs in:** the agent sidecar (the runtime next to each agent), or any process that calls `runInference`. No hub is required.
- **Plugs into:** the [`@intx/inference`](https://github.com/faremeter/interchange/tree/main/packages/inference) adapter registry, as the factory for the `openai-responses` provider id.
- **Pairs with:** [`@corbits/ollama-adapter`](https://github.com/corbitsdev/corbits-ollama-adapter) and [`@corbits/system-one`](https://github.com/corbitsdev/corbits-system-one), the other Corbits inference providers.

## Reference

| Export                                    | Description                                                                         |
| ----------------------------------------- | ----------------------------------------------------------------------------------- |
| `createOpenAIResponsesAdapter`            | `AdapterFactory`. Reads quirks from the source on every resolve.                    |
| `responsesAdapterFactory(quirks, hooks?)` | Returns an `AdapterFactory` with fixed quirks. Any per-source `quirks` are ignored. |
| `OPENAI_RESPONSES_PROVIDER`               | The `"openai-responses"` provider id.                                               |
| `OPENAI_COMPATIBLE_RESPONSES_PROVIDER`    | Deprecated `"openai-compatible-responses"` id from 0.1. Removed in 0.3.0.           |
| `responsesAdapterFactories`               | Record mapping both provider ids to `createOpenAIResponsesAdapter`.                 |
| `ResponsesQuirks`                         | Schema and type for the `quirks` object.                                            |
| `ResponsesHooks`                          | Code hooks: `wrapSystemPrompt`, `includeReasoningEffort`.                           |

### Quirks

Every field is optional. An absent field keeps the protocol default. Unknown keys are rejected.

| Quirk                    | Type                                                            | Default                               | Effect on the request                                                         |
| ------------------------ | --------------------------------------------------------------- | ------------------------------------- | ----------------------------------------------------------------------------- |
| `path`                   | `string`                                                        | `"/responses"`                        | Request path appended to the source's `baseURL`.                              |
| `headers.static`         | `Record<string, string>`                                        | `{}`                                  | Headers added verbatim; they can override the stock headers.                  |
| `headers.modelHeader`    | `string`                                                        | unset                                 | Header name that carries the model id.                                        |
| `headers.fromOption`     | `{ optionKey, header }[]`                                       | `[]`                                  | Copies each non-empty string `providerOptions[optionKey]` into `header`.      |
| `sessionIdOption`        | `string`                                                        | unset                                 | `providerOptions` key whose value is sent as `prompt_cache_key`.              |
| `sessionIdHeader`        | `string`                                                        | unset                                 | Also sends that session id in this header. Ignored without `sessionIdOption`. |
| `systemPrompt`           | `{ role: "system" \| "developer", shape: "string" \| "parts" }` | `{ role: "system", shape: "string" }` | Role and content shape of the leading system-prompt item.                     |
| `contentShape`           | `"typed" \| "flat"`                                             | `"typed"`                             | `flat` sends text-only content as a plain string instead of typed parts.      |
| `parallelToolCalls`      | `boolean`                                                       | unset                                 | Unset omits `parallel_tool_calls`; a boolean is sent as given.                |
| `maxOutputTokens`        | `boolean`                                                       | `true`                                | `false` omits `max_output_tokens` even when the caller sets `maxTokens`.      |
| `temperature`            | `boolean`                                                       | `true`                                | `false` omits `temperature` even when the caller sets it.                     |
| `store`                  | `boolean`                                                       | `false`                               | Sent as `store`.                                                              |
| `stream`                 | `boolean`                                                       | `true`                                | Sent as `stream`; `false` also sends `accept: application/json`.              |
| `reasoning.summary`      | `"auto" \| "detailed"`                                          | unset                                 | Sent as `reasoning.summary`.                                                  |
| `reasoning.effortOption` | `string`                                                        | unset                                 | `providerOptions` key whose value is sent as `reasoning.effort`.              |
| `instructions`           | `string`                                                        | unset                                 | Sent as `instructions`.                                                       |

## Using with Interchange

Interchange loads custom adapters from an operator-configured `AdapterManifest`. Add one entry per provider id you serve:

```ts
import { createDependencies, type AdapterManifest } from "@intx/inference";
import { loadAdapterRegistry } from "@intx/inference/providers";

const manifest: AdapterManifest = [
  {
    provider: "openai-responses",
    specifier: "@corbits/openai-responses",
    export: "createOpenAIResponsesAdapter",
  },
  {
    provider: "openai-compatible-responses",
    specifier: "@corbits/openai-responses",
    export: "createOpenAIResponsesAdapter",
  },
];
const deps = createDependencies(await loadAdapterRegistry(manifest));
```

Pass `deps` to `runInference`. Sources with `provider: "openai-responses"` (or the deprecated `"openai-compatible-responses"`) then resolve to this adapter, and each source's `quirks` configures its backend. To serve a vendor under its own id, add another entry with that `provider` and the same export. Manifest entries override built-in adapters with the same id, so don't reuse `openai` or another built-in id.

For a vendor that needs code hooks, bake its quirks into a factory in your own module and point a manifest entry at that export:

```ts
import { responsesAdapterFactory } from "@corbits/openai-responses";

export const createVendorAdapter = responsesAdapterFactory(
  { path: "/v1/responses", contentShape: "flat", temperature: false },
  { wrapSystemPrompt: (prompt) => `<system>${prompt}</system>` },
);
```

Its manifest entry names your module and that export, for example `{ provider: "vendor-responses", specifier: "./vendor-adapter.js", export: "createVendorAdapter" }`.

## Upgrading from 0.1

- No host change is needed. `responsesAdapterFactories` still maps `openai-compatible-responses` to `createOpenAIResponsesAdapter`, so sources stored under that id keep resolving and running.
- `OPENAI_COMPATIBLE_RESPONSES_PROVIDER` is deprecated and removed in 0.3.0. Move stored sources to `openai-responses` before then.
- `isResponsesStreamTerminal` is no longer exported. The adapter still applies it.
- `@intx/inference` and `@intx/types` peers are now `^0.4.0`.
- Quirks and reasoning signatures are unchanged.

## License

[LGPL-2.1-only](https://github.com/corbitsdev/corbits-openai-responses/blob/main/LICENSE)
