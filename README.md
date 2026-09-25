# @corbits/openai-responses

Use this adapter to run Interchange inference against any OpenAI Responses
API endpoint (OpenAI, Codex, xAI/Grok), with text, tool calls, reasoning
replay, and image/PDF input.

## Quickstart

```sh
npm add @corbits/openai-responses @intx/inference @intx/types
```

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
  if (event.type === "inference.text.delta") {
    process.stdout.write(event.data.token);
  }
}
```

`@intx/inference` and `@intx/types` are peer dependencies; the host supplies
its own copy. Requires Node >= 24 or Bun >= 1.2.

## Using with Interchange

Register `createOpenAIResponsesAdapter` under the `openai-responses`
provider id in your host's `AdapterRegistry`, as the quickstart does. A
source's `quirks` bag is passed through as the factory's second argument.

For a vendor whose backend deviates from the protocol, bake its quirks into a
factory once with `responsesAdapterFactory`:

```ts
import { responsesAdapterFactory } from "@corbits/openai-responses";

export const createVendorAdapter = responsesAdapterFactory({
  path: "/v1/responses",
  contentShape: "flat",
  temperature: false,
});
```

The optional second argument, `ResponsesHooks`, carries code-shaped
accommodations (`wrapSystemPrompt`, `includeReasoningEffort`) that cannot
live in a JSON quirks bag.

## Quirks

Every field is optional. An absent field keeps the protocol-native default.
Unknown keys are rejected.

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

## License

LGPL-2.1-only.
