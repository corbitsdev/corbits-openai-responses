# @corbits/openai-responses — Architecture

## Shape

One adapter for the OpenAI Responses API wire protocol. Vendor deviation is
config on that adapter, not a second implementation.

```
host (provider id, InferenceSource)
        │
        ▼
responsesAdapterFactory(quirks, hooks)   ← bake once
        │
        ▼
AdapterFactory(source) → ProviderAdapter
        │
        ▼
Responses wire (SSE or JSON)
```

For a source with no prior wire shape, the host loads
`createOpenAIResponsesAdapter` by provider id instead of currying quirks.

## Quirks vs hooks

| | `quirks` | `hooks` |
| --- | --- | --- |
| Shape | JSON on `InferenceSource` | TypeScript functions |
| Lifetime | Persisted and sent over the wire | Applied once at factory construction |
| Typical content | Content shape, session-id option/header, reasoning summary, static headers, max-output-tokens / temperature opt-out | `wrapSystemPrompt` |

A function-valued field cannot ride in `quirks`; code-shaped accommodations
belong in `hooks`.

## Defaults

Protocol-native unless a quirk opts a backend out: system prompt,
`maxTokens`, and `temperature` go through. An absent quirks bag is the same
as `{}` — native defaults, not an error.

## Provider ids

The host owns provider ids. This package does not mint them.

`responsesAdapterFactories` maps two ids onto the same protocol-native
factory:

- `openai-responses` → `createOpenAIResponsesAdapter`
- `openai-compatible-responses` → `createOpenAIResponsesAdapter`

The wire protocol is the same; only the quirks bag each source carries
differs.

## Reasoning signatures

A reasoning signature is tagged with the provider id in effect when it was
issued. Replay is only valid for that id; a foreign backend must not receive
another vendor's `encrypted_content`.

## Non-goals (adapter surface)

Structured output (`text.format`) and url-form file input are out of the
adapter's implemented surface.
