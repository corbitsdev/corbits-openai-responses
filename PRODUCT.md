# @corbits/openai-responses — Product

## What it is

An Interchange `ProviderAdapter` for the OpenAI Responses API wire protocol.
Hosts speak one adapter. Vendor differences (Codex, xAI/Grok, plain OpenAI)
are a `ResponsesQuirks` bag, not a forked adapter.

## Why it exists

Responses backends share a protocol and diverge in headers, session routing,
content shape, and which request fields they accept. Forking an adapter per
vendor copies the same parse and replay logic. A quirks factory lets a host
bake a vendor's wire shape once and keep its own provider id.

## Who it is for

Interchange hosts that already resolve `@intx/inference` and `@intx/types`,
and need to talk to a Responses-shaped backend without owning a vendor fork.

## What users can do

- Bake a vendor's wire shape into `responsesAdapterFactory` and export a
  factory that takes only the host's source.
- Load the protocol-native factory by provider id
  (`createOpenAIResponsesAdapter`) when there is no prior wire shape.
- Register `openai-responses` and `openai-compatible-responses` from
  `responsesAdapterFactories` onto the host's adapter-loading mechanism.
- Send text, tool calls, reasoning with `encrypted_content` replay, image
  and PDF input, over SSE or non-streaming.

## Non-goals

- Structured output (`text.format`) is not implemented.
- Url-form file input is not implemented.
- The host owns provider ids; this package does not assign them.

## License

LGPL-2.1-only.
