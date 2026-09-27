# Contributing

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
bun run check
```

`bun run check` is typecheck + lint + format:check + test. `bun run format`
rewrites the tree.
