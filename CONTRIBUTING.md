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

Unit tests sit next to the code in `src/`; end-to-end tests live in `e2e/`.
`e2e/live.test.ts` calls a real Responses API and runs only when
`OPENAI_RESPONSES_LIVE_URL` is set:

```sh
OPENAI_RESPONSES_LIVE_URL=http://localhost:11434/v1 \
OPENAI_RESPONSES_LIVE_MODEL=gpt-oss:20b bun run test:e2e
```

`OPENAI_RESPONSES_LIVE_KEY` is optional.

## Commit messages

Commit subjects and PR titles follow [Conventional Commits](https://www.conventionalcommits.org): `feat`, `fix`, `refactor`, `test`, `docs`, `build`, `ci`, `perf`, and `chore(release): x.y.z` for releases.
Add `!` only for public API breaks: removed or renamed exports, changed signatures, newly required params. Peer and dependency range changes are `build(deps):` with no `!`.
Keep subjects imperative, lowercase after the colon, 72 characters or less, and free of ticket IDs.
Every PR links its issue with a `Closes <issue id>` line in the PR body.
