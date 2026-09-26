import type { AdapterFactory } from "@intx/inference";
import { isResponsesStreamTerminal } from "./protocol/iterator.js";
import {
  createOpenAIResponsesAdapter,
  responsesAdapterFactory,
  ResponsesQuirks,
} from "./responses.js";
import type { ResponsesHooks } from "./responses.js";

export {
  createOpenAIResponsesAdapter,
  isResponsesStreamTerminal,
  responsesAdapterFactory,
  ResponsesQuirks,
};
export type { ResponsesHooks };

export const OPENAI_RESPONSES_PROVIDER = "openai-responses";
export const OPENAI_COMPATIBLE_RESPONSES_PROVIDER =
  "openai-compatible-responses";

/**
 * A provider-id -> factory record a host can register with its own
 * adapter-loading mechanism (e.g. an `AdapterManifest` entry per id, or a
 * hand-built `AdapterRegistry`), covering both provider ids this package
 * serves: a plain OpenAI Responses source and an OpenAI-compatible one wire
 * up identically, since the wire protocol — and therefore the code — is the
 * same; only the `quirks` bag each source carries differs.
 */
export const responsesAdapterFactories: Readonly<
  Record<string, AdapterFactory>
> = {
  [OPENAI_RESPONSES_PROVIDER]: createOpenAIResponsesAdapter,
  [OPENAI_COMPATIBLE_RESPONSES_PROVIDER]: createOpenAIResponsesAdapter,
};
