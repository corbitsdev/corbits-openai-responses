import type { AdapterFactory } from "@intx/inference";
import { createOpenAIResponsesAdapter } from "./responses.js";

export {
  createOpenAIResponsesAdapter,
  responsesAdapterFactory,
  ResponsesQuirks,
} from "./responses.js";
export type { ResponsesHooks } from "./responses.js";

export const OPENAI_RESPONSES_PROVIDER = "openai-responses";

/**
 * Provider id stored on 0.1 sources; resolves to the same adapter.
 *
 * @deprecated Use `OPENAI_RESPONSES_PROVIDER`. Removed in 0.3.0.
 */
export const OPENAI_COMPATIBLE_RESPONSES_PROVIDER =
  "openai-compatible-responses";

/**
 * Provider-id -> factory record for a host's adapter registry. Maps both
 * provider ids to `createOpenAIResponsesAdapter`.
 */
export const responsesAdapterFactories: Readonly<
  Record<string, AdapterFactory>
> = {
  [OPENAI_RESPONSES_PROVIDER]: createOpenAIResponsesAdapter,
  [OPENAI_COMPATIBLE_RESPONSES_PROVIDER]: createOpenAIResponsesAdapter,
};
