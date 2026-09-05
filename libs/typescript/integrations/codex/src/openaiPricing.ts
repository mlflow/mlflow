/**
 * GENERATED FILE - DO NOT EDIT BY HAND.
 *
 * Snapshot of OpenAI per-model pricing from MLflow's bundled model catalog
 * (mlflow/utils/model_catalog/openai.json). Regenerate with:
 *
 *   npm run sync:pricing
 *
 * tests/openaiPricing.test.ts fails when this file drifts from the catalog.
 */

/** Per-million-token USD rates for one OpenAI model. */
export interface OpenAIModelRate {
  /** USD per million regular (non-cached) input tokens. */
  input: number;
  /** USD per million output tokens. */
  output: number;
  /** USD per million cache-read (cached) input tokens. */
  cacheRead?: number;
  /** USD per million cache-write input tokens (unused by OpenAI today). */
  cacheWrite?: number;
}

export const OPENAI_MODEL_RATES: Readonly<Record<string, OpenAIModelRate>> = {
  'chat-latest': { input: 5, output: 30, cacheRead: 0.5 },
  'chatgpt-4o-latest': { input: 5, output: 15 },
  'daybreak-blue-latest': { input: 4, output: 20, cacheRead: 0.4, cacheWrite: 5 },
  'daybreak-red-latest': { input: 12.5, output: 75, cacheRead: 1.25, cacheWrite: 15.625 },
  'gpt-3.5-turbo': { input: 0.5, output: 1.5 },
  'gpt-3.5-turbo-0125': { input: 0.5, output: 1.5 },
  'gpt-3.5-turbo-1106': { input: 1, output: 2 },
  'gpt-3.5-turbo-16k': { input: 3, output: 4 },
  'gpt-4': { input: 30, output: 60 },
  'gpt-4-0125-preview': { input: 10, output: 30 },
  'gpt-4-0314': { input: 30, output: 60 },
  'gpt-4-0613': { input: 30, output: 60 },
  'gpt-4-1106-preview': { input: 10, output: 30 },
  'gpt-4-turbo': { input: 10, output: 30 },
  'gpt-4-turbo-2024-04-09': { input: 10, output: 30 },
  'gpt-4-turbo-preview': { input: 10, output: 30 },
  'gpt-4.1': { input: 2, output: 8, cacheRead: 0.5 },
  'gpt-4.1-2025-04-14': { input: 2, output: 8, cacheRead: 0.5 },
  'gpt-4.1-mini': { input: 0.4, output: 1.6, cacheRead: 0.1 },
  'gpt-4.1-mini-2025-04-14': { input: 0.4, output: 1.6, cacheRead: 0.1 },
  'gpt-4.1-nano': { input: 0.1, output: 0.4, cacheRead: 0.025 },
  'gpt-4.1-nano-2025-04-14': { input: 0.1, output: 0.4, cacheRead: 0.025 },
  'gpt-4o': { input: 2.5, output: 10, cacheRead: 1.25 },
  'gpt-4o-2024-05-13': { input: 5, output: 15 },
  'gpt-4o-2024-08-06': { input: 2.5, output: 10, cacheRead: 1.25 },
  'gpt-4o-2024-11-20': { input: 2.5, output: 10, cacheRead: 1.25 },
  'gpt-4o-audio-preview': { input: 2.5, output: 10 },
  'gpt-4o-audio-preview-2024-12-17': { input: 2.5, output: 10 },
  'gpt-4o-audio-preview-2025-06-03': { input: 2.5, output: 10 },
  'gpt-4o-mini': { input: 0.15, output: 0.6, cacheRead: 0.075 },
  'gpt-4o-mini-2024-07-18': { input: 0.15, output: 0.6, cacheRead: 0.075 },
  'gpt-4o-mini-audio-preview': { input: 0.15, output: 0.6 },
  'gpt-4o-mini-audio-preview-2024-12-17': { input: 0.15, output: 0.6 },
  'gpt-4o-mini-realtime-preview': { input: 0.6, output: 2.4, cacheRead: 0.3 },
  'gpt-4o-mini-realtime-preview-2024-12-17': { input: 0.6, output: 2.4, cacheRead: 0.3 },
  'gpt-4o-mini-search-preview': { input: 0.15, output: 0.6, cacheRead: 0.075 },
  'gpt-4o-mini-search-preview-2025-03-11': { input: 0.15, output: 0.6, cacheRead: 0.075 },
  'gpt-4o-realtime-preview': { input: 5, output: 20, cacheRead: 2.5 },
  'gpt-4o-realtime-preview-2024-12-17': { input: 5, output: 20, cacheRead: 2.5 },
  'gpt-4o-realtime-preview-2025-06-03': { input: 5, output: 20, cacheRead: 2.5 },
  'gpt-4o-search-preview': { input: 2.5, output: 10, cacheRead: 1.25 },
  'gpt-4o-search-preview-2025-03-11': { input: 2.5, output: 10, cacheRead: 1.25 },
  'gpt-5': { input: 1.25, output: 10, cacheRead: 0.125 },
  'gpt-5-2025-08-07': { input: 1.25, output: 10, cacheRead: 0.125 },
  'gpt-5-chat': { input: 1.25, output: 10, cacheRead: 0.125 },
  'gpt-5-chat-latest': { input: 1.25, output: 10, cacheRead: 0.125 },
  'gpt-5-mini': { input: 0.25, output: 2, cacheRead: 0.025 },
  'gpt-5-mini-2025-08-07': { input: 0.25, output: 2, cacheRead: 0.025 },
  'gpt-5-nano': { input: 0.05, output: 0.4, cacheRead: 0.005 },
  'gpt-5-nano-2025-08-07': { input: 0.05, output: 0.4, cacheRead: 0.005 },
  'gpt-5-search-api': { input: 1.25, output: 10, cacheRead: 0.125 },
  'gpt-5-search-api-2025-10-14': { input: 1.25, output: 10, cacheRead: 0.125 },
  'gpt-5.1': { input: 1.25, output: 10, cacheRead: 0.125 },
  'gpt-5.1-2025-11-13': { input: 1.25, output: 10, cacheRead: 0.125 },
  'gpt-5.1-chat-latest': { input: 1.25, output: 10, cacheRead: 0.125 },
  'gpt-5.2': { input: 1.75, output: 14, cacheRead: 0.175 },
  'gpt-5.2-2025-12-11': { input: 1.75, output: 14, cacheRead: 0.175 },
  'gpt-5.2-chat-latest': { input: 1.75, output: 14, cacheRead: 0.175 },
  'gpt-5.3-chat-latest': { input: 1.75, output: 14, cacheRead: 0.175 },
  'gpt-5.4': { input: 2.5, output: 15, cacheRead: 0.25 },
  'gpt-5.4-2026-03-05': { input: 2.5, output: 15, cacheRead: 0.25 },
  'gpt-5.4-mini': { input: 0.75, output: 4.5, cacheRead: 0.075 },
  'gpt-5.4-mini-2026-03-17': { input: 0.75, output: 4.5, cacheRead: 0.075 },
  'gpt-5.4-nano': { input: 0.2, output: 1.25, cacheRead: 0.02 },
  'gpt-5.4-nano-2026-03-17': { input: 0.2, output: 1.25, cacheRead: 0.02 },
  'gpt-5.5': { input: 5, output: 30, cacheRead: 0.5 },
  'gpt-5.5-2026-04-23': { input: 5, output: 30, cacheRead: 0.5 },
  'gpt-5.6': { input: 4, output: 20, cacheRead: 0.4, cacheWrite: 5 },
  'gpt-5.6-cyber': { input: 12.5, output: 75, cacheRead: 1.25, cacheWrite: 15.625 },
  'gpt-5.6-luna': { input: 0.2, output: 1.2, cacheRead: 0.02, cacheWrite: 0.25 },
  'gpt-5.6-sol': { input: 4, output: 20, cacheRead: 0.4, cacheWrite: 5 },
  'gpt-5.6-terra': { input: 2, output: 12, cacheRead: 0.2, cacheWrite: 2.5 },
  'gpt-6-astra': { input: 10, output: 50, cacheRead: 1, cacheWrite: 12.5 },
  'gpt-audio': { input: 2.5, output: 10 },
  'gpt-audio-1.5': { input: 2.5, output: 10 },
  'gpt-audio-2025-08-28': { input: 2.5, output: 10 },
  'gpt-audio-mini': { input: 0.6, output: 2.4 },
  'gpt-audio-mini-2025-10-06': { input: 0.6, output: 2.4 },
  'gpt-audio-mini-2025-12-15': { input: 0.6, output: 2.4 },
  'gpt-image-1.5': { input: 5, output: 10, cacheRead: 1.25 },
  'gpt-image-1.5-2025-12-16': { input: 5, output: 10, cacheRead: 1.25 },
  'gpt-image-2': { input: 5, output: 10, cacheRead: 1.25 },
  'gpt-image-2-2026-04-21': { input: 5, output: 10, cacheRead: 1.25 },
  'gpt-realtime': { input: 4, output: 16, cacheRead: 0.4 },
  'gpt-realtime-1.5': { input: 4, output: 16, cacheRead: 0.4 },
  'gpt-realtime-2': { input: 4, output: 16, cacheRead: 0.4 },
  'gpt-realtime-2.1': { input: 4, output: 24, cacheRead: 0.4 },
  'gpt-realtime-2.1-mini': { input: 0.6, output: 2.4, cacheRead: 0.06 },
  'gpt-realtime-2025-08-28': { input: 4, output: 16, cacheRead: 0.4 },
  'gpt-realtime-mini': { input: 0.6, output: 2.4 },
  'gpt-realtime-mini-2025-10-06': { input: 0.6, output: 2.4, cacheRead: 0.06 },
  'gpt-realtime-mini-2025-12-15': { input: 0.6, output: 2.4, cacheRead: 0.06 },
  o1: { input: 15, output: 60, cacheRead: 7.5 },
  'o1-2024-12-17': { input: 15, output: 60, cacheRead: 7.5 },
  o3: { input: 2, output: 8, cacheRead: 0.5 },
  'o3-2025-04-16': { input: 2, output: 8, cacheRead: 0.5 },
  'o3-mini': { input: 1.1, output: 4.4, cacheRead: 0.55 },
  'o3-mini-2025-01-31': { input: 1.1, output: 4.4, cacheRead: 0.55 },
  'o4-mini': { input: 1.1, output: 4.4, cacheRead: 0.275 },
  'o4-mini-2025-04-16': { input: 1.1, output: 4.4, cacheRead: 0.275 },
  'text-embedding-3-large': { input: 0.13, output: 0 },
  'text-embedding-3-small': { input: 0.02, output: 0 },
  'text-embedding-ada-002': { input: 0.1, output: 0 },
  'text-embedding-ada-002-v2': { input: 0.1, output: 0 },
};
