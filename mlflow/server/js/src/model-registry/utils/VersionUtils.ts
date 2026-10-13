/**
 * Extract artifact path from provided `modelSource` string
 */
export function extractArtifactPathFromModelSource(modelSource: string, runId: string) {
  return modelSource.match(new RegExp(`/${runId}/artifacts/(.+)`))?.[1];
}

/**
 * Extract the logged model ID from a `models:/<model_id>` source URI.
 * Mirrors `_parse_model_uri` in the Python client: a single path segment without `@` is a model ID.
 */
export function extractLoggedModelIdFromModelSource(modelSource?: string) {
  return modelSource?.match(/^models:\/([^/@]+)$/)?.[1];
}
