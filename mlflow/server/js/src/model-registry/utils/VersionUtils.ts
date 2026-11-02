/**
 * Extract artifact path from provided `modelSource` string
 */
export function extractArtifactPathFromModelSource(modelSource: string, runId: string) {
  const escapedRunId = runId.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const runsUriMatch = modelSource.match(new RegExp(`^runs:/${escapedRunId}(?:/(.*))?$`));
  return runsUriMatch ? (runsUriMatch[1] ?? '') : modelSource.match(new RegExp(`/${escapedRunId}/artifacts/(.+)`))?.[1];
}

/**
 * Extract the logged model ID from a `models:/<model_id>` source URI.
 * Mirrors `_parse_model_uri` in the Python client: a single path segment without `@` is a model ID.
 */
export function extractLoggedModelIdFromModelSource(modelSource?: string) {
  return modelSource?.match(/^models:\/([^/@]+)$/)?.[1];
}
