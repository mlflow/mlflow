/**
 * Extract artifact path from provided `modelSource` string
 */
export function extractArtifactPathFromModelSource(modelSource: string, runId: string) {
  return modelSource.match(new RegExp(`/${runId}/artifacts/(.+)`))?.[1];
}
