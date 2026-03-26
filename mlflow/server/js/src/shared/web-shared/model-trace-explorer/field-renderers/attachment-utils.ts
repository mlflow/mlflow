/**
 * Parses an mlflow-attachment:// URI into its component parts.
 * Returns null if the URI is not a valid attachment URI.
 */
export function parseAttachmentUri(uri: string): { attachmentId: string; traceId: string; contentType: string } | null {
  try {
    const parsed = new URL(uri);
    if (parsed.protocol !== 'mlflow-attachment:') {
      return null;
    }
    const attachmentId = parsed.hostname;
    const contentType = parsed.searchParams.get('content_type');
    const traceId = parsed.searchParams.get('trace_id');
    if (!attachmentId || !contentType || !traceId) {
      return null;
    }
    return { attachmentId, contentType, traceId };
  } catch {
    return null;
  }
}
