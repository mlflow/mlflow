/**
 * Returns the OSS docs URL for a known eval-dataset doc anchor. Kept as a lookup (not a
 * `${base}/${suffix}` concat) because the OSS docs site organizes datasets under
 * `/genai/datasets` rather than the `/genai/eval-monitor/<page>` layout universe uses.
 */
const OSS_DOCS_BASE = 'https://mlflow.org/docs/latest';

const EVAL_MONITOR_DOC_PATHS = {
  'build-eval-dataset': '/genai/datasets',
} satisfies Record<string, string>;

type KnownDocAnchor = keyof typeof EVAL_MONITOR_DOC_PATHS;

const isKnownDocAnchor = (suffix: string): suffix is KnownDocAnchor =>
  Object.prototype.hasOwnProperty.call(EVAL_MONITOR_DOC_PATHS, suffix);

export const getEvalMonitorDocsLink = (suffix: string): string => {
  if (!isKnownDocAnchor(suffix)) {
    // Caller passed an anchor we don't know — fall back to the docs root so the link
    // still resolves to something useful instead of 404ing.
    return OSS_DOCS_BASE;
  }
  return `${OSS_DOCS_BASE}${EVAL_MONITOR_DOC_PATHS[suffix]}`;
};
