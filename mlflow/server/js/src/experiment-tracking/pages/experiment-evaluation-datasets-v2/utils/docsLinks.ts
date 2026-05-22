/**
 * Build a docs URL under the OSS MLflow docs site. OSS only has one docs host (unlike
 * universe, which routes per-cloud), so this is a simple string concat.
 */
const OSS_DOCS_BASE = 'https://mlflow.org/docs/latest';

export const getEvalMonitorDocsLink = (suffix: string): string => {
  return `${OSS_DOCS_BASE}/genai/eval-monitor/${suffix}`;
};
