/**
 * Read an env var and require a strictly positive integer.
 */
export const readPositiveInt = (envName: string, fallback: number): number => {
  const raw = process.env[envName];
  if (raw === undefined || raw === '') {
    return fallback;
  }
  // `Number` (vs `Number.parseInt`) refuses trailing garbage and unit
  // suffixes — e.g. "15000abc", "15s", and "10.9" all become NaN or a
  // non-integer, so the `isInteger` check below rejects them cleanly.
  const parsed = Number(raw);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    console.warn(
      `[mlflow] Ignoring invalid ${envName}=${JSON.stringify(raw)}; ` +
        `expected a positive integer. Falling back to default ${fallback}.`,
    );
    return fallback;
  }
  return parsed;
};

export const readNonNegativeInt = (envName: string, fallback: number): number => {
  const raw = process.env[envName];
  if (raw === undefined || raw === '') {
    return fallback;
  }
  const parsed = Number(raw);
  if (!Number.isInteger(parsed) || parsed < 0) {
    console.warn(
      `[mlflow] Ignoring invalid ${envName}=${JSON.stringify(raw)}; ` +
        `expected a non-negative integer. Falling back to default ${fallback}.`,
    );
    return fallback;
  }
  return parsed;
};

/**
 * Whether to route trace export through an on-disk WAL queue
 * drained by a background daemon, instead of the synchronous
 * HTTP exporter.
 */
export function asyncExportEnabled(): boolean {
  const raw = process.env.MLFLOW_ENABLE_ASYNC_TRACE_LOGGING;
  if (raw === undefined || raw === '') {
    return false;
  }
  const lower = raw.toLowerCase();
  return lower === '1' || lower === 'true';
}
