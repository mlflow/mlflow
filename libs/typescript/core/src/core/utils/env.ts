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
