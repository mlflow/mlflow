/**
 * Known SQL warehouse timeout error message patterns from the backend.
 * These correspond to timeout errors in MlflowSqlExecUtils.scala.
 */
const SQL_TIMEOUT_PATTERNS = [
  'Timeout while getting statement result',
  'Timeout while issuing SQL query',
  'Timeout while waiting for SQL query to complete',
  'underlying SQL request timed out',
];

/**
 * Detects whether the given error is a SQL warehouse timeout error.
 * This helps distinguish timeout errors (which may be resolved by selecting
 * a larger warehouse) from other types of failures.
 */
export const isSqlWarehouseTimeoutError = (error: Error | undefined | null): boolean => {
  if (!error?.message) return false;
  return SQL_TIMEOUT_PATTERNS.some((pattern) => error.message.includes(pattern));
};
