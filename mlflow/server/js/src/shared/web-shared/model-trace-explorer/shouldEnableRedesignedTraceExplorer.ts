/**
 * Gates the redesigned (v2) model trace explorer. Off by default in OSS; the
 * Databricks build flips it via the enableRedesignedTraceExplorer SAFE flag.
 */
export const shouldEnableRedesignedTraceExplorer = (): boolean => {
  return false;
};
