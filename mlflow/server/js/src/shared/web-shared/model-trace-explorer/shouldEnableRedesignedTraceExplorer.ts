/**
 * Gates the redesigned (v2) model trace explorer. Enabled in OSS; the
 * Databricks build gates it via the enableRedesignedTraceExplorer SAFE flag.
 */
export const shouldEnableRedesignedTraceExplorer = (): boolean => {
  return true;
};
