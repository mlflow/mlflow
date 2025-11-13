export const shouldBlockLargeTraceDisplay = () => {
  return false;
};

// controls the size (in bytes) of a trace that is considered too large
// to display. default to 1gb for a safe limit to always display traces
export const getLargeTraceDisplaySizeThreshold = () => {
  return 1e9;
};

/**
 * Determines if traces V4 API should be used to fetch traces
 */
export const shouldUseTracesV4API = () => {
  return false;
};

/**
 * Determines if the new labeling schemas UI in trace assessments pane is enabled.
 * This feature allows users to configure feedback schemas at the experiment level
 * for labeling traces in the Traces tab.
 */
export const shouldEnableTracesTabLabelingSchemas = () => {
  return false;
};

export const shouldEnableChatSessionsTab = () => {
  return false;
};
