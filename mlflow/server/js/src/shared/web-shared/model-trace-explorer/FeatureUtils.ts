export const shouldBlockLargeTraceDisplay = () => {
  return false;
};

// controls the size (in bytes) of a trace that is considered too large
// to display. default to 1gb for a safe limit to always display traces
export const getLargeTraceDisplaySizeThreshold = () => {
  return 1e9;
};
