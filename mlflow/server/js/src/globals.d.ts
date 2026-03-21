export type Timeout = ReturnType<typeof window.setTimeout>;

declare global {
  //
  interface Window {
    /** Used by a few unit tests  */
    isTestingIframe?: boolean;
  }
}
