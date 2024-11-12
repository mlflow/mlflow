export interface LoggedModelArtifactViewerProps {
  /**
   * Artifact viewer can also work with logged models instead of runs.
   * Set this to `true` to enable the logged models mode.
   */
  isLoggedModelsMode?: boolean;

  /**
   * ID of the logged model to display artifacts for. Works only in logged models mode.
   */
  loggedModelId?: string;
}
