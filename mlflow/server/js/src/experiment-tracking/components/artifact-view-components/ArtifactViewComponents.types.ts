import type { KeyValueEntity } from '../../../common/types';

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

  /**
   * Indicates if the artifact viewer falls back from displaying run artifacts to logged model artifacts.
   * This is used when the run does not have the artifacts but the related logged model does.
   */
  isFallbackToLoggedModelArtifacts?: boolean;

  experimentId: string;

  entityTags?: Partial<KeyValueEntity>[];
}
