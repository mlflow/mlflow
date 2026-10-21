import {
  type ArtifactRequestOptions,
  getArtifactContent,
  getArtifactLocationUrl,
  getLoggedModelArtifactLocationUrl,
} from '../../../../common/utils/ArtifactUtils';
import { fetchArtifactPreferPresigned } from '../../../utils/PresignedArtifactUtils';

export type FetchArtifactParams = {
  runUuid: string;
  path: string;
  isLoggedModelsMode?: boolean;
  loggedModelId?: string;
  artifactRootUri?: string;
};

type GetArtifactContentFn<T = unknown> = (artifactLocation: string, options?: ArtifactRequestOptions) => Promise<T>;

// Internal util that generates the artifact location URL for the workspace API
const getWorkspaceArtifactLocationUrl = (params: FetchArtifactParams) => {
  const { runUuid, path, isLoggedModelsMode, loggedModelId } = params;
  if (isLoggedModelsMode && loggedModelId) {
    return getLoggedModelArtifactLocationUrl(path, loggedModelId);
  }
  return getArtifactLocationUrl(path, runUuid);
};

/**
 * Fetch a run or logged-model artifact through the configured artifact service.
 */
export function fetchArtifactUnified(params: FetchArtifactParams): Promise<string>;
export function fetchArtifactUnified<T>(
  params: FetchArtifactParams,
  getArtifactDataFn: GetArtifactContentFn<T>,
): Promise<T>;
export function fetchArtifactUnified(
  params: FetchArtifactParams,
  getArtifactDataFn: GetArtifactContentFn = getArtifactContent,
): Promise<unknown> {
  const workspaceAPIArtifactLocation = getWorkspaceArtifactLocationUrl(params);

  return fetchArtifactPreferPresigned(params, workspaceAPIArtifactLocation, getArtifactDataFn);
}

export type FetchArtifactUnifiedFn<T = string> = (
  params: FetchArtifactParams,
  getArtifactDataFn: GetArtifactContentFn<T>,
) => Promise<T>;
