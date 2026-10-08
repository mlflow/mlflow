import { useCallback } from 'react';
import { getArtifactBlob, getLoggedModelArtifactLocationUrl } from '../../../../common/utils/ArtifactUtils';
import { MLMODEL_FILE_NAME } from '../../../constants';
import type { LoggedModelProto } from '../../../types';
import { fetchArtifactWithPresignedUrl } from '../../../utils/PresignedArtifactUtils';

const lazyJsYaml = () => import('js-yaml');

export const useValidateLoggedModelSignature = (loggedModel?: LoggedModelProto | null) =>
  useCallback(async () => {
    if (!loggedModel?.info?.model_id || !loggedModel?.info?.artifact_uri) {
      return true;
    }

    const artifactLocation = getLoggedModelArtifactLocationUrl(MLMODEL_FILE_NAME, loggedModel.info.model_id);
    const blob = await fetchArtifactWithPresignedUrl(
      {
        runUuid: '',
        path: MLMODEL_FILE_NAME,
        artifactRootUri: loggedModel.info.artifact_uri,
        isLoggedModelsMode: true,
        loggedModelId: loggedModel.info.model_id,
      },
      artifactLocation,
      getArtifactBlob,
    );

    const yamlContent = (await lazyJsYaml()).safeLoad(await blob.text());

    const isValid = yamlContent?.signature?.inputs !== undefined && yamlContent?.signature?.outputs !== undefined;

    return isValid;
  }, [loggedModel]);
