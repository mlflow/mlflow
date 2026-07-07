import { useCallback } from 'react';
import { getArtifactBlob, getLoggedModelArtifactLocationUrl } from '../../../../common/utils/ArtifactUtils';
import { MLMODEL_FILE_NAME } from '../../../constants';
import type { LoggedModelProto } from '../../../types';

const lazyJsYaml = () => import('js-yaml');

export const useValidateLoggedModelSignature = (loggedModel?: LoggedModelProto | null) =>
  useCallback(async () => {
    if (!loggedModel?.info?.model_id || !loggedModel?.info?.artifact_uri) {
      return true;
    }

    const artifactLocation = getLoggedModelArtifactLocationUrl(MLMODEL_FILE_NAME, loggedModel.info.model_id);
    const blob = await getArtifactBlob(artifactLocation);

    const yamlContent = (await lazyJsYaml()).load(await blob.text());

    const isValid = yamlContent?.signature?.inputs !== undefined && yamlContent?.signature?.outputs !== undefined;

    return isValid;
  }, [loggedModel]);
