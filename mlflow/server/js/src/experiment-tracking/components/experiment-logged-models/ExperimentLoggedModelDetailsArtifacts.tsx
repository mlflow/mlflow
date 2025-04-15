import type { LoggedModelProto } from '../../types';
import ArtifactPage from '../ArtifactPage';

export const ExperimentLoggedModelDetailsArtifacts = ({ loggedModel }: { loggedModel: LoggedModelProto }) => (
  <div css={{ height: '100%', overflow: 'hidden', display: 'flex' }}>
    <ArtifactPage
      isLoggedModelsMode
      loggedModelId={loggedModel.info?.model_id ?? ''}
      artifactRootUri={loggedModel?.info?.artifact_uri ?? ''}
      useAutoHeight
    />
  </div>
);
