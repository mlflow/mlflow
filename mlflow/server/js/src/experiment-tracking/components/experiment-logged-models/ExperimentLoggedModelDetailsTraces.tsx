import { useMemo } from 'react';
import type { LoggedModelProto } from '../../types';
import { ExperimentLoggedModelDetailsTracesIntroductionText } from './ExperimentLoggedModelDetailsTracesIntroductionText';
import { TracesViewTableNoTracesQuickstartContextProvider } from '../traces/quickstart/TracesViewTableNoTracesQuickstartContext';
import { TracesV3Logs } from '../experiment-page/components/traces-v3/TracesV3Logs';

export const ExperimentLoggedModelDetailsTraces = ({ loggedModel }: { loggedModel: LoggedModelProto }) => {
  const experimentIds = useMemo(() => [loggedModel.info?.experiment_id ?? ''], [loggedModel.info?.experiment_id]);

  if (!loggedModel.info?.experiment_id) {
    return null;
  }
  return (
    <div css={{ height: '100%' }}>
      <TracesViewTableNoTracesQuickstartContextProvider
        introductionText={
          loggedModel.info?.model_id && (
            <ExperimentLoggedModelDetailsTracesIntroductionText modelId={loggedModel.info.model_id} />
          )
        }
        displayVersionWarnings={false}
      >
        <TracesComponent experimentIds={experimentIds} loggedModelId={loggedModel.info?.model_id} />
      </TracesViewTableNoTracesQuickstartContextProvider>
    </div>
  );
};

const TracesComponent = ({
  experimentIds,
  loggedModelId,
}: {
  experimentIds: string[];
  loggedModelId: string | undefined;
}) => {
  return experimentIds.length > 0 ? (
    <TracesV3Logs experimentId={experimentIds[0]} endpointName="" loggedModelId={loggedModelId} />
  ) : null;
};
