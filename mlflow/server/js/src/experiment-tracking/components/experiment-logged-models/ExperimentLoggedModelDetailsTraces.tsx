import { useEffect, useMemo } from 'react';
import type { LoggedModelProto } from '../../types';
import { TracesView } from '../traces/TracesView';
import { ExperimentLoggedModelDetailsTracesIntroductionText } from './ExperimentLoggedModelDetailsTracesIntroductionText';
import { TracesViewTableNoTracesQuickstartContextProvider } from '../traces/quickstart/TracesViewTableNoTracesQuickstartContext';

export const ExperimentLoggedModelDetailsTraces = ({ loggedModel }: { loggedModel: LoggedModelProto }) => {
  const experimentIds = useMemo(() => [loggedModel.info?.experiment_id ?? ''], [loggedModel.info?.experiment_id]);

  if (!loggedModel.info?.experiment_id) {
    return null;
  }
  return (
    <div css={{ height: '100%', overflow: 'hidden' }}>
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
  return (
    <TracesView
      experimentIds={experimentIds}
      loggedModelId={loggedModelId}
      baseComponentId="mlflow.logged_model.traces"
    />
  );
};
