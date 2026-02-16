import { useEffect, useMemo, useState } from 'react';
import type { LoggedModelProto } from '../../types';
import { ExperimentLoggedModelDetailsTracesIntroductionText } from './ExperimentLoggedModelDetailsTracesIntroductionText';
import { TracesViewTableNoTracesQuickstartContextProvider } from '../traces/quickstart/TracesViewTableNoTracesQuickstartContext';
import { TracesV3Logs } from '../experiment-page/components/traces-v3/TracesV3Logs';
import { shouldUseTracesV4API } from '@databricks/web-shared/genai-traces-table';

export const ExperimentLoggedModelDetailsTraces = ({
  loggedModel,
  experimentTags,
  isLoadingExperiment,
}: {
  loggedModel: LoggedModelProto;
  experimentTags?: {
    key: string | null;
    value: string | null;
  }[];
  isLoadingExperiment?: boolean;
}) => {
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
        {/* prettier-ignore */}
        <TracesComponent
          experimentIds={experimentIds}
          loggedModelId={loggedModel.info?.model_id}
          isLoadingExperiment={isLoadingExperiment}
        />
      </TracesViewTableNoTracesQuickstartContextProvider>
    </div>
  );
};

const TracesComponent = ({
  experimentIds,
  loggedModelId,
  isLoadingExperiment,
}: {
  experimentIds: string[];
  loggedModelId: string | undefined;
  isLoadingExperiment?: boolean;
}) => {
  // prettier-ignore
  return experimentIds.length > 0 ? (
    <TracesV3Logs
      experimentId={experimentIds[0]}
      endpointName=""
      loggedModelId={loggedModelId}
      isLoadingExperiment={isLoadingExperiment}
    />
  ) : null;
};
