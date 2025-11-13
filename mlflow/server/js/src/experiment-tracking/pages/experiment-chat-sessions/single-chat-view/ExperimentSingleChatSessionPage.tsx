import ErrorUtils from '@mlflow/mlflow/src/common/utils/ErrorUtils';
import { withErrorBoundary } from '@mlflow/mlflow/src/common/utils/withErrorBoundary';
import { FormattedMessage } from '@mlflow/mlflow/src/i18n/i18n';
import type { GetTraceFunction } from '@databricks/web-shared/genai-traces-table';
import {
  createTraceLocationForExperiment,
  createTraceLocationForUCSchema,
  doesTraceSupportV4API,
  useGetTraces,
  useSearchMlflowTraces,
} from '@databricks/web-shared/genai-traces-table';

import { useParams } from '@mlflow/mlflow/src/common/utils/RoutingUtils';
import invariant from 'invariant';
import { useGetExperimentQuery } from '../../../hooks/useExperimentQuery';
import { useCallback, useMemo, useRef, useState } from 'react';
import { TracesV3Toolbar } from '../../../components/experiment-page/components/traces-v3/TracesV3Toolbar';
import type { ModelTrace } from '@databricks/web-shared/model-trace-explorer';
import {
  getModelTraceId,
  isV3ModelTraceInfo,
  ModelTraceExplorer,
  ModelTraceExplorerUpdateTraceContextProvider,
  shouldUseTracesV4API,
} from '@databricks/web-shared/model-trace-explorer';
import {
  ExperimentSingleChatSessionSidebar,
  ExperimentSingleChatSessionSidebarSkeleton,
} from './ExperimentSingleChatSessionSidebar';
import { shouldEnableChatSessionsTab } from '@mlflow/mlflow/src/common/utils/FeatureUtils';
import { getTrace as getTraceV3 } from '@mlflow/mlflow/src/experiment-tracking/utils/TraceUtils';
import { getChatSessionsFilter } from '../utils';
import {
  ExperimentSingleChatConversation,
  ExperimentSingleChatConversationSkeleton,
} from './ExperimentSingleChatConversation';
import { Drawer, useDesignSystemTheme } from '@databricks/design-system';

const ContextProviders = ({ children }: { children: React.ReactNode }) => {
  return <>{children}</>;
};

const ExperimentSingleChatSessionPageImpl = () => {
  const { theme } = useDesignSystemTheme();
  const { experimentId, sessionId } = useParams();
  const [selectedTurnIndex, setSelectedTurnIndex] = useState<number | null>(null);
  const [selectedTrace, setSelectedTrace] = useState<ModelTrace | null>(null);
  const chatRefs = useRef<{ [traceId: string]: HTMLDivElement }>({});

  invariant(experimentId, 'Experiment ID must be defined');
  invariant(sessionId, 'Session ID must be defined');

  const { loading: isLoadingExperiment } = useGetExperimentQuery({
    experimentId,
    options: {
      fetchPolicy: 'cache-only',
    },
  });

  const traceSearchLocations = useMemo(
    () => {
      return [createTraceLocationForExperiment(experimentId)];
    },
    // prettier-ignore
    [
      experimentId,
    ],
  );

  const filters = useMemo(() => getChatSessionsFilter({ sessionId }), [sessionId]);

  const { data: traceInfos, isLoading: isLoadingTraceInfos } = useSearchMlflowTraces({
    locations: traceSearchLocations,
    filters,
    disabled: false,
  });

  const sortedTraceInfos = useMemo(() => {
    return traceInfos?.sort((a, b) => new Date(a.request_time).getTime() - new Date(b.request_time).getTime());
  }, [traceInfos]);

  const getTrace = getTraceV3;
  const { data: traces, isLoading: isLoadingTraceDatas } = useGetTraces(getTrace, sortedTraceInfos);

  if (!shouldEnableChatSessionsTab()) {
    return <div />;
  }

  return (
    <div css={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0 }}>
      <TracesV3Toolbar
        // prettier-ignore
        viewState="single-chat-session"
        sessionId={sessionId}
      />
      {isLoadingTraceDatas || isLoadingTraceInfos ? (
        <div css={{ display: 'flex', flex: 1, minHeight: 0 }}>
          <ExperimentSingleChatSessionSidebarSkeleton />
          <ExperimentSingleChatConversationSkeleton />
        </div>
      ) : (
        <div css={{ display: 'flex', flex: 1, minHeight: 0 }}>
          <ExperimentSingleChatSessionSidebar
            traces={traces ?? []}
            selectedTurnIndex={selectedTurnIndex}
            setSelectedTurnIndex={setSelectedTurnIndex}
            setSelectedTrace={setSelectedTrace}
            chatRefs={chatRefs}
          />
          <ExperimentSingleChatConversation
            traces={traces ?? []}
            selectedTurnIndex={selectedTurnIndex}
            setSelectedTurnIndex={setSelectedTurnIndex}
            setSelectedTrace={setSelectedTrace}
            chatRefs={chatRefs}
          />
        </div>
      )}
      <Drawer.Root
        open={selectedTrace !== null}
        onOpenChange={(open) => {
          if (!open) {
            setSelectedTrace(null);
          }
        }}
      >
        <Drawer.Content
          componentId="mlflow.experiment.chat-session.trace-drawer"
          title={selectedTrace ? getModelTraceId(selectedTrace) : ''}
          width="90vw"
          expandContentToFullHeight
        >
          <div
            css={{
              height: '100%',
              marginLeft: -theme.spacing.lg,
              marginRight: -theme.spacing.lg,
              marginBottom: -theme.spacing.lg,
            }}
          >
            <ContextProviders // prettier-ignore
            >
              {selectedTrace && <ModelTraceExplorer modelTrace={selectedTrace as ModelTrace} />}
            </ContextProviders>
          </div>
        </Drawer.Content>
      </Drawer.Root>
    </div>
  );
};

const ExperimentSingleChatSessionPage = withErrorBoundary(
  ErrorUtils.mlflowServices.CHAT_SESSIONS,
  ExperimentSingleChatSessionPageImpl,
  <FormattedMessage
    defaultMessage="An error occurred while rendering the chat session."
    description="Generic error message for uncaught errors when rendering a single chat session in MLflow experiment page"
  />,
);

export default ExperimentSingleChatSessionPage;
