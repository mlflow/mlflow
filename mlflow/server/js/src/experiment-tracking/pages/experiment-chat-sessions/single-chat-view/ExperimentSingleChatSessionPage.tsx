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

import { useParams, useLocation } from '@mlflow/mlflow/src/common/utils/RoutingUtils';
import invariant from 'invariant';
import { useGetExperimentQuery } from '../../../hooks/useExperimentQuery';
import { useCallback, useMemo, useRef, useState, useEffect } from 'react';
import { ExperimentSingleChatSessionScoreResults } from './ExperimentSingleChatSessionScoreResults';
import { TracesV3Toolbar } from '../../../components/experiment-page/components/traces-v3/TracesV3Toolbar';
import type { ModelTrace } from '@databricks/web-shared/model-trace-explorer';
import {
  getModelTraceId,
  isV3ModelTraceInfo,
  ModelTraceExplorer,
  ModelTraceExplorerUpdateTraceContextProvider,
  shouldEnableAssessmentsInSessions,
  shouldUseTracesV4API,
} from '@databricks/web-shared/model-trace-explorer';
import {
  ExperimentSingleChatSessionSidebar,
  ExperimentSingleChatSessionSidebarSkeleton,
} from './ExperimentSingleChatSessionSidebar';
import { getTrace as getTraceV3 } from '@mlflow/mlflow/src/experiment-tracking/utils/TraceUtils';
import { getChatSessionsFilter } from '../utils';
import {
  ExperimentSingleChatConversation,
  ExperimentSingleChatConversationSkeleton,
} from './ExperimentSingleChatConversation';
import { Drawer, useDesignSystemTheme } from '@databricks/design-system';
import { SELECTED_TRACE_ID_QUERY_PARAM } from '../../../constants';
import { useExperimentSingleChatMetrics } from './useExperimentSingleChatMetrics';
import { ExperimentSingleChatSessionMetrics } from './ExperimentSingleChatSessionMetrics';
import { useRegisterAssistantContext } from '@mlflow/mlflow/src/assistant';

const ContextProviders = ({
  children,
  invalidateTraceQuery,
}: {
  children: React.ReactNode;
  invalidateTraceQuery?: (traceId?: string) => void;
}) => {
  return (
    <ModelTraceExplorerUpdateTraceContextProvider invalidateTraceQuery={invalidateTraceQuery}>
      {children}
    </ModelTraceExplorerUpdateTraceContextProvider>
  );
};

const ExperimentSingleChatSessionPageImpl = () => {
  const { theme } = useDesignSystemTheme();
  const { experimentId, sessionId } = useParams();
  const location = useLocation();
  const [selectedTurnIndex, setSelectedTurnIndex] = useState<number | null>(null);
  const [selectedTrace, setSelectedTrace] = useState<ModelTrace | null>(null);
  const chatRefs = useRef<{ [traceId: string]: HTMLDivElement }>({});

  invariant(experimentId, 'Experiment ID must be defined');
  invariant(sessionId, 'Session ID must be defined');

  useRegisterAssistantContext('sessionId', sessionId);

  const selectedTraceIdFromUrl = useMemo(() => {
    const searchParams = new URLSearchParams(location.search);
    return searchParams.get(SELECTED_TRACE_ID_QUERY_PARAM);
  }, [location.search]);

  const { loading: isLoadingExperiment } = useGetExperimentQuery({
    experimentId,
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

  const chatSessionMetrics = useExperimentSingleChatMetrics({ traceInfos: sortedTraceInfos });

  const getTrace = getTraceV3;
  const getAssessmentTitle = useCallback((assessmentName: string) => assessmentName, []);
  const {
    data: traces,
    isLoading: isLoadingTraceDatas,
    invalidateSingleTraceQuery,
  } = useGetTraces(getTrace, sortedTraceInfos);

  useEffect(() => {
    if (selectedTraceIdFromUrl && traces && traces.length > 0 && !isLoadingTraceDatas) {
      const traceIndex = traces.findIndex((trace) => getModelTraceId(trace) === selectedTraceIdFromUrl);
      if (traceIndex !== -1) {
        setSelectedTurnIndex(traceIndex);
        const traceRef = chatRefs.current[selectedTraceIdFromUrl];
        if (traceRef) {
          traceRef.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
      }
    }
  }, [selectedTraceIdFromUrl, traces, isLoadingTraceDatas]);

  return (
    <ContextProviders // prettier-ignore
      invalidateTraceQuery={invalidateSingleTraceQuery}
    >
      <div css={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0 }}>
        <TracesV3Toolbar
          // prettier-ignore
          viewState="single-chat-session"
          sessionId={sessionId}
          css={
            shouldEnableAssessmentsInSessions()
              ? {
                  borderBottom: 'none',
                }
              : undefined
          }
        />

        {shouldEnableAssessmentsInSessions() && (
          <ExperimentSingleChatSessionMetrics chatSessionMetrics={chatSessionMetrics} />
        )}
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
              getAssessmentTitle={getAssessmentTitle}
            />
            {shouldEnableAssessmentsInSessions() && (
              <ExperimentSingleChatSessionScoreResults traces={traces ?? []} sessionId={sessionId} />
            )}
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
              {selectedTrace && <ModelTraceExplorer modelTrace={selectedTrace} collapseAssessmentPane="force-open" />}
            </div>
          </Drawer.Content>
        </Drawer.Root>
      </div>
    </ContextProviders>
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
