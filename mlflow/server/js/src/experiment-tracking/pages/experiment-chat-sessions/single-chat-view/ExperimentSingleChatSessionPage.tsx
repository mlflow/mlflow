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
// BEGIN-EDGE
// Commented out because these don't exist in OSS:
// import { useFetchTraceV4LazyQuery } from '@databricks/web-shared/genai-traces-table';
// import { useMonitoringSqlWarehouseId } from '../../experiment-evaluation-monitoring/hooks/useMonitoringSqlWarehouseId';
// import { useUpdateExperimentUCSchemaStorage } from '../../../components/experiment-page/components/traces-v3/hooks/useUpdateExperimentUCSchemaStorage';
// import { LabelingSchemaContextProvider } from '@databricks/web-shared/model-trace-explorer';
// import { useExperimentLabelingSchemas } from '../../../hooks/useExperimentLabelingSchemas';
// END-EDGE
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
import { shouldEnableChatSessionsTab } from '@mlflow/mlflow/src/common/utils/FeatureUtils';
import { getTrace as getTraceV3 } from '@mlflow/mlflow/src/experiment-tracking/utils/TraceUtils';
import { getChatSessionsFilter } from '../utils';
import {
  ExperimentSingleChatConversation,
  ExperimentSingleChatConversationSkeleton,
} from './ExperimentSingleChatConversation';
import { Drawer, useDesignSystemTheme } from '@databricks/design-system';
import { useExperimentSingleChatMetrics } from './useExperimentSingleChatMetrics';
import { ExperimentSingleChatSessionMetrics } from './ExperimentSingleChatSessionMetrics';
// BEGIN-EDGE
// Commented out because LabelingSchemaContextProvider and useExperimentLabelingSchemas don't exist in OSS:
// const ContextProviders = ({
//   sqlWarehouseId,
//   modelTraceInfo,
//   children,
//   labelingSchemasConfig,
//   invalidateTraceQuery,
// }: {
//   sqlWarehouseId?: string;
//   modelTraceInfo?: ModelTrace['info'];
//   children: React.ReactNode;
//   labelingSchemasConfig: ReturnType<typeof useExperimentLabelingSchemas>;
//   invalidateTraceQuery?: (traceId?: string) => void;
// }) => {
//   return (
//     <ModelTraceExplorerUpdateTraceContextProvider
//       sqlWarehouseId={sqlWarehouseId}
//       modelTraceInfo={modelTraceInfo}
//       invalidateTraceQuery={invalidateTraceQuery}
//     >
//       <LabelingSchemaContextProvider
//         schemas={labelingSchemasConfig.schemas}
//         allAvailableSchemas={labelingSchemasConfig.allAvailableSchemas}
//         isLoading={labelingSchemasConfig.isLoading}
//         onAddSchema={labelingSchemasConfig.addSchema}
//         onRemoveSchema={labelingSchemasConfig.removeSchema}
//       >
//         {children}
//       </LabelingSchemaContextProvider>
//     </ModelTraceExplorerUpdateTraceContextProvider>
//   );
// };
// END-EDGE

const oss_ContextProviders = ({
  children,
  modelTraceInfo,
  invalidateTraceQuery,
}: {
  children: React.ReactNode;
  modelTraceInfo?: ModelTrace['info'];
  invalidateTraceQuery?: (traceId?: string) => void;
}) => {
  return (
    <ModelTraceExplorerUpdateTraceContextProvider
      modelTraceInfo={modelTraceInfo}
      invalidateTraceQuery={invalidateTraceQuery}
    >
      {children}
    </ModelTraceExplorerUpdateTraceContextProvider>
  );
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
  });

  // BEGIN-EDGE
  // Commented out because useUpdateExperimentUCSchemaStorage doesn't exist in OSS:
  // const { storageUCSchema, setStorageUCSchema, isUpdatingStorageUCSchema } = useUpdateExperimentUCSchemaStorage({
  //   experimentId,
  // });
  // END-EDGE
  const traceSearchLocations = useMemo(
    () => {
      // BEGIN-EDGE
      // Commented out because UC schema doesn't exist in OSS:
      // // If the experiment is still loading, wait with determining the trace location
      // if (isLoadingExperiment && shouldUseTracesV4API()) {
      //   return [];
      // }
      // if (storageUCSchema && shouldUseTracesV4API()) {
      //   return [createTraceLocationForUCSchema(storageUCSchema)];
      // }
      // END-EDGE
      return [createTraceLocationForExperiment(experimentId)];
    },
    // prettier-ignore
    [
      experimentId,
      // BEGIN-EDGE
      // storageUCSchema,
      // isLoadingExperiment,
      // END-EDGE
    ],
  );

  const filters = useMemo(() => getChatSessionsFilter({ sessionId }), [sessionId]);

  // BEGIN-EDGE
  // Commented out because useMonitoringSqlWarehouseId doesn't exist in OSS:
  // const [selectedWarehouseId, setSelectedWarehouseId] = useMonitoringSqlWarehouseId();
  // const [warehousesLoading, setWarehousesLoading] = useState(false);
  // END-EDGE
  const { data: traceInfos, isLoading: isLoadingTraceInfos } = useSearchMlflowTraces({
    locations: traceSearchLocations,
    filters,
    disabled: false,
    // BEGIN-EDGE
    // sqlWarehouseId: selectedWarehouseId || undefined,
    // END-EDGE
  });

  const sortedTraceInfos = useMemo(() => {
    return traceInfos?.sort((a, b) => new Date(a.request_time).getTime() - new Date(b.request_time).getTime());
  }, [traceInfos]);

  const chatSessionMetrics = useExperimentSingleChatMetrics({ traceInfos: sortedTraceInfos });

  // BEGIN-EDGE
  // Commented out because useFetchTraceV4LazyQuery and useExperimentLabelingSchemas don't exist in OSS:
  // // A function used to fetch trace using V4 API
  // const getTraceV4 = useFetchTraceV4LazyQuery({ selectedSqlWarehouseId: selectedWarehouseId || undefined });
  //
  // // A wrapper function that decides whether to use V4 API or original API to fetch trace
  // const getTrace = useCallback<GetTraceFunction>(
  //   (traceId, traceInfo) => {
  //     if (shouldUseTracesV4API() && traceInfo && isV3ModelTraceInfo(traceInfo) && doesTraceSupportV4API(traceInfo)) {
  //       return getTraceV4(traceInfo);
  //     }
  //     return getTraceV3(traceId, traceInfo);
  //   },
  //   [getTraceV4],
  // );
  //
  // const labelingSchemasConfig = useExperimentLabelingSchemas(experimentId);
  //
  // const getAssessmentTitle = useCallback(
  //   (assessmentName: string) => {
  //     const matchingSchema = labelingSchemasConfig.schemas?.find((schema) => schema?.name === assessmentName);
  //     if (matchingSchema) {
  //       return matchingSchema.title ?? assessmentName;
  //     }
  //     return assessmentName;
  //   },
  //   [labelingSchemasConfig.schemas],
  // );
  // END-EDGE
  const getTrace = getTraceV3;
  const getAssessmentTitle = useCallback((assessmentName: string) => assessmentName, []);
  const {
    data: traces,
    isLoading: isLoadingTraceDatas,
    invalidateSingleTraceQuery,
  } = useGetTraces(getTrace, sortedTraceInfos);

  const firstTraceInfo = traces?.[0]?.info;

  return (
    <oss_ContextProviders modelTraceInfo={firstTraceInfo} invalidateTraceQuery={invalidateSingleTraceQuery}>
      <div css={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0 }}>
        <div
          css={
            shouldEnableAssessmentsInSessions()
              ? {
                  '& > div': {
                    borderBottom: 'none',
                  },
                }
              : undefined
          }
        >
          <TracesV3Toolbar
            // prettier-ignore
            viewState="single-chat-session"
            sessionId={sessionId}
            // BEGIN-EDGE
            // Commented out EDGE-only props:
            // experimentId={experimentId}
            // selectedWarehouseId={selectedWarehouseId || undefined}
            // setSelectedWarehouseId={setSelectedWarehouseId}
            // setWarehousesLoading={setWarehousesLoading}
            // isListMonitorsLoading={false}
            // isShinkansenEnabled
            // shinkansenTraceArchiveTableName={undefined}
            // selectedStorageUCSchema={storageUCSchema}
            // setSelectedStorageUCSchema={setStorageUCSchema}
            // isUpdatingStorageUCSchema={isUpdatingStorageUCSchema}
            // END-EDGE
          />
        </div>

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
              <oss_ContextProviders
                modelTraceInfo={selectedTrace?.info}
                invalidateTraceQuery={invalidateSingleTraceQuery}
              >
                {selectedTrace && <ModelTraceExplorer modelTrace={selectedTrace} collapseAssessmentPane="force-open" />}
              </oss_ContextProviders>
            </div>
          </Drawer.Content>
        </Drawer.Root>
      </div>
    </oss_ContextProviders>
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
