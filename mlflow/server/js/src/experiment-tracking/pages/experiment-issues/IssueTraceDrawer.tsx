import React, { useCallback, useMemo, useRef, type ReactNode } from 'react';
import { GenericSkeleton } from '@databricks/design-system';
import {
  convertTraceInfoV3ToRunEvalEntry,
  GenAiEvaluationTracesReviewModal,
  useActiveEvaluation,
  type AssessmentInfo,
  type EvalTraceComparisonEntry,
} from '../../../shared/web-shared/genai-traces-table';
import {
  ModelTraceExplorerContextProvider,
  ModelTraceExplorerPreferencesProvider,
  shouldEnableModelTraceExplorerCustomTraceView,
  type ModelTraceInfoV3,
} from '../../../shared/web-shared/model-trace-explorer';
import { AssistantAwareDrawer } from '../../../common/components/AssistantAwareDrawer';
import { useSearchParams } from '../../../common/utils/RoutingUtils';
import { SELECTED_ISSUE_SPAN_ID_PARAM } from '../../constants';
import { getTrace } from '../../utils/TraceUtils';
import { ExportTracesToDatasetModal } from '../experiment-evaluation-datasets/components/ExportTracesToDatasetModal';
import { AddToReviewQueueDropdown } from '../experiment-review-queue/AddToReviewQueueDropdown';
import { getAddressableTraceId } from './issueEvidence.utils';

// Custom View pulls in @a2ui (ESM-only) transitively via ExperimentCustomViewProvider.
// Lazy-load it so the Issues tab doesn't carry @a2ui on its static module graph
// (mirrors the V3 and V4 traces pages).
const LazyExperimentCustomViewProvider = React.lazy(() =>
  import('../../components/experiment-page/components/traces-v3/ExperimentCustomViewProvider').then((module) => ({
    default: module.ExperimentCustomViewProvider,
  })),
);

/**
 * The contexts the trace drawer's own controls read, which a traces table would otherwise supply.
 *
 * The drawer is a shared component whose header and span tree reach for host-provided behavior:
 * display-metric preferences, the Add to dataset modal, the review-queue picker and the experiment's
 * saved custom views. Their defaults are inert -- no-op setters and a modal renderer that returns
 * nothing -- so a host that omits them gets controls that render and do nothing rather than controls
 * that stay hidden. Mounted above the detail pane, not inside the drawer, so a preference set while
 * reading one trace survives closing the drawer and opening another, as it does on the Traces tab.
 */
export const IssueTraceDrawerProviders = ({
  experimentId,
  children,
}: {
  experimentId: string;
  children: ReactNode;
}) => {
  const content = shouldEnableModelTraceExplorerCustomTraceView() ? (
    <React.Suspense fallback={<GenericSkeleton css={{ flex: 1, margin: 16 }} />}>
      <LazyExperimentCustomViewProvider key={experimentId} experimentId={experimentId}>
        {children}
      </LazyExperimentCustomViewProvider>
    </React.Suspense>
  ) : (
    children
  );

  return (
    <ModelTraceExplorerPreferencesProvider>
      <ModelTraceExplorerContextProvider
        renderExportTracesToDatasetsModal={ExportTracesToDatasetModal}
        renderAddToReviewQueueDropdown={AddToReviewQueueDropdown}
        DrawerComponent={AssistantAwareDrawer}
      >
        {content}
      </ModelTraceExplorerContextProvider>
    </ModelTraceExplorerPreferencesProvider>
  );
};

/**
 * The drawer's assessment editing is driven by the assessment columns a traces *table* aggregates
 * from its search results, and this pane has no table. Empty is what makes the drawer render the
 * trace explorer alone, which is exactly what an Evidence card links the reader to.
 */
const NO_ASSESSMENT_INFOS: AssessmentInfo[] = [];

export interface IssueTraceDrawerProps {
  experimentId: string;
  /**
   * The affected traces whose info has arrived, in the issue's own order. These are what the
   * drawer's Previous/Next walk, so the reader can only step to a trace they can also see.
   */
  traceInfos: ModelTraceInfoV3[];
  /** The span the issue blamed for one of its traces, by trace id; `undefined` where it named none. */
  getOffendingSpanId: (traceId: string) => string | undefined;
  /** Writes the drawer's selection the way an Evidence card does -- trace and span together. */
  onSelectTrace: (linkableTraceId: string, spanId?: string) => void;
}

/**
 * Hosts the single-trace drawer for the issue pane, so `Open trace` on a card works regardless of
 * what the rest of the page is doing: the cards do not depend on a traces search, so neither does
 * the drawer they open.
 */
export const IssueTraceDrawer = ({
  experimentId,
  traceInfos,
  getOffendingSpanId,
  onSelectTrace,
}: IssueTraceDrawerProps) => {
  // Resolves against the page's search params, so the selection stays in the URL and a copied
  // link reopens the same trace on the same span.
  const [selectedTraceId, setSelectedTraceId] = useActiveEvaluation();
  const [params] = useSearchParams();
  const selectedSpanId = params.get(SELECTED_ISSUE_SPAN_ID_PARAM) ?? undefined;

  const evaluations = useMemo<EvalTraceComparisonEntry[]>(
    () => traceInfos.map((info) => ({ currentRunValue: convertTraceInfoV3ToRunEvalEntry(info) })),
    [traceInfos],
  );

  // Keyed by both forms, because a selection arrives as either: a card and the URL carry the
  // addressable id, while the drawer's own arrows hand back the bare `evaluationId`.
  const infoByTraceId = useMemo(() => {
    const byTraceId = new Map<string, ModelTraceInfoV3>();
    traceInfos.forEach((info) => {
      byTraceId.set(info.trace_id, info);
      byTraceId.set(getAddressableTraceId(info), info);
    });
    return byTraceId;
  }, [traceInfos]);

  // Held in a ref, not a dependency: this is rebuilt whenever the Evidence section's span reads
  // settle, and stepping through traces must not depend on a card's fetch state.
  const getOffendingSpanIdRef = useRef(getOffendingSpanId);
  getOffendingSpanIdRef.current = getOffendingSpanId;

  // A step writes the trace and its own offending span, through the same callback an Evidence card
  // uses, so stepping to a trace and clicking its card agree. Closing drops both together.
  const handleChangeEvaluationId = useCallback(
    (nextTraceId: string | undefined) => {
      if (!nextTraceId) {
        setSelectedTraceId(undefined);
        return;
      }
      const nextInfo = infoByTraceId.get(nextTraceId);
      onSelectTrace(
        nextInfo ? getAddressableTraceId(nextInfo) : nextTraceId,
        getOffendingSpanIdRef.current(nextInfo?.trace_id ?? nextTraceId),
      );
    },
    [setSelectedTraceId, infoByTraceId, onSelectTrace],
  );

  if (!selectedTraceId) {
    return null;
  }

  return (
    <GenAiEvaluationTracesReviewModal
      experimentId={experimentId}
      evaluations={evaluations}
      selectedEvaluationId={selectedTraceId}
      onChangeEvaluationId={handleChangeEvaluationId}
      assessmentInfos={NO_ASSESSMENT_INFOS}
      getTrace={getTrace}
      selectedSpanId={selectedSpanId}
    />
  );
};
