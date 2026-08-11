import { type ReactNode } from 'react';
import type { RenderExportTracesToDatasetsModalParams } from '@databricks/web-shared/model-trace-explorer';
import { isEvaluatingTracesInDetailsViewEnabled } from '@databricks/web-shared/model-trace-explorer';
import { type GetTraceFunction } from '@databricks/web-shared/genai-traces-table';
import { getTrace as getTraceV3 } from '@mlflow/mlflow/src/experiment-tracking/utils/TraceUtils';
import {
  useRunScorerInTracesViewConfiguration,
  useRunJudgesOnTracesConfiguration,
} from '@mlflow/mlflow/src/experiment-tracking/pages/experiment-scorers/hooks/useRunScorerInTracesViewConfiguration';
import { ExportTracesToDatasetModal } from '@mlflow/mlflow/src/experiment-tracking/pages/experiment-evaluation-datasets/components/ExportTracesToDatasetModal';
import { AddToReviewQueueDropdown } from '@mlflow/mlflow/src/experiment-tracking/pages/experiment-review-queue/AddToReviewQueueDropdown';

/**
 * The shared trace-action building blocks the v4 Traces tab wires into both the detail drawer and
 * the toolbar's "Actions" menu. Assembled once (see {@link useTracesV4TraceActions}) so the modals
 * and their state live at the page level and every consumer shares one instance.
 */
export interface TracesV4TraceActions {
  /** Fetches a single trace's spans for the OSS dataset-export modal (via GenAITracesTableProvider). */
  getTrace: GetTraceFunction;
  /** OSS "Add to evaluation dataset" modal. */
  renderExportTracesToDatasetsModal?: (params: RenderExportTracesToDatasetsModalParams) => ReactNode;
  /**
   * OSS "Flag for review" flow: the controlled review-queue picker that wraps the Actions trigger.
   * Present in OSS exactly as v3 wires it (`TracesV3Logs`); the Actions menu drives its open state.
   */
  AddToReviewQueueDropdown: typeof AddToReviewQueueDropdown;
  /** Run-judges flow: opener, the modal element, the in-progress banner, and the in-explorer config. */
  runJudges?: {
    showRunJudgesModal: (traceIds: string[]) => void;
    RunJudgesModal: ReactNode;
    JudgesStatusBanner: ReactNode;
    runJudgeConfiguration: ReturnType<typeof useRunScorerInTracesViewConfiguration>;
  };
}

/**
 * Assembles the shared trace-action pieces (get-trace, run-judges, add-to-dataset) once for the v4 Traces tab.
 * Mirrors the `TraceActions` assembly in `TracesV3Logs`, re-expressed here so v4 stays isolated from v3.
 * Each block is feature-gated exactly as v3 gates it.
 */
export const useTracesV4TraceActions = (experimentId: string): TracesV4TraceActions => {
  const getTrace: GetTraceFunction = getTraceV3;

  const runJudgeConfiguration = useRunScorerInTracesViewConfiguration();
  const { showRunJudgesModal, RunJudgesModal, JudgesStatusBanner } = useRunJudgesOnTracesConfiguration(
    runJudgeConfiguration.evaluateTraces,
    runJudgeConfiguration.allEvaluations,
    runJudgeConfiguration.subscribeToScorerFinished,
  );
  const runJudges = isEvaluatingTracesInDetailsViewEnabled()
    ? { showRunJudgesModal, RunJudgesModal, JudgesStatusBanner, runJudgeConfiguration }
    : undefined;

  const actions: TracesV4TraceActions = {
    getTrace,
    renderExportTracesToDatasetsModal: ExportTracesToDatasetModal,
    AddToReviewQueueDropdown,
    runJudges,
  };
  return actions;
};
