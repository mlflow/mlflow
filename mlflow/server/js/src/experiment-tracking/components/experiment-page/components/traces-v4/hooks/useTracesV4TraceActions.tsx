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
  /** Run-judges flow: opener, the modal element, the in-progress banner, and the in-explorer config. */
  runJudges?: {
    showRunJudgesModal: (traceIds: string[]) => void;
    RunJudgesModal: ReactNode;
    JudgesStatusBanner: ReactNode;
    runJudgeConfiguration: ReturnType<typeof useRunScorerInTracesViewConfiguration>;
  };
  flags: {
    /** Show "Add to labeling session" (always true for OSS; in Databricks gated on review queues off). */
    addToLabelingSession: boolean;
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
    runJudges,
    flags: { addToLabelingSession: true },
  };
  return actions;
};
