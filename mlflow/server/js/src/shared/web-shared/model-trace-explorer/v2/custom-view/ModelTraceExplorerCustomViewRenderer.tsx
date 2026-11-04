import type { ModelTrace } from '../ModelTrace.types';
import { ModelTraceExplorerPreferencesProvider } from '../ModelTraceExplorerPreferencesContext';
import { ModelTraceExplorerViewStateProvider } from '../ModelTraceExplorerViewStateContext';
import { ModelTraceExplorerCustomView } from './ModelTraceExplorerCustomView';

export interface ModelTraceExplorerCustomViewRendererProps {
  modelTrace: ModelTrace;
  enableFeedback: boolean;
  feedbackSourceId?: string;
}

/**
 * Supplies the trace-derived state needed to bind a saved custom-view template
 * when rendering outside the full trace explorer.
 */
export const ModelTraceExplorerCustomViewRenderer = ({
  modelTrace,
  enableFeedback,
  feedbackSourceId,
}: ModelTraceExplorerCustomViewRendererProps): JSX.Element => (
  <ModelTraceExplorerPreferencesProvider>
    <ModelTraceExplorerViewStateProvider modelTrace={modelTrace} assessmentsPaneEnabled={false}>
      <ModelTraceExplorerCustomView
        modelTraceInfo={modelTrace.info}
        enableFeedback={enableFeedback}
        feedbackSourceId={feedbackSourceId}
      />
    </ModelTraceExplorerViewStateProvider>
  </ModelTraceExplorerPreferencesProvider>
);
