import type { ModelTrace } from '../ModelTrace.types';
import { ModelTraceExplorerPreferencesProvider } from '../ModelTraceExplorerPreferencesContext';
import { ModelTraceExplorerViewStateProvider } from '../ModelTraceExplorerViewStateContext';
import { ModelTraceExplorerCustomView } from './ModelTraceExplorerCustomView';

export interface ModelTraceExplorerCustomViewRendererProps {
  modelTrace: ModelTrace;
  enableFeedback: boolean;
}

/**
 * Supplies the trace-derived state needed to bind a saved custom-view template
 * when rendering outside the full trace explorer.
 */
export const ModelTraceExplorerCustomViewRenderer = ({
  modelTrace,
  enableFeedback,
}: ModelTraceExplorerCustomViewRendererProps): JSX.Element => (
  <ModelTraceExplorerPreferencesProvider>
    <ModelTraceExplorerViewStateProvider modelTrace={modelTrace} assessmentsPaneEnabled={false}>
      <ModelTraceExplorerCustomView modelTraceInfo={modelTrace.info} enableFeedback={enableFeedback} />
    </ModelTraceExplorerViewStateProvider>
  </ModelTraceExplorerPreferencesProvider>
);
