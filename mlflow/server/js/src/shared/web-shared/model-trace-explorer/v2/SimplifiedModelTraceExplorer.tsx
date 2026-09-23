import { useState } from 'react';
import { ErrorBoundary } from 'react-error-boundary';

import type { Assessment, ModelTrace } from './ModelTrace.types';
import { getModelTraceId } from './ModelTraceExplorer.utils';
import { ModelTraceExplorerDetailView } from './ModelTraceExplorerDetailView';
import { ModelTraceExplorerErrorState } from '../ModelTraceExplorerErrorState';
import { ModelTraceExplorerGenericErrorState } from '../ModelTraceExplorerGenericErrorState';
import ModelTraceExplorerResizablePane from './ModelTraceExplorerResizablePane';
import {
  ModelTraceExplorerViewStateProvider,
  useModelTraceExplorerViewState,
} from './ModelTraceExplorerViewStateContext';
import { ModelTraceExplorerPreferencesProvider } from './ModelTraceExplorerPreferencesContext';
import { SimplifiedAssessmentView, SIMPLIFIED_ASSESSMENT_VIEW_MIN_WIDTH } from './right-pane/SimplifiedAssessmentView';

const TRACE_DETAILS_MIN_WIDTH = 400;

const SimplifiedModelTraceExplorerContent = ({
  assessments,
  modelTraceInfo,
}: {
  assessments: Assessment[];
  modelTraceInfo: ModelTrace['info'];
}) => {
  const [paneWidth, setPaneWidth] = useState(500);
  const { rootNode } = useModelTraceExplorerViewState();

  if (!rootNode) {
    return null;
  }

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        overflow: 'hidden',
      }}
    >
      <ModelTraceExplorerResizablePane
        initialRatio={0.5}
        paneWidth={paneWidth}
        setPaneWidth={setPaneWidth}
        leftChild={<ModelTraceExplorerDetailView modelTraceInfo={modelTraceInfo} />}
        rightChild={<SimplifiedAssessmentView assessments={assessments} />}
        leftMinWidth={TRACE_DETAILS_MIN_WIDTH}
        rightMinWidth={SIMPLIFIED_ASSESSMENT_VIEW_MIN_WIDTH}
      />
    </div>
  );
};

export const ContextProviders = ({ children }: { traceId: string; children: React.ReactNode }): JSX.Element => {
  return <ErrorBoundary fallbackRender={ModelTraceExplorerErrorState}>{children}</ErrorBoundary>;
};

export const SimplifiedModelTraceExplorerImpl = ({
  modelTrace: initialModelTrace,
  assessments,
}: {
  modelTrace: ModelTrace;
  assessments: Assessment[];
}): JSX.Element => {
  const traceId = getModelTraceId(initialModelTrace);

  return (
    <ContextProviders traceId={traceId}>
      <ModelTraceExplorerPreferencesProvider>
        <ModelTraceExplorerViewStateProvider modelTrace={initialModelTrace} assessmentsPaneEnabled={false}>
          <SimplifiedModelTraceExplorerContent assessments={assessments} modelTraceInfo={initialModelTrace.info} />
        </ModelTraceExplorerViewStateProvider>
      </ModelTraceExplorerPreferencesProvider>
    </ContextProviders>
  );
};

export const SimplifiedModelTraceExplorer: typeof SimplifiedModelTraceExplorerImpl = SimplifiedModelTraceExplorerImpl;
