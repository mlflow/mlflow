import { useState } from 'react';
import { ErrorBoundary } from 'react-error-boundary';

import type { Assessment, ModelTrace } from './ModelTrace.types';
import { getModelTraceId, useIntermediateNodes } from './ModelTraceExplorer.utils';
import { ModelTraceExplorerErrorState } from './ModelTraceExplorerErrorState';
import { ModelTraceExplorerGenericErrorState } from './ModelTraceExplorerGenericErrorState';
import ModelTraceExplorerResizablePane from './ModelTraceExplorerResizablePane';
import {
  ModelTraceExplorerViewStateProvider,
  useModelTraceExplorerViewState,
} from './ModelTraceExplorerViewStateContext';
import { SimplifiedAssessmentView, SIMPLIFIED_ASSESSMENT_VIEW_MIN_WIDTH } from './right-pane/SimplifiedAssessmentView';
import { ModelTraceExplorerSummarySpans, SUMMARY_SPANS_MIN_WIDTH } from './summary-view/ModelTraceExplorerSummarySpans';

const SimplifiedModelTraceExplorerContent = ({ assessments }: { assessments: Assessment[] }) => {
  const [paneWidth, setPaneWidth] = useState(500);
  const { rootNode } = useModelTraceExplorerViewState();
  const intermediateNodes = useIntermediateNodes(rootNode);

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
        leftChild={
          <ModelTraceExplorerSummarySpans
            rootNode={rootNode}
            intermediateNodes={intermediateNodes}
            hideRenderModeSelector
          />
        }
        rightChild={<SimplifiedAssessmentView assessments={assessments} />}
        leftMinWidth={SUMMARY_SPANS_MIN_WIDTH}
        rightMinWidth={SIMPLIFIED_ASSESSMENT_VIEW_MIN_WIDTH}
      />
    </div>
  );
};

const ContextProviders = ({ children }: { traceId: string; children: React.ReactNode }) => {
  return <ErrorBoundary fallbackRender={ModelTraceExplorerErrorState}>{children}</ErrorBoundary>;
};

export const SimplifiedModelTraceExplorerImpl = ({
  modelTrace: initialModelTrace,
  assessments,
}: {
  modelTrace: ModelTrace;
  assessments: Assessment[];
}) => {
  const traceId = getModelTraceId(initialModelTrace);

  return (
    <ContextProviders traceId={traceId}>
      <ModelTraceExplorerViewStateProvider modelTrace={initialModelTrace} assessmentsPaneEnabled>
        <SimplifiedModelTraceExplorerContent assessments={assessments} />
      </ModelTraceExplorerViewStateProvider>
    </ContextProviders>
  );
};

export const SimplifiedModelTraceExplorer = SimplifiedModelTraceExplorerImpl;
