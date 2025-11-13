import { useMemo, useState } from 'react';

import { Empty, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { ModelTraceExplorerSummarySpans, SUMMARY_SPANS_MIN_WIDTH } from './ModelTraceExplorerSummarySpans';
import { useIntermediateNodes } from '../ModelTraceExplorer.utils';
import ModelTraceExplorerResizablePane from '../ModelTraceExplorerResizablePane';
import { useModelTraceExplorerViewState } from '../ModelTraceExplorerViewStateContext';
import { AssessmentsPane } from '../assessments-pane/AssessmentsPane';
import { ASSESSMENT_PANE_MIN_WIDTH } from '../assessments-pane/AssessmentsPane.utils';

export const ModelTraceExplorerSummaryView = () => {
  const { theme } = useDesignSystemTheme();
  const [paneWidth, setPaneWidth] = useState(500);
  const { rootNode, nodeMap, assessmentsPaneEnabled, assessmentsPaneExpanded } = useModelTraceExplorerViewState();

  const allAssessments = useMemo(() => Object.values(nodeMap).flatMap((node) => node.assessments), [nodeMap]);

  const intermediateNodes = useIntermediateNodes(rootNode);

  if (!rootNode) {
    return (
      <div css={{ marginTop: theme.spacing.lg }}>
        <Empty
          description={
            <FormattedMessage
              defaultMessage="No span data to display"
              description="Title for the empty state in the model trace explorer summary view"
            />
          }
        />
      </div>
    );
  }
  const AssessmentsPaneComponent = (
    <AssessmentsPane assessments={allAssessments} traceId={rootNode.traceId} activeSpanId={undefined} />
  );

  return assessmentsPaneEnabled && assessmentsPaneExpanded ? (
    <ModelTraceExplorerResizablePane
      initialRatio={0.75}
      paneWidth={paneWidth}
      setPaneWidth={setPaneWidth}
      leftChild={<ModelTraceExplorerSummarySpans rootNode={rootNode} intermediateNodes={intermediateNodes} />}
      rightChild={AssessmentsPaneComponent}
      leftMinWidth={SUMMARY_SPANS_MIN_WIDTH}
      rightMinWidth={ASSESSMENT_PANE_MIN_WIDTH}
    />
  ) : (
    <ModelTraceExplorerSummarySpans rootNode={rootNode} intermediateNodes={intermediateNodes} />
  );
};
