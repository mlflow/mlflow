import { useMemo, useState } from 'react';

import { Empty, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { ModelTraceExplorerSummarySpans, SUMMARY_SPANS_MIN_WIDTH } from './ModelTraceExplorerSummarySpans';
import { ModelSpanType } from '../ModelTrace.types';
import type { ModelTraceSpanNode, ModelTrace } from '../ModelTrace.types';
import { getSpanExceptionCount } from '../ModelTraceExplorer.utils';
import ModelTraceExplorerResizablePane from '../ModelTraceExplorerResizablePane';
import { useModelTraceExplorerViewState } from '../ModelTraceExplorerViewStateContext';
import { AssessmentsPane } from '../assessments-pane/AssessmentsPane';
import { ASSESSMENT_PANE_MIN_WIDTH } from '../assessments-pane/AssessmentsPane.utils';
import { getTimelineTreeNodesList } from '../timeline-tree/TimelineTree.utils';

const isNodeImportant = (node: ModelTraceSpanNode) => {
  // root node is shown at top level, so we don't need to
  // show it in the intermediate nodes list
  if (!node.parentId) {
    return false;
  }

  return (
    [
      ModelSpanType.AGENT,
      ModelSpanType.RETRIEVER,
      ModelSpanType.CHAT_MODEL,
      ModelSpanType.TOOL,
      ModelSpanType.LLM,
    ].includes(node.type ?? ModelSpanType.UNKNOWN) || getSpanExceptionCount(node) > 0
  );
};

export const ModelTraceExplorerSummaryView = ({ modelTrace }: { modelTrace: ModelTrace }) => {
  const { theme } = useDesignSystemTheme();
  const [paneWidth, setPaneWidth] = useState(500);
  const { rootNode, nodeMap, assessmentsPaneEnabled, assessmentsPaneExpanded } = useModelTraceExplorerViewState();

  const allAssessments = useMemo(() => Object.values(nodeMap).flatMap((node) => node.assessments), [nodeMap]);

  const intermediateNodes = useMemo(() => {
    if (!rootNode) {
      return [];
    }

    // the summary view is meant to be a high-level view of the trace,
    // so we show "important" nodes as a flat list between the inputs
    // and outputs of the root node.
    const nodes = getTimelineTreeNodesList([rootNode]);
    const intermediateNodes = nodes.filter(isNodeImportant);

    return intermediateNodes;
  }, [rootNode]);

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

  return assessmentsPaneEnabled && assessmentsPaneExpanded ? (
    <ModelTraceExplorerResizablePane
      initialRatio={0.75}
      paneWidth={paneWidth}
      setPaneWidth={setPaneWidth}
      leftChild={<ModelTraceExplorerSummarySpans rootNode={rootNode} intermediateNodes={intermediateNodes} />}
      rightChild={<AssessmentsPane assessments={allAssessments} traceId={rootNode.traceId} activeSpanId={undefined} />}
      leftMinWidth={SUMMARY_SPANS_MIN_WIDTH}
      rightMinWidth={ASSESSMENT_PANE_MIN_WIDTH}
    />
  ) : (
    <ModelTraceExplorerSummarySpans rootNode={rootNode} intermediateNodes={intermediateNodes} />
  );
};
