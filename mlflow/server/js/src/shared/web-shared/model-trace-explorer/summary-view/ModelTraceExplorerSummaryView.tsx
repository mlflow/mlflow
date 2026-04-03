import { useCallback, useMemo, useState } from 'react';

import { Empty, SegmentedControlButton, SegmentedControlGroup, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { ModelTraceExplorerCompactSummaryView } from './ModelTraceExplorerCompactSummaryView';
import { ModelTraceExplorerRangesView } from './ModelTraceExplorerRangesView';
import { ModelTraceExplorerSummarySpans, SUMMARY_SPANS_MIN_WIDTH } from './ModelTraceExplorerSummarySpans';
import { getTraceLevelAssessments, useIntermediateNodes } from '../ModelTraceExplorer.utils';
import ModelTraceExplorerResizablePane from '../ModelTraceExplorerResizablePane';
import { useModelTraceExplorerViewState } from '../ModelTraceExplorerViewStateContext';
import { AssessmentsPane } from '../assessments-pane/AssessmentsPane';
import { ASSESSMENT_PANE_MIN_WIDTH } from '../assessments-pane/AssessmentsPane.utils';
import { useTraceViewSpanMatches } from '../hooks/useTraceViewFiltering';

type SummaryLayout = 'classic' | 'compact';

export const ModelTraceExplorerSummaryView = () => {
  const { theme } = useDesignSystemTheme();
  const [paneWidth, setPaneWidth] = useState(500);
  const [layout, setLayout] = useState<SummaryLayout>('classic');

  const {
    rootNode,
    nodeMap,
    assessmentsPaneEnabled,
    assessmentsPaneExpanded,
    updatePaneSizeRatios,
    getPaneSizeRatios,
    activeTraceView,
    topLevelNodes,
  } = useModelTraceExplorerViewState();

  const allAssessments = useMemo(() => Object.values(nodeMap).flatMap((node) => node.assessments), [nodeMap]);

  const onSizeRatioChange = useCallback(
    (ratio: number) => {
      updatePaneSizeRatios({ summarySidebar: ratio });
    },
    [updatePaneSizeRatios],
  );

  const intermediateNodes = useIntermediateNodes(rootNode);
  const viewMatchedSpanKeys = useTraceViewSpanMatches(topLevelNodes, activeTraceView);

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

  const LayoutToggle = (
    <div
      css={{ display: 'flex', justifyContent: 'flex-end', padding: `${theme.spacing.xs}px ${theme.spacing.md}px 0` }}
    >
      <SegmentedControlGroup
        name="summary-layout"
        componentId="shared.model-trace-explorer.summary-view.layout-toggle"
        value={layout}
        size="small"
        onChange={(event) => setLayout(event.target.value as SummaryLayout)}
      >
        <SegmentedControlButton value="classic">
          <FormattedMessage defaultMessage="Classic" description="Label for classic summary layout in trace explorer" />
        </SegmentedControlButton>
        <SegmentedControlButton value="compact">
          <FormattedMessage defaultMessage="Compact" description="Label for compact summary layout in trace explorer" />
        </SegmentedControlButton>
      </SegmentedControlGroup>
    </div>
  );

  // When a multi-range view is active, render the ranges view directly
  const isMultiRangeView = activeTraceView && activeTraceView.ranges.length > 1;

  if (isMultiRangeView) {
    return (
      <div css={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0 }}>
        <ModelTraceExplorerRangesView activeTraceView={activeTraceView} />
      </div>
    );
  }

  if (layout === 'compact') {
    return (
      <div css={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0 }}>
        {LayoutToggle}
        <ModelTraceExplorerCompactSummaryView />
      </div>
    );
  }

  const AssessmentsPaneComponent = (
    <AssessmentsPane assessments={allAssessments} traceId={rootNode.traceId} activeSpanId={undefined} />
  );

  return (
    <div css={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0 }}>
      {LayoutToggle}
      {assessmentsPaneEnabled && assessmentsPaneExpanded ? (
        <ModelTraceExplorerResizablePane
          initialRatio={getPaneSizeRatios().summarySidebar}
          paneWidth={paneWidth}
          setPaneWidth={setPaneWidth}
          leftChild={
            <ModelTraceExplorerSummarySpans
              rootNode={rootNode}
              intermediateNodes={intermediateNodes}
              activeTraceView={activeTraceView}
              viewMatchedSpanKeys={viewMatchedSpanKeys}
            />
          }
          rightChild={AssessmentsPaneComponent}
          leftMinWidth={SUMMARY_SPANS_MIN_WIDTH + 2 * theme.spacing.md}
          rightMinWidth={ASSESSMENT_PANE_MIN_WIDTH + 2 * theme.spacing.sm}
          onRatioChange={onSizeRatioChange}
        />
      ) : (
        <ModelTraceExplorerSummarySpans
          rootNode={rootNode}
          intermediateNodes={intermediateNodes}
          activeTraceView={activeTraceView}
          viewMatchedSpanKeys={viewMatchedSpanKeys}
        />
      )}
    </div>
  );
};
