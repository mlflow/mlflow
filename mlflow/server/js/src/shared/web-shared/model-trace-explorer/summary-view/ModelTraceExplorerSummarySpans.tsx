import { useCallback, useEffect, useState } from 'react';

import {
  SegmentedControlButton,
  SegmentedControlGroup,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { ModelTraceExplorerSummaryIntermediateNode } from './ModelTraceExplorerSummaryIntermediateNode';
import { ModelTraceExplorerSummarySection } from './ModelTraceExplorerSummarySection';
import { ModelTraceExplorerSummaryViewExceptionsSection } from './ModelTraceExplorerSummaryViewExceptionsSection';
import type { ModelTraceExplorerRenderMode, ModelTraceSpanNode } from '../ModelTrace.types';
import { createListFromObject, getSpanExceptionEvents } from '../ModelTraceExplorer.utils';
import { useModelTraceExplorerViewState } from '../ModelTraceExplorerViewStateContext';
import { AddToDatasetButton } from '../assessments-pane/AddToDatasetButton';
import { AssessmentPaneToggle } from '../assessments-pane/AssessmentPaneToggle';
import { useModelTraceExplorerPreferences } from '../ModelTraceExplorerPreferencesContext';

export const SUMMARY_SPANS_MIN_WIDTH = 400;
const INTERMEDIATE_NODES_TRUNCATION_LIMIT = 3;

export const ModelTraceExplorerSummarySpans = ({
  rootNode,
  intermediateNodes,
  hideRenderModeSelector = false,
}: {
  rootNode: ModelTraceSpanNode;
  intermediateNodes: ModelTraceSpanNode[];
  hideRenderModeSelector?: boolean;
}) => {
  const { theme } = useDesignSystemTheme();
  const preferences = useModelTraceExplorerPreferences();
  const [renderMode, setRenderModeInternal] = useState<ModelTraceExplorerRenderMode>(preferences.renderMode);
  const [intermediateNodesExpanded, setIntermediateNodesExpanded] = useState(false);
  const { readOnly, assessmentsPaneExpanded } = useModelTraceExplorerViewState();

  useEffect(() => {
    setRenderModeInternal(preferences.renderMode);
  }, [preferences.renderMode]);

  const setRenderMode = useCallback(
    (mode: ModelTraceExplorerRenderMode) => {
      setRenderModeInternal(mode);
      preferences.setRenderMode(mode);
    },
    [preferences],
  );

  const rootInputs = rootNode.inputs;
  const rootOutputs = rootNode.outputs;
  const chatMessageFormat = rootNode.chatMessageFormat;
  const exceptions = getSpanExceptionEvents(rootNode);
  const hasIntermediateNodes = intermediateNodes.length > 0;
  const hasExceptions = exceptions.length > 0;

  const inputList = createListFromObject(rootInputs).filter(({ value }) => value !== 'null');
  const outputList = createListFromObject(rootOutputs).filter(({ value }) => value !== 'null');
  const shouldTruncateNodes = intermediateNodes.length > INTERMEDIATE_NODES_TRUNCATION_LIMIT;
  const displayedIntermediateNodes = intermediateNodesExpanded
    ? intermediateNodes
    : intermediateNodes.slice(0, INTERMEDIATE_NODES_TRUNCATION_LIMIT);

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        flex: 1,
        minHeight: 0,
        overflow: 'auto',
        minWidth: SUMMARY_SPANS_MIN_WIDTH,
      }}
    >
      {!hideRenderModeSelector && (
        <div css={{ display: 'flex', flexDirection: 'row', justifyContent: 'flex-end', marginBlock: theme.spacing.sm }}>
          <div css={{ display: 'flex', gap: theme.spacing.sm, paddingInline: theme.spacing.sm }}>
            <SegmentedControlGroup
              name="render-mode"
              componentId="shared.model-trace-explorer.summary-view.render-mode"
              value={renderMode}
              size="small"
              onChange={(event) => setRenderMode(event.target.value)}
            >
              <SegmentedControlButton value="default">
                <FormattedMessage
                  defaultMessage="Default"
                  description="Label for the default render mode selector in the model trace explorer summary view"
                />
              </SegmentedControlButton>
              <SegmentedControlButton value="json">
                <FormattedMessage
                  defaultMessage="JSON"
                  description="Label for the JSON render mode selector in the model trace explorer summary view"
                />
              </SegmentedControlButton>
              <SegmentedControlButton value="table">
                <FormattedMessage
                  defaultMessage="Table"
                  description="Label for the Table render mode selector in the model trace explorer summary view"
                />
              </SegmentedControlButton>
            </SegmentedControlGroup>
            {!readOnly && (
              <>
                <AddToDatasetButton />
                {!assessmentsPaneExpanded && <AssessmentPaneToggle />}
              </>
            )}
          </div>
        </div>
      )}
      {hasExceptions && <ModelTraceExplorerSummaryViewExceptionsSection node={rootNode} />}
      <ModelTraceExplorerSummarySection
        title={
          <FormattedMessage
            defaultMessage="Inputs"
            description="Model trace explorer > selected span > inputs header"
          />
        }
        css={{ marginBottom: hasIntermediateNodes ? 0 : theme.spacing.md }}
        sectionKey="summary-inputs"
        data={inputList}
        renderMode={renderMode}
        chatMessageFormat={chatMessageFormat}
      />
      {displayedIntermediateNodes.map((node, index) => (
        <ModelTraceExplorerSummaryIntermediateNode
          key={node.key}
          node={node}
          renderMode={renderMode}
          css={{
            borderTop: index === 0 ? `1px solid ${theme.colors.border}` : undefined,
            borderBottom:
              !shouldTruncateNodes && index === displayedIntermediateNodes.length - 1
                ? undefined
                : `1px solid ${theme.colors.border}`,
          }}
        />
      ))}
      {shouldTruncateNodes && (
        <div css={{ paddingBlock: theme.spacing.sm, display: 'flex', justifyContent: 'center' }}>
          {intermediateNodesExpanded ? (
            <Typography.Link
              componentId="shared.model-trace-explorer.summary-view.collapse-intermediate-nodes"
              onClick={() => setIntermediateNodesExpanded(false)}
            >
              <FormattedMessage
                defaultMessage="Show less"
                description="Link that collapses an expanded list when clicked"
              />
            </Typography.Link>
          ) : (
            <Typography.Link
              componentId="shared.model-trace-explorer.summary-view.expand-intermediate-nodes"
              onClick={() => setIntermediateNodesExpanded(true)}
            >
              <FormattedMessage
                defaultMessage="Show {count} more intermediate {count, plural, =1 {step} other {steps}}"
                description="Link that expands a collapsed list of intermediate function execution steps when clicked"
                values={{
                  count: intermediateNodes.length - INTERMEDIATE_NODES_TRUNCATION_LIMIT,
                }}
              />
            </Typography.Link>
          )}
        </div>
      )}
      <ModelTraceExplorerSummarySection
        title={
          <FormattedMessage
            defaultMessage="Outputs"
            description="Model trace explorer > selected span > outputs header"
          />
        }
        sectionKey="summary-outputs"
        data={outputList}
        renderMode={renderMode}
        chatMessageFormat={chatMessageFormat}
        assessments={rootNode.assessments}
      />
    </div>
  );
};
