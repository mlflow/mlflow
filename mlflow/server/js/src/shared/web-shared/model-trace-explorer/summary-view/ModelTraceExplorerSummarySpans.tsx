import { useState } from 'react';

import { SegmentedControlButton, SegmentedControlGroup, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { ModelTraceExplorerSummaryIntermediateNode } from './ModelTraceExplorerSummaryIntermediateNode';
import { ModelTraceExplorerSummarySection } from './ModelTraceExplorerSummarySection';
import { ModelTraceExplorerSummaryViewExceptionsSection } from './ModelTraceExplorerSummaryViewExceptionsSection';
import type { ModelTraceExplorerRenderMode, ModelTraceSpanNode } from '../ModelTrace.types';
import { createListFromObject, getSpanExceptionEvents } from '../ModelTraceExplorer.utils';
import { AssessmentPaneToggle } from '../assessments-pane/AssessmentPaneToggle';

export const SUMMARY_SPANS_MIN_WIDTH = 400;

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
  const [renderMode, setRenderMode] = useState<ModelTraceExplorerRenderMode>('default');

  const rootInputs = rootNode.inputs;
  const rootOutputs = rootNode.outputs;
  const chatMessageFormat = rootNode.chatMessageFormat;
  const exceptions = getSpanExceptionEvents(rootNode);
  const hasIntermediateNodes = intermediateNodes.length > 0;
  const hasExceptions = exceptions.length > 0;

  const inputList = createListFromObject(rootInputs).filter(({ value }) => value !== 'null');
  const outputList = createListFromObject(rootOutputs).filter(({ value }) => value !== 'null');

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        flex: 1,
        minHeight: 0,
        padding: theme.spacing.md,
        paddingTop: theme.spacing.sm,
        overflow: 'auto',
        minWidth: SUMMARY_SPANS_MIN_WIDTH,
      }}
    >
      {!hideRenderModeSelector && (
        <div
          css={{ display: 'flex', flexDirection: 'row', justifyContent: 'flex-end', marginBottom: theme.spacing.sm }}
        >
          <div css={{ display: 'flex', gap: theme.spacing.sm }}>
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
            </SegmentedControlGroup>
            <AssessmentPaneToggle />
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
      {hasIntermediateNodes &&
        intermediateNodes.map((node) => (
          <ModelTraceExplorerSummaryIntermediateNode key={node.key} node={node} renderMode={renderMode} />
        ))}
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
        chatMessageFormat={chatMessageFormat ?? 'openai'}
      />
    </div>
  );
};
