import { useCallback, useEffect, useMemo, useState } from 'react';

import { SegmentedControlButton, SegmentedControlGroup, Typography, useDesignSystemTheme } from '@databricks/design-system';
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
import type { TraceView } from '../hooks/useTraceViews';
import { applyJsonPathToObject } from '../hooks/useTraceViewFiltering';

export const SUMMARY_SPANS_MIN_WIDTH = 400;

export const ModelTraceExplorerSummarySpans = ({
  rootNode,
  intermediateNodes,
  hideRenderModeSelector = false,
  activeTraceView = null,
  viewMatchedSpanKeys = null,
}: {
  rootNode: ModelTraceSpanNode;
  intermediateNodes: ModelTraceSpanNode[];
  hideRenderModeSelector?: boolean;
  activeTraceView?: TraceView | null;
  viewMatchedSpanKeys?: Set<string | number> | null;
}) => {
  const { theme } = useDesignSystemTheme();
  const preferences = useModelTraceExplorerPreferences();
  const [renderMode, setRenderModeInternal] = useState<ModelTraceExplorerRenderMode>(preferences.renderMode);
  const { readOnly } = useModelTraceExplorerViewState();

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

  const filteredInputs = useMemo(
    () => applyJsonPathToObject(rootInputs, activeTraceView?.input_path),
    [rootInputs, activeTraceView?.input_path],
  );
  const filteredOutputs = useMemo(
    () => applyJsonPathToObject(rootOutputs, activeTraceView?.output_path),
    [rootOutputs, activeTraceView?.output_path],
  );

  const inputList = useMemo(
    () => createListFromObject(filteredInputs as any).filter(({ value }) => value !== 'null'),
    [filteredInputs],
  );
  const outputList = useMemo(
    () => createListFromObject(filteredOutputs as any).filter(({ value }) => value !== 'null'),
    [filteredOutputs],
  );

  const hasJsonPathFilter = !!(activeTraceView?.input_path || activeTraceView?.output_path);

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
            {!readOnly && (
              <>
                <AddToDatasetButton />
                <AssessmentPaneToggle />
              </>
            )}
          </div>
        </div>
      )}
      {hasJsonPathFilter && activeTraceView && (
        <div
          css={{
            display: 'flex',
            alignItems: 'center',
            gap: theme.spacing.xs,
            marginBottom: theme.spacing.sm,
            padding: `${theme.spacing.xs}px ${theme.spacing.sm}px`,
            backgroundColor: theme.colors.backgroundSecondary,
            borderRadius: theme.borders.borderRadiusMd,
          }}
        >
          <Typography.Text size="sm" color="secondary">
            Filtered by: {activeTraceView.name}
          </Typography.Text>
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
          <ModelTraceExplorerSummaryIntermediateNode
            key={node.key}
            node={node}
            renderMode={renderMode}
            activeTraceView={activeTraceView}
            isDimmedByView={viewMatchedSpanKeys != null && !viewMatchedSpanKeys.has(node.key)}
            isMatchedByView={viewMatchedSpanKeys != null && viewMatchedSpanKeys.has(node.key)}
          />
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
        chatMessageFormat={chatMessageFormat}
        assessments={rootNode.assessments}
      />
    </div>
  );
};
