import { useCallback, useEffect, useMemo, useState } from 'react';

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
import type { TraceView } from '../hooks/useTraceViews';
import { applyJsonPathToObject } from '../hooks/useTraceViewFiltering';
import { SpanRangeOverlay } from '../edit-mode/SpanRangeOverlay';
import { TraceViewEditToolbar } from '../edit-mode/TraceViewEditToolbar';
import { useCreateTraceView, useUpdateTraceView } from '../hooks/useTraceViewMutations';

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
  const { readOnly, editMode } = useModelTraceExplorerViewState();

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

  const traceId = rootNode.traceId;
  const createMutation = useCreateTraceView(traceId);
  const updateMutation = useUpdateTraceView(traceId);

  const flattenedNodes = useMemo(() => {
    const flatten = (nodes: ModelTraceSpanNode[]): ModelTraceSpanNode[] =>
      nodes.flatMap((node) => [node, ...(node.children ? flatten(node.children) : [])]);
    return flatten(intermediateNodes);
  }, [intermediateNodes]);

  const handleSave = useCallback(async () => {
    if (!editMode.draftView) return;
    const { view_id, name, ranges } = editMode.draftView;
    if (view_id) {
      await updateMutation.mutateAsync({ viewId: view_id, name, ranges });
    } else {
      await createMutation.mutateAsync({ name, ranges });
    }
    editMode.exitEditMode();
  }, [editMode, createMutation, updateMutation]);

  const rootInputs = rootNode.inputs;
  const rootOutputs = rootNode.outputs;
  const chatMessageFormat = rootNode.chatMessageFormat;
  const exceptions = getSpanExceptionEvents(rootNode);
  const hasIntermediateNodes = intermediateNodes.length > 0;
  const hasExceptions = exceptions.length > 0;

  const firstRange = activeTraceView?.ranges?.[0];
  const filteredInputs = useMemo(
    () => applyJsonPathToObject(rootInputs, firstRange?.input_path),
    [rootInputs, firstRange?.input_path],
  );
  const filteredOutputs = useMemo(
    () => applyJsonPathToObject(rootOutputs, firstRange?.output_path),
    [rootOutputs, firstRange?.output_path],
  );

  const inputList = useMemo(
    () => createListFromObject(filteredInputs as any).filter(({ value }) => value !== 'null'),
    [filteredInputs],
  );
  const outputList = useMemo(
    () => createListFromObject(filteredOutputs as any).filter(({ value }) => value !== 'null'),
    [filteredOutputs],
  );

  const hasJsonPathFilter = !!(firstRange?.input_path || firstRange?.output_path);

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
      {editMode.isEditMode && editMode.draftView && (
        <>
          <TraceViewEditToolbar
            name={editMode.draftView.name}
            onNameChange={editMode.setName}
            ranges={editMode.draftView.ranges}
            onCancel={editMode.exitEditMode}
            onSave={handleSave}
            isSaving={createMutation.isLoading || updateMutation.isLoading}
          />
          <div css={{ marginTop: theme.spacing.sm }}>
            <SpanRangeOverlay
              nodes={flattenedNodes}
              ranges={editMode.draftView.ranges}
              onAddRange={editMode.addRange}
              onRemoveRange={editMode.removeRange}
              onUpdateRange={editMode.updateRange}
            />
          </div>
        </>
      )}
      {!editMode.isEditMode && hasJsonPathFilter && activeTraceView && (
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
      {!editMode.isEditMode && hasExceptions && <ModelTraceExplorerSummaryViewExceptionsSection node={rootNode} />}
      {!editMode.isEditMode && <ModelTraceExplorerSummarySection
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
      />}
      {!editMode.isEditMode && hasIntermediateNodes &&
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
      {!editMode.isEditMode && <ModelTraceExplorerSummarySection
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
      />}
    </div>
  );
};
