import { useEffect, useMemo, useState } from 'react';

import { Button, Checkbox, ChevronRightIcon, ChevronDownIcon, GearIcon, useDesignSystemTheme, Typography } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { ModelTraceExplorerSummaryViewExceptionsSection } from './ModelTraceExplorerSummaryViewExceptionsSection';
import { type ModelTraceSpanNode } from '../ModelTrace.types';
import { createListFromObject, getSpanExceptionEvents } from '../ModelTraceExplorer.utils';
import { ModelTraceExplorerCollapsibleSection } from '../ModelTraceExplorerCollapsibleSection';
import { useModelTraceExplorerViewState } from '../ModelTraceExplorerViewStateContext';
import { SpanNameDetailViewLink } from '../assessments-pane/SpanNameDetailViewLink';
import { ModelTraceExplorerFieldRenderer } from '../field-renderers/ModelTraceExplorerFieldRenderer';
import { spanTimeFormatter } from '../timeline-tree/TimelineTree.utils';
import type { SpanRange, TraceView } from '../hooks/useTraceViews';
import { applyJsonPathToObject } from '../hooks/useTraceViewFiltering';
import type { SpanEditState, SpanRangeSelectionResult } from '../edit-mode/useSpanRangeSelection';
import { RangeBadge } from '../edit-mode/RangeBadge';
import { JsonFieldSelector } from '../edit-mode/JsonFieldSelector';

const CONNECTOR_WIDTH = 12;
const ROW_HEIGHT = 48;

export const ModelTraceExplorerSummaryIntermediateNode = ({
  node,
  renderMode,
  activeTraceView = null,
  isDimmedByView = false,
  isMatchedByView = false,
  editState,
  editSelection,
  editRanges,
  onRemoveRange,
  onUpdateRange,
}: {
  node: ModelTraceSpanNode;
  renderMode: 'default' | 'json';
  activeTraceView?: TraceView | null;
  isDimmedByView?: boolean;
  isMatchedByView?: boolean;
  editState?: SpanEditState | null;
  editSelection?: SpanRangeSelectionResult;
  editRanges?: SpanRange[];
  onRemoveRange?: (index: number) => void;
  onUpdateRange?: (index: number, updates: Partial<SpanRange>) => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const [expanded, setExpanded] = useState(isMatchedByView);

  // Auto-expand when a trace view matches this span, collapse when view is cleared
  useEffect(() => {
    setExpanded(isMatchedByView);
  }, [isMatchedByView]);
  const firstRange = activeTraceView?.ranges?.[0];
  const filteredInputs = useMemo(
    () => applyJsonPathToObject(node.inputs, firstRange?.input_path),
    [node.inputs, firstRange?.input_path],
  );
  const filteredOutputs = useMemo(
    () => applyJsonPathToObject(node.outputs, firstRange?.output_path),
    [node.outputs, firstRange?.output_path],
  );

  const inputList = useMemo(() => createListFromObject(filteredInputs as any), [filteredInputs]);
  const outputList = useMemo(() => createListFromObject(filteredOutputs as any), [filteredOutputs]);
  const exceptionEvents = getSpanExceptionEvents(node);
  const chatMessageFormat = node.chatMessageFormat;

  const hasException = exceptionEvents.length > 0;
  const containsInputs = inputList.length > 0;
  const containsOutputs = outputList.length > 0;

  const { setSelectedNode, setActiveView, setShowTimelineTreeGantt } = useModelTraceExplorerViewState();

  const editRangeColor = editState?.inRange ? editState.color : null;
  const editDragHighlight = editState?.inDrag ? 'rgba(59, 130, 246, 0.08)' : null;
  const flatIndex = editSelection?.spanKeyToFlatIndex.get(String(node.key));

  return (
    <div>
      {editState?.isFirstInRange && editState.rangeIdx !== undefined && editState.color && editRanges && (
        <div css={{ padding: `${theme.spacing.xs}px 0` }}>
          <RangeBadge
            label={editRanges[editState.rangeIdx].label}
            color={editState.color}
            onDelete={() => onRemoveRange?.(editState.rangeIdx as number)}
          />
        </div>
      )}
      <div
        css={{
          display: 'flex',
          flexDirection: 'row',
          minHeight: ROW_HEIGHT,
          flexShrink: 0,
          opacity: isDimmedByView ? 0.3 : 1,
          transition: 'opacity 150ms ease, background-color 150ms ease',
          backgroundColor: editDragHighlight ?? (editRangeColor ? editRangeColor.background : 'transparent'),
          borderLeft: editRangeColor ? `3px solid ${editRangeColor.primary}` : '3px solid transparent',
          borderRadius: theme.borders.borderRadiusSm,
        }}
        onPointerDown={editSelection && flatIndex !== undefined ? () => editSelection.handlePointerDown(flatIndex) : undefined}
        onPointerMove={editSelection && flatIndex !== undefined ? () => editSelection.handlePointerMove(flatIndex) : undefined}
      >
        {editSelection && flatIndex !== undefined && (
          <div css={{ height: ROW_HEIGHT, display: 'flex', alignItems: 'center', marginRight: theme.spacing.xs }}>
            <Checkbox
              componentId={`summary-intermediate-node.edit-checkbox-${node.key}`}
              isChecked={editState?.inRange ?? false}
              onChange={() => editSelection.handleCheckboxClick(flatIndex)}
              aria-label={`Select ${node.title}`}
            />
          </div>
        )}
        <div css={{ height: ROW_HEIGHT, display: 'flex', alignItems: 'center' }}>
          <Button
            size="small"
            data-testid={`toggle-span-expanded-${node.key}`}
            css={{ flexShrink: 0, marginRight: theme.spacing.xs }}
            icon={expanded ? <ChevronDownIcon /> : <ChevronRightIcon />}
            onClick={() => setExpanded(!expanded)}
            componentId="shared.model-trace-explorer.toggle-span"
          />
        </div>
      <div
        css={{
          position: 'relative',
          boxSizing: 'border-box',
          height: ROW_HEIGHT,
          borderLeft: `2px solid ${theme.colors.border}`,
          width: CONNECTOR_WIDTH,
        }}
      >
        <div
          css={{
            position: 'absolute',
            left: -2,
            top: 14,
            height: CONNECTOR_WIDTH,
            width: CONNECTOR_WIDTH,
            boxSizing: 'border-box',
            borderBottomLeftRadius: theme.borders.borderRadiusMd,
            borderBottom: `2px solid ${theme.colors.border}`,
            borderLeft: `2px solid ${theme.colors.border}`,
          }}
        />
      </div>
      <div css={{ display: 'flex', flexDirection: 'column', flex: 1, minWidth: 0 }}>
        <div css={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <Typography.Text color="secondary" css={{ display: 'inline-flex', alignItems: 'center', height: ROW_HEIGHT }}>
            <FormattedMessage
              defaultMessage="{spanName} was called"
              description="Label for an intermediate node in the trace explorer summary view, indicating that a span/function was called in the course of execution."
              values={{
                spanName: <SpanNameDetailViewLink node={node} />,
              }}
            />
          </Typography.Text>
          <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
            <span
              onClick={() => {
                setSelectedNode(node);
                setActiveView('detail');
                setShowTimelineTreeGantt(true);
              }}
            >
              <Typography.Text
                css={{
                  '&:hover': {
                    textDecoration: 'underline',
                    cursor: 'pointer',
                  },
                }}
                color="secondary"
              >
                {spanTimeFormatter(node.end - node.start)}
              </Typography.Text>
            </span>
            {editState?.inRange && editSelection && (
              <Button
                componentId={`summary-intermediate-node.gear-${node.key}`}
                type="tertiary"
                size="small"
                icon={<GearIcon />}
                onClick={(e: React.MouseEvent) => {
                  e.stopPropagation();
                  editSelection.toggleExpandedSpan(String(node.key));
                }}
                aria-label="Configure JSON fields"
              />
            )}
          </div>
        </div>
        {expanded && (
          <div>
            {hasException && <ModelTraceExplorerSummaryViewExceptionsSection node={node} />}
            {containsInputs && (
              <ModelTraceExplorerCollapsibleSection
                sectionKey="input"
                title={
                  <FormattedMessage
                    defaultMessage="Inputs"
                    description="Model trace explorer > selected span > inputs header"
                  />
                }
              >
                <div
                  css={{
                    display: 'flex',
                    flexDirection: 'column',
                    gap: theme.spacing.sm,
                    paddingLeft: theme.spacing.lg,
                    marginBottom: theme.spacing.sm,
                  }}
                >
                  {inputList.map(({ key, value }, index) => (
                    <ModelTraceExplorerFieldRenderer
                      key={key || index}
                      title={key}
                      data={value}
                      renderMode={renderMode}
                      chatMessageFormat={chatMessageFormat}
                    />
                  ))}
                </div>
              </ModelTraceExplorerCollapsibleSection>
            )}
            {containsOutputs && (
              <ModelTraceExplorerCollapsibleSection
                sectionKey="output"
                title={
                  <FormattedMessage
                    defaultMessage="Outputs"
                    description="Model trace explorer > selected span > outputs header"
                  />
                }
              >
                <div
                  css={{
                    display: 'flex',
                    flexDirection: 'column',
                    gap: theme.spacing.sm,
                    paddingLeft: theme.spacing.lg,
                    marginBottom: theme.spacing.sm,
                  }}
                >
                  {outputList.map(({ key, value }) => (
                    <ModelTraceExplorerFieldRenderer
                      key={key}
                      title={key}
                      data={value}
                      renderMode={renderMode}
                      chatMessageFormat={chatMessageFormat}
                      assessments={node.assessments}
                    />
                  ))}
                </div>
              </ModelTraceExplorerCollapsibleSection>
            )}
          </div>
        )}
      </div>
      </div>
      {editSelection && editSelection.expandedSpanId === String(node.key) && editState?.inRange && editState.rangeIdx !== undefined && editRanges && onUpdateRange && (
        <div
          css={{
            marginLeft: theme.spacing.lg + theme.spacing.md,
            marginRight: theme.spacing.sm,
            padding: theme.spacing.sm,
            backgroundColor: theme.colors.backgroundPrimary,
            border: `1px solid ${theme.colors.border}`,
            borderRadius: theme.borders.borderRadiusMd,
            marginBottom: theme.spacing.xs,
          }}
        >
          <div
            css={{
              color: theme.colors.textSecondary,
              fontSize: theme.typography.fontSizeSm,
              marginBottom: theme.spacing.sm,
            }}
          >
            These paths apply to all spans in this range
          </div>
          <div css={{ display: 'flex', gap: theme.spacing.md }}>
            <div css={{ flex: 1 }}>
              <JsonFieldSelector
                data={node.inputs}
                selectedPath={editRanges[editState.rangeIdx].input_path ?? null}
                onPathChange={(path) => onUpdateRange(editState.rangeIdx as number, { input_path: path })}
                label="Input Fields"
              />
            </div>
            <div css={{ flex: 1 }}>
              <JsonFieldSelector
                data={node.outputs}
                selectedPath={editRanges[editState.rangeIdx].output_path ?? null}
                onPathChange={(path) => onUpdateRange(editState.rangeIdx as number, { output_path: path })}
                label="Output Fields"
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
