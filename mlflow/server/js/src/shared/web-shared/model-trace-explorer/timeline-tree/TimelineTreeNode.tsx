import {
  Button,
  Checkbox,
  Typography,
  useDesignSystemTheme,
  ChevronDownIcon,
  ChevronRightIcon,
  Tag,
  GavelIcon,
  GearIcon,
  LinkIcon,
  Tooltip,
} from '@databricks/design-system';

import type { HierarchyBar } from './TimelineTree.types';
import { getActiveChildIndex, TimelineTreeZIndex } from './TimelineTree.utils';
import { TimelineTreeHierarchyBars } from './TimelineTreeHierarchyBars';
import { TimelineTreeSpanTooltip } from './TimelineTreeSpanTooltip';
import { type ModelTraceSpanNode } from '../ModelTrace.types';
import { getSpanExceptionCount } from '../ModelTraceExplorer.utils';
import { useModelTraceExplorerViewState } from '../ModelTraceExplorerViewStateContext';
import { useGatewayTraceLink } from '../hooks/useGatewayTraceLink';
import { Link } from '../RoutingUtils';
import type { SpanEditState, SpanRangeSelectionResult } from '../edit-mode/useSpanRangeSelection';
import type { SpanRange } from '../hooks/useTraceViews';
import type { SpanViewRangeInfo } from '../hooks/useTraceViewFiltering';
import { getRangeColor } from '../edit-mode/rangeColors';
import { RangeBadge } from '../edit-mode/RangeBadge';
import { JsonFieldSelector } from '../edit-mode/JsonFieldSelector';

export interface TimelineTreeEditModeProps {
  selection: SpanRangeSelectionResult;
  ranges: SpanRange[];
  onRemoveRange: (index: number) => void;
  onUpdateRange: (index: number, updates: Partial<SpanRange>) => void;
}

export const TimelineTreeNode = ({
  node,
  selectedKey,
  expandedKeys,
  setExpandedKeys,
  traceStartTime,
  traceEndTime,
  onSelect,
  linesToRender,
  viewMatchedSpanKeys,
  viewRangeMap,
  viewRanges,
  editModeProps,
}: {
  node: ModelTraceSpanNode;
  selectedKey: string | number;
  expandedKeys: Set<string | number>;
  setExpandedKeys: (keys: Set<string | number>) => void;
  traceStartTime: number;
  traceEndTime: number;
  onSelect: ((node: ModelTraceSpanNode) => void) | undefined;
  // a boolean array that signifies whether or not a vertical
  // connecting line is supposed to in at the `i`th spacer. see
  // TimelineTreeHierarchyBars for more details.
  linesToRender: Array<HierarchyBar>;
  // When a trace view with a span filter is active, this set contains the keys
  // of spans that match the filter. Spans not in the set are visually dimmed.
  // null means no filter is active (all spans shown normally).
  viewMatchedSpanKeys?: Set<string | number> | null;
  // Per-span range info for the active view (range index, isFirst). Used to
  // render range badges and colored borders when viewing a saved trace view.
  viewRangeMap?: Map<string | number, SpanViewRangeInfo> | null;
  // The ranges from the active trace view, used alongside viewRangeMap for labels.
  viewRanges?: SpanRange[] | null;
  editModeProps?: TimelineTreeEditModeProps;
}) => {
  const expanded = expandedKeys.has(node.key);
  const { theme } = useDesignSystemTheme();
  const hasChildren = (node.children ?? []).length > 0;
  const { setAssessmentsPaneExpanded, selectedViewRangeIdx, setSelectedViewRangeIdx } = useModelTraceExplorerViewState();

  const isActive = selectedKey === node.key;
  const activeChildIndex = getActiveChildIndex(node, String(selectedKey));
  // true if a span has active children OR is the active span
  const isInActiveChain = activeChildIndex > -1;

  const hasException = getSpanExceptionCount(node) > 0;
  const gatewayTraceHref = useGatewayTraceLink(node.linkedGatewayTraceId);

  // Edit mode state
  const flatIndex = editModeProps?.selection.spanKeyToFlatIndex.get(String(node.key));
  const editState: SpanEditState | null =
    editModeProps && flatIndex !== undefined ? editModeProps.selection.getNodeEditState(flatIndex) : null;

  // View mode range state (when viewing a saved trace view, not editing)
  const viewRangeInfo = viewRangeMap?.get(node.key) ?? null;
  const viewRangeColor = viewRangeInfo ? getRangeColor(viewRangeInfo.rangeIdx) : null;

  const isDimmedByView = editState
    ? editState.isDimmed
    : viewMatchedSpanKeys != null && !viewMatchedSpanKeys.has(node.key);

  const editRangeColor = editState?.inRange ? editState.color : null;
  const activeRangeColor = editRangeColor ?? viewRangeColor;
  const editDragHighlight = editState?.inDrag ? 'rgba(59, 130, 246, 0.08)' : null;

  const backgroundColor = editDragHighlight
    ?? (activeRangeColor ? activeRangeColor.background : null)
    ?? (isActive ? theme.colors.actionDefaultBackgroundHover : 'transparent');

  return (
    <>
      {/* Edit mode badge */}
      {editModeProps && editState?.isFirstInRange && editState.rangeIdx !== undefined && editState.color && (
        <div css={{ padding: `${theme.spacing.xs}px ${theme.spacing.sm}px` }}>
          <RangeBadge
            label={editModeProps.ranges[editState.rangeIdx].label}
            color={editState.color}
            onDelete={() => editModeProps.onRemoveRange(editState.rangeIdx as number)}
          />
        </div>
      )}
      {/* View mode badge */}
      {!editModeProps && viewRangeInfo?.isFirstInRange && viewRangeColor && viewRanges && (
        <div css={{ padding: `${theme.spacing.xs}px ${theme.spacing.sm}px` }}>
          <RangeBadge
            label={viewRanges[viewRangeInfo.rangeIdx].label}
            color={viewRangeColor}
            onClick={() => {
              setSelectedViewRangeIdx(
                selectedViewRangeIdx === viewRangeInfo.rangeIdx ? null : viewRangeInfo.rangeIdx,
              );
            }}
            isSelected={selectedViewRangeIdx === viewRangeInfo.rangeIdx}
          />
        </div>
      )}
      <TimelineTreeSpanTooltip span={node}>
        <div
          data-testid={`timeline-tree-node-${node.key}`}
          css={{
            display: 'flex',
            flexDirection: 'column',
            width: '100%',
            cursor: 'pointer',
            boxSizing: 'border-box',
            backgroundColor,
            borderLeft: activeRangeColor ? `3px solid ${activeRangeColor.primary}` : '3px solid transparent',
            opacity: isDimmedByView ? 0.3 : 1,
            transition: 'opacity 150ms ease, background-color 150ms ease',
            ':hover': {
              backgroundColor: theme.colors.actionDefaultBackgroundHover,
            },
            ':active': {
              backgroundColor: theme.colors.actionDefaultBackgroundPress,
            },
          }}
          onClick={() => {
            onSelect?.(node);
          }}
          onPointerDown={editModeProps && flatIndex !== undefined ? () => editModeProps.selection.handlePointerDown(flatIndex) : undefined}
          onPointerMove={editModeProps && flatIndex !== undefined ? () => editModeProps.selection.handlePointerMove(flatIndex) : undefined}
        >
          <div
            css={{
              display: 'flex',
              flexDirection: 'row',
              alignItems: 'center',
              // add padding to root nodes, because they have no connecting lines
              padding: `0px ${theme.spacing.sm}px`,
              justifyContent: 'space-between',
              overflow: 'hidden',
              flex: 1,
            }}
          >
            <div css={{ display: 'flex', flexDirection: 'row', alignItems: 'center', overflow: 'hidden', flex: 1 }}>
              {editModeProps && flatIndex !== undefined && (
                <Checkbox
                  componentId={`timeline-tree-node.edit-checkbox-${node.key}`}
                  isChecked={editState?.inRange ?? false}
                  onChange={() => {
                    editModeProps.selection.handleCheckboxClick(flatIndex);
                  }}
                  onClick={(e: React.MouseEvent) => e.stopPropagation()}
                  aria-label={`Select ${node.title}`}
                  css={{ flexShrink: 0, marginRight: theme.spacing.xs }}
                />
              )}
              {hasChildren ? (
                <Button
                  size="small"
                  data-testid={`toggle-span-expanded-${node.key}`}
                  css={{ flexShrink: 0, marginRight: theme.spacing.xs }}
                  icon={expanded ? <ChevronDownIcon /> : <ChevronRightIcon />}
                  onClick={(event) => {
                    // prevent the node from being selected when the expand button is clicked
                    event.stopPropagation();
                    const newExpandedKeys = new Set(expandedKeys);
                    if (expanded) {
                      newExpandedKeys.delete(node.key);
                    } else {
                      newExpandedKeys.add(node.key);
                    }
                    setExpandedKeys(newExpandedKeys);
                  }}
                  componentId="shared.model-trace-explorer.toggle-span"
                />
              ) : (
                <div css={{ width: 24, marginRight: theme.spacing.xs }} />
              )}
              <TimelineTreeHierarchyBars
                isActiveSpan={isActive}
                isInActiveChain={isInActiveChain}
                linesToRender={linesToRender}
                hasChildren={hasChildren}
                isExpanded={expanded}
              />
              <span
                css={{
                  flexShrink: 0,
                  marginRight: theme.spacing.xs,
                  borderRadius: theme.borders.borderRadiusSm,
                  border: `1px solid ${
                    activeChildIndex > -1 ? theme.colors.blue500 : theme.colors.backgroundSecondary
                  }`,
                  zIndex: TimelineTreeZIndex.NORMAL,
                }}
              >
                {node.icon}
              </span>
              <Typography.Text
                color={hasException ? 'error' : 'primary'}
                css={{
                  overflow: 'hidden',
                  whiteSpace: 'nowrap',
                  textOverflow: 'ellipsis',
                  flex: 1,
                }}
              >
                {node.title}
              </Typography.Text>
              {gatewayTraceHref && (
                <Tooltip
                  content="View linked gateway trace"
                  componentId="shared.model-trace-explorer.gateway-trace-link"
                >
                  <Link
                    componentId="mlflow.model_trace_explorer.timeline.gateway_trace_link"
                    to={gatewayTraceHref}
                    target="_blank"
                    rel="noreferrer"
                    data-testid={`gateway-trace-link-${node.key}`}
                    onClick={(e: React.MouseEvent) => e.stopPropagation()}
                    css={{
                      flexShrink: 0,
                      display: 'flex',
                      alignItems: 'center',
                      marginLeft: theme.spacing.xs,
                      color: theme.colors.actionPrimaryBackgroundDefault,
                    }}
                  >
                    <LinkIcon css={{ fontSize: 14 }} />
                  </Link>
                </Tooltip>
              )}
              {node.assessments.length > 0 && (
                <Tag
                  color="indigo"
                  data-testid={`assessment-tag-${node.key}`}
                  componentId="shared.model-trace-explorer.assessment-count"
                  css={{
                    margin: 0,
                    borderRadius: theme.borders.borderRadiusSm,
                  }}
                  onClick={() => setAssessmentsPaneExpanded?.(true)}
                >
                  <GavelIcon />
                  <Typography.Text css={{ marginLeft: theme.spacing.xs }}>{node.assessments.length}</Typography.Text>
                </Tag>
              )}
              {editModeProps && editState?.inRange && (
                <Button
                  componentId={`timeline-tree-node.gear-${node.key}`}
                  type="tertiary"
                  size="small"
                  icon={<GearIcon />}
                  onClick={(e: React.MouseEvent) => {
                    e.stopPropagation();
                    editModeProps.selection.toggleExpandedSpan(String(node.key));
                  }}
                  aria-label="Configure JSON fields"
                  css={{ flexShrink: 0, marginLeft: theme.spacing.xs }}
                />
              )}
            </div>
          </div>
        </div>
      </TimelineTreeSpanTooltip>
      {editModeProps && editModeProps.selection.expandedSpanId === String(node.key) && editState?.inRange && editState.rangeIdx !== undefined && (
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
                selectedPath={editModeProps.ranges[editState.rangeIdx].input_path ?? null}
                onPathChange={(path) => editModeProps.onUpdateRange(editState.rangeIdx as number, { input_path: path })}
                label="Input Fields"
              />
            </div>
            <div css={{ flex: 1 }}>
              <JsonFieldSelector
                data={node.outputs}
                selectedPath={editModeProps.ranges[editState.rangeIdx].output_path ?? null}
                onPathChange={(path) => editModeProps.onUpdateRange(editState.rangeIdx as number, { output_path: path })}
                label="Output Fields"
              />
            </div>
          </div>
        </div>
      )}
      {expanded &&
        node.children?.map((child, idx) => (
          <TimelineTreeNode
            key={child.key}
            node={child}
            expandedKeys={expandedKeys}
            setExpandedKeys={setExpandedKeys}
            selectedKey={selectedKey}
            traceStartTime={traceStartTime}
            traceEndTime={traceEndTime}
            onSelect={onSelect}
            viewMatchedSpanKeys={viewMatchedSpanKeys}
            viewRangeMap={viewRangeMap}
            viewRanges={viewRanges}
            editModeProps={editModeProps}
            linesToRender={linesToRender.concat({
              // render the connecting line at this depth
              // if there are more children to render
              shouldRender: idx < (node.children?.length ?? 0) - 1,
              // make the vertical line blue if the active span
              // is below this child
              isActive: idx < activeChildIndex,
            })}
          />
        ))}
    </>
  );
};
