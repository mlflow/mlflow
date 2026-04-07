import { useState, useCallback, useMemo } from 'react';
import { Button, Checkbox, useDesignSystemTheme } from '@databricks/design-system';
import { CloseIcon, GearIcon } from '@databricks/design-system';
import type { ModelTraceSpanNode } from '../ModelTrace.types';
import type { SpanRange, SpanSelector } from '../hooks/useTraceViews';
import { getRangeColor } from './rangeColors';
import { JsonFieldSelector } from './JsonFieldSelector';

interface SpanRangeOverlayProps {
  nodes: ModelTraceSpanNode[];
  ranges: SpanRange[];
  onAddRange: (from: SpanSelector, to?: SpanSelector) => void;
  onRemoveRange: (index: number) => void;
  onUpdateRange: (index: number, updates: Partial<SpanRange>) => void;
}

const findRangeForSpan = (
  spanId: string,
  nodes: ModelTraceSpanNode[],
  ranges: SpanRange[],
): number | null => {
  for (let rangeIdx = 0; rangeIdx < ranges.length; rangeIdx++) {
    const range = ranges[rangeIdx];
    const fromId = range.from_selector.span_id;
    const toId = range.to_selector?.span_id;

    if (!toId) {
      if (spanId === fromId) return rangeIdx;
    } else {
      let inRange = false;
      for (const node of nodes) {
        const nodeId = String(node.key);
        if (nodeId === fromId) inRange = true;
        if (inRange && nodeId === spanId) return rangeIdx;
        if (nodeId === toId && inRange) break;
      }
    }
  }
  return null;
};

export const SpanRangeOverlay = ({
  nodes,
  ranges,
  onAddRange,
  onRemoveRange,
  onUpdateRange,
}: SpanRangeOverlayProps) => {
  const { theme } = useDesignSystemTheme();
  const [expandedSpanId, setExpandedSpanId] = useState<string | null>(null);
  const [dragState, setDragState] = useState<{ startIdx: number; currentIdx: number } | null>(null);

  const spanRangeMap = useMemo(() => {
    const map = new Map<string, number>();
    for (const node of nodes) {
      const rangeIdx = findRangeForSpan(String(node.key), nodes, ranges);
      if (rangeIdx !== null) map.set(String(node.key), rangeIdx);
    }
    return map;
  }, [nodes, ranges]);

  const rangeFirstSpanId = useMemo(() => {
    const result = new Map<number, string>();
    for (let rangeIdx = 0; rangeIdx < ranges.length; rangeIdx++) {
      const fromId = ranges[rangeIdx].from_selector.span_id;
      if (fromId) result.set(rangeIdx, fromId);
    }
    return result;
  }, [ranges]);

  const handleCheckboxClick = useCallback(
    (nodeIdx: number) => {
      const node = nodes[nodeIdx];
      const spanId = String(node.key);
      const existingRange = spanRangeMap.get(spanId);
      if (existingRange !== undefined) {
        onRemoveRange(existingRange);
      } else {
        onAddRange({ span_id: spanId });
      }
    },
    [nodes, spanRangeMap, onAddRange, onRemoveRange],
  );

  const handlePointerDown = useCallback((nodeIdx: number) => {
    setDragState({ startIdx: nodeIdx, currentIdx: nodeIdx });
  }, []);

  const handlePointerMove = useCallback(
    (nodeIdx: number) => {
      if (dragState) {
        setDragState((prev) => (prev ? { ...prev, currentIdx: nodeIdx } : prev));
      }
    },
    [dragState],
  );

  const handlePointerUp = useCallback(() => {
    if (dragState) {
      const { startIdx, currentIdx } = dragState;
      if (startIdx !== currentIdx) {
        const fromIdx = Math.min(startIdx, currentIdx);
        const toIdx = Math.max(startIdx, currentIdx);
        const fromNode = nodes[fromIdx];
        const toNode = nodes[toIdx];
        onAddRange({ span_id: String(fromNode.key) }, { span_id: String(toNode.key) });
      }
      setDragState(null);
    }
  }, [dragState, nodes, onAddRange]);

  const isDragging = dragState !== null && dragState.startIdx !== dragState.currentIdx;
  const dragMin = dragState ? Math.min(dragState.startIdx, dragState.currentIdx) : -1;
  const dragMax = dragState ? Math.max(dragState.startIdx, dragState.currentIdx) : -1;

  return (
    <div
      css={{ userSelect: 'none' }}
      onPointerUp={handlePointerUp}
      onPointerLeave={handlePointerUp}
    >
      {nodes.map((node, idx) => {
        const spanId = String(node.key);
        const rangeIdx = spanRangeMap.get(spanId);
        const inRange = rangeIdx !== undefined;
        const color = inRange ? getRangeColor(rangeIdx) : null;
        const isFirstInRange = inRange && rangeFirstSpanId.get(rangeIdx) === spanId;
        const inDrag = isDragging && idx >= dragMin && idx <= dragMax;
        const isDimmed = !inRange && !inDrag;

        return (
          <div key={spanId}>
            {isFirstInRange && rangeIdx !== undefined && (
              <div
                css={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: theme.spacing.xs,
                  padding: `${theme.spacing.xs}px ${theme.spacing.sm}px`,
                }}
              >
                <div
                  css={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: theme.spacing.xs,
                    padding: `2px ${theme.spacing.sm}px`,
                    backgroundColor: color!.background,
                    border: `1px solid ${color!.primary}33`,
                    borderRadius: theme.borders.borderRadiusMd,
                  }}
                >
                  <div
                    css={{
                      width: 8,
                      height: 8,
                      borderRadius: '50%',
                      backgroundColor: color!.primary,
                    }}
                  />
                  <span
                    css={{
                      color: color!.primary,
                      fontSize: theme.typography.fontSizeSm,
                      fontWeight: 600,
                    }}
                  >
                    {ranges[rangeIdx].label}
                  </span>
                  <Button
                    componentId={`span-range-overlay.delete-range-${rangeIdx}`}
                    type="tertiary"
                    size="small"
                    icon={<CloseIcon />}
                    onClick={() => onRemoveRange(rangeIdx)}
                    aria-label="Delete range"
                    css={{ marginLeft: theme.spacing.xs }}
                  />
                </div>
              </div>
            )}

            <div
              css={{
                display: 'flex',
                alignItems: 'center',
                gap: theme.spacing.sm,
                padding: `${theme.spacing.xs}px ${theme.spacing.sm}px`,
                opacity: isDimmed ? 0.4 : 1,
                backgroundColor: inDrag
                  ? 'rgba(59, 130, 246, 0.08)'
                  : color?.background ?? 'transparent',
                borderLeft: inRange ? `3px solid ${color!.primary}` : '3px solid transparent',
                transition: 'opacity 0.15s, background-color 0.15s',
              }}
              onPointerDown={() => handlePointerDown(idx)}
              onPointerMove={() => handlePointerMove(idx)}
            >
              <Checkbox
                componentId={`span-range-overlay.checkbox-${spanId}`}
                isChecked={inRange}
                onChange={() => handleCheckboxClick(idx)}
                aria-label={String(node.title)}
              />
              <span
                css={{
                  flex: 1,
                  fontSize: theme.typography.fontSizeSm,
                  color: theme.colors.textPrimary,
                }}
              >
                {node.title}
              </span>
              {inRange && (
                <Button
                  componentId={`span-range-overlay.gear-${spanId}`}
                  type="tertiary"
                  size="small"
                  icon={<GearIcon />}
                  onClick={() => setExpandedSpanId(expandedSpanId === spanId ? null : spanId)}
                  aria-label="Configure JSON fields"
                />
              )}
            </div>

            {expandedSpanId === spanId && inRange && rangeIdx !== undefined && (
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
                      selectedPath={ranges[rangeIdx].input_path ?? null}
                      onPathChange={(path) => onUpdateRange(rangeIdx, { input_path: path })}
                      label="Input Fields"
                    />
                  </div>
                  <div css={{ flex: 1 }}>
                    <JsonFieldSelector
                      data={node.outputs}
                      selectedPath={ranges[rangeIdx].output_path ?? null}
                      onPathChange={(path) => onUpdateRange(rangeIdx, { output_path: path })}
                      label="Output Fields"
                    />
                  </div>
                </div>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
};
