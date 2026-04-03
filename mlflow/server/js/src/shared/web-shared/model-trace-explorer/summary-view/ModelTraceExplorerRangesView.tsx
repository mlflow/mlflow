import { useMemo, useState } from 'react';

import { ChevronDownIcon, ChevronRightIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';

import type { ModelTraceSpanNode } from '../ModelTrace.types';
import { createListFromObject } from '../ModelTraceExplorer.utils';
import { useModelTraceExplorerViewState } from '../ModelTraceExplorerViewStateContext';
import { ModelTraceExplorerFieldRenderer } from '../field-renderers/ModelTraceExplorerFieldRenderer';
import { spanMatchesSelector, applyJsonPathToObject } from '../hooks/useTraceViewFiltering';
import type { SpanRange, SpanSelector, TraceView } from '../hooks/useTraceViews';

/**
 * If the value is a JSON string, parse it so it gets rendered as structured data.
 */
function tryParseJson(value: unknown): unknown {
  if (typeof value !== 'string') return value;
  try {
    return JSON.parse(value);
  } catch {
    return value;
  }
}

/**
 * Find the first span in the nodeMap that matches the given selector.
 */
function findMatchingSpan(
  nodeMap: Record<string, ModelTraceSpanNode>,
  selector: SpanSelector | null | undefined,
): ModelTraceSpanNode | null {
  if (!selector) return null;
  for (const node of Object.values(nodeMap)) {
    if (spanMatchesSelector(node, selector)) {
      return node;
    }
  }
  return null;
}

/**
 * Renders a trace view's ranges as an ordered list of cards.
 * Each card shows the range's label, description, and extracted input/output content.
 * This replaces the span-based summary when a multi-range view is active.
 */
export const ModelTraceExplorerRangesView = ({ activeTraceView }: { activeTraceView: TraceView }) => {
  const { theme } = useDesignSystemTheme();
  const { nodeMap } = useModelTraceExplorerViewState();
  const sortedRanges = [...activeTraceView.ranges].sort((a, b) => a.position - b.position);

  // Position 0 is the summary; the rest are steps
  const summaryRange = sortedRanges.find((r) => r.position === 0);
  const stepRanges = sortedRanges.filter((r) => r.position > 0);

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        flex: 1,
        minHeight: 0,
        overflow: 'auto',
        padding: theme.spacing.md,
        gap: theme.spacing.md,
      }}
    >
      {/* View header */}
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
        <Typography.Title level={4} withoutMargins>
          {activeTraceView.name}
        </Typography.Title>
        {activeTraceView.created_by && (
          <Typography.Text size="sm" color="secondary">
            Created by {activeTraceView.created_by}
          </Typography.Text>
        )}
      </div>

      {/* Summary range (position 0) */}
      {summaryRange?.description && (
        <div
          css={{
            padding: theme.spacing.md,
            backgroundColor: theme.colors.backgroundSecondary,
            borderRadius: theme.borders.borderRadiusMd,
            border: `1px solid ${theme.colors.border}`,
          }}
        >
          <Typography.Text>{summaryRange.description}</Typography.Text>
        </div>
      )}

      {/* Step ranges */}
      {stepRanges.length > 0 && (
        <div css={{ display: 'flex', flexDirection: 'column', gap: 0 }}>
          {stepRanges.map((range, index) => (
            <RangeStepCard
              key={range.range_id ?? index}
              range={range}
              nodeMap={nodeMap}
              stepNumber={index + 1}
              isLast={index === stepRanges.length - 1}
            />
          ))}
        </div>
      )}
    </div>
  );
};

const RangeStepCard = ({
  range,
  nodeMap,
  stepNumber,
  isLast,
}: {
  range: SpanRange;
  nodeMap: Record<string, ModelTraceSpanNode>;
  stepNumber: number;
  isLast: boolean;
}) => {
  const { theme } = useDesignSystemTheme();
  const [inputExpanded, setInputExpanded] = useState(!!range.input_path);
  const [outputExpanded, setOutputExpanded] = useState(!!range.output_path);

  // Resolve spans: input from from_selector span, output from to_selector span (or from_selector if no to)
  const fromSpan = useMemo(() => findMatchingSpan(nodeMap, range.from_selector), [nodeMap, range.from_selector]);
  const toSpan = useMemo(
    () => (range.to_selector ? findMatchingSpan(nodeMap, range.to_selector) : fromSpan),
    [nodeMap, range.to_selector, fromSpan],
  );

  // Extract content via JSONPath (applyJsonPathToObject returns raw data when path is null)
  const extractedInput = useMemo(
    () => (fromSpan ? applyJsonPathToObject(fromSpan.inputs, range.input_path) : null),
    [fromSpan, range.input_path],
  );
  const extractedOutput = useMemo(
    () => (toSpan ? applyJsonPathToObject(toSpan.outputs, range.output_path) : null),
    [toSpan, range.output_path],
  );

  const inputList = useMemo(
    () => (extractedInput != null ? createListFromObject(tryParseJson(extractedInput) as any) : []),
    [extractedInput],
  );
  const outputList = useMemo(
    () => (extractedOutput != null ? createListFromObject(tryParseJson(extractedOutput) as any) : []),
    [extractedOutput],
  );

  return (
    <div css={{ display: 'flex', flexDirection: 'row' }}>
      {/* Timeline rail */}
      <div
        css={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          width: 28,
          flexShrink: 0,
          paddingTop: 2,
        }}
      >
        <div
          css={{
            width: 22,
            height: 22,
            borderRadius: '50%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: 11,
            fontWeight: 600,
            lineHeight: 1,
            backgroundColor: theme.colors.actionPrimaryBackgroundDefault,
            color: '#fff',
            border: `1px solid ${theme.colors.actionPrimaryBackgroundDefault}`,
          }}
        >
          {stepNumber}
        </div>
        {!isLast && (
          <div
            css={{
              width: 2,
              flex: 1,
              minHeight: theme.spacing.sm,
              backgroundColor: theme.colors.border,
            }}
          />
        )}
      </div>

      {/* Card content */}
      <div
        css={{
          flex: 1,
          minWidth: 0,
          marginLeft: theme.spacing.sm,
          marginBottom: isLast ? 0 : theme.spacing.sm,
        }}
      >
        <Typography.Text bold css={{ display: 'block', padding: `${theme.spacing.xs}px 0` }}>
          {range.label || `Range ${range.position}`}
        </Typography.Text>

        <div
          css={{
            display: 'flex',
            flexDirection: 'column',
            gap: theme.spacing.sm,
            padding: theme.spacing.sm,
            border: `1px solid ${theme.colors.border}`,
            borderRadius: theme.borders.borderRadiusMd,
            backgroundColor: theme.colors.backgroundSecondary,
          }}
        >
          {range.description && <Typography.Text size="sm">{range.description}</Typography.Text>}
          {inputList.length > 0 && (
            <CollapsibleIOSection label="Input" expanded={inputExpanded} onToggle={() => setInputExpanded(!inputExpanded)}>
              {inputList.map(({ key, value }, index) => (
                <ModelTraceExplorerFieldRenderer key={key || index} title={key} data={value} renderMode="default" />
              ))}
            </CollapsibleIOSection>
          )}
          {outputList.length > 0 && (
            <CollapsibleIOSection
              label="Output"
              expanded={outputExpanded}
              onToggle={() => setOutputExpanded(!outputExpanded)}
            >
              {outputList.map(({ key, value }, index) => (
                <ModelTraceExplorerFieldRenderer key={key || index} title={key} data={value} renderMode="default" />
              ))}
            </CollapsibleIOSection>
          )}
        </div>
      </div>
    </div>
  );
};

const CollapsibleIOSection = ({
  label,
  expanded,
  onToggle,
  children,
}: {
  label: string;
  expanded: boolean;
  onToggle: () => void;
  children: React.ReactNode;
}) => {
  const { theme } = useDesignSystemTheme();
  return (
    <div>
      <div
        role="button"
        tabIndex={0}
        onClick={onToggle}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') onToggle();
        }}
        css={{
          display: 'flex',
          alignItems: 'center',
          gap: theme.spacing.xs,
          cursor: 'pointer',
          userSelect: 'none',
        }}
      >
        {expanded ? (
          <ChevronDownIcon css={{ color: theme.colors.textSecondary, flexShrink: 0, width: 14, height: 14 }} />
        ) : (
          <ChevronRightIcon css={{ color: theme.colors.textSecondary, flexShrink: 0, width: 14, height: 14 }} />
        )}
        <Typography.Text size="sm" bold color="secondary">
          {label}
        </Typography.Text>
      </div>
      {expanded && (
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs, paddingLeft: theme.spacing.md }}>
          {children}
        </div>
      )}
    </div>
  );
};
