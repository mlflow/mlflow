import { useMemo, useState } from 'react';

import { ChevronDownIcon, ChevronRightIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import type { ModelTraceExplorerRenderMode, ModelTraceSpanNode } from '../ModelTrace.types';
import { createListFromObject, getSpanExceptionEvents, useIntermediateNodes } from '../ModelTraceExplorer.utils';
import { ModelTraceExplorerFieldRenderer } from '../field-renderers/ModelTraceExplorerFieldRenderer';
import { useModelTraceExplorerViewState } from '../ModelTraceExplorerViewStateContext';
import { useTraceViewSpanMatches, applyJsonPathToObject } from '../hooks/useTraceViewFiltering';
import { spanTimeFormatter } from '../timeline-tree/TimelineTree.utils';

/**
 * A compact, mobile-friendly summary view designed for sharing with business users.
 * Uses a card-based flow layout: Input → Steps → Output with side-by-side rendering
 * when horizontal space permits.
 */
export const ModelTraceExplorerCompactSummaryView = () => {
  const { theme } = useDesignSystemTheme();
  const { rootNode, activeTraceView, topLevelNodes } = useModelTraceExplorerViewState();
  const intermediateNodes = useIntermediateNodes(rootNode);
  const viewMatchedSpanKeys = useTraceViewSpanMatches(topLevelNodes, activeTraceView);

  const rootInputs = rootNode?.inputs;
  const rootOutputs = rootNode?.outputs;

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

  if (!rootNode) {
    return null;
  }

  const hasSteps = intermediateNodes.length > 0;

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
      {/* Top-level I/O: side by side on wide screens, stacked on narrow */}
      <div
        css={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
          gap: theme.spacing.md,
        }}
      >
        <CompactIOCard label="Input" items={inputList} renderMode="default" chatMessageFormat={rootNode.chatMessageFormat} theme={theme} />
        <CompactIOCard label="Output" items={outputList} renderMode="default" chatMessageFormat={rootNode.chatMessageFormat} theme={theme} assessments={rootNode.assessments} />
      </div>

      {/* Steps section — only shown when there are intermediate spans */}
      {hasSteps && (
        <div css={{ display: 'flex', flexDirection: 'column', gap: 0 }}>
          <Typography.Title level={4} withoutMargins css={{ marginBottom: theme.spacing.sm, color: theme.colors.textSecondary }}>
            <FormattedMessage
              defaultMessage="{count, plural, one {# step} other {# steps}}"
              description="Compact summary view steps count header"
              values={{ count: intermediateNodes.length }}
            />
          </Typography.Title>
          {intermediateNodes.map((node, index) => (
            <CompactStepCard
              key={node.key}
              node={node}
              stepNumber={index + 1}
              isDimmed={viewMatchedSpanKeys != null && !viewMatchedSpanKeys.has(node.key)}
              isHighlighted={viewMatchedSpanKeys != null && viewMatchedSpanKeys.has(node.key)}
              activeTraceView={activeTraceView}
              isLast={index === intermediateNodes.length - 1}
            />
          ))}
        </div>
      )}
    </div>
  );
};

const CompactIOCard = ({
  label,
  items,
  renderMode,
  chatMessageFormat,
  theme,
  assessments,
}: {
  label: string;
  items: { key: string; value: string }[];
  renderMode: ModelTraceExplorerRenderMode;
  chatMessageFormat?: string;
  theme: any;
  assessments?: any[];
}) => {
  if (items.length === 0) {
    return null;
  }

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        border: `1px solid ${theme.colors.border}`,
        borderRadius: theme.borders.borderRadiusMd,
        overflow: 'hidden',
      }}
    >
      <div
        css={{
          padding: `${theme.spacing.xs}px ${theme.spacing.sm}px`,
          backgroundColor: theme.colors.backgroundSecondary,
          borderBottom: `1px solid ${theme.colors.border}`,
        }}
      >
        <Typography.Text bold size="sm" color="secondary">
          {label}
        </Typography.Text>
      </div>
      <div
        css={{
          display: 'flex',
          flexDirection: 'column',
          gap: theme.spacing.sm,
          padding: theme.spacing.sm,
        }}
      >
        {items.map(({ key, value }, index) => (
          <ModelTraceExplorerFieldRenderer
            key={key || index}
            title={key}
            data={value}
            renderMode={renderMode}
            chatMessageFormat={chatMessageFormat}
            assessments={assessments}
          />
        ))}
      </div>
    </div>
  );
};

const CompactStepCard = ({
  node,
  stepNumber,
  isDimmed,
  isHighlighted,
  activeTraceView,
  isLast,
}: {
  node: ModelTraceSpanNode;
  stepNumber: number;
  isDimmed: boolean;
  isHighlighted: boolean;
  activeTraceView: any;
  isLast: boolean;
}) => {
  const { theme } = useDesignSystemTheme();
  const [collapsed, setCollapsed] = useState(false);

  const filteredInputs = useMemo(
    () => applyJsonPathToObject(node.inputs, activeTraceView?.input_path),
    [node.inputs, activeTraceView?.input_path],
  );
  const filteredOutputs = useMemo(
    () => applyJsonPathToObject(node.outputs, activeTraceView?.output_path),
    [node.outputs, activeTraceView?.output_path],
  );

  const inputList = useMemo(() => createListFromObject(filteredInputs as any), [filteredInputs]);
  const outputList = useMemo(() => createListFromObject(filteredOutputs as any), [filteredOutputs]);
  const exceptions = getSpanExceptionEvents(node);
  const hasError = exceptions.length > 0;
  const duration = spanTimeFormatter(node.end - node.start);

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'row',
        opacity: isDimmed ? 0.35 : 1,
        transition: 'opacity 150ms ease',
      }}
    >
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
            backgroundColor: hasError
              ? theme.colors.actionDangerPrimaryBackgroundDefault
              : isHighlighted
                ? theme.colors.actionPrimaryBackgroundDefault
                : theme.colors.backgroundSecondary,
            color: hasError || isHighlighted ? '#fff' : theme.colors.textSecondary,
            border: `1px solid ${hasError ? theme.colors.actionDangerPrimaryBackgroundDefault : isHighlighted ? theme.colors.actionPrimaryBackgroundDefault : theme.colors.border}`,
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
        <div
          role="button"
          tabIndex={0}
          onClick={() => setCollapsed(!collapsed)}
          onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') setCollapsed(!collapsed); }}
          css={{
            display: 'flex',
            alignItems: 'center',
            gap: theme.spacing.xs,
            cursor: 'pointer',
            padding: `${theme.spacing.xs}px 0`,
            userSelect: 'none',
          }}
        >
          {collapsed ? (
            <ChevronRightIcon css={{ color: theme.colors.textSecondary, flexShrink: 0 }} />
          ) : (
            <ChevronDownIcon css={{ color: theme.colors.textSecondary, flexShrink: 0 }} />
          )}
          <Typography.Text bold css={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            {node.title}
          </Typography.Text>
          <Typography.Text size="sm" color="secondary" css={{ flexShrink: 0, marginLeft: 'auto' }}>
            {duration}
          </Typography.Text>
        </div>

        {!collapsed && (
          <div
            css={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))',
              gap: theme.spacing.sm,
              padding: theme.spacing.sm,
              border: `1px solid ${theme.colors.border}`,
              borderRadius: theme.borders.borderRadiusMd,
              backgroundColor: theme.colors.backgroundSecondary,
            }}
          >
            {inputList.length > 0 && (
              <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
                <Typography.Text size="sm" bold color="secondary">
                  Input
                </Typography.Text>
                {inputList.map(({ key, value }, index) => (
                  <ModelTraceExplorerFieldRenderer
                    key={key || index}
                    title={key}
                    data={value}
                    renderMode="default"
                    chatMessageFormat={node.chatMessageFormat}
                  />
                ))}
              </div>
            )}
            {outputList.length > 0 && (
              <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
                <Typography.Text size="sm" bold color="secondary">
                  Output
                </Typography.Text>
                {outputList.map(({ key, value }, index) => (
                  <ModelTraceExplorerFieldRenderer
                    key={key || index}
                    title={key}
                    data={value}
                    renderMode="default"
                    chatMessageFormat={node.chatMessageFormat}
                    assessments={node.assessments}
                  />
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
