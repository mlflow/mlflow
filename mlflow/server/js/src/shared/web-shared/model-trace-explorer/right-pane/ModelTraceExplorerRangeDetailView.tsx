import { useMemo } from 'react';

import { Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import type { ModelTraceSpanNode } from '../ModelTrace.types';
import { createListFromObject } from '../ModelTraceExplorer.utils';
import { ModelTraceExplorerCodeSnippet } from '../ModelTraceExplorerCodeSnippet';
import { ModelTraceExplorerCollapsibleSection } from '../ModelTraceExplorerCollapsibleSection';
import { useModelTraceExplorerViewState } from '../ModelTraceExplorerViewStateContext';
import { applyJsonPathToObject, spanMatchesSelector } from '../hooks/useTraceViewFiltering';
import { getTimelineTreeNodesList } from '../timeline-tree/TimelineTree.utils';
import type { SpanRange } from '../hooks/useTraceViews';

export function ModelTraceExplorerRangeDetailView({
  range,
  activeSpan: _activeSpan,
}: {
  range: SpanRange;
  activeSpan: ModelTraceSpanNode | undefined;
}) {
  const { theme } = useDesignSystemTheme();
  const { topLevelNodes } = useModelTraceExplorerViewState();

  // Find the spans matching from_selector and to_selector so we show the
  // range's own inputs/outputs rather than the currently selected span's.
  const { fromSpan, toSpan } = useMemo(() => {
    const flatNodes = getTimelineTreeNodesList(topLevelNodes);
    const from = flatNodes.find((n) => spanMatchesSelector(n, range.from_selector));
    const to = range.to_selector ? flatNodes.find((n) => spanMatchesSelector(n, range.to_selector)) : null;
    return { fromSpan: from, toSpan: to };
  }, [topLevelNodes, range.from_selector, range.to_selector]);

  const inputSpan = fromSpan;
  const outputSpan = toSpan ?? fromSpan;

  const filteredInputs = useMemo(
    () => applyJsonPathToObject(inputSpan?.inputs, range.input_path),
    [inputSpan?.inputs, range.input_path],
  );
  const filteredOutputs = useMemo(
    () => applyJsonPathToObject(outputSpan?.outputs, range.output_path),
    [outputSpan?.outputs, range.output_path],
  );

  const inputList = useMemo(() => createListFromObject(filteredInputs as any), [filteredInputs]);
  const outputList = useMemo(() => createListFromObject(filteredOutputs as any), [filteredOutputs]);

  const containsInputs = inputList.length > 0;
  const containsOutputs = outputList.length > 0;

  return (
    <div css={{ padding: theme.spacing.md }}>
      <div css={{ marginBottom: theme.spacing.md }}>
        <Typography.Title level={4} css={{ marginBottom: theme.spacing.xs }}>
          {range.label}
        </Typography.Title>
        {range.description && (
          <Typography.Text color="secondary" size="sm">
            {range.description}
          </Typography.Text>
        )}
      </div>
      {(range.input_path || range.output_path) && (
        <div
          css={{
            display: 'flex',
            flexDirection: 'column',
            gap: theme.spacing.xs,
            marginBottom: theme.spacing.md,
            padding: theme.spacing.sm,
            backgroundColor: theme.colors.backgroundSecondary,
            borderRadius: theme.borders.borderRadiusMd,
          }}
        >
          {range.input_path && (
            <Typography.Text size="sm" color="secondary">
              Input path: <code>{range.input_path}</code>
            </Typography.Text>
          )}
          {range.output_path && (
            <Typography.Text size="sm" color="secondary">
              Output path: <code>{range.output_path}</code>
            </Typography.Text>
          )}
        </div>
      )}
      {containsInputs && (
        <ModelTraceExplorerCollapsibleSection
          withBorder
          css={{ marginBottom: theme.spacing.sm }}
          sectionKey="range-input"
          title={
            <FormattedMessage
              defaultMessage="Inputs"
              description="Model trace explorer > range detail > inputs header"
            />
          }
        >
          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
            {inputList.map(({ key, value }, index) => (
              <ModelTraceExplorerCodeSnippet
                key={key || index}
                title={key}
                data={value}
                searchFilter=""
                activeMatch={null}
                containsActiveMatch={false}
              />
            ))}
          </div>
        </ModelTraceExplorerCollapsibleSection>
      )}
      {containsOutputs && (
        <ModelTraceExplorerCollapsibleSection
          withBorder
          sectionKey="range-output"
          title={
            <FormattedMessage
              defaultMessage="Outputs"
              description="Model trace explorer > range detail > outputs header"
            />
          }
        >
          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
            {outputList.map(({ key, value }) => (
              <ModelTraceExplorerCodeSnippet
                key={key}
                title={key}
                data={value}
                searchFilter=""
                activeMatch={null}
                containsActiveMatch={false}
              />
            ))}
          </div>
        </ModelTraceExplorerCollapsibleSection>
      )}
    </div>
  );
}
