import { useMemo } from 'react';

import { Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import type { ModelTraceSpanNode } from '../ModelTrace.types';
import { createListFromObject } from '../ModelTraceExplorer.utils';
import { ModelTraceExplorerCodeSnippet } from '../ModelTraceExplorerCodeSnippet';
import { ModelTraceExplorerCollapsibleSection } from '../ModelTraceExplorerCollapsibleSection';
import { useModelTraceExplorerViewState } from '../ModelTraceExplorerViewStateContext';
import { DeeplinkText } from '../DeeplinkText';
import { applyJsonPathToObject, spanMatchesSelector } from '../hooks/useTraceViewFiltering';
import { getTimelineTreeNodesList } from '../timeline-tree/TimelineTree.utils';
import type { PathSelection, SpanRange } from '../hooks/useTraceViews';

/**
 * Resolve a list of PathSelections into {key, value} items for display.
 * Each item's key is prefixed with the source span name so it's clear
 * where the data came from when multiple spans contribute.
 */
function resolveSelections(
  selections: PathSelection[],
  flatNodes: ModelTraceSpanNode[],
  dataKey: 'inputs' | 'outputs',
): { key: string; value: string }[] {
  const items: { key: string; value: string }[] = [];
  for (const sel of selections) {
    const span = flatNodes.find((n) => spanMatchesSelector(n, sel.span_selector));
    if (!span) continue;
    const raw = dataKey === 'inputs' ? span.inputs : span.outputs;
    const filtered = applyJsonPathToObject(raw, sel.path);
    const list = createListFromObject(filtered as any);
    const spanName = String(span.title);
    for (const item of list) {
      items.push({ key: item.key ? `${spanName} / ${item.key}` : spanName, value: item.value });
    }
  }
  return items;
}

export function ModelTraceExplorerRangeDetailView({
  range,
  activeSpan: _activeSpan,
}: {
  range: SpanRange;
  activeSpan: ModelTraceSpanNode | undefined;
}) {
  const { theme } = useDesignSystemTheme();
  const { topLevelNodes } = useModelTraceExplorerViewState();

  const flatNodes = useMemo(() => getTimelineTreeNodesList(topLevelNodes), [topLevelNodes]);

  const hasInputSelections = (range.input_selections?.length ?? 0) > 0;
  const hasOutputSelections = (range.output_selections?.length ?? 0) > 0;

  // Resolve inputs and outputs independently: use selections when present,
  // otherwise fall back to legacy input_path/output_path on boundary spans.
  const { inputList, outputList } = useMemo(() => {
    let inputItems: { key: string; value: string }[];
    let outputItems: { key: string; value: string }[];

    if (hasInputSelections) {
      inputItems = resolveSelections(range.input_selections!, flatNodes, 'inputs');
    } else {
      const from = flatNodes.find((n) => spanMatchesSelector(n, range.from_selector));
      const filteredInputs = applyJsonPathToObject(from?.inputs, range.input_path);
      inputItems = createListFromObject(filteredInputs as any);
    }

    if (hasOutputSelections) {
      outputItems = resolveSelections(range.output_selections!, flatNodes, 'outputs');
    } else {
      const from = flatNodes.find((n) => spanMatchesSelector(n, range.from_selector));
      const to = range.to_selector ? flatNodes.find((n) => spanMatchesSelector(n, range.to_selector)) : null;
      const outputSpan = to ?? from;
      const filteredOutputs = applyJsonPathToObject(outputSpan?.outputs, range.output_path);
      outputItems = createListFromObject(filteredOutputs as any);
    }

    return { inputList: inputItems, outputList: outputItems };
  }, [flatNodes, range, hasInputSelections, hasOutputSelections]);

  const containsInputs = inputList.length > 0;
  const containsOutputs = outputList.length > 0;

  return (
    <div css={{ padding: theme.spacing.md }}>
      <div css={{ marginBottom: theme.spacing.md }}>
        <Typography.Title level={4} css={{ marginBottom: theme.spacing.xs }}>
          {range.label}
        </Typography.Title>
        {range.description && <DeeplinkText text={range.description} size="sm" color="secondary" />}
      </div>
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
            {outputList.map(({ key, value }, index) => (
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
    </div>
  );
}
