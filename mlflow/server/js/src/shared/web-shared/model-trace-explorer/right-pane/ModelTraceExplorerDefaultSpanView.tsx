import { isNil } from 'lodash';
import { useMemo } from 'react';

import { Typography, useDesignSystemTheme } from '@databricks/design-system';

import { FormattedMessage } from '@databricks/i18n';

import type { ModelTraceSpanNode, SearchMatch } from '../ModelTrace.types';
import { createListFromObject } from '../ModelTraceExplorer.utils';
import { ModelTraceExplorerCodeSnippet } from '../ModelTraceExplorerCodeSnippet';
import { ModelTraceExplorerCollapsibleSection } from '../ModelTraceExplorerCollapsibleSection';
import { useModelTraceExplorerViewState } from '../ModelTraceExplorerViewStateContext';
import { applyJsonPathToObject } from '../hooks/useTraceViewFiltering';

export function ModelTraceExplorerDefaultSpanView({
  activeSpan,
  className,
  searchFilter,
  activeMatch,
}: {
  activeSpan: ModelTraceSpanNode | undefined;
  className?: string;
  searchFilter: string;
  activeMatch: SearchMatch | null;
}) {
  const { theme } = useDesignSystemTheme();
  const { activeTraceView } = useModelTraceExplorerViewState();

  const filteredInputs = useMemo(
    () => applyJsonPathToObject(activeSpan?.inputs, activeTraceView?.input_path),
    [activeSpan?.inputs, activeTraceView?.input_path],
  );
  const filteredOutputs = useMemo(
    () => applyJsonPathToObject(activeSpan?.outputs, activeTraceView?.output_path),
    [activeSpan?.outputs, activeTraceView?.output_path],
  );

  const inputList = useMemo(() => createListFromObject(filteredInputs as any), [filteredInputs]);
  const outputList = useMemo(() => createListFromObject(filteredOutputs as any), [filteredOutputs]);

  if (isNil(activeSpan)) {
    return null;
  }

  const containsInputs = inputList.length > 0;
  const containsOutputs = outputList.length > 0;

  const isActiveMatchSpan = !isNil(activeMatch) && activeMatch.span.key === activeSpan.key;

  const hasJsonPathFilter = !!(activeTraceView?.input_path || activeTraceView?.output_path);

  return (
    <div data-testid="model-trace-explorer-default-span-view">
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
      {containsInputs && (
        <ModelTraceExplorerCollapsibleSection
          withBorder
          css={{ marginBottom: theme.spacing.sm }}
          sectionKey="input"
          title={
            <div
              css={{
                display: 'flex',
                flexDirection: 'row',
                alignItems: 'center',
                justifyContent: 'space-between',
                width: '100%',
              }}
            >
              <FormattedMessage
                defaultMessage="Inputs"
                description="Model trace explorer > selected span > inputs header"
              />
            </div>
          }
        >
          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
            {inputList.map(({ key, value }, index) => (
              <ModelTraceExplorerCodeSnippet
                key={key || index}
                title={key}
                data={value}
                searchFilter={searchFilter}
                activeMatch={activeMatch}
                containsActiveMatch={isActiveMatchSpan && activeMatch.section === 'inputs' && activeMatch.key === key}
              />
            ))}
          </div>
        </ModelTraceExplorerCollapsibleSection>
      )}
      {containsOutputs && (
        <ModelTraceExplorerCollapsibleSection
          withBorder
          sectionKey="output"
          title={
            <div css={{ display: 'flex', flexDirection: 'row', justifyContent: 'space-between', width: '100%' }}>
              <FormattedMessage
                defaultMessage="Outputs"
                description="Model trace explorer > selected span > outputs header"
              />
            </div>
          }
        >
          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
            {outputList.map(({ key, value }) => (
              <ModelTraceExplorerCodeSnippet
                key={key}
                title={key}
                data={value}
                searchFilter={searchFilter}
                activeMatch={activeMatch}
                containsActiveMatch={isActiveMatchSpan && activeMatch.section === 'outputs' && activeMatch.key === key}
              />
            ))}
          </div>
        </ModelTraceExplorerCollapsibleSection>
      )}
    </div>
  );
}
