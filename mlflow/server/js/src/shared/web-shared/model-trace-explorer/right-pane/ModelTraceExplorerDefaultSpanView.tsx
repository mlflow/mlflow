import { isNil } from 'lodash';
import { useMemo } from 'react';

import { useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import type { ModelTraceSpanNode, SearchMatch } from '../ModelTrace.types';
import { CodeSnippetRenderMode } from '../ModelTrace.types';
import { createListFromObject, buildAggregatedJsonFromKeyValueList } from '../ModelTraceExplorer.utils';
import { ModelTraceExplorerCodeSnippet } from '../ModelTraceExplorerCodeSnippet';
import { ModelTraceExplorerCollapsibleSection } from '../ModelTraceExplorerCollapsibleSection';
import { ModelTraceExplorerFieldRenderer } from '../field-renderers/ModelTraceExplorerFieldRenderer';

export function ModelTraceExplorerDefaultSpanView({
  activeSpan,
  className,
  searchFilter,
  activeMatch,
  renderMode = 'default',
}: {
  activeSpan: ModelTraceSpanNode | undefined;
  className?: string;
  searchFilter: string;
  activeMatch: SearchMatch | null;
  renderMode?: 'default' | 'json' | 'table';
}) {
  const { theme } = useDesignSystemTheme();
  const inputList = useMemo(() => createListFromObject(activeSpan?.inputs), [activeSpan]);
  const outputList = useMemo(() => createListFromObject(activeSpan?.outputs), [activeSpan]);
  const aggregatedInputJson = useMemo(() => buildAggregatedJsonFromKeyValueList(inputList), [inputList]);
  const aggregatedOutputJson = useMemo(() => buildAggregatedJsonFromKeyValueList(outputList), [outputList]);

  if (isNil(activeSpan)) {
    return null;
  }

  const containsInputs = inputList.length > 0;
  const containsOutputs = outputList.length > 0;

  const isActiveMatchSpan = !isNil(activeMatch) && activeMatch.span.key === activeSpan.key;

  return (
    <div data-testid="model-trace-explorer-default-span-view">
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
          {renderMode === 'table' ? (
            <ModelTraceExplorerCodeSnippet
              title=""
              data={aggregatedInputJson}
              initialRenderMode={CodeSnippetRenderMode.TABLE}
              hideRenderModeDropdown
            />
          ) : renderMode === 'default' ? (
            <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
              {inputList.map(({ key, value }, index) => (
                <ModelTraceExplorerFieldRenderer
                  key={key || index}
                  title={key}
                  data={value}
                  renderMode={renderMode}
                  assessments={activeSpan?.assessments}
                />
              ))}
            </div>
          ) : (
            <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
              {inputList.map(({ key, value }, index) => (
                <ModelTraceExplorerCodeSnippet
                  key={key || index}
                  title={key}
                  data={value}
                  initialRenderMode={renderMode === 'json' ? CodeSnippetRenderMode.JSON : undefined}
                  searchFilter={searchFilter}
                  activeMatch={activeMatch}
                  containsActiveMatch={isActiveMatchSpan && activeMatch.section === 'inputs' && activeMatch.key === key}
                />
              ))}
            </div>
          )}
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
          {renderMode === 'table' ? (
            <ModelTraceExplorerCodeSnippet
              title=""
              data={aggregatedOutputJson}
              initialRenderMode={CodeSnippetRenderMode.TABLE}
              hideRenderModeDropdown
            />
          ) : renderMode === 'default' ? (
            <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
              {outputList.map(({ key, value }, index) => (
                <ModelTraceExplorerFieldRenderer
                  key={key || index}
                  title={key}
                  data={value}
                  renderMode={renderMode}
                  assessments={activeSpan?.assessments}
                />
              ))}
            </div>
          ) : (
            <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
              {outputList.map(({ key, value }) => (
                <ModelTraceExplorerCodeSnippet
                  key={key}
                  title={key}
                  data={value}
                  initialRenderMode={renderMode === 'json' ? CodeSnippetRenderMode.JSON : undefined}
                  searchFilter={searchFilter}
                  activeMatch={activeMatch}
                  containsActiveMatch={
                    isActiveMatchSpan && activeMatch.section === 'outputs' && activeMatch.key === key
                  }
                />
              ))}
            </div>
          )}
        </ModelTraceExplorerCollapsibleSection>
      )}
    </div>
  );
}
