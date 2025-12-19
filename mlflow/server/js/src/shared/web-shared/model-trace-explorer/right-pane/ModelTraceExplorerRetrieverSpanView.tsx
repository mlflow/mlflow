import { isNil } from 'lodash';
import { useMemo, useState } from 'react';

import { Tag, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { ModelTraceExplorerRetrieverDocument } from './ModelTraceExplorerRetrieverDocument';
import type { ModelTraceSpanNode, RetrieverDocument, SearchMatch } from '../ModelTrace.types';
import { createListFromObject } from '../ModelTraceExplorer.utils';
import { ModelTraceExplorerCodeSnippet } from '../ModelTraceExplorerCodeSnippet';
import { ModelTraceExplorerCollapsibleSection } from '../ModelTraceExplorerCollapsibleSection';
import { ModelTraceExplorerRenderModeToggle } from '../ModelTraceExplorerRenderModeToggle';

export function ModelTraceExplorerRetrieverSpanView({
  activeSpan,
  className,
  searchFilter,
  activeMatch,
}: {
  activeSpan: ModelTraceSpanNode;
  className?: string;
  searchFilter: string;
  activeMatch: SearchMatch | null;
}) {
  const { theme } = useDesignSystemTheme();
  const [shouldRenderMarkdown, setShouldRenderMarkdown] = useState(true);
  const inputList = useMemo(() => createListFromObject(activeSpan.inputs), [activeSpan]);

  const outputs = activeSpan.outputs as RetrieverDocument[];

  const containsInputs = inputList.length > 0;

  // search highlighting is not supported in markdown rendering, so
  // if there is an active match in the documents, we have to render
  // them as code snippets.
  const isActiveMatchSpan = !isNil(activeMatch) && activeMatch.span.key === activeSpan.key;
  const outputsContainsActiveMatch = isActiveMatchSpan && activeMatch.section === 'outputs';

  return (
    <div className={className} data-testid="model-trace-explorer-retriever-span-view">
      {containsInputs && (
        <ModelTraceExplorerCollapsibleSection
          sectionKey="input"
          css={{ marginBottom: theme.spacing.sm }}
          title={
            <FormattedMessage
              defaultMessage="Inputs"
              description="Model trace explorer > selected span > inputs header"
            />
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

      <ModelTraceExplorerCollapsibleSection
        sectionKey="output"
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
            <div css={{ display: 'flex', flexDirection: 'row', gap: theme.spacing.sm }}>
              <FormattedMessage
                defaultMessage="Documents"
                description="Model trace explorer > retriever span > documents header"
              />
              <Tag componentId="shared.model-trace-explorer.document-count">{outputs.length}</Tag>
            </div>
            {!outputsContainsActiveMatch && (
              <ModelTraceExplorerRenderModeToggle
                shouldRenderMarkdown={shouldRenderMarkdown}
                setShouldRenderMarkdown={setShouldRenderMarkdown}
              />
            )}
          </div>
        }
      >
        {shouldRenderMarkdown && !outputsContainsActiveMatch ? (
          <div
            css={{
              border: `1px solid ${theme.colors.border}`,
              borderRadius: theme.legacyBorders.borderRadiusMd,
            }}
          >
            {outputs.map((document, idx) => (
              <div
                key={idx}
                css={{ borderBottom: idx !== outputs.length - 1 ? `1px solid ${theme.colors.border}` : '' }}
              >
                <ModelTraceExplorerRetrieverDocument
                  key={idx}
                  text={document.page_content}
                  metadata={document.metadata}
                />
              </div>
            ))}
          </div>
        ) : (
          <div
            css={{
              border: `1px solid ${theme.colors.border}`,
              borderRadius: theme.legacyBorders.borderRadiusMd,
              padding: theme.spacing.md,
            }}
          >
            <ModelTraceExplorerCodeSnippet
              title=""
              data={JSON.stringify(outputs, null, 2)}
              searchFilter={searchFilter}
              activeMatch={activeMatch}
              containsActiveMatch={isActiveMatchSpan && activeMatch.section === 'outputs'}
            />
          </div>
        )}
      </ModelTraceExplorerCollapsibleSection>
    </div>
  );
}
