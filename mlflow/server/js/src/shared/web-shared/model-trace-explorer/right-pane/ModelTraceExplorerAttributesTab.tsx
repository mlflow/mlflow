import { isNil, keys } from 'lodash';

import { Empty, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { CodeSnippetRenderMode, type ModelTraceSpanNode, type SearchMatch } from '../ModelTrace.types';
import { ModelTraceExplorerCodeSnippet } from '../ModelTraceExplorerCodeSnippet';
import { useModelTraceExplorerPreferences } from '../ModelTraceExplorerPreferencesContext';
import { SpanModelCostBadge } from './SpanModelCostBadge';

export function ModelTraceExplorerAttributesTab({
  activeSpan,
  searchFilter,
  activeMatch,
}: {
  activeSpan: ModelTraceSpanNode;
  searchFilter: string;
  activeMatch: SearchMatch | null;
}) {
  const { theme } = useDesignSystemTheme();
  const { renderMode } = useModelTraceExplorerPreferences();
  const { attributes } = activeSpan;
  const containsAttributes = keys(attributes).length > 0;
  const isActiveMatchSpan = !isNil(activeMatch) && activeMatch.span.key === activeSpan.key;
  const initialRenderMode =
    renderMode === 'json'
      ? CodeSnippetRenderMode.JSON
      : renderMode === 'table'
        ? CodeSnippetRenderMode.TABLE
        : undefined;

  if (!containsAttributes || isNil(attributes)) {
    return (
      <div css={{ marginTop: theme.spacing.sm }}>
        <SpanModelCostBadge css={{ marginLeft: theme.spacing.sm }} activeSpan={activeSpan} />
        <div css={{ marginTop: theme.spacing.sm }}>
          <Empty
            description={
              <FormattedMessage
                defaultMessage="No attributes found"
                description="Empty state for the attributes tab in the model trace explorer. Attributes are properties of a span that the user defines."
              />
            }
          />
        </div>
      </div>
    );
  }

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        gap: theme.spacing.sm,
        padding: theme.spacing.sm,
      }}
    >
      <SpanModelCostBadge activeSpan={activeSpan} />
      {Object.entries(attributes).map(([key, value]) => (
        <ModelTraceExplorerCodeSnippet
          key={key}
          title={key}
          data={JSON.stringify(value, null, 2)}
          searchFilter={searchFilter}
          activeMatch={activeMatch}
          containsActiveMatch={isActiveMatchSpan && activeMatch.section === 'attributes' && activeMatch.key === key}
          initialRenderMode={initialRenderMode}
        />
      ))}
    </div>
  );
}
