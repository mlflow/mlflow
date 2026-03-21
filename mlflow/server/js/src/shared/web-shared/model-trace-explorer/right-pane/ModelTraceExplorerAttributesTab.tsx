import { isNil, keys } from 'lodash';

import { Empty, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import type { ModelTraceSpanNode, SearchMatch } from '../ModelTrace.types';
import { ModelTraceExplorerCodeSnippet } from '../ModelTraceExplorerCodeSnippet';

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
  const { attributes } = activeSpan;
  const containsAttributes = keys(attributes).length > 0;
  const isActiveMatchSpan = !isNil(activeMatch) && activeMatch.span.key === activeSpan.key;

  if (!containsAttributes || isNil(attributes)) {
    return (
      <div css={{ marginTop: theme.spacing.md }}>
        <Empty
          description={
            <FormattedMessage
              defaultMessage="No attributes found"
              description="Empty state for the attributes tab in the model trace explorer. Attributes are properties of a span that the user defines."
            />
          }
        />
      </div>
    );
  }

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        gap: theme.spacing.sm,
        padding: theme.spacing.md,
      }}
    >
      {Object.entries(attributes).map(([key, value]) => (
        <ModelTraceExplorerCodeSnippet
          key={key}
          title={key}
          data={JSON.stringify(value, null, 2)}
          searchFilter={searchFilter}
          activeMatch={activeMatch}
          containsActiveMatch={isActiveMatchSpan && activeMatch.section === 'attributes' && activeMatch.key === key}
        />
      ))}
    </div>
  );
}
