import { isNil } from 'lodash';

import { Empty, Typography, useDesignSystemTheme, XCircleIcon } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { CodeSnippetRenderMode, type ModelTraceSpanNode, type SearchMatch } from '../ModelTrace.types';
import { getEventAttributeKey } from '../ModelTraceExplorer.utils';
import { ModelTraceExplorerCodeSnippet } from '../ModelTraceExplorerCodeSnippet';
import { ModelTraceExplorerCollapsibleSection } from '../ModelTraceExplorerCollapsibleSection';

export function ModelTraceExplorerEventsTab({
  activeSpan,
  searchFilter,
  activeMatch,
}: {
  activeSpan: ModelTraceSpanNode;
  searchFilter: string;
  activeMatch: SearchMatch | null;
}) {
  const { theme } = useDesignSystemTheme();
  const { events } = activeSpan;
  const isActiveMatchSpan = !isNil(activeMatch) && activeMatch.span.key === activeSpan.key;

  if (!Array.isArray(events) || events.length === 0) {
    return (
      <div css={{ marginTop: theme.spacing.md }}>
        <Empty
          description={
            <FormattedMessage
              defaultMessage="No events found"
              description="Empty state for the events tab in the model trace explorer. Events are logs of arbitrary things (e.g. exceptions) that occur during the execution of a span, and can be user-defined."
            />
          }
        />
      </div>
    );
  }

  return (
    <div css={{ padding: theme.spacing.md }}>
      {events.map((event, index) => {
        const attributes = event.attributes;
        const title =
          event.name === 'exception' ? (
            <>
              <XCircleIcon css={{ marginRight: theme.spacing.sm }} color="danger" />
              <Typography.Text color="error" bold>
                Exception
              </Typography.Text>
            </>
          ) : (
            event.name
          );

        if (!attributes) return null;

        return (
          <ModelTraceExplorerCollapsibleSection
            key={`${event.name}-${index}`}
            sectionKey={event.name}
            title={title}
            withBorder
          >
            <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
              {Object.keys(attributes).map((attribute) => {
                const key = getEventAttributeKey(event.name, index, attribute);

                return (
                  <ModelTraceExplorerCodeSnippet
                    key={key}
                    title={attribute}
                    data={JSON.stringify(attributes[attribute], null, 2)}
                    searchFilter={searchFilter}
                    activeMatch={activeMatch}
                    containsActiveMatch={
                      isActiveMatchSpan && activeMatch.section === 'events' && activeMatch.key === key
                    }
                    initialRenderMode={CodeSnippetRenderMode.TEXT}
                  />
                );
              })}
            </div>
          </ModelTraceExplorerCollapsibleSection>
        );
      })}
    </div>
  );
}
