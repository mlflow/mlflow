import { isNil } from 'lodash';

import { Empty, Tooltip, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import type { ModelTraceSpanLink, ModelTraceSpanNode, SearchMatch } from '../ModelTrace.types';
import { CodeSnippetRenderMode } from '../ModelTrace.types';
import { getLinkFieldKey } from '../ModelTraceExplorer.utils';
import { ModelTraceExplorerCodeSnippet } from '../ModelTraceExplorerCodeSnippet';
import { ModelTraceExplorerCollapsibleSection } from '../ModelTraceExplorerCollapsibleSection';
import { useSpanLinkHref } from '../hooks/useSpanLinkHref';

function SpanLinkEntry({
  link,
  index,
  searchFilter,
  activeMatch,
  isActiveMatchSpan,
}: {
  link: ModelTraceSpanLink;
  index: number;
  searchFilter: string;
  activeMatch: SearchMatch | null;
  isActiveMatchSpan: boolean;
}) {
  const { theme } = useDesignSystemTheme();
  const href = useSpanLinkHref(link.trace_id);

  const title = (
    <Tooltip
      componentId="mlflow.model_trace_explorer.span_link.tooltip"
      content={`span_id: ${link.span_id}`}
    >
      {href ? (
        <Typography.Link
          componentId="mlflow.model_trace_explorer.span_link"
          href={href}
          openInNewTab
          onClick={(e: React.MouseEvent) => e.stopPropagation()}
          css={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
        >
          {link.trace_id}
        </Typography.Link>
      ) : (
        <Typography.Text bold css={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
          {link.trace_id}
        </Typography.Text>
      )}
    </Tooltip>
  );

  const attributes = link.attributes && Object.keys(link.attributes).length > 0 ? link.attributes : null;

  return (
    <ModelTraceExplorerCollapsibleSection
      key={`link-${index}`}
      css={{
        marginBottom: theme.spacing.sm,
        '& > div:first-of-type': { borderTop: 'none' },
      }}
      sectionKey={`link-${index}`}
      title={title}
      withBorder
    >
      {attributes && (
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
          {Object.entries(attributes).map(([key, value]) => (
            <ModelTraceExplorerCodeSnippet
              key={key}
              title={key}
              data={JSON.stringify(value, null, 2)}
              searchFilter={searchFilter}
              activeMatch={activeMatch}
              containsActiveMatch={
                isActiveMatchSpan &&
                activeMatch?.section === 'links' &&
                activeMatch.key === getLinkFieldKey(index, key)
              }
              initialRenderMode={CodeSnippetRenderMode.TEXT}
            />
          ))}
        </div>
      )}
    </ModelTraceExplorerCollapsibleSection>
  );
}

export function ModelTraceExplorerLinksTab({
  activeSpan,
  searchFilter,
  activeMatch,
}: {
  activeSpan: ModelTraceSpanNode;
  searchFilter: string;
  activeMatch: SearchMatch | null;
}) {
  const { theme } = useDesignSystemTheme();
  const { links } = activeSpan;
  const isActiveMatchSpan = !isNil(activeMatch) && activeMatch.span.key === activeSpan.key;

  if (!Array.isArray(links) || links.length === 0) {
    return (
      <div css={{ marginTop: theme.spacing.sm }}>
        <Empty
          description={
            <FormattedMessage
              defaultMessage="No links found"
              description="Empty state for the links tab in the model trace explorer. Links connect spans across different traces."
            />
          }
        />
      </div>
    );
  }

  return (
    <div css={{ marginTop: theme.spacing.sm }}>
      {links.map((link, index) => (
        <SpanLinkEntry
          key={`${link.trace_id}-${link.span_id}-${index}`}
          link={link}
          index={index}
          searchFilter={searchFilter}
          activeMatch={activeMatch}
          isActiveMatchSpan={isActiveMatchSpan}
        />
      ))}
    </div>
  );
}
