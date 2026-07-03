import { isNil } from 'lodash';

import { Empty, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import type { ModelTraceSpanLink, ModelTraceSpanNode, SearchMatch } from '../ModelTrace.types';
import { CodeSnippetRenderMode } from '../ModelTrace.types';
import { getLinkFieldKey } from '../ModelTraceExplorer.utils';
import { ModelTraceExplorerCodeSnippet } from '../ModelTraceExplorerCodeSnippet';
import { ModelTraceExplorerCollapsibleSection } from '../ModelTraceExplorerCollapsibleSection';
import { Link } from '../RoutingUtils';
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

  const title = href ? (
    <Link
      componentId="mlflow.model_trace_explorer.span_link"
      to={href}
      target="_blank"
      rel="noreferrer"
      onClick={(e: React.MouseEvent) => e.stopPropagation()}
      css={{
        overflow: 'hidden',
        textOverflow: 'ellipsis',
        whiteSpace: 'nowrap',
        fontWeight: theme.typography.typographyBoldFontWeight,
        color: theme.colors.actionPrimaryBackgroundDefault,
        '&:hover': { textDecoration: 'underline' },
      }}
    >
      {link.trace_id}
    </Link>
  ) : (
    <Typography.Text bold css={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
      {link.trace_id}
    </Typography.Text>
  );

  const hasAttributes = link.attributes && Object.keys(link.attributes).length > 0;

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
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
        <ModelTraceExplorerCodeSnippet
          title="span_id"
          data={JSON.stringify(link.span_id)}
          searchFilter={searchFilter}
          activeMatch={activeMatch}
          containsActiveMatch={
            isActiveMatchSpan &&
            activeMatch?.section === 'links' &&
            activeMatch.key === getLinkFieldKey(index, 'span_id')
          }
          initialRenderMode={CodeSnippetRenderMode.TEXT}
        />
        {hasAttributes && (
          <ModelTraceExplorerCodeSnippet
            title="attributes"
            data={JSON.stringify(link.attributes, null, 2)}
            searchFilter={searchFilter}
            activeMatch={activeMatch}
            containsActiveMatch={
              isActiveMatchSpan &&
              activeMatch?.section === 'links' &&
              activeMatch.key === getLinkFieldKey(index, 'attributes')
            }
            initialRenderMode={CodeSnippetRenderMode.JSON}
          />
        )}
      </div>
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
