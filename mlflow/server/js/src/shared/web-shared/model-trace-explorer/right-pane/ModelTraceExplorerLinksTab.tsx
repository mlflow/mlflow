import { Empty, LinkIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import type { ModelTraceSpanLink, ModelTraceSpanNode } from '../ModelTrace.types';
import { CodeSnippetRenderMode } from '../ModelTrace.types';
import { ModelTraceExplorerCodeSnippet } from '../ModelTraceExplorerCodeSnippet';
import { ModelTraceExplorerCollapsibleSection } from '../ModelTraceExplorerCollapsibleSection';
import { Link } from '../RoutingUtils';
import { useSpanLinkHref } from '../hooks/useSpanLinkHref';

function SpanLinkEntry({ link, index }: { link: ModelTraceSpanLink; index: number }) {
  const { theme } = useDesignSystemTheme();
  const href = useSpanLinkHref(link.trace_id);

  const title = (
    <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
      <Typography.Text bold css={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
        {link.trace_id}
      </Typography.Text>
      {href && (
        <Link
          componentId="mlflow.model_trace_explorer.span_link"
          to={href}
          target="_blank"
          rel="noreferrer"
          onClick={(e: React.MouseEvent) => e.stopPropagation()}
          css={{
            flexShrink: 0,
            display: 'flex',
            alignItems: 'center',
            color: theme.colors.actionPrimaryBackgroundDefault,
          }}
        >
          <LinkIcon css={{ fontSize: 14 }} />
        </Link>
      )}
    </div>
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
          searchFilter=""
          activeMatch={null}
          containsActiveMatch={false}
          initialRenderMode={CodeSnippetRenderMode.TEXT}
        />
        {hasAttributes && (
          <ModelTraceExplorerCodeSnippet
            title="attributes"
            data={JSON.stringify(link.attributes, null, 2)}
            searchFilter=""
            activeMatch={null}
            containsActiveMatch={false}
            initialRenderMode={CodeSnippetRenderMode.JSON}
          />
        )}
      </div>
    </ModelTraceExplorerCollapsibleSection>
  );
}

export function ModelTraceExplorerLinksTab({ activeSpan }: { activeSpan: ModelTraceSpanNode }) {
  const { theme } = useDesignSystemTheme();
  const { links } = activeSpan;

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
        <SpanLinkEntry key={`${link.trace_id}-${link.span_id}-${index}`} link={link} index={index} />
      ))}
    </div>
  );
}
