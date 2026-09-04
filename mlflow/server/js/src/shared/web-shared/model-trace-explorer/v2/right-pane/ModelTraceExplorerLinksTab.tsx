import { isNil } from 'lodash';
import { useMemo, useState } from 'react';

import { Button, Card, NewWindowIcon, Tag, Tooltip, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { useIntl } from '@databricks/i18n';

import type { ModelTraceSpanLink, ModelTraceSpanNode, SearchMatch } from '../ModelTrace.types';
import { getIconTypeForSpan, getLinkFieldKey, getSpanExceptionCount } from '../ModelTraceExplorer.utils';
import { ModelTraceExplorerIcon } from '../ModelTraceExplorerIcon';
import { ModelTraceExplorerFieldRenderer } from '../field-renderers/ModelTraceExplorerFieldRenderer';
import { useSpanLinkHrefs } from '../../hooks/useSpanLinkHref';
import { useCopyController } from '../../../copy/useCopyController';
import { useHref } from '../../RoutingUtils';

const MAX_VISIBLE_ATTRIBUTES = 3;

const CopyableIdTag = ({ id, copyTooltip }: { id: string; copyTooltip: string }): JSX.Element => {
  const { copy, tooltipMessage, tooltipOpen, handleTooltipOpenChange } = useCopyController(id, copyTooltip);

  return (
    <Tooltip
      componentId="mlflow.model_trace_explorer.span_link.id_tooltip"
      content={tooltipMessage}
      open={tooltipOpen}
      onOpenChange={handleTooltipOpenChange}
    >
      <Tag
        componentId="mlflow.model_trace_explorer.span_link.id"
        color="default"
        onClick={copy}
        onKeyDown={(event) => {
          if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault();
            copy();
          }
        }}
        role="button"
        tabIndex={0}
        css={{ alignSelf: 'flex-start', cursor: 'pointer', margin: 0, maxWidth: 96 }}
      >
        <Typography.Text size="sm" color="secondary" css={{ fontFamily: 'monospace', whiteSpace: 'nowrap' }}>
          {id.replace(/^tr-/, '').slice(0, 8)}
        </Typography.Text>
      </Tag>
    </Tooltip>
  );
};

const SpanLinkEntry = ({
  link,
  index,
  href,
  searchFilter,
  activeMatch,
  isActiveMatchSpan,
  isSameTrace,
  linkedSpan,
  onSelectSpan,
}: {
  link: ModelTraceSpanLink;
  index: number;
  href: string | undefined;
  searchFilter: string;
  activeMatch: SearchMatch | null;
  isActiveMatchSpan: boolean;
  isSameTrace: boolean;
  linkedSpan?: ModelTraceSpanNode;
  onSelectSpan?: () => void;
}): JSX.Element => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const routedHref = useHref(href ?? '');
  const [showAllAttributes, setShowAllAttributes] = useState(false);
  const attributeEntries = link.attributes ? Object.entries(link.attributes) : [];
  const hasHiddenAttributes = attributeEntries.length > MAX_VISIBLE_ATTRIBUTES;
  const activeMatchEntry = attributeEntries.find(
    ([key]) => isActiveMatchSpan && activeMatch?.section === 'links' && activeMatch.key === getLinkFieldKey(index, key),
  );
  const collapsedAttributeEntries = attributeEntries.slice(0, MAX_VISIBLE_ATTRIBUTES);
  if (activeMatchEntry && !collapsedAttributeEntries.includes(activeMatchEntry)) {
    collapsedAttributeEntries.push(activeMatchEntry);
  }
  const visibleAttributeEntries = showAllAttributes ? attributeEntries : collapsedAttributeEntries;
  const jumpButtonCss = {
    whiteSpace: 'nowrap' as const,
    '&:not(:hover):not(:focus-visible)': {
      borderColor: `${theme.colors.borderDecorative} !important`,
    },
  };
  const openButton = isSameTrace ? (
    <Button
      componentId="mlflow.model_trace_explorer.span_link"
      size="small"
      css={jumpButtonCss}
      disabled={!linkedSpan || !onSelectSpan}
      onClick={onSelectSpan}
    >
      Jump to linked span
    </Button>
  ) : (
    <Button
      componentId="mlflow.model_trace_explorer.span_link"
      size="small"
      href={href ? routedHref : undefined}
      target="_blank"
      disabled={!href}
      endIcon={<NewWindowIcon css={{ fontSize: 12 }} />}
      css={jumpButtonCss}
    >
      Jump to linked span
    </Button>
  );

  return (
    <Card
      componentId="mlflow.model_trace_explorer.span_link.card"
      css={{
        boxSizing: 'border-box',
        marginBottom: theme.spacing.sm,
        padding: theme.spacing.md,
        width: '100%',
      }}
      disableHover
    >
      <div
        css={{
          alignItems: 'flex-start',
          display: 'grid',
          gap: theme.spacing.md,
          gridTemplateColumns: 'minmax(0, 1fr) auto',
        }}
      >
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm, minWidth: 0 }}>
          {!isSameTrace && (
            <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
              <Typography.Hint>Trace ID</Typography.Hint>
              <CopyableIdTag
                id={link.trace_id}
                copyTooltip={intl.formatMessage({
                  defaultMessage: 'Copy trace ID',
                  description: 'Tooltip for copying a linked trace ID',
                })}
              />
            </div>
          )}
          {linkedSpan ? (
            <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
              <Typography.Hint>Linked span</Typography.Hint>
              <div css={{ alignItems: 'center', display: 'flex', gap: theme.spacing.sm, minWidth: 0 }}>
                <ModelTraceExplorerIcon
                  type={getIconTypeForSpan(linkedSpan.type ?? '')}
                  hasException={getSpanExceptionCount(linkedSpan) > 0}
                />
                <Typography.Text css={{ overflowWrap: 'anywhere' }}>{linkedSpan.title}</Typography.Text>
              </div>
            </div>
          ) : (
            <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
              <Typography.Hint>Span ID</Typography.Hint>
              <CopyableIdTag
                id={link.span_id}
                copyTooltip={intl.formatMessage({
                  defaultMessage: 'Copy span ID',
                  description: 'Tooltip for copying a linked span ID',
                })}
              />
            </div>
          )}
        </div>
        {openButton}
      </div>
      <div
        css={{
          borderTop: `1px solid ${theme.colors.borderDecorative}`,
          display: 'flex',
          flexDirection: 'column',
          gap: theme.spacing.sm,
          marginTop: theme.spacing.md,
          paddingTop: theme.spacing.md,
        }}
      >
        <Typography.Hint>Attributes</Typography.Hint>
        {attributeEntries.length > 0 ? (
          <>
            {visibleAttributeEntries.map(([key, value]) => (
              <ModelTraceExplorerFieldRenderer
                key={key}
                title={key}
                data={JSON.stringify(value, null, 2)}
                renderMode="default"
                searchFilter={searchFilter}
                activeMatch={activeMatch}
                containsActiveMatch={
                  isActiveMatchSpan &&
                  activeMatch?.section === 'links' &&
                  activeMatch.key === getLinkFieldKey(index, key)
                }
              />
            ))}
            {hasHiddenAttributes && (
              <Button
                componentId="mlflow.model_trace_explorer.span_link.attributes_toggle"
                size="small"
                type="link"
                css={{ alignSelf: 'flex-start' }}
                onClick={() => setShowAllAttributes((value) => !value)}
              >
                {showAllAttributes ? 'See less' : 'See more'}
              </Button>
            )}
          </>
        ) : (
          <Typography.Hint>No attributes</Typography.Hint>
        )}
      </div>
    </Card>
  );
};

export const ModelTraceExplorerLinksTab = ({
  activeSpan,
  searchFilter,
  activeMatch,
  onSelectSpan,
  spanNodes = {},
}: {
  activeSpan: ModelTraceSpanNode;
  searchFilter: string;
  activeMatch: SearchMatch | null;
  onSelectSpan?: (spanId: string) => void;
  spanNodes?: Record<string, ModelTraceSpanNode>;
}): JSX.Element | null => {
  const { theme } = useDesignSystemTheme();
  const { links } = activeSpan;
  const isActiveMatchSpan = !isNil(activeMatch) && activeMatch.span.key === activeSpan.key;
  const traceIds = useMemo(() => (links ?? []).map((link) => link.trace_id), [links]);
  const hrefMap = useSpanLinkHrefs(traceIds);

  if (!links?.length) {
    return null;
  }

  return (
    <div
      css={{
        paddingLeft: theme.spacing.md + theme.spacing.xs,
        paddingRight: theme.spacing.md + theme.spacing.xs,
        paddingTop: theme.spacing.md,
      }}
    >
      {links.map((link, index) => {
        const isSameTrace = link.trace_id === activeSpan.traceId;
        return (
          <SpanLinkEntry
            key={`${link.trace_id}-${link.span_id}-${index}`}
            link={link}
            index={index}
            href={hrefMap[link.trace_id]}
            searchFilter={searchFilter}
            activeMatch={activeMatch}
            isActiveMatchSpan={isActiveMatchSpan}
            isSameTrace={isSameTrace}
            linkedSpan={isSameTrace ? spanNodes[link.span_id] : undefined}
            onSelectSpan={isSameTrace && onSelectSpan ? () => onSelectSpan(link.span_id) : undefined}
          />
        );
      })}
    </div>
  );
};
